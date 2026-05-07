"""Dynamic tool creation — agent creates new tools via natural language.

Agent describes what a tool should do → LLM generates Python code →
code is validated and registered as a LangChain tool.

Tools are persisted in workspace/tools/ and loaded on next startup.

Security: generated code runs in a restricted scope (no file system
access, no network, no imports beyond allowlist).
"""

import ast
import inspect
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field, create_model

from kronos.config import settings
from kronos.llm import ModelTier, get_model
from kronos.workspace import ws

log = logging.getLogger("kronos.tools.dynamic")

TOOLS_DIR = ws.dynamic_tools_dir

# Allowed imports in generated code
SAFE_IMPORTS = {
    "json", "re", "math", "datetime", "collections",
    "itertools", "functools", "hashlib", "base64",
    "urllib.parse", "statistics",
}

# Forbidden patterns in generated code
FORBIDDEN_PATTERNS = [
    r"import\s+os\b",
    r"import\s+subprocess",
    r"import\s+shutil",
    r"import\s+pathlib",
    r"__import__",
    r"eval\s*\(",
    r"exec\s*\(",
    r"open\s*\(",
    r"compile\s*\(",
    r"globals\s*\(",
    r"locals\s*\(",
    r"getattr\s*\(",
    r"setattr\s*\(",
    r"delattr\s*\(",
    r"os\.\w+",
    r"sys\.\w+",
    r"subprocess\.",
    r"shutil\.",
]

PYTHON_TYPE_MAP = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list[Any],
    "dict": dict[str, Any],
}


@dataclass(frozen=True)
class ToolFunctionSpec:
    name: str
    description: str
    args_schema: type

GENERATE_PROMPT = """Create a Python function for a LangChain tool.

Tool description: {description}
Tool name: {name}

Requirements:
- Write a single async function with type hints
- Function name must match tool name (snake_case)
- Include a docstring (this becomes the tool description for the LLM)
- Use only these imports: {safe_imports}
- Function must return a string
- No file I/O, no network calls, no subprocess, no eval/exec
- Handle errors gracefully (return error message, don't raise)

Return ONLY the Python code, no markdown fences, no explanation.

Example:
```python
import math

async def calculate_compound_interest(principal: float, rate: float, years: int) -> str:
    \"\"\"Calculate compound interest. Args: principal, annual rate (%), years.\"\"\"
    try:
        amount = principal * math.pow(1 + rate / 100, years)
        return f"After {{years}} years: ${{amount:,.2f}} (interest: ${{amount - principal:,.2f}})"
    except Exception as e:
        return f"Calculation error: {{e}}"
```
"""


def validate_code(code: str) -> tuple[bool, str]:
    """Validate generated code for safety."""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            return False, f"Forbidden pattern: {pattern}"

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    func_defs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if base_module not in SAFE_IMPORTS:
                    return False, f"Unsafe import: {alias.name}"
            continue
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            base_module = module.split(".")[0]
            if base_module not in SAFE_IMPORTS:
                return False, f"Unsafe import: {module}"
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            func_defs.append(node)
            continue
        return False, f"Top-level statements are not allowed: {type(node).__name__}"

    if len(func_defs) != 1:
        return False, f"Expected 1 function, found {len(func_defs)}"

    func_def = func_defs[0]
    if func_def.decorator_list:
        return False, "Function decorators are not allowed"
    if func_def.args.vararg or func_def.args.kwarg:
        return False, "Variadic *args/**kwargs are not allowed"

    for default in [*func_def.args.defaults, *[item for item in func_def.args.kw_defaults if item is not None]]:
        try:
            ast.literal_eval(default)
        except (ValueError, TypeError):
            return False, "Only literal argument defaults are allowed"

    return True, ""


def _extract_function_spec(code: str, description: str) -> ToolFunctionSpec:
    tree = ast.parse(code)
    module_doc = ast.get_docstring(tree) or ""
    func_def = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )
    docstring = ast.get_docstring(func_def) or module_doc or description
    return ToolFunctionSpec(
        name=func_def.name,
        description=docstring,
        args_schema=_build_args_schema(func_def.name, func_def),
    )


def _build_args_schema(name: str, func_def: ast.FunctionDef | ast.AsyncFunctionDef) -> type:
    fields: dict[str, tuple[type, Any]] = {}

    positional_args = [*func_def.args.posonlyargs, *func_def.args.args]
    defaults = [None] * (len(positional_args) - len(func_def.args.defaults)) + list(func_def.args.defaults)
    for arg, default_node in zip(positional_args, defaults, strict=True):
        fields[arg.arg] = _field_for_arg(arg, default_node)

    for arg, default_node in zip(func_def.args.kwonlyargs, func_def.args.kw_defaults, strict=True):
        fields[arg.arg] = _field_for_arg(arg, default_node)

    return create_model(f"{name}_Args", **fields)


def _field_for_arg(arg: ast.arg, default_node: ast.expr | None) -> tuple[type, Any]:
    annotation = _annotation_to_type(arg.annotation)
    if default_node is None:
        return annotation, Field(..., description=arg.arg)
    default = ast.literal_eval(default_node)
    if default is None:
        annotation = Any
    return annotation, Field(default, description=arg.arg)


def _annotation_to_type(annotation: ast.expr | None) -> type:
    if annotation is None:
        return Any
    if isinstance(annotation, ast.Name):
        return PYTHON_TYPE_MAP.get(annotation.id, Any)
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        return PYTHON_TYPE_MAP.get(annotation.value.id, Any)
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return PYTHON_TYPE_MAP.get(annotation.value, Any)
    return Any


def _build_runner_code(code: str, func_name: str, args: tuple, kwargs: dict) -> str:
    payload = json.dumps({"args": args, "kwargs": kwargs}, default=str)
    return (
        code
        + "\n\nimport asyncio\n"
        + "import inspect\n"
        + "import json\n"
        + f"_payload = json.loads({payload!r})\n"
        + f"_result = {func_name}(*_payload['args'], **_payload['kwargs'])\n"
        + "if inspect.isawaitable(_result):\n"
        + "    _result = asyncio.run(_result)\n"
        + "print(_result)\n"
    )


async def _run_locally_for_dev(code: str, func_name: str, args: tuple, kwargs: dict):
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102
    result = namespace[func_name](*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _build_dynamic_tool(name: str, code: str, spec: ToolFunctionSpec) -> BaseTool:
    async def _sandboxed_wrapper(*args, **kwargs):
        """Execute the dynamic tool in a Docker sandbox."""
        from kronos.tools.sandbox import execute_sandboxed, sandbox_ready

        if sandbox_ready():
            runner_code = _build_runner_code(code, spec.name, args, kwargs)
            stdout, stderr = await execute_sandboxed(runner_code, timeout=30)
            if stderr:
                log.warning("Sandbox stderr for %s: %s", spec.name, stderr[:200])
            return stdout or stderr or "No output"
        if settings.require_dynamic_tool_sandbox:
            from kronos.tools.sandbox import sandbox_unavailable_message

            return f"Blocked: Docker sandbox is required for dynamic tools. {sandbox_unavailable_message()}"

        return await _run_locally_for_dev(code, spec.name, args, kwargs)

    _sandboxed_wrapper.__name__ = spec.name
    _sandboxed_wrapper.__doc__ = spec.description

    return StructuredTool.from_function(
        coroutine=_sandboxed_wrapper,
        name=name,
        description=spec.description,
        args_schema=spec.args_schema,
    )


async def create_tool(name: str, description: str) -> tuple[BaseTool | None, str]:
    """Generate a tool from natural language description.

    Returns (tool, message). Tool is None on failure.
    """
    if not settings.enable_dynamic_tools:
        return None, (
            "Dynamic tool creation is disabled. "
            "Set ENABLE_DYNAMIC_TOOLS=true in a trusted local environment."
        )

    # Sanitize name
    clean_name = re.sub(r"[^a-z0-9_]", "_", name.lower().strip())
    if not clean_name:
        return None, "Invalid tool name"

    # Generate code via LLM
    prompt = GENERATE_PROMPT.format(
        description=description,
        name=clean_name,
        safe_imports=", ".join(sorted(SAFE_IMPORTS)),
    )

    model = get_model(ModelTier.STANDARD)
    from langchain_core.messages import HumanMessage
    response = model.invoke([HumanMessage(content=prompt)])
    code = response.content if isinstance(response.content, str) else str(response.content)

    # Strip markdown fences if present
    code = re.sub(r"^```python\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    code = code.strip()

    # Validate
    valid, reason = validate_code(code)
    if not valid:
        return None, f"Generated code rejected: {reason}"

    spec = _extract_function_spec(code, description)
    if spec.name != clean_name:
        return None, f"Generated code rejected: function name must match tool name '{clean_name}'"

    if settings.require_dynamic_tool_sandbox:
        from kronos.tools.sandbox import sandbox_ready, sandbox_unavailable_message

        if not sandbox_ready():
            return None, (
                "Docker sandbox is required before dynamic tools can be created. "
                f"{sandbox_unavailable_message()} "
                "Set REQUIRE_DYNAMIC_TOOL_SANDBOX=false for local development only."
            )

    tool = _build_dynamic_tool(clean_name, code, spec)

    # Persist
    _save_tool(clean_name, code, description)

    log.info("Dynamic tool created: %s", clean_name)
    return tool, f"Tool '{clean_name}' created successfully."


def load_persisted_tools() -> list[BaseTool]:
    """Load previously created dynamic tools from disk."""
    if not settings.enable_dynamic_tools:
        return []

    if settings.require_dynamic_tool_sandbox:
        from kronos.tools.sandbox import sandbox_ready, sandbox_unavailable_message

        if not sandbox_ready():
            log.warning("Skipping persisted dynamic tools: %s", sandbox_unavailable_message())
            return []

    if not TOOLS_DIR.exists():
        return []

    tools = []
    for tool_file in TOOLS_DIR.glob("*.py"):
        try:
            code = tool_file.read_text(encoding="utf-8")
            valid, reason = validate_code(code)
            if not valid:
                log.warning("Skipping invalid persisted tool %s: %s", tool_file.name, reason)
                continue

            spec = _extract_function_spec(code, f"Dynamic tool: {tool_file.stem}")
            tools.append(_build_dynamic_tool(spec.name, code, spec))

        except Exception as e:
            log.error("Failed to load dynamic tool %s: %s", tool_file.name, e)

    if tools:
        log.info("Loaded %d persisted dynamic tools", len(tools))
    return tools


def _save_tool(name: str, code: str, description: str) -> None:
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    path = TOOLS_DIR / f"{name}.py"
    header = f'"""{description}"""\n\n'
    path.write_text(header + code, encoding="utf-8")
