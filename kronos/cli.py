"""KAOS command line interface.

Primary commands:
    kaos doctor              # validate local environment
    kaos chat                # local chat without Telegram
    kaos demo                # safe local demo chat
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from kronos import __version__
from kronos.config import settings
from kronos.llm import ModelTier, describe_provider_chain, is_runtime_llm_configured
from kronos.logging import install_pii_filter
from kronos.security.pii import mask_pii

log = logging.getLogger("kronos.cli")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    install_pii_filter()


def _runtime_llm_configured() -> bool:
    """Return whether the current runtime LLM factory can create a chat model."""
    return is_runtime_llm_configured()


def _print_missing_runtime_llm() -> None:
    print("KAOS chat requires at least one configured LLM provider.")
    print("Set FIREWORKS_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY, or configure a provider chain in .env.")
    print("Run `kaos doctor` to inspect providers, or `kaos demo` for the offline walkthrough.")


_SECRET_ARG_NAMES = {"token", "secret", "password", "api_key", "apikey", "key", "hash", "authorization"}


def _redact_tool_payload(value: Any, key: str = "") -> Any:
    key_name = key.lower().replace("-", "_")
    if key_name in _SECRET_ARG_NAMES or key_name.endswith(("_token", "_secret", "_password", "_api_key", "_key")):
        return "***REDACTED***"
    if isinstance(value, dict):
        return {str(k): _redact_tool_payload(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_tool_payload(item) for item in value[:10]]
    if isinstance(value, tuple):
        return [_redact_tool_payload(item) for item in value[:10]]
    if isinstance(value, str):
        redacted = mask_pii(value)
        return redacted if len(redacted) <= 160 else f"{redacted[:157]}..."
    return value


def _format_tool_payload(payload: dict[str, Any]) -> str:
    redacted = _redact_tool_payload(payload)
    return json.dumps(redacted, ensure_ascii=False, sort_keys=True, default=str)


def _make_tool_event_printer():
    def printer(event: str, payload: dict[str, Any]) -> None:
        name = str(payload.get("name") or "unknown")
        if event == "tool_call":
            args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
            print(f"[tool] {name} args={_format_tool_payload(args)}", file=sys.stderr)
            return
        if event == "tool_result":
            status = "ok" if payload.get("ok") else "error"
            content = str(payload.get("content", "")).replace("\n", " ")
            if len(content) > 180:
                content = f"{content[:177]}..."
            print(f"[tool:{status}] {name} {content}", file=sys.stderr)

    return printer


def _print_chat_runtime_summary(agent_tool_count: int, enable_memory: bool) -> None:
    gates = {
        "memory": "on" if enable_memory else "off",
        "tools": agent_tool_count,
        "dynamic-tools": "on" if settings.enable_dynamic_tools else "off",
        "dynamic-mcp": "on" if settings.enable_dynamic_mcp_servers else "off",
        "server-ops": "on" if settings.enable_server_ops else "off",
    }
    print(f"[approval] {_format_tool_payload(gates)}", file=sys.stderr)


async def run_cli(
    use_tools: bool = False,
    thread_id: str = "cli-test",
    prompt: str | None = None,
    enable_memory: bool = True,
) -> int:
    """Interactive CLI for testing the agent."""
    if not _runtime_llm_configured():
        _print_missing_runtime_llm()
        return 1

    _configure_logging()

    try:
        from kronos.graph import KronosAgent
        from kronos.session import SessionStore
        from kronos.tools.manager import managed_mcp_tools
    except ModuleNotFoundError as e:
        print(f"Missing Python dependency: {e.name}")
        print('Install KAOS first with: pip install -e ".[dev]"')
        print("Or run `kaos demo` for the offline walkthrough.")
        return 1

    log.info("KAOS chat mode (workspace: %s, tools: %s)", settings.workspace_path, use_tools)

    if use_tools:
        ctx = managed_mcp_tools()
    else:
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _no_tools():
            yield []

        ctx = _no_tools()

    async with ctx as tools:
        session_store = SessionStore(settings.db_path, agent_name=settings.agent_name)
        agent = KronosAgent(
            tools=tools or None,
            enable_memory=enable_memory,
            session_store=session_store,
            tool_event_callback=_make_tool_event_printer(),
        )
        _print_chat_runtime_summary(agent.tool_count, enable_memory)

        if prompt is not None:
            try:
                reply = await agent.ainvoke(
                    message=prompt,
                    thread_id=thread_id,
                    user_id="cli-user",
                    session_id="cli-session",
                )
                print(reply)
                return 0
            except Exception as e:
                print(f"[Error] {e}")
                return 1

        log.info("Agent ready (%d tools). Type messages, Ctrl+C to exit.\n", len(tools))

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "/q"):
                break

            if user_input.lower() in ("/clear", "/reset"):
                result = await agent.clear_context(thread_id)
                print(f"\n{result}")
                continue

            try:
                reply = await agent.ainvoke(
                    message=user_input,
                    thread_id=thread_id,
                    user_id="cli-user",
                    session_id="cli-session",
                )
                print(f"\nKronos: {reply}")
            except Exception as e:
                print(f"\n[Error] {e}")

    return 0


def _force_demo_safety() -> None:
    """Keep demo mode conservative even if local env enables risky features."""
    settings.enable_dynamic_tools = False
    settings.enable_mcp_gateway_management = False
    settings.enable_dynamic_mcp_servers = False
    settings.enable_server_ops = False
    settings.require_dynamic_tool_sandbox = True


_DEMO_EVENTS: tuple[tuple[str, str], ...] = (
    (
        "Runtime",
        "User asks for a launch plan. KAOS creates one session, keeps state local, and routes through the same runtime used by CLI, Telegram, cron, and dashboard.",
    ),
    (
        "Memory",
        "The agent stores durable preferences such as 'prefer concise technical answers' and can recall them in later sessions.",
    ),
    (
        "Skills",
        "A skill packages reusable behavior: instructions, references, and tool policy. Users can version and review skills instead of hiding behavior in prompts.",
    ),
    (
        "Tool Gateway",
        "Safe tools can run immediately. Dynamic tools, dynamic MCP registration, and server ops stay blocked until explicit local opt-in.",
    ),
    (
        "Automations",
        "Scheduled jobs call the same runtime for briefs, monitors, reports, and maintenance without waiting for a chat message.",
    ),
    (
        "Swarm",
        "Optional sub-agents can debate or split work, then write back a single synthesized result through the main runtime.",
    ),
)


def _demo_reply(prompt: str) -> str:
    """Small deterministic demo brain for offline quickstart."""
    text = prompt.lower()
    if "memory" in text:
        return "Memory demo: KAOS would recall durable user facts, inject only relevant context, and avoid storing ephemeral peer reactions."
    if "skill" in text:
        return "Skill demo: package a repeatable workflow in workspaces/<agent>/self/skills, then expose it through reviewed skill tools."
    if "mcp" in text or "tool" in text:
        return "Tool demo: static MCP tools are allowed; dynamic tool creation and dynamic MCP server registration are disabled by default."
    if "swarm" in text or "sub" in text:
        return "Swarm demo: run specialist agents with separate workspaces and merge their output through the main KAOS session."
    if "dashboard" in text:
        return "Dashboard demo: bind to 127.0.0.1 by default, generate a temporary password if none is configured, and inspect runtime state locally."
    return "KAOS demo: runtime + memory + skills + MCP + automations + optional swarm, with risky capabilities disabled until explicit opt-in."


def run_demo(interactive: bool = False, live: bool = False, use_tools: bool = False) -> int:
    """Run a safe local demo that does not require Telegram, Docker, or LLM keys."""
    _force_demo_safety()

    if live:
        print("Starting KAOS live demo mode. Dynamic tools, dynamic MCP, and server ops are disabled.")
        asyncio.run(run_cli(use_tools=use_tools, thread_id="kaos-demo"))
        return 0

    print("KAOS safe demo\n")
    print("No Telegram, Docker, server registry, or LLM key is required for this walkthrough.")
    print("Risky capabilities forced off: dynamic tools, dynamic MCP, server ops.\n")

    for title, detail in _DEMO_EVENTS:
        print(f"[{title}] {detail}")

    if interactive:
        print("\nAsk about memory, skills, tools, MCP, dashboard, or swarm. Type 'exit' to stop.")
        while True:
            try:
                prompt = input("\nDemo> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if prompt.lower() in {"exit", "quit", "/q"}:
                break
            if prompt:
                print(_demo_reply(prompt))

    print("\nNext commands:")
    print("  kaos init my-agent --dry-run")
    print("  kaos doctor")
    print("  kaos chat")
    print("\nFor a real LLM-backed demo: kaos demo --live")
    return 0


def run_doctor() -> int:
    """Run local environment checks."""
    checks: list[tuple[str, str, str]] = []

    def ok(name: str, detail: str) -> None:
        checks.append(("OK", name, detail))

    def warn(name: str, detail: str) -> None:
        checks.append(("WARN", name, detail))

    def fail(name: str, detail: str) -> None:
        checks.append(("FAIL", name, detail))

    ok("Python", sys.version.split()[0])

    project_root = Path.cwd()
    if (project_root / "pyproject.toml").exists():
        ok("Project", str(project_root))
    else:
        warn("Project", "Run doctor from the KAOS repo root for best results")

    provider_lines: list[str] = []
    for tier in (ModelTier.STANDARD, ModelTier.LITE):
        rows = describe_provider_chain(tier)
        configured = [
            f"{row['provider']}:{row['model']}"
            for row in rows
            if row["configured"]
        ]
        missing = [str(row["provider"]) for row in rows if not row["configured"]]
        if configured:
            ok(f"LLM {tier.value}", " -> ".join(configured))
        else:
            warn(f"LLM {tier.value}", f"No configured providers in chain: {', '.join(missing) or '(empty)'}")
        provider_lines.extend(configured)

    if provider_lines:
        ok("Runtime LLM provider", "configured")
    else:
        warn("Runtime LLM provider", "Set provider keys or configure KAOS_*_PROVIDER_CHAIN before chat")

    if settings.openai_api_key:
        ok("OpenAI optional key", "configured")

    fallback_workspace = Path("workspaces") / settings.agent_name
    workspace = Path(settings.workspace_path) if settings.workspace_path else fallback_workspace
    if workspace.exists():
        ok("Workspace", str(workspace))
    elif settings.workspace_path and fallback_workspace.exists():
        warn(
            "Workspace",
            f"WORKSPACE_PATH points to missing {workspace}; fallback exists at {fallback_workspace}",
        )
    elif not settings.workspace_path:
        warn(
            "Workspace",
            f"No workspace for AGENT_NAME={settings.agent_name} yet; run `kaos init {settings.agent_name}`",
        )
    else:
        fail("Workspace", f"Missing workspace for AGENT_NAME={settings.agent_name}: {workspace}")

    db_dir = Path(settings.db_dir)
    if db_dir.parent.exists():
        ok("Data path", str(db_dir))
    else:
        warn("Data path", f"Parent directory does not exist yet: {db_dir.parent}")

    if settings.enable_dynamic_tools:
        if settings.require_dynamic_tool_sandbox:
            from kronos.tools.sandbox import sandbox_status

            status = sandbox_status()
            if not status["docker_available"]:
                fail("Dynamic tools", "ENABLE_DYNAMIC_TOOLS=true but Docker is unavailable")
            elif not status["image_available"]:
                fail(
                    "Dynamic tools",
                    f"ENABLE_DYNAMIC_TOOLS=true but sandbox image is missing; run {status['build_script']}",
                )
            else:
                warn("Dynamic tools", f"Enabled with required sandbox image {status['image']}")
        else:
            warn("Dynamic tools", "Enabled; keep this only for trusted local deployments")
    else:
        ok("Dynamic tools", "disabled by default")

    if settings.enable_mcp_gateway_management:
        warn("MCP gateway management", "Enabled; agent can add/remove/reload MCP servers")
    else:
        ok("MCP gateway management", "disabled by default")

    if settings.enable_dynamic_mcp_servers:
        warn("Dynamic MCP registry", "Enabled; persisted local MCP servers will be loaded")
    else:
        ok("Dynamic MCP registry", "disabled by default")

    if settings.enable_server_ops:
        registry = Path(os.environ.get("SERVER_REGISTRY_PATH", "servers.yaml"))
        if registry.exists():
            warn("Server ops", f"Enabled with registry {registry}")
        else:
            fail("Server ops", "ENABLE_SERVER_OPS=true but no server registry was found")
    else:
        ok("Server ops", "disabled by default")

    invalid_allowed_users = settings.invalid_allowed_user_tokens
    if invalid_allowed_users:
        fail("Telegram access", f"Invalid ALLOWED_USERS entries: {', '.join(invalid_allowed_users)}")
    elif settings.allowed_user_ids:
        ok("Telegram access", settings.telegram_access_description)
    elif settings.allow_all_users:
        warn("Telegram access", settings.telegram_access_description)
    else:
        warn("Telegram access", "DMs blocked until ALLOWED_USERS is set")

    print("KAOS doctor\n")
    failed = 0
    for status, name, detail in checks:
        print(f"[{status}] {name}: {detail}")
        if status == "FAIL":
            failed += 1

    if failed:
        print(f"\n{failed} hard check(s) failed.")
        return 1

    print("\nNo hard blockers found.")
    return 0


def _agent_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-").lower()
    return slug


def _display_name(slug: str) -> str:
    return " ".join(part.capitalize() for part in re.split(r"[-_]+", slug) if part) or slug


def _repo_root() -> Path:
    return Path.cwd()


def _agent_template_dir(template: str) -> Path:
    return _repo_root() / "templates" / "agents" / template


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _available_agent_templates() -> list[Path]:
    root = _repo_root() / "templates" / "agents"
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if (path / "template.yaml").is_file())


def run_templates(command: str, name: str = "", workspace: str = "", role: str = "", force: bool = False, dry_run: bool = False) -> int:
    if command == "list":
        templates = _available_agent_templates()
        if not templates:
            print("No agent templates found.")
            return 1
        print("KAOS agent templates\n")
        for path in templates:
            meta = _load_yaml(path / "template.yaml")
            print(f"- {path.name}: {meta.get('description', 'No description')}")
        print("\nNext:")
        print("  kaos templates show personal-operator")
        print("  kaos templates install personal-operator personal-demo --force")
        return 0

    template_dir = _agent_template_dir(name)
    meta = _load_yaml(template_dir / "template.yaml")
    if not meta:
        print(f"Template not found: {name}")
        return 1

    if command == "show":
        print(f"{meta.get('name', name)}")
        print(f"Role: {meta.get('role', 'general-purpose local AI agent')}")
        print(f"Description: {meta.get('description', '')}\n")
        for section in ("skills", "capability_policy", "example_prompts"):
            values = meta.get(section, [])
            if values:
                print(section.replace("_", " ").title() + ":")
                for value in values:
                    print(f"  - {value}")
                print()
        return 0

    if command == "install":
        if not workspace:
            print("Workspace name is required.")
            return 1
        template_role = role or str(meta.get("role", "general-purpose local AI agent"))
        result = run_init(workspace, role=template_role, force=force, dry_run=dry_run)
        if result != 0 or dry_run:
            return result
        workspace_dir = _repo_root() / "workspaces" / _agent_slug(workspace)
        profile = workspace_dir / "ops" / "TEMPLATE.md"
        lines = [
            f"# Template: {meta.get('name', name)}",
            "",
            str(meta.get("description", "")),
            "",
            "## Memory Defaults",
        ]
        for item in meta.get("memory_defaults", []):
            lines.append(f"- {item}")
        lines.append("")
        lines.append("## Capability Policy")
        for item in meta.get("capability_policy", []):
            lines.append(f"- {item}")
        lines.append("")
        lines.append("## Example Prompts")
        for item in meta.get("example_prompts", []):
            lines.append(f"- {item}")
        profile.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        print(f"\nTemplate profile written: {profile}")
        return 0

    print(f"Unknown templates command: {command}")
    return 1


def _available_skill_packs() -> list[Path]:
    root = _repo_root() / "templates" / "skill-packs"
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if (path / "pack.yaml").is_file())


def _skill_pack_dir(name: str) -> Path:
    return _repo_root() / "templates" / "skill-packs" / name


def _skills_workspace_root(agent: str = "") -> Path:
    if agent:
        return _repo_root() / "workspaces" / _agent_slug(agent)
    if settings.workspace_path:
        return Path(settings.workspace_path)
    return _repo_root() / "workspaces" / _agent_slug(settings.agent_name)


def run_skills(
    command: str,
    pack: str = "",
    agent: str = "",
    force: bool = False,
    dry_run: bool = False,
    source: str = "",
    skill: str = "",
    output: str = "",
) -> int:
    if command == "packs":
        packs = _available_skill_packs()
        if not packs:
            print("No skill packs found.")
            return 1
        print("KAOS skill packs\n")
        for path in packs:
            meta = _load_yaml(path / "pack.yaml")
            print(f"- {path.name}: {meta.get('description', 'No description')}")
        print("\nNext:")
        print("  kaos skills show-pack productivity")
        print("  kaos skills install-pack productivity --agent personal-demo --force")
        return 0

    if command == "import":
        from kronos.skills.hub import import_skill
        from kronos.skills.store import SkillStore

        workspace_root = _skills_workspace_root(agent)
        store = SkillStore(str(workspace_root))
        print(f"Importing skill into {workspace_root / 'self' / 'skills'}")
        result = import_skill(source, store)
        print(result)
        return 0 if "imported successfully" in result else 1

    if command == "export":
        from kronos.skills.hub import export_skill
        from kronos.skills.store import SkillStore

        workspace_root = _skills_workspace_root(agent)
        store = SkillStore(str(workspace_root))
        content = export_skill(skill, store)
        if content is None:
            print(f"Skill '{skill}' not found in {workspace_root / 'self' / 'skills'}")
            return 1
        if output:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(f"Exported skill '{skill}' to {out_path}")
        else:
            print(content)
        return 0

    pack_dir = _skill_pack_dir(pack)
    meta = _load_yaml(pack_dir / "pack.yaml")
    if not meta:
        print(f"Skill pack not found: {pack}")
        return 1

    if command == "show-pack":
        print(f"{meta.get('name', pack)}")
        print(f"Description: {meta.get('description', '')}\n")
        for section in ("capabilities", "skills", "examples"):
            values = meta.get(section, [])
            if values:
                print(section.title() + ":")
                for value in values:
                    print(f"  - {value}")
                print()
        return 0

    if command == "install-pack":
        target_agent = _agent_slug(agent or settings.agent_name)
        skills_src = pack_dir / "skills"
        if not skills_src.is_dir():
            print(f"Pack has no skills directory: {pack}")
            return 1
        dest_root = _repo_root() / "workspaces" / target_agent / "self" / "skills"
        print(f"Installing skill pack '{pack}' into {dest_root}")
        if not dry_run:
            dest_root.mkdir(parents=True, exist_ok=True)
        for skill_dir in sorted(path for path in skills_src.iterdir() if path.is_dir()):
            dest = dest_root / skill_dir.name
            if dest.exists() and not force:
                print(f"Skip existing skill: {skill_dir.name} (use --force to overwrite)")
                continue
            if dry_run:
                print(f"Would install: {skill_dir.name}")
                continue
            shutil.copytree(skill_dir, dest, dirs_exist_ok=force)
            print(f"Installed: {skill_dir.name}")
        return 0

    print(f"Unknown skills command: {command}")
    return 1


def run_init(name: str, role: str, force: bool = False, dry_run: bool = False) -> int:
    """Create a new local KAOS workspace from the bundled template."""
    slug = _agent_slug(name)
    if not slug:
        print("Invalid agent name. Use letters, numbers, dash, or underscore.")
        return 1

    repo_root = _repo_root()
    template = repo_root / "workspaces" / "_template"
    dest = repo_root / "workspaces" / slug

    if not template.exists():
        print(f"Template not found: {template}")
        return 1

    if dest.exists() and not force:
        print(f"Workspace already exists: {dest}")
        print("Use --force to merge template files into it.")
        return 1

    print(f"KAOS init: {slug}")
    print(f"Workspace: {dest}")

    if dry_run:
        print("Dry run only. No files were written.")
        return 0

    shutil.copytree(template, dest, dirs_exist_ok=force)

    display_name = _display_name(slug)
    replacements = {
        "{Agent Name}": display_name,
        "{One-line role description}": role,
        "{domain expertise}": role,
    }

    for path in dest.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")

    for directory in [
        dest / "notes" / "user",
        dest / "notes" / "world",
        dest / "notes" / "inbox",
        dest / "ops" / "sessions",
        dest / "ops" / "queue",
        dest / "ops" / "dynamic_tools",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    for path, content in {
        dest / "notes" / "user" / "USER.md": f"# User Model for {display_name}\n\nAdd durable user facts here.\n",
        dest / "ops" / "HEARTBEAT.md": f"# {display_name} Heartbeat\n\nRuntime notes and health updates.\n",
        dest / "ops" / "TOOLS.md": "# Tools\n\nDocument enabled tools and capability decisions here.\n",
    }.items():
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    print("\nNext steps:")
    print(f"  Edit workspaces/{slug}/self/IDENTITY.md")
    print(f"  kaos skills install-pack productivity --agent {slug} --force")
    print(f"  AGENT_NAME={slug} kaos doctor")
    print(f"  AGENT_NAME={slug} kaos chat")
    print("\nOptional: add this agent to agents.yaml for swarm/group routing.")
    return 0


def run_connect_telegram() -> int:
    """Print guided Telegram setup checks without exposing secrets."""
    print("KAOS Telegram connector check\n")

    checks = [
        ("TG_API_ID", bool(settings.tg_api_id)),
        ("TG_API_HASH", bool(settings.tg_api_hash)),
    ]
    for name, present in checks:
        status = "OK" if present else "MISSING"
        print(f"[{status}] {name}")

    if settings.allowed_users:
        print(f"[OK] Telegram access: {settings.telegram_access_description}")
        print("[OK] ALLOW_ALL_USERS=false (safe default)")
    elif settings.allow_all_users:
        print("[WARN] Telegram access: ALL (ALLOW_ALL_USERS=true)")
    else:
        print("[WARN] Telegram access: DMs blocked until ALLOWED_USERS is set")
        print("[OK] ALLOW_ALL_USERS=false (safe default)")

    print("\nSetup:")
    print("  1. Create Telegram API credentials at https://my.telegram.org")
    print("  2. Put TG_API_ID and TG_API_HASH into .env")
    print("  3. Set ALLOWED_USERS to comma-separated Telegram user IDs")
    print("  4. Run: python scripts/auth-userbot.py")
    print("  5. Run: python -m kronos")

    if not settings.allowed_users and not settings.allow_all_users:
        print("\nWarning: DMs are blocked until ALLOWED_USERS is set or ALLOW_ALL_USERS=true.")
    elif settings.allow_all_users:
        print("\nWarning: ALLOW_ALL_USERS=true allows any Telegram user who can message this account.")

    return 0


def run_dashboard_command() -> int:
    """Start the local dashboard API/UI without starting Telegram bridges."""
    _configure_logging()
    try:
        from dashboard.config import DASHBOARD_HOST, DASHBOARD_PORT
        from dashboard.server import run_dashboard
    except ModuleNotFoundError as e:
        print(f"Missing Python dependency: {e.name}")
        print('Install KAOS first with: pip install -e ".[dev]"')
        return 1

    print(f"Starting KAOS dashboard on http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print("Press Ctrl+C to stop.")
    try:
        asyncio.run(run_dashboard())
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    return 0


def run_demo_seed(data_dir: str, workspace: str, swarm_db: str, reset: bool) -> int:
    """Seed deterministic, public-safe dashboard demo state."""
    from kronos.demo_seed import seed_demo_state

    result = seed_demo_state(Path(data_dir), Path(workspace), Path(swarm_db), reset=reset)
    print("KAOS demo state seeded:")
    print(_format_tool_payload(result))
    print("\nRun dashboard with:")
    print(f"  AGENT_NAME=demo DB_DIR={result['data_dir']} DB_PATH={result['data_dir']}/session.db SWARM_DB_PATH={result['swarm_db']} WORKSPACE_PATH={result['workspace_dir']} kaos dashboard")
    return 0


async def _run_sessions_backfill_search(agent: str = "") -> int:
    """Backfill existing persisted sessions into the shared FTS search index."""
    from kronos.session import SessionStore

    target_agent = agent or settings.agent_name
    store = SessionStore(settings.db_path, agent_name=target_agent)
    indexed = await store.backfill_swarm_fts()
    print(f"Session search backfill complete: {indexed} new messages indexed for agent '{target_agent}'.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaos",
        description="Kronos Agent OS (KAOS) local control CLI",
    )
    parser.add_argument("--version", action="version", version=f"kaos {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="validate local environment and safety defaults")
    sub.add_parser("dashboard", help="start local dashboard API/UI")

    init = sub.add_parser("init", help="create a local agent workspace")
    init.add_argument("name", help="agent/workspace name, e.g. personal-operator")
    init.add_argument("--role", default="general-purpose local AI agent", help="one-line role description")
    init.add_argument("--force", action="store_true", help="merge into an existing workspace")
    init.add_argument("--dry-run", action="store_true", help="show what would be created without writing files")

    templates = sub.add_parser("templates", help="list, show, and install agent templates")
    templates_sub = templates.add_subparsers(dest="templates_command")
    templates_sub.add_parser("list", help="list bundled agent templates")
    template_show = templates_sub.add_parser("show", help="show an agent template")
    template_show.add_argument("name")
    template_install = templates_sub.add_parser("install", help="install an agent template into a workspace")
    template_install.add_argument("template")
    template_install.add_argument("workspace")
    template_install.add_argument("--role", default="", help="override the template role")
    template_install.add_argument("--force", action="store_true", help="merge into an existing workspace")
    template_install.add_argument("--dry-run", action="store_true", help="show what would be installed")

    skills = sub.add_parser("skills", help="list, show, and install skill packs")
    skills_sub = skills.add_subparsers(dest="skills_command")
    skills_sub.add_parser("packs", help="list bundled skill packs")
    skill_show = skills_sub.add_parser("show-pack", help="show a bundled skill pack")
    skill_show.add_argument("pack")
    skill_install = skills_sub.add_parser("install-pack", help="install a bundled skill pack")
    skill_install.add_argument("pack")
    skill_install.add_argument("--agent", default="", help="target AGENT_NAME/workspace; defaults to current settings")
    skill_install.add_argument("--force", action="store_true", help="overwrite existing skills")
    skill_install.add_argument("--dry-run", action="store_true", help="show what would be installed")
    skill_import = skills_sub.add_parser("import", help="import an external SKILL.md as a draft")
    skill_import.add_argument("source", help="URL to SKILL.md or github:user/repo/skill-name")
    skill_import.add_argument("--agent", default="", help="target AGENT_NAME/workspace; defaults to current settings")
    skill_export = skills_sub.add_parser("export", help="export a local skill as SKILL.md")
    skill_export.add_argument("skill")
    skill_export.add_argument("--agent", default="", help="source AGENT_NAME/workspace; defaults to current settings")
    skill_export.add_argument("--output", "-o", default="", help="write to file instead of stdout")

    chat = sub.add_parser("chat", help="start local CLI chat")
    chat.add_argument("--tools", action="store_true", help="load configured static MCP tools")
    chat.add_argument("--no-memory", action="store_true", help="disable long-term memory for this chat session")
    chat.add_argument("--prompt", "-p", help="send one message and exit")

    demo = sub.add_parser("demo", help="run safe local demo")
    demo.add_argument("--interactive", action="store_true", help="open deterministic offline demo prompt")
    demo.add_argument("--live", action="store_true", help="start real LLM-backed chat with demo safety gates")
    demo.add_argument("--tools", action="store_true", help="load configured static MCP tools in --live mode")

    demo_seed = sub.add_parser("demo-seed", help="seed deterministic dashboard demo state")
    demo_seed.add_argument("--data-dir", default="data/demo", help="target demo data directory")
    demo_seed.add_argument("--workspace", default="workspaces/demo", help="target demo workspace")
    demo_seed.add_argument("--swarm-db", default="data/demo/swarm.db", help="target demo swarm database")
    demo_seed.add_argument("--reset", action="store_true", help="delete existing demo data before seeding")

    sessions = sub.add_parser("sessions", help="maintain local session history")
    sessions_sub = sessions.add_subparsers(dest="sessions_command")
    sessions_backfill = sessions_sub.add_parser("backfill-search", help="backfill existing sessions into session_search")
    sessions_backfill.add_argument("--agent", default="", help="agent name to index under; defaults to AGENT_NAME")

    connect = sub.add_parser("connect", help="guided connector setup")
    connect_sub = connect.add_subparsers(dest="connector")
    connect_sub.add_parser("telegram", help="check Telegram connector setup")

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Backward compatibility with: python -m kronos.cli --tools
    if not argv or argv == ["--tools"]:
        use_tools = "--tools" in argv
        return asyncio.run(run_cli(use_tools=use_tools))

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return run_doctor()
    if args.command == "dashboard":
        return run_dashboard_command()
    if args.command == "init":
        return run_init(args.name, role=args.role, force=args.force, dry_run=args.dry_run)
    if args.command == "templates":
        if args.templates_command == "list":
            return run_templates("list")
        if args.templates_command == "show":
            return run_templates("show", name=args.name)
        if args.templates_command == "install":
            return run_templates(
                "install",
                name=args.template,
                workspace=args.workspace,
                role=args.role,
                force=args.force,
                dry_run=args.dry_run,
            )
        parser.parse_args(["templates", "--help"])
        return 0
    if args.command == "skills":
        if args.skills_command == "packs":
            return run_skills("packs")
        if args.skills_command == "show-pack":
            return run_skills("show-pack", pack=args.pack)
        if args.skills_command == "install-pack":
            return run_skills(
                "install-pack",
                pack=args.pack,
                agent=args.agent,
                force=args.force,
                dry_run=args.dry_run,
            )
        if args.skills_command == "import":
            return run_skills("import", source=args.source, agent=args.agent)
        if args.skills_command == "export":
            return run_skills("export", skill=args.skill, agent=args.agent, output=args.output)
        parser.parse_args(["skills", "--help"])
        return 0
    if args.command == "chat":
        return asyncio.run(run_cli(use_tools=args.tools, prompt=args.prompt, enable_memory=not args.no_memory))
    if args.command == "demo":
        return run_demo(interactive=args.interactive, live=args.live, use_tools=args.tools)
    if args.command == "demo-seed":
        return run_demo_seed(args.data_dir, args.workspace, args.swarm_db, reset=args.reset)
    if args.command == "sessions":
        if args.sessions_command == "backfill-search":
            return asyncio.run(_run_sessions_backfill_search(agent=args.agent))
        parser.parse_args(["sessions", "--help"])
        return 0
    if args.command == "connect":
        if args.connector == "telegram":
            return run_connect_telegram()
        parser.parse_args(["connect", "--help"])
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
