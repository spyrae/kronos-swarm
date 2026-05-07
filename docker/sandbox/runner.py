"""Runner script inside sandbox container.

Executes /code/tool.py in a restricted environment.
"""

import json
import sys
from pathlib import Path


def main() -> None:
    try:
        tool_path = Path("/code/tool.py")
        if not tool_path.exists():
            print("sandbox ready")
            return

        code = tool_path.read_text(encoding="utf-8")

        namespace = {"__builtins__": __builtins__}
        exec(code, namespace)  # noqa: S102

        # If the code defines a main() function, call it
        if "main" in namespace and callable(namespace["main"]):
            result = namespace["main"]()
            if result is not None:
                print(json.dumps(result) if not isinstance(result, str) else result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
