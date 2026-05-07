"""Docker sandbox for dynamic tool execution.

Runs untrusted code in isolated Docker containers instead of exec() in-process.
Public-safe default fails closed if Docker is unavailable.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("kronos.tools.sandbox")

SANDBOX_IMAGE = "kronos-sandbox:latest"
SANDBOX_BUILD_SCRIPT = "scripts/build-sandbox.sh"
DEFAULT_TIMEOUT = 30
DEFAULT_MEMORY = "256m"


def _docker_available() -> bool:
    """Check if Docker is available."""
    return shutil.which("docker") is not None


def _docker_image_available(image: str = SANDBOX_IMAGE) -> bool:
    """Check if the sandbox image exists locally."""
    if not _docker_available():
        return False

    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def sandbox_ready() -> bool:
    """Return whether Docker and the local sandbox image are ready."""
    return _docker_available() and _docker_image_available()


def sandbox_status() -> dict[str, object]:
    """Return operator-facing sandbox readiness details."""
    docker_available = _docker_available()
    return {
        "docker_available": docker_available,
        "image": SANDBOX_IMAGE,
        "image_available": _docker_image_available() if docker_available else False,
        "build_script": SANDBOX_BUILD_SCRIPT,
    }


def sandbox_unavailable_message() -> str:
    """Return a concise remediation hint for sandbox setup."""
    status = sandbox_status()
    if not status["docker_available"]:
        return "Docker is required for dynamic tool sandboxing."
    return f"Sandbox image {SANDBOX_IMAGE} is missing. Run `{SANDBOX_BUILD_SCRIPT}`."


def build_sandbox_command(
    tmpdir: str,
    memory_limit: str = DEFAULT_MEMORY,
    network: bool = False,
) -> list[str]:
    """Build the Docker command for a single sandboxed execution."""
    network_flag = "bridge" if network else "none"
    return [
        "docker", "run",
        "--rm",
        f"--memory={memory_limit}",
        f"--network={network_flag}",
        "--cpus=1",
        "--pids-limit=50",
        "--read-only",
        "--cap-drop=ALL",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m",
        "--security-opt=no-new-privileges",
        "--user=10001:10001",
        "--workdir=/code",
        "-v", f"{tmpdir}:/code:ro",
        SANDBOX_IMAGE,
        "python", "/sandbox/runner.py",
    ]


async def execute_sandboxed(
    code: str,
    timeout: int = DEFAULT_TIMEOUT,
    memory_limit: str = DEFAULT_MEMORY,
    network: bool = False,
) -> tuple[str, str]:
    """Execute Python code in a Docker sandbox.

    Args:
        code: Python source code to execute
        timeout: Max execution time in seconds
        memory_limit: Docker memory limit (e.g. '256m')
        network: Whether to allow network access

    Returns:
        Tuple of (stdout, stderr)
    """
    from kronos.config import settings

    if not _docker_available():
        if settings.require_dynamic_tool_sandbox:
            return "", "Sandbox unavailable: Docker is required for dynamic tool execution."

        log.warning("Docker not available, falling back to in-process exec")
        return _exec_in_process(code, timeout)

    if not _docker_image_available():
        if settings.require_dynamic_tool_sandbox:
            return "", f"Sandbox unavailable: {sandbox_unavailable_message()}"

        log.warning("Docker sandbox image not available, falling back to in-process exec")
        return _exec_in_process(code, timeout)

    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="kronos-sandbox-")
        code_file = Path(tmpdir) / "tool.py"
        code_file.write_text(code, encoding="utf-8")

        cmd = build_sandbox_command(tmpdir, memory_limit=memory_limit, network=network)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return "", f"Execution timed out after {timeout}s"

        return (
            stdout.decode("utf-8", errors="replace").strip(),
            stderr.decode("utf-8", errors="replace").strip(),
        )

    except FileNotFoundError:
        from kronos.config import settings

        if settings.require_dynamic_tool_sandbox:
            return "", "Sandbox unavailable: Docker binary not found."

        log.warning("Docker binary not found, falling back to in-process exec")
        return _exec_in_process(code, timeout)
    except Exception as e:
        log.error("Sandbox execution failed: %s", e)
        return "", f"Sandbox error: {e}"
    finally:
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


def _exec_in_process(code: str, timeout: int = DEFAULT_TIMEOUT) -> tuple[str, str]:
    """Fallback: execute code in-process (unsafe, for dev/testing only)."""
    import io
    import sys

    log.warning("Executing dynamic tool in-process (no sandbox isolation)")

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_out = io.StringIO()
    sys.stderr = captured_err = io.StringIO()

    try:
        namespace: dict = {}
        exec(code, namespace)  # noqa: S102
        return captured_out.getvalue().strip(), captured_err.getvalue().strip()
    except Exception as e:
        return "", f"Execution error: {e}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
