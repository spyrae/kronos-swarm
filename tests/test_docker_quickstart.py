from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compose_uses_localhost_only_host_ports():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["kaos"]

    ports = service["ports"]
    assert ports
    assert all(str(port).startswith("127.0.0.1:") for port in ports)


def test_compose_does_not_mount_optional_local_agent_registry():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["kaos"]

    volumes = service.get("volumes", [])
    assert not any(str(volume).startswith("./agents.yaml:") for volume in volumes)
    assert not any(str(volume).startswith("./servers.yaml:") for volume in volumes)


def test_compose_sets_container_ports_explicitly():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["kaos"]
    env = service["environment"]

    assert service["command"] == ["kaos", "dashboard"]
    assert "WEBHOOK_PORT=8788" in env
    assert "DASHBOARD_PORT=8789" in env
    assert "DASHBOARD_HOST=0.0.0.0" in env


def test_compose_example_uses_generic_agent_names():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "kaos-worker" in text
    assert "nex" + "us" not in text


def test_dockerfile_installs_after_copying_package_source():
    lines = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()

    install_idx = next(i for i, line in enumerate(lines) if "pip install --no-cache-dir -e ." in line)
    metadata_copy_idx = next(i for i, line in enumerate(lines) if line.strip() == "COPY pyproject.toml README.md LICENSE MANIFEST.in ./")
    kronos_copy_idx = next(i for i, line in enumerate(lines) if line.strip() == "COPY kronos ./kronos")
    dashboard_copy_idx = next(i for i, line in enumerate(lines) if line.strip() == "COPY dashboard ./dashboard")
    aso_copy_idx = next(i for i, line in enumerate(lines) if line.strip() == "COPY aso ./aso")

    assert metadata_copy_idx < install_idx
    assert kronos_copy_idx < install_idx
    assert dashboard_copy_idx < install_idx
    assert aso_copy_idx < install_idx


def test_dockerfile_uses_lightweight_public_install_and_safe_default_command():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert 'pip install --no-cache-dir -e ".[memory]"' not in text
    assert 'CMD ["kaos", "dashboard"]' in text


def test_dockerfile_builds_dashboard_ui_after_full_copy():
    lines = (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()

    full_copy_idx = next(i for i, line in enumerate(lines) if line.strip() == "COPY . .")
    dashboard_build_idx = next(i for i, line in enumerate(lines) if "cd dashboard-ui" in line)
    npm_ci_idx = next(i for i, line in enumerate(lines) if "npm ci" in line)
    npm_build_idx = next(i for i, line in enumerate(lines) if "npm run build" in line)
    cleanup_idx = next(i for i, line in enumerate(lines) if "rm -rf node_modules" in line)

    assert full_copy_idx < dashboard_build_idx < npm_ci_idx < npm_build_idx < cleanup_idx


def test_dockerignore_excludes_private_runtime_state_but_keeps_template():
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in dockerignore
    assert "servers.yaml" in dockerignore
    assert "agents.yaml" in dockerignore
    assert "data/" in dockerignore
    assert "workspaces/*" in dockerignore
    assert "!workspaces/_template/**" in dockerignore


def test_sandbox_dockerfile_uses_non_root_pure_python_runner():
    text = (ROOT / "docker/sandbox/Dockerfile").read_text(encoding="utf-8")

    assert "PIP_NO_INDEX=1" in text
    assert "RUN pip install" not in text
    assert "USER 10001:10001" in text
    assert "COPY runner.py /sandbox/runner.py" in text
    assert 'CMD ["python", "/sandbox/runner.py"]' in text


def test_sandbox_build_script_allows_image_name_override():
    text = (ROOT / "scripts/build-sandbox.sh").read_text(encoding="utf-8")

    assert 'IMAGE_NAME="${SANDBOX_IMAGE:-kronos-sandbox:latest}"' in text
    assert 'docker build -t "$IMAGE_NAME" "$DOCKER_DIR"' in text
