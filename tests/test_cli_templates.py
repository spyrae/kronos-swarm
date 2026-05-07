from pathlib import Path

from kronos.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_templates_list_and_show(capsys):
    result = main(["templates", "list"])
    out = capsys.readouterr().out

    assert result == 0
    assert "personal-operator" in out
    assert "research-agent" in out
    assert "kaos templates install personal-operator personal-demo --force" in out

    result = main(["templates", "show", "personal-operator"])
    out = capsys.readouterr().out

    assert result == 0
    assert "Personal Operator" in out
    assert "Capability Policy" in out


def test_template_install_dry_run_does_not_write(capsys):
    target = ROOT / "workspaces" / "template-smoke-agent"
    assert not target.exists()

    result = main([
        "templates",
        "install",
        "research-agent",
        "template-smoke-agent",
        "--dry-run",
    ])
    out = capsys.readouterr().out

    assert result == 0
    assert "Dry run only" in out
    assert not target.exists()


def test_skill_packs_list_show_and_dry_run(capsys):
    result = main(["skills", "packs"])
    out = capsys.readouterr().out

    assert result == 0
    assert "research" in out
    assert "finance-lite" in out
    assert "kaos skills install-pack productivity --agent personal-demo --force" in out

    result = main(["skills", "show-pack", "research"])
    out = capsys.readouterr().out

    assert result == 0
    assert "Research Pack" in out
    assert "research-brief" in out

    result = main(["skills", "install-pack", "research", "--agent", "template-smoke-agent", "--dry-run"])
    out = capsys.readouterr().out

    assert result == 0
    assert "Would install: research-brief" in out
    assert not (ROOT / "workspaces" / "template-smoke-agent").exists()


def test_skills_import_and_export_cli(tmp_path, monkeypatch, capsys):
    import kronos.cli as cli
    from kronos.skills import hub

    skill_md = """---
name: external-cli
description: External CLI skill
version: 1.2.3
author: Ada
---
# External CLI

Follow the reviewed protocol.
"""
    monkeypatch.setattr(cli, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(hub, "_fetch_url", lambda _url: skill_md)

    result = main(["skills", "import", "https://example.com/SKILL.md", "--agent", "hub-agent"])
    out = capsys.readouterr().out

    assert result == 0
    assert "imported successfully as draft" in out
    skill_file = tmp_path / "workspaces" / "hub-agent" / "self" / "skills" / "external-cli" / "SKILL.md"
    assert "status: draft" in skill_file.read_text(encoding="utf-8")

    export_path = tmp_path / "exported" / "SKILL.md"
    result = main([
        "skills",
        "export",
        "external-cli",
        "--agent",
        "hub-agent",
        "--output",
        str(export_path),
    ])
    out = capsys.readouterr().out

    assert result == 0
    assert "Exported skill 'external-cli'" in out
    assert "# External CLI" in export_path.read_text(encoding="utf-8")


def test_bundled_templates_and_packs_have_required_metadata():
    agent_templates = sorted((ROOT / "templates" / "agents").glob("*/template.yaml"))
    skill_packs = sorted((ROOT / "templates" / "skill-packs").glob("*/pack.yaml"))

    assert len(agent_templates) >= 5
    assert len(skill_packs) >= 5

    for path in agent_templates:
        text = path.read_text(encoding="utf-8")
        assert "capability_policy:" in text
        assert "memory_defaults:" in text
        assert "example_prompts:" in text

    for path in skill_packs:
        pack_dir = path.parent
        assert "capabilities:" in path.read_text(encoding="utf-8")
        assert (pack_dir / "fixtures" / "smoke.md").is_file()
        assert list((pack_dir / "skills").glob("*/SKILL.md"))
