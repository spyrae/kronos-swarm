import json


def _portable_skill(name: str = "external-research") -> str:
    return f"""---
name: {name}
description: Portable research workflow
version: 2.0.0
author: Ada
status: active
tags: [research]
tools: [session_search, brave_search]
---
# Portable Research

Use session history and search tools to produce a concise research brief.
"""


def test_parse_github_source_supports_skill_directories_and_files():
    from kronos.skills.hub import _parse_github_source

    assert _parse_github_source("github:acme/skills/research-brief") == (
        "https://raw.githubusercontent.com/acme/skills/main/research-brief/SKILL.md"
    )
    assert _parse_github_source("github:acme/skills/packs/research/SKILL.md") == (
        "https://raw.githubusercontent.com/acme/skills/main/packs/research/SKILL.md"
    )
    assert _parse_github_source("github:acme/skills/../secret") is None


def test_import_skill_from_url_is_reviewable_draft(tmp_path, monkeypatch):
    from kronos.skills import hub
    from kronos.skills.store import SkillStore

    monkeypatch.setattr(hub, "_fetch_url", lambda _url: _portable_skill())
    store = SkillStore(str(tmp_path / "workspace"))

    message = hub.import_skill("https://example.com/SKILL.md", store)

    assert "imported successfully as draft" in message
    skill = store.get("external-research")
    assert skill is not None
    assert skill.status == "draft"
    assert skill.imported_from == "https://example.com/SKILL.md"
    assert skill.source_url == "https://example.com/SKILL.md"
    assert skill.review_required is True
    assert skill.tags == ["research", "external", "imported"]

    raw = skill.path.read_text(encoding="utf-8")
    assert "status: draft" in raw
    assert "review_required: true" in raw
    assert "imported_original_status: active" in raw

    manifest = json.loads((tmp_path / "workspace" / "self" / "skills" / "skills.json").read_text(encoding="utf-8"))
    row = next(item for item in manifest["skills"] if item["name"] == "external-research")
    assert manifest["standard"] == "agentskills-compatible"
    assert row["status"] == "draft"
    assert row["source_url"] == "https://example.com/SKILL.md"
    assert row["review_required"] is True


def test_import_skill_rejects_unsafe_or_empty_packages(tmp_path, monkeypatch):
    from kronos.skills import hub
    from kronos.skills.store import SkillStore

    store = SkillStore(str(tmp_path / "workspace"))

    monkeypatch.setattr(hub, "_fetch_url", lambda _url: _portable_skill("Bad Name"))
    assert "safe slug" in hub.import_skill("https://example.com/bad.md", store)

    monkeypatch.setattr(hub, "_fetch_url", lambda _url: "---\nname: empty-skill\ndescription: Empty\n---\n")
    assert "body is empty" in hub.import_skill("https://example.com/empty.md", store)


def test_export_skill_returns_full_skill_markdown(tmp_path, monkeypatch):
    from kronos.skills import hub
    from kronos.skills.store import SkillStore

    monkeypatch.setattr(hub, "_fetch_url", lambda _url: _portable_skill())
    store = SkillStore(str(tmp_path / "workspace"))
    hub.import_skill("https://example.com/SKILL.md", store)

    exported = hub.export_skill("external-research", store)

    assert exported is not None
    assert "name: external-research" in exported
    assert "# Portable Research" in exported
