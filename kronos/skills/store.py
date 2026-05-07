"""Skill store — loads, indexes, and serves skill definitions.

Progressive disclosure:
- L1 (catalog): name + description — always in system prompt
- L2 (full): complete SKILL.md content — loaded via tool call
- L3 (references): supporting files — loaded on demand
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("kronos.skills.store")


@dataclass
class Skill:
    """A loaded skill definition."""

    name: str
    description: str
    content: str  # full SKILL.md body (without frontmatter)
    path: Path
    references: dict[str, Path] = field(default_factory=dict)  # name -> path
    status: str = "active"  # 'active' | 'draft'
    version: str = "1.0.0"
    author: str = ""
    tags: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    tier: str = ""
    imported_from: str = ""
    source_url: str = ""
    imported_at: str = ""
    review_required: bool = False


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter from markdown. Returns (metadata, body).

    Supports simple key: value, YAML folded scalars (>), and inline lists
    in bracket notation: key: [item1, item2].
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not match:
        return {}, text

    meta_text, body = match.group(1), match.group(2)
    meta: dict[str, str] = {}

    lines = meta_text.strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if ":" in line and not line[0].isspace():
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            if value in (">", "|", ">-", "|-"):
                # Multiline scalar — collect indented continuation lines
                parts = []
                i += 1
                while i < len(lines) and (lines[i].startswith("  ") or lines[i].startswith("\t")):
                    parts.append(lines[i].strip())
                    i += 1
                meta[key] = " ".join(parts)
                continue
            else:
                meta[key] = value
        i += 1

    return meta, body.strip()


def _parse_list_field(raw: str) -> list[str]:
    """Parse an inline YAML list value like '[item1, item2]' into a Python list.

    Also handles plain comma-separated strings without brackets.
    Returns an empty list for empty/missing values.
    """
    if not raw:
        return []
    stripped = raw.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    return [t.strip().strip("\"'") for t in stripped.split(",") if t.strip()]


def _format_frontmatter_value(value: object) -> str:
    """Format simple metadata values for YAML-ish frontmatter."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


class SkillStore:
    """Central store for all skills. Loads from workspace/self/skills/."""

    def __init__(self, workspace_path: str | Path | None = None):
        if workspace_path:
            from kronos.workspace import Workspace

            self._workspace = Workspace(workspace_path)
        else:
            from kronos.workspace import ws

            self._workspace = ws
        self._skills_dir = self._workspace.skills_dir
        self._skills: dict[str, Skill] = {}
        self._load_all()

    def _load_all(self) -> None:
        skills_dir = self._skills_dir
        if not skills_dir.is_dir():
            log.warning("Skills directory not found: %s", skills_dir)
            return

        for skill_dir in sorted(skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.is_file():
                continue

            raw = skill_file.read_text(encoding="utf-8").strip()
            meta, body = _parse_frontmatter(raw)

            name = meta.get("name", skill_dir.name)
            description = meta.get("description", "")
            status = meta.get("status", "active")
            version = meta.get("version", "1.0.0")
            author = meta.get("author", "")
            tags = _parse_list_field(meta.get("tags", ""))
            tools = _parse_list_field(meta.get("tools", ""))
            tier = meta.get("tier", "")
            imported_from = meta.get("imported_from", "")
            source_url = meta.get("source_url", "")
            imported_at = meta.get("imported_at", "")
            review_required = meta.get("review_required", "").lower() == "true"

            if not description:
                # Fallback: extract first paragraph as description
                first_para = body.split("\n\n")[0] if body else ""
                description = first_para[:200]

            # Discover reference files
            refs: dict[str, Path] = {}
            refs_dir = skill_dir / "references"
            if refs_dir.is_dir():
                for ref_file in refs_dir.iterdir():
                    if ref_file.is_file() and ref_file.suffix == ".md":
                        refs[ref_file.stem] = ref_file

            self._skills[name] = Skill(
                name=name,
                description=description,
                content=body,
                path=skill_file,
                references=refs,
                status=status,
                version=version,
                author=author,
                tags=tags,
                tools=tools,
                tier=tier,
                imported_from=imported_from,
                source_url=source_url,
                imported_at=imported_at,
                review_required=review_required,
            )

        log.info("Loaded %d skills: %s", len(self._skills), list(self._skills.keys()))
        self._generate_manifest_file()

    def list_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def get_reference(self, skill_name: str, ref_name: str) -> str | None:
        """Load a reference file for a skill."""
        skill = self._skills.get(skill_name)
        if not skill:
            return None
        ref_path = skill.references.get(ref_name)
        if not ref_path or not ref_path.is_file():
            return None
        return ref_path.read_text(encoding="utf-8").strip()

    def add_skill(self, name: str, content: str, meta: dict) -> Path:
        """Create a new skill file and register it in the store."""
        skill_dir = self._skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"

        # Build frontmatter
        fm_lines = ["---"]
        for k, v in meta.items():
            fm_lines.append(f"{k}: {_format_frontmatter_value(v)}")
        fm_lines.append("---")
        fm_lines.append("")
        fm_lines.append(content)

        skill_file.write_text("\n".join(fm_lines), encoding="utf-8")

        # Register in memory
        self._skills[name] = Skill(
            name=name,
            description=meta.get("description", ""),
            content=content,
            path=skill_file,
            status=meta.get("status", "active"),
            version=meta.get("version", "1.0.0"),
            author=meta.get("author", ""),
            tags=_parse_list_field(meta.get("tags", "")),
            tools=_parse_list_field(meta.get("tools", "")),
            tier=meta.get("tier", ""),
            imported_from=meta.get("imported_from", ""),
            source_url=meta.get("source_url", ""),
            imported_at=meta.get("imported_at", ""),
            review_required=str(meta.get("review_required", "")).lower() == "true",
        )
        log.info("Skill added: %s (status=%s)", name, meta.get("status", "active"))
        self._generate_manifest_file()
        return skill_file

    def update_status(self, name: str, status: str) -> bool:
        """Update a skill status in frontmatter and refresh the manifest."""
        skill = self._skills.get(name)
        if not skill:
            return False

        raw = skill.path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(raw)
        meta["status"] = status

        fm_lines = ["---"]
        for key, value in meta.items():
            fm_lines.append(f"{key}: {_format_frontmatter_value(value)}")
        fm_lines.extend(["---", "", body])
        skill.path.write_text("\n".join(fm_lines), encoding="utf-8")

        skill.status = status
        self._generate_manifest_file()
        return True

    def build_catalog(self) -> str:
        """Build L1 catalog string for system prompt injection.

        Compact format: name + description + available references + tags + tier.
        ~50-100 tokens per skill.
        """
        if not self._skills:
            return ""

        lines = []
        for skill in self._skills.values():
            refs_note = ""
            if skill.references:
                ref_names = ", ".join(skill.references.keys())
                refs_note = f" [refs: {ref_names}]"
            tier_note = f" ({skill.tier})" if skill.tier else ""
            tags_note = f" #{' #'.join(skill.tags)}" if skill.tags else ""
            status_note = " [draft; load for review before relying on it]" if skill.status == "draft" else ""
            lines.append(
                f"- **{skill.name}**{tier_note}: {skill.description}"
                f"{status_note}{refs_note}{tags_note}"
            )

        return "\n".join(lines)

    def generate_manifest(self) -> dict:
        """Generate skills.json manifest for the hub."""
        skills_list = []
        for skill in self._skills.values():
            row = {
                "name": skill.name,
                "version": skill.version,
                "description": skill.description,
                "author": skill.author,
                "tags": skill.tags,
                "tools": skill.tools,
                "tier": skill.tier,
                "status": skill.status,
            }
            if skill.imported_from:
                row["imported_from"] = skill.imported_from
            if skill.source_url:
                row["source_url"] = skill.source_url
            if skill.imported_at:
                row["imported_at"] = skill.imported_at
            if skill.review_required:
                row["review_required"] = True
            skills_list.append(row)
        return {
            "version": "1.0.0",
            "standard": "agentskills-compatible",
            "agent": self._workspace.root.name,
            "skills": skills_list,
            "generated_at": datetime.now(UTC).isoformat(),
        }

    def _generate_manifest_file(self) -> None:
        """Write skills.json manifest to skills directory."""
        import json

        manifest = self.generate_manifest()
        manifest_path = self._skills_dir / "skills.json"
        try:
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            log.warning("Failed to write skills.json: %s", e)
