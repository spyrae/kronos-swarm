"""Skills Hub — import/export skills following agentskills.io standard.

Supports:
- Import from URL or github:user/repo/skill-name
- Export skill as standalone package
- Manifest generation
"""

import logging
import re
import urllib.request
from datetime import UTC, datetime

from kronos.skills.store import SkillStore, _parse_frontmatter, _parse_list_field

log = logging.getLogger("kronos.skills.hub")

GITHUB_RAW_URL = "https://raw.githubusercontent.com/{user}/{repo}/main/{path}"
SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")


def _fetch_url(url: str, timeout: int = 15) -> str:
    """Fetch content from URL.

    Args:
        url: Target URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Decoded response body as string.

    Raises:
        urllib.error.URLError: On network errors.
        urllib.error.HTTPError: On non-2xx HTTP responses.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "kaos/1.0"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp.read().decode("utf-8")


def _parse_github_source(source: str) -> str | None:
    """Parse github:user/repo/path into a raw GitHub SKILL.md URL.

    Args:
        source: Source string in 'github:user/repo/skill-name' format.

    Returns:
        Raw GitHub URL or None if format does not match.
    """
    match = re.match(r"^github:([\w.-]+)/([\w.-]+)/(.+)$", source)
    if not match:
        return None
    user, repo, raw_path = match.groups()
    path = raw_path.strip().strip("/")
    if not path or ".." in path.split("/"):
        return None
    if not path.endswith("SKILL.md"):
        path = f"{path}/SKILL.md"
    return GITHUB_RAW_URL.format(user=user, repo=repo, path=path)


def _resolve_source(source: str) -> str | None:
    url = _parse_github_source(source)
    if url:
        return url
    if source.startswith(("http://", "https://")):
        return source
    return None


def _validate_skill_package(content: str) -> tuple[bool, str, dict[str, str], str]:
    """Validate a portable SKILL.md package."""
    if not content.strip():
        return False, "Empty skill content received.", {}, ""

    meta, body = _parse_frontmatter(content)
    name = meta.get("name", "").strip()
    if not name:
        return False, "Skill has no 'name' in frontmatter.", meta, body
    if not SKILL_NAME_RE.match(name):
        return (
            False,
            "Skill name must be a safe slug: lowercase letters, numbers, dash, or underscore.",
            meta,
            body,
        )

    description = meta.get("description", "").strip()
    if not description:
        return False, "Skill has no 'description' in frontmatter.", meta, body
    if not body.strip():
        return False, "Skill body is empty.", meta, body

    return True, "", meta, body


def import_skill(source: str, store: SkillStore) -> str:
    """Import a skill from URL or github:user/repo/skill-name.

    Fetches the remote SKILL.md, validates its frontmatter, checks for
    name conflicts, and registers the skill via the store.

    Args:
        source: URL to SKILL.md or 'github:user/repo/skill-name'.
        store: SkillStore instance to register the imported skill into.

    Returns:
        Human-readable status message describing the outcome.
    """
    # Resolve source to a concrete URL
    url = _resolve_source(source)
    if not url:
        return (
            f"Invalid source: '{source}'. "
            "Use a URL or 'github:user/repo/skill-name' format."
        )

    # Fetch SKILL.md content
    try:
        content = _fetch_url(url)
    except Exception as e:
        return f"Failed to fetch skill from '{url}': {e}"

    # Parse and validate frontmatter
    valid, reason, meta, body = _validate_skill_package(content)
    if not valid:
        return reason

    # Guard against name collisions
    name = meta["name"].strip()
    existing = store.get(name)
    if existing:
        return (
            f"Skill '{name}' already exists. Remove it first to re-import."
        )

    # External skills are reviewable drafts by default. Approval is explicit.
    original_status = meta.get("status", "active")
    tags = _parse_list_field(meta.get("tags", ""))
    for tag in ("external", "imported"):
        if tag not in tags:
            tags.append(tag)

    imported_at = datetime.now(UTC).isoformat()
    meta = {
        **meta,
        "status": "draft",
        "review_required": "true",
        "imported_from": source,
        "source_url": url,
        "imported_at": imported_at,
        "created_by": meta.get("created_by", "external-import"),
        "tags": "[" + ", ".join(tags) + "]",
    }
    if original_status and original_status != "draft":
        meta["imported_original_status"] = original_status

    # Log required tools for informational purposes (no enforcement yet).
    required_tools = _parse_list_field(meta.get("tools", ""))
    if required_tools:
        log.info("Imported skill '%s' requires tools: %s", name, required_tools)

    # Persist and register
    store.add_skill(name, body, meta)

    version = meta.get("version", "unknown")
    author = meta.get("author", "unknown")
    tools_display = ", ".join(required_tools) if required_tools else "none"

    return (
        f"Skill '{name}' v{version} imported successfully "
        f"as draft (author: {author}, tools: {tools_display}). "
        "Review it with load_skill, then approve_skill if it is safe."
    )


def export_skill(name: str, store: SkillStore) -> str | None:
    """Export a skill as its full SKILL.md content.

    Args:
        name: Skill name to export.
        store: SkillStore instance to look up the skill in.

    Returns:
        Full SKILL.md file content as string, or None if skill not found.
    """
    skill = store.get(name)
    if not skill:
        return None
    return skill.path.read_text(encoding="utf-8")
