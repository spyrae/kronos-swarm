"""Skill tools — LangChain tools for lazy skill loading.

These tools are added to the agent's toolset so the LLM can
load skill protocols on demand when it recognizes a matching request.
"""

import logging

from langchain_core.tools import tool

from kronos.skills.store import SkillStore

log = logging.getLogger("kronos.skills.tools")

# Module-level reference, set by init_skill_tools()
_store: SkillStore | None = None


def init_skill_tools(store: SkillStore) -> None:
    """Initialize skill tools with a SkillStore instance."""
    global _store
    _store = store


@tool
def load_skill(skill_name: str) -> str:
    """Load the full protocol for a skill. Use this when a user's request
    matches a skill from the Available Skills catalog. The skill protocol
    contains detailed instructions, pipeline steps, and output format
    that you must follow.

    Args:
        skill_name: Name of the skill to load (e.g. 'deep-research', 'food-advisor')
    """
    if not _store:
        return "Error: skill store not initialized"

    skill = _store.get(skill_name)
    if not skill:
        available = [s.name for s in _store.list_skills()]
        return f"Skill '{skill_name}' not found. Available: {', '.join(available)}"

    log.info("Loaded skill: %s (%d chars)", skill_name, len(skill.content))

    result = skill.content
    if skill.status == "draft":
        result = (
            "⚡ Это черновик навыка, созданного автоматически. "
            "Если он полезен — скажи 'одобрить skill " + skill_name + "'.\n\n"
            + result
        )
    return result


@tool
def load_skill_reference(skill_name: str, reference_name: str) -> str:
    """Load a reference file for a skill. Reference files contain supporting
    data like watchlists, criteria, budgets, etc. Only load when the skill
    protocol instructs you to use reference data.

    Args:
        skill_name: Name of the skill (e.g. 'news-monitor')
        reference_name: Name of the reference file (e.g. 'WATCHLIST', 'CRITERIA')
    """
    if not _store:
        return "Error: skill store not initialized"

    content = _store.get_reference(skill_name, reference_name)
    if content is None:
        skill = _store.get(skill_name)
        if not skill:
            return f"Skill '{skill_name}' not found"
        available_refs = list(skill.references.keys()) if skill.references else []
        return f"Reference '{reference_name}' not found for skill '{skill_name}'. Available: {available_refs}"

    log.info("Loaded reference: %s/%s (%d chars)", skill_name, reference_name, len(content))
    return content


@tool
def approve_skill(skill_name: str) -> str:
    """Approve a draft skill, changing its status from 'draft' to 'active'.

    Args:
        skill_name: Name of the draft skill to approve
    """
    if not _store:
        return "Error: skill store not initialized"

    skill = _store.get(skill_name)
    if not skill:
        return f"Skill '{skill_name}' not found."

    if skill.status != "draft":
        return f"Skill '{skill_name}' is already {skill.status}."

    if not _store.update_status(skill_name, "active"):
        return f"Skill '{skill_name}' could not be approved."

    log.info("Skill approved: %s", skill_name)
    return f"Skill '{skill_name}' одобрен и теперь активен."


@tool
def import_skill_from_source(source: str) -> str:
    """Import a skill from an external source (URL or GitHub).

    Fetches a remote SKILL.md, validates it, and registers the skill
    in the local store as a draft for review.

    Args:
        source: URL to a SKILL.md file, or 'github:user/repo/skill-name'
    """
    if not _store:
        return "Error: skill store not initialized"

    from kronos.skills.hub import import_skill

    return import_skill(source, _store)
