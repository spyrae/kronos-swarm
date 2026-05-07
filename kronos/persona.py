"""Persona system — loads workspace files into system prompt.

Three-Space layout:
  self/  → IDENTITY, SOUL, AGENTS (who I am)
  notes/ → USER, MEMORY, USER-MODEL (what I know)
  ops/   → TOOLS, WORKFLOW_AUTO (what I do)

Skills are loaded via SkillStore with progressive disclosure:
- L1 (catalog with descriptions) → always in system prompt
- L2 (full protocol) → loaded via load_skill() tool call
- L3 (reference files) → loaded via load_skill_reference() tool call
"""

import logging

import kronos.workspace as _workspace

log = logging.getLogger("kronos.persona")

# Core persona file attribute names on Workspace, loaded in order
_CORE_ATTRS = ["identity", "soul", "user", "tools", "agents", "workflow"]
MAX_USER_MODEL_CHARS = 6000


def load_persona(workspace_path: str | None = None) -> str:
    """Load core persona files into a single system prompt string.

    Args:
        workspace_path: Ignored (kept for backward compatibility). Uses ws paths.
    """
    ws = _workspace.ws
    parts: list[str] = []
    for attr in _CORE_ATTRS:
        filepath = getattr(ws, attr)
        if filepath.is_file():
            content = filepath.read_text(encoding="utf-8").strip()
            if content:
                parts.append(content)
                log.info("Loaded persona: %s (%d chars)", filepath.name, len(content))
        else:
            log.debug("Persona file not found: %s", filepath)

    return "\n\n---\n\n".join(parts)


def load_memory(workspace_path: str | None = None) -> str:
    """Load MEMORY.md for long-term facts context."""
    ws = _workspace.ws
    if ws.memory.is_file():
        content = ws.memory.read_text(encoding="utf-8").strip()
        log.info("Loaded memory: %d chars", len(content))
        return content
    return ""


def load_user_model(workspace_path: str | None = None) -> str:
    """Load dialectic USER-MODEL.md context."""
    ws = _workspace.ws
    if ws.user_model.is_file():
        content = ws.user_model.read_text(encoding="utf-8").strip()
        if len(content) > MAX_USER_MODEL_CHARS:
            content = content[:MAX_USER_MODEL_CHARS].rstrip() + "\n\n[User model truncated]"
        log.info("Loaded user model: %d chars", len(content))
        return content
    return ""


def load_handoff(workspace_path: str | None = None) -> str:
    """Load handoff.md for session continuity after compaction/restart."""
    ws = _workspace.ws
    if ws.handoff.is_file():
        content = ws.handoff.read_text(encoding="utf-8").strip()
        if content:
            log.info("Loaded handoff: %d chars", len(content))
            return content
    return ""


def build_system_prompt(workspace_path: str | None = None, skill_catalog: str = "") -> str:
    """Build complete system prompt from persona + memory + skill catalog.

    Loading order follows WORKFLOW_AUTO.md:
    handoff → persona (IDENTITY, SOUL, USER, TOOLS, AGENTS, WORKFLOW_AUTO) → memory → skills

    Args:
        workspace_path: Ignored (kept for backward compatibility). Uses ws paths.
        skill_catalog: Pre-built L1 skill catalog string from SkillStore.
    """
    # 1. Handoff first (interrupted work context)
    handoff = load_handoff()

    # 2. Persona files
    persona = load_persona()

    # 3. Memory and dialectic user model
    memory = load_memory()
    user_model = load_user_model()

    sections = []

    if handoff:
        sections.append(f"# Session Handoff (from previous session)\n\n{handoff}")

    sections.append(persona)

    if memory:
        sections.append(f"# Long-term Memory\n\n{memory}")

    if user_model:
        sections.append(f"# Dialectic User Model\n\n{user_model}")

    if skill_catalog:
        sections.append(
            "# Available Skills\n\n"
            "When a user's request matches a skill below, call `load_skill(skill_name)` "
            "to get the full protocol, then follow it precisely.\n"
            "For reference data (watchlists, criteria, etc.), use `load_skill_reference(skill_name, ref_name)`.\n\n"
            f"{skill_catalog}"
        )

    return "\n\n---\n\n".join(sections)
