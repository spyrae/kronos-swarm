"""Skill Improve — weekly auto-improvement of skill files.

Reads audit log, matches interactions to skills by keywords,
proposes minimal improvements to SKILL.md files with versioned backups.
"""

import json
import logging
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from kronos.config import settings
from kronos.cron.notify import TOPIC_GENERAL, send_bot_api
from kronos.llm import ModelTier, get_model
from kronos.skills.store import Skill

log = logging.getLogger("kronos.cron.skill_improve")

LOOKBACK_DAYS = 7
MIN_INTERACTIONS = 3

# Keyword → skill mapping
SKILL_KEYWORDS = {
    "expense-tracker": ["расход", "expense", "трат", "бюджет", "budget", "потратил"],
    "investment-analysis": ["акци", "stock", "инвестиц", "портфел", "invest", "market"],
    "heartbeat": ["HEARTBEAT", "heartbeat"],
    "news-monitor": ["NEWS MONITOR", "дайджест", "новост"],
    "deep-research": ["исследуй", "research", "проверь идею", "анализ рынка"],
    "food-advisor": ["еда", "food", "калори", "рецепт", "диет"],
}


def _tokenize_skill_text(text: str) -> set[str]:
    import re

    return {
        token
        for token in re.findall(r"[a-zA-Zа-яА-Я0-9]{4,}", text.lower())
        if token not in {"skill", "auto", "created", "draft", "status"}
    }


def _match_skill(text: str, skills: list[Skill] | None = None) -> str | None:
    lower = text.lower()
    for skill, keywords in SKILL_KEYWORDS.items():
        if any(kw.lower() in lower for kw in keywords):
            return skill
    if not skills:
        return None
    text_tokens = _tokenize_skill_text(text)
    if not text_tokens:
        return None
    best_name = None
    best_score = 0
    for skill in skills:
        skill_tokens = _tokenize_skill_text(f"{skill.name} {skill.description}")
        score = len(text_tokens & skill_tokens)
        if score > best_score:
            best_name = skill.name
            best_score = score
    return best_name if best_score >= 2 else None


async def run_skill_improve() -> None:
    """Analyze interactions and improve relevant skills."""
    audit_file = Path(settings.db_path).parent / "logs" / "audit.jsonl"
    if not audit_file.exists():
        log.info("No audit log, skipping skill-improve")
        return

    cutoff = time.time() - (LOOKBACK_DAYS * 86400)

    from kronos.skills.store import SkillStore
    skill_store = SkillStore(settings.workspace_path)
    known_skills = skill_store.list_skills()

    # Collect interactions per skill
    skill_interactions: dict[str, list[dict]] = defaultdict(list)

    with open(audit_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                ts = entry.get("ts", "")
                if ts:
                    dt = datetime.fromisoformat(ts)
                    if dt.timestamp() < cutoff:
                        continue
                inp = entry.get("input_preview", "")
                skill = _match_skill(inp, known_skills)
                if skill:
                    skill_interactions[skill].append(entry)
            except (json.JSONDecodeError, ValueError):
                continue

    # Filter skills with enough interactions
    candidates = {
        skill: ints for skill, ints in skill_interactions.items()
        if len(ints) >= MIN_INTERACTIONS
    }

    if not candidates:
        log.info("No skills with >= %d interactions, skipping", MIN_INTERACTIONS)
        return

    improvements = []
    model = get_model(ModelTier.LITE)
    from langchain_core.messages import HumanMessage

    for skill_name, interactions in candidates.items():
        from kronos.workspace import ws
        skill = skill_store.get(skill_name)
        skill_path = skill.path if skill else ws.skill_path(skill_name)
        if not skill_path.exists():
            continue

        current_content = skill_path.read_text(encoding="utf-8")
        recent = interactions[-10:]
        interactions_text = "\n".join(
            f"- [{e.get('tier', '?')}] {e.get('input_preview', '')[:80]}"
            for e in recent
        )

        # Add feedback data for this skill
        feedback_text = ""
        try:
            from kronos.swarm_store import get_swarm
            swarm = get_swarm()
            satisfaction = swarm.get_satisfaction_rate(
                agent_name=settings.agent_name,
                days=LOOKBACK_DAYS,
            )
            feedback_text = (
                f"\n\nFeedback за {LOOKBACK_DAYS} дней: "
                f"{satisfaction['positive']}👍 / {satisfaction['negative']}👎 "
                f"(satisfaction: {satisfaction['satisfaction_rate']}%)"
            )
        except Exception:
            pass

        prompt = f"""Вот текущий SKILL.md для скилла "{skill_name}":

{current_content[:3000]}

Последние {len(recent)} взаимодействий:
{interactions_text}{feedback_text}

Предложи ОДНО минимальное улучшение для SKILL.md.
Если улучшения не нужны — ответь "без изменений".
Если есть предложение — верни полный обновлённый SKILL.md."""

        response = model.invoke([HumanMessage(content=prompt)])
        reply = response.content if isinstance(response.content, str) else str(response.content)

        if "без изменений" in reply.lower():
            continue

        # Backup current version
        versions_dir = skill_path.parent / ".versions"
        versions_dir.mkdir(exist_ok=True)
        existing = list(versions_dir.glob("SKILL.v*.md"))
        next_version = len(existing) + 1
        shutil.copy2(skill_path, versions_dir / f"SKILL.v{next_version}.md")

        # Write updated skill
        skill_path.write_text(reply, encoding="utf-8")
        improvements.append(f"**{skill_name}** — updated (v{next_version} backup)")
        log.info("Skill improved: %s (v%d backup)", skill_name, next_version)

    if improvements:
        text = "🔧 Skill Improvement\n\n" + "\n".join(f"• {i}" for i in improvements)
        send_bot_api(text, topic_id=TOPIC_GENERAL)
    else:
        log.info("No skill improvements needed")
