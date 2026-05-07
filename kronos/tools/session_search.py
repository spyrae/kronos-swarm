"""Session search tool -- FTS5 search across all agent sessions."""

import logging
from datetime import datetime

from langchain_core.tools import tool

from kronos.llm import ModelTier, get_model
from kronos.swarm_store import get_swarm

log = logging.getLogger("kronos.tools.session_search")


@tool
def session_search(
    query: str,
    agent: str = "",
    days: int = 30,
    limit: int = 5,
    summarize: bool = True,
) -> str:
    """Search across conversation history of all agents. Use to find what was discussed before.

    Args:
        query: Search keywords (Russian or English)
        agent: Optional agent name filter (e.g. 'kronos', 'nexus')
        days: How many days back to search (default 30)
        limit: Max results (default 5)
        summarize: Whether to include a short LITE-model summary when available
    """
    if not query.strip():
        return "Укажи поисковый запрос."

    swarm = get_swarm()
    results = swarm.search_sessions(
        query=query,
        agent_name=agent,
        days=days,
        limit=limit,
    )

    if not results:
        return f"Ничего не найдено по запросу '{query}' за последние {days} дней."

    lines = []
    if summarize:
        summary = _summarize_results(query, results)
        if summary:
            lines.append(f"Краткое резюме:\n{summary}")

    for r in results:
        ts = datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M")
        snippet = r["content"][:200]
        lines.append(f"[{ts}] {r['agent_name']}/{r['role']}: {snippet}")

    return f"Найдено {len(results)} результатов:\n\n" + "\n---\n".join(lines)


def _summarize_results(query: str, results: list[dict]) -> str:
    """Best-effort summary. Search remains useful if LLM summarization fails."""
    snippets = []
    for r in results[:5]:
        ts = datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M")
        snippets.append(
            f"[{ts}] {r['agent_name']}/{r['role']}: {r['content'][:600]}"
        )
    prompt = f"""Summarize these session-search hits for the user.

Query: {query}

Hits:
{chr(10).join(snippets)}

Return 2-4 concise Russian bullet points. Mention uncertainty if the hits are weak."""
    try:
        from langchain_core.messages import HumanMessage

        model = get_model(ModelTier.LITE)
        response = model.invoke([HumanMessage(content=prompt)])
        summary = response.content if isinstance(response.content, str) else str(response.content)
        return summary.strip()
    except Exception as e:
        log.debug("Session-search summarization failed: %s", e)
        return ""
