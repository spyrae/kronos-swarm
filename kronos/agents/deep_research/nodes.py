"""Nodes for the Deep Research agent graph.

Pipeline: classify → plan_queries → execute_searches → evaluate → synthesize

Search execution uses a ReAct agent (LLM decides how to call tools)
instead of hardcoded tool invocations. This lets the LLM use each tool's
correct parameter schema automatically.
"""

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from kronos.agents.deep_research.state import DeepResearchState, SearchResult
from kronos.engine import create_agent
from kronos.llm import ModelTier, get_model

log = logging.getLogger("kronos.agents.deep_research")

# Tools and search agent (set during graph build)
_tools: list[BaseTool] = []
_search_agent = None

MAX_ITERATIONS = 2
MIN_QUALITY_SCORE = 45
SELF_CORRECTION_SCORE = 75
MAX_CORRECTIONS = 1


def set_tools(tools: list[BaseTool], on_tool_event=None) -> None:
    """Register search tools and build a ReAct search agent."""
    global _tools, _search_agent
    _tools = [
        t for t in tools
        if any(kw in t.name.lower() for kw in (
            "brave", "exa", "fetch", "content", "extract", "reddit", "search", "transcript",
        ))
    ]
    if _tools:
        _search_agent = create_agent(
            model=get_model(ModelTier.LITE),
            tools=_tools,
            system_prompt="You are a search assistant. Execute the given search queries using the available tools. If a tool returns an error, skip it and try an alternative. Return all results.",
            name="search_executor",
            on_tool_event=on_tool_event,
        )
    log.info("Research tools registered: %d tools", len(_tools))


def classify_mode(state: DeepResearchState) -> DeepResearchState:
    """Classify research mode from user query."""
    # Find the actual user message (skip system/handoff messages)
    query = ""
    for msg in reversed(state["messages"]):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if isinstance(msg, HumanMessage) and "transferred" not in content.lower():
            query = content
            break
    if not query:
        # Fallback: use last message regardless
        last_msg = state["messages"][-1]
        query = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)

    model = get_model(ModelTier.LITE)
    prompt = f"""Classify this research request into exactly one mode.
Respond with ONLY the mode name, nothing else.

Modes:
- topic: deep dive into a subject ("расскажи про", "что такое", "как работает")
- validation: validate an idea ("проверь идею", "есть ли конкуренты", "стоит ли делать")
- market: market research ("анализ рынка", "pain points", "что людям нужно")
- competitive: analyze a competitor ("разбери [продукт]", "конкурентный анализ")
- trends: trend analysis ("тренды в", "что растёт", "trend analysis")

Request: {query}

Mode:"""

    response = model.invoke([HumanMessage(content=prompt)])
    mode_text = response.content.strip().lower()

    # Normalize
    mode_map = {
        "topic": "topic", "validation": "validation", "market": "market",
        "competitive": "competitive", "trends": "trends",
    }
    mode = mode_map.get(mode_text, "topic")

    log.info("Research mode: %s, topic: %s", mode, query[:80])
    return {"topic": query, "mode": mode, "iteration": 0, "search_results": [], "search_queries": []}


def plan_queries(state: DeepResearchState) -> DeepResearchState:
    """Plan search queries based on mode and topic."""
    model = get_model(ModelTier.LITE)
    available = [t.name for t in _tools]

    prompt = f"""Plan search queries for a {state['mode']} research on: "{state['topic']}"

Available search tools: {available}

Generate 5-7 search queries. For each, specify which tool to use.
Respond in JSON format:
[
  {{"query": "search query text", "tool": "brave"}},
  {{"query": "another query", "tool": "exa"}},
  ...
]

Rules:
- Use "brave" for broad web search and site:-specific searches
- Use "exa" for deep content search (academic, technical)
- Use "reddit" for community discussions and opinions
- Use "fetch" to extract full content from a specific URL
- Vary formulations — don't repeat the same query
- Use Russian and English queries as appropriate
- For validation: include site:github.com, site:producthunt.com searches

Return ONLY the JSON array, no other text."""

    response = model.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    # Parse JSON from response
    try:
        # Handle markdown code blocks
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        queries = json.loads(content)
    except (json.JSONDecodeError, IndexError):
        log.warning("Failed to parse search plan, using defaults")
        queries = [
            {"query": state["topic"], "tool": "brave"},
            {"query": f"{state['topic']} обзор анализ", "tool": "brave"},
            {"query": state["topic"], "tool": "exa"} if "exa" in available else {"query": state["topic"], "tool": "brave"},
        ]

    log.info("Planned %d queries for '%s'", len(queries), state["topic"][:50])
    return {"search_queries": queries}


async def execute_searches(state: DeepResearchState) -> DeepResearchState:
    """Execute searches via ReAct agent (LLM picks correct tool params)."""
    if not _search_agent:
        log.warning("No search agent available, skipping searches")
        return {"iteration": state.get("iteration", 0) + 1}

    results: list[SearchResult] = list(state.get("search_results", []))
    queries = state.get("search_queries", [])

    # Build a single prompt for the search agent with all queries
    queries_text = "\n".join(
        f"- {q.get('query', '')} (use {q.get('tool', 'any available')} tool)"
        for q in queries if q.get("query")
    )

    search_prompt = f"""Execute these search queries and return ALL results.
For each query, use the most appropriate tool. Return the raw results.

Queries:
{queries_text}

Execute each query one by one. Return all results you find."""

    try:
        agent_result = await _search_agent([HumanMessage(content=search_prompt)])

        # Extract content from search agent's messages
        for msg in agent_result.messages:
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if len(content) > 100:
                    results.append(SearchResult(
                        query=state["topic"],
                        source="search_agent",
                        content=content[:5000],
                        url="",
                    ))

        log.info("Search agent returned %d results, total chars: %d",
                 len(results), sum(len(r["content"]) for r in results))

    except Exception as e:
        log.error("Search agent failed: %s", e)

    return {"search_results": results, "iteration": state.get("iteration", 0) + 1}


def evaluate_quality(state: DeepResearchState) -> DeepResearchState:
    """Evaluate if we have enough data or need more searches."""
    results = state.get("search_results", [])
    iteration = state.get("iteration", 0)
    mode = state.get("mode", "topic")
    topic = state.get("topic", "")

    judged = _judge_research_quality(results, mode, topic)
    if judged:
        score, feedback = judged
    else:
        score, feedback = _score_research_quality(results, mode)

    log.info(
        "Quality evaluation: score=%d, results=%d, feedback=%s, iteration=%d",
        score, len(results), feedback[:200], iteration,
    )
    return {"quality_score": score, "quality_feedback": feedback}


def _judge_research_quality(
    results: list[SearchResult],
    mode: str,
    topic: str,
) -> tuple[int, str] | None:
    """Use a cheap LLM judge for source quality, falling back on any failure."""
    if not results:
        return None

    context = _build_research_context(results, max_chars=8000)
    prompt = f"""You are a strict quality judge for a Deep Research pipeline.

Research topic: {topic}
Mode: {mode}

Collected source material:
{context}

Score the collected material from 0 to 100 by these criteria:
- relevance: does the material answer the user's research request?
- completeness: are the important aspects covered for this mode?
- evidence: are claims backed by enough independent source material?

Return ONLY a JSON object:
{{
  "relevance": 0,
  "completeness": 0,
  "evidence": 0,
  "score": 0,
  "weak_areas": ["..."],
  "feedback": "Concrete instructions for search or synthesis."
}}"""

    try:
        model = get_model(ModelTier.LITE)
        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content if isinstance(response.content, str) else str(response.content)
        data = _parse_json_object(content)
    except Exception as e:
        log.warning("Quality judge failed, falling back to heuristic: %s", e)
        return None

    if not data:
        return None

    score = _coerce_score(data.get("score"))
    if score is None:
        subscores = [
            _coerce_score(data.get("relevance")),
            _coerce_score(data.get("completeness")),
            _coerce_score(data.get("evidence")),
        ]
        valid_subscores = [s for s in subscores if s is not None]
        if not valid_subscores:
            return None
        score = round(sum(valid_subscores) / len(valid_subscores))

    weak_areas_raw = data.get("weak_areas", [])
    weak_areas = weak_areas_raw if isinstance(weak_areas_raw, list) else [str(weak_areas_raw)]
    feedback = data.get("feedback", "")
    feedback_text = feedback if isinstance(feedback, str) else str(feedback)
    return _clamp_score(score), _format_judge_feedback(_clamp_score(score), weak_areas, feedback_text)


def _score_research_quality(results: list[SearchResult], mode: str = "topic") -> tuple[int, str]:
    """Score collected research data and explain concrete gaps."""
    total_chars = sum(len(r["content"]) for r in results)
    unique_sources = {r["source"] for r in results if r.get("source")}
    source_count = len(unique_sources)

    score = 20
    if total_chars > 10000 and source_count >= 2:
        score = 75
    elif total_chars > 5000:
        score = 60
    elif total_chars > 2000:
        score = 40

    gaps: list[str] = []
    strengths: list[str] = []

    if not results:
        gaps.append("No source data was collected.")
    if total_chars < 2000:
        gaps.append("Very little source material; claims may be under-supported.")
    elif total_chars < 5000:
        gaps.append("Limited source material; final report should avoid broad claims.")
    else:
        strengths.append(f"Collected {total_chars} source characters.")

    if source_count < 2:
        gaps.append("Fewer than two independent source types; cross-checking is weak.")
    else:
        strengths.append(f"Collected {source_count} source types: {', '.join(sorted(unique_sources))}.")

    if len(results) < 3:
        gaps.append("Few result blocks; synthesis should call out evidence limitations.")

    if mode == "validation":
        gaps.append("For validation, explicitly cover competitors, demand signal, risks, and go/no-go conditions.")
    elif mode == "market":
        gaps.append("For market research, explicitly cover audience, pain points, alternatives, and buying triggers.")
    elif mode == "competitive":
        gaps.append("For competitive research, explicitly cover positioning, differentiators, pricing, and weaknesses.")
    elif mode == "trends":
        gaps.append("For trends, distinguish current evidence from speculation and include counter-signals.")

    if score >= SELF_CORRECTION_SCORE and not gaps:
        return score, "Quality looks sufficient: enough material and source diversity for synthesis."

    sections = []
    if strengths:
        sections.append("Strengths:\n" + "\n".join(f"- {item}" for item in strengths))
    if gaps:
        sections.append("Gaps to address:\n" + "\n".join(f"- {item}" for item in gaps))
    sections.append(
        "Synthesis instruction: address these gaps directly, label weak evidence, "
        "and avoid unsupported certainty."
    )
    return score, "\n\n".join(sections)


def _build_research_context(results: list[SearchResult], max_chars: int) -> str:
    """Build compact source context for judge and synthesis prompts."""
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"[Source {i}: {r['source']}] Query: {r['query']}\n{r['content'][:2000]}"
        )
    return "\n\n---\n\n".join(context_parts)[:max_chars]


def _parse_json_object(content: str) -> dict | None:
    """Parse a JSON object from a model response."""
    cleaned = content.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_score(value) -> int | None:
    """Convert model score fields to int when possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return round(value)
    if isinstance(value, str):
        try:
            return round(float(value.strip()))
        except ValueError:
            return None
    return None


def _clamp_score(score: int) -> int:
    return max(0, min(100, score))


def _format_judge_feedback(score: int, weak_areas: list, feedback: str) -> str:
    """Format judge output for later search planning and synthesis."""
    sections = [f"LLM judge score: {score}/100."]
    cleaned_weak_areas = [str(item).strip() for item in weak_areas if str(item).strip()]
    if cleaned_weak_areas:
        sections.append("Weak areas:\n" + "\n".join(f"- {item}" for item in cleaned_weak_areas))
    if feedback.strip():
        sections.append(f"Feedback:\n{feedback.strip()}")
    sections.append(
        "Synthesis instruction: address these gaps directly, label weak evidence, "
        "and avoid unsupported certainty."
    )
    return "\n\n".join(sections)


def should_search_more(state: DeepResearchState) -> str:
    """Decide: search more or synthesize."""
    score = state.get("quality_score", 0)
    iteration = state.get("iteration", 0)

    if score >= MIN_QUALITY_SCORE or iteration >= MAX_ITERATIONS:
        return "synthesize"
    return "plan_more_queries"


def should_self_correct(state: DeepResearchState) -> bool:
    """Decide if the final draft needs one bounded self-correction pass."""
    score = state.get("quality_score", 0)
    return (
        MIN_QUALITY_SCORE <= score < SELF_CORRECTION_SCORE
        and state.get("correction_count", 0) < MAX_CORRECTIONS
        and bool(state.get("report"))
        and bool(state.get("quality_feedback"))
    )


def plan_more_queries(state: DeepResearchState) -> DeepResearchState:
    """Plan additional queries based on gaps in current results."""
    model = get_model(ModelTier.LITE)
    existing = [r["query"] for r in state.get("search_results", [])]
    feedback = state.get("quality_feedback", "")

    prompt = f"""I'm researching: "{state['topic']}" (mode: {state['mode']})

Already searched:
{chr(10).join(f'- {q}' for q in existing)}

Quality feedback:
{feedback or "The current data is insufficient."}

Plan 3 additional search queries that:
- Cover different angles than what we already have
- Are more specific or targeted
- Use different tools if possible
- Directly address the quality feedback above

Available tools: {[t.name for t in _tools]}

Respond in JSON format: [{{"query": "...", "tool": "..."}}]
Return ONLY the JSON array."""

    response = model.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    try:
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        queries = json.loads(content)
    except (json.JSONDecodeError, IndexError):
        queries = [{"query": f"{state['topic']} detailed analysis", "tool": "brave"}]

    return {"search_queries": queries}


def synthesize_report(state: DeepResearchState) -> DeepResearchState:
    """Synthesize all search results into a structured report."""
    model = get_model(ModelTier.STANDARD)
    results = state.get("search_results", [])

    context = _build_research_context(results, max_chars=12000)

    # Load report format from skill definition
    mode = state.get("mode", "topic")
    quality_score = state.get("quality_score", 0)
    quality_feedback = state.get("quality_feedback", "")
    quality_feedback_block = ""
    if quality_feedback and quality_score < SELF_CORRECTION_SCORE:
        quality_feedback_block = f"Structured feedback для self-correction:\n{quality_feedback}"

    from kronos.workspace import ws
    skill_path = str(ws.skill_path("deep-research"))

    try:
        with open(skill_path, encoding="utf-8") as f:
            skill_content = f.read()
    except FileNotFoundError:
        skill_content = ""

    prompt = f"""Ты — Deep Research Agent (INTJ). Составь структурированный отчёт.

Тема: {state['topic']}
Режим: {mode}

{f"Следуй формату отчёта из skill definition для режима '{mode}':" if skill_content else ""}
{skill_content[:3000] if skill_content else ""}

Собранные данные ({len(results)} источников):

{context[:12000]}

Оценка качества собранных данных: {quality_score}/100
{quality_feedback_block}

Правила:
- Только факты с источниками. Не выдумывай.
- Если данных нет — пиши "данных нет"
- Если quality feedback указывает на пробелы — явно закрой их в отчёте или пометь как limitation.
- При score ниже {SELF_CORRECTION_SCORE} сначала исправь слабые места: добавь caveats, missing evidence, risks, and next checks.
- Начинай с TL;DR
- Русский язык, термины на EN
- Actionable recommendations в конце"""

    response = model.invoke([HumanMessage(content=prompt)])
    report = response.content if isinstance(response.content, str) else str(response.content)

    log.info("Report synthesized: %d chars from %d sources", len(report), len(results))
    return {
        "report": report,
        "messages": [AIMessage(content=report)],
    }


def self_correct_report(state: DeepResearchState) -> DeepResearchState:
    """Rewrite the report once using judge feedback, without doing new search."""
    report = state.get("report", "")
    quality_feedback = state.get("quality_feedback", "")
    correction_count = state.get("correction_count", 0)
    if not report or not quality_feedback:
        return {"correction_count": correction_count}

    prompt = f"""Ты — Deep Research Agent. Улучши draft отчёта по feedback от quality judge.

Оригинальная тема: {state.get("topic", "")}
Режим: {state.get("mode", "topic")}
Quality score: {state.get("quality_score", 0)}/100

Judge feedback:
{quality_feedback}

Draft report:
{report[:12000]}

Правила:
- Не добавляй факты, которых нет в draft или source-derived feedback.
- Исправь слабые места, добавь limitations/caveats, риски и next checks.
- Сохрани русский язык и структуру с TL;DR.
- Верни только улучшенный отчёт."""

    try:
        model = get_model(ModelTier.STANDARD)
        response = model.invoke([HumanMessage(content=prompt)])
    except Exception as e:
        log.warning("Self-correction failed, keeping draft report: %s", e)
        return {"correction_count": correction_count}

    corrected = response.content if isinstance(response.content, str) else str(response.content)
    log.info("Report self-corrected: %d -> %d chars", len(report), len(corrected))
    return {
        "report": corrected,
        "messages": [AIMessage(content=corrected)],
        "correction_count": correction_count + 1,
    }
