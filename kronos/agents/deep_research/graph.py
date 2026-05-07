"""Deep Research agent — multi-step research pipeline.

Pipeline:
  classify_mode → plan_queries → execute_searches → evaluate_quality
                                      ↑                    ↓
                            plan_more_queries ← [need more?] → synthesize → self-correct?

No LangGraph — plain async workflow.
"""

import logging

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from kronos.agents.deep_research.nodes import (
    classify_mode,
    evaluate_quality,
    execute_searches,
    plan_more_queries,
    plan_queries,
    self_correct_report,
    set_tools,
    should_search_more,
    should_self_correct,
    synthesize_report,
)
from kronos.agents.deep_research.state import DeepResearchState
from kronos.engine import AgentResult

log = logging.getLogger("kronos.agents.deep_research")


def create_deep_research_agent(tools: list[BaseTool], on_tool_event=None):
    """Create the deep research agent as an async callable.

    Returns an async function that takes messages and returns AgentResult.
    """
    set_tools(tools, on_tool_event=on_tool_event)

    async def run(messages: list[BaseMessage]) -> AgentResult:
        """Run the deep research pipeline."""
        state: DeepResearchState = {
            "messages": list(messages),
            "topic": "",
            "mode": "topic",
            "user_id": "",
            "search_queries": [],
            "search_results": [],
            "iteration": 0,
            "report": "",
            "quality_score": 0,
            "quality_feedback": "",
            "correction_count": 0,
        }

        # Step 1: classify mode
        update = classify_mode(state)
        state.update(update)

        # Step 2: iterative search loop
        while True:
            # Plan queries
            update = plan_queries(state) if state["iteration"] == 0 else plan_more_queries(state)
            state.update(update)

            # Execute searches
            update = await execute_searches(state)
            state.update(update)

            # Evaluate quality
            update = evaluate_quality(state)
            state.update(update)

            # Check if we need more data
            decision = should_search_more(state)
            if decision == "synthesize":
                break
            # else: loop back to plan_more_queries

        # Step 3: synthesize report
        update = synthesize_report(state)
        state.update(update)

        if should_self_correct(state):
            update = self_correct_report(state)
            state.update(update)

        report = state.get("report", "")
        return AgentResult(
            messages=state["messages"] + [AIMessage(content=report)] if report else state["messages"],
            content=report or "Не удалось сформировать отчёт.",
        )

    run.__name__ = "deep_research_agent"
    run.__qualname__ = "deep_research_agent"
    log.info("Deep Research agent created (tools: %d)", len(tools))
    return run
