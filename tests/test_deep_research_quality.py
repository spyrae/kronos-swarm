from langchain_core.messages import AIMessage, HumanMessage


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class CapturingModel:
    def __init__(self, content: str):
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        return FakeResponse(self.content)


def _state(**overrides):
    state = {
        "messages": [HumanMessage(content="Проверь идею AI travel planner")],
        "topic": "AI travel planner",
        "mode": "validation",
        "user_id": "test",
        "search_queries": [],
        "search_results": [],
        "iteration": 1,
        "report": "",
        "quality_score": 0,
        "quality_feedback": "",
        "correction_count": 0,
    }
    state.update(overrides)
    return state


def test_evaluate_quality_uses_lite_judge(monkeypatch):
    from kronos.agents.deep_research import nodes

    model = CapturingModel(
        '{"relevance": 70, "completeness": 50, "evidence": 60, '
        '"score": 60, "weak_areas": ["Need competitors"], '
        '"feedback": "Add demand signal and limitations."}'
    )
    monkeypatch.setattr(nodes, "get_model", lambda _tier: model)

    result = nodes.evaluate_quality(_state(search_results=[
        {"query": "q1", "source": "brave", "content": "x" * 6000, "url": ""},
    ]))

    assert result["quality_score"] == 60
    assert "LLM judge score: 60/100" in result["quality_feedback"]
    assert "Need competitors" in result["quality_feedback"]
    assert "Add demand signal and limitations." in result["quality_feedback"]
    assert "Score the collected material" in model.prompts[0]


def test_evaluate_quality_falls_back_to_heuristic_feedback(monkeypatch):
    from kronos.agents.deep_research import nodes

    def fail_get_model(_tier):
        raise RuntimeError("no model")

    monkeypatch.setattr(nodes, "get_model", fail_get_model)

    result = nodes.evaluate_quality(_state(search_results=[
        {"query": "q1", "source": "brave", "content": "x" * 6000, "url": ""},
    ]))

    assert result["quality_score"] == 60
    assert "Gaps to address" in result["quality_feedback"]
    assert "Fewer than two independent source types" in result["quality_feedback"]
    assert "competitors" in result["quality_feedback"]


def test_should_search_more_keeps_low_score_search_loop_but_synthesizes_medium_score():
    from kronos.agents.deep_research.nodes import should_search_more

    assert should_search_more({"quality_score": 40, "iteration": 1}) == "plan_more_queries"
    assert should_search_more({"quality_score": 60, "iteration": 1}) == "synthesize"


def test_should_self_correct_only_for_one_medium_score_retry():
    from kronos.agents.deep_research.nodes import should_self_correct

    base = _state(
        quality_score=60,
        quality_feedback="Need stronger evidence.",
        report="draft report",
    )

    assert should_self_correct(base)
    assert not should_self_correct({**base, "correction_count": 1})
    assert not should_self_correct({**base, "quality_score": 80})
    assert not should_self_correct({**base, "quality_score": 40})


def test_plan_more_queries_uses_quality_feedback(monkeypatch):
    from kronos.agents.deep_research import nodes

    model = CapturingModel('[{"query": "AI travel planner competitors", "tool": "brave"}]')
    monkeypatch.setattr(nodes, "get_model", lambda _tier: model)

    result = nodes.plan_more_queries(_state(
        search_results=[{"query": "q1", "source": "brave", "content": "short", "url": ""}],
        quality_feedback="Need independent competitors and demand signal.",
    ))

    assert result["search_queries"][0]["query"] == "AI travel planner competitors"
    assert "Need independent competitors and demand signal." in model.prompts[0]
    assert "Directly address the quality feedback" in model.prompts[0]


def test_synthesize_report_includes_self_correction_feedback(monkeypatch):
    from kronos.agents.deep_research import nodes

    model = CapturingModel("final report")
    monkeypatch.setattr(nodes, "get_model", lambda _tier: model)

    result = nodes.synthesize_report(_state(
        search_results=[{"query": "q1", "source": "brave", "content": "x" * 6000, "url": ""}],
        quality_score=60,
        quality_feedback="Need stronger evidence and explicit limitations.",
    ))

    assert result["report"] == "final report"
    assert result["messages"] == [AIMessage(content="final report")]
    assert "Structured feedback" in model.prompts[0]
    assert "Need stronger evidence and explicit limitations." in model.prompts[0]
    assert "сначала исправь слабые места" in model.prompts[0]


def test_self_correct_report_rewrites_draft_once(monkeypatch):
    from kronos.agents.deep_research import nodes

    model = CapturingModel("corrected report")
    monkeypatch.setattr(nodes, "get_model", lambda _tier: model)

    result = nodes.self_correct_report(_state(
        report="draft report",
        quality_score=60,
        quality_feedback="Need stronger evidence and explicit limitations.",
        correction_count=0,
    ))

    assert result["report"] == "corrected report"
    assert result["messages"] == [AIMessage(content="corrected report")]
    assert result["correction_count"] == 1
    assert "Need stronger evidence and explicit limitations." in model.prompts[0]
    assert "Draft report:" in model.prompts[0]
