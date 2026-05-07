"""State for the Deep Research agent pipeline."""

from typing import Literal, TypedDict

ResearchMode = Literal[
    "topic",        # глубокое погружение в тему
    "validation",   # проверка идеи (конкуренты, рынок)
    "market",       # анализ рынка и pain points
    "competitive",  # разбор конкурента
    "trends",       # трендовый анализ
]


class SearchResult(TypedDict):
    """A single search result with source."""
    query: str
    source: str  # brave, exa, reddit, youtube, content-core
    content: str
    url: str


class DeepResearchState(TypedDict):
    """State for the deep research pipeline."""

    messages: list  # list[BaseMessage]

    # Research parameters
    topic: str
    mode: ResearchMode
    user_id: str

    # Planning
    search_queries: list[str]  # planned queries

    # Execution
    search_results: list[SearchResult]  # collected data
    iteration: int  # current research iteration (max 3)

    # Output
    report: str  # final structured report
    quality_score: int  # 0-100, self-assessed quality
    quality_feedback: str  # structured gaps for search/synthesis correction
    correction_count: int  # bounded self-correction attempts
