# state.py
from typing import TypedDict, List

class AgentState(TypedDict):
    query: str
    history: list
    context: list
    intent: str
    is_clarified: bool
    mode: str
    confidence_score: float
    research_data: list
    final_report: str
    token_usage: int
    budget_limit: int
    gaps: list
    iterations: int
    streaming_chunk: str  # For real-time token streaming
    query_id: str  # Unique ID for streaming buffer