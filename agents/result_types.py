"""Result types for debate system components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from agents.heat import HeatMetrics


@dataclass
class DebateArgumentResult:
    """Result of generating a debate argument."""
    content: str
    strategy: str
    heat_score: float
    heat_metrics: HeatMetrics
    retrieved_passages: List[str] = None
    generation_params: Dict[str, Any] = None


@dataclass
class MediatorInterventionResult:
    """Result of a mediator intervention."""
    content: str
    cooling_effect: float
    should_continue: bool
    suggested_direction: Optional[str] = None
    intervention_type: str = "diplomatic"


@dataclass
class RetrievalResult:
    """Result from document retrieval."""
    passages: List[str]
    scores: List[float]
    sources: List[str]
    total_found: int


