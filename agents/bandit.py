"""Contextual bandit system for adaptive strategy learning."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class DebateOutcome(Enum):
    """Possible outcomes for a debate turn."""
    WIN = "win"           # Agent's argument was stronger
    LOSE = "lose"         # Opponent's argument was stronger  
    DRAW = "draw"         # Arguments were roughly equal
    INTERRUPTED = "interrupted"  # Mediator intervened


@dataclass
class StrategyResult:
    """Result of using a particular strategy."""
    strategy: str
    outcome: DebateOutcome
    heat_generated: float
    opponent_heat: float
    audience_engagement: float  # Placeholder for future audience scoring
    turn_number: int


@dataclass
class StrategyStats:
    """Statistics for a debate strategy."""
    name: str
    times_used: int
    wins: int
    losses: int
    draws: int
    interruptions: int
    avg_heat_generated: float
    avg_opponent_heat: float
    success_rate: float
    
    def update(self, result: StrategyResult) -> None:
        """Update statistics with a new result."""
        self.times_used += 1
        
        if result.outcome == DebateOutcome.WIN:
            self.wins += 1
        elif result.outcome == DebateOutcome.LOSE:
            self.losses += 1
        elif result.outcome == DebateOutcome.DRAW:
            self.draws += 1
        elif result.outcome == DebateOutcome.INTERRUPTED:
            self.interruptions += 1
        
        # Update averages
        self.avg_heat_generated = (
            (self.avg_heat_generated * (self.times_used - 1) + result.heat_generated) 
            / self.times_used
        )
        self.avg_opponent_heat = (
            (self.avg_opponent_heat * (self.times_used - 1) + result.opponent_heat)
            / self.times_used
        )
        
        # Calculate success rate (wins + draws, penalize interruptions)
        successful_outcomes = self.wins + self.draws * 0.5
        self.success_rate = successful_outcomes / self.times_used


class ContextualBandit:
    """Multi-armed bandit for learning optimal debate strategies."""
    
    def __init__(
        self, 
        strategies: List[str],
        epsilon: float = 0.1,
        learning_rate: float = 0.05,
        save_path: Optional[Path] = None
    ):
        self.strategies = strategies
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.save_path = save_path
        
        # Initialize strategy statistics
        self.strategy_stats: Dict[str, StrategyStats] = {}
        for strategy in strategies:
            self.strategy_stats[strategy] = StrategyStats(
                name=strategy,
                times_used=0,
                wins=0,
                losses=0,
                draws=0,
                interruptions=0,
                avg_heat_generated=0.0,
                avg_opponent_heat=0.0,
                success_rate=0.5  # Start with neutral assumption
            )
        
        # Context-aware adjustments
        self.context_modifiers: Dict[str, Dict[str, float]] = {
            "high_heat": {  # When conversation is already heated
                "logical_reasoning": 0.2,   # Boost logical approaches
                "emotional_appeal": -0.3,   # Reduce emotional approaches
                "citation_heavy": 0.1,      # Slightly boost evidence-based
                "questioning": 0.0,         # Neutral
                "analogy_based": -0.1       # Slightly reduce analogies
            },
            "early_debate": {  # First few turns
                "logical_reasoning": 0.1,
                "emotional_appeal": 0.0,
                "citation_heavy": 0.2,      # Start with strong evidence
                "questioning": 0.1,
                "analogy_based": 0.0
            },
            "late_debate": {  # Later turns, need to be more decisive
                "logical_reasoning": 0.0,
                "emotional_appeal": 0.1,
                "citation_heavy": -0.1,
                "questioning": -0.2,        # Less questioning, more asserting
                "analogy_based": 0.1
            }
        }
    
    def select_strategy(
        self, 
        context: Dict[str, any] = None
    ) -> str:
        """Select a strategy using epsilon-greedy with context awareness."""
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(self.strategies)
        
        # Calculate adjusted success rates based on context
        adjusted_rates = {}
        for strategy in self.strategies:
            base_rate = self.strategy_stats[strategy].success_rate
            
            # Apply context modifiers
            if context:
                modifier = 0.0
                
                # Heat-based adjustments
                if context.get('heat_level', 0) > 0.6:
                    modifier += self.context_modifiers["high_heat"].get(strategy, 0)
                
                # Turn-based adjustments
                turn_num = context.get('turn_number', 0)
                if turn_num <= 3:
                    modifier += self.context_modifiers["early_debate"].get(strategy, 0)
                elif turn_num >= 8:
                    modifier += self.context_modifiers["late_debate"].get(strategy, 0)
                
                adjusted_rates[strategy] = max(0.0, min(1.0, base_rate + modifier))
            else:
                adjusted_rates[strategy] = base_rate
        
        # Select strategy with highest adjusted rate
        best_strategy = max(adjusted_rates.keys(), key=lambda s: adjusted_rates[s])
        return best_strategy
    
    def update_strategy(
        self, 
        strategy: str, 
        result: StrategyResult
    ) -> None:
        """Update strategy statistics based on outcome."""
        if strategy in self.strategy_stats:
            self.strategy_stats[strategy].update(result)
            
            # Adjust epsilon based on learning progress
            self._adjust_epsilon()
            
            # Save progress if path is specified
            if self.save_path:
                self.save_stats()
    
    def _adjust_epsilon(self) -> None:
        """Gradually reduce exploration as we learn more."""
        total_trials = sum(stats.times_used for stats in self.strategy_stats.values())
        
        # Reduce epsilon as we gain experience
        if total_trials > 50:
            self.epsilon = max(0.05, self.epsilon * 0.995)
    
    def get_strategy_rankings(self) -> List[Tuple[str, float]]:
        """Get strategies ranked by success rate."""
        rankings = [
            (strategy, stats.success_rate) 
            for strategy, stats in self.strategy_stats.items()
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_recommendations(self, context: Dict[str, any] = None) -> Dict[str, str]:
        """Get strategy recommendations based on current learning."""
        rankings = self.get_strategy_rankings()
        
        recommendations = {
            "best_overall": rankings[0][0],
            "most_experienced": max(
                self.strategy_stats.keys(), 
                key=lambda s: self.strategy_stats[s].times_used
            ),
            "least_heat_generating": min(
                self.strategy_stats.keys(),
                key=lambda s: self.strategy_stats[s].avg_heat_generated
            )
        }
        
        if context:
            recommended = self.select_strategy(context)
            recommendations["context_recommended"] = recommended
        
        return recommendations
    
    def save_stats(self) -> None:
        """Save strategy statistics to file."""
        if not self.save_path:
            return
        
        data = {
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "strategies": self.strategies,
            "stats": {
                name: asdict(stats) 
                for name, stats in self.strategy_stats.items()
            }
        }
        
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_stats(self) -> bool:
        """Load strategy statistics from file."""
        if not self.save_path or not self.save_path.exists():
            return False
        
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            
            self.epsilon = data.get("epsilon", self.epsilon)
            self.learning_rate = data.get("learning_rate", self.learning_rate)
            
            # Load strategy stats
            for name, stats_dict in data.get("stats", {}).items():
                if name in self.strategy_stats:
                    self.strategy_stats[name] = StrategyStats(**stats_dict)
            
            return True
        except Exception as e:
            print(f"Failed to load bandit stats: {e}")
            return False
    
    def reset_stats(self) -> None:
        """Reset all strategy statistics."""
        for strategy in self.strategies:
            self.strategy_stats[strategy] = StrategyStats(
                name=strategy,
                times_used=0,
                wins=0,
                losses=0,
                draws=0,
                interruptions=0,
                avg_heat_generated=0.0,
                avg_opponent_heat=0.0,
                success_rate=0.5
            )
        self.epsilon = 0.1




