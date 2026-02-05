"""Strategy and passage selection utilities."""

from __future__ import annotations

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from rag.chunker import DocumentChunk


@dataclass
class SelectionCriteria:
    """Criteria for selecting passages and strategies."""
    relevance_threshold: float = 0.3
    diversity_weight: float = 0.2
    recency_weight: float = 0.1
    authority_weight: float = 0.1


class PassageSelector:
    """Selects the most relevant passages for debate arguments."""
    
    def __init__(self, criteria: SelectionCriteria = None):
        self.criteria = criteria or SelectionCriteria()
    
    def select_passages(
        self, 
        retrieved_passages: List[Tuple[DocumentChunk, float]],
        query: str,
        max_passages: int = 3,
        context: Dict[str, Any] = None
    ) -> List[str]:
        """Select the best passages for the current argument."""
        
        if not retrieved_passages:
            return []
        
        # Score passages based on multiple criteria
        scored_passages = []
        
        for chunk, relevance_score in retrieved_passages:
            # Base relevance score
            total_score = relevance_score
            
            # Diversity bonus (avoid repetitive sources)
            diversity_score = self._calculate_diversity_score(chunk, scored_passages)
            total_score += diversity_score * self.criteria.diversity_weight
            
            # Authority bonus (longer, more detailed passages might be more authoritative)
            authority_score = min(1.0, len(chunk.content) / 1000)
            total_score += authority_score * self.criteria.authority_weight
            
            # Context-specific adjustments
            if context:
                context_score = self._calculate_context_score(chunk, context)
                total_score += context_score * 0.1
            
            scored_passages.append((chunk, total_score))
        
        # Sort by total score and select top passages
        scored_passages.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        for chunk, score in scored_passages[:max_passages]:
            if score >= self.criteria.relevance_threshold:
                selected.append(chunk.content)
        
        return selected
    
    def _calculate_diversity_score(
        self, 
        chunk: DocumentChunk, 
        existing_passages: List[Tuple[DocumentChunk, float]]
    ) -> float:
        """Calculate diversity score to avoid repetitive sources."""
        if not existing_passages:
            return 1.0
        
        # Check source diversity
        existing_sources = {p[0].source_file for p in existing_passages}
        if chunk.source_file in existing_sources:
            return 0.5  # Penalty for same source
        
        # Check content similarity (simple word overlap)
        chunk_words = set(chunk.content.lower().split())
        
        max_overlap = 0
        for existing_chunk, _ in existing_passages:
            existing_words = set(existing_chunk.content.lower().split())
            overlap = len(chunk_words & existing_words) / len(chunk_words | existing_words)
            max_overlap = max(max_overlap, overlap)
        
        return 1.0 - max_overlap
    
    def _calculate_context_score(self, chunk: DocumentChunk, context: Dict[str, Any]) -> float:
        """Calculate context-specific relevance score."""
        score = 0.0
        
        # Strategy-specific preferences
        strategy = context.get('strategy', '')
        
        if strategy == 'citation_heavy':
            # Prefer chunks with numbers, statistics, studies
            if any(word in chunk.content.lower() for word in ['study', 'research', 'data', '%', 'percent']):
                score += 0.3
        
        elif strategy == 'emotional_appeal':
            # Prefer chunks with emotional language
            if any(word in chunk.content.lower() for word in ['impact', 'affect', 'people', 'community', 'family']):
                score += 0.3
        
        elif strategy == 'questioning':
            # Prefer chunks that raise questions or show uncertainty
            if any(word in chunk.content.lower() for word in ['question', 'unclear', 'debate', 'controversy']):
                score += 0.3
        
        # Heat level adjustments
        heat_level = context.get('heat_level', 0.0)
        if heat_level > 0.7:
            # When heated, prefer more factual, less controversial content
            if any(word in chunk.content.lower() for word in ['fact', 'evidence', 'proven', 'established']):
                score += 0.2
        
        return min(1.0, score)


class StrategySelector:
    """Selects debate strategies based on context and learning."""
    
    def __init__(self):
        self.strategy_contexts = {
            'logical_reasoning': {
                'best_when': ['early_debate', 'low_heat', 'factual_topic'],
                'avoid_when': ['high_heat', 'emotional_topic'],
                'effectiveness_factors': ['evidence_quality', 'opponent_logic_weakness']
            },
            'emotional_appeal': {
                'best_when': ['high_stakes_topic', 'personal_impact', 'mid_debate'],
                'avoid_when': ['technical_topic', 'high_heat'],
                'effectiveness_factors': ['audience_values', 'personal_stories']
            },
            'citation_heavy': {
                'best_when': ['disputed_facts', 'early_debate', 'academic_topic'],
                'avoid_when': ['late_debate', 'well_established_facts'],
                'effectiveness_factors': ['source_quality', 'data_availability']
            },
            'questioning': {
                'best_when': ['opponent_weak_point', 'complex_topic', 'mid_debate'],
                'avoid_when': ['need_strong_position', 'late_debate'],
                'effectiveness_factors': ['opponent_certainty', 'topic_complexity']
            },
            'analogy_based': {
                'best_when': ['complex_topic', 'abstract_concepts', 'audience_confusion'],
                'avoid_when': ['simple_topic', 'technical_accuracy_needed'],
                'effectiveness_factors': ['analogy_quality', 'audience_familiarity']
            }
        }
    
    def recommend_strategy(
        self, 
        context: Dict[str, Any],
        available_strategies: List[str],
        strategy_performance: Dict[str, float] = None
    ) -> List[Tuple[str, float]]:
        """Recommend strategies with confidence scores."""
        
        recommendations = []
        
        for strategy in available_strategies:
            if strategy not in self.strategy_contexts:
                continue
            
            confidence = self._calculate_strategy_confidence(strategy, context)
            
            # Adjust based on historical performance
            if strategy_performance and strategy in strategy_performance:
                performance_factor = strategy_performance[strategy]
                confidence = confidence * 0.7 + performance_factor * 0.3
            
            recommendations.append((strategy, confidence))
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations
    
    def _calculate_strategy_confidence(self, strategy: str, context: Dict[str, Any]) -> float:
        """Calculate confidence score for a strategy given the context."""
        
        strategy_info = self.strategy_contexts[strategy]
        confidence = 0.5  # Base confidence
        
        # Check favorable conditions
        best_when = strategy_info['best_when']
        for condition in best_when:
            if self._check_condition(condition, context):
                confidence += 0.15
        
        # Check unfavorable conditions
        avoid_when = strategy_info['avoid_when']
        for condition in avoid_when:
            if self._check_condition(condition, context):
                confidence -= 0.2
        
        # Check effectiveness factors
        effectiveness_factors = strategy_info['effectiveness_factors']
        for factor in effectiveness_factors:
            factor_score = self._evaluate_effectiveness_factor(factor, context)
            confidence += factor_score * 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _check_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Check if a specific condition is met in the current context."""
        
        turn_number = context.get('turn_number', 0)
        heat_level = context.get('heat_level', 0.0)
        topic = context.get('topic', '').lower()
        
        condition_checks = {
            'early_debate': turn_number <= 3,
            'mid_debate': 3 < turn_number <= 8,
            'late_debate': turn_number > 8,
            'low_heat': heat_level < 0.4,
            'high_heat': heat_level > 0.7,
            'factual_topic': any(word in topic for word in ['data', 'study', 'research', 'evidence']),
            'emotional_topic': any(word in topic for word in ['rights', 'freedom', 'justice', 'family']),
            'technical_topic': any(word in topic for word in ['technology', 'algorithm', 'system', 'method']),
            'academic_topic': any(word in topic for word in ['research', 'study', 'analysis', 'theory']),
            'complex_topic': len(topic.split()) > 5,
            'simple_topic': len(topic.split()) <= 3,
        }
        
        return condition_checks.get(condition, False)
    
    def _evaluate_effectiveness_factor(self, factor: str, context: Dict[str, Any]) -> float:
        """Evaluate how well an effectiveness factor is satisfied."""
        
        # This is a simplified implementation
        # In a full system, these would be more sophisticated evaluations
        
        factor_evaluations = {
            'evidence_quality': 0.7,  # Assume good evidence available
            'opponent_logic_weakness': 0.5,  # Neutral assumption
            'audience_values': 0.6,  # Moderate alignment
            'personal_stories': 0.4,  # Limited personal content
            'source_quality': 0.8,  # Assume high-quality sources
            'data_availability': 0.7,  # Good data availability
            'opponent_certainty': 0.5,  # Neutral
            'topic_complexity': min(1.0, len(context.get('topic', '').split()) / 10),
            'analogy_quality': 0.6,  # Moderate analogy potential
            'audience_familiarity': 0.5,  # Neutral assumption
        }
        
        return factor_evaluations.get(factor, 0.5)
