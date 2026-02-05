"""Mediator agent for managing debate flow and heat."""

from __future__ import annotations

import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from agents.heat import HeatDetector, HeatMetrics, EmotionLevel
from agents.llm_wrapper import VicunaWrapper, GenerationParams
from agents.result_types import MediatorInterventionResult


@dataclass
class MediatorPersona:
    """Defines the mediator's personality and approach."""
    name: str
    intervention_style: str  # "diplomatic", "firm", "gentle"
    cooling_effectiveness: float  # 0.0 to 1.0
    intervention_threshold: float  # Heat level that triggers intervention


# InterventionResult moved to result_types.py


class Mediator:
    """Mediator agent that monitors and manages debate heat."""
    
    def __init__(
        self,
        persona: MediatorPersona,
        llm_wrapper: Optional[VicunaWrapper] = None,
        heat_detector: Optional[HeatDetector] = None
    ):
        self.persona = persona
        self.llm = llm_wrapper
        self.heat_detector = heat_detector or HeatDetector()
        
        # Intervention history
        self.interventions: List[Dict[str, Any]] = []
        self.intervention_count = 0
        
        # Cooling strategies
        self.cooling_strategies = {
            "diplomatic": [
                "I can see both sides have strong feelings about this. Let's take a moment to focus on the core issues.",
                "Both perspectives bring valuable insights. Perhaps we can explore the common ground here.",
                "This is clearly an important topic for everyone. Let's ensure we're addressing the key points constructively."
            ],
            "firm": [
                "Let's pause here and refocus on respectful dialogue.",
                "I need to interrupt - let's keep this discussion productive and evidence-based.",
                "We're getting off track. Let's return to the substantive arguments."
            ],
            "gentle": [
                "I notice the conversation is getting quite intense. Would it help to take a different approach?",
                "Perhaps we could step back and consider this from another angle?",
                "I wonder if we might find more common ground by focusing on shared values?"
            ]
        }
    
    def should_intervene(
        self, 
        current_heat: HeatMetrics,
        opponent_heat: HeatMetrics,
        turn_number: int,
        conversation_history: List[Dict[str, str]]
    ) -> bool:
        """Determine if mediator should intervene."""
        
        # Basic heat threshold
        if current_heat.heat_score > self.persona.intervention_threshold:
            return True
        
        if opponent_heat.heat_score > self.persona.intervention_threshold:
            return True
        
        # Check for escalation
        if current_heat.escalation_trend > 0.3 or opponent_heat.escalation_trend > 0.3:
            return True
        
        # Check for personal attacks or extreme language
        if (current_heat.emotion_level == EmotionLevel.ANGRY or 
            opponent_heat.emotion_level == EmotionLevel.ANGRY):
            return True
        
        # Probabilistic intervention based on overall heat
        avg_heat = (current_heat.heat_score + opponent_heat.heat_score) / 2
        intervention_probability = max(0, (avg_heat - 0.5) * 0.6)  # 0 to 0.3 probability
        
        if random.random() < intervention_probability:
            return True
        
        # Don't intervene too frequently
        recent_interventions = sum(
            1 for intervention in self.interventions[-5:] 
            if intervention.get('turn_number', 0) > turn_number - 3
        )
        
        if recent_interventions >= 2:
            return False
        
        return False
    
    def intervene(
        self,
        current_heat: HeatMetrics,
        opponent_heat: HeatMetrics,
        turn_number: int,
        topic: str,
        conversation_history: List[Dict[str, str]]
    ) -> MediatorInterventionResult:
        """Generate a mediator intervention."""
        
        self.intervention_count += 1
        
        # Use the passed heat metrics
        max_heat = max(current_heat.heat_score, opponent_heat.heat_score)
        
        # Determine intervention approach
        if max_heat > 0.9:
            approach = "firm"
        elif max_heat > 0.6:
            approach = "diplomatic"
        else:
            approach = "gentle"
        
        # Generate intervention message
        if self.llm and self.llm.is_loaded():
            message = self._generate_llm_intervention(
                max_heat, turn_number, conversation_history, approach
            )
        else:
            message = self._generate_template_intervention(approach, max_heat)
        
        # Calculate cooling effect
        cooling_effect = self._calculate_cooling_effect(max_heat, approach)
        
        # Determine if debate should continue
        should_continue = self._should_continue_debate(max_heat, turn_number)
        
        # Suggest direction for conversation
        suggested_direction = self._suggest_direction(conversation_history)
        
        # Record intervention
        self.interventions.append({
            "turn_number": turn_number,
            "approach": approach,
            "max_heat": max_heat,
            "cooling_effect": cooling_effect,
            "content": message
        })
        
        return MediatorInterventionResult(
            content=message,
            cooling_effect=cooling_effect,
            should_continue=should_continue,
            suggested_direction=suggested_direction,
            intervention_type=approach
        )
    
    def _generate_llm_intervention(
        self,
        heat_score: float,
        turn_number: int,
        conversation_history: List[Dict[str, str]],
        approach: str
    ) -> str:
        """Generate intervention using LLM."""
        
        # Build context
        recent_messages = conversation_history[-4:] if conversation_history else []
        history_text = "\n".join([
            f"{msg['speaker']}: {msg['content']}" for msg in recent_messages
        ])
        
        # Determine tone
        tone_instructions = {
            "diplomatic": "Be diplomatic and balanced. Acknowledge both sides while guiding toward productive discussion.",
            "firm": "Be firm but respectful. Clearly redirect the conversation toward constructive dialogue.",
            "gentle": "Be gentle and understanding. Help participants find common ground and reduce tension."
        }
        
        # Extract topic from conversation history
        topic = "the current debate topic"
        if conversation_history:
            first_msg = conversation_history[0]
            if 'topic' in first_msg:
                topic = first_msg['topic']
        
        prompt = f"""You are {self.persona.name}, a skilled human debate mediator. The conversation about "{topic}" is getting emotionally intense and needs your intervention. You understand human emotions and can help balance the situation.

Recent conversation:
{history_text}

Current situation:
- Heat level is high (emotional intensity detected)
- Participants are expressing strong emotions (anger, frustration, fear, etc.)
- Need to acknowledge their feelings while guiding toward productive discussion
- Act like a human mediator who understands and empathizes with emotions

Your approach: {approach}
Instructions: {tone_instructions[approach]}

As a human mediator, you should:
1. Acknowledge the emotions being expressed
2. Validate their concerns while redirecting constructively
3. Show empathy and understanding
4. Guide them toward finding common ground
5. Use warm, human language

Respond as the mediator in 1-2 sentences. Be natural, empathetic, and helpful.

{self.persona.name}:"""
        
        params = GenerationParams(
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        try:
            response = self.llm.generate_response(prompt, params)
            return response.strip()
        except Exception:
            # Fallback to template
            return self._generate_template_intervention(approach, max_heat)
    
    def _generate_template_intervention(
        self, 
        approach: str, 
        max_heat: float
    ) -> str:
        """Generate intervention using templates."""
        
        strategies = self.cooling_strategies.get(approach, self.cooling_strategies["diplomatic"])
        base_message = random.choice(strategies)
        
        # Add specific guidance based on heat level
        if max_heat > 0.8:
            base_message += " Let's focus on understanding different perspectives rather than proving who's right."
        elif max_heat > 0.6:
            base_message += " All viewpoints deserve respectful consideration."
        else:
            base_message += " Let's keep our discussion constructive and evidence-based."
        
        return base_message
    
    def _calculate_cooling_effect(self, max_heat: float, approach: str) -> float:
        """Calculate how much the intervention should cool the conversation."""
        
        base_cooling = self.persona.cooling_effectiveness
        
        # Adjust based on approach
        approach_multipliers = {
            "firm": 1.2,
            "diplomatic": 1.0,
            "gentle": 0.8
        }
        
        cooling = base_cooling * approach_multipliers.get(approach, 1.0)
        
        # More effective on very high heat
        if max_heat > 0.8:
            cooling *= 1.3
        
        return min(1.0, cooling)
    
    def _should_continue_debate(
        self, 
        heat_score: float, 
        turn_number: int
    ) -> bool:
        """Determine if the debate should continue after intervention."""
        
        # Stop if heat is extremely high
        if heat_score > 0.95:
            return turn_number < 12  # Give some chances to cool down
        
        # Stop if too many recent interventions
        recent_interventions = sum(
            1 for intervention in self.interventions[-3:] 
            if intervention.get('turn_number', 0) > turn_number - 2
        )
        
        if recent_interventions >= 2:
            return False
        
        return True
    
    def _suggest_direction(
        self, 
        conversation_history: List[Dict[str, str]]
    ) -> Optional[str]:
        """Suggest a direction for the conversation."""
        
        # Extract topic from conversation history
        topic = "this topic"
        if conversation_history:
            first_msg = conversation_history[0]
            if 'topic' in first_msg:
                topic = first_msg['topic']
        
        suggestions = [
            f"Perhaps we could explore the practical implications of different approaches to {topic}?",
            f"What evidence would be most helpful in understanding {topic} better?",
            f"Are there any areas of agreement we can build on regarding {topic}?",
            f"What are the key trade-offs we should consider with {topic}?",
        ]
        
        return random.choice(suggestions)
    
    def get_intervention_stats(self) -> Dict[str, Any]:
        """Get statistics about mediator interventions."""
        if not self.interventions:
            return {"total_interventions": 0}
        
        approaches = [i["approach"] for i in self.interventions]
        avg_cooling = sum(i["cooling_effect"] for i in self.interventions) / len(self.interventions)
        
        return {
            "total_interventions": len(self.interventions),
            "approaches_used": {
                approach: approaches.count(approach) for approach in set(approaches)
            },
            "average_cooling_effect": avg_cooling,
            "intervention_frequency": len(self.interventions) / max(1, len(set(i["turn_number"] for i in self.interventions)))
        }
    
    def reset_for_new_debate(self) -> None:
        """Reset mediator state for a new debate."""
        self.interventions.clear()
        self.intervention_count = 0
        self.heat_detector.reset_history()
