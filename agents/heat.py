"""Heat/emotion detection and tracking system."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class EmotionLevel(Enum):
    """Emotion intensity levels."""
    CALM = "calm"
    ENGAGED = "engaged"
    HEATED = "heated"
    ANGRY = "angry"
    ENTHUSIASTIC = "enthusiastic"
    FEARFUL = "fearful"
    SAD = "sad"
    FRUSTRATED = "frustrated"


@dataclass
class HeatMetrics:
    """Metrics for measuring debate heat."""
    emotion_level: EmotionLevel
    heat_score: float
    sentiment_score: float
    aggression_indicators: List[str]
    escalation_trend: float
    primary_emotion: str  # anger, enthusiasm, fear, sadness, etc.
    emotional_intensity: float  # 0.0 to 1.0


class HeatDetector:
    """Detects and tracks emotional heat in debate conversations."""
    
    def __init__(self):
        # Aggression indicators with weights (enhanced for more sensitivity)
        self.aggression_patterns = {
            # Strong disagreement
            r'\b(absolutely wrong|completely false|totally incorrect|dead wrong)\b': 0.9,
            r'\b(ridiculous|absurd|nonsense|garbage|trash)\b': 1.0,
            r'\b(stupid|idiotic|moronic|dumb)\b': 1.0,
            r'\b(unacceptable|outrageous|disgusting|appalling)\b': 0.9,
            
            # Dismissive language
            r'\b(obviously|clearly|any fool|everyone knows)\b': 0.7,
            r'\b(wake up|get real|face reality)\b': 0.8,
            r'\b(seriously|come on|give me a break)\b': 0.6,
            
            # Personal attacks
            r'\b(you people|your kind|typical)\b': 0.9,
            r'\b(liar|lying|dishonest|deceptive)\b': 1.0,
            r'\b(ignorant|clueless|naive)\b': 0.8,
            
            # Emotional intensifiers
            r'[!]{2,}': 0.7,
            r'[A-Z]{3,}': 0.8,  # ALL CAPS
            r'\b(never|always|everyone|no one) ': 0.5,
            r'\b(how dare|how can you)\b': 0.9,
            
            # Confrontational language
            r'\b(you\'re wrong|that\'s wrong|you don\'t understand)\b': 0.7,
            r'\b(that\'s ridiculous|that\'s absurd|that\'s nonsense)\b': 0.9,
            r'\b(I can\'t believe|unbelievable|incredible)\b': 0.6,
            # More generic strong disagreement / dismissive words
            r"\b(weak|pathetic|embarrassing|fail(ed)?|you fail|demand|shame)\b": 0.6,
            r"\b(shut up|you should be quiet|silence)\b": 0.9,
        }
        
        # Emotional indicators for different emotions
        self.emotion_patterns = {
            # Anger indicators
            'anger': {
                r'\b(furious|enraged|outraged|livid|seething)\b': 0.9,
                r'\b(angry|mad|upset|irritated|annoyed)\b': 0.7,
                r'\b(disgusting|appalling|outrageous|unacceptable)\b': 0.8,
                r'\b(how dare|how can you|this is wrong)\b': 0.8,
            },
            # Enthusiasm indicators
            'enthusiasm': {
                r'\b(excited|thrilled|amazing|fantastic|brilliant)\b': 0.8,
                r'\b(wonderful|incredible|outstanding|excellent)\b': 0.7,
                r'\b(I love|I adore|this is great|perfect)\b': 0.8,
                r'\b(enthusiastic|passionate|eager|optimistic)\b': 0.7,
            },
            # Fear indicators
            'fear': {
                r'\b(scared|afraid|terrified|worried|concerned)\b': 0.8,
                r'\b(dangerous|risky|threatening|alarming)\b': 0.7,
                r'\b(what if|I fear|concerned about|worried that)\b': 0.8,
                r'\b(disaster|catastrophe|crisis|emergency)\b': 0.9,
            },
            # Sadness indicators
            'sadness': {
                r'\b(sad|depressed|disappointed|heartbroken|devastated)\b': 0.8,
                r'\b(tragic|unfortunate|regrettable|sorrowful)\b': 0.7,
                r'\b(I wish|if only|it breaks my heart|so sad)\b': 0.8,
                r'\b(loss|grief|mourning|melancholy)\b': 0.7,
            },
            # Frustration indicators
            'frustration': {
                r'\b(frustrated|exasperated|fed up|sick of)\b': 0.8,
                r'\b(why can\'t|this is ridiculous|enough already)\b': 0.7,
                r'\b(impossible|hopeless|pointless|useless)\b': 0.8,
                r'\b(argh|ugh|seriously|come on)\b': 0.6,
            }
        }
        
        # Positive engagement patterns (reduce heat)
        self.cooling_patterns = {
            r'\b(I understand|I see your point|good point|fair enough)\b': -0.3,
            r'\b(perhaps|maybe|could be|might be)\b': -0.2,
            r'\b(let me clarify|to be fair|on the other hand)\b': -0.2,
        }
        
        # Track conversation history for trend analysis
        self.heat_history: List[float] = []
        self.max_history = 10
    
    def analyze_heat(self, message: str, speaker: str) -> HeatMetrics:
        """Analyze the emotional heat of a message with robust handling."""
        # Handle empty or invalid messages
        if not message or not isinstance(message, str) or not message.strip():
            return HeatMetrics(
                emotion_level=EmotionLevel.CALM,
                heat_score=0.0,
                sentiment_score=0.0,
                aggression_indicators=[],
                escalation_trend=0.0,
                primary_emotion="neutral",
                emotional_intensity=0.0
            )
        message_stripped = message.strip()

        # 1) Count matches for aggression and cooling patterns (robust to multiple matches)
        aggression_raw = 0.0
        found_indicators = []

        for pattern, weight in self.aggression_patterns.items():
            try:
                flags = 0 if pattern == r'[A-Z]{3,}' else re.IGNORECASE
                matches = list(re.finditer(pattern, message_stripped, flags=flags))
                if matches:
                    aggression_raw += weight * len(matches)
                    found_indicators.extend([m.group(0) for m in matches])
            except re.error as e:
                # don't fail the detector on bad regex
                continue

        cooling_raw = 0.0
        for pattern, weight in self.cooling_patterns.items():
            try:
                matches = list(re.finditer(pattern, message_stripped, flags=re.IGNORECASE))
                if matches:
                    cooling_raw += weight * len(matches)
                    found_indicators.extend([m.group(0) for m in matches])
            except re.error:
                continue

        # 2) Additional intensity signals: exclamation marks and ALL-CAPS word ratio
        exclamations = message_stripped.count('!')
        words = re.findall(r"\b\w+\b", message_stripped)
        if words:
            caps_words = sum(1 for w in words if w.isalpha() and w.upper() == w and len(w) >= 3)
            caps_ratio = caps_words / len(words)
        else:
            caps_ratio = 0.0

        # Map exclamation count into 0..1 (saturate at 6)
        exclaim_score = min(6, exclamations) / 6.0

        # 3) Base sentiment and emotion intensity
        sentiment_score = self._calculate_sentiment(message_stripped)
        primary_emotion, emotional_intensity = self._detect_primary_emotion(message_stripped)

        # 4) Combine raw signals into a raw heat value
        # aggression_raw can be >1; scale it down with a soft normalization
        agg_scaled = aggression_raw / (1.0 + aggression_raw)  # compress to 0..1
        cool_scaled = max(-1.0, min(0.0, cooling_raw))  # cooling_raw is negative

        raw_heat = agg_scaled + 0.5 * exclaim_score + 0.6 * caps_ratio + 0.4 * abs(sentiment_score) + 0.3 * emotional_intensity + cool_scaled

        # 5) Normalize using tanh for good dynamic range, then rescale to 0..1
        import math
        heat_score = (math.tanh((raw_heat - 0.2) * 1.5) + 1.0) / 2.0
        heat_score = max(0.0, min(1.0, heat_score))

        # 6) Final outputs
        emotion_level = self._classify_emotion(heat_score)
        escalation_trend = self._calculate_escalation_trend(heat_score)

        return HeatMetrics(
            emotion_level=emotion_level,
            heat_score=heat_score,
            sentiment_score=sentiment_score,
            aggression_indicators=found_indicators,
            escalation_trend=escalation_trend,
            primary_emotion=primary_emotion,
            emotional_intensity=emotional_intensity
        )
    
    def _calculate_sentiment(self, message: str) -> float:
        """Simple sentiment analysis (-1 to 1)."""
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'enjoy', 'appreciate', 'agree', 'support',
            'helpful', 'useful', 'valuable', 'important', 'beneficial'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
            'disagree', 'oppose', 'against', 'wrong', 'false', 'incorrect',
            'harmful', 'dangerous', 'useless', 'worthless', 'problematic'
        ]
        
        message_lower = message.lower()
        words = re.findall(r'\b\w+\b', message_lower)
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment * 5))  # Scale up the effect
    
    def _classify_emotion(self, heat_score: float) -> EmotionLevel:
        """Classify emotion level based on heat score."""
        if heat_score >= 0.8:
            return EmotionLevel.ANGRY
        elif heat_score >= 0.6:
            return EmotionLevel.HEATED
        elif heat_score >= 0.3:
            return EmotionLevel.ENGAGED
        else:
            return EmotionLevel.CALM
    
    def _calculate_escalation_trend(self, current_heat: float) -> float:
        """Calculate if the conversation is escalating or de-escalating."""
        self.heat_history.append(current_heat)
        
        # Keep only recent history
        if len(self.heat_history) > self.max_history:
            self.heat_history = self.heat_history[-self.max_history:]
        
        if len(self.heat_history) < 3:
            return 0.0
        
        # Calculate trend over last few messages
        recent_scores = self.heat_history[-3:]
        trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        
        return trend
    
    def should_intervene(self, heat_metrics: HeatMetrics, threshold: float = 0.8) -> bool:
        """Determine if mediator should intervene."""
        return (
            heat_metrics.heat_score > threshold or
            heat_metrics.emotion_level == EmotionLevel.ANGRY or
            heat_metrics.escalation_trend > 0.3
        )
    
    def get_cooling_suggestion(self, heat_metrics: HeatMetrics) -> str:
        """Get a suggestion for cooling down the conversation."""
        if heat_metrics.emotion_level == EmotionLevel.ANGRY:
            return "Let's take a step back and focus on the core issues rather than personal attacks."
        elif heat_metrics.emotion_level == EmotionLevel.HEATED:
            return "I can see both sides feel strongly about this. Let's examine the evidence more carefully."
        elif heat_metrics.escalation_trend > 0.2:
            return "The discussion seems to be getting more intense. Let's refocus on the facts."
        else:
            return "Let's continue with a respectful exchange of ideas."
    
    def _detect_primary_emotion(self, message_lower: str) -> Tuple[str, float]:
        """Detect the primary emotion and its intensity."""
        emotion_scores = {}
        for emotion, patterns in self.emotion_patterns.items():
            score = 0.0
            for pattern, weight in patterns.items():
                try:
                    matches = re.findall(pattern, message_lower, flags=re.IGNORECASE)
                    score += weight * len(matches)
                except re.error:
                    continue
            emotion_scores[emotion] = score
        
        # Find the emotion with the highest score
        if emotion_scores:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            max_score = emotion_scores[primary_emotion]
            # Normalize intensity to 0-1 range
            emotional_intensity = min(1.0, max_score / 2.0)
        else:
            primary_emotion = "neutral"
            emotional_intensity = 0.0
        
        return primary_emotion, emotional_intensity
    
    def reset_history(self) -> None:
        """Reset the heat history (e.g., for a new debate)."""
        self.heat_history.clear()



