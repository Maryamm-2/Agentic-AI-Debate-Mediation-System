"""Response realization and formatting utilities."""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from agents.heat import EmotionLevel


@dataclass
class RealizationParams:
    """Parameters for response realization."""
    max_sentences: int = 4
    min_sentences: int = 2
    max_words: int = 150
    include_citations: bool = True
    emotion_markers: bool = True
    clarity_level: float = 0.8  # 0.0 to 1.0


class ResponseRealizer:
    """Formats and refines debate responses for clarity and impact."""
    
    def __init__(self, params: RealizationParams = None):
        self.params = params or RealizationParams()
        
        # Emotion-based formatting
        self.emotion_formats = {
            EmotionLevel.CALM: {
                'prefix': '',
                'connector': 'Furthermore,',
                'emphasis': 'clearly',
                'conclusion': 'Therefore,'
            },
            EmotionLevel.ENGAGED: {
                'prefix': '',
                'connector': 'Moreover,',
                'emphasis': 'importantly',
                'conclusion': 'Thus,'
            },
            EmotionLevel.HEATED: {
                'prefix': '',
                'connector': 'However,',
                'emphasis': 'crucially',
                'conclusion': 'Clearly,'
            },
            EmotionLevel.ANGRY: {
                'prefix': '',
                'connector': 'But',
                'emphasis': 'obviously',
                'conclusion': 'Obviously,'
            }
        }
    
    def realize_response(
        self,
        raw_response: str,
        strategy: str,
        emotion_level: EmotionLevel,
        retrieved_passages: List[str] = None,
        context: Dict[str, Any] = None
    ) -> str:
        """Transform raw response into polished debate argument."""
        
        # Clean and structure the response
        cleaned = self._clean_response(raw_response)
        
        # Apply strategy-specific formatting
        formatted = self._apply_strategy_formatting(cleaned, strategy, emotion_level)
        
        # Add citations if requested and available
        if self.params.include_citations and retrieved_passages:
            formatted = self._add_citations(formatted, retrieved_passages)
        
        # Apply emotion-based formatting
        if self.params.emotion_markers:
            formatted = self._apply_emotion_formatting(formatted, emotion_level)
        
        # Ensure appropriate length
        formatted = self._adjust_length(formatted)
        
        # Final polish for clarity
        formatted = self._polish_clarity(formatted)
        
        return formatted.strip()
    
    def _clean_response(self, response: str) -> str:
        """Clean up raw response text."""
        # Remove extra whitespace
        response = re.sub(r'\s+', ' ', response)
        
        # Remove incomplete sentences at the end
        sentences = re.split(r'[.!?]+', response)
        complete_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Minimum sentence length
                complete_sentences.append(sentence)
        
        # Rejoin sentences
        if complete_sentences:
            response = '. '.join(complete_sentences)
            if not response.endswith(('.', '!', '?')):
                response += '.'
        
        return response
    
    def _apply_strategy_formatting(
        self, 
        response: str, 
        strategy: str, 
        emotion_level: EmotionLevel
    ) -> str:
        """Apply strategy-specific formatting."""
        
        strategy_formats = {
            'logical_reasoning': self._format_logical,
            'emotional_appeal': self._format_emotional,
            'citation_heavy': self._format_citation_heavy,
            'questioning': self._format_questioning,
            'analogy_based': self._format_analogy_based
        }
        
        formatter = strategy_formats.get(strategy, self._format_default)
        return formatter(response, emotion_level)
    
    def _format_logical(self, response: str, emotion_level: EmotionLevel) -> str:
        """Format response for logical reasoning strategy."""
        # Add logical connectors
        sentences = response.split('. ')
        if len(sentences) > 1:
            # Add logical flow
            connectors = ['First,', 'Second,', 'Additionally,', 'Finally,']
            for i, sentence in enumerate(sentences[:len(connectors)]):
                if i < len(connectors) and not sentence.strip().startswith(('First', 'Second', 'Third', 'Finally')):
                    sentences[i] = f"{connectors[i]} {sentence.strip()}"
        
        return '. '.join(sentences)
    
    def _format_emotional(self, response: str, emotion_level: EmotionLevel) -> str:
        """Format response for emotional appeal strategy."""
        # Add emotional emphasis
        if emotion_level in [EmotionLevel.HEATED, EmotionLevel.ANGRY]:
            # Stronger emotional language
            response = re.sub(r'\bimportant\b', 'crucial', response, flags=re.IGNORECASE)
            response = re.sub(r'\bbad\b', 'harmful', response, flags=re.IGNORECASE)
            response = re.sub(r'\bgood\b', 'beneficial', response, flags=re.IGNORECASE)
        
        return response
    
    def _format_citation_heavy(self, response: str, emotion_level: EmotionLevel) -> str:
        """Format response for citation-heavy strategy."""
        # Add evidential language
        response = re.sub(
            r'\b(shows?|indicates?|suggests?)\b', 
            r'the evidence \1', 
            response, 
            flags=re.IGNORECASE
        )
        
        # Add qualifying language for precision
        response = re.sub(
            r'\b(all|every|never|always)\b', 
            r'most', 
            response, 
            flags=re.IGNORECASE
        )
        
        return response
    
    def _format_questioning(self, response: str, emotion_level: EmotionLevel) -> str:
        """Format response for questioning strategy."""
        # Convert some statements to questions
        sentences = response.split('. ')
        modified_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence.endswith('?'):
                # Convert some statements to rhetorical questions
                if any(word in sentence.lower() for word in ['consider', 'think', 'believe', 'assume']):
                    sentence = f"But should we really {sentence.lower()}?"
                elif len(modified_sentences) < 2 and random.choice([True, False]):
                    sentence = f"Isn't it true that {sentence.lower()}?"
            
            modified_sentences.append(sentence)
        
        return '. '.join(modified_sentences)
    
    def _format_analogy_based(self, response: str, emotion_level: EmotionLevel) -> str:
        """Format response for analogy-based strategy."""
        # Add analogical language
        analogy_starters = [
            "It's like", "Consider this:", "Think of it as", "Imagine if"
        ]
        
        sentences = response.split('. ')
        if len(sentences) > 1 and not any(starter in response for starter in analogy_starters):
            # Try to add an analogy connector
            middle_idx = len(sentences) // 2
            sentences[middle_idx] = f"Think of it this way: {sentences[middle_idx].strip()}"
        
        return '. '.join(sentences)
    
    def _format_default(self, response: str, emotion_level: EmotionLevel) -> str:
        """Default formatting."""
        return response
    
    def _add_citations(self, response: str, passages: List[str]) -> str:
        """Add citations to the response."""
        if not passages:
            return response
        
        # Simple citation approach - add reference to evidence
        if "evidence" not in response.lower() and "research" not in response.lower():
            # Add a reference to supporting evidence
            sentences = response.split('. ')
            if len(sentences) > 1:
                # Add citation reference to the first substantial claim
                sentences[0] += " (as supported by the research)"
        
        return '. '.join(sentences) if '. ' in response else response
    
    def _apply_emotion_formatting(self, response: str, emotion_level: EmotionLevel) -> str:
        """Apply emotion-based formatting."""
        if emotion_level not in self.emotion_formats:
            return response
        
        format_dict = self.emotion_formats[emotion_level]
        
        # Add emphasis words
        emphasis = format_dict['emphasis']
        if emphasis not in response.lower():
            sentences = response.split('. ')
            if sentences:
                # Add emphasis to the first sentence
                sentences[0] = f"{sentences[0].strip()}, {emphasis}"
        
        return '. '.join(sentences) if '. ' in response else response
    
    def _adjust_length(self, response: str) -> str:
        """Adjust response length to meet parameters."""
        sentences = [s.strip() for s in response.split('. ') if s.strip()]
        words = response.split()
        
        # Adjust sentence count
        if len(sentences) > self.params.max_sentences:
            sentences = sentences[:self.params.max_sentences]
        elif len(sentences) < self.params.min_sentences and len(sentences) > 0:
            # If too short, we keep what we have rather than padding
            pass
        
        response = '. '.join(sentences)
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        # Adjust word count
        words = response.split()
        if len(words) > self.params.max_words:
            # Truncate at sentence boundary
            truncated_words = words[:self.params.max_words]
            truncated_text = ' '.join(truncated_words)
            
            # Find last complete sentence
            last_period = truncated_text.rfind('.')
            if last_period > len(truncated_text) * 0.7:  # If we're not cutting too much
                response = truncated_text[:last_period + 1]
            else:
                response = truncated_text + '...'
        
        return response
    
    def _polish_clarity(self, response: str) -> str:
        """Final polish for clarity and readability."""
        # Remove redundant phrases
        redundant_patterns = [
            (r'\b(very very|really really|quite quite)\b', r'\1'.split()[0]),
            (r'\b(in my opinion, I think|I believe that I think)\b', 'I believe'),
            (r'\b(the fact that the fact)\b', 'the fact'),
        ]
        
        for pattern, replacement in redundant_patterns:
            response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
        
        # Improve sentence flow
        response = re.sub(r'\. And ', '. Additionally, ', response)
        response = re.sub(r'\. But ', '. However, ', response)
        response = re.sub(r'\. So ', '. Therefore, ', response)
        
        return response


# Import random for the questioning strategy
import random
