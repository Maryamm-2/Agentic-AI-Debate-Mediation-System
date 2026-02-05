"""Utility functions for the debate system."""

from __future__ import annotations

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class DebateConfig:
    """Configuration for a debate session."""
    # Model settings (all with defaults to avoid dataclass ordering error)
    model_name: str = "microsoft/DialoGPT-medium"
    device: str = "auto"
    load_in_8bit: bool = False
    max_new_tokens: int = 100
    temperature: float = 0.7
    # Backend selection: 'transformers' or 'llama.cpp'
    backend: str = "transformers"
    # GGUF-specific params (used when backend == 'llama.cpp')
    gguf_model_path: str = ""
    gguf_n_ctx: int = 4096
    gguf_n_threads: int = 0
    gguf_n_gpu_layers: int = 0
    
    # RAG settings
    use_embeddings: bool = False
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    bm25_weight: float = 0.7
    embedding_weight: float = 0.3
    max_retrieved_chunks: int = 3
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Debate settings
    max_turns: int = 6
    heat_threshold: float = 0.8
    mediator_intervention_probability: float = 0.3
    rolling_history_size: int = 10
    
    # Agent settings
    debater_pro: Dict[str, Any] = None
    debater_anti: Dict[str, Any] = None
    mediator: Dict[str, Any] = None
    
    # Bandit settings
    bandit_epsilon: float = 0.1
    bandit_strategies: List[str] = None
    bandit_learning_rate: float = 0.05
    
    # Polish settings
    polish_enabled: bool = False
    polish_model: str = "google/flan-t5-small"
    polish_max_tokens: int = 40
    
    def __post_init__(self):
        """Initialize mutable defaults after object creation"""
        if self.debater_pro is None:
            self.debater_pro = {}
        if self.debater_anti is None:
            self.debater_anti = {}
        if self.mediator is None:
            self.mediator = {}
        if self.bandit_strategies is None:
            self.bandit_strategies = ["logical_reasoning", "emotional_appeal", "citation_heavy", "questioning", "analogy_based"]


def load_config(config_path: Path) -> DebateConfig:
    """Load configuration from YAML file with error handling."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if not config_dict:
            raise ValueError("Configuration file is empty or invalid")
        
        # Validate required sections exist
        required_sections = ['model', 'rag', 'debate', 'agents', 'bandit', 'polish']
        for section in required_sections:
            if section not in config_dict:
                raise ValueError(f"Missing required configuration section: {section}")
        
        return DebateConfig(
            # Model settings
            model_name=config_dict['model']['name'],
            device=config_dict['model']['device'],
            load_in_8bit=config_dict['model']['load_in_8bit'],
            max_new_tokens=config_dict['model']['max_new_tokens'],
            temperature=config_dict['model']['temperature'],
            backend=config_dict['model'].get('backend', 'transformers'),
            gguf_model_path=config_dict['model'].get('gguf_model_path', ''),
            gguf_n_ctx=config_dict['model'].get('gguf_n_ctx', 4096),
            gguf_n_threads=config_dict['model'].get('gguf_n_threads', 0),
            gguf_n_gpu_layers=config_dict['model'].get('gguf_n_gpu_layers', 0),
            
            # RAG settings
            use_embeddings=config_dict['rag']['use_embeddings'],
            embedding_model=config_dict['rag']['embedding_model'],
            bm25_weight=config_dict['rag']['bm25_weight'],
            embedding_weight=config_dict['rag']['embedding_weight'],
            max_retrieved_chunks=config_dict['rag']['max_retrieved_chunks'],
            chunk_size=config_dict['rag']['chunk_size'],
            chunk_overlap=config_dict['rag']['chunk_overlap'],
            
            # Debate settings
            max_turns=config_dict['debate']['max_turns'],
            heat_threshold=config_dict['debate']['heat_threshold'],
            mediator_intervention_probability=config_dict['debate']['mediator_intervention_probability'],
            rolling_history_size=config_dict['debate']['rolling_history_size'],
            
            # Agent settings
            debater_pro=config_dict['agents']['debater_pro'],
            debater_anti=config_dict['agents']['debater_anti'],
            mediator=config_dict['agents']['mediator'],
            
            # Bandit settings
            bandit_epsilon=config_dict['bandit']['epsilon'],
            bandit_strategies=config_dict['bandit']['strategies'],
            bandit_learning_rate=config_dict['bandit']['learning_rate'],
            
            # Polish settings
            polish_enabled=config_dict['polish']['enabled'],
            polish_model=config_dict['polish']['model'],
            polish_max_tokens=config_dict['polish']['max_tokens'],
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")
    except KeyError as e:
        raise ValueError(f"Missing required configuration key: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def validate_config(config: DebateConfig) -> Dict[str, Any]:
    """Validate configuration object and return validation results."""
    errors = []
    warnings = []
    
    # Validate model settings
    if not config.model_name or not isinstance(config.model_name, str):
        errors.append("Model name must be a non-empty string")
    
    if config.device not in ['auto', 'cpu', 'cuda']:
        errors.append("Device must be 'auto', 'cpu', or 'cuda'")
    
    if not isinstance(config.load_in_8bit, bool):
        errors.append("load_in_8bit must be a boolean")
    
    if not (1 <= config.max_new_tokens <= 1000):
        errors.append("max_new_tokens must be between 1 and 1000")
    
    if not (0.0 <= config.temperature <= 2.0):
        errors.append("temperature must be between 0.0 and 2.0")
    
    # Backend validation
    if config.backend not in ['transformers', 'llama.cpp']:
        errors.append("backend must be 'transformers' or 'llama.cpp'")

    if config.backend == 'llama.cpp':
        if not config.gguf_model_path:
            errors.append("gguf_model_path is required when backend == 'llama.cpp'")
        if not (512 <= config.gguf_n_ctx <= 16384):
            errors.append("gguf_n_ctx must be between 512 and 16384")
        if not (0 <= config.gguf_n_gpu_layers <= 80):
            errors.append("gguf_n_gpu_layers must be between 0 and 80")

    # Validate RAG settings
    if not isinstance(config.use_embeddings, bool):
        errors.append("use_embeddings must be a boolean")
    
    if not (0.0 <= config.bm25_weight <= 1.0):
        errors.append("bm25_weight must be between 0.0 and 1.0")
    
    if not (0.0 <= config.embedding_weight <= 1.0):
        errors.append("embedding_weight must be between 0.0 and 1.0")
    
    if abs(config.bm25_weight + config.embedding_weight - 1.0) > 0.01:
        warnings.append("bm25_weight and embedding_weight should sum to 1.0")
    
    if not (1 <= config.max_retrieved_chunks <= 20):
        errors.append("max_retrieved_chunks must be between 1 and 20")
    
    if not (100 <= config.chunk_size <= 2000):
        errors.append("chunk_size must be between 100 and 2000")
    
    if not (0 <= config.chunk_overlap <= config.chunk_size // 2):
        errors.append("chunk_overlap must be between 0 and chunk_size/2")
    
    # Validate debate settings
    if not (1 <= config.max_turns <= 50):
        errors.append("max_turns must be between 1 and 50")
    
    if not (0.0 <= config.heat_threshold <= 1.0):
        errors.append("heat_threshold must be between 0.0 and 1.0")
    
    if not (0.0 <= config.mediator_intervention_probability <= 1.0):
        errors.append("mediator_intervention_probability must be between 0.0 and 1.0")
    
    if not (5 <= config.rolling_history_size <= 50):
        errors.append("rolling_history_size must be between 5 and 50")
    
    # Validate agent settings
    required_agent_keys = ['name', 'personality', 'aggression_level']
    for agent_type in ['debater_pro', 'debater_anti']:
        agent = getattr(config, agent_type)
        for key in required_agent_keys:
            if key not in agent:
                errors.append(f"Missing {key} in {agent_type}")
        
        if 'aggression_level' in agent:
            if not (0.0 <= agent['aggression_level'] <= 1.0):
                errors.append(f"{agent_type} aggression_level must be between 0.0 and 1.0")
    
    # Validate bandit settings
    if not (0.0 <= config.bandit_epsilon <= 1.0):
        errors.append("bandit_epsilon must be between 0.0 and 1.0")
    
    if not config.bandit_strategies or not isinstance(config.bandit_strategies, list):
        errors.append("bandit_strategies must be a non-empty list")
    
    valid_strategies = ['logical_reasoning', 'emotional_appeal', 'citation_heavy', 'questioning', 'analogy_based']
    for strategy in config.bandit_strategies:
        if strategy not in valid_strategies:
            errors.append(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}")
    
    if not (0.0 <= config.bandit_learning_rate <= 1.0):
        errors.append("bandit_learning_rate must be between 0.0 and 1.0")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def create_default_config(config_path: Path) -> None:
    """Create a default configuration file."""
    default_config = {
        "model": {
            "name": "HuggingFaceH4/zephyr-7b-alpha",
            "device": "auto",
            "load_in_8bit": False,
            "max_new_tokens": 300,
            "temperature": 0.8,
            "do_sample": True,
            "top_p": 0.9,
            "backend": "transformers",  # or 'llama.cpp'
            # GGUF backend parameters (used when backend == 'llama.cpp')
            "gguf_model_path": "",
            "gguf_n_ctx": 4096,
            "gguf_n_threads": 0,
            "gguf_n_gpu_layers": 0
        },
        "rag": {
            "use_embeddings": True,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "bm25_weight": 0.7,
            "embedding_weight": 0.3,
            "max_retrieved_chunks": 5,
            "chunk_size": 512,
            "chunk_overlap": 50
        },
        "debate": {
            "max_turns": 16,
            "heat_threshold": 0.8,
            "mediator_intervention_probability": 0.3,
            "rolling_history_size": 10
        },
        "agents": {
            "debater_pro": {
                "name": "Alex",
                "role": "pro",
                "personality": "analytical and evidence-focused",
                "aggression_level": 0.6
            },
            "debater_anti": {
                "name": "Blair",
                "role": "anti",
                "personality": "skeptical and detail-oriented",
                "aggression_level": 0.7
            },
            "mediator": {
                "name": "Morgan",
                "intervention_style": "diplomatic",
                "cooling_effectiveness": 0.8
            }
        },
        "bandit": {
            "epsilon": 0.1,
            "strategies": [
                "logical_reasoning",
                "emotional_appeal",
                "citation_heavy",
                "questioning",
                "analogy_based"
            ],
            "learning_rate": 0.05
        },
        "polish": {
            "enabled": False,
            "model": "google/flan-t5-small",
            "max_tokens": 40
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)


def format_debate_output(
    speaker: str,
    message: str,
    strategy: str,
    heat_score: float,
    turn_number: int
) -> str:
    """Format debate output for display."""
    # Use ASCII-compatible heat indicator for Windows
    heat_level = min(3, int(heat_score * 3))
    heat_indicator = "*" * heat_level if heat_level > 0 else ""
    strategy_short = strategy.replace("_", " ").title()
    
    return f"""
Turn {turn_number} | {speaker} | {strategy_short} {heat_indicator}
{message}
Heat: {heat_score:.2f}
"""


def format_mediator_output(message: str, turn_number: int) -> str:
    """Format mediator intervention output."""
    return f"""
MEDIATOR INTERVENTION (Turn {turn_number}) [STOP]
{message}
Cooling the discussion...
"""


def calculate_debate_outcome(
    pro_stats: Dict[str, Any],
    anti_stats: Dict[str, Any],
    mediator_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate overall debate outcome and statistics."""
    
    pro_heat = pro_stats.get("avg_heat_generated", 0)
    anti_heat = anti_stats.get("avg_heat_generated", 0)
    interventions = mediator_stats.get("total_interventions", 0)
    
    # Simple scoring system
    pro_score = 0
    anti_score = 0
    
    # Reward lower heat generation (more controlled arguments)
    if pro_heat < anti_heat:
        pro_score += 1
    elif anti_heat < pro_heat:
        anti_score += 1
    
    # Reward fewer interventions caused
    pro_interventions = pro_stats.get("interventions_caused", 0)
    anti_interventions = anti_stats.get("interventions_caused", 0)
    
    if pro_interventions < anti_interventions:
        pro_score += 1
    elif anti_interventions < pro_interventions:
        anti_score += 1
    
    # Determine winner
    if pro_score > anti_score:
        winner = "Pro"
    elif anti_score > pro_score:
        winner = "Anti"
    else:
        winner = "Draw"
    
    return {
        "winner": winner,
        "pro_score": pro_score,
        "anti_score": anti_score,
        "total_interventions": interventions,
        "avg_heat": (pro_heat + anti_heat) / 2,
        "debate_quality": max(0, 1 - (interventions * 0.1) - ((pro_heat + anti_heat) / 2) * 0.5)
    }


def save_debate_log(
    debate_log: List[Dict[str, Any]],
    outcome: Dict[str, Any],
    topic: str,
    output_path: Path
) -> None:
    """Save complete debate log to file."""
    
    log_data = {
        "topic": topic,
        "outcome": outcome,
        "turns": debate_log,
        "statistics": {
            "total_turns": len([turn for turn in debate_log if turn.get("type") == "argument"]),
            "total_interventions": len([turn for turn in debate_log if turn.get("type") == "intervention"]),
            "final_heat": debate_log[-1].get("heat_score", 0) if debate_log else 0
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(log_data, f, default_flow_style=False, indent=2)


class SafetyFilter:
    """Basic safety filter for debate content."""
    
    def __init__(self):
        self.blocked_terms = [
            # Add terms that should be blocked
            # This is a basic implementation - in production, use more sophisticated filtering
        ]
        
        self.warning_terms = [
            "hate", "violence", "attack", "destroy", "kill"
        ]
    
    def check_content(self, content: str) -> Dict[str, Any]:
        """Check content for safety issues."""
        content_lower = content.lower()
        
        blocked = any(term in content_lower for term in self.blocked_terms)
        warnings = [term for term in self.warning_terms if term in content_lower]
        
        return {
            "blocked": blocked,
            "warnings": warnings,
            "safe": not blocked and len(warnings) == 0
        }
    
    def filter_content(self, content: str) -> str:
        """Filter and clean content if needed."""
        # Basic implementation - could be enhanced
        safety_check = self.check_content(content)
        
        if safety_check["blocked"]:
            return "[Content filtered for safety]"
        
        return content

