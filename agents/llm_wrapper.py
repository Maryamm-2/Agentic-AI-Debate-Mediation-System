"""LLM wrappers for debate generation.

Includes:
- Transformers-based wrapper (Vicuna/Zephyr etc.) with bitsandbytes 4-bit/8-bit
- Optional llama.cpp-based wrapper for running GGUF-quantized models
"""

from __future__ import annotations

import torch
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import os

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)

try:
    # Optional dependency for GGUF via llama.cpp
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    Llama = None  # type: ignore


@dataclass
class GenerationParams:
    """Parameters for text generation."""
    max_new_tokens: int = 300
    temperature: float = 0.8
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1


class VicunaWrapper:
    """Wrapper for chat LLM inference with quantization support.

    Notes:
        - Primary model is taken from configuration (e.g., Vicuna or Zephyr).
        - If the primary model fails to load, falls back to microsoft/DialoGPT-medium.
        - Supports 8-bit (int8) and 4-bit (int4) quantization via bitsandbytes.
    """
    
    def __init__(
        self, 
        model_name: str = "lmsys/vicuna-7b-v1.5",
        device: str = "auto",
        load_in_8bit: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.generation_config: Optional[GenerationConfig] = None
        self.using_fallback: bool = False
        self.actual_model_name: str = model_name
        
    def load_model(self) -> None:
        """Load the Vicuna model and tokenizer with comprehensive error handling."""
        from rich.console import Console
        console = Console()
        
        console.print(f"[blue]🤖 Loading model: {self.model_name}[/blue]")
        
        # Track if we're using fallback
        using_fallback = False
        fallback_model = "microsoft/DialoGPT-medium"
        
        # Load tokenizer with DialoGPT-specific handling
        try:
            console.print("[blue]📝 Loading tokenizer...[/blue]")
            
            # Special handling for DialoGPT
            if "DialoGPT" in self.model_name or "dialogpt" in self.model_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    padding_side="left"
                )
                console.print("[green]✅ DialoGPT tokenizer loaded successfully[/green]")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    padding_side="left",
                    use_fast=False  # Use slow tokenizer to avoid tiktoken issues
                )
                console.print("[green]✅ Primary tokenizer loaded successfully[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Warning: Failed to load primary tokenizer: {e}[/yellow]")
            console.print(f"[yellow]🔄 Trying fallback tokenizer: {fallback_model}[/yellow]")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    fallback_model,
                    trust_remote_code=True,
                    padding_side="left"
                )
                using_fallback = True
                console.print(f"[yellow]⚠️ Using fallback tokenizer: {fallback_model}[/yellow]")
            except Exception as e2:
                console.print(f"[red]❌ Error: Fallback tokenizer also failed: {e2}[/red]")
                raise RuntimeError(f"Could not load any tokenizer. Primary error: {e}, Fallback error: {e2}")
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            console.print("[yellow]💡 Added missing pad token[/yellow]")
        
        # Configure quantization (skip for DialoGPT)
        quantization_config = None
        is_dialogpt = "DialoGPT" in self.model_name or "dialogpt" in self.model_name.lower()
        
        if not using_fallback and not is_dialogpt:
            # Prefer 4-bit for Zephyr or when 8-bit not requested
            try:
                if self.load_in_8bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                    )
                    console.print("[blue]⚡ Using 8-bit quantization[/blue]")
                else:
                    # Default to 4-bit to minimize memory usage
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    console.print("[blue]⚡ Using 4-bit quantization (NF4) via bitsandbytes[/blue]")
            except Exception as e:
                console.print(f"[yellow]⚠️ Warning: Could not configure quantization: {e}[/yellow]")
                console.print("[yellow]💡 Continuing without quantization[/yellow]")
        elif is_dialogpt:
            console.print("[blue]💡 DialoGPT detected - skipping quantization for compatibility[/blue]")
        
        # Prepare cache/offload directories
        cache_env = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or os.getenv("HF_HUB_CACHE")
        cache_dir = Path(cache_env) if cache_env else Path("D:/HFCache")
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        offload_dir = cache_dir / "offload"
        try:
            offload_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Load model with fallback
        try:
            console.print("[blue]🧠 Loading model...[/blue]")
            
            # DialoGPT-specific loading parameters
            if is_dialogpt:
                model_kwargs = {
                    "torch_dtype": torch.float32,  # DialoGPT works better with float32
                    "cache_dir": str(cache_dir),
                }
                # Don't use device_map="auto" for DialoGPT, it can cause issues
                if self.device != "auto":
                    model_kwargs["device_map"] = self.device
            else:
                model_kwargs = {
                    "quantization_config": quantization_config,
                    "device_map": self.device,
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if quantization_config else torch.float32,
                    "cache_dir": str(cache_dir),
                    "offload_folder": str(offload_dir),
                }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move DialoGPT to device manually if needed
            if is_dialogpt and self.device != "auto":
                device = self.device if self.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
                self.model = self.model.to(device)
            
            console.print("[green]✅ Primary model loaded successfully[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Warning: Failed to load primary model: {e}[/yellow]")
            console.print(f"[yellow]🔄 Trying fallback model: {fallback_model}[/yellow]")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    fallback_model,
                    torch_dtype=torch.float32,
                    cache_dir=str(cache_dir),
                )
                # Move to device
                device = self.device if self.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
                self.model = self.model.to(device)
                using_fallback = True
                console.print(f"[yellow]⚠️ Using fallback model: {fallback_model}[/yellow]")
                console.print("[yellow]💡 Note: This is a smaller model with different capabilities[/yellow]")
            except Exception as e2:
                console.print(f"[red]❌ Error: Fallback model also failed: {e2}[/red]")
                raise RuntimeError(f"Could not load any model. Primary error: {e}, Fallback error: {e2}")
        
        # Set up generation config (optimized for DialoGPT)
        if is_dialogpt or using_fallback:
            # DialoGPT-specific generation settings
            self.generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.7,  # Lower temperature for DialoGPT
                top_p=0.9,
                max_new_tokens=100,  # Shorter responses for conversation
                repetition_penalty=1.1,
                length_penalty=1.0,
                pad_token_id=self.tokenizer.eos_token_id,  # Use eos as pad for DialoGPT
                eos_token_id=self.tokenizer.eos_token_id,
            )
        else:
            # Standard generation config for other models
            self.generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                max_new_tokens=300,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        if using_fallback:
            console.print(f"[yellow]⚠️ Model fallback active: Using {fallback_model} instead of {self.model_name}[/yellow]")
            console.print("[yellow]💡 For full functionality, ensure you have sufficient resources for the primary model[/yellow]")
        else:
            console.print("[green]✅ Model loaded successfully[/green]")
        
        # Store fallback status for later reference
        self.using_fallback = using_fallback
        self.actual_model_name = fallback_model if using_fallback else self.model_name
    
    def generate_response(
        self, 
        prompt: str, 
        params: Optional[GenerationParams] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate a response using Vicuna or DialoGPT."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if params is None:
            params = GenerationParams()
        
        # Check if this is DialoGPT
        is_dialogpt = "DialoGPT" in self.actual_model_name or "dialogpt" in self.actual_model_name.lower()
        
        # Format input for DialoGPT conversation style
        if is_dialogpt:
            # For DialoGPT, use the conversation history parameter if available
            formatted_input = self._format_dialogpt_conversation(prompt, conversation_history) if conversation_history else prompt
        else:
            formatted_input = prompt
        
        # Update generation config with custom params
        gen_config = GenerationConfig(
            do_sample=params.do_sample,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k if hasattr(params, 'top_k') else 50,
            repetition_penalty=params.repetition_penalty,
            max_new_tokens=params.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,  # Use eos as pad for DialoGPT
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Tokenize input
        try:
            inputs = self.tokenizer(
                formatted_input, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=1000 if is_dialogpt else 2048  # Shorter context for DialoGPT
            )
        except Exception as e:
            # Fallback for tokenization issues
            inputs = self.tokenizer.encode(
                formatted_input + self.tokenizer.eos_token,
                return_tensors="pt"
            )
            inputs = {"input_ids": inputs}
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
                use_cache=True,
            )
        
        # Decode response (skip input tokens)
        input_length = inputs['input_ids'].shape[1]
        response_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Clean up response for DialoGPT
        if is_dialogpt:
            response = self._clean_dialogpt_response(response)
        
        return response.strip()
    
    def _format_dialogpt_conversation(self, current_prompt: str, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history for DialoGPT."""
        # Keep only recent history to avoid context length issues
        recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        
        conversation_parts = []
        for turn in recent_history:
            if isinstance(turn, dict):
                content = turn.get('content', str(turn))
            else:
                content = str(turn)
            conversation_parts.append(content)
        
        # Add current prompt
        conversation_parts.append(current_prompt)
        
        # Join with EOS token for DialoGPT
        return self.tokenizer.eos_token.join(conversation_parts) + self.tokenizer.eos_token
    
    def _clean_dialogpt_response(self, response: str) -> str:
        """Clean and format DialoGPT response."""
        if not response:
            return "I need more context to provide a meaningful response."
        
        # Remove extra whitespace and special tokens
        response = response.strip()
        response = response.replace(self.tokenizer.eos_token, "")
        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token:
            response = response.replace(self.tokenizer.pad_token, "")
        
        # Ensure response isn't empty
        if not response:
            return "Let me think about that..."
        
        return response


class LlamaCppWrapper:
    """Wrapper for llama.cpp GGUF models via llama-cpp-python.

    Notes:
        - Runs fully locally using a `.gguf` model file
        - Supports CPU by default; GPU offloading if compiled with cuBLAS
        - Streaming is not used here; we generate then return full text
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 0,
        n_gpu_layers: int = 0,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_new_tokens: int = 300,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.max_new_tokens = max_new_tokens

        self.llm: Optional["Llama"] = None

    def load_model(self) -> None:
        from rich.console import Console
        console = Console()

        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it to use GGUF backend."
            )

        console.print(f"[blue]🤖 Loading GGUF model from: {self.model_path}[/blue]")
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads or None,
                n_gpu_layers=self.n_gpu_layers,
                logits_all=False,
                embedding=False,
                verbose=False,
            )
            console.print("[green]✅ GGUF model loaded successfully[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to load GGUF model: {e}[/red]")
            raise

    def is_loaded(self) -> bool:
        return self.llm is not None

    def get_model_info(self) -> Dict[str, Any]:
        if not self.is_loaded():
            return {"status": "not_loaded"}
        return {
            "status": "loaded",
            "backend": "llama.cpp",
            "model_path": self.model_path,
            "model_name": "zephyr-7b-alpha",
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
        }

    def generate_response(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
    ) -> str:
        if not self.llm:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if params is None:
            params = GenerationParams(
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repeat_penalty,
            )

        # llama.cpp expects the full prompt; no separate tokenizer step here.
        result = self.llm(
            prompt=prompt,
            max_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            repeat_penalty=params.repetition_penalty,
            stop=None,
        )

        text = result.get("choices", [{}])[0].get("text", "")
        return text.strip()
    
    def format_debate_prompt(
        self,
        role: str,
        agent_name: str,
        personality: str,
        topic: str,
        stance: str,
        conversation_history: List[Dict[str, str]],
        retrieved_passages: List[str],
        opponent_message: str,
        heat_level: float,
        strategy: str = "logical_reasoning"
    ) -> str:
        """Format a debate prompt for Vicuna."""
        
        # Build conversation context
        history_text = ""
        if conversation_history:
            history_text = "\n".join([
                f"{msg['speaker']}: {msg['content']}" 
                for msg in conversation_history[-5:]  # Last 5 messages
            ])
        
        # Build evidence context
        evidence_text = ""
        if retrieved_passages:
            evidence_text = "\n".join([
                f"Evidence {i+1}: {passage}" 
                for i, passage in enumerate(retrieved_passages[:3])
            ])
        
        # Determine tone based on heat level
        tone_instruction = ""
        if heat_level > 0.8:
            tone_instruction = "You are feeling heated and passionate. Be more aggressive but stay factual."
        elif heat_level > 0.5:
            tone_instruction = "You are engaged and assertive. Push back firmly on weak points."
        else:
            tone_instruction = "You are calm and analytical. Focus on logical reasoning."
        
        # Strategy-specific instructions
        strategy_instructions = {
            "logical_reasoning": "Focus on logical arguments and evidence. Point out flaws in reasoning.",
            "emotional_appeal": "Appeal to values and emotions while staying grounded in facts.",
            "citation_heavy": "Reference the evidence extensively. Quote specific passages.",
            "questioning": "Ask probing questions to expose weaknesses in the opponent's position.",
            "analogy_based": "Use analogies and examples to make your points more relatable."
        }
        
        strategy_instruction = strategy_instructions.get(strategy, strategy_instructions["logical_reasoning"])
        
        prompt = f"""A conversation between a human and an AI assistant.

[Role: {role} Debater - {agent_name}]
You are {agent_name}, a {personality} debater taking the {stance} position on: "{topic}"

{tone_instruction}
{strategy_instruction}

Conversation History:
{history_text}

Retrieved Evidence:
{evidence_text}

Opponent just said:
"{opponent_message}"

Instructions:
- Respond naturally as {agent_name} would
- Reference the evidence when relevant
- Stay focused on the topic: {topic}
- Keep response to 2-4 sentences
- Be engaging but respectful

{agent_name}:"""

        return prompt

