"""Main debate orchestration loop."""

from __future__ import annotations

import random
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag.chunker import DocumentChunker
from rag.bm25_index import BM25Index
from rag.embeddings import EmbeddingIndex, HybridRetriever
from agents.llm_wrapper import VicunaWrapper, LlamaCppWrapper
from agents.debater import Debater, DebaterPersona, DebateContext
from agents.mediator import Mediator, MediatorPersona
from agents.heat import HeatDetector
from agents.bandit import ContextualBandit, DebateOutcome
from .utils import DebateConfig, format_debate_output, format_mediator_output, calculate_debate_outcome, save_debate_log, SafetyFilter


class DebateOrchestrator:
    """Orchestrates multi-agent debates with RAG and learning."""
    
    def __init__(self, config: DebateConfig, data_dir: Path):
        self.config = config
        self.data_dir = data_dir
        self.console = Console()
        
        # Initialize components
        self.llm_wrapper: Optional[VicunaWrapper] = None
        self.retriever: Optional[HybridRetriever] = None
        self.heat_detector = HeatDetector()
        self.safety_filter = SafetyFilter()
        
        # Initialize agents
        self.pro_debater: Optional[Debater] = None
        self.anti_debater: Optional[Debater] = None
        self.mediator: Optional[Mediator] = None
        
        # Debate state
        self.current_topic = ""
        self.conversation_history: List[Dict[str, str]] = []
        self.debate_log: List[Dict[str, Any]] = []
        
    def initialize(self) -> None:
        """Initialize all components."""
        self.console.print("[bold blue]Initializing Advanced Debate System...[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Load LLM
            task1 = progress.add_task(f"Loading {self.config.model_name} model...", total=None)
            self._initialize_llm()
            progress.update(task1, completed=True)
            
            # Build RAG system
            task2 = progress.add_task("Building RAG indices...", total=None)
            self._initialize_rag()
            progress.update(task2, completed=True)
            
            # Initialize agents
            task3 = progress.add_task("Initializing agents...", total=None)
            self._initialize_agents()
            progress.update(task3, completed=True)
        
        self.console.print("[bold green]System initialized successfully![/bold green]")
    
    def _initialize_llm(self) -> None:
        """Initialize the chat LLM wrapper using configured model."""
        # Select backend
        if getattr(self.config, 'backend', 'transformers') == 'llama.cpp':
            # GGUF backend
            self.llm_wrapper = LlamaCppWrapper(
                model_path=self.config.gguf_model_path,
                n_ctx=self.config.gguf_n_ctx,
                n_threads=self.config.gguf_n_threads,
                n_gpu_layers=self.config.gguf_n_gpu_layers,
                temperature=self.config.temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                max_new_tokens=self.config.max_new_tokens,
            )
            self.llm_wrapper.load_model()
        else:
            # Transformers backend (Vicuna/Zephyr/etc.)
            self.llm_wrapper = VicunaWrapper(
                model_name=self.config.model_name,
                device=self.config.device,
                load_in_8bit=self.config.load_in_8bit
            )
            self.llm_wrapper.load_model()
    
    def _initialize_rag(self) -> None:
        """Initialize the RAG retrieval system with comprehensive error handling."""
        try:
            source_docs_dir = self.data_dir / "source_docs"
            embeddings_dir = self.data_dir / "embeddings"
            
            # Check if source documents exist
            if not source_docs_dir.exists():
                self.console.print(f"[yellow]⚠️ Warning: No source documents found in {source_docs_dir}[/yellow]")
                self.console.print("[yellow]💡 Creating empty retriever - agents will debate without document grounding[/yellow]")
                self.retriever = EmptyRetriever()
                return
            
            # Check if directory is empty
            doc_files = list(source_docs_dir.glob("*.md")) + list(source_docs_dir.glob("*.txt")) + list(source_docs_dir.glob("*.rst"))
            if not doc_files:
                self.console.print(f"[yellow]⚠️ Warning: No supported documents found in {source_docs_dir}[/yellow]")
                self.console.print("[yellow]💡 Supported formats: .md, .txt, .rst[/yellow]")
                self.retriever = EmptyRetriever()
                return
            
            self.console.print(f"[blue]📄 Found {len(doc_files)} document files[/blue]")
            
            # Chunk documents
            self.console.print("[blue]✂️ Chunking documents...[/blue]")
            try:
                chunker = DocumentChunker(
                    chunk_size=self.config.chunk_size,
                    overlap=self.config.chunk_overlap
                )
                chunks = chunker.chunk_directory(source_docs_dir)
                
                if not chunks:
                    self.console.print("[yellow]⚠️ Warning: No document chunks created[/yellow]")
                    self.retriever = EmptyRetriever()
                    return
                
                self.console.print(f"[green]✅ Created {len(chunks)} document chunks[/green]")
            except Exception as e:
                self.console.print(f"[red]❌ Error chunking documents: {e}[/red]")
                self.retriever = EmptyRetriever()
                return
            
            # Build BM25 index
            self.console.print("[blue]🔍 Building BM25 keyword index...[/blue]")
            try:
                bm25_index = BM25Index()
                bm25_index.build_index(chunks)
                self.console.print("[green]✅ BM25 index built successfully[/green]")
            except Exception as e:
                self.console.print(f"[red]❌ Error building BM25 index: {e}[/red]")
                self.retriever = EmptyRetriever()
                return
            
            # Build embedding index if enabled
            embedding_index = None
            if self.config.use_embeddings:
                self.console.print("[blue]🧠 Building semantic embedding index...[/blue]")
                try:
                    embedding_index = EmbeddingIndex(self.config.embedding_model)
                    embedding_index.build_index(chunks)
                    self.console.print("[green]✅ Embedding index built successfully[/green]")
                except Exception as e:
                    self.console.print(f"[yellow]⚠️ Warning: Could not build embedding index: {e}[/yellow]")
                    self.console.print("[yellow]💡 Falling back to BM25-only retrieval[/yellow]")
                    embedding_index = None
            
            # Create hybrid retriever
            if embedding_index:
                self.console.print("[blue]🔗 Creating hybrid retriever (BM25 + Embeddings)...[/blue]")
                try:
                    self.retriever = HybridRetriever(
                        bm25_index=bm25_index,
                        embedding_index=embedding_index,
                        bm25_weight=self.config.bm25_weight
                    )
                    self.console.print("[green]✅ Hybrid retriever created successfully[/green]")
                except Exception as e:
                    self.console.print(f"[yellow]⚠️ Warning: Could not create hybrid retriever: {e}[/yellow]")
                    self.console.print("[yellow]💡 Falling back to BM25-only retrieval[/yellow]")
                    self.retriever = BM25OnlyRetriever(bm25_index)
            else:
                self.console.print("[blue]🔍 Creating BM25-only retriever...[/blue]")
                self.retriever = BM25OnlyRetriever(bm25_index)
                self.console.print("[green]✅ BM25-only retriever created successfully[/green]")
                
        except Exception as e:
            self.console.print(f"[red]❌ Fatal error initializing RAG system: {e}[/red]")
            self.retriever = EmptyRetriever()
    
    def _initialize_agents(self) -> None:
        """Initialize the debate agents."""
        # Create bandit learners
        pro_bandit = ContextualBandit(
            strategies=self.config.bandit_strategies,
            epsilon=self.config.bandit_epsilon,
            learning_rate=self.config.bandit_learning_rate,
            save_path=self.data_dir / "bandit_pro.json"
        )
        pro_bandit.load_stats()
        
        anti_bandit = ContextualBandit(
            strategies=self.config.bandit_strategies,
            epsilon=self.config.bandit_epsilon,
            learning_rate=self.config.bandit_learning_rate,
            save_path=self.data_dir / "bandit_anti.json"
        )
        anti_bandit.load_stats()
        
        # Create debater personas
        pro_persona = DebaterPersona(
            name=self.config.debater_pro["name"],
            role="pro",
            personality=self.config.debater_pro["personality"],
            aggression_level=self.config.debater_pro["aggression_level"],
            preferred_strategies=self.config.bandit_strategies
        )
        
        anti_persona = DebaterPersona(
            name=self.config.debater_anti["name"],
            role="anti",
            personality=self.config.debater_anti["personality"],
            aggression_level=self.config.debater_anti["aggression_level"],
            preferred_strategies=self.config.bandit_strategies
        )
        
        # Create debaters
        self.pro_debater = Debater(
            persona=pro_persona,
            llm_wrapper=self.llm_wrapper,
            bandit=pro_bandit,
            heat_detector=HeatDetector()
        )
        
        self.anti_debater = Debater(
            persona=anti_persona,
            llm_wrapper=self.llm_wrapper,
            bandit=anti_bandit,
            heat_detector=HeatDetector()
        )
        
        # Create mediator
        mediator_persona = MediatorPersona(
            name=self.config.mediator["name"],
            intervention_style=self.config.mediator["intervention_style"],
            cooling_effectiveness=self.config.mediator["cooling_effectiveness"],
            intervention_threshold=self.config.heat_threshold
        )
        
        self.mediator = Mediator(
            persona=mediator_persona,
            llm_wrapper=self.llm_wrapper,
            heat_detector=self.heat_detector
        )
    
    def run_debate(self, topic: str, max_turns: Optional[int] = None) -> Dict[str, Any]:
        """Run a complete debate on the given topic."""
        max_turns = max_turns or self.config.max_turns
        self.current_topic = topic
        
        # Reset state
        self.conversation_history.clear()
        self.debate_log.clear()
        self.pro_debater.reset_for_new_debate()
        self.anti_debater.reset_for_new_debate()
        self.mediator.reset_for_new_debate()
        
        # Display debate start
        self.console.print(Panel.fit(
            f"[bold]Debate Topic:[/bold] {topic}\n"
            f"[bold]Pro:[/bold] {self.pro_debater.persona.name}\n"
            f"[bold]Anti:[/bold] {self.anti_debater.persona.name}\n"
            f"[bold]Mediator:[/bold] {self.mediator.persona.name}",
            title="Starting Debate",
            style="bold blue"
        ))
        
        # Main debate loop
        current_speaker = "pro"  # Pro goes first
        pro_heat = 0.0
        anti_heat = 0.0
        
        for turn in range(max_turns):
            try:
                # Determine current debater
                if current_speaker == "pro":
                    debater = self.pro_debater
                    opponent_heat = anti_heat
                else:
                    debater = self.anti_debater
                    opponent_heat = pro_heat
                
                # Retrieve relevant passages
                retrieved_passages = self._retrieve_passages(topic)
                
                # Create debate context
                context = DebateContext(
                    topic=topic,
                    turn_number=turn + 1,
                    conversation_history=self.conversation_history[-self.config.rolling_history_size:],
                    retrieved_passages=retrieved_passages,
                    opponent_last_message=self.conversation_history[-1]["content"] if self.conversation_history else "",
                    current_heat_level=pro_heat if current_speaker == "pro" else anti_heat,
                    opponent_heat_level=opponent_heat,
                    current_speaker=current_speaker
                )
                
                # Generate argument
                result = debater.generate_argument(context)
                argument = result.content
                strategy = result.strategy
                heat_metrics = result.heat_metrics
                
                # Apply safety filter
                argument = self.safety_filter.filter_content(argument)
                
                # Update heat levels
                if current_speaker == "pro":
                    pro_heat = heat_metrics.heat_score
                else:
                    anti_heat = heat_metrics.heat_score
                
                # Add to conversation history
                self.conversation_history.append({
                    "speaker": debater.persona.name,
                    "content": argument,
                    "turn": turn + 1,
                    "strategy": strategy
                })
                
                # Log the turn
                self.debate_log.append({
                    "type": "argument",
                    "turn": turn + 1,
                    "speaker": current_speaker,
                    "speaker_name": debater.persona.name,
                    "content": argument,
                    "strategy": strategy,
                    "heat_score": heat_metrics.heat_score,
                    "retrieved_passages": len(retrieved_passages)
                })
                
                # Display the argument
                output = format_debate_output(
                    debater.persona.name,
                    argument,
                    strategy,
                    heat_metrics.heat_score,
                    turn + 1
                )
                
                color = "red" if current_speaker == "anti" else "blue"
                self.console.print(Panel.fit(output, style=color))
                
                # Check for mediator intervention
                opponent_debater = self.anti_debater if current_speaker == "pro" else self.pro_debater
                opponent_heat_metrics = self.heat_detector.analyze_heat(
                    self.conversation_history[-2]["content"] if len(self.conversation_history) >= 2 else "",
                    opponent_debater.persona.name
                )
                
                if self.mediator.should_intervene(
                    heat_metrics, opponent_heat_metrics, turn + 1, self.conversation_history
                ):
                    intervention = self.mediator.intervene(
                        heat_metrics, opponent_heat_metrics, turn + 1, topic, self.conversation_history
                    )
                    
                    # Display intervention
                    intervention_output = format_mediator_output(intervention.content, turn + 1)
                    self.console.print(Panel.fit(intervention_output, style="yellow"))
                    
                    # Log intervention
                    self.debate_log.append({
                        "type": "intervention",
                        "turn": turn + 1,
                        "content": intervention.content,
                        "cooling_effect": intervention.cooling_effect
                    })

                    # Append mediator message to conversation history so agents perceive it
                    mediator_entry = {
                        "speaker": self.mediator.persona.name,
                        "content": intervention.content,
                        "turn": turn + 1,
                        "strategy": "mediator"
                    }
                    self.conversation_history.append(mediator_entry)

                    # Let both debaters process the mediator intervention so they adapt
                    try:
                        if self.pro_debater:
                            # newer interface: pass mediator name/content/cooling
                            self.pro_debater.process_mediator_intervention(
                                self.mediator.persona.name,
                                intervention.content,
                                intervention.cooling_effect,
                                ttl=2
                            )
                        if self.anti_debater:
                            self.anti_debater.process_mediator_intervention(
                                self.mediator.persona.name,
                                intervention.content,
                                intervention.cooling_effect,
                                ttl=2
                            )
                    except Exception:
                        # Non-fatal: continue debate even if debaters fail to process
                        pass

                    # Analyze mediator message heat (should be low/neutral) and update heat history
                    mediator_heat_metrics = self.heat_detector.analyze_heat(intervention.content, self.mediator.persona.name)

                    # Apply multiplicative cooling so mediator has stronger effect when cooling_effect is higher
                    # Also consider mediator emotional intensity (if mediator is empathetic, it cools more)
                    cooling_factor = 1.0 - intervention.cooling_effect * (0.5 + mediator_heat_metrics.emotional_intensity * 0.5)
                    cooling_factor = max(0.0, min(1.0, cooling_factor))

                    pro_heat = max(0.0, pro_heat * cooling_factor)
                    anti_heat = max(0.0, anti_heat * cooling_factor)

                    # Inform both debaters so they "perceive" and respond to mediator guidance
                    try:
                        if self.pro_debater:
                            self.pro_debater.process_mediator_intervention(self.mediator.persona.name, intervention.content, intervention.cooling_effect)
                    except Exception:
                        pass
                    try:
                        if self.anti_debater:
                            self.anti_debater.process_mediator_intervention(self.mediator.persona.name, intervention.content, intervention.cooling_effect)
                    except Exception:
                        pass
                    
                    # Check if debate should continue
                    if not intervention.should_continue:
                        self.console.print("[yellow]Mediator has ended the debate due to excessive heat.[/yellow]")
                        break
                
                # Switch speakers
                current_speaker = "anti" if current_speaker == "pro" else "pro"
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Debate interrupted by user.[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error during debate turn {turn + 1}: {e}[/red]")
                break
        
        # Calculate final outcome
        outcome = self._calculate_final_outcome()
        
        # Display results
        self._display_results(outcome)
        
        return outcome
    
    def _retrieve_passages(self, query: str) -> List[str]:
        """Retrieve relevant passages for the current query."""
        if not self.retriever:
            return []
        
        try:
            results = self.retriever.search(query, self.config.max_retrieved_chunks)
            return [chunk.content for chunk, score in results]
        except Exception as e:
            self.console.print(f"[yellow]Warning: RAG retrieval failed: {e}[/yellow]")
            return []
    
    def _calculate_final_outcome(self) -> Dict[str, Any]:
        """Calculate the final debate outcome."""
        pro_stats = self.pro_debater.get_stats()
        anti_stats = self.anti_debater.get_stats()
        mediator_stats = self.mediator.get_intervention_stats()
        
        outcome = calculate_debate_outcome(
            pro_stats["performance"],
            anti_stats["performance"],
            mediator_stats
        )
        
        # Add detailed statistics
        outcome.update({
            "topic": self.current_topic,
            "total_turns": len([log for log in self.debate_log if log["type"] == "argument"]),
            "pro_stats": pro_stats,
            "anti_stats": anti_stats,
            "mediator_stats": mediator_stats
        })
        
        return outcome
    
    def _display_results(self, outcome: Dict[str, Any]) -> None:
        """Display the final debate results."""
        winner = outcome["winner"]
        quality = outcome["debate_quality"]
        
        result_text = f"""
[bold]Winner:[/bold] {winner}
[bold]Debate Quality:[/bold] {quality:.2f}/1.0
[bold]Total Turns:[/bold] {outcome['total_turns']}
[bold]Interventions:[/bold] {outcome['total_interventions']}
[bold]Average Heat:[/bold] {outcome['avg_heat']:.2f}
        """
        
        self.console.print(Panel.fit(
            result_text,
            title="Debate Results",
            style="bold green"
        ))
        
        # Show strategy learning
        pro_strategies = outcome["pro_stats"]["strategy_preferences"]
        anti_strategies = outcome["anti_stats"]["strategy_preferences"]
        
        strategy_text = "[bold]Strategy Learning:[/bold]\n"
        strategy_text += f"[blue]{self.pro_debater.persona.name}:[/blue] "
        strategy_text += ", ".join([f"{s}: {v:.2f}" for s, v in list(pro_strategies.items())[:3]])
        strategy_text += f"\n[red]{self.anti_debater.persona.name}:[/red] "
        strategy_text += ", ".join([f"{s}: {v:.2f}" for s, v in list(anti_strategies.items())[:3]])
        
        self.console.print(Panel.fit(strategy_text, title="📊 Learning Progress"))


class EmptyRetriever:
    """Empty retriever for when no documents are available."""
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Any, float]]:
        """Return empty results."""
        return []


class BM25OnlyRetriever:
    """BM25-only retriever when embeddings are not available."""
    
    def __init__(self, bm25_index):
        self.bm25_index = bm25_index
    
    def search(self, query: str, top_k: int = 5):
        results = self.bm25_index.search(query, top_k)
        return [(result.chunk, result.score) for result in results]


def run_debate_session(config: 'DebateConfig', topic: str = None, save_log: bool = True, verbose: bool = False) -> Dict[str, Any]:
    """Run a complete debate session with comprehensive error handling."""
    try:
        # Create data directory path
        data_dir = Path("data")
        
        # Initialize orchestrator
        console = Console()
        console.print("[blue]🏗️ Initializing debate orchestrator...[/blue]")
        
        try:
            orchestrator = DebateOrchestrator(config, data_dir)
            orchestrator.initialize()
            console.print("[green]✅ Orchestrator initialized successfully[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error initializing orchestrator: {e}[/red]")
            if verbose:
                console.print(traceback.format_exc())
            raise
        
        # Select topic
        if not topic:
            topics = [
                "Universal Basic Income should be implemented globally",
                "AI systems should be regulated like pharmaceutical drugs",
                "Social media platforms should be treated as public utilities",
                "Climate change requires immediate economic transformation",
                "Privacy is more important than security in digital systems"
            ]
            topic = random.choice(topics)
            console.print(f"[yellow]🎲 Selected random topic: {topic}[/yellow]")
        else:
            console.print(f"[blue]📝 Using provided topic: {topic}[/blue]")
        
        # Run debate
        console.print("[blue]🚀 Starting debate...[/blue]")
        try:
            outcome = orchestrator.run_debate(topic)
            console.print("[green]✅ Debate completed successfully[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error during debate: {e}[/red]")
            if verbose:
                console.print(traceback.format_exc())
            raise
        
        # Save log if requested
        if save_log:
            try:
                log_path = data_dir / "logs" / f"debate_{topic[:30].replace(' ', '_')}.yaml"
                save_debate_log(orchestrator.debate_log, outcome, topic, log_path)
                console.print(f"[green]💾 Debate log saved to {log_path}[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️ Warning: Could not save debate log: {e}[/yellow]")
        
        return outcome
        
    except Exception as e:
        console.print(f"[red]❌ Fatal error in debate session: {e}[/red]")
        if verbose:
            console.print(traceback.format_exc())
        raise
