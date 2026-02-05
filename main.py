"""Main entry point for the Advanced Agentic Debate System."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

try:
    from runtime.loop import run_debate_session
    from runtime.utils import load_config, validate_config as validate_config_object
except ImportError as e:
    print(f"[red]Import error: {e}[/red]")
    print("[yellow]Some dependencies may be missing. Please install: pip install -r requirements.txt[/yellow]")
    sys.exit(1)


app = typer.Typer(add_completion=False, help="Advanced Agentic Debate System with Vicuna-7B")
console = Console()


@app.command()
def debate(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to configuration file"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Debate topic (random if not specified)"),
    turns: Optional[int] = typer.Option(None, "--turns", help="Maximum number of turns"),
    save_log: bool = typer.Option(True, "--save-log/--no-save-log", help="Save debate log to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run a debate between two AI agents with mediator oversight."""
    
    try:
        # Validate config file exists
        config_path = Path(config)
        if not config_path.exists():
            console.print(f"[red]❌ Error: Configuration file {config_path} not found[/red]")
            console.print(f"[yellow]💡 Tip: Run 'python main.py setup' to create a default configuration[/yellow]")
            raise typer.Exit(1)
        
        # Load and validate configuration
        console.print("[blue]📋 Loading configuration...[/blue]")
        try:
            debate_config = load_config(config_path)
            console.print("[green]✅ Configuration loaded successfully[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error loading configuration: {e}[/red]")
            if verbose:
                console.print(traceback.format_exc())
            raise typer.Exit(1)
        
        # Validate configuration
        console.print("[blue]🔍 Validating configuration...[/blue]")
        try:
            validation_result = validate_config_object(debate_config)
            if not validation_result["valid"]:
                console.print(f"[red]❌ Configuration validation failed:[/red]")
                for error in validation_result["errors"]:
                    console.print(f"  • {error}")
                raise typer.Exit(1)
            console.print("[green]✅ Configuration is valid[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error validating configuration: {e}[/red]")
            if verbose:
                console.print(traceback.format_exc())
            raise typer.Exit(1)
        
        # Apply command line overrides
        if turns:
            debate_config.max_turns = turns
            console.print(f"[yellow]📝 Overriding max turns to {turns}[/yellow]")
        
        # Display system info
        console.print(Panel.fit(
            "[bold blue]Advanced Agentic Debate System[/bold blue]\n"
            f"Model: {debate_config.model_name}\n"
            f"RAG: {'Enabled' if debate_config.use_embeddings else 'BM25 Only'}\n"
            f"Max Turns: {debate_config.max_turns}\n"
            f"Topic: {topic or 'Random selection'}",
            title="🤖 System Info"
        ))
        
        # Run the debate
        console.print("[blue]🚀 Starting debate session...[/blue]")
        try:
            outcome = run_debate_session(debate_config, topic, save_log, verbose)
            console.print("[green]✅ Debate completed successfully[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error during debate: {e}[/red]")
            if verbose:
                console.print(traceback.format_exc())
            raise typer.Exit(1)
        
        # Display results
        if verbose:
            console.print("\n[bold]📊 Detailed Results:[/bold]")
            console.print(outcome)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Debate session interrupted by user[/yellow]")
        raise typer.Exit(0)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")
        if verbose:
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def setup(
    data_dir: str = typer.Option("data", "--data-dir", "-d", help="Data directory path"),
    download_model: bool = typer.Option(False, "--download-model", help="Download Vicuna model"),
    create_sample_docs: bool = typer.Option(True, "--create-sample-docs", help="Create sample documents"),
    create_config: bool = typer.Option(True, "--create-config", help="Create default config.yaml")
):
    """Set up the debate system with initial data and models."""
    
    try:
        data_path = Path(data_dir)
        
        console.print("[bold blue]🚀 Setting up Advanced Debate System...[/bold blue]")
        
        # Create directory structure
        console.print("[blue]📁 Creating directory structure...[/blue]")
        try:
            (data_path / "source_docs").mkdir(parents=True, exist_ok=True)
            (data_path / "embeddings").mkdir(parents=True, exist_ok=True)
            (data_path / "logs").mkdir(parents=True, exist_ok=True)
            console.print(f"[green]✅ Created directory structure in {data_path}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error creating directories: {e}[/red]")
            raise typer.Exit(1)
        
        # Create default config if requested
        if create_config:
            console.print("[blue]⚙️ Creating default configuration...[/blue]")
            try:
                config_path = Path("config.yaml")
                if not config_path.exists():
                    from runtime.utils import create_default_config
                    create_default_config(config_path)
                    console.print(f"[green]✅ Created default config: {config_path}[/green]")
                else:
                    console.print(f"[yellow]⚠️ Config file already exists: {config_path}[/yellow]")
            except Exception as e:
                console.print(f"[red]❌ Error creating config: {e}[/red]")
                raise typer.Exit(1)
        
        # Create sample documents if requested
        if create_sample_docs:
            console.print("[blue]📄 Creating sample documents...[/blue]")
            try:
                sample_doc = """# Universal Basic Income: A Comprehensive Analysis

## Introduction
Universal Basic Income (UBI) is a social program where all citizens receive regular, unconditional cash payments from the government. This concept has gained significant attention as a potential solution to economic inequality and job displacement due to automation.

## Arguments in Favor of UBI

### Economic Security
UBI provides a financial safety net that ensures basic economic security for all citizens. This can reduce poverty and provide stability in an increasingly uncertain job market.

### Simplification of Welfare
UBI could replace complex welfare systems with a single, universal program, reducing bureaucratic overhead and ensuring that help reaches those who need it most.

### Freedom and Dignity
By providing unconditional income, UBI respects individual autonomy and dignity, allowing people to make choices about work and life without the stigma often associated with traditional welfare programs.

### Economic Stimulus
Regular cash payments to all citizens can stimulate economic activity, as people spend money on goods and services, creating a multiplier effect throughout the economy.

## Arguments Against UBI

### Cost and Funding
The cost of providing meaningful UBI payments to all citizens would be enormous, requiring significant tax increases or reallocation of government spending.

### Work Disincentives
Critics argue that guaranteed income might reduce the incentive to work, potentially leading to decreased productivity and economic output.

### Inflation Risk
Injecting large amounts of cash into the economy through UBI could lead to inflation, potentially eroding the purchasing power of the payments themselves.

### Political Feasibility
The political challenges of implementing UBI are substantial, requiring broad consensus and significant changes to existing social and economic systems.

## International Examples

### Finland's Experiment
Finland conducted a two-year UBI experiment from 2017-2018, providing €560 per month to 2,000 unemployed individuals. Results showed modest improvements in well-being and employment.

### Kenya's GiveDirectly Program
The charity GiveDirectly has been providing unconditional cash transfers to villages in Kenya since 2016, offering insights into the long-term effects of basic income in developing contexts.

## Conclusion
The debate over Universal Basic Income involves complex trade-offs between economic security, work incentives, fiscal responsibility, and social values. While pilot programs provide some evidence, the full effects of large-scale UBI implementation remain uncertain and continue to be subjects of intense debate among economists, policymakers, and citizens worldwide.
"""
                
                sample_path = data_path / "source_docs" / "ubi_analysis.md"
                sample_path.write_text(sample_doc)
                console.print(f"[green]✅ Created sample document: {sample_path}[/green]")
            except Exception as e:
                console.print(f"[red]❌ Error creating sample documents: {e}[/red]")
                raise typer.Exit(1)
        
        # Model download info
        if download_model:
            console.print("[yellow]📥 Note: Model downloading will happen automatically when first running a debate.[/yellow]")
            console.print("[yellow]💾 Ensure you have sufficient disk space (~13GB for Vicuna-7B).[/yellow]")
        
        console.print("\n[bold green]🎉 Setup complete![/bold green]")
        console.print("[blue]💡 Next steps:[/blue]")
        console.print("  1. Run 'python main.py debate' to start your first debate")
        console.print("  2. Run 'python main.py validate-config' to check your configuration")
        console.print("  3. Add your own documents to the 'data/source_docs' folder")
        
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]❌ Unexpected error during setup: {e}[/red]")
        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def validate_config(
    config: str = typer.Option("config.yaml", "--config", "-c", help="Path to configuration file")
):
    """Validate the configuration file."""
    
    try:
        config_path = Path(config)
        if not config_path.exists():
            console.print(f"[red]❌ Error: Configuration file {config_path} not found[/red]")
            raise typer.Exit(1)
        
        console.print("[blue]📋 Loading configuration...[/blue]")
        try:
            debate_config = load_config(config_path)
            console.print("[green]✅ Configuration loaded successfully[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error loading configuration: {e}[/red]")
            raise typer.Exit(1)
        
        console.print("[blue]🔍 Validating configuration...[/blue]")
        try:
            validation_result = validate_config_object(debate_config)
            if not validation_result["valid"]:
                console.print(f"[red]❌ Configuration validation failed:[/red]")
                for error in validation_result["errors"]:
                    console.print(f"  • {error}")
                raise typer.Exit(1)
            console.print("[green]✅ Configuration is valid[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error validating configuration: {e}[/red]")
            raise typer.Exit(1)
        
        # Display key settings
        console.print(Panel.fit(
            f"Model: {debate_config.model_name}\n"
            f"Device: {debate_config.device}\n"
            f"8-bit Loading: {debate_config.load_in_8bit}\n"
            f"Max Turns: {debate_config.max_turns}\n"
            f"RAG Enabled: {debate_config.use_embeddings}\n"
            f"Strategies: {', '.join(debate_config.bandit_strategies)}",
            title="📊 Configuration Summary"
        ))
        
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")
        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def interactive():
    """Run the debate system in interactive mode."""
    
    try:
        console.print(Panel.fit(
            "[bold blue]Interactive Debate Mode[/bold blue]\n"
            "Enter topics and watch AI agents debate with real-time learning!",
            title="🎭 Interactive Mode"
        ))
        
        config_path = Path("config.yaml")
        if not config_path.exists():
            console.print("[red]❌ Error: config.yaml not found. Run 'python main.py setup' first.[/red]")
            raise typer.Exit(1)
        
        # Load and validate config once
        console.print("[blue]📋 Loading configuration...[/blue]")
        try:
            debate_config = load_config(config_path)
            validation_result = validate_config_object(debate_config)
            if not validation_result["valid"]:
                console.print(f"[red]❌ Configuration validation failed:[/red]")
                for error in validation_result["errors"]:
                    console.print(f"  • {error}")
                raise typer.Exit(1)
            console.print("[green]✅ Configuration loaded and validated[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error loading configuration: {e}[/red]")
            raise typer.Exit(1)
        
        console.print("[green]🎉 Ready for interactive debates![/green]")
        
        while True:
            try:
                topic = typer.prompt("\n💬 Enter a debate topic (or 'quit' to exit)")
                
                if topic.lower() in ['quit', 'exit', 'q']:
                    console.print("[blue]👋 Goodbye![/blue]")
                    break
                
                console.print(f"\n[bold blue]🚀 Starting debate on: {topic}[/bold blue]")
                
                try:
                    outcome = run_debate_session(debate_config, topic, save_log=True, verbose=False)
                    console.print("[green]✅ Debate completed![/green]")
                    
                    # Ask if user wants to continue
                    continue_debate = typer.confirm("\n🔄 Would you like to start another debate?")
                    if not continue_debate:
                        console.print("[blue]👋 Goodbye![/blue]")
                        break
                        
                except KeyboardInterrupt:
                    console.print("\n[yellow]⚠️ Debate interrupted. Returning to topic selection.[/yellow]")
                    continue
                except Exception as e:
                    console.print(f"[red]❌ Error during debate: {e}[/red]")
                    console.print("[yellow]💡 Try a different topic or check your configuration[/yellow]")
                    continue
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]⚠️ Interactive mode ended.[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]❌ Unexpected error: {e}[/red]")
                console.print("[yellow]💡 Returning to topic selection...[/yellow]")
                continue
                
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]❌ Fatal error in interactive mode: {e}[/red]")
        console.print(traceback.format_exc())
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
