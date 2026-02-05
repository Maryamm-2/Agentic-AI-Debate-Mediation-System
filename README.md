# Agentic AI Debate & Mediation System

A sophisticated multi-agent debate system featuring Vicuna-7B, document-grounded arguments (RAG), adaptive learning, and intelligent mediation.

## Project Overview

The **Agentic AI Debate & Mediation System** is a research system designed to facilitate and study **autonomous debates between AI agents**.

### Project Purpose & Research Goal
The core purpose of this project is **"Setting AI to talk to each other"** in a controlled, adversarial environment to observe their behavior. Specifically, it aims to:
1.  **Simulate Autonomous Discourse**: Create a loop where two AI agents (Pro & Anti) debate a topic endlessly without human input, reacting only to each other.
2.  **Study Mediation Effects**: Quantify how an intelligent **Mediator Agent** influences the debate. Does a mediator actually reduce "heat" (aggression)? Can it steer the conversation back to facts?
3.  **Analyze Agentic Dynamics**: Observe how agents adapt their strategies (e.g., becoming more aggressive or more logical) in response to their opponent and the mediator.

This serves as a platform for **Multi-Agent Systems (MAS)** research, focusing on conflict resolution and safe agent interaction.

### Key Features
*   **Vicuna-7B Integration**: Lifelike debate generation using state-of-the-art open-source language models.
*   **RAG System**: Arguments are grounded in retrieved facts using a hybrid BM25 + FAISS pipeline.
*   **Adaptive Learning**: Agents use Contextual Bandits to learn and switch strategies (e.g., from "Logical" to "Emotional") based on performance.
*   **Intelligent Mediation**: A dedicated Mediator agent monitors "Heat" levels and intervenes to maintain productive discourse.
*   **Real-time Analytics**: Tracks strategy effectiveness, win/loss ratios, and sentiment trends.

### Why is this "Agentic AI"?
This project goes beyond standard LLM chatbots. It implements true **Agentic Architecture**:
1.  **Autonomy**: Agents operate in a self-directed loop (`Perceive` -> `Think` -> `Act`), managing their own state over 12+ turns without human intervention.
2.  **Tool Use**: Agents actively invoke external tools (Retrieval/RAG) to fetch evidence and ground their claims.
3.  **Adaptive Decision Making**: Unlike static prompts, these agents use **Reinforcement Learning (Bandits)** to *dynamically choose* the best strategy (e.g., Logical vs. Emotional) based on real-time feedback.
4.  **Goal-Oriented Behavior**: Each agent has a specific objective (Win Debate vs. Reduce Heat) and optimizes its actions to achieve it.

## Core Technical Competencies & Implementation

This project demonstrates mastery in the following areas of AI and Software Engineering:

### Core AI & Machine Learning
*   **Large Language Models (LLMs)**
    *   **Local Inference**: Implementation of 7B models (Zephyr, Mistral) on consumer hardware using 4-bit (NF4) and 8-bit quantization.
    *   **Prompt Engineering**: Dynamic system prompting with context injection, persona adoption (Pro/Anti/Mediator), and sliding context windows.
    *   **Backends**: Integration with **HuggingFace Transformers** and **llama.cpp** (GGUF format) for optimized CPU/GPU execution.

*   **Retrieval Augmented Generation (RAG)**
    *   **Hybrid Search**: Combining **Semantic Search** (Dense Vector Embeddings) with **Keyword Search** (BM25) for high-precision retrieval.
    *   **Vector Database**: High-performance similarity search implementation using **FAISS** (Facebook AI Similarity Search).
    *   **Data Processing**: Advanced document chunking strategies with overlap preservation and `sentence-transformers` tokenization.

*   **Reinforcement Learning (RL)**
    *   **Contextual Bandits**: Implementation of Epsilon-Greedy algorithms for adaptive strategy selection.
    *   **Reward Modeling**: Feedback loops that optimize agent behavior based on debate outcomes and heat reduction.

### Software Engineering & Architecture
*   **Design Patterns**: Agentic workflow design, Orchestrator pattern for state management, and modular architecture.
*   **Python Mastery**: Extensive use of Type Hinting, Dataclasses, Decorators, and Enums for robust codebases.
*   **Algorithms**: Custom Rule-Based NLP for deterministic emotion and heat detection using Regex patterns.

### Libraries & Tools
*   **AI/ML**: `transformers`, `sentence-transformers`, `accelerate`, `bitsandbytes`, `llama-cpp-python`
*   **Data/Math**: `numpy`, `faiss-cpu`, `rank_bm25`
*   **Infrastructure**: `typer` (CLI), `rich` (Terminal UI), `pyyaml` (Configuration)

---

## System Architecture

The system follows an **Orchestrator Pattern**, where a central loop manages agent interactions, retrieval, and state transitions.

### 1. Execution Flow

The debate cycle follows a strict turn-based loop (`runtime/loop.py`):

1.  **Initialize**: Load LLM (Vicuna/Zephyr), build RAG indices (BM25+FAISS), and spawn Agents.
2.  **Retrieve (RAG)**: The `HybridRetriever` fetches relevant chunks (BM25 + Semantic Search) for the topic.
3.  **Think (Bandit)**: The active Agent's `ContextualBandit` selects a strategy (e.g., "Logical" vs "Aggressive") using Epsilon-Greedy logic.
4.  **Generate (LLM)**: The Agent generates an argument. **Dynamic System Prompts** are injected to reflect the current "Heat" level.
5.  **Monitor (Heat)**: The `HeatDetector` scans the response for keywords/patterns.
6.  **Intervene**: If Heat > 0.8, the `Mediator` inserts a cooling message.
7.  **Learn**: The Agent receives a reward (Win/Cooling) and updates its strategy weights.

### 2. System Modules & Detailed Component Analysis

#### A. The Agents (`agents/`)
*   **Debater (`debater.py`)**: Uses **Contextual Bandits** to adaptively switch strategies. It perceives "Heat" and can be "cooled down" by the Mediator.
*   **Mediator (`mediator.py`)**: The referee. Triggers interventions if `Heat > 0.8` or if there is a rapid escalation trend.
*   **Bandit (`bandit.py`)**: Implements Reinforcement Learning logic to optimize strategy selection over time.

#### B. Heat Detection (`agents/heat.py`)
A fast, **Deterministic (Rule-Based)** system:
*   **Regex Patterns**: Scans for weighted aggressive keywords (e.g., "liar", "stupid").
*   **Intensity Check**: Analyzes CAPS LOCK ratio and punctuation (!!!).
*   **Output**: Returns a normalized `HeatScore` (0.0 - 1.0).

#### C. RAG System (`rag/`)
A **Hybrid Retrieval** pipeline ensuring arguments are factual:
*   **BM25**: Keyword search for exact terminology matches.
*   **FAISS**: Semantic search (Dense Vectors) for conceptual matches.
*   **Fusion**: Results are merged (70% BM25, 30% FAISS) and top chunks are injected into the context.

### 3. Directory Structure
```
Agentic_AI_Debate_System/
├── main.py                  # Entry point with CLI interface
├── config.yaml             # System configuration
├── requirements.txt        # Dependencies
├── runtime/
│   ├── loop.py             # Main orchestration loop
│   └── utils.py            # Utilities and configuration
├── agents/
│   ├── llm_wrapper.py      # LLM integration (Transformers/Llama.cpp)
│   ├── debater.py          # Pro/Anti debate agents
│   ├── mediator.py         # Heat monitoring and intervention
│   └── heat.py             # Emotion/sentiment analysis
│   └── bandit.py           # Adaptive strategy learning
├── rag/
│   ├── chunker.py          # Document chunking
│   ├── bm25_index.py       # Keyword-based retrieval
│   └── embeddings.py       # Semantic search with FAISS
├── data/
│   ├── source_docs/        # Input documents for grounding
│   ├── embeddings/         # Stored FAISS indices
│   └── logs/              # Debate transcripts and analytics
├── tools/                  # Utility scripts
└── venv/                   # Virtual environment (optional)
```

## Installation

### Prerequisites & Requirements

This project relies on the following key libraries (full list in `requirements.txt`):

*   **Core Systems**: `python >= 3.10`
*   **Machine Learning**: `torch`, `transformers`, `accelerate`, `bitsandbytes` (for quantization)
*   **LLM Inference**: `llama-cpp-python` (for GGUF models)
*   **RAG & Search**: `faiss-cpu` (Vector DB), `sentence-transformers` (Embeddings), `rank-bm25` (Keyword Search)
*   **Data Science**: `numpy`, `pandas`, `scikit-learn`
*   **Interface**: `typer` (CLI), `rich` (UI), `pyyaml`, `pydantic`, `orjson`, `requests`

> **Note**: For GPU acceleration, ensure you have the appropriate CUDA toolkit installed for PyTorch.

### Recommended System Specifications
*   **RAM**: 16GB+ (to run 7B models in 8-bit/4-bit)
*   **Storage**: ~10GB for models and vector indices
*   **GPU**: NVIDIA GPU (6GB+ VRAM) recommended but not required (runs on CPU via llama.cpp).

### Setup
```bash
# Clone the repository
cd Agentic_AI_Debate_System

# Install dependencies
pip install -r requirements.txt

# Initialize the system (creates sample docs)
python main.py setup --create-sample-docs
```

## Configuration

The system is highly configurable via `config.yaml`.

### Model Options
You can choose between HuggingFace Transformers (GPU) or Llama.cpp (CPU/Apple Silicon).

#### Option 1: HuggingFace (Default)
```yaml
model:
  name: "lmsys/vicuna-7b-v1.5"
  device: "auto"
  load_in_8bit: true
  backend: "transformers"
```

#### Option 2: Llama.cpp (GGUF)
```yaml
model:
  backend: "llama.cpp"
  gguf_model_path: "path/to/zephyr-7b.gguf"
  gguf_n_ctx: 4096
```

### RAG & Debate Settings
```yaml
rag:
  use_embeddings: true
  bm25_weight: 0.7  # Balance keyword vs semantic search

debate:
  max_turns: 12
  heat_threshold: 0.8
  mediator_intervention_probability: 0.1
```

## Usage

### Run a Standard Debate
```bash
python main.py debate --topic "The impact of AI on employment"
```

### Run with Custom Configuration
```bash
python main.py debate --config custom_config.yaml --turns 20
```

### Interactive Mode
```bash
python main.py interactive
```

## Research Applications

This system is suitable for:
*   **AI Safety Research**: Analyzes how agents behave under stress and high-heat conditions.
*   **Discourse Analysis**: Studies persuasion techniques and argument structure.
*   **Educational Tools**: Demonstrates the mechanics of RAG and Reinforcement Learning in a tangible application.

## License

This project is licensed under the MIT License.
