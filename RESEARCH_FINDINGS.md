# Research Report: Agentic AI Debate & Mediation System

## 1. Research Observations & Findings

### A. Agent Behavior & Strategy Adaptation (RL)
*   **Convergence**: Agents using Contextual Bandits tended to converge on specific strategies over long debates. If "Aggressive" prompting initially yielded higher "wins" (as perceived by the reward model), agents would spiral into aggression unless checked.
*   **Exploration vs. Exploitation**: The Epsilon-Greedy mechanism was crucial. Without sufficient exploration (random strategy selection), agents often got stuck in sub-optimal local minima (e.g., repeating the same weak arguments).

### B. Mediation Efficacy
*   **Heat Reduction**: The Mediator proved effective at "breaking the cycle". By inserting a neutral, fact-focused message, the `DebateContext` was fundamentally altered. Subsequent agent responses (even with aggressive personas) showed a marked decrease in sentiment intensity immediately following an intervention.
*   **Contextual Awareness**: The simple Rule-Based/Regex heat detection, while fast, sometimes missed sarcasm or passive-aggressive behavior, leading to missed interventions or false positives.

### C. RAG Grounding
*   **Hallucination Reduction**: The Hybrid Retrieval (BM25 + Semantic) significantly reduced hallucinations compared to raw generation. Agents were forced to incorporate specific statistics or quotes from the provided chunks.
*   **Context Window Pressure**: Heavy use of RAG evidence quickly filled the context window (4096 tokens). The `rolling_history_size` implementation was critical to prevent the debate from crashing after 4-5 turns.

## 2. Technical Limitations & Challenges

1.  **Context Limitations**: Local 7B models have limited context. Long debates lose coherence as early arguments roll out of memory.
2.  **Heat Detection nuance**: The deterministic Regex approach lacks the nuance of an LLM-based classifier. It cannot detect "cold anger" or polite condescension.
3.  **Inference Latency**: On CPU (Llama.cpp), generation times can be slow (5-10s per turn), making real-time interactive testing cumbersome compared to GPU execution.

## 3. Future Roadmap & Improvements

### Phase 1: Enhanced Intelligence
*   **LLM-as-Judge**: Replace regex-based Heat Detection with a small, specialized LLM (e.g., TinyLlama) to classify sentiment more accurately.
*   **Chain-of-Thought (CoT)**: Implement CoT prompting to let agents "plan" their argument before speaking.

### Phase 2: System Scalability
*   **Vector DB Upgrade**: Migrate from FAISS-CPU to a persistent solution like ChromaDB or Pinecone for larger document sets.
*   **Web Search Integration**: Allow agents to fetch *live* data (Serper/Tavily API) instead of relying solely on static local documents.

### Phase 3: Multi-Agent Expansion
*   **Audience Agents**: Introduce passive "Audience" agents who vote on the winner, providing a more robust Reward Model for the Reinforcement Learning component.
