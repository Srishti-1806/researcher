# 🔬 Developer Research AI Agent with Chemical Translation

A sophisticated AI-powered research assistant that combines **deep technical research capabilities** with **chemical molecule translation** between SMILES and IUPAC formats. Built with LangGraph for workflow orchestration, Groq LLM for generation, Qdrant for vector memory, and RDKit for chemical validation.

## 🎯 Key Features

### 🤖 Dual Research Modes
- **Quick Mode**: Direct LLM responses for simple queries
- **Deep Mode**: Iterative research with web search, gap analysis, and synthesis

### 🧪 Chemical Translation Engine
- **SMILES ↔ IUPAC Conversion**: Bidirectional molecule format translation
- **RDKit Validation**: Chemical structure validation and parsing
- **Confidence Scoring**: Round-trip similarity verification
- **Research Loop**: Web search for low-confidence translations

### 🧠 Intelligent Memory System
- **Qdrant Vector Database**: Persistent context and research history
- **Semantic Search**: Context-aware query enhancement
- **Conversation Threads**: Multi-session research continuity

### 🔍 Advanced Research Tools
- **Web Search Integration**: Tavily API for deep research, DuckDuckGo fallback
- **Intent Classification**: Automatic query categorization
- **Gap Analysis**: Iterative refinement of research findings
- **Structured Synthesis**: Professional markdown report generation

## 🏗️ Architecture & Data Flow

### High-Level Pipeline
```
Input Query → Guard Layer → Preprocessing → Intent Classification → Execution → Synthesis → Output
```

### Detailed Workflow

```mermaid
graph TD
<<<<<<< HEAD

Start([Input: SMILES or IUPAC]) --> Guard[Input Guard / Sanitization]
Guard --> PreProc[Chemical Preprocessor]

subgraph Knowledge Retrieval
    PreProc --> Embed[Embedding Model]
    Embed --> Vector[(Vector DB: Chemical Data / Past Queries)]
end

Vector --> Classify{Intent Classifier}

Classify -- Off topic --> End([Reject or Clarify])
Classify -- Quick query --> Tools[Search Tools Tavily or Web]
Classify -- Deep chemical query --> Deep[Deep Mode Orchestrator]

subgraph Neural Translation
    Deep --> Tokenizer[SMILES or IUPAC Tokenizer]
    Tokenizer --> Model[Transformer Translation Model]
    Model --> RawOutput[Raw Translation Output]
end

RawOutput --> Validity{RDKit Validity Check}

Validity -- Invalid --> Retry[Beam Search or Temperature Retry]
Retry --> Model

Validity -- Valid --> Confidence[Confidence Scoring Engine]

subgraph Confidence Loop
    Confidence --> Score{Confidence >= 0.8}
    Score -- No --> Iterate[Refinement Iteration]
    Iterate --> Model
end

Score -- Yes --> CrossCheck[Bi Directional Translation]

subgraph Verification
    CrossCheck --> Reverse[Translate Back]
    Reverse --> Similarity[Tanimoto or String Similarity]
end

Similarity --> Verify{Similarity >= 0.9}

subgraph Deep Research Loop
    Tools --> Research[Tavily Deep Search]
    Research --> Evidence[Evidence Aggregation]
end

Verify -- Low --> Research
Verify -- High --> Formatter

Evidence --> Formatter

Formatter[Output Formatter]

Formatter --> Metrics[Update Accuracy Benchmarks]
Formatter --> Memory[(Save to Vector DB)]
Formatter --> End([Final Molecule Output])
```
=======
    A[User Input] --> B[Guard Layer]
    B --> C[Chemical Preprocessor]
    C --> D[Query Embedding]
    D --> E[Vector Retrieval]
    E --> F[Intent Classifier]
>>>>>>> 946d395 (origin main)

    F --> G{Intent Type}
    G -->|OffTopic| H[Clarification Node]
    G -->|Quick Query| I[Quick Mode]
    G -->|Deep Chemical| J[Deep Mode Orchestrator]

    I --> K[Format Output]
    H --> L[End]

    J --> M[LLM Translation]
    M --> N[RDKit Validation]
    N --> O{Valid?}
    O -->|No| P[Temperature Retry]
    P --> M
    O -->|Yes| Q[Round-trip Check]
    Q --> R{Similarity ≥ 0.9?}
    R -->|No| S[Web Research]
    S --> T[Evidence Aggregation]
    T --> M
    R -->|Yes| U[Gap Analysis]
    U --> V{Confidence ≥ 0.8?}
    V -->|No| J
    V -->|Yes| W[Structured Synthesis]
    W --> K
    K --> X[Save to Memory]
    X --> Y[Final Output]
```


## 📂 Project Structure

```
researcher/
├── 📄 main.py                 # LangGraph workflow builder and compilation
├── 📄 config.py               # Configuration (API keys, model settings, Qdrant URL)
├── 📄 state.py                # AgentState TypedDict definition
├── 📄 memory.py               # Qdrant vector database integration
├── 📄 app.py                  # Streamlit web interface
├── 📄 ui.py                   # UI components and rendering functions
├── 📄 requirements.txt        # Python dependencies
├── 📄 run.sh                  # Startup script (Docker + Streamlit)
├── 📄 verify_setup.py         # Environment verification script
├── 📄 check_deps.py           # Dependency checker
├── 📄 .env                    # Environment variables (API keys)
│
├── 🔧 graph/                  # LangGraph workflow nodes
│   ├── 📄 nodes_pre.py        # Input preprocessing, embedding, intent classification
│   ├── 📄 nodes_exec.py       # Execution modes (quick/deep), gap analysis, synthesis
│   └── 📄 nodes_post.py       # Output formatting
│
├── 🧠 prompts/                # LLM prompt templates
│   ├── 📄 research_prompts.py # Research and analysis prompts
│   └── 📄 report_templates.py # Output formatting templates
│
├── 🔍 tools/                  # External tool integrations
│   ├── 📄 search_tools.py     # Web search (Tavily/DuckDuckGo)
│   └── 📄 memory_tools.py     # Memory operations
│
├── 🧪 testing/                # Test suites
│   ├── 📄 test_memory.py      # Memory persistence tests
│   └── 📄 test_research_agent.py # End-to-end workflow tests
│
├── 🧪 utils/                  # Utility modules
│   ├── 📄 groq_client.py      # Groq API wrapper
│   └── 📄 streaming.py        # Real-time response streaming
│
├── 📁 output/                 # Generated research reports
├── 📁 qdrant_db/              # Local Qdrant database (fallback)
└── 📁 prompts/                # Additional prompt templates
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **Docker** (for Qdrant vector database)
- **API Keys**: Groq API key (required), Tavily API key (optional)

### Installation & Setup

1. **Clone and navigate to the project**:
   ```bash
   cd researcher
   ```

2. **Run the startup script** (automatically handles everything):
   ```bash
   ./run.sh
   ```

   The script will:
   - ✅ Start Qdrant Docker container
   - 📦 Install Python dependencies
   - 📝 Create `.env` file if missing
   - 🔍 Verify setup
   - 🌐 Launch Streamlit app

3. **Access the application**:
   - **Local**: http://localhost:8503
   - **Network**: http://10.0.1.55:8503
   - **External**: http://20.192.21.51:8503

### Manual Setup (Alternative)

If you prefer manual setup:

```bash
# 1. Start Qdrant database
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env  # Edit with your API keys

# 4. Run verification
python verify_setup.py

# 5. Start the app
streamlit run app.py
```

## 🔧 Configuration

### Environment Variables (.env)
```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (improves research quality)
TAVILY_API_KEY=your_tavily_api_key_here

# Model Configuration
GROQ_MODEL_NAME=llama-3.1-8b-instant
GROQ_API_URL=https://api.groq.com/v1/models/openai/v1/chat/completions

# Database
QDRANT_URL=http://localhost:6333
```

### Model Parameters (config.py)
- **Temperature**: 0.7 (balanced creativity/accuracy)
- **Max Tokens**: 1024 per generation
- **Confidence Threshold**: 0.8 for research loops
- **Max Iterations**: 3 for deep mode

## 💡 Usage Examples

### Chemical Translation
```
Input: "Convert SMILES CC(=O)O to IUPAC"
Output: acetic acid (with validation and confidence score)
```

### Research Queries
```
Input: "How does React's virtual DOM work?"
Output: Comprehensive technical analysis with sources
```

### Deep Research Mode
```
Input: "Compare GraphQL vs REST APIs for microservices"
Output: Detailed comparison with benchmarks and trade-offs
```

## 🔄 What Was Implemented

### Phase 1: Core Infrastructure
- ✅ **LangGraph Workflow**: Multi-stage agent orchestration
- ✅ **Groq LLM Integration**: Fast, high-quality text generation
- ✅ **Qdrant Vector Database**: Semantic memory with Docker deployment
- ✅ **Streamlit Web Interface**: User-friendly research interface

### Phase 2: Chemical Translation Engine
- ✅ **SMILES/IUPAC Conversion**: Bidirectional molecule translation
- ✅ **RDKit Integration**: Chemical structure validation
- ✅ **Confidence Scoring**: Similarity-based verification
- ✅ **Research Loop**: Web search for uncertain translations
- ✅ **Temperature Retry**: Improved generation for invalid outputs

### Phase 3: Research Capabilities
- ✅ **Dual Mode Execution**: Quick responses + deep iterative research
- ✅ **Web Search Integration**: Tavily API + DuckDuckGo fallback
- ✅ **Intent Classification**: Automatic query categorization
- ✅ **Gap Analysis**: Research completeness assessment
- ✅ **Structured Synthesis**: Professional report generation

### Phase 4: Advanced Features
- ✅ **Persistent Memory**: Context retention across sessions
- ✅ **Conversation Threads**: Multi-query research continuity
- ✅ **Real-time Streaming**: Progressive response display
- ✅ **Error Handling**: Robust failure recovery
- ✅ **Testing Suite**: Automated validation

## 🛠️ Technical Stack

- **Framework**: LangGraph (workflow orchestration)
- **LLM**: Groq API (llama-3.1-8b-instant)
- **Vector DB**: Qdrant (Docker container)
- **Web Search**: Tavily API, DuckDuckGo
- **Chemical Processing**: RDKit
- **Embeddings**: FastEmbed (BAAI/bge-small-en-v1.5)
- **UI**: Streamlit
- **Language**: Python 3.10+

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Memory persistence tests
python testing/test_memory.py

# End-to-end workflow tests
python testing/test_research_agent.py

# Dependency verification
python check_deps.py
```

## 📊 Performance & Validation

- **Chemical Translation Accuracy**: RDKit validation + similarity scoring
- **Research Quality**: Multi-source evidence aggregation
- **Response Time**: ~2-5 seconds for quick mode, ~10-30 seconds for deep research
- **Memory Efficiency**: Optimized embeddings with Qdrant indexing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangGraph**: For the robust workflow orchestration framework
- **Groq**: For providing fast and reliable LLM inference
- **Qdrant**: For the efficient vector database solution
- **RDKit**: For chemical structure processing capabilities
- **Streamlit**: For the intuitive web interface framework

---

**Built with ❤️ for advancing AI-assisted research and chemical informatics**

