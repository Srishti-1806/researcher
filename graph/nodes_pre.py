from state import AgentState
from config import Config
from memory import memory
from utils.groq_client import get_llm
import uuid
import sys
import re

# Initialize a simple Groq client wrapper
# Acquire LLM (Groq if configured, otherwise DummyLLM fallback)
llm = get_llm()
print(f"✅ LLM initialized: {getattr(llm, 'model', 'unknown')}")

def guard_layer(state: AgentState):
    """Input Guard / Sanitization and state initialization."""
    return {
        "token_usage": 0,
        "budget_limit": Config.MAX_TOKENS_PER_QUERY,
        "research_data": [],
        "gaps": [],
        "iterations": 0,
        "history": state.get("history", []),
        "query_id": str(uuid.uuid4()),  # Generate unique ID for streaming
    }


def chemical_preprocessor(state: AgentState):
    """Sanitize the query and detect whether it is a SMILES or IUPAC string."""
    raw = state.get("query", "") or ""
    sanitized = " ".join(raw.strip().split())

    # Guess input format using simple heuristics
    input_format = "IUPAC"
    if "smiles" in sanitized.lower() or any(c in sanitized for c in "=#@/\\[]"):
        input_format = "SMILES"

    # Allow explicit labels like "SMILES: ..." or "IUPAC: ..."
    match = re.search(r"(?:SMILES|smiles|IUPAC|iupac)\s*[:\-]?\s*(['\"]?)([^'\"]+)\1", sanitized)
    if match:
        sanitized = match.group(2).strip()

    # Default output format is the opposite
    output_format = "IUPAC" if input_format == "SMILES" else "SMILES"

    return {
        "sanitized_query": sanitized,
        "input_format": input_format,
        "output_format": output_format,
        "input_value": sanitized,
    }


def embed_query(state: AgentState):
    """Generate an embedding for the sanitized query for later vector DB retrieval."""
    query = state.get("sanitized_query", state.get("query", ""))
    try:
        # Use the same embedding model as MemoryManager for consistency
        embeddings = list(memory.embedding_model.embed([query]))
        query_embedding = embeddings[0] if embeddings else None
    except Exception as e:
        print(f"⚠️ Error embedding query: {e}")
        query_embedding = None

    return {"query_embedding": query_embedding}


def vector_retrieval(state: AgentState):
    """Retrieve context from the vector DB (past queries, chemical data)."""
    query = state.get("sanitized_query", state.get("query", ""))
    context_docs = memory.get_context(query)
    formatted_context = "\n".join(context_docs) if context_docs else "No prior context found."

    return {
        "context": [formatted_context],
        "vector_hits": context_docs,
    }

def context_retrieval(state: AgentState):
    """
    Memory and context retrieval vector DB.
    """
    query = state["query"]
    print(f"DEBUG: Retrieving context for: {query}")
    
    context_docs = memory.get_context(query)
    formatted_context = "\n".join(context_docs) if context_docs else "No prior context found."
    
    return {"context": [formatted_context]}

def intent_classifier(state: AgentState):
    """Classify the intent of the query.

    Categories:
    - OffTopic: Not a chemical / molecule conversion query.
    - Quick_Query: Simple factual question (use quick web search).
    - Deep_Chemical_Query: Requires SMILES/IUPAC translation / deep validation.
    """
    if not llm:
        print("⚠️ Error: LLM (Groq) not available.")
        return {
            "intent": "OffTopic",
            "is_clarified": False,
            "token_usage": 0
        }

    query = state.get("sanitized_query", state.get("query", ""))
    prompt_text = (
        "Classify the following query into one of: OffTopic, Quick_Query, Deep_Chemical_Query.\n"
        "- OffTopic: unrelated to chemical structure conversion or chemistry knowledge.\n"
        "- Quick_Query: a quick factual question answerable with a web search.\n"
        "- Deep_Chemical_Query: requires SMILES/IUPAC translation or chemistry reasoning.\n"
        "Return ONLY: Intent: <category>\n\n"
        f"Query: {query}"
    )

    try:
        response = llm.generate(prompt_text)
    except Exception as e:
        print(f"⚠️ Error calling Groq API: {e}")
        return {
            "intent": "OffTopic",
            "is_clarified": False,
            "token_usage": 0
        }

    tokens_used = len(str(response).split()) * 2
    content = str(response).strip()
    intent = "OffTopic"

    if "Deep_Chemical_Query" in content or "deep" in content.lower():
        intent = "Deep_Chemical_Query"
    elif "Quick_Query" in content or "quick" in content.lower():
        intent = "Quick_Query"

    return {
        "intent": intent,
        "is_clarified": intent != "OffTopic",
        "token_usage": state.get("token_usage", 0) + tokens_used
    }


def clarification_node(state: AgentState):
    """
    Ask the user to rephrase when the query is not a research-related query.
    """
    message = "This tool is only meant for research related queries...."
    history = state.get("history", [])
    history.append({"role": "system", "content": message})

    return {
        "clarification_prompt": message,
        "awaiting_user_input": True,
        "history": history
    }