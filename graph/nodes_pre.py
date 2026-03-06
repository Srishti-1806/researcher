from state import AgentState
from config import Config
from memory import memory
from utils.groq_client import get_llm
import uuid
import sys

# Initialize a simple Groq client wrapper
# Acquire LLM (Groq if configured, otherwise DummyLLM fallback)
llm = get_llm()
print(f"✅ LLM initialized: {getattr(llm, 'model', 'unknown')}")

def guard_layer(state: AgentState):
    """
    Guard Budget and Token and telemetry.
    Initializes the state if needed.
    """
    return {
        "token_usage": 0, 
        "budget_limit": 5000, 
        "research_data": [], 
        "gaps": [], 
        "iterations": 0,
        "history": state.get("history", []),
        "query_id": str(uuid.uuid4())  # Generate unique ID for streaming
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
    """
    Intent classifier clarification Orchestrator.
    Determines if the query is clear and what the intent is.
    """
    if not llm:
        print("⚠️ Error: LLM (Groq) not available.")
        return {
            "intent": "Error",
            "is_clarified": False,
            "token_usage": 0
        }

    prompt_text = (
        "Classify the following query into one of these categories: Research, Bug Fix, Architecture, General Question. "
        "Also determine if the query is clear (True/False). Return the output as 'Category: <category>, Clear: <True/False>'.\n\n"
        f"Query: {state['query']}"
    )

    try:
        response = llm.generate(prompt_text)
    except Exception as e:
        print(f"⚠️ Error calling Groq API: {e}")
        return {
            "intent": "Error",
            "is_clarified": False,
            "token_usage": 0
        }

    # Track tokens (estimate)
    tokens_used = len(str(response).split()) * 2

    content = str(response).strip()
    print(f"DEBUG intent_classifier - LLM raw response: '{content}'")
    
    # Simple parsing
    intent = "Research"
    is_clarified = True
    
    print(f"DEBUG intent_classifier - Checking for Clear: False or false in content...")
    if "Clear: False" in content or "false" in content.lower():
        print(f"DEBUG intent_classifier - Found Clear: False or false, setting is_clarified to False")
        is_clarified = False
    else:
        print(f"DEBUG intent_classifier - Did not find Clear: False or false, keeping is_clarified as True")
    
    # Extract intent
    if "Bug Fix" in content: intent = "Bug Fix"
    elif "Architecture" in content: intent = "Architecture"
    elif "General Question" in content: intent = "General Question"
    
    result = {
        "intent": intent, 
        "is_clarified": is_clarified,
        "token_usage": state.get("token_usage", 0) + tokens_used
    }
    print(f"DEBUG intent_classifier returning: {result}")
    return result


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