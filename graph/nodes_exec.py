from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from config import Config
import os
from prompts.research_prompts import GAP_ANALYSIS_PROMPT, RESEARCH_SYNTHESIS_PROMPT
from utils.streaming import get_streaming_buffer
from utils.groq_client import get_llm

# Initialize Groq client
# Acquire LLM (Groq if configured, otherwise DummyLLM fallback)
llm = get_llm()
print(f"✅ LLM initialized: {getattr(llm, 'model', 'unknown')}")

# Custom DDG Tool to bypass langchain-community import issues
from duckduckgo_search import DDGS

class CustomDuckDuckGoSearch:
    def invoke(self, query):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                return str(results)
        except Exception as e:
            return f"Search error: {e}"

# Initialize Search Tools
tavily_api_key = os.getenv("TAVILY_API_KEY")
if tavily_api_key:
    search_tool = TavilySearchResults(max_results=3)
else:
    search_tool = CustomDuckDuckGoSearch()

def planner_router(state: AgentState):
    """
    Planner and router.
    Decides between 'quick' and 'deep' mode.
    """
    if not llm:
        print("⚠️ Error: Groq LLM not available")
        return {"mode": "quick", "token_usage": 0}
    
    # Build conversation history context
    history = state.get("history", [])
    history_text = ""
    if history:
        history_text = "\n\nPrevious conversation:\n" + "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-4:]  # Last 4 messages
        ]) + "\n"
    
    prompt_text = (
        "Analyze the query complexity. If it requires simple fact checking or code snippet, choose 'quick'. "
        "If it requires extensive research, comparison, or architectural design, choose 'deep'. Return ONLY the mode: 'quick' or 'deep'.\n\n"
        f"{history_text}\nCurrent Query: {state['query']}"
    )

    try:
        response = llm.generate(prompt_text)
    except Exception as e:
        print(f"⚠️ Error in planner_router: {e}")
        return {"mode": "quick", "token_usage": 0}

    mode = str(response).strip().lower()
    tokens_used = len(str(response).split()) * 2
    
    if "deep" in mode:
        return {"mode": "deep", "token_usage": state.get("token_usage", 0) + tokens_used}
    return {"mode": "quick", "token_usage": state.get("token_usage", 0) + tokens_used}

def quick_mode_executor(state: AgentState):
    """
    Quick mode executor - generates response directly without streaming.
    """
    if not llm:
        return {
            "research_data": [{"content": "Error: LLM not available", "source": "Error"}],
            "final_report": "Error: LLM not available",
            "token_usage": 0
        }
    
    query_id = state.get("query_id", "")
    buffer = get_streaming_buffer(query_id) if query_id else None
    
    # Build conversation history for context
    history = state.get("history", [])
    context_text = ""
    
    if history:
        context_text = "Previous conversation:\n" + "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-6:]
        ]) + "\n\n"
    
    full_prompt = context_text + f"Query: {state['query']}\n\nProvide a detailed response:"
    
    try:
        full_response = llm.invoke(full_prompt)
    except Exception as e:
        print(f"⚠️ Error in quick_mode_executor: {e}")
        return {
            "research_data": [{"content": f"Error: {e}", "source": "Error"}],
            "final_report": f"Error: {e}",
            "token_usage": 0
        }
    
    # Add to buffer for real-time display
    if buffer:
        buffer.add_chunk(str(full_response))
        buffer.mark_complete()
    
    # Estimate tokens
    tokens_used = len(str(full_response).split()) * 2
    
    return {
        "research_data": [{"content": str(full_response), "source": "LLM Knowledge"}],
        "final_report": str(full_response),
        "token_usage": state.get("token_usage", 0) + tokens_used
    }

def deep_mode_orchestrator(state: AgentState):
    """
    Deep mode executor: Orchestrator -> Agent Research
    Performs search and evidence gathering.
    """
    query = state["query"]
    iteration = state.get("iterations", 0)
    gaps = state.get("gaps", [])
    history = state.get("history", [])
    
    # Build context-aware query from history
    search_query = query
    if history:
        # Get last user message for context
        last_messages = [msg['content'] for msg in history[-2:] if msg['role'] == 'user']
        if last_messages:
            context_hint = " (in context of: " + ", ".join(last_messages) + ")"
            search_query = query + context_hint
    
    # If we have gaps, refine the search query
    if gaps:
        search_query = f"{search_query} focusing on {', '.join(gaps)}"
        
    print(f"DEBUG: Executing Search for: {search_query} (Iter: {iteration})")
    
    try:
        if isinstance(search_tool, TavilySearchResults):
            # Tavily returns list of dicts
            results = search_tool.invoke({"query": search_query})
            # Format Tavily results to extract content
            if isinstance(results, list):
                formatted_content = []
                for r in results:
                    if isinstance(r, dict):
                        title = r.get('title', r.get('url', 'Untitled'))
                        content_text = r.get('content', r.get('snippet', ''))
                        url = r.get('url', '')
                        formatted_content.append(f"**{title}**\n{content_text}\nSource: {url}\n")
                content = "\n---\n".join(formatted_content) if formatted_content else str(results)
            else:
                content = str(results)
        else:
            # Custom duckduckgo returns formatted string via custom invoke
            content = search_tool.invoke(search_query)

    except Exception as e:
        content = f"Search failed: {e}"

    return {
        "research_data": state.get("research_data", []) + [{"content": content, "source": "Web Search"}],
        "iterations": iteration + 1
    }

def gap_analysis_node(state: AgentState):
    """
    Cross references and Gap analysis.
    Checks if enough information is gathered.
    """
    if not llm:
        return {
            "confidence_score": 0.5,
            "gaps": ["LLM not available"],
            "token_usage": 0
        }
    
    data = state.get("research_data", [])
    combined_content = "\n".join([d["content"] for d in data])
    history = state.get("history", [])
    
    # Build history context
    history_text = ""
    if history:
        history_text = "\nConversation history: " + "; ".join([
            f"{msg['role']}: {msg['content'][:100]}" for msg in history[-3:]
        ])
    
    # Build prompt text for gap analysis (use prompt template from prompts/research_prompts)
    gap_prompt = (
        "You are a technical analyst evaluating research progress.\n"
        f"Current Research Findings:\n{combined_content}\n\n"
        "Task:\n1. Identify missing technical details required to answer the query:\n"
        f'"{state["query"]}"\n'
        "2. Detect any contradictions between different data sources.\n"
        "3. Assign a Confidence Score (0.0 to 1.0) based on source agreement and detail depth.\n\n"
        "Format your response as a JSON object with keys: \"gaps\" (list), \"contradictions\" (list), \"confidence_score\" (float)."
    )

    try:
        response = llm.generate(gap_prompt)
    except Exception as e:
        print(f"Error in gap analysis: {e}")
        return {
            "confidence_score": 0.5,
            "gaps": ["Error in analysis"],
            "token_usage": 0
        }

    tokens_used = len(str(response).split()) * 2
    
    # Simple parsing
    try:
        import re
        response_text = str(response)
        
        # Looking for "Confidence: 0.X"
        score_match = re.search(r"Confidence:\s*([0-9.]+)", response_text)
        score = float(score_match.group(1)) if score_match else 0.5
        
        gaps = []
        if "Gaps:" in response_text or "gap" in response_text.lower():
            # Extract gaps if mentioned
            parts = response_text.split("Gaps:" if "Gaps:" in response_text else "gaps:") 
            if len(parts) > 1:
                gaps_str = parts[1].split("\n")[0].strip()
                gaps = [g.strip() for g in gaps_str.split(",")]
            
    except Exception as parse_error:
        score = 0.5
        gaps = ["Could not parse analysis"]

    return {
        "confidence_score": score, 
        "gaps": gaps,
        "token_usage": state.get("token_usage", 0) + tokens_used
    }

def structured_synthesis_node(state: AgentState):
    """
    Structure reasoning and synthesis.
    """
    if not llm:
        return {
            "final_report": "Error: LLM not available",
            "token_usage": 0
        }
    
    query_id = state.get("query_id", "")
    buffer = get_streaming_buffer(query_id) if query_id else None
    
    data = state.get("research_data", [])
    combined_content = "\n".join([d["content"] for d in data])
    history = state.get("history", [])
    
    # Build conversation context
    history_context = ""
    if history:
        history_context = "\n\nConversation context:\n" + "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-4:]
        ]) + "\n"
    
    # Build synthesis prompt
    synth_prompt = (
        "You are a world-class Technical Documentation Engineer.\n"
        "Synthesize the following research data into a production-grade report.\n\n"
        f"Research Context:\n{combined_content}\n\n"
        f"Original Query: {state['query']}\n\n"
        "Instructions:\n- Use professional Markdown.\n- Ensure every claim is linked to the research findings (Evidence Trace).\n- Highlight risks and performance trade-offs.\n- Avoid fluff; optimize for senior developer readability.\n"
    )

    try:
        full_response = llm.generate(synth_prompt)
    except Exception as e:
        print(f"Error in synthesis: {e}")
        return {
            "final_report": f"Error during synthesis: {e}",
            "token_usage": 0
        }
    
    # Add to buffer for real-time display
    if buffer:
        buffer.add_chunk(str(full_response))
        buffer.mark_complete()
    
    cleaned_report = str(full_response).replace("```markdown", "").replace("```", "").strip()
    tokens_used = len(cleaned_report.split()) * 2  # Estimate
    
    return {
        "final_report": cleaned_report,
        "token_usage": state.get("token_usage", 0) + tokens_used
    }