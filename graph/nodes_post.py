from state import AgentState
from memory import memory
import os
from datetime import datetime
from config import Config
from prompts.report_templates import OUTPUT_WRAPPER

def format_output(state: AgentState):
    """
    Formatting final output.
    """
    report = state.get("final_report", "No report generated.")
    sources = [d.get("source") for d in state.get("research_data", [])]
    
    formatted = OUTPUT_WRAPPER.format(
        report=report,
        sources=set(sources),
        confidence_score=state.get("confidence_score", 0.0),
        mode=state.get("mode", "unknown"),
        token_usage=state.get("token_usage", 0)
    )
   
    # Save interaction to memory
    memory.add_memory(
        text=f"Query: {state['query']}\nResponse: {report}",
        metadata={"confidence": state.get("confidence_score", 0.0)}
    )
    
    # Save to Markdown file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_report_{timestamp}.md"
    filepath = os.path.join(Config.OUTPUT_DIR, filename)
    
    with open(filepath, "w") as f:
        f.write(formatted)
        
    print(f"✅ Report saved to: {filepath}")

    return {"final_report": formatted}