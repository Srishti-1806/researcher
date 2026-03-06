# prompts/report_templates.py

# Template for the final output formatting (wrapping the generated report)
OUTPUT_WRAPPER = """
# Final Response

{report}

---
**Sources:** {sources}
**Confidence:** {confidence_score}
**Mode:** {mode}
**Token Usage:** {token_usage} tokens
"""

# Template for the LLM to structure its report (used in synthesis if needed)
REPORT_STRUCTURE_INSTRUCTION = """
Follow this structure for the report:
1. Executive Summary
2. Technical Deep Analysis
3. Key Findings & Trade-offs
4. Evidence Trace (Citations)
"""