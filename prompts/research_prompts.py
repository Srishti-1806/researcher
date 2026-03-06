# prompts/research_prompts.py
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Gap Analysis (Deep Mode Loop)
GAP_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""
You are a technical analyst evaluating research progress.
Current Research Findings:
{research_data}

Task:
1. Identify missing technical details required to answer the query: "{query}"
2. Detect any contradictions between different data sources.
3. Assign a Confidence Score (0.0 to 1.0) based on source agreement and detail depth.

Format your response as a JSON object with keys: "gaps" (list), "contradictions" (list), "confidence_score" (float).
""")

# Step 2: Synthesis (The "Writer")
RESEARCH_SYNTHESIS_PROMPT = ChatPromptTemplate.from_template("""
You are a world-class Technical Documentation Engineer. 
Synthesize the following research data into a production-grade report.

Research Context:
{context}

Original Query: {query}

Instructions:
- Use professional Markdown.
- Ensure every claim is linked to the research findings (Evidence Trace).
- Highlight risks and performance trade-offs.
- Avoid fluff; optimize for senior developer readability.
""")