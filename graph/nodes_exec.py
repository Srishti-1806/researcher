try:
    from langchain_tavily import TavilySearch
    search_tool_class = TavilySearch
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults
    search_tool_class = TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from config import Config
import os
import difflib
from prompts.research_prompts import GAP_ANALYSIS_PROMPT, RESEARCH_SYNTHESIS_PROMPT
from utils.streaming import get_streaming_buffer
from utils.groq_client import get_llm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit not available - chemical validation will be limited")

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

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

if tavily_api_key and TAVILY_AVAILABLE:
    search_tool = TavilySearchResults(max_results=3)
else:
    search_tool = CustomDuckDuckGoSearch()
def planner_router(state: AgentState):
    """
    Planner and router.
    Decides between 'quick' and 'deep' mode.
    """

    if not llm:
        return {"mode": "quick", "token_usage": 0}

    # ✅ define query FIRST
    query = state.get("query", "")

    history = state.get("history", [])
    history_text = ""

    if history:
        history_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in history[-4:]
        ])

    # ✅ Hard rule for chemical queries
    if is_chemical_query(query):
        return {
            "mode": "deep",
            "token_usage": state.get("token_usage", 0)
        }

    prompt_text = f"""
You are a routing system for an AI agent.

Decide if the query requires:
- QUICK mode → simple question, small explanation, code snippet.
- DEEP mode → research, chemical translation, architecture design, multi-step reasoning.

Rules:
- SMILES or chemical queries → deep
- long technical problems → deep
- short factual answers → quick

Conversation History:
{history_text}

Current Query:
{query}

Return ONLY one word:
quick
or
deep
"""

    try:
        response = llm.generate(prompt_text)
        mode = str(response).strip().lower()
    except Exception:
        mode = "quick"
        response = ""

    tokens_used = len(str(response).split()) * 2 if response else 0

    if "deep" in mode:
        mode = "deep"
    else:
        mode = "quick"

    return {
        "mode": mode,
        "token_usage": state.get("token_usage", 0) + tokens_used
    }
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

def _validate_molecule(smiles: str) -> tuple[bool, str]:
    if not RDKIT_AVAILABLE:
        return True, "Skipping"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES structure"
        
        # catches extra double bonds, unclosed rings, etc.
        Chem.SanitizeMol(mol) 
        
        return True, "Valid"
    except Exception as e:
        # if valence error occurs (e.g. Carbon with 5 bonds), caught here
        return False, f"Valence Error: {str(e)}"


def canonicalize_smiles(smiles: str) -> str:
    """Convert SMILES to canonical form using RDKit."""
    if not RDKIT_AVAILABLE:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
    except Exception:
        return None

def compute_similarity(smiles1: str, smiles2: str) -> float:
    """
    Compute molecular similarity
    """

    if not smiles1 or not smiles2:
        return 0.0

    if not RDKIT_AVAILABLE:
        return difflib.SequenceMatcher(a=smiles1, b=smiles2).ratio()

    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 and mol2:
            fp1 = Chem.RDKFingerprint(mol1)
            fp2 = Chem.RDKFingerprint(mol2)

            return DataStructs.FingerprintSimilarity(fp1, fp2)

    except Exception:
        pass

    return difflib.SequenceMatcher(a=smiles1, b=smiles2).ratio()

def clean_smiles_string(smiles: str) -> str:
    import re

    if not smiles:
        return ""

    smiles = smiles.strip()

    # remove spaces
    smiles = re.sub(r"\s+", "", smiles)

    # remove empty brackets
    smiles = smiles.replace("()", "")

    # collapse double parentheses
    while "((" in smiles:
        smiles = smiles.replace("((", "(")

    while "))" in smiles:
        smiles = smiles.replace("))", ")")

    return smiles

def validate_iupac_name(name: str) -> bool:
    """
    Basic IUPAC validation
    """

    if not name:
        return False

    name = name.strip()

    if len(name) < 3:
        return False

    import re

    pattern = r'^[A-Za-z0-9,\-\(\)\s]+$'

    if not re.match(pattern, name):
        return False

    return True

def is_chemical_query(query: str) -> bool:

    keywords = [
        "smiles",
        "iupac",
        "molecule",
        "chemical",
        "pubchem",
        "rdkit"
    ]

    q = query.lower()

    if any(k in q for k in keywords):
        return True

    if any(c.isdigit() for c in query) and "(" in query:
        return True

    return False

def rule_based_fallback(smiles: str) -> str:
    """Apply rule-based corrections for common SMILES issues."""
    if not RDKIT_AVAILABLE:
        return smiles

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Try to standardize the molecule
            AllChem.Compute2DCoords(mol)
            return Chem.MolToSmiles(mol, canonical=True)
        return smiles
    except Exception:
        return smiles

def check_carbon_count(iupac: str, smiles: str) -> bool:
    if not RDKIT_AVAILABLE: return True
    
    # Simple mapping
    counts = {"meth": 1, "eth": 2, "prop": 3, "but": 4, "pent": 5, "hex": 6}
    expected = 0
    for key, val in counts.items():
        if key in iupac.lower():
            # Basic logic: finds the largest prefix
            expected = max(expected, val)
            
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        actual = len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'])
        return actual == expected # Or logic to handle substituents
    return False

def deep_mode_orchestrator(state: AgentState):
    """Enhanced deep mode executor with canonical SMILES, confidence scoring, and advanced validation."""
    query = state.get("query", "")
    if not llm:
        return {
            "research_data": [{"content": "Error: LLM not available", "source": "Error"}],
            "final_report": "Error: LLM not available",
            "confidence_score": 0.0,
            "iterations": state.get("iterations", 0),
            "token_usage": 0,
        }

    # Gather inputs from preprocessing
    input_value = state.get("input_value", state.get("query", ""))
    input_format = state.get("input_format", "SMILES")
    output_format = state.get("output_format", "IUPAC")
    iteration = state.get("iterations", 0) + 1

    # Step 1: Canonicalize input SMILES
    original_input = input_value
    if input_format.upper() == "SMILES":
        canonical_input = canonicalize_smiles(input_value)
        if canonical_input:
            input_value = canonical_input
            print(f"🔄 Canonicalized input: {original_input} → {input_value}")

    # Initialize variables for the retry loop
    max_iterations = 3
    best_output = None
    best_confidence = 0.0
    best_similarity = 0.0
    research_data = state.get("research_data", [])

    for attempt in range(max_iterations):
        print(f"🔄 Translation attempt {attempt + 1}/{max_iterations}")

        # Step 2: Generate translation with adaptive temperature
        temperature = min(1.0, Config.TEMPERATURE + (attempt * 0.1))

        if output_format.upper() == "SMILES":
            # Specific rules for SMILES generation
            translate_prompt = (
                f"You are a chemistry expert. Convert the following IUPAC name to SMILES notation.\n\n"
                f"Input IUPAC: {input_value}\n\n"
                "SMILES Rules:\n"
                "- Output ONLY the SMILES string, no explanations or extra text\n"
                "- Use standard SMILES syntax: C, N, O, S, F, Cl, Br, I for atoms\n"
                "- Use lowercase for aromatic atoms (c, n, o, s)\n"
                "- Use = for double bonds, # for triple bonds\n"
                "- Use ( ) for branches\n"
                "- Use numbers for ring closures when needed\n"
                "- NEVER include explicit hydrogens (H)\n"
                "- Keep it simple and canonical\n"
                "- Examples:\n"
                "  ethane → CC\n"
                "  ethanol → CCO\n"
                "  2-methylpropane → CC(C)C\n"
                "  2,3-dimethylbutane → CC(C)C(C)C\n"
                "  propane → CCC\n"
                "  2-methylbutane → CCC(C)C\n"
                "  benzene → c1ccccc1\n"
                f"Output SMILES:"
            )
        else:
            # Rules for IUPAC generation
            translate_prompt = (
                f"You are a chemistry expert. Convert the following SMILES to IUPAC name.\n\n"
                f"Input SMILES: {input_value}\n\n"
                "IUPAC Rules:\n"
                "- Output ONLY the IUPAC name, no explanations\n"
                "- Use proper systematic nomenclature\n"
                "- Number carbons from the end that gives lowest numbers\n"
                "- Use correct prefixes for substituents\n"
                "- Examples:\n"
                "  CC → ethane\n"
                "  CCO → ethanol\n"
                "  CC(C)C → 2-methylpropane\n"
                "  CC(C)C(C)C → 2,3-dimethylbutane\n"
                "NEGATIVE CONSTRAINTS:\n"
                "- DO NOT invent carbons. The total carbon count must match the IUPAC prefix (e.g., 'pent' must have exactly 5 carbons).\n"
                "- DO NOT leave empty brackets '()' or unclosed rings 'c1...'.\n"
                "- DO NOT use explicit hydrogens like '[CH3]'; use 'C' instead.\n"
                "- Ensure valence rules: Carbon must have 4 bonds, Nitrogen 3 or 5, Oxygen 2.\n"
                f"Output IUPAC:"
            )

        try:
            raw_output = llm.generate(translate_prompt, temperature=temperature, max_new_tokens=Config.MAX_NEW_TOKENS)
            raw_output = str(raw_output).strip()
            raw_output = clean_smiles_string(raw_output)
        except Exception as e:
            raw_output = f"[LLM Error] {e}"
            continue

        # Step 3: Canonicalize output if it's SMILES
        if output_format.upper() == "SMILES":
            canonical_output = canonicalize_smiles(raw_output)
            if canonical_output:
                raw_output = canonical_output

        # Step 4: Validate output
        if output_format.upper() == "SMILES":
            valid = _is_valid_smiles(raw_output)
        else:
            valid = validate_iupac_name(raw_output)

        if not valid:
            print(f"❌ Invalid output: {raw_output}")
            continue

        # Step 5: Reverse translation check
        if input_format.upper() == "SMILES":
            # Reverse: SMILES → IUPAC
            reverse_prompt = (
                f"You are a chemistry expert. Convert the following SMILES back to IUPAC name.\n\n"
                f"Input SMILES: {raw_output}\n\n"
                "IUPAC Rules:\n"
                "- Output ONLY the IUPAC name, no explanations\n"
                "- Use proper systematic nomenclature\n"
                "- Number carbons from the end that gives lowest numbers\n"
                "- Identify the longest carbon chain as parent\n"
                "- Number substituents with lowest possible numbers\n"
                "- Examples:\n"
                "  CC → ethane\n"
                "  CCO → ethanol\n"
                "  CC(C)C → 2-methylpropane\n"
                "  CC(C)C(C)C → 2,3-dimethylbutane\n"
                "  CCC → propane\n"
                f"Output IUPAC:"
            )
        else:
            # Reverse: IUPAC → SMILES
            reverse_prompt = (
                f"You are a chemistry expert. Convert the following IUPAC name back to SMILES.\n\n"
                f"Input IUPAC: {raw_output}\n\n"
                "SMILES Rules:\n"
                "- Output ONLY the SMILES string, no explanations\n"
                "- Use standard SMILES syntax: C, N, O, S, F, Cl, Br, I\n"
                "- Use ( ) for branches\n"
                "- NEVER include explicit hydrogens\n"
                "- Keep it simple and canonical\n"
                "- Examples:\n"
                "  ethane → CC\n"
                "  ethanol → CCO\n"
                "  2-methylpropane → CC(C)C\n"
                f"Output SMILES:"
            )

        try:
            reverse_output = llm.generate(reverse_prompt, temperature=temperature, max_new_tokens=Config.MAX_NEW_TOKENS)
            reverse_output = str(reverse_output).strip()
        except Exception as e:
            # Better fallback: use original input for reverse validation
            if input_format.upper() == "SMILES" and output_format.upper() == "IUPAC":
                reverse_output = input_value  # Original SMILES for comparison
            elif input_format.upper() == "IUPAC" and output_format.upper() == "SMILES":
                reverse_output = raw_output  # Generated SMILES as fallback
            else:
                reverse_output = raw_output

        # Canonicalize reverse output if SMILES
        if input_format.upper() == "SMILES":
            canonical_reverse = canonicalize_smiles(reverse_output)
            if canonical_reverse:
                reverse_output = canonical_reverse

        # Step 6: Compute similarity (round-trip accuracy)
        if input_format.upper() == "SMILES" and output_format.upper() == "IUPAC":
            # SMILES → IUPAC: reverse should be SMILES, compare with original input
            if _is_valid_smiles(reverse_output):
                similarity = compute_similarity(input_value, reverse_output)
            else:
                similarity = 0.0
        elif input_format.upper() == "IUPAC" and output_format.upper() == "SMILES":
            # IUPAC → SMILES: Check SMILES validity and basic structure
            if valid and _is_valid_smiles(raw_output):
                # For IUPAC → SMILES, we give higher confidence if SMILES is valid
                # and the reverse translation contains key parts of the original IUPAC name
                reverse_clean = reverse_output.lower().strip()
                input_clean = input_value.lower().strip()
                # Check if key components are present in reverse translation
                key_parts = input_clean.replace(',', '').replace('-', '').split()
                matches = sum(1 for part in key_parts if part in reverse_clean)
                similarity = matches / len(key_parts) if key_parts else 0.0
                # Boost similarity if reverse contains the core name
                if any(word in reverse_clean for word in ['butane', 'propane', 'ethane', 'methane']):
                    similarity = min(1.0, similarity + 0.3)
            else:
                similarity = 0.0
        elif input_format.upper() == "SMILES" and output_format.upper() == "SMILES":
            # SMILES → SMILES: direct comparison
            similarity = compute_similarity(input_value, raw_output)
        else:
            # IUPAC → IUPAC: string similarity
            similarity = difflib.SequenceMatcher(a=input_value.lower().strip(), b=raw_output.lower().strip()).ratio()

        # Step 7: Compute confidence score
        if input_format.upper() == "IUPAC" and output_format.upper() == "SMILES":
            # For IUPAC → SMILES, base confidence on SMILES validity and structure
            base_confidence = 0.8 if valid else 0.0  # High base confidence for valid SMILES
        else:
            base_confidence = similarity if valid else 0.0

        # Additional validation bonuses
        confidence_score = base_confidence

        # Bonus for valid molecules with high round-trip similarity
        if valid and similarity >= 0.95:
            confidence_score = min(1.0, confidence_score + 0.05)  # Perfect round-trip
        elif valid and similarity >= 0.85:
            confidence_score = min(1.0, confidence_score + 0.02)  # Good round-trip

        # Penalty for low similarity
        if similarity < 0.7:
            confidence_score = max(0.0, confidence_score - 0.2)

        # Additional validation for SMILES output
        if output_format.upper() == "SMILES" and valid:
            # Check if SMILES is canonical and simple
            try:
                if RDKIT_AVAILABLE:
                    mol = Chem.MolFromSmiles(raw_output)
                    if mol:
                        canonical = Chem.MolToSmiles(mol, canonical=True)
                        # Bonus if already canonical
                        if canonical == raw_output:
                            confidence_score = min(1.0, confidence_score + 0.05)
                        # Penalty for overcomplicated SMILES
                        if len(raw_output) > len(canonical) * 1.5:
                            confidence_score = max(0.0, confidence_score - 0.1)
            except:
                pass

        print(f"📊 Attempt {attempt + 1}: confidence={confidence_score:.3f}, similarity={similarity:.3f}, valid={valid}")

        # Keep track of best result
        if confidence_score > best_confidence:
            best_output = raw_output
            best_confidence = confidence_score
            best_similarity = similarity

        # If we have a very good result, break early
        if confidence_score >= 0.9 and similarity >= 0.9:
            break

    # Step 8: Deep research if confidence is still low
    if best_confidence < 0.8 and iteration <= Config.MAX_ITERATIONS_DEEP_MODE:
        print("🔍 Performing deep research for low confidence result")

        if input_format.upper() == "SMILES":
            research_query = f"Convert SMILES {input_value} to IUPAC name PubChem chemical nomenclature"
        else:
            research_query = f"Convert IUPAC {input_value} to SMILES notation PubChem"

        try:
            if hasattr(search_tool, 'invoke'):
                if "TavilySearchResults" in str(type(search_tool)):
                    search_results = search_tool.invoke({"query": research_query})
                    content = str(search_results)
                else:
                    content = search_tool.invoke(research_query)
            else:
                content = "Search tool not available"

            research_data.append({"content": content, "source": f"Deep Research (iteration {iteration})"})

            # Try to extract correct answer from PubChem/ChEMBL results
            import re
            if input_format.upper() == "SMILES" and "PubChem" in content:
                # Look for IUPAC patterns
                iupac_patterns = [
                    r'IUPAC.*?name.*?[:;]\s*([A-Za-z0-9\-,\(\)\s]+)',
                    r'Systematic.*?name.*?[:;]\s*([A-Za-z0-9\-,\(\)\s]+)',
                    r'([A-Za-z][a-z]+(?:\s+[a-z]+)*\d*(?:-\d+)*(?:\([^)]+\))*)'
                ]
                for pattern in iupac_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        candidate = match.group(1).strip()
                        if validate_iupac_name(candidate):
                            best_output = candidate
                            best_confidence = 0.85  # High confidence from verified source
                            print(f"✅ Found verified IUPAC from research: {candidate}")
                            break

        except Exception as e:
            research_data.append({"content": f"Research failed: {e}", "source": "Research Error"})

    # Step 9: Rule-based fallback for very complex molecules
    if best_confidence < 0.6 and input_format.upper() == "SMILES":
        print("🔧 Applying rule-based fallback")
        fallback_result = rule_based_fallback(input_value)
        if fallback_result != input_value:
            best_output = fallback_result
            best_confidence = 0.7  # Moderate confidence from rules

    # Prepare final results
    gaps = []
    if best_confidence < 0.8:
        gaps.append("Low confidence in translation")
    if best_similarity < 0.8:
        gaps.append("Low round-trip similarity")

    return {
        "research_data": research_data,
        "final_translation": best_output or "Translation failed",
        "reverse_translation": reverse_output if 'reverse_output' in locals() else "",
        "similarity_score": best_similarity,
        "confidence_score": best_confidence,
        "iterations": iteration,
        "gaps": gaps,
        "token_usage": state.get("token_usage", 0) + len(str(best_output).split()) * 2,
    }


def _is_valid_smiles(smiles: str) -> tuple[bool, str]:
    """
    Validate SMILES using RDKit
    """
    if not RDKIT_AVAILABLE:
        return True, "RDKit not available"

    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return False, "Invalid SMILES"

        Chem.SanitizeMol(mol)

        return True, "Valid"

    except Exception as e:
        return False, f"Validation error: {e}"

def gap_analysis_node(state: AgentState):
    """Cross references and Gap analysis.

    Uses the confidence score computed by the deep translation step when available.
    """
    # If we already have a computed confidence score from translation, reuse it.
    if "confidence_score" in state:
        score = state.get("confidence_score", 0.0)
        gaps = state.get("gaps", []) or []
        if not isinstance(gaps, list):
            gaps = [str(gaps)]
        return {
            "confidence_score": score,
            "gaps": gaps,
            "token_usage": state.get("token_usage", 0)
        }

    # Fallback: use LLM to assess the quality of web research.
    if not llm:
        return {
            "confidence_score": 0.5,
            "gaps": ["LLM not available"],
            "token_usage": 0
        }

    data = state.get("research_data", [])
    combined_content = "\n".join([d.get("content", "") for d in data])
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
        f'"{state.get("query", "")}"\n'
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
            parts = response_text.split("Gaps:" if "Gaps:" in response_text else "gaps:")
            if len(parts) > 1:
                gaps_str = parts[1].split("\n")[0].strip()
                gaps = [g.strip() for g in gaps_str.split(",") if g.strip()]

    except Exception:
        score = 0.5
        gaps = ["Could not parse analysis"]

    return {
        "confidence_score": score,
        "gaps": gaps,
        "token_usage": state.get("token_usage", 0) + tokens_used
    }


def structured_synthesis_node(state: AgentState):
        """Structure reasoning and synthesis into a final formatted report."""
        # Prefer returning a direct translation if available (deep chemical mode)
        if "final_translation" in state:
            input_value = state.get("input_value", state.get("query", ""))
            input_format = state.get("input_format", "")
            output_format = state.get("output_format", "")
            translation = state.get("final_translation")
            similarity = state.get("similarity_score")
            gaps = state.get("gaps", [])

            report_lines = [
                "### ✅ Chemical Translation Result",
                f"- **Input ({input_format})**: `{input_value}`",
                f"- **Output ({output_format})**: `{translation}`",
            ]
            if similarity is not None:
                report_lines.append(f"- **Round-trip similarity**: {similarity:.2f}")
            if gaps:
                report_lines.append(f"- **Notes**: {gaps}")
            return {"final_report": "\n".join(report_lines), "token_usage": state.get("token_usage", 0)}

        # Fallback: use the existing researcher synthesis flow
        if not llm:
            return {
                "final_report": "Error: LLM not available",
                "token_usage": 0
            }

        query_id = state.get("query_id", "")
        buffer = get_streaming_buffer(query_id) if query_id else None

        data = state.get("research_data", [])
        combined_content = "\n".join([d.get("content", "") for d in data])
        history = state.get("history", [])

        # Build conversation context
        history_context = ""
        if history:
            history_context = "\n\nConversation context:\n" + "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-4:]
            ]) + "\n\n"

        # Build synthesis prompt
        synth_prompt = (
            "You are a world-class Technical Documentation Engineer.\n"
            "Synthesize the following research data into a production-grade report.\n"
            f"Research Context:\n{combined_content}\n"
            f"Original Query: {state.get('query', '')}\n"
            "Instructions:\n"
            "- Use professional Markdown.\n"
            "- Ensure every claim is linked to the research findings (Evidence Trace).\n"
            "- Highlight risks and performance trade-offs.\n"
            "- Avoid fluff; optimize for senior developer readability.\n"
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
