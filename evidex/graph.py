"""
LangGraph workflow for the Evidex Q&A system.

This module provides a LangGraph-based workflow that wraps the existing
explain_question logic. The behavior is IDENTICAL to the original function.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from evidex.models import Document, Paragraph, Equation, Entities
from evidex.llm import LLMInterface, LLMResponse, parse_llm_response
from evidex.qa import build_context_block, build_prompt, build_equations_block
from evidex.entities import extract_entities_as_model


# =============================================================================
# State Definition
# =============================================================================

class QAState(TypedDict, total=False):
    """State for the Q&A workflow.
    
    Attributes:
        document: The Document being queried
        paragraph_ids: List of paragraph IDs to use as context (can be provided or planned)
        question: The user's question
        llm: The LLM interface (passed through state)
        include_debug: Whether to include debug info in final output (default: False)
        candidate_paragraph_ids: Paragraph IDs selected by planner (set by planner_node)
        planner_selected_automatically: True if planner selected paragraphs (not provided by user)
        paragraphs: Retrieved paragraphs (set by retrieve_paragraphs_node)
        equations: Retrieved equations associated with paragraphs (set by retrieve_paragraphs_node)
        llm_response: Raw LLM response (set by explain_node)
        final_response: Final structured response dict (set by explain_node)
        verification_passed: Whether the response passed verification (set by verifier_node)
        planner_reason: Explanation of planner's decision (set by planner_node)
        verifier_reason: Explanation of verifier's decision (set by verifier_node)
        linked_evidence: Evidence links based on shared entities (set by evidence_linker_node)
    """
    # Inputs
    document: Document
    paragraph_ids: list[str]  # Optional: if provided, planner is skipped
    question: str
    llm: LLMInterface
    include_debug: bool  # Optional: if True, include debug info in output
    
    # Intermediate state (set by nodes)
    candidate_paragraph_ids: list[str]  # Set by planner_node
    planner_selected_automatically: bool  # True if planner selected paragraphs
    paragraphs: list[Paragraph]
    equations: list[Equation]  # Equations associated with retrieved paragraphs
    llm_response: LLMResponse | None
    
    # Debug/introspection state (set by nodes)
    planner_reason: str  # Explanation of planner's paragraph selection
    verifier_reason: str  # Explanation of verifier's decision
    
    # Output
    final_response: dict
    verification_passed: bool
    linked_evidence: list[dict]  # Evidence links from evidence_linker_node
    composed_explanation: str | None  # Composed explanation from composer_node
    composer_verification_passed: bool  # Whether composed explanation passed verification


# =============================================================================
# Node Functions
# =============================================================================

def extract_keywords(text: str) -> set[str]:
    """Extract keywords from text for matching.
    
    Extracts individual words and important phrases, normalizes to lowercase.
    Filters out common stop words that don't carry meaning.
    
    Args:
        text: Text to extract keywords from
        
    Returns:
        Set of lowercase keywords
    """
    import re
    
    # Common stop words to filter out
    STOP_WORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and',
        'but', 'if', 'or', 'because', 'until', 'while', 'although', 'though',
        'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves', 'about', 'also',
        'paper', 'defined', 'describe', 'explain', 'discussed', 'mentioned',
    }
    
    # Extract words (letters and numbers, at least 2 chars)
    words = re.findall(r'\b[a-zA-Z0-9]{2,}\b', text.lower())
    
    # Filter stop words
    keywords = {w for w in words if w not in STOP_WORDS}
    
    return keywords


def planner_node(state: QAState) -> dict:
    """Select candidate paragraphs based on keyword matching.
    
    This node performs RESTRICTED planning:
    - ONLY selects from existing paragraph IDs in the document
    - Does NOT generate answers or summarize content
    - Does NOT invent new paragraph IDs
    - Does NOT call the LLM for reasoning
    - Is conservative: selects more paragraphs rather than fewer
    
    The planner uses keyword matching between the question and paragraph text.
    
    Args:
        state: Current workflow state with document and question
        
    Returns:
        Dict with 'candidate_paragraph_ids' key containing selected paragraph IDs
    """
    document = state["document"]
    question = state["question"]
    
    # If paragraph_ids were explicitly provided, use them directly
    if state.get("paragraph_ids"):
        provided_ids = state["paragraph_ids"]
        reason = f"Using {len(provided_ids)} explicitly provided paragraph IDs."
        return {
            "candidate_paragraph_ids": provided_ids,
            "planner_reason": reason,
            "planner_selected_automatically": False,  # User provided paragraphs
        }
    
    # Extract keywords from the question
    question_keywords = extract_keywords(question)
    
    if not question_keywords:
        # No meaningful keywords - return empty (will trigger "Not defined")
        reason = "No meaningful keywords extracted from question. Cannot match paragraphs."
        return {
            "candidate_paragraph_ids": [],
            "planner_reason": reason,
            "planner_selected_automatically": True,  # Planner made the selection
        }
    
    # Score each paragraph by keyword overlap
    scored_paragraphs: list[tuple[str, int, int, set[str]]] = []  # (id, score, position, matched_keywords)
    position = 0
    
    for section in document.sections:
        for para in section.paragraphs:
            para_keywords = extract_keywords(para.text)
            
            # Count keyword matches (intersection)
            matches = question_keywords & para_keywords
            score = len(matches)
            
            if score > 0:
                scored_paragraphs.append((para.paragraph_id, score, position, matches))
            
            position += 1
    
    if not scored_paragraphs:
        # No matches found - return empty (will trigger "Not defined")
        reason = f"No paragraphs matched keywords: {sorted(question_keywords)}. Topic may not be in document."
        return {
            "candidate_paragraph_ids": [],
            "planner_reason": reason,
            "planner_selected_automatically": True,  # Planner made the selection
        }
    
    # Sort by score (descending), then by position (ascending) for stability
    scored_paragraphs.sort(key=lambda x: (-x[1], x[2]))
    
    # Be conservative: include more paragraphs rather than fewer
    # Take all paragraphs with at least 1 match, up to a reasonable limit
    # Also include paragraphs adjacent to high-scoring ones for context
    
    selected_ids = [pid for pid, score, pos, matches in scored_paragraphs]
    
    # Limit to top 10 paragraphs to avoid overwhelming the LLM
    # But be conservative - include all with score >= 1
    max_paragraphs = 10
    selected_ids = selected_ids[:max_paragraphs]
    
    # Build reason string
    top_matches = scored_paragraphs[:3]  # Show top 3 for reason
    match_details = "; ".join(
        f"{pid}({score} keywords: {', '.join(sorted(list(kw)[:3]))}{'...' if len(kw) > 3 else ''})"
        for pid, score, pos, kw in top_matches
    )
    reason = f"Selected {len(selected_ids)} paragraphs via keyword matching. Top matches: {match_details}."
    
    return {
        "candidate_paragraph_ids": selected_ids,
        "planner_reason": reason,
        "planner_selected_automatically": True,  # Planner made the selection
    }


def retrieve_paragraphs_node(state: QAState) -> dict:
    """Retrieve paragraphs and associated equations from the document.
    
    This node:
    1. Fetches paragraphs using candidate_paragraph_ids or paragraph_ids
    2. Automatically retrieves equations associated with those paragraphs
    
    Equations are treated as first-class citizens - if a paragraph contains
    or references an equation, that equation is included in the context.
    
    Args:
        state: Current workflow state with document and paragraph IDs
        
    Returns:
        Dict with 'paragraphs' and 'equations' keys
    """
    document = state["document"]
    
    # Use candidate_paragraph_ids from planner if available
    paragraph_ids = state.get("candidate_paragraph_ids") or state.get("paragraph_ids", [])
    
    # Retrieve paragraphs
    paragraphs = document.get_paragraphs(paragraph_ids)
    
    # Retrieve equations associated with these paragraphs
    # This includes:
    # 1. Equations where associated_paragraph_id matches
    # 2. Equations referenced via paragraph.equation_refs
    equations = document.get_equations_for_paragraphs(paragraph_ids)
    
    # Also get equations referenced by equation_refs in paragraphs
    for para in paragraphs:
        for eq_ref in para.equation_refs:
            eq = document.get_equation(eq_ref)
            if eq and eq not in equations:
                equations.append(eq)
    
    return {
        "paragraphs": paragraphs,
        "equations": equations,
    }


def explain_node(state: QAState) -> dict:
    """Generate an explanation using the LLM.
    
    This node:
    1. Checks if paragraphs were retrieved
    2. If no paragraphs, returns "Not defined in the paper"
    3. Otherwise, builds prompt with paragraphs AND equations
    4. Parses and validates the response
    
    Equations are included as first-class content, clearly marked
    separately from prose paragraphs.
    
    Args:
        state: Current workflow state with paragraphs, equations, question, and llm
        
    Returns:
        Dict with 'llm_response' and 'final_response' keys
    """
    paragraphs = state["paragraphs"]
    equations = state.get("equations", [])
    question = state["question"]
    llm = state["llm"]
    
    # Handle case where no paragraphs were found
    if not paragraphs:
        return {
            "llm_response": None,
            "final_response": {
                "answer": "Not defined in the paper",
                "citations": [],
                # Note: confidence will be computed by verifier_node
            }
        }
    
    # Build context with paragraphs
    context = build_context_block(paragraphs)
    
    # Build equations block if we have equations
    equations_context = build_equations_block(equations) if equations else ""
    
    # Build prompt with both paragraphs and equations
    prompt = build_prompt(context, question, equations_context)
    
    # Get LLM response
    response = llm.generate(prompt)
    
    # Parse and validate response
    parsed = parse_llm_response(response)
    
    # Validate citations - only include citations that match provided paragraph IDs
    valid_paragraph_ids = {p.paragraph_id for p in paragraphs}
    valid_citations = [
        cid for cid in parsed.get("citations", [])
        if cid in valid_paragraph_ids
    ]
    
    # Note: confidence is computed by verifier_node based on system rules:
    # high = citations non-empty + verifier passed + planner selected automatically
    final_response = {
        "answer": parsed.get("answer", "Not defined in the paper"),
        "citations": valid_citations,
        # confidence will be set by verifier_node
    }
    
    return {
        "llm_response": response,
        "final_response": final_response,
    }


def verifier_node(state: QAState) -> dict:
    """Verify that the response is grounded and compute system-derived confidence.
    
    This node enforces hallucination control by verifying:
    1. If answer != "Not defined in the paper", citations must be non-empty
    2. All citations must exist in the provided paragraph_ids
    
    System-derived confidence rules:
    - confidence = "high" ONLY if:
      1. citations are non-empty
      2. verifier passed without overrides
      3. planner selected paragraphs automatically
    - Otherwise: confidence = "low"
    
    If verification fails, the response is overridden to reject the answer.
    
    Args:
        state: Current workflow state with final_response and paragraphs
        
    Returns:
        Dict with 'final_response', 'verification_passed', and 'verifier_reason' keys
    """
    final_response = state["final_response"]
    paragraphs = state["paragraphs"]
    include_debug = state.get("include_debug", False)
    planner_selected_automatically = state.get("planner_selected_automatically", False)
    
    answer = final_response.get("answer", "")
    citations = final_response.get("citations", [])
    
    # Get valid paragraph IDs from the retrieved paragraphs
    valid_paragraph_ids = {p.paragraph_id for p in paragraphs}
    
    # Check 1: If answer is NOT "Not defined in the paper", citations must be non-empty
    is_not_defined = answer == "Not defined in the paper"
    has_citations = len(citations) > 0
    
    if not is_not_defined and not has_citations:
        # Answer claims to have information but provides no citations
        reason = "REJECTED: Answer provided without citations. Substantive answers must cite sources."
        rejected_response = {
            "answer": "Not defined in the paper",
            "citations": [],
            "confidence": "low",  # Rejection always = low confidence
        }
        if include_debug:
            rejected_response["debug"] = {
                "planner_reason": state.get("planner_reason", ""),
                "verifier_reason": reason,
            }
        return {
            "final_response": rejected_response,
            "verification_passed": False,
            "verifier_reason": reason,
        }
    
    # Check 2: All citations must exist in the provided paragraph_ids
    invalid_citations = [cid for cid in citations if cid not in valid_paragraph_ids]
    
    if invalid_citations:
        # Some citations reference paragraphs not in the provided context
        reason = f"REJECTED: Invalid citations {invalid_citations} not in provided context {sorted(valid_paragraph_ids)}."
        rejected_response = {
            "answer": "Not defined in the paper",
            "citations": [],
            "confidence": "low",  # Rejection always = low confidence
        }
        if include_debug:
            rejected_response["debug"] = {
                "planner_reason": state.get("planner_reason", ""),
                "verifier_reason": reason,
            }
        return {
            "final_response": rejected_response,
            "verification_passed": False,
            "verifier_reason": reason,
        }
    
    # Verification passed - now compute system-derived confidence
    # confidence = "high" ONLY if:
    #   1. citations are non-empty
    #   2. verifier passed (we're here, so yes)
    #   3. planner selected paragraphs automatically
    verification_passed_without_override = True  # We passed all checks
    
    if has_citations and verification_passed_without_override and planner_selected_automatically:
        confidence = "high"
    else:
        confidence = "low"
    
    # Build reason
    if is_not_defined:
        reason = "PASSED: Answer correctly indicates topic not defined in paper."
    else:
        reason = f"PASSED: Answer grounded with {len(citations)} valid citations: {citations}."
    
    # Add confidence reasoning to verifier reason
    confidence_factors = []
    if not has_citations:
        confidence_factors.append("no citations")
    if not planner_selected_automatically:
        confidence_factors.append("paragraphs provided manually")
    
    if confidence_factors:
        reason += f" Confidence=low due to: {', '.join(confidence_factors)}."
    else:
        reason += " Confidence=high (citations present, verified, planner auto-selected)."
    
    # Add debug info to final response if requested
    output_response = dict(final_response)
    output_response["confidence"] = confidence
    if include_debug:
        output_response["debug"] = {
            "planner_reason": state.get("planner_reason", ""),
            "verifier_reason": reason,
        }
    
    return {
        "final_response": output_response,
        "verification_passed": True,
        "verifier_reason": reason,
    }


def evidence_linker_node(state: QAState) -> dict:
    """Link related evidence based on shared entities.
    
    This node connects paragraphs and equations that share common entities
    (variables or concepts). It does NOT call an LLM and does NOT generate
    prose - it only groups existing evidence.
    
    Evidence is linked if and only if they share at least one extracted entity.
    
    Args:
        state: Current workflow state with paragraphs, equations
        
    Returns:
        Dict with 'linked_evidence' key containing list of evidence links:
        [
            {
                "source_ids": [paragraph_id | equation_id, ...],
                "shared_entities": {"variables": [...], "concepts": [...]}
            },
            ...
        ]
    """
    paragraphs = state.get("paragraphs", [])
    equations = state.get("equations", [])
    
    # If no evidence to link, return empty list
    if not paragraphs and not equations:
        return {"linked_evidence": []}
    
    # Extract entities for each piece of evidence
    # Structure: {id: {"variables": set, "concepts": set}}
    evidence_entities: dict[str, dict[str, set[str]]] = {}
    
    for para in paragraphs:
        # Use pre-extracted entities if available, otherwise extract on-the-fly
        if para.entities:
            entities = para.entities
        else:
            entities = extract_entities_as_model(para.text)
        
        evidence_entities[para.paragraph_id] = {
            "variables": set(entities.variables),
            "concepts": set(entities.concepts),
        }
    
    for eq in equations:
        # Extract entities from equation text
        entities = extract_entities_as_model(eq.equation_text)
        evidence_entities[eq.equation_id] = {
            "variables": set(entities.variables),
            "concepts": set(entities.concepts),
        }
    
    # Find all pairs that share at least one entity
    # Use Union-Find to group connected evidence
    all_ids = list(evidence_entities.keys())
    
    if len(all_ids) < 2:
        # Need at least 2 pieces of evidence to form a link
        return {"linked_evidence": []}
    
    # Build adjacency list for entities that are shared
    # Map each entity to the evidence IDs that contain it
    variable_to_evidence: dict[str, set[str]] = {}
    concept_to_evidence: dict[str, set[str]] = {}
    
    for eid, ent in evidence_entities.items():
        for var in ent["variables"]:
            if var not in variable_to_evidence:
                variable_to_evidence[var] = set()
            variable_to_evidence[var].add(eid)
        
        for concept in ent["concepts"]:
            if concept not in concept_to_evidence:
                concept_to_evidence[concept] = set()
            concept_to_evidence[concept].add(eid)
    
    # Find all connected components (groups of linked evidence)
    # Using simple Union-Find
    parent = {eid: eid for eid in all_ids}
    
    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x: str, y: str) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union evidence that shares variables
    for var, eids in variable_to_evidence.items():
        eids_list = list(eids)
        for i in range(1, len(eids_list)):
            union(eids_list[0], eids_list[i])
    
    # Union evidence that shares concepts
    for concept, eids in concept_to_evidence.items():
        eids_list = list(eids)
        for i in range(1, len(eids_list)):
            union(eids_list[0], eids_list[i])
    
    # Group by connected component
    components: dict[str, list[str]] = {}
    for eid in all_ids:
        root = find(eid)
        if root not in components:
            components[root] = []
        components[root].append(eid)
    
    # Build linked_evidence output
    # Only include components with 2+ pieces of evidence (actual links)
    linked_evidence = []
    
    for root, source_ids in components.items():
        if len(source_ids) < 2:
            # No link - single piece of evidence
            continue
        
        # Find shared entities among all sources in this component
        # (intersection of all entity sets)
        shared_vars = None
        shared_concepts = None
        
        for eid in source_ids:
            ent = evidence_entities[eid]
            if shared_vars is None:
                shared_vars = ent["variables"].copy()
                shared_concepts = ent["concepts"].copy()
            else:
                # For shared entities, we want entities that appear in at least 2 sources
                # Not strict intersection, but entities connecting any pair
                pass
        
        # Actually, for "shared" we want the union of entities that caused links
        # i.e., entities that appear in 2+ sources
        all_vars_in_group: dict[str, int] = {}
        all_concepts_in_group: dict[str, int] = {}
        
        for eid in source_ids:
            ent = evidence_entities[eid]
            for var in ent["variables"]:
                all_vars_in_group[var] = all_vars_in_group.get(var, 0) + 1
            for concept in ent["concepts"]:
                all_concepts_in_group[concept] = all_concepts_in_group.get(concept, 0) + 1
        
        # Shared = appears in 2+ sources
        shared_variables = sorted([v for v, count in all_vars_in_group.items() if count >= 2])
        shared_concepts_list = sorted([c for c, count in all_concepts_in_group.items() if count >= 2])
        
        linked_evidence.append({
            "source_ids": sorted(source_ids),
            "shared_entities": {
                "variables": shared_variables,
                "concepts": shared_concepts_list,
            }
        })
    
    return {"linked_evidence": linked_evidence}


# =============================================================================
# Composer Prompt
# =============================================================================

COMPOSER_SYSTEM_PROMPT = """You are a research paper explanation composer. Your task is to compose a clear, grounded explanation by ONLY paraphrasing the provided verified evidence.

CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:
1. You may ONLY paraphrase existing evidence - do NOT introduce new information.
2. Every sentence MUST cite exactly one source using format: "Statement. [source_id]"
3. You may NOT introduce new entities (variables, concepts) not present in the evidence.
4. You may NOT make claims that combine information in ways not supported by the evidence.
5. You may NOT add assumptions, implications, or conclusions beyond what is stated.
6. You may NOT use any external knowledge.
7. If the evidence is insufficient, compose what you can and cite appropriately.

Format each sentence as:
"Paraphrased statement from source. [source_id]"

You must respond ONLY with a JSON object in this exact format:
{
    "composed_explanation": "First sentence. [source_id_1] Second sentence. [source_id_2]",
    "sentences": [
        {"text": "First sentence.", "citation": "source_id_1"},
        {"text": "Second sentence.", "citation": "source_id_2"}
    ]
}

Do not include any text outside the JSON object."""


def build_composer_prompt(
    paragraphs: list[Paragraph],
    equations: list[Equation],
    linked_evidence: list[dict],
    question: str,
) -> str:
    """Build the composer prompt with verified evidence.
    
    Args:
        paragraphs: Verified paragraphs to compose from
        equations: Verified equations to compose from
        linked_evidence: Links between related evidence
        question: The original question being answered
        
    Returns:
        Complete prompt string for the composer
    """
    # Build evidence block
    evidence_blocks = []
    
    for para in paragraphs:
        evidence_blocks.append(f"[{para.paragraph_id}] (paragraph)\n{para.text}")
    
    for eq in equations:
        evidence_blocks.append(f"[{eq.equation_id}] (equation from {eq.associated_paragraph_id})\n{eq.equation_text}")
    
    evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "No evidence provided."
    
    # Build linked evidence section
    links_text = ""
    if linked_evidence:
        link_descriptions = []
        for link in linked_evidence:
            source_ids = ", ".join(link["source_ids"])
            shared_vars = link["shared_entities"].get("variables", [])
            shared_concepts = link["shared_entities"].get("concepts", [])
            shared_items = shared_vars + shared_concepts
            shared_str = ", ".join(shared_items) if shared_items else "none"
            link_descriptions.append(f"  - Sources [{source_ids}] share: {shared_str}")
        links_text = "\n\n=== LINKED EVIDENCE ===\nThe following evidence is connected:\n" + "\n".join(link_descriptions) + "\n=== END LINKED EVIDENCE ==="
    
    return f"""{COMPOSER_SYSTEM_PROMPT}

=== VERIFIED EVIDENCE ===
{evidence_text}
=== END VERIFIED EVIDENCE ==={links_text}

QUESTION: {question}

Compose an explanation using ONLY the verified evidence above. Each sentence must cite its source.

JSON Response:"""


def parse_composer_response(response) -> dict:
    """Parse the composer LLM response.
    
    Args:
        response: LLM response (LLMResponse or str)
        
    Returns:
        Dict with 'composed_explanation' and 'sentences' keys
    """
    import json
    import re
    
    # Get text from response
    if hasattr(response, 'content'):
        text = response.content
    else:
        text = str(response)
    
    # Try to extract JSON from the response
    # Handle markdown code blocks
    if '```json' in text:
        match = re.search(r'```json\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1)
    elif '```' in text:
        match = re.search(r'```\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1)
    
    # Find JSON object in text
    text = text.strip()
    start = text.find('{')
    end = text.rfind('}') + 1
    
    if start >= 0 and end > start:
        json_str = text[start:end]
        try:
            parsed = json.loads(json_str)
            return {
                "composed_explanation": parsed.get("composed_explanation", ""),
                "sentences": parsed.get("sentences", []),
            }
        except json.JSONDecodeError:
            pass
    
    # Fallback: return empty
    return {
        "composed_explanation": "",
        "sentences": [],
    }


def verify_composed_explanation(
    sentences: list[dict],
    valid_source_ids: set[str],
    paragraphs: list[Paragraph],
    equations: list[Equation],
) -> tuple[bool, str]:
    """Verify that the composed explanation follows all rules.
    
    Rules:
    1. Every sentence must have a citation
    2. Citations must reference valid source IDs
    3. No new entities introduced (checked against source entities)
    
    Args:
        sentences: List of {"text": str, "citation": str} dicts
        valid_source_ids: Set of valid source IDs (paragraph + equation IDs)
        paragraphs: Original paragraphs for entity checking
        equations: Original equations for entity checking
        
    Returns:
        Tuple of (passed: bool, reason: str)
    """
    if not sentences:
        return False, "REJECTED: No sentences in composed explanation."
    
    # Check each sentence has a citation
    for i, sent in enumerate(sentences):
        text = sent.get("text", "")
        citation = sent.get("citation", "")
        
        if not citation:
            return False, f"REJECTED: Sentence {i+1} lacks a citation: '{text[:50]}...'"
        
        if citation not in valid_source_ids:
            return False, f"REJECTED: Invalid citation '{citation}' not in sources {sorted(valid_source_ids)}."
    
    # Build set of allowed entities from sources
    allowed_entities = set()
    for para in paragraphs:
        if para.entities:
            allowed_entities.update(para.entities.variables)
            allowed_entities.update(para.entities.concepts)
        else:
            # Extract on-the-fly
            entities = extract_entities_as_model(para.text)
            allowed_entities.update(entities.variables)
            allowed_entities.update(entities.concepts)
    
    for eq in equations:
        entities = extract_entities_as_model(eq.equation_text)
        allowed_entities.update(entities.variables)
        allowed_entities.update(entities.concepts)
    
    # Check for new entities in composed text
    for sent in sentences:
        text = sent.get("text", "")
        sent_entities = extract_entities_as_model(text)
        
        # Variables must be from allowed set
        for var in sent_entities.variables:
            if var not in allowed_entities:
                return False, f"REJECTED: New variable '{var}' introduced in composed explanation."
        
        # Concepts are more lenient - common words are OK
        # Only reject if it's a technical concept not in sources
        TECHNICAL_CONCEPTS = {
            'attention', 'transformer', 'encoder', 'decoder', 'embedding',
            'softmax', 'layer normalization', 'dropout', 'residual',
            'multi-head', 'self-attention', 'cross-attention', 'positional encoding',
            'feedforward', 'bleu', 'bleu score', 'perplexity', 'accuracy',
        }
        for concept in sent_entities.concepts:
            if concept in TECHNICAL_CONCEPTS and concept not in allowed_entities:
                return False, f"REJECTED: New technical concept '{concept}' introduced in composed explanation."
    
    return True, f"PASSED: Composed explanation verified with {len(sentences)} cited sentences."


def composer_node(state: QAState) -> dict:
    """Compose a grounded explanation from verified evidence.
    
    This node:
    1. Takes verified paragraphs, equations, and linked evidence
    2. Prompts the LLM to compose an explanation
    3. Verifies each sentence has a citation
    4. Verifies no new entities are introduced
    
    The composer may ONLY paraphrase existing evidence.
    It may NOT introduce new entities or claims.
    
    Args:
        state: Current workflow state with paragraphs, equations, linked_evidence
        
    Returns:
        Dict with 'composed_explanation' and 'composer_verification_passed' keys
    """
    paragraphs = state.get("paragraphs", [])
    equations = state.get("equations", [])
    linked_evidence = state.get("linked_evidence", [])
    question = state.get("question", "")
    llm = state.get("llm")
    
    # If no evidence, return empty
    if not paragraphs and not equations:
        return {
            "composed_explanation": None,
            "composer_verification_passed": False,
        }
    
    # Build valid source IDs
    valid_source_ids = set()
    for para in paragraphs:
        valid_source_ids.add(para.paragraph_id)
    for eq in equations:
        valid_source_ids.add(eq.equation_id)
    
    # Build prompt
    prompt = build_composer_prompt(paragraphs, equations, linked_evidence, question)
    
    # Get LLM response
    response = llm.generate(prompt)
    
    # Parse response
    parsed = parse_composer_response(response)
    composed_explanation = parsed.get("composed_explanation", "")
    sentences = parsed.get("sentences", [])
    
    # Verify the composed explanation
    passed, reason = verify_composed_explanation(
        sentences, valid_source_ids, paragraphs, equations
    )
    
    # If verification failed, set composed_explanation to None
    if not passed:
        return {
            "composed_explanation": None,
            "composer_verification_passed": False,
            "verifier_reason": state.get("verifier_reason", "") + f" Composer: {reason}",
        }
    
    return {
        "composed_explanation": composed_explanation,
        "composer_verification_passed": passed,
        "verifier_reason": state.get("verifier_reason", "") + f" Composer: {reason}",
    }


# =============================================================================
# Graph Construction
# =============================================================================

def create_qa_graph() -> StateGraph:
    """Create the Q&A workflow graph.
    
    The graph has six nodes:
    1. planner: Selects candidate paragraphs via keyword matching
    2. retrieve_paragraphs: Fetches paragraphs from document
    3. explain: Generates answer using LLM
    4. verify: Validates response is grounded in provided paragraphs
    5. evidence_linker: Links related evidence based on shared entities
    6. composer: Composes grounded explanation from verified evidence
    
    Flow: START -> planner -> retrieve_paragraphs -> explain -> verify -> evidence_linker -> composer -> END
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the graph with our state schema
    builder = StateGraph(QAState)
    
    # Add nodes
    builder.add_node("planner", planner_node)
    builder.add_node("retrieve_paragraphs", retrieve_paragraphs_node)
    builder.add_node("explain", explain_node)
    builder.add_node("verify", verifier_node)
    builder.add_node("evidence_linker", evidence_linker_node)
    builder.add_node("composer", composer_node)
    
    # Define the flow: planner -> retrieve -> explain -> verify -> evidence_linker -> composer -> END
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retrieve_paragraphs")
    builder.add_edge("retrieve_paragraphs", "explain")
    builder.add_edge("explain", "verify")
    builder.add_edge("verify", "evidence_linker")
    builder.add_edge("evidence_linker", "composer")
    builder.add_edge("composer", END)
    
    # Compile and return
    return builder.compile()


# Create a default graph instance
qa_graph = create_qa_graph()


# =============================================================================
# High-level API
# =============================================================================

def explain_question_graph(
    document: Document,
    paragraph_ids: list[str],
    question: str,
    llm: LLMInterface,
    include_debug: bool = False,
) -> dict:
    """Answer a question using the LangGraph workflow.
    
    This function provides the same interface as explain_question()
    but uses the LangGraph workflow internally.
    
    Args:
        document: The Document to query
        paragraph_ids: List of paragraph IDs to use as context
        question: The question to answer
        llm: The LLM interface to use for generation
        include_debug: If True, include debug info (planner_reason, verifier_reason)
        
    Returns:
        Dict with keys:
        - answer: The answer string
        - citations: List of paragraph IDs used
        - confidence: "high" or "low"
        - debug (optional): Dict with planner_reason and verifier_reason
    """
    # Prepare initial state
    initial_state: QAState = {
        "document": document,
        "paragraph_ids": paragraph_ids,
        "question": question,
        "llm": llm,
        "include_debug": include_debug,
    }
    
    # Run the graph
    result = qa_graph.invoke(initial_state)
    
    # Extract and return the final response
    return result["final_response"]
