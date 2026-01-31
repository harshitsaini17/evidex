"""
LangGraph workflow for the Evidex Q&A system.

This module provides a LangGraph-based workflow that wraps the existing
explain_question logic. The behavior is IDENTICAL to the original function.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from evidex.models import Document, Paragraph
from evidex.llm import LLMInterface, LLMResponse, parse_llm_response
from evidex.qa import build_context_block, build_prompt


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
        paragraphs: Retrieved paragraphs (set by retrieve_paragraphs_node)
        llm_response: Raw LLM response (set by explain_node)
        final_response: Final structured response dict (set by explain_node)
        verification_passed: Whether the response passed verification (set by verifier_node)
        planner_reason: Explanation of planner's decision (set by planner_node)
        verifier_reason: Explanation of verifier's decision (set by verifier_node)
    """
    # Inputs
    document: Document
    paragraph_ids: list[str]  # Optional: if provided, planner is skipped
    question: str
    llm: LLMInterface
    include_debug: bool  # Optional: if True, include debug info in output
    
    # Intermediate state (set by nodes)
    candidate_paragraph_ids: list[str]  # Set by planner_node
    paragraphs: list[Paragraph]
    llm_response: LLMResponse | None
    
    # Debug/introspection state (set by nodes)
    planner_reason: str  # Explanation of planner's paragraph selection
    verifier_reason: str  # Explanation of verifier's decision
    
    # Output
    final_response: dict
    verification_passed: bool


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
        }
    
    # Extract keywords from the question
    question_keywords = extract_keywords(question)
    
    if not question_keywords:
        # No meaningful keywords - return empty (will trigger "Not defined")
        reason = "No meaningful keywords extracted from question. Cannot match paragraphs."
        return {
            "candidate_paragraph_ids": [],
            "planner_reason": reason,
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
    }


def retrieve_paragraphs_node(state: QAState) -> dict:
    """Retrieve paragraphs from the document based on candidate_paragraph_ids.
    
    This node wraps Document.get_paragraphs() to fetch the specified
    paragraphs from the document. It uses candidate_paragraph_ids from
    the planner if available, otherwise falls back to paragraph_ids.
    
    Args:
        state: Current workflow state with document and paragraph IDs
        
    Returns:
        Dict with 'paragraphs' key containing the retrieved paragraphs
    """
    document = state["document"]
    
    # Use candidate_paragraph_ids from planner if available
    paragraph_ids = state.get("candidate_paragraph_ids") or state.get("paragraph_ids", [])
    
    paragraphs = document.get_paragraphs(paragraph_ids)
    
    return {"paragraphs": paragraphs}


def explain_node(state: QAState) -> dict:
    """Generate an explanation using the LLM.
    
    This node:
    1. Checks if paragraphs were retrieved
    2. If no paragraphs, returns "Not defined in the paper"
    3. Otherwise, builds prompt and calls LLM
    4. Parses and validates the response
    
    Args:
        state: Current workflow state with paragraphs, question, and llm
        
    Returns:
        Dict with 'llm_response' and 'final_response' keys
    """
    paragraphs = state["paragraphs"]
    question = state["question"]
    llm = state["llm"]
    
    # Handle case where no paragraphs were found
    if not paragraphs:
        return {
            "llm_response": None,
            "final_response": {
                "answer": "Not defined in the paper",
                "citations": [],
                "confidence": "high",
            }
        }
    
    # Build context and prompt
    context = build_context_block(paragraphs)
    prompt = build_prompt(context, question)
    
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
    
    # Validate confidence value
    confidence = parsed.get("confidence", "low")
    if confidence not in ("high", "low"):
        confidence = "low"
    
    final_response = {
        "answer": parsed.get("answer", "Not defined in the paper"),
        "citations": valid_citations,
        "confidence": confidence,
    }
    
    return {
        "llm_response": response,
        "final_response": final_response,
    }


def verifier_node(state: QAState) -> dict:
    """Verify that the response is grounded in the provided paragraphs.
    
    This node enforces hallucination control by verifying:
    1. If answer != "Not defined in the paper", citations must be non-empty
    2. All citations must exist in the provided paragraph_ids
    
    If verification fails, the response is overridden to reject the answer.
    
    Args:
        state: Current workflow state with final_response and paragraphs
        
    Returns:
        Dict with 'final_response', 'verification_passed', and 'verifier_reason' keys
    """
    final_response = state["final_response"]
    paragraphs = state["paragraphs"]
    include_debug = state.get("include_debug", False)
    
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
            "confidence": "low",
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
            "confidence": "low",
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
    
    # Verification passed
    if is_not_defined:
        reason = "PASSED: Answer correctly indicates topic not defined in paper."
    else:
        reason = f"PASSED: Answer grounded with {len(citations)} valid citations: {citations}."
    
    # Add debug info to final response if requested
    output_response = dict(final_response)
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


# =============================================================================
# Graph Construction
# =============================================================================

def create_qa_graph() -> StateGraph:
    """Create the Q&A workflow graph.
    
    The graph has four nodes:
    1. planner: Selects candidate paragraphs via keyword matching
    2. retrieve_paragraphs: Fetches paragraphs from document
    3. explain: Generates answer using LLM
    4. verify: Validates response is grounded in provided paragraphs
    
    Flow: START -> planner -> retrieve_paragraphs -> explain -> verify -> END
    
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
    
    # Define the flow: planner -> retrieve -> explain -> verify -> END
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retrieve_paragraphs")
    builder.add_edge("retrieve_paragraphs", "explain")
    builder.add_edge("explain", "verify")
    builder.add_edge("verify", END)
    
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
