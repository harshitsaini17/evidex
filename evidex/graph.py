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
        paragraph_ids: List of paragraph IDs to use as context
        question: The user's question
        llm: The LLM interface (passed through state)
        paragraphs: Retrieved paragraphs (set by retrieve_paragraphs_node)
        llm_response: Raw LLM response (set by explain_node)
        final_response: Final structured response dict (set by explain_node)
    """
    # Inputs
    document: Document
    paragraph_ids: list[str]
    question: str
    llm: LLMInterface
    
    # Intermediate state
    paragraphs: list[Paragraph]
    llm_response: LLMResponse | None
    
    # Output
    final_response: dict


# =============================================================================
# Node Functions
# =============================================================================

def retrieve_paragraphs_node(state: QAState) -> dict:
    """Retrieve paragraphs from the document based on paragraph_ids.
    
    This node wraps Document.get_paragraphs() to fetch the specified
    paragraphs from the document.
    
    Args:
        state: Current workflow state with document and paragraph_ids
        
    Returns:
        Dict with 'paragraphs' key containing the retrieved paragraphs
    """
    document = state["document"]
    paragraph_ids = state["paragraph_ids"]
    
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


# =============================================================================
# Graph Construction
# =============================================================================

def create_qa_graph() -> StateGraph:
    """Create the Q&A workflow graph.
    
    The graph has two nodes:
    1. retrieve_paragraphs: Fetches paragraphs from document
    2. explain: Generates answer using LLM
    
    Flow: START -> retrieve_paragraphs -> explain -> END
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the graph with our state schema
    builder = StateGraph(QAState)
    
    # Add nodes
    builder.add_node("retrieve_paragraphs", retrieve_paragraphs_node)
    builder.add_node("explain", explain_node)
    
    # Define the flow
    builder.add_edge(START, "retrieve_paragraphs")
    builder.add_edge("retrieve_paragraphs", "explain")
    builder.add_edge("explain", END)
    
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
) -> dict:
    """Answer a question using the LangGraph workflow.
    
    This function provides the same interface as explain_question()
    but uses the LangGraph workflow internally.
    
    Args:
        document: The Document to query
        paragraph_ids: List of paragraph IDs to use as context
        question: The question to answer
        llm: The LLM interface to use for generation
        
    Returns:
        Dict with keys:
        - answer: The answer string
        - citations: List of paragraph IDs used
        - confidence: "high" or "low"
    """
    # Prepare initial state
    initial_state: QAState = {
        "document": document,
        "paragraph_ids": paragraph_ids,
        "question": question,
        "llm": llm,
    }
    
    # Run the graph
    result = qa_graph.invoke(initial_state)
    
    # Extract and return the final response
    return result["final_response"]
