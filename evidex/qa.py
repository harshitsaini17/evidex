"""
Question-answering functionality for the Evidex system.

This module provides the core explain_question function that answers
questions using ONLY the content from the provided document paragraphs.
"""

from evidex.models import Document, Paragraph, Equation, QAResponse
from evidex.llm import LLMInterface, LLMResponse, parse_llm_response


# System prompt that strictly forbids external knowledge
SYSTEM_PROMPT = """You are a research paper analysis assistant. Your ONLY task is to answer questions using EXCLUSIVELY the provided document content.

CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:
1. You may ONLY use information that is EXPLICITLY stated in the provided paragraphs and equations.
2. You must NEVER use any external knowledge, even if you know the answer from training.
3. If a concept, term, or fact is NOT defined or explained in the provided text, you MUST respond with "Not defined in the paper".
4. Every claim in your answer MUST be directly traceable to the provided paragraphs or equations.
5. You MUST ALWAYS include citations in your response.
   - ALWAYS cite the paragraph IDs you used to formulate your answer
   - Even for summaries or explanations, cite ALL paragraphs you reference
   - NEVER return an answer without citations unless the answer is exactly "Not defined in the paper"
6. If you are uncertain whether the text supports the answer, set confidence to "low".
7. EQUATIONS are provided separately and are CRITICAL to understanding. Do NOT simplify or modify equations.

RESPONSE FORMAT - You MUST respond with ONLY a JSON object, no other text:

Example 1 (specific question):
Question: "What is attention?"
Paragraph [p1]: "Attention is a mechanism..."
Response:
{
    "answer": "Attention is a mechanism...",
    "citations": ["p1"],
    "confidence": "high"
}

Example 2 (explanation request):
Question: "Explain this section"
Paragraph [s1_p7]: "The model uses convolutional layers..."
Response:
{
    "answer": "This section describes convolutional layers and their computational complexity...",
    "citations": ["s1_p7"],
    "confidence": "low"
}

Example 3 (not found):
Question: "What is quantum computing?"
Paragraph [p1]: "Deep learning uses neural networks..."
Response:
{
    "answer": "Not defined in the paper",
    "citations": [],
    "confidence": "low"
}

RESPONSE SCHEMA:
{
    "answer": string,
    "citations": [string],  // REQUIRED unless answer is "Not defined in the paper"
    "confidence": "high" | "low"
}

Do not include any text outside the JSON object."""


def build_context_block(paragraphs: list[Paragraph]) -> str:
    """Build a formatted context block from paragraphs.
    
    Args:
        paragraphs: List of paragraphs to include
        
    Returns:
        Formatted string with paragraph IDs and text
    """
    if not paragraphs:
        return "No paragraphs provided."
    
    blocks = []
    for para in paragraphs:
        blocks.append(f"[{para.paragraph_id}]\n{para.text}")
    
    return "\n\n".join(blocks)


def build_equations_block(equations: list[Equation]) -> str:
    """Build a formatted equations block.
    
    Equations are treated as first-class citizens and are clearly
    marked separately from prose paragraphs.
    
    Args:
        equations: List of equations to include
        
    Returns:
        Formatted string with equation IDs and text
    """
    if not equations:
        return ""
    
    blocks = []
    for eq in equations:
        blocks.append(f"[{eq.equation_id}] (from {eq.associated_paragraph_id})\n{eq.equation_text}")
    
    return "\n\n".join(blocks)


def build_prompt(context: str, question: str, equations_context: str = "") -> str:
    """Build the complete prompt for the LLM.
    
    Args:
        context: The formatted paragraph context
        question: The user's question
        equations_context: Optional formatted equations block
        
    Returns:
        Complete prompt string
    """
    # Build equations section if present
    equations_section = ""
    if equations_context:
        equations_section = f"""
=== EQUATIONS ===
The following equations are critical to understanding the document content.
Do NOT simplify or modify these equations - they must be preserved exactly.

{equations_context}
=== END EQUATIONS ===
"""
    
    return f"""{SYSTEM_PROMPT}

=== DOCUMENT CONTENT ===
{context}
=== END DOCUMENT CONTENT ===
{equations_section}
QUESTION: {question}

CRITICAL: You MUST respond with valid JSON including citations array.

Example Response Format:
{{
    "answer": "Your answer summarizing the content from the paragraphs above",
    "citations": ["s1_p7"],
    "confidence": "low"
}}

Now provide your response as JSON:"""


def explain_question(
    document: Document,
    paragraph_ids: list[str],
    question: str,
    llm: LLMInterface,
    include_debug: bool = False,
) -> dict:
    """Answer a question using only specified paragraphs from a document.
    
    This function uses the LangGraph workflow internally to:
    1. Retrieve the specified paragraphs from the document
    2. Build a strict prompt forbidding external knowledge
    3. Send to the LLM and parse the response
    4. Verify the response is grounded in the provided paragraphs
    5. Return a structured response with citations
    
    Args:
        document: The Document to query
        paragraph_ids: List of paragraph IDs to use as context
        question: The question to answer
        llm: The LLM interface to use for generation
        include_debug: If True, include debug info in response (default: False)
        
    Returns:
        Dict with keys:
        - answer: The answer string
        - citations: List of paragraph IDs used
        - confidence: "high" or "low"
        - debug (optional): Dict with planner_reason and verifier_reason
        
    Raises:
        ValueError: If no valid paragraphs found or LLM response invalid
    """
    # Import here to avoid circular imports
    from evidex.graph import qa_graph
    
    # Initialize graph state
    initial_state = {
        "document": document,
        "paragraph_ids": paragraph_ids,
        "question": question,
        "llm": llm,
        "include_debug": include_debug,
    }
    
    # Execute the graph
    result = qa_graph.invoke(initial_state)
    
    # Return the final response
    return result["final_response"]
