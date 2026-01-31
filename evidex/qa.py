"""
Question-answering functionality for the Evidex system.

This module provides the core explain_question function that answers
questions using ONLY the content from the provided document paragraphs.
"""

from evidex.models import Document, Paragraph, QAResponse
from evidex.llm import LLMInterface, LLMResponse, parse_llm_response


# System prompt that strictly forbids external knowledge
SYSTEM_PROMPT = """You are a research paper analysis assistant. Your ONLY task is to answer questions using EXCLUSIVELY the provided document content.

CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:
1. You may ONLY use information that is EXPLICITLY stated in the provided paragraphs.
2. You must NEVER use any external knowledge, even if you know the answer from training.
3. If a concept, term, or fact is NOT defined or explained in the provided text, you MUST respond with "Not defined in the paper".
4. Every claim in your answer MUST be directly traceable to the provided paragraphs.
5. You MUST cite the paragraph IDs that support your answer.
6. If you are uncertain whether the text supports the answer, set confidence to "low".

You must respond ONLY with a JSON object in this exact format:
{
    "answer": "Your answer based solely on the provided text, or 'Not defined in the paper' if the information is not present",
    "citations": ["paragraph_id_1", "paragraph_id_2"],
    "confidence": "high" or "low"
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


def build_prompt(context: str, question: str) -> str:
    """Build the complete prompt for the LLM.
    
    Args:
        context: The formatted paragraph context
        question: The user's question
        
    Returns:
        Complete prompt string
    """
    return f"""{SYSTEM_PROMPT}

=== DOCUMENT CONTENT ===
{context}
=== END DOCUMENT CONTENT ===

QUESTION: {question}

Remember: Answer ONLY using the document content above. If the answer is not in the text, respond with "Not defined in the paper".

JSON Response:"""


def explain_question(
    document: Document,
    paragraph_ids: list[str],
    question: str,
    llm: LLMInterface,
) -> dict:
    """Answer a question using only specified paragraphs from a document.
    
    This function:
    1. Retrieves the specified paragraphs from the document
    2. Builds a strict prompt forbidding external knowledge
    3. Sends to the LLM and parses the response
    4. Returns a structured response with citations
    
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
        
    Raises:
        ValueError: If no valid paragraphs found or LLM response invalid
    """
    # Retrieve the specified paragraphs
    paragraphs = document.get_paragraphs(paragraph_ids)
    
    if not paragraphs:
        return {
            "answer": "Not defined in the paper",
            "citations": [],
            "confidence": "high",
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
    
    return {
        "answer": parsed.get("answer", "Not defined in the paper"),
        "citations": valid_citations,
        "confidence": confidence,
    }
