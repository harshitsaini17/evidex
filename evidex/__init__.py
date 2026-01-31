"""
Evidex: Research Paper Understanding System

A system that answers questions ONLY using content from ingested documents,
with strict citation requirements and no external knowledge.
"""

from evidex.models import Paragraph, Section, Document, QAResponse
from evidex.qa import explain_question
from evidex.llm import LLMInterface, MockLLM, GroqLLM
from evidex.graph import (
    QAState,
    planner_node,
    extract_keywords,
    retrieve_paragraphs_node,
    explain_node,
    verifier_node,
    create_qa_graph,
    qa_graph,
    explain_question_graph,
)
from evidex.ingest import (
    parse_pdf_to_document,
    extract_text_from_pdf,
    get_all_paragraph_ids,
    search_paragraphs,
)

__all__ = [
    # Models
    "Paragraph",
    "Section", 
    "Document",
    "QAResponse",
    # Original Q&A
    "explain_question",
    # LLM
    "LLMInterface",
    "MockLLM",
    "GroqLLM",
    # LangGraph components
    "QAState",
    "planner_node",
    "extract_keywords",
    "retrieve_paragraphs_node",
    "explain_node",
    "verifier_node",
    "create_qa_graph",
    "qa_graph",
    "explain_question_graph",
    # Ingestion
    "parse_pdf_to_document",
    "extract_text_from_pdf",
    "get_all_paragraph_ids",
    "search_paragraphs",
]
