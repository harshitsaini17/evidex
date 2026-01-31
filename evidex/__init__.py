"""
Evidex: Research Paper Understanding System

A system that answers questions ONLY using content from ingested documents,
with strict citation requirements and no external knowledge.
"""

from evidex.models import Paragraph, Section, Document, Equation, Entities, QAResponse
from evidex.qa import explain_question
from evidex.llm import LLMInterface, MockLLM, GroqLLM
from evidex.graph import (
    QAState,
    planner_node,
    extract_keywords,
    retrieve_paragraphs_node,
    explain_node,
    verifier_node,
    evidence_linker_node,
    create_qa_graph,
    qa_graph,
    explain_question_graph,
)
from evidex.ingest import (
    parse_pdf_to_document,
    extract_text_from_pdf,
    get_all_paragraph_ids,
    search_paragraphs,
    extract_equations_from_document,
    search_equations,
)
from evidex.entities import (
    extract_entities,
    extract_entities_as_model,
    extract_variables,
    extract_concepts,
    extract_entities_for_document,
    get_all_variables,
    get_all_concepts,
)

__all__ = [
    # Models
    "Paragraph",
    "Section", 
    "Document",
    "Equation",
    "Entities",
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
    "evidence_linker_node",
    "create_qa_graph",
    "qa_graph",
    "explain_question_graph",
    # Ingestion
    "parse_pdf_to_document",
    "extract_text_from_pdf",
    "get_all_paragraph_ids",
    "search_paragraphs",
    "extract_equations_from_document",
    "search_equations",
    # Entity extraction (GraphRAG scaffolding)
    "extract_entities",
    "extract_entities_as_model",
    "extract_variables",
    "extract_concepts",
    "extract_entities_for_document",
    "get_all_variables",
    "get_all_concepts",
]
