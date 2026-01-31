"""
Evidex: Research Paper Understanding System

A system that answers questions ONLY using content from ingested documents,
with strict citation requirements and no external knowledge.
"""

from evidex.models import Paragraph, Section, Document, Equation, Entities, Motivation, QAResponse
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
    composer_node,
    build_composer_prompt,
    parse_composer_response,
    verify_composed_explanation,
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
from evidex.motivations import (
    extract_motivations,
    extract_motivations_as_list,
    has_motivation,
    extract_motivations_for_paragraph,
    extract_motivations_for_document,
    search_motivations,
    get_motivation_summary,
)

__all__ = [
    # Models
    "Paragraph",
    "Section", 
    "Document",
    "Equation",
    "Entities",
    "Motivation",
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
    "composer_node",
    "build_composer_prompt",
    "parse_composer_response",
    "verify_composed_explanation",
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
    # Motivation extraction
    "extract_motivations",
    "extract_motivations_as_list",
    "has_motivation",
    "extract_motivations_for_paragraph",
    "extract_motivations_for_document",
    "search_motivations",
    "get_motivation_summary",
]
