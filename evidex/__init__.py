"""
Evidex: Research Paper Understanding System

A system that answers questions ONLY using content from ingested documents,
with strict citation requirements and no external knowledge.
"""

from evidex.models import Paragraph, Section, Document, QAResponse
from evidex.qa import explain_question
from evidex.llm import LLMInterface, MockLLM
from evidex.graph import (
    QAState,
    retrieve_paragraphs_node,
    explain_node,
    create_qa_graph,
    qa_graph,
    explain_question_graph,
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
    # LangGraph components
    "QAState",
    "retrieve_paragraphs_node",
    "explain_node",
    "create_qa_graph",
    "qa_graph",
    "explain_question_graph",
]
