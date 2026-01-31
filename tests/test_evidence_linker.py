"""
Unit tests for the EvidenceLinker node.

Tests that evidence (paragraphs and equations) are correctly linked
based on shared entities (variables and concepts).

The EvidenceLinker:
- Does NOT call an LLM
- Does NOT generate prose
- Only groups/links existing evidence based on shared entities
"""

import pytest
from pathlib import Path

from evidex.models import Paragraph, Section, Document, Equation, Entities
from evidex.llm import MockLLM
from evidex.graph import (
    QAState,
    evidence_linker_node,
    create_qa_graph,
    retrieve_paragraphs_node,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def attention_document() -> Document:
    """Create a document about attention with equations and entities."""
    return Document(
        title="Attention Is All You Need",
        sections=[
            Section(
                title="Attention",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s1_p1",
                        text="An attention function maps a query Q and key-value pairs K, V to an output.",
                        entities=Entities(
                            variables=["Q", "K", "V"],
                            concepts=["attention", "query"],
                        ),
                    ),
                    Paragraph(
                        paragraph_id="s1_p2",
                        text="We compute the attention using scaled dot-product with softmax.",
                        entities=Entities(
                            variables=[],
                            concepts=["attention", "softmax"],
                        ),
                    ),
                    Paragraph(
                        paragraph_id="s1_p3",
                        text="The dimension d_k determines the scaling factor.",
                        entities=Entities(
                            variables=["d_k"],
                            concepts=[],
                        ),
                    ),
                ]
            ),
            Section(
                title="Results",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s2_p1",
                        text="Our model achieves a BLEU score of 28.4 on the WMT 2014 task.",
                        entities=Entities(
                            variables=[],
                            concepts=["bleu", "bleu score"],
                        ),
                    ),
                    Paragraph(
                        paragraph_id="s2_p2",
                        text="The evaluation shows strong performance on translation quality.",
                        entities=Entities(
                            variables=[],
                            concepts=["bleu"],
                        ),
                    ),
                ]
            ),
        ],
        equations=[
            Equation(
                equation_id="eq1",
                equation_text="Attention(Q, K, V) = softmax(QK^T / √d_k)V",
                associated_paragraph_id="s1_p1",
            ),
        ],
    )


@pytest.fixture
def unrelated_document() -> Document:
    """Create a document with unrelated paragraphs (no shared entities)."""
    return Document(
        title="Unrelated Topics",
        sections=[
            Section(
                title="Topic A",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s1_p1",
                        text="This paragraph discusses topic A exclusively.",
                        entities=Entities(
                            variables=["X"],
                            concepts=["topic_a"],
                        ),
                    ),
                ]
            ),
            Section(
                title="Topic B",
                paragraphs=[
                    Paragraph(
                        paragraph_id="s2_p1",
                        text="This paragraph discusses topic B exclusively.",
                        entities=Entities(
                            variables=["Y"],
                            concepts=["topic_b"],
                        ),
                    ),
                ]
            ),
        ],
        equations=[],
    )


# =============================================================================
# Basic EvidenceLinker Tests
# =============================================================================

class TestEvidenceLinkerBasic:
    """Basic tests for the evidence_linker_node function."""
    
    def test_returns_linked_evidence_key(self, attention_document: Document):
        """Test that the node returns a dict with linked_evidence key."""
        state: QAState = {
            "document": attention_document,
            "paragraphs": attention_document.sections[0].paragraphs[:2],
            "equations": [],
            "llm": MockLLM(),
        }
        
        result = evidence_linker_node(state)
        
        assert "linked_evidence" in result
        assert isinstance(result["linked_evidence"], list)
    
    def test_empty_paragraphs_returns_empty_list(self):
        """Test that empty input returns empty linked_evidence."""
        state: QAState = {
            "paragraphs": [],
            "equations": [],
        }
        
        result = evidence_linker_node(state)
        
        assert result["linked_evidence"] == []
    
    def test_single_paragraph_returns_empty_list(self, attention_document: Document):
        """Test that a single piece of evidence returns empty list (no links)."""
        state: QAState = {
            "paragraphs": [attention_document.sections[0].paragraphs[0]],
            "equations": [],
        }
        
        result = evidence_linker_node(state)
        
        # Need at least 2 pieces of evidence to form a link
        assert result["linked_evidence"] == []


# =============================================================================
# Attention Equation ↔ Attention Paragraph Tests
# =============================================================================

class TestAttentionEquationParagraphLink:
    """Tests for linking attention equation with attention paragraph."""
    
    def test_links_attention_equation_to_paragraph(self, attention_document: Document):
        """Test that attention equation is linked to attention paragraph via shared Q, K, V."""
        # Get the attention paragraph and equation
        attention_para = attention_document.sections[0].paragraphs[0]  # Contains Q, K, V
        attention_eq = attention_document.equations[0]  # Attention(Q, K, V) = ...
        
        state: QAState = {
            "paragraphs": [attention_para],
            "equations": [attention_eq],
        }
        
        result = evidence_linker_node(state)
        
        # Should have exactly one link
        assert len(result["linked_evidence"]) == 1
        
        link = result["linked_evidence"][0]
        # Both should be in source_ids
        assert "s1_p1" in link["source_ids"]
        assert "eq1" in link["source_ids"]
        
        # Should share Q, K, V variables
        shared_vars = link["shared_entities"]["variables"]
        assert "Q" in shared_vars
        assert "K" in shared_vars
        assert "V" in shared_vars
    
    def test_links_via_attention_concept(self, attention_document: Document):
        """Test that paragraphs sharing 'attention' concept are linked."""
        # Paragraphs s1_p1 and s1_p2 both have 'attention' concept
        para1 = attention_document.sections[0].paragraphs[0]  # attention concept
        para2 = attention_document.sections[0].paragraphs[1]  # attention concept
        
        state: QAState = {
            "paragraphs": [para1, para2],
            "equations": [],
        }
        
        result = evidence_linker_node(state)
        
        assert len(result["linked_evidence"]) == 1
        
        link = result["linked_evidence"][0]
        assert "s1_p1" in link["source_ids"]
        assert "s1_p2" in link["source_ids"]
        assert "attention" in link["shared_entities"]["concepts"]


# =============================================================================
# BLEU Score ↔ Evaluation Paragraph Tests
# =============================================================================

class TestBLEUEvaluationLink:
    """Tests for linking BLEU score paragraphs with evaluation paragraphs."""
    
    def test_links_bleu_paragraphs(self, attention_document: Document):
        """Test that BLEU-related paragraphs are linked via shared concept."""
        # s2_p1 has 'bleu', 'bleu score'; s2_p2 has 'bleu'
        bleu_para1 = attention_document.sections[1].paragraphs[0]
        bleu_para2 = attention_document.sections[1].paragraphs[1]
        
        state: QAState = {
            "paragraphs": [bleu_para1, bleu_para2],
            "equations": [],
        }
        
        result = evidence_linker_node(state)
        
        assert len(result["linked_evidence"]) == 1
        
        link = result["linked_evidence"][0]
        assert "s2_p1" in link["source_ids"]
        assert "s2_p2" in link["source_ids"]
        assert "bleu" in link["shared_entities"]["concepts"]


# =============================================================================
# No Links for Unrelated Questions Tests
# =============================================================================

class TestNoLinksUnrelated:
    """Tests that unrelated evidence does not get linked."""
    
    def test_no_links_for_unrelated_paragraphs(self, unrelated_document: Document):
        """Test that paragraphs with no shared entities are not linked."""
        para1 = unrelated_document.sections[0].paragraphs[0]  # topic_a, X
        para2 = unrelated_document.sections[1].paragraphs[0]  # topic_b, Y
        
        state: QAState = {
            "paragraphs": [para1, para2],
            "equations": [],
        }
        
        result = evidence_linker_node(state)
        
        # No shared entities, so no links
        assert result["linked_evidence"] == []
    
    def test_no_links_attention_to_bleu(self, attention_document: Document):
        """Test that attention paragraphs are not linked to BLEU paragraphs."""
        attention_para = attention_document.sections[0].paragraphs[0]  # Q, K, V, attention
        bleu_para = attention_document.sections[1].paragraphs[0]  # bleu
        
        state: QAState = {
            "paragraphs": [attention_para, bleu_para],
            "equations": [],
        }
        
        result = evidence_linker_node(state)
        
        # No shared entities between attention and BLEU
        assert result["linked_evidence"] == []
    
    def test_d_k_only_paragraph_not_linked_to_bleu(self, attention_document: Document):
        """Test that d_k paragraph is not linked to BLEU paragraph."""
        dk_para = attention_document.sections[0].paragraphs[2]  # d_k only
        bleu_para = attention_document.sections[1].paragraphs[0]  # bleu only
        
        state: QAState = {
            "paragraphs": [dk_para, bleu_para],
            "equations": [],
        }
        
        result = evidence_linker_node(state)
        
        assert result["linked_evidence"] == []


# =============================================================================
# Entity Extraction On-The-Fly Tests
# =============================================================================

class TestOnTheFlyExtraction:
    """Tests that entities are extracted on-the-fly if not pre-extracted."""
    
    def test_extracts_entities_when_not_present(self):
        """Test that entities are extracted from text if not pre-extracted."""
        # Create paragraphs WITHOUT pre-extracted entities
        para1 = Paragraph(
            paragraph_id="p1",
            text="The attention function uses Q and K matrices.",
            entities=None,  # No pre-extracted entities
        )
        para2 = Paragraph(
            paragraph_id="p2",
            text="We compute Q times K transpose for attention.",
            entities=None,
        )
        
        state: QAState = {
            "paragraphs": [para1, para2],
            "equations": [],
        }
        
        result = evidence_linker_node(state)
        
        # Should still find links via Q and K extracted on-the-fly
        assert len(result["linked_evidence"]) == 1
        link = result["linked_evidence"][0]
        assert "Q" in link["shared_entities"]["variables"]
        assert "K" in link["shared_entities"]["variables"]
    
    def test_extracts_entities_from_equation_text(self, attention_document: Document):
        """Test that entities are extracted from equation text."""
        # Use only the equation (no paragraphs)
        eq = attention_document.equations[0]  # Contains Q, K, V, d_k, softmax
        
        # Create a paragraph with Q, K, V to link with
        para = Paragraph(
            paragraph_id="p1",
            text="The query Q and key K are used in attention.",
            entities=None,
        )
        
        state: QAState = {
            "paragraphs": [para],
            "equations": [eq],
        }
        
        result = evidence_linker_node(state)
        
        # Should link via Q and K
        assert len(result["linked_evidence"]) == 1
        link = result["linked_evidence"][0]
        assert "p1" in link["source_ids"]
        assert "eq1" in link["source_ids"]


# =============================================================================
# Multiple Links Tests
# =============================================================================

class TestMultipleLinks:
    """Tests for scenarios with multiple independent link groups."""
    
    def test_creates_separate_link_groups(self, attention_document: Document):
        """Test that independent groups are created correctly."""
        # Group 1: attention paragraphs (share 'attention' concept)
        # Group 2: BLEU paragraphs (share 'bleu' concept)
        # These should form 2 separate link groups
        
        attention_para1 = attention_document.sections[0].paragraphs[0]  # attention
        attention_para2 = attention_document.sections[0].paragraphs[1]  # attention
        bleu_para1 = attention_document.sections[1].paragraphs[0]  # bleu
        bleu_para2 = attention_document.sections[1].paragraphs[1]  # bleu
        
        state: QAState = {
            "paragraphs": [attention_para1, attention_para2, bleu_para1, bleu_para2],
            "equations": [],
        }
        
        result = evidence_linker_node(state)
        
        # Should have 2 link groups
        assert len(result["linked_evidence"]) == 2
        
        # Verify the groups are correct
        group_ids = [set(link["source_ids"]) for link in result["linked_evidence"]]
        
        attention_group = {"s1_p1", "s1_p2"}
        bleu_group = {"s2_p1", "s2_p2"}
        
        assert attention_group in group_ids
        assert bleu_group in group_ids


# =============================================================================
# Graph Integration Tests
# =============================================================================

class TestEvidenceLinkerInGraph:
    """Tests for evidence_linker_node integration in the full graph."""
    
    def test_graph_includes_evidence_linker(self):
        """Test that the graph includes the evidence_linker node."""
        graph = create_qa_graph()
        
        # The graph should be able to invoke
        assert graph is not None
    
    def test_full_graph_returns_linked_evidence(self, attention_document: Document):
        """Test that the full graph workflow returns linked_evidence."""
        mock_llm = MockLLM(
            default_response=MockLLM.create_response(
                answer="Attention uses Q, K, V matrices.",
                citations=["s1_p1"],
                confidence="high"
            )
        )
        
        graph = create_qa_graph()
        
        initial_state: QAState = {
            "document": attention_document,
            "question": "How is attention computed?",
            "llm": mock_llm,
        }
        
        result = graph.invoke(initial_state)
        
        # Should have linked_evidence in the result
        assert "linked_evidence" in result
        assert isinstance(result["linked_evidence"], list)


# =============================================================================
# Real PDF Integration Tests
# =============================================================================

ATTENTION_PAPER_PATH = Path(__file__).parent.parent / "NIPS-2017-attention-is-all-you-need-Paper.pdf"


@pytest.mark.skipif(
    not ATTENTION_PAPER_PATH.exists(),
    reason="Attention paper PDF not found"
)
class TestEvidenceLinkerRealPDF:
    """Integration tests for evidence linking with the real Attention paper."""
    
    @pytest.fixture
    def attention_paper(self) -> Document:
        """Load the real Attention paper."""
        from evidex.ingest import parse_pdf_to_document
        from evidex.entities import extract_entities_for_document
        
        doc = parse_pdf_to_document(
            ATTENTION_PAPER_PATH,
            title="Attention Is All You Need",
            extract_equations=True,
        )
        # Extract entities for all paragraphs
        extract_entities_for_document(doc)
        return doc
    
    def test_finds_attention_related_links(self, attention_paper: Document):
        """Test that attention-related evidence is linked in the real paper."""
        # Get some attention-related paragraphs
        from evidex.ingest import search_paragraphs
        
        attention_ids = search_paragraphs(attention_paper, "attention")[:5]
        paragraphs = attention_paper.get_paragraphs(attention_ids)
        
        state: QAState = {
            "paragraphs": paragraphs,
            "equations": attention_paper.equations[:3] if attention_paper.equations else [],
        }
        
        result = evidence_linker_node(state)
        
        # Should find some links (attention is a common concept)
        # But we don't assert specific structure since it depends on PDF parsing
        assert isinstance(result["linked_evidence"], list)
