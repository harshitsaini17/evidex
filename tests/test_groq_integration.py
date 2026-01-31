"""
Integration tests using real Groq API.

These tests make actual API calls to Groq and verify the system
works end-to-end on the Attention paper.

Run with: pytest tests/test_groq_integration.py -v -s
"""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from evidex.ingest import parse_pdf_to_document, search_paragraphs, get_all_paragraph_ids
from evidex.llm import GroqLLM
from evidex.qa import explain_question


# Path to the test PDF
ATTENTION_PAPER_PATH = Path(__file__).parent.parent / "NIPS-2017-attention-is-all-you-need-Paper.pdf"


def has_groq_api_key() -> bool:
    """Check if Groq API key is available."""
    return bool(os.environ.get("GROQ_API_KEY"))


@pytest.fixture
def attention_paper():
    """Load the Attention paper as a Document."""
    if not ATTENTION_PAPER_PATH.exists():
        pytest.skip(f"Test PDF not found: {ATTENTION_PAPER_PATH}")
    return parse_pdf_to_document(ATTENTION_PAPER_PATH, title="Attention Is All You Need")


@pytest.fixture
def groq_llm():
    """Create a Groq LLM instance."""
    if not has_groq_api_key():
        pytest.skip("GROQ_API_KEY not set")
    return GroqLLM(
        model=os.environ.get("GROQ_MODEL", "llama3-70b-8192"),
        temperature=0.0,
    )


# =============================================================================
# Real API Integration Tests
# =============================================================================

@pytest.mark.skipif(not has_groq_api_key(), reason="GROQ_API_KEY not set")
class TestGroqIntegration:
    """Integration tests using real Groq API calls."""
    
    def test_attention_definition_question(self, attention_paper, groq_llm):
        """Test: 'How is attention defined?' using real Groq API.
        
        This is the key test - verifying real LLM behavior on a real paper.
        """
        # Find paragraphs about attention mechanism
        attention_ids = search_paragraphs(attention_paper, "attention")
        scaled_dot_ids = search_paragraphs(attention_paper, "scaled dot-product")
        query_key_ids = search_paragraphs(attention_paper, "query")
        
        # Combine relevant paragraphs (deduplicated)
        all_relevant = list(dict.fromkeys(attention_ids[:5] + scaled_dot_ids[:3] + query_key_ids[:3]))
        context_ids = all_relevant[:8]  # Limit context size
        
        print(f"\n{'='*60}")
        print(f"Question: How is attention defined?")
        print(f"Context paragraph IDs: {context_ids}")
        print(f"{'='*60}")
        
        # Show the actual paragraphs being used
        for pid in context_ids[:3]:
            para = attention_paper.get_paragraph(pid)
            if para:
                print(f"\n[{pid}]: {para.text[:200]}...")
        
        result = explain_question(
            document=attention_paper,
            paragraph_ids=context_ids,
            question="How is attention defined in this paper?",
            llm=groq_llm
        )
        
        print(f"\n{'='*60}")
        print(f"RESULT:")
        print(f"Answer: {result['answer']}")
        print(f"Citations: {result['citations']}")
        print(f"Confidence: {result['confidence']}")
        print(f"{'='*60}")
        
        # Verify we got a substantive answer
        assert result["answer"] != "Not defined in the paper", \
            "Attention should be defined in the attention paper!"
        
        # Verify citations are valid
        assert len(result["citations"]) > 0, "Should have citations"
        for cid in result["citations"]:
            assert cid in context_ids, f"Citation {cid} not in provided context"
        
        # Verify answer mentions key concepts from the paper
        answer_lower = result["answer"].lower()
        assert any(term in answer_lower for term in ["attention", "query", "key", "value", "scaled"]), \
            "Answer should mention attention-related concepts"
    
    def test_transformer_architecture_question(self, attention_paper, groq_llm):
        """Test asking about the Transformer architecture."""
        # Find relevant paragraphs
        transformer_ids = search_paragraphs(attention_paper, "transformer")
        encoder_ids = search_paragraphs(attention_paper, "encoder")
        decoder_ids = search_paragraphs(attention_paper, "decoder")
        
        context_ids = list(dict.fromkeys(
            transformer_ids[:3] + encoder_ids[:2] + decoder_ids[:2]
        ))[:7]
        
        print(f"\n{'='*60}")
        print(f"Question: What is the Transformer architecture?")
        print(f"Context paragraph IDs: {context_ids}")
        print(f"{'='*60}")
        
        result = explain_question(
            document=attention_paper,
            paragraph_ids=context_ids,
            question="What is the Transformer architecture and what are its main components?",
            llm=groq_llm
        )
        
        print(f"\n{'='*60}")
        print(f"RESULT:")
        print(f"Answer: {result['answer']}")
        print(f"Citations: {result['citations']}")
        print(f"Confidence: {result['confidence']}")
        print(f"{'='*60}")
        
        # Verify substantive answer
        assert result["answer"] != "Not defined in the paper"
        assert len(result["citations"]) > 0
        
        # Should mention encoder/decoder
        answer_lower = result["answer"].lower()
        assert any(term in answer_lower for term in ["encoder", "decoder", "attention", "layer"])
    
    def test_question_not_in_paper(self, attention_paper, groq_llm):
        """Test that questions about things not in the paper are rejected."""
        # Use some paragraphs but ask about something not there
        context_ids = get_all_paragraph_ids(attention_paper)[:5]
        
        print(f"\n{'='*60}")
        print(f"Question: What does the paper say about quantum computing?")
        print(f"Context paragraph IDs: {context_ids}")
        print(f"{'='*60}")
        
        result = explain_question(
            document=attention_paper,
            paragraph_ids=context_ids,
            question="What does this paper say about quantum computing applications?",
            llm=groq_llm
        )
        
        print(f"\n{'='*60}")
        print(f"RESULT:")
        print(f"Answer: {result['answer']}")
        print(f"Citations: {result['citations']}")
        print(f"Confidence: {result['confidence']}")
        print(f"{'='*60}")
        
        # Should indicate not in paper OR have low confidence
        # The verifier should catch if LLM tries to answer without citations
        if result["answer"] != "Not defined in the paper":
            # If LLM tried to answer, verifier should have caught it
            # (no citations = rejection)
            assert result["confidence"] == "low" or len(result["citations"]) == 0
    
    def test_bleu_score_question(self, attention_paper, groq_llm):
        """Test asking about BLEU scores mentioned in the paper."""
        # Find paragraphs about BLEU scores
        bleu_ids = search_paragraphs(attention_paper, "BLEU")
        
        if not bleu_ids:
            pytest.skip("No paragraphs found mentioning BLEU")
        
        context_ids = bleu_ids[:5]
        
        print(f"\n{'='*60}")
        print(f"Question: What BLEU scores did the model achieve?")
        print(f"Context paragraph IDs: {context_ids}")
        print(f"{'='*60}")
        
        result = explain_question(
            document=attention_paper,
            paragraph_ids=context_ids,
            question="What BLEU scores did the Transformer model achieve on translation tasks?",
            llm=groq_llm
        )
        
        print(f"\n{'='*60}")
        print(f"RESULT:")
        print(f"Answer: {result['answer']}")
        print(f"Citations: {result['citations']}")
        print(f"Confidence: {result['confidence']}")
        print(f"{'='*60}")
        
        # Should have a substantive answer about BLEU scores
        assert len(result["citations"]) > 0
        # Answer should contain numbers if discussing scores
        assert any(char.isdigit() for char in result["answer"]), \
            "BLEU score answer should include numbers"


# =============================================================================
# Quick Verification Script
# =============================================================================

if __name__ == "__main__":
    """Run a quick verification without pytest."""
    load_dotenv()
    
    if not has_groq_api_key():
        print("ERROR: GROQ_API_KEY not set in environment")
        exit(1)
    
    print("Loading Attention paper...")
    paper = parse_pdf_to_document(ATTENTION_PAPER_PATH, title="Attention Is All You Need")
    print(f"Loaded {len(paper.sections)} sections")
    
    total_paras = sum(len(s.paragraphs) for s in paper.sections)
    print(f"Total paragraphs: {total_paras}")
    
    print("\nInitializing Groq LLM...")
    llm = GroqLLM(
        model=os.environ.get("GROQ_MODEL", "llama3-70b-8192"),
        temperature=0.0,
    )
    
    # Test question
    print("\n" + "="*60)
    print("TEST: How is attention defined?")
    print("="*60)
    
    attention_ids = search_paragraphs(paper, "attention")
    scaled_ids = search_paragraphs(paper, "scaled dot-product")
    context_ids = list(dict.fromkeys(attention_ids[:5] + scaled_ids[:3]))[:8]
    
    print(f"Using {len(context_ids)} context paragraphs: {context_ids}")
    
    result = explain_question(
        document=paper,
        paragraph_ids=context_ids,
        question="How is attention defined in this paper?",
        llm=llm
    )
    
    print(f"\nANSWER: {result['answer']}")
    print(f"\nCITATIONS: {result['citations']}")
    print(f"CONFIDENCE: {result['confidence']}")
    
    # Verify the citations exist
    print("\n" + "="*60)
    print("CITATION VERIFICATION:")
    print("="*60)
    for cid in result["citations"]:
        para = paper.get_paragraph(cid)
        if para:
            print(f"\n[{cid}]: {para.text[:300]}...")
        else:
            print(f"\n[{cid}]: NOT FOUND!")
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
