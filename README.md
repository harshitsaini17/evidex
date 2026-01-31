# Evidex - Research Paper Understanding System

A system that answers questions **ONLY** using content from ingested documents, with strict citation requirements and no external knowledge.

## Key Principles

- **No External Knowledge**: The system NEVER introduces information not present in the document
- **Mandatory Citations**: Every answer must cite the specific paragraph IDs used
- **Explicit Rejection**: If a concept is not defined, the system explicitly says "Not defined in the paper"
- **Traceability Over Helpfulness**: Correctness and traceability matter more than being helpful

## Project Structure

```
evidex/
├── __init__.py          # Package exports
├── models.py            # Data models (Paragraph, Section, Document, QAResponse)
├── llm.py               # LLM interface and mock implementation
└── qa.py                # Core explain_question function
tests/
└── test_qa.py           # Unit tests
pyproject.toml           # Project configuration
```

## Installation

```bash
# Install in development mode
pip install -e ".[dev]"
```

## Usage

### Document Structure

Documents are represented as a hierarchy of sections containing paragraphs:

```python
from evidex import Document, Section, Paragraph

doc = Document(
    title="My Research Paper",
    sections=[
        Section(
            title="Introduction",
            paragraphs=[
                Paragraph(paragraph_id="s1_p1", text="First paragraph..."),
                Paragraph(paragraph_id="s1_p2", text="Second paragraph..."),
            ]
        ),
    ]
)
```

### Answering Questions

Use `explain_question` to answer questions using specific paragraphs:

```python
from evidex import explain_question, MockLLM

# For testing, use the MockLLM
llm = MockLLM(
    default_response=MockLLM.create_response(
        answer="The answer based on document content",
        citations=["s1_p1"],
        confidence="high"
    )
)

result = explain_question(
    document=doc,
    paragraph_ids=["s1_p1", "s1_p2"],
    question="What is the main topic?",
    llm=llm
)

# Result structure:
# {
#     "answer": "The answer based on document content",
#     "citations": ["s1_p1"],
#     "confidence": "high"
# }
```

### Response Format

The `explain_question` function returns a dictionary with:

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | The answer text, or "Not defined in the paper" |
| `citations` | list[str] | Paragraph IDs that support the answer |
| `confidence` | "high" \| "low" | System's confidence in the answer |

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=evidex

# Run specific test class
pytest tests/test_qa.py::TestExplainQuestion -v
```

## MVP Development Phases

This MVP is being built in three parts:

1. ✅ **Part 1**: Deterministic document parsing + Q&A (no agents) - CURRENT
2. ⬜ **Part 2**: Agentic workflow using LangGraph
3. ⬜ **Part 3**: Verification, guardrails, and evaluation hooks

## Design Decisions

### Why Mock LLM?

The `MockLLM` class allows testing the entire Q&A pipeline without external API calls:
- Deterministic test results
- No API costs during development
- Fast test execution
- Easy to simulate edge cases (missing info, low confidence, etc.)

### Why Validate Citations?

The system validates that returned citations match the provided paragraph IDs:
- Prevents LLM from hallucinating citations
- Ensures traceability to actual document content
- Makes the system auditable
