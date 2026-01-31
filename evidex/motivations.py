"""
Author motivation extraction from research papers.

This module extracts EXPLICIT motivations stated by authors using
trigger phrases like "because", "to address", "in order to".

Rules:
- ONLY extract motivations that are explicitly stated
- No inferred reasoning
- Must have a clear trigger phrase

NOTE: Used to answer "why" questions about author decisions.
"""

import re
from dataclasses import dataclass, field

from evidex.models import Document, Paragraph


# =============================================================================
# Motivation Model
# =============================================================================

@dataclass
class Motivation:
    """An explicit author motivation extracted from text.
    
    Attributes:
        text: The motivation statement (the "why")
        trigger_phrase: The phrase that introduced the motivation
        full_sentence: The complete sentence containing the motivation
    """
    text: str
    trigger_phrase: str
    full_sentence: str


# =============================================================================
# Motivation Trigger Patterns
# =============================================================================

# Patterns that signal explicit author motivation
# Order matters: longer patterns first to avoid partial matches
MOTIVATION_TRIGGERS = [
    # Purpose/goal patterns
    r'in order to\b',
    r'so that\b',
    r'so as to\b',
    r'with the aim of\b',
    r'with the goal of\b',
    r'for the purpose of\b',
    r'with the purpose of\b',
    r'to enable\b',
    r'to allow\b',
    r'to achieve\b',
    r'to improve\b',
    r'to reduce\b',
    r'to avoid\b',
    r'to address\b',
    r'to solve\b',
    r'to overcome\b',
    r'to mitigate\b',
    r'to facilitate\b',
    r'to support\b',
    r'to counteract\b',
    r'to prevent\b',
    r'to ensure\b',
    
    # Reason patterns
    r'because\b',
    r'since\b(?=\s+(?:it|they|this|these|the|we|our|dividing|using|having))',  # "since dividing" but not "since 2014"
    r'due to\b',
    r'owing to\b',
    r'given that\b',
    
    # NOTE: "as" is intentionally excluded - too many false positives ("as well as", "as a result")
    
    # Motivation/rationale patterns
    r'the reason (?:is|being|for)\b',
    r'this is because\b',
    r'this allows\b',
    r'this enables\b',
    r'this ensures\b',
    r'this prevents\b',
    r'this helps\b',
    r'this makes\b',
    
    # Standalone allows/enables (subject + allows)
    r'(?:attention|model|approach|method|technique|architecture) allows\b',
    
    # Advantage/benefit patterns
    r'the advantage (?:is|of)\b',
    r'the benefit (?:is|of)\b',
    r'which allows\b',
    r'which enables\b',
    r'which ensures\b',
    r'which prevents\b',
    r'which helps\b',
    r'which makes\b',
    r'allowing\b',
    r'enabling\b',
    r'ensuring\b',
    r'preventing\b',
    
    # Problem/solution patterns
    r'to handle\b',
    r'to deal with\b',
    r'to cope with\b',
    r'rather than\b',
    r'instead of\b',
]

# Compile into a single pattern
_TRIGGER_PATTERN = re.compile(
    r'(' + '|'.join(MOTIVATION_TRIGGERS) + r')',
    re.IGNORECASE
)

# Pattern to split text into sentences
_SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


# =============================================================================
# Extraction Functions
# =============================================================================

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences.
    
    Args:
        text: The text to split
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting
    sentences = _SENTENCE_PATTERN.split(text)
    return [s.strip() for s in sentences if s.strip()]


def extract_motivations(text: str) -> list[Motivation]:
    """Extract explicit author motivations from text.
    
    Finds sentences containing motivation trigger phrases and
    extracts the motivation statement.
    
    ONLY extracts explicitly stated motivations.
    Does NOT infer reasoning.
    
    Args:
        text: The text to extract motivations from
        
    Returns:
        List of Motivation objects found
    """
    motivations = []
    sentences = _split_sentences(text)
    
    for sentence in sentences:
        # Find all trigger phrases in this sentence
        matches = list(_TRIGGER_PATTERN.finditer(sentence))
        
        for match in matches:
            trigger = match.group(1).lower().strip()
            
            # Extract the motivation text (what comes after the trigger)
            start_pos = match.end()
            motivation_text = sentence[start_pos:].strip()
            
            # Clean up: remove leading punctuation/whitespace
            motivation_text = re.sub(r'^[,\s]+', '', motivation_text)
            
            # Skip if motivation text is too short or empty
            if len(motivation_text) < 10:
                continue
            
            # Truncate at next sentence boundary within the text
            period_pos = motivation_text.find('.')
            if period_pos > 0:
                motivation_text = motivation_text[:period_pos + 1]
            
            motivations.append(Motivation(
                text=motivation_text.strip(),
                trigger_phrase=trigger,
                full_sentence=sentence.strip(),
            ))
    
    return motivations


def extract_motivations_as_list(text: str) -> list[dict]:
    """Extract motivations and return as list of dicts.
    
    Convenience function for JSON serialization.
    
    Args:
        text: The text to extract motivations from
        
    Returns:
        List of dicts with 'text', 'trigger_phrase', 'full_sentence' keys
    """
    motivations = extract_motivations(text)
    return [
        {
            "text": m.text,
            "trigger_phrase": m.trigger_phrase,
            "full_sentence": m.full_sentence,
        }
        for m in motivations
    ]


def has_motivation(text: str) -> bool:
    """Check if text contains any explicit motivation.
    
    Args:
        text: The text to check
        
    Returns:
        True if at least one motivation trigger is found
    """
    return bool(_TRIGGER_PATTERN.search(text))


# =============================================================================
# Document-level Functions
# =============================================================================

def extract_motivations_for_paragraph(paragraph: Paragraph) -> list[Motivation]:
    """Extract motivations from a single paragraph.
    
    Args:
        paragraph: The paragraph to process
        
    Returns:
        List of Motivation objects found
    """
    return extract_motivations(paragraph.text)


def extract_motivations_for_document(document: Document) -> dict[str, list[Motivation]]:
    """Extract motivations from all paragraphs in a document.
    
    Args:
        document: The document to process
        
    Returns:
        Dict mapping paragraph_id to list of Motivation objects
    """
    result = {}
    
    for section in document.sections:
        for para in section.paragraphs:
            motivations = extract_motivations(para.text)
            if motivations:
                result[para.paragraph_id] = motivations
    
    return result


def search_motivations(document: Document, keyword: str) -> list[tuple[str, Motivation]]:
    """Search for motivations related to a keyword.
    
    Finds motivations where the keyword appears in either the
    motivation text or the full sentence.
    
    Args:
        document: The document to search
        keyword: The keyword to search for (case-insensitive)
        
    Returns:
        List of (paragraph_id, Motivation) tuples
    """
    keyword_lower = keyword.lower()
    results = []
    
    motivations_by_para = extract_motivations_for_document(document)
    
    for para_id, motivations in motivations_by_para.items():
        for m in motivations:
            if keyword_lower in m.full_sentence.lower():
                results.append((para_id, m))
    
    return results


def get_motivation_summary(document: Document) -> dict:
    """Get a summary of all motivations in a document.
    
    Args:
        document: The document to analyze
        
    Returns:
        Dict with summary statistics and motivations grouped by trigger
    """
    all_motivations = extract_motivations_for_document(document)
    
    # Group by trigger phrase
    by_trigger: dict[str, list[str]] = {}
    total_count = 0
    
    for para_id, motivations in all_motivations.items():
        for m in motivations:
            trigger = m.trigger_phrase
            if trigger not in by_trigger:
                by_trigger[trigger] = []
            by_trigger[trigger].append(m.text)
            total_count += 1
    
    return {
        "total_motivations": total_count,
        "paragraphs_with_motivations": len(all_motivations),
        "by_trigger": by_trigger,
    }
