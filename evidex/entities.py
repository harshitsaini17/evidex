"""
Entity extraction for GraphRAG preparation.

This module provides simple heuristic-based entity extraction from text.
Extracts:
- variables: Mathematical variables like Q, K, V, d_k, d_model
- concepts: Domain concepts like attention, transformer, encoder

NOTE: These entities are NOT used for reasoning yet.
This is scaffolding for future GraphRAG integration.
"""

import re
from evidex.models import Entities


# =============================================================================
# Variable Extraction
# =============================================================================

# Common mathematical variables in ML/attention papers
# Pattern matches: single uppercase letters, subscripted vars (d_k), 
# common notation (W^Q, W_i^K), dimension variables
VARIABLE_PATTERNS = [
    # Subscripted variables: d_k, d_v, d_model, d_ff
    r'\bd_(?:k|v|model|ff)\b',
    # Single uppercase letters that are variables: Q, K, V, W, X, Y, Z
    r'\b[QKVWXYZ]\b',
    # Superscripted/subscripted weight matrices: W^Q, W^K, W^V, W^O, W_i
    r'\bW(?:\^[QKVO]|_[io0-9])*\b',
    # Head notation: head_i, head_1
    r'\bhead_?[_i0-9]*\b',
    # Position encoding: PE
    r'\bPE\b',
    # Sequence length: n
    r'\bn\b(?=\s*[,.\)]|\s+(?:is|are|represents?))',
    # Hidden dimension: h
    r'\bh\b(?=\s*(?:heads?|=|,))',
    # Layer indices
    r'\blayer_?[0-9]*\b',
]

# Compile patterns for efficiency
_VARIABLE_REGEX = re.compile(
    '|'.join(f'({p})' for p in VARIABLE_PATTERNS),
    re.IGNORECASE
)


def extract_variables(text: str) -> list[str]:
    """Extract mathematical variables from text.
    
    Identifies common ML/attention paper variables like Q, K, V, d_k, etc.
    
    Args:
        text: The text to extract variables from
        
    Returns:
        List of unique variable names found (preserving first occurrence order)
    """
    # Find all matches
    matches = _VARIABLE_REGEX.findall(text)
    
    # Flatten the tuple groups and filter empty strings
    found = []
    seen = set()
    for match_tuple in matches:
        for m in match_tuple:
            if m and m not in seen:
                # Normalize: keep original case but dedupe case-insensitively
                normalized = m.strip()
                if normalized.lower() not in {v.lower() for v in seen}:
                    found.append(normalized)
                    seen.add(normalized)
    
    return found


# =============================================================================
# Concept Extraction
# =============================================================================

# Domain concepts common in ML/attention papers
CONCEPT_KEYWORDS = {
    # Attention mechanisms
    'attention', 'self-attention', 'self attention', 'multi-head attention',
    'multi-head', 'multihead', 'scaled dot-product', 'dot-product attention',
    'cross-attention', 'cross attention',
    
    # Architecture components
    'transformer', 'encoder', 'decoder', 'layer', 'sublayer', 'sub-layer',
    'embedding', 'embeddings', 'positional encoding', 'position encoding',
    'feed-forward', 'feedforward', 'ffn', 'residual connection', 'residual',
    'layer normalization', 'layer norm', 'layernorm', 'dropout',
    
    # Operations
    'softmax', 'linear projection', 'projection', 'concatenation', 'concat',
    'matrix multiplication', 'dot product', 'weighted sum',
    
    # Training concepts
    'training', 'inference', 'regularization', 'label smoothing',
    'learning rate', 'warmup', 'optimizer', 'adam', 'loss', 'cross-entropy',
    
    # Evaluation
    'bleu', 'bleu score', 'perplexity', 'accuracy', 'f1', 'precision', 'recall',
    
    # Data
    'sequence', 'token', 'tokens', 'vocabulary', 'batch', 'batch size',
    'input', 'output', 'query', 'key', 'value', 'mask', 'padding',
}

# Build regex pattern for concept matching (word boundaries, case insensitive)
_CONCEPT_PATTERNS = sorted(CONCEPT_KEYWORDS, key=len, reverse=True)  # Longest first
_CONCEPT_REGEX = re.compile(
    r'\b(' + '|'.join(re.escape(c) for c in _CONCEPT_PATTERNS) + r')\b',
    re.IGNORECASE
)


def extract_concepts(text: str) -> list[str]:
    """Extract domain concepts from text.
    
    Identifies ML/attention domain concepts like attention, transformer, etc.
    
    Args:
        text: The text to extract concepts from
        
    Returns:
        List of unique concept names found (normalized to lowercase)
    """
    matches = _CONCEPT_REGEX.findall(text)
    
    # Dedupe and normalize to lowercase
    seen = set()
    found = []
    for m in matches:
        normalized = m.lower().strip()
        if normalized not in seen:
            found.append(normalized)
            seen.add(normalized)
    
    return found


# =============================================================================
# Combined Entity Extraction
# =============================================================================

def extract_entities(text: str) -> dict:
    """Extract all entities from text.
    
    This is the main entry point for entity extraction.
    Returns a dict with 'variables' and 'concepts' lists.
    
    NOTE: Entities are NOT used for reasoning yet.
    This is scaffolding for future GraphRAG integration.
    
    Args:
        text: The text to extract entities from
        
    Returns:
        Dict with keys:
        - variables: List of mathematical variables found
        - concepts: List of domain concepts found
    """
    return {
        'variables': extract_variables(text),
        'concepts': extract_concepts(text),
    }


def extract_entities_as_model(text: str) -> Entities:
    """Extract entities and return as Entities model.
    
    Convenience function that returns an Entities dataclass
    instead of a dict.
    
    Args:
        text: The text to extract entities from
        
    Returns:
        Entities model with variables and concepts
    """
    result = extract_entities(text)
    return Entities(
        variables=result['variables'],
        concepts=result['concepts'],
    )


# =============================================================================
# Document-level Entity Extraction
# =============================================================================

def extract_entities_for_document(document: 'Document') -> None:
    """Extract entities for all paragraphs in a document.
    
    Modifies the document in place, setting the entities field
    on each paragraph.
    
    NOTE: Entities are NOT used for reasoning yet.
    This is scaffolding for future GraphRAG integration.
    
    Args:
        document: The Document to extract entities for (modified in place)
    """
    from evidex.models import Document  # Avoid circular import at module level
    
    for section in document.sections:
        for paragraph in section.paragraphs:
            paragraph.entities = extract_entities_as_model(paragraph.text)


def get_all_variables(document: 'Document') -> list[str]:
    """Get all unique variables across a document.
    
    Args:
        document: The Document to get variables from
        
    Returns:
        List of unique variable names (preserving first occurrence order)
    """
    seen = set()
    variables = []
    for section in document.sections:
        for paragraph in section.paragraphs:
            if paragraph.entities:
                for var in paragraph.entities.variables:
                    if var not in seen:
                        seen.add(var)
                        variables.append(var)
    return variables


def get_all_concepts(document: 'Document') -> list[str]:
    """Get all unique concepts across a document.
    
    Args:
        document: The Document to get concepts from
        
    Returns:
        List of unique concept names (preserving first occurrence order)
    """
    seen = set()
    concepts = []
    for section in document.sections:
        for paragraph in section.paragraphs:
            if paragraph.entities:
                for concept in paragraph.entities.concepts:
                    if concept not in seen:
                        seen.add(concept)
                        concepts.append(concept)
    return concepts
