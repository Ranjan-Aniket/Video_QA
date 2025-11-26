"""
Centralized Validation Patterns

Shared regex patterns for detecting quality issues across all modules:
- Hedging language (appears, seems, might, etc.)
- Pronouns without clear context (he, she, they, etc.)
- Proper names (person names, brands, teams, etc.)

Used by:
- hedging_fixer.py
- question_quality_checker.py
- phase8_vision_generator.py
- pass2a_sonnet_selector.py
- pass2b_opus_selector.py
"""

from typing import List

# ============================================================================
# HEDGING PATTERNS
# ============================================================================
# Detect uncertain/hedging language in questions, answers, and cues.
# Per Guidelines: Use definitive language instead of hedging.
#
# ✅ GOOD: "shows", "displays", "contains", "indicates"
# ❌ BAD: "appears to", "seems to", "might be", "probably"
# ============================================================================

HEDGING_PATTERNS: List[str] = [
    # Appearance-based hedging
    r'\bappears?\s+to\b',        # "appears to", "appear to"
    r'\bappears?\b',             # "appears", "appear" (standalone)
    r'\bseems?\s+to\b',          # "seems to", "seem to"
    r'\bseems?\b',               # "seems", "seem" (standalone)
    r'\blooks?\s+like\b',        # "looks like", "look like"

    # Modal hedging
    r'\bcould\s+be\b',           # "could be"
    r'\bmay\s+be\b',             # "may be"
    r'\bmight\s+be\b',           # "might be"
    r'\bmight\b',                # "might" (standalone)
    r'\bmay\b',                  # "may" (standalone)
    r'\bmaybe\b',                # "maybe"

    # Probability hedging
    r'\blikely\b',               # "likely"
    r'\bprobably\b',             # "probably"
    r'\bpossibly\b',             # "possibly"
    r'\bperhaps\b',              # "perhaps"

    # Suggestion-based hedging
    r'\bsuggest(s|ing|ed)?\b',   # "suggests", "suggesting", "suggested"

    # Note: "indicates" is NOT included - it's more definitive than "suggests"
]


# ============================================================================
# PRONOUN PATTERNS
# ============================================================================
# Detect pronouns without clear antecedents in questions and answers.
# Per Guidelines: Use descriptors ("person", "speaker", "individual")
# instead of pronouns.
#
# ✅ GOOD: "person", "speaker", "individual", "woman", "man"
# ❌ BAD: "he", "she", "they", "him", "her"
# ============================================================================

PRONOUN_PATTERNS: List[str] = [
    # Subject pronouns
    r'\bhe\b',
    r'\bshe\b',
    r'\bthey\b',

    # Object pronouns
    r'\bhim\b',
    r'\bher\b',
    r'\bthem\b',

    # Possessive pronouns
    r'\bhis\b',
    r'\bhers\b',
    r'\btheir\b',
    r'\btheirs\b',
]


# ============================================================================
# VAGUE REFERENCE PATTERNS (for question_quality_checker.py)
# ============================================================================
# Detect vague references without sufficient context.
# ============================================================================

VAGUE_REFERENCE_PATTERNS: List[str] = [
    r'\bthe figure\b(?! with | in )',   # "the figure" without descriptor
    r'\bthe model\b(?! with | in )',    # "the model" without descriptor
    r'\bthe person\b(?! with | in | who | wearing )',  # "the person" without descriptor
    r'\bthe object\b(?! with | in )',   # "the object" without descriptor
    r'\bit\b(?! is | was | appears | shows)',  # "it" without clear antecedent
]


# ============================================================================
# PROPER NAME PATTERNS
# ============================================================================
# Detect proper names (person names, brands, teams, etc.)
# Per Guidelines: Use generic descriptors instead of proper names.
#
# ✅ GOOD: "player", "brand logo", "team", "character"
# ❌ BAD: "LeBron James", "Nike", "Lakers", "Harry Potter"
# ============================================================================

NAME_PATTERNS: List[str] = [
    # Common first names with action verbs
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:says|mentions|performs|holds|wears|stands|moves)',

    # Possessive names
    r'\b[A-Z][a-z]+(?:\'s|s\')\s+(?:hand|arm|action|position|movement)',

    # Team/Brand names (extend as needed)
    r'\b(?:LEGO|Nike|Adidas|Lakers|Warriors|Ferrari|Tesla|Apple|Google|Microsoft|Amazon)\b',

    # Title + Name
    r'\bMr\.\s+[A-Z][a-z]+\b',
    r'\bMrs\.\s+[A-Z][a-z]+\b',
    r'\bMs\.\s+[A-Z][a-z]+\b',
    r'\bDr\.\s+[A-Z][a-z]+\b',

    # Character/Person names in questions
    r'\b(?:when|where|what|how|why)\s+[A-Z][a-z]+\s+(?:does|is|moves|says|performs)',

    # Mid-sentence capitalized words (likely proper nouns)
    r'(?<=[.!?]\s)[A-Z][a-z]+\s+(?:is|does|has|moves|says|performs)',
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_quality_patterns():
    """
    Get all quality validation patterns as a dict.

    Returns:
        Dict with pattern categories
    """
    return {
        'hedging': HEDGING_PATTERNS,
        'pronouns': PRONOUN_PATTERNS,
        'vague_references': VAGUE_REFERENCE_PATTERNS,
        'names': NAME_PATTERNS,
    }


def get_pattern_count():
    """
    Get count of patterns in each category.

    Returns:
        Dict with pattern counts
    """
    return {
        'hedging': len(HEDGING_PATTERNS),
        'pronouns': len(PRONOUN_PATTERNS),
        'vague_references': len(VAGUE_REFERENCE_PATTERNS),
        'names': len(NAME_PATTERNS),
    }
