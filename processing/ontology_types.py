"""
Centralized Ontology Type Definitions

Maps official PDF type names to internal aliases and provides
normalization to ensure consistency across Pass 2A, 2B, 3, Validator, and Phase 8.

Official Types (from PDF):
1. Temporal Understanding
2. Sequential
3. Subscene
4. General Holistic Reasoning
5. Inference
6. Context
7. Needle
8. Referential Grounding
9. Counting
10. Comparative
11. Object Interaction Reasoning
12. Audio-Visual Stitching
13. Tackling Spurious Correlations
"""

from typing import Dict, List

# Official type names from PDF (ground truth)
OFFICIAL_TYPES = [
    "Temporal Understanding",
    "Sequential",
    "Subscene",
    "General Holistic Reasoning",
    "Inference",
    "Context",
    "Needle",
    "Referential Grounding",
    "Counting",
    "Comparative",
    "Object Interaction Reasoning",
    "Audio-Visual Stitching",
    "Tackling Spurious Correlations"
]

# Mapping of aliases to official names
TYPE_ALIASES: Dict[str, str] = {
    # Temporal Understanding
    "Temporal": "Temporal Understanding",
    "temporal": "Temporal Understanding",
    "temporal_understanding": "Temporal Understanding",

    # Sequential (no aliases needed)
    "Sequential": "Sequential",
    "sequential": "Sequential",

    # Subscene (no aliases needed)
    "Subscene": "Subscene",
    "subscene": "Subscene",

    # General Holistic Reasoning
    "Holistic": "General Holistic Reasoning",
    "holistic": "General Holistic Reasoning",
    "General Holistic Reasoning": "General Holistic Reasoning",

    # Inference (no aliases needed)
    "Inference": "Inference",
    "inference": "Inference",

    # Context (no aliases needed)
    "Context": "Context",
    "context": "Context",

    # Needle (no aliases needed)
    "Needle": "Needle",
    "needle": "Needle",

    # Referential Grounding
    "Referential": "Referential Grounding",
    "referential": "Referential Grounding",
    "Referential Grounding": "Referential Grounding",

    # Counting (no aliases needed)
    "Counting": "Counting",
    "counting": "Counting",

    # Comparative (no aliases needed)
    "Comparative": "Comparative",
    "comparative": "Comparative",

    # Object Interaction Reasoning
    "ObjectInteraction": "Object Interaction Reasoning",
    "object_interaction": "Object Interaction Reasoning",
    "Object Interaction Reasoning": "Object Interaction Reasoning",

    # Audio-Visual Stitching
    "AVStitching": "Audio-Visual Stitching",
    "av_stitching": "Audio-Visual Stitching",
    "Audio-Visual Stitching": "Audio-Visual Stitching",

    # Tackling Spurious Correlations
    "Spurious": "Tackling Spurious Correlations",
    "spurious": "Tackling Spurious Correlations",
    "Tackling Spurious Correlations": "Tackling Spurious Correlations",
}


def normalize_type(raw_type: str) -> str:
    """
    Normalize a type name to official PDF naming.

    Args:
        raw_type: Type name from any source (Pass 2A/2B/3/Validator/Phase8)

    Returns:
        Official type name from PDF, or original if not found

    Examples:
        >>> normalize_type("Temporal")
        "Temporal Understanding"
        >>> normalize_type("ObjectInteraction")
        "Object Interaction Reasoning"
    """
    return TYPE_ALIASES.get(raw_type, raw_type)


def get_short_name(official_type: str) -> str:
    """
    Get the short alias for an official type name.

    Args:
        official_type: Official type name from PDF

    Returns:
        Short alias (for backward compatibility with existing code)

    Examples:
        >>> get_short_name("Temporal Understanding")
        "Temporal"
        >>> get_short_name("Object Interaction Reasoning")
        "ObjectInteraction"
    """
    reverse_map = {
        "Temporal Understanding": "Temporal",
        "General Holistic Reasoning": "Holistic",
        "Referential Grounding": "Referential",
        "Object Interaction Reasoning": "ObjectInteraction",
        "Audio-Visual Stitching": "AVStitching",
        "Tackling Spurious Correlations": "Spurious",
    }
    return reverse_map.get(official_type, official_type)


# Minimum temporal windows by type (from Guidelines PDF)
# These define the minimum time span (start_timestamp to end_timestamp)
# required for each question type to have sufficient context.
MIN_TEMPORAL_WINDOWS = {
    # Very high complexity (40-60 seconds) - requires extended temporal reasoning
    "Comparative": 40.0,
    "Tackling Spurious Correlations": 40.0,

    # High complexity (30-40 seconds) - requires multi-step temporal reasoning
    "Sequential": 30.0,
    "Temporal Understanding": 30.0,
    "Subscene": 30.0,
    "Object Interaction Reasoning": 30.0,

    # Moderate complexity (20-25 seconds) - requires context integration
    "Inference": 20.0,
    "Context": 20.0,
    "General Holistic Reasoning": 20.0,
    "Audio-Visual Stitching": 20.0,

    # Simple observation (10-15 seconds) - single-moment observation
    "Needle": 10.0,
    "Counting": 10.0,
    "Referential Grounding": 10.0,
}


# ✅ OFFICIAL Sub-task type assignments (from Question Types_ Skills.pdf)
# These 8 categories are the authoritative taxonomy defined in the official PDF.
# DO NOT modify without consulting the official Question Types PDF.
SUB_TASK_TYPES = {
    "Human Behavior Understanding": [
        "Object Interaction Reasoning"  # Actions performed on objects and their effects
    ],
    "Scene Recognition": [
        "Context",    # Background/setting elements
        "Subscene"    # Captioning relevant segments of video
    ],
    "OCR Recognition": [
        "Needle",                # Text, numbers, small visual details
        "Referential Grounding"  # When text/OCR syncs with audio cues
    ],
    "Causal Reasoning": [
        "Inference",   # Unstated purposes, intentions, cause-effect
        "Sequential"   # Ordered events with causal relationships
    ],
    "Intent Understanding": [
        # No direct main type mapping - detected from question content keywords
        # Examples: "purpose", "goal", "trying to", "intends", "aims"
    ],
    "Hallucination": [
        # ✅ OFFICIAL NAME per PDF - tests model's ability to avoid fabricating false events
        "Tackling Spurious Correlations"  # Unexpected/unintuitive events
    ],
    "Multi-Detail Understanding": [
        "Temporal Understanding",       # Multiple temporal details (before/after/when)
        "Comparative",                  # Comparing multiple elements
        "General Holistic Reasoning",   # Entire video comprehension
        "Counting",                     # Multiple instances/occurrences
        "Audio-Visual Stitching"        # Multiple spliced clips/segments
    ]
}


def get_min_temporal_window(ontology_type: str) -> float:
    """
    Get minimum temporal window (in seconds) for a question type.

    Per Guidelines PDF requirements:
    - Simple observation: 10-15s
    - Moderate complexity: 20-25s
    - High complexity: 30-40s
    - Very high complexity: 40-60s

    Args:
        ontology_type: Official ontology type name

    Returns:
        Minimum window duration in seconds (default 10.0)
    """
    return MIN_TEMPORAL_WINDOWS.get(ontology_type, 10.0)


def get_sub_task_type(ontology_type: str) -> str:
    """
    Get the sub-task type for a given ontology type.

    Args:
        ontology_type: Official ontology type name

    Returns:
        Sub-task type category
    """
    for sub_task, types in SUB_TASK_TYPES.items():
        if ontology_type in types:
            return sub_task
    return "Unknown"


# ===============================================================================
# MODE CONFIGURATION (Single Source of Truth)
# ===============================================================================
# These configurations control moment detection modes across Pass 2A and 2B.
# DO NOT duplicate in individual files - import from here.

# Mode duration ranges (min_seconds, max_seconds)
# Controls what duration moments are allowed for each mode
MODE_DURATION_RANGES = {
    "precise": (1.0, 5.0),           # Single frame or very short burst
    "micro_temporal": (3.0, 8.0),    # Reserved for future use
    "inference_window": (8.0, 15.0), # Medium-duration reasoning
    "cluster": (15.0, 60.0),         # Extended temporal sequences
}

# Protected window radii (seconds)
# Controls how much temporal separation is enforced between moments
MODE_PROTECTED_RADII = {
    "precise": 5.0,           # ±5s around precise moments
    "micro_temporal": 3.0,    # Reserved for future use
    "inference_window": 2.0,  # ±2s around inference windows
    "cluster": 0.0,           # No protection (clusters can overlap)
}


def get_mode_duration_range(mode: str) -> tuple[float, float]:
    """
    Get the allowed duration range for a mode.

    Args:
        mode: Mode name (precise, micro_temporal, inference_window, cluster)

    Returns:
        (min_duration, max_duration) in seconds
    """
    return MODE_DURATION_RANGES.get(mode, (0.0, 999.0))


def get_mode_protected_radius(mode: str) -> float:
    """
    Get the protected window radius for a mode.

    Args:
        mode: Mode name (precise, micro_temporal, inference_window, cluster)

    Returns:
        Radius in seconds
    """
    return MODE_PROTECTED_RADII.get(mode, 5.0)
