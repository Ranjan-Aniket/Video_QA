"""
Shared validation functions for Pass 2A and Pass 2B moment detection.

Extracted from pass2a_sonnet_selector.py and pass2b_opus_selector.py
to eliminate code duplication.

All validation functions are stateless and accept configuration parameters.
"""

import re
from typing import Dict, List, Tuple, Any

from .ontology_types import normalize_type, get_min_temporal_window


def validate_temporal_window_for_type(
    moment: Dict[str, Any],
    **kwargs
) -> Tuple[bool, str]:
    """
    Validate temporal window meets minimum for question type.

    Args:
        moment: Moment dict with protected_window and primary_ontology
        **kwargs: Unused (for API compatibility)

    Returns:
        (is_valid, error_message)
    """
    ontology_type = moment.get('primary_ontology', '')

    # Normalize type name
    normalized = normalize_type(ontology_type)

    # Use centralized temporal window function
    min_window = get_min_temporal_window(normalized)

    window = moment.get('protected_window', {})
    start = window.get('start', 0)
    end = window.get('end', 0)
    actual_window = end - start

    if actual_window < min_window:
        return False, f"{normalized} requires {min_window}s window, got {actual_window:.1f}s"

    return True, ""


def validate_moment_duration(
    moment: Dict[str, Any],
    mode_duration_ranges: Dict[str, Tuple[float, float]]
) -> Tuple[bool, str]:
    """
    Validate moment duration matches mode requirements.

    ✅ FIXED: For cluster mode, validate against question type requirements
    instead of just mode duration range.

    Args:
        moment: Moment dict with 'mode' and 'duration' fields
        mode_duration_ranges: Dict mapping mode -> (min_duration, max_duration)

    Returns:
        (is_valid, error_message)
    """
    mode = moment.get('mode', 'precise')
    duration = moment.get('duration', 0)

    min_dur, max_dur = mode_duration_ranges.get(mode, (0, 999))

    # ✅ CLUSTER MODE FIX: Validate against question type minimum temporal window
    # Cluster mode (15-60s) is too permissive for high-complexity types like
    # Comparative (40s min) and Spurious Correlations (40s min).
    # This ensures cluster duration respects the question type requirements.
    if mode == 'cluster':
        primary_ontology = moment.get('primary_ontology', '')
        if primary_ontology:
            min_required = get_min_temporal_window(primary_ontology)

            # Cluster must span at least the question type's minimum window
            if duration < min_required:
                return False, f"Duration {duration}s too short for {primary_ontology} (min: {min_required}s)"

            # But still can't exceed cluster max
            if duration > max_dur:
                return False, f"Duration {duration}s too long for cluster (max: {max_dur}s)"

            return True, ""

    # Original validation for non-cluster modes
    if duration < min_dur:
        return False, f"Duration {duration}s too short for {mode} (min: {min_dur}s)"
    if duration > max_dur:
        return False, f"Duration {duration}s too long for {mode} (max: {max_dur}s)"

    return True, ""


def validate_cue_quality(
    moment: Dict[str, Any],
    hedging_patterns: List[str],
    pronoun_patterns: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate visual_cues and audio_cues for quality issues.

    Checks for:
    - Hedging language (appears, seems, possibly, etc.)
    - Pronouns (he, she, they, etc.)

    Args:
        moment: Moment dict with 'visual_cues' and 'audio_cues' fields
        hedging_patterns: List of regex patterns for hedging language
        pronoun_patterns: List of regex patterns for pronouns

    Returns:
        (is_valid, list of issues found)
    """
    issues = []

    visual_cues = moment.get('visual_cues', [])
    audio_cues = moment.get('audio_cues', [])
    combined_text = ' '.join(visual_cues + audio_cues)

    # Check hedging
    for pattern in hedging_patterns:
        match = re.search(pattern, combined_text, re.IGNORECASE)
        if match:
            issues.append(f"Hedging: '{match.group()}'")

    # Check pronouns
    for pattern in pronoun_patterns:
        match = re.search(pattern, combined_text, re.IGNORECASE)
        if match:
            issues.append(f"Pronoun: '{match.group()}'")

    return len(issues) == 0, issues


def validate_protected_radius(
    moment: Dict[str, Any],
    protected_radii: Dict[str, float]
) -> Tuple[bool, str]:
    """
    Validate protected window radius matches mode.

    Args:
        moment: Moment dict with 'mode' and 'protected_window' fields
        protected_radii: Dict mapping mode -> expected_radius

    Returns:
        (is_valid, error_message)
    """
    mode = moment.get('mode', 'precise')
    expected_radius = protected_radii.get(mode, 5.0)

    window = moment.get('protected_window', {})
    actual_radius = window.get('radius', 0)

    # Allow 1s tolerance for floating point variations
    # Skip for cluster mode (no protection needed)
    if mode != 'cluster' and abs(actual_radius - expected_radius) > 1.0:
        return False, f"Radius {actual_radius}s should be ~{expected_radius}s for {mode}"

    return True, ""
