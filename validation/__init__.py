"""
Validation Package

10-layer validation system that ensures 99%+ accuracy:

Layer 1: Evidence Grounding - All facts verified against evidence
Layer 2: Dual Cue Check - Both audio AND visual cues present
Layer 3: Name Detection - No names detected
Layer 4: Timestamp Validation - Timestamps precise and valid
Layer 5: Single Cue Answerable - Cannot be answered with single cue
Layer 6: Intro/Outro Check - Not in intro/outro segments
Layer 7: Complexity Check - Sufficiently challenging
Layer 8: Descriptor Validation - Descriptors used correctly
Layer 9: Cue Necessity - Both cues truly necessary
Layer 10: Final QC - Grammar, formatting, quality

CRITICAL: Question is REJECTED if ANY layer fails (no correction).
"""

from validation.layer01_evidence_grounding import EvidenceGroundingValidator
from validation.layer02_05 import (
    DualCueValidator,
    NameDetectionValidator,
    TimestampValidator,
    SingleCueAnswerableValidator
)
from validation.layer06_10 import (
    IntroOutroValidator,
    ComplexityValidator,
    DescriptorValidator,
    CueNecessityValidator,
    FinalQCValidator
)
from validation.validator import ValidationOrchestrator

__all__ = [
    # Individual validators
    'EvidenceGroundingValidator',
    'DualCueValidator',
    'NameDetectionValidator',
    'TimestampValidator',
    'SingleCueAnswerableValidator',
    'IntroOutroValidator',
    'ComplexityValidator',
    'DescriptorValidator',
    'CueNecessityValidator',
    'FinalQCValidator',
    
    # Orchestrator
    'ValidationOrchestrator'
]