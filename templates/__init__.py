"""
Templates Package - Enhanced Multi-Type Question Generation

Provides mixin-based template system for generating adversarial multimodal questions.

ARCHITECTURE:
- Base classes: QuestionTemplate, EvidenceDatabase, GeneratedQuestion
- Mixins: 13 reusable components for individual question types
- Combinations: 20+ templates combining multiple question types
- Registry: Enhanced registry with compatibility scoring

USAGE:
    from templates import EnhancedTemplateRegistry, get_all_templates
    from templates.base import EvidenceDatabase, QuestionType
    
    # Initialize registry
    registry = EnhancedTemplateRegistry()
    
    # Get applicable templates for evidence
    templates = registry.get_applicable_templates(evidence)
    
    # Generate questions
    questions = registry.generate_tier1_questions(evidence, target_count=25)
"""

# Base classes and types
from templates.base import (
    QuestionTemplate,
    GeneratedQuestion,
    EvidenceDatabase,
    QuestionType,
    Cue,
    CueType,
)

# Descriptor generator
from templates.descriptor_generator import (
    DescriptorGenerator,
    VisualEvidence,
)

# Mixin components (13 individual question type components)
from templates.mixins import (
    # Core mixins
    TemporalMixin,
    SequentialMixin,
    CountingMixin,
    NeedleMixin,
    ReferentialGroundingMixin,
    ContextMixin,
    ComparativeMixin,
    InferenceMixin,
    ObjectInteractionMixin,
    SubsceneMixin,
    HolisticReasoningMixin,
    AudioVisualStitchingMixin,
    SpuriousCorrelationMixin,
    
    # Helper utilities
    MixinHelpers,
)

# Multi-type combination templates (20+ templates)
from templates.combinations import (
    # Temporal combinations
    TemporalSequentialCountingTemplate,
    TemporalSequentialNeedleTemplate,
    TemporalSequentialInferenceTemplate,

    # Subscene combinations
    SubsceneNeedleTemplate,

    # Inference combinations
    HolisticInferenceTemplate,
    ContextInferenceCountingTemplate,

    # Sequential combinations
    SequentialObjectInteractionTemplate,

    # Spurious correlation combinations
    SpuriousContextTemplate,
    ReferentialSpuriousTemplate,

    # Audio-visual combinations
    AudioVisualStitchingReferentialTemplate,

    # Complex multi-type combinations
    SequentialSubsceneHolisticTemplate,
)

# Registry
from templates.registry import (
    EnhancedTemplateRegistry,
)


# ============================================================================
# PUBLIC API
# ============================================================================

def get_registry() -> EnhancedTemplateRegistry:
    """
    Get enhanced template registry instance
    
    Returns:
        EnhancedTemplateRegistry with all templates loaded
    """
    return EnhancedTemplateRegistry()


def get_all_templates():
    """
    Get list of all available template classes
    
    Returns:
        List of template classes
    """
    registry = EnhancedTemplateRegistry()
    return registry.templates


def get_question_types():
    """
    Get list of all question types
    
    Returns:
        List of QuestionType enum values
    """
    return list(QuestionType)


def get_template_statistics():
    """
    Get statistics about available templates
    
    Returns:
        Dictionary with template counts and coverage
    """
    registry = EnhancedTemplateRegistry()
    return registry.get_statistics()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Base classes
    'QuestionTemplate',
    'GeneratedQuestion',
    'EvidenceDatabase',
    'QuestionType',
    'Cue',
    'CueType',
    'VisualEvidence',
    
    # Descriptor generator
    'DescriptorGenerator',
    
    # Mixins
    'TemporalMixin',
    'SequentialMixin',
    'CountingMixin',
    'NeedleMixin',
    'ReferentialGroundingMixin',
    'ContextMixin',
    'ComparativeMixin',
    'InferenceMixin',
    'ObjectInteractionMixin',
    'SubsceneMixin',
    'HolisticReasoningMixin',
    'AudioVisualStitchingMixin',
    'SpuriousCorrelationMixin',
    'MixinHelpers',
    
    # Combination templates
    'TemporalSequentialCountingTemplate',
    'TemporalSequentialNeedleTemplate',
    'TemporalSequentialInferenceTemplate',
    'SubsceneNeedleTemplate',
    'HolisticInferenceTemplate',
    'ContextInferenceCountingTemplate',
    'SequentialObjectInteractionTemplate',
    'SpuriousContextTemplate',
    'ReferentialSpuriousTemplate',
    'AudioVisualStitchingReferentialTemplate',
    'SequentialSubsceneHolisticTemplate',
    
    # Registry
    'EnhancedTemplateRegistry',

    # Convenience functions
    'get_registry',
    'get_all_templates',
    'get_question_types',
    'get_template_statistics',
]

__version__ = '2.0.0'  # Enhanced with mixin-based templates