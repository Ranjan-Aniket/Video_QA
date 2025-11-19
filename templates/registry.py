"""
Enhanced Template Registry with Multi-Type Support

Manages all 13 single-type templates + ~20 multi-type combinations.
Includes compatibility matrix to ensure valid combinations.

CRITICAL: All templates enforce guidelines:
1. Both audio AND visual cues
2. NO names - only descriptors
3. Precise timestamps
4. Evidence-first generation
"""

from typing import List, Dict, Optional, Set, FrozenSet, Tuple
from templates.base import QuestionTemplate, GeneratedQuestion, EvidenceDatabase, QuestionType

# Import all single-type templates
from templates.counting import CountingTemplate
from templates.temporal_sequential import (
    TemporalUnderstandingTemplate,
    SequentialTemplate
)
from templates.needle_referential_context import (
    NeedleTemplate,
    ReferentialGroundingTemplate,
    ContextTemplate
)
from templates.comparative_inference_interaction import (
    ComparativeTemplate,
    InferenceTemplate,
    ObjectInteractionTemplate
)
from templates.subscene_holistic_stitching_spurious import (
    SubsceneTemplate,
    HolisticReasoningTemplate,
    AudioVisualStitchingTemplate,
    SpuriousCorrelationsTemplate
)

# Import all multi-type combination templates
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
    
    # Object & Sequential
    SequentialObjectInteractionTemplate,
    
    # Spurious combinations
    SpuriousContextTemplate,
    ReferentialSpuriousTemplate,
    
    # Audio-Visual Stitching
    AudioVisualStitchingReferentialTemplate,
    
    # Three-type combinations
    SequentialSubsceneHolisticTemplate
)


class EnhancedTemplateRegistry:
    """
    Registry managing all question templates
    
    Features:
    - 13 single-type templates
    - ~20 multi-type combination templates
    - Compatibility matrix for valid combinations
    - Evidence-based template selection
    """
    
    def __init__(self):
        """Initialize registry with all templates"""
        
        # Single-type templates (13 total)
        self.single_type_templates: List[QuestionTemplate] = [
            CountingTemplate(),
            TemporalUnderstandingTemplate(),
            SequentialTemplate(),
            NeedleTemplate(),
            ReferentialGroundingTemplate(),
            ContextTemplate(),
            ComparativeTemplate(),
            InferenceTemplate(),
            ObjectInteractionTemplate(),
            SubsceneTemplate(),
            HolisticReasoningTemplate(),
            AudioVisualStitchingTemplate(),
            SpuriousCorrelationsTemplate()
        ]
        
        # Multi-type combination templates (~20 total)
        self.multi_type_templates: List[QuestionTemplate] = [
            # Temporal combinations (3)
            TemporalSequentialCountingTemplate(),
            TemporalSequentialNeedleTemplate(),
            TemporalSequentialInferenceTemplate(),
            
            # Subscene combinations (1)
            SubsceneNeedleTemplate(),
            
            # Inference combinations (2)
            HolisticInferenceTemplate(),
            ContextInferenceCountingTemplate(),
            
            # Object & Sequential (1)
            SequentialObjectInteractionTemplate(),
            
            # Spurious combinations (2)
            SpuriousContextTemplate(),
            ReferentialSpuriousTemplate(),
            
            # Audio-Visual Stitching (1)
            AudioVisualStitchingReferentialTemplate(),
            
            # Three-type combinations (1)
            SequentialSubsceneHolisticTemplate()
        ]
        
        # All templates combined
        self.all_templates = self.single_type_templates + self.multi_type_templates
        
        # Build compatibility matrix
        self.compatibility_matrix = self._build_compatibility_matrix()
    
    def _build_compatibility_matrix(self) -> Dict[FrozenSet[QuestionType], float]:
        """
        Build compatibility matrix for question type combinations
        
        Based on real examples from guidelines and sample data.
        Score: 0.0 (invalid) to 1.0 (highly compatible)
        
        Returns:
            Dict mapping frozenset of question types to compatibility score
        """
        matrix = {}
        
        # ====================================================================
        # HIGHLY COMPATIBLE COMBINATIONS (0.8 - 1.0)
        # Based on real examples from documents
        # ====================================================================
        
        # Temporal + Sequential + Counting (very common)
        matrix[frozenset([
            QuestionType.TEMPORAL_UNDERSTANDING,
            QuestionType.SEQUENTIAL,
            QuestionType.COUNTING
        ])] = 1.0
        
        # Temporal + Sequential + Needle (common)
        matrix[frozenset([
            QuestionType.TEMPORAL_UNDERSTANDING,
            QuestionType.SEQUENTIAL,
            QuestionType.NEEDLE
        ])] = 0.9
        
        # Temporal + Sequential + Inference (common)
        matrix[frozenset([
            QuestionType.TEMPORAL_UNDERSTANDING,
            QuestionType.SEQUENTIAL,
            QuestionType.INFERENCE
        ])] = 0.9
        
        # Subscene + Needle (common)
        matrix[frozenset([
            QuestionType.SUBSCENE,
            QuestionType.NEEDLE
        ])] = 0.9
        
        # Holistic + Inference (common - "why does overall pattern exist?")
        matrix[frozenset([
            QuestionType.GENERAL_HOLISTIC_REASONING,
            QuestionType.INFERENCE
        ])] = 0.9
        
        # Context + Inference + Counting (from examples)
        matrix[frozenset([
            QuestionType.CONTEXT,
            QuestionType.INFERENCE,
            QuestionType.COUNTING
        ])] = 0.8
        
        # Sequential + Object Interaction (transformation sequences)
        matrix[frozenset([
            QuestionType.SEQUENTIAL,
            QuestionType.OBJECT_INTERACTION_REASONING
        ])] = 0.8
        
        # Spurious + Context (unexpected reference in scene)
        matrix[frozenset([
            QuestionType.TACKLING_SPURIOUS_CORRELATIONS,
            QuestionType.CONTEXT
        ])] = 0.9
        
        # Referential + Spurious (grounding unexpected elements)
        matrix[frozenset([
            QuestionType.REFERENTIAL_GROUNDING,
            QuestionType.TACKLING_SPURIOUS_CORRELATIONS
        ])] = 0.8
        
        # Audio-Visual Stitching + Referential (editing with grounding)
        matrix[frozenset([
            QuestionType.AUDIO_VISUAL_STITCHING,
            QuestionType.REFERENTIAL_GROUNDING
        ])] = 0.8
        
        # Sequential + Subscene + Holistic (complex three-type)
        matrix[frozenset([
            QuestionType.SEQUENTIAL,
            QuestionType.SUBSCENE,
            QuestionType.GENERAL_HOLISTIC_REASONING
        ])] = 0.8
        
        # ====================================================================
        # MODERATELY COMPATIBLE (0.5 - 0.7)
        # Possible but less common
        # ====================================================================
        
        # Temporal + Counting (without Sequential)
        matrix[frozenset([
            QuestionType.TEMPORAL_UNDERSTANDING,
            QuestionType.COUNTING
        ])] = 0.6
        
        # Sequential + Counting (without Temporal)
        matrix[frozenset([
            QuestionType.SEQUENTIAL,
            QuestionType.COUNTING
        ])] = 0.6
        
        # Temporal + Comparative
        matrix[frozenset([
            QuestionType.TEMPORAL_UNDERSTANDING,
            QuestionType.COMPARATIVE
        ])] = 0.7
        
        # Comparative + Inference
        matrix[frozenset([
            QuestionType.COMPARATIVE,
            QuestionType.INFERENCE
        ])] = 0.6
        
        # Needle + Referential
        matrix[frozenset([
            QuestionType.NEEDLE,
            QuestionType.REFERENTIAL_GROUNDING
        ])] = 0.6
        
        # Context + Counting (without Inference)
        matrix[frozenset([
            QuestionType.CONTEXT,
            QuestionType.COUNTING
        ])] = 0.5
        
        # ====================================================================
        # INCOMPATIBLE COMBINATIONS (0.0 - 0.3)
        # Logically inconsistent or nonsensical
        # ====================================================================
        
        # Audio-Visual Stitching + Counting (doesn't make sense)
        matrix[frozenset([
            QuestionType.AUDIO_VISUAL_STITCHING,
            QuestionType.COUNTING
        ])] = 0.0
        
        # Subscene + Audio-Visual Stitching (conflicting concepts)
        matrix[frozenset([
            QuestionType.SUBSCENE,
            QuestionType.AUDIO_VISUAL_STITCHING
        ])] = 0.0
        
        # Comparative + Counting (awkward combination)
        matrix[frozenset([
            QuestionType.COMPARATIVE,
            QuestionType.COUNTING
        ])] = 0.2
        
        # Needle + Holistic (opposite granularities)
        matrix[frozenset([
            QuestionType.NEEDLE,
            QuestionType.GENERAL_HOLISTIC_REASONING
        ])] = 0.1
        
        # Object Interaction + Audio-Visual Stitching (unrelated)
        matrix[frozenset([
            QuestionType.OBJECT_INTERACTION_REASONING,
            QuestionType.AUDIO_VISUAL_STITCHING
        ])] = 0.0
        
        return matrix
    
    def is_valid_combination(
        self,
        question_types: List[QuestionType],
        min_score: float = 0.5
    ) -> bool:
        """
        Check if combination of question types is valid
        
        Args:
            question_types: List of question types to combine
            min_score: Minimum compatibility score (default 0.5)
            
        Returns:
            True if valid combination, False otherwise
        """
        if len(question_types) == 1:
            return True  # Single types always valid
        
        key = frozenset(question_types)
        score = self.compatibility_matrix.get(key, 0.0)
        
        return score >= min_score
    
    def get_compatibility_score(
        self,
        question_types: List[QuestionType]
    ) -> float:
        """
        Get compatibility score for combination
        
        Returns:
            Score from 0.0 (invalid) to 1.0 (highly compatible)
        """
        if len(question_types) == 1:
            return 1.0
        
        key = frozenset(question_types)
        return self.compatibility_matrix.get(key, 0.0)
    
    def generate_tier1_questions(
        self,
        evidence: EvidenceDatabase,
        target_count: int = 8,
        prefer_multi_type: bool = True
    ) -> List[GeneratedQuestion]:
        """
        Generate Tier 1 (template-based) questions
        
        Args:
            evidence: Evidence database
            target_count: Target number of questions (default 8)
            prefer_multi_type: Prefer multi-type over single-type (default True)
            
        Returns:
            List of generated questions
        """
        questions = []
        
        # Strategy: Try multi-type first (more challenging), then single-type
        templates_to_try = []
        
        if prefer_multi_type:
            templates_to_try = self.multi_type_templates + self.single_type_templates
        else:
            templates_to_try = self.single_type_templates + self.multi_type_templates
        
        # Try each template
        for template in templates_to_try:
            if len(questions) >= target_count:
                break
            
            # Check if template can apply
            if not template.can_apply(evidence):
                continue
            
            # Generate question
            try:
                question = template.generate(evidence)
                if question:
                    questions.append(question)
            except Exception as e:
                # Log error but continue
                print(f"[Tier1] Error in {template.name}: {e}")
                continue
        
        return questions[:target_count]
    
    def get_templates_by_type(
        self,
        question_type: QuestionType,
        include_combinations: bool = True
    ) -> List[QuestionTemplate]:
        """
        Get all templates that generate a specific question type
        
        Args:
            question_type: Question type to find
            include_combinations: Include multi-type templates (default True)
            
        Returns:
            List of matching templates
        """
        matching = []
        
        # Search single-type templates
        for template in self.single_type_templates:
            types = template.get_question_types()
            if question_type in types:
                matching.append(template)
        
        # Search multi-type templates
        if include_combinations:
            for template in self.multi_type_templates:
                types = template.get_question_types()
                if question_type in types:
                    matching.append(template)
        
        return matching
    
    def get_template_by_name(self, name: str) -> Optional[QuestionTemplate]:
        """Get template by class name"""
        for template in self.all_templates:
            if template.name == name:
                return template
        return None
    
    def get_multi_type_templates(self) -> List[QuestionTemplate]:
        """Get all multi-type combination templates"""
        return self.multi_type_templates
    
    def get_single_type_templates(self) -> List[QuestionTemplate]:
        """Get all single-type templates"""
        return self.single_type_templates
    
    def get_statistics(self) -> Dict:
        """Get registry statistics"""
        
        # Count questions types covered
        all_types = set()
        for template in self.all_templates:
            all_types.update(template.get_question_types())
        
        # Count combinations
        multi_type_combos = [
            frozenset(t.get_question_types())
            for t in self.multi_type_templates
        ]
        
        return {
            "total_templates": len(self.all_templates),
            "single_type_templates": len(self.single_type_templates),
            "multi_type_templates": len(self.multi_type_templates),
            "question_types_covered": len(all_types),
            "template_names": [t.name for t in self.all_templates],
            "multi_type_combinations": len(multi_type_combos),
            "compatibility_entries": len(self.compatibility_matrix)
        }
    
    def validate_question_types(
        self,
        question: GeneratedQuestion
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate question's type combination
        
        Args:
            question: Generated question to validate
            
        Returns:
            (is_valid, error_message)
        """
        types = question.question_types
        
        # Single type always valid
        if len(types) == 1:
            return True, None
        
        # Check compatibility
        if not self.is_valid_combination(types):
            score = self.get_compatibility_score(types)
            return False, f"Invalid type combination (compatibility: {score})"
        
        return True, None
    
    def get_recommended_combinations(
        self,
        evidence: EvidenceDatabase,
        min_score: float = 0.7
    ) -> List[Tuple[List[QuestionType], float]]:
        """
        Get recommended question type combinations for this evidence
        
        Args:
            evidence: Evidence database
            min_score: Minimum compatibility score
            
        Returns:
            List of (question_types, score) tuples, sorted by score
        """
        recommendations = []
        
        # Check each multi-type template
        for template in self.multi_type_templates:
            if not template.can_apply(evidence):
                continue
            
            types = template.get_question_types()
            score = self.get_compatibility_score(types)
            
            if score >= min_score:
                recommendations.append((types, score))
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations

    def get_applicable_templates(
        self,
        evidence: EvidenceDatabase
    ) -> List[QuestionTemplate]:
        """
        Get all templates that can be applied to the given evidence.

        Args:
            evidence: Evidence database to check against

        Returns:
            List of applicable templates
        """
        applicable = []

        # Check all templates (both single and multi-type)
        for template in self.all_templates:
            try:
                if template.can_apply(evidence):
                    applicable.append(template)
            except Exception as e:
                # Log but don't fail if a template check fails
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Template {template.name} failed applicability check: {e}")
                continue

        return applicable

    def select_templates_with_compatibility(
        self,
        evidence: EvidenceDatabase,
        target_count: int = 25,
        min_compatibility_score: float = 0.7
    ) -> List[Tuple[QuestionTemplate, float]]:
        """
        Select templates based on compatibility scores and evidence.

        Args:
            evidence: Evidence database
            target_count: Target number of templates to select
            min_compatibility_score: Minimum compatibility score threshold

        Returns:
            List of (template, compatibility_score) tuples sorted by score
        """
        import logging
        logger = logging.getLogger(__name__)

        # Get applicable templates
        applicable = self.get_applicable_templates(evidence)

        if not applicable:
            logger.warning("No applicable templates found")
            return []

        # Score each template
        scored_templates = []
        for template in applicable:
            try:
                # Get question types for this template
                types = template.get_question_types()

                # Get compatibility score
                score = self.get_compatibility_score(types)

                # Only include if meets minimum score
                if score >= min_compatibility_score:
                    scored_templates.append((template, score))

            except Exception as e:
                logger.warning(f"Failed to score template {template.name}: {e}")
                continue

        # Sort by compatibility score (highest first)
        scored_templates.sort(key=lambda x: x[1], reverse=True)

        # Return top N templates
        return scored_templates[:target_count]


# Global registry instance
_registry = None


def get_enhanced_registry() -> EnhancedTemplateRegistry:
    """Get global enhanced template registry (singleton)"""
    global _registry
    if _registry is None:
        _registry = EnhancedTemplateRegistry()
    return _registry


def list_all_templates() -> List[str]:
    """List all available template names"""
    registry = get_enhanced_registry()
    return [t.name for t in registry.all_templates]


def list_valid_combinations() -> List[Tuple[List[str], float]]:
    """
    List all valid question type combinations with scores
    
    Returns:
        List of (type_names, score) sorted by score
    """
    registry = get_enhanced_registry()
    
    combinations = []
    for types_set, score in registry.compatibility_matrix.items():
        if score > 0.0:
            type_names = [qt.value for qt in types_set]
            combinations.append((type_names, score))
    
    # Sort by score
    combinations.sort(key=lambda x: x[1], reverse=True)
    
    return combinations


def validate_combination(question_types: List[str]) -> Tuple[bool, float]:
    """
    Validate if question type names form valid combination
    
    Args:
        question_types: List of question type names (strings)
        
    Returns:
        (is_valid, compatibility_score)
    """
    registry = get_enhanced_registry()
    
    # Convert strings to QuestionType enums
    try:
        types = [QuestionType(name) for name in question_types]
    except ValueError as e:
        return False, 0.0
    
    score = registry.get_compatibility_score(types)
    is_valid = registry.is_valid_combination(types)
    
    return is_valid, score


# ============================================================================
# TESTING & DEBUGGING UTILITIES
# ============================================================================

def print_registry_info():
    """Print registry information for debugging"""
    registry = get_enhanced_registry()
    stats = registry.get_statistics()
    
    print("=" * 80)
    print("ENHANCED TEMPLATE REGISTRY")
    print("=" * 80)
    print(f"Total Templates: {stats['total_templates']}")
    print(f"  - Single-type: {stats['single_type_templates']}")
    print(f"  - Multi-type: {stats['multi_type_templates']}")
    print(f"Question Types Covered: {stats['question_types_covered']}/13")
    print(f"Compatibility Matrix Entries: {stats['compatibility_entries']}")
    print()
    
    print("SINGLE-TYPE TEMPLATES:")
    for i, template in enumerate(registry.single_type_templates, 1):
        types = [qt.value for qt in template.get_question_types()]
        print(f"  {i}. {template.name}")
        print(f"     Types: {types}")
    print()
    
    print("MULTI-TYPE TEMPLATES:")
    for i, template in enumerate(registry.multi_type_templates, 1):
        types = [qt.value for qt in template.get_question_types()]
        score = registry.get_compatibility_score(template.get_question_types())
        print(f"  {i}. {template.name}")
        print(f"     Types: {types}")
        print(f"     Compatibility: {score:.2f}")
    print()
    
    print("TOP 10 VALID COMBINATIONS:")
    combos = list_valid_combinations()
    for i, (types, score) in enumerate(combos[:10], 1):
        print(f"  {i}. {' + '.join(types)}")
        print(f"     Score: {score:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    # Test registry
    print_registry_info()
    
    # Test validation
    print("\nTESTING VALIDATION:")
    test_combos = [
        ["Temporal Understanding", "Sequential", "Counting"],
        ["Audio-Visual Stitching", "Counting"],  # Invalid
        ["Subscene", "Needle"],
        ["Needle", "General Holistic Reasoning"]  # Invalid
    ]
    
    for combo in test_combos:
        is_valid, score = validate_combination(combo)
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"{status} ({score:.2f}): {' + '.join(combo)}")