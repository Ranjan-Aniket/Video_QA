"""
Validation Module - Complete Guidelines Enforcement

This module enforces ALL guidelines and question type requirements:
- 15 critical validation rules from Guidelines document
- 13 NVIDIA question types from Question Types & Skills PDF
- Generalized name/pronoun replacement (NER-based, domain-agnostic)

Usage:
    from validation.generalized_name_replacer import GeneralizedNameReplacer
    from validation.complete_guidelines_validator import CompleteGuidelinesValidator
    from validation.question_type_classifier import QuestionTypeClassifier
"""

__version__ = "1.0.0"

__all__ = [
    "DynamicNameDetector",
    "DynamicDescriptorExtractor",
    "GeneralizedNameReplacer",
    "CompleteGuidelinesValidator",
    "QuestionTypeClassifier",
]
