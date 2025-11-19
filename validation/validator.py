"""
Validation Orchestrator

Runs all 10 validation layers in sequence.

CRITICAL: Question is REJECTED if ANY layer fails.
No correction - only rejection (prevents hallucinations).
"""

from typing import List, Tuple, Optional, Dict
from templates.base import GeneratedQuestion, EvidenceDatabase
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


class ValidationOrchestrator:
    """
    Run all 10 validation layers
    
    Rejects question if ANY layer fails.
    """
    
    def __init__(self):
        """Initialize all validators"""
        self.validators = [
            ("Evidence Grounding", EvidenceGroundingValidator()),
            ("Dual Cue Check", DualCueValidator()),
            ("Name Detection", NameDetectionValidator()),
            ("Timestamp Validation", TimestampValidator()),
            ("Single Cue Answerable", SingleCueAnswerableValidator()),
            ("Intro/Outro Check", IntroOutroValidator()),
            ("Complexity Check", ComplexityValidator()),
            ("Descriptor Validation", DescriptorValidator()),
            ("Cue Necessity", CueNecessityValidator()),
            ("Final QC", FinalQCValidator())
        ]
    
    def validate(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Validate question through all layers
        
        Args:
            question: Generated question
            evidence: Evidence database
            
        Returns:
            (is_valid, error_message, validation_report)
        """
        report = {
            "total_layers": len(self.validators),
            "passed_layers": 0,
            "failed_layer": None,
            "error_message": None,
            "layer_results": []
        }
        
        # Run each validator
        for layer_name, validator in self.validators:
            is_valid, error_msg = validator.validate(question, evidence)
            
            report["layer_results"].append({
                "layer": layer_name,
                "passed": is_valid,
                "error": error_msg
            })
            
            if not is_valid:
                # FAIL - reject question
                report["failed_layer"] = layer_name
                report["error_message"] = error_msg
                return False, f"Layer '{layer_name}' failed: {error_msg}", report
            
            report["passed_layers"] += 1
        
        # All layers passed
        return True, None, report
    
    def validate_batch(
        self,
        questions: List[GeneratedQuestion],
        evidence: EvidenceDatabase
    ) -> Tuple[List[GeneratedQuestion], List[Dict]]:
        """
        Validate batch of questions
        
        Args:
            questions: List of generated questions
            evidence: Evidence database
            
        Returns:
            (valid_questions, validation_reports)
        """
        valid_questions = []
        all_reports = []
        
        for question in questions:
            is_valid, error_msg, report = self.validate(question, evidence)
            
            all_reports.append({
                "question": question.question_text,
                "is_valid": is_valid,
                "error": error_msg,
                "report": report
            })
            
            if is_valid:
                valid_questions.append(question)
        
        return valid_questions, all_reports
    
    def get_statistics(self, reports: List[Dict]) -> Dict:
        """Get validation statistics"""
        total = len(reports)
        passed = sum(1 for r in reports if r["is_valid"])
        failed = total - passed
        
        # Count failures by layer
        layer_failures = {}
        for report in reports:
            if not report["is_valid"]:
                layer = report["report"]["failed_layer"]
                layer_failures[layer] = layer_failures.get(layer, 0) + 1
        
        return {
            "total_questions": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "layer_failures": layer_failures
        }
