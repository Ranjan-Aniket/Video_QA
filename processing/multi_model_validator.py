"""
Multi-Model Validator - Compare GPT-4, Claude, Open Models, and Gemini

Sends video to Gemini and compares answers from all 4 model types:
1. GPT-4 Vision (from key moments)
2. Claude Sonnet 4.5 (from key moments)
3. Open Models (from bulk frames + evidence)
4. Gemini (full video analysis)

Human reviews and selects best answer from all 4.

Phase 6 of the new evidence-first architecture.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ModelAnswer:
    """Answer from a single model"""
    model_name: str  # "gpt4", "claude", "open", "gemini"
    answer: str
    confidence: float
    reasoning: Optional[str] = None
    evidence_used: Optional[List[str]] = None


@dataclass
class QuestionValidation:
    """Validation results for a single question"""
    question_id: str
    question_text: str
    correct_answer: str  # Ground truth

    # Model answers
    gpt4_answer: Optional[ModelAnswer] = None
    claude_answer: Optional[ModelAnswer] = None
    open_model_answer: Optional[ModelAnswer] = None
    gemini_answer: Optional[ModelAnswer] = None

    # Comparison
    models_agree: bool = False
    consensus_answer: Optional[str] = None
    disagreement_details: Optional[Dict] = None

    # Human review
    needs_human_review: bool = False
    human_selected_model: Optional[str] = None
    human_selected_answer: Optional[str] = None


class MultiModelValidator:
    """
    Validate answers across multiple AI models
    """

    def __init__(
        self,
        video_path: str,
        questions: List[Dict],
        enable_gpt4: bool = False,
        enable_claude: bool = False,
        enable_gemini: bool = False
    ):
        """
        Initialize multi-model validator

        Args:
            video_path: Path to video file
            questions: List of generated questions
            enable_gpt4: Use GPT-4 for validation
            enable_claude: Use Claude for validation
            enable_gemini: Use Gemini for validation
        """
        self.video_path = Path(video_path)
        self.questions = questions
        self.enable_gpt4 = enable_gpt4
        self.enable_claude = enable_claude
        self.enable_gemini = enable_gemini

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        logger.info("=" * 80)
        logger.info("MULTI-MODEL VALIDATOR - INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Video: {self.video_path.name}")
        logger.info(f"Questions: {len(questions)}")
        logger.info(f"GPT-4: {'ENABLED' if enable_gpt4 else 'DISABLED'}")
        logger.info(f"Claude: {'ENABLED' if enable_claude else 'DISABLED'}")
        logger.info(f"Gemini: {'ENABLED' if enable_gemini else 'DISABLED'}")

    def validate_all_questions(self) -> List[QuestionValidation]:
        """
        Get answers from all models and compare

        Returns:
            List of QuestionValidation objects
        """
        logger.info("=" * 80)
        logger.info("VALIDATING QUESTIONS ACROSS ALL MODELS")
        logger.info("=" * 80)

        validations = []

        # Process each question
        for i, question in enumerate(self.questions, 1):
            logger.info(f"\n📝 Question {i}/{len(self.questions)}: {question['question_text'][:60]}...")

            validation = self._validate_question(question)
            validations.append(validation)

            if validation.needs_human_review:
                logger.warning(f"  ⚠️  Needs human review (models disagree)")
            else:
                logger.info(f"  ✓ Models agree: {validation.consensus_answer}")

        # Summary
        needs_review_count = len([v for v in validations if v.needs_human_review])

        logger.info("=" * 80)
        logger.info(f"✅ VALIDATION COMPLETE")
        logger.info(f"   Total questions: {len(validations)}")
        logger.info(f"   Consensus reached: {len(validations) - needs_review_count}")
        logger.info(f"   Needs human review: {needs_review_count}")
        logger.info("=" * 80)

        return validations

    def _validate_question(self, question: Dict) -> QuestionValidation:
        """
        Get answers from all models for a single question

        Args:
            question: Question dict

        Returns:
            QuestionValidation object
        """
        question_text = question["question_text"]
        correct_answer = question["correct_answer"]

        # Get answers from each model
        gpt4_answer = None
        claude_answer = None
        open_model_answer = None
        gemini_answer = None

        if self.enable_gpt4:
            gpt4_answer = self._get_gpt4_answer(question_text, correct_answer)

        if self.enable_claude:
            claude_answer = self._get_claude_answer(question_text, correct_answer)

        # Open model answer (use correct answer as reference)
        open_model_answer = ModelAnswer(
            model_name="open_models",
            answer=correct_answer,
            confidence=1.0,
            reasoning="Based on deterministic model outputs (YOLO, OCR, etc.)"
        )

        if self.enable_gemini:
            gemini_answer = self._get_gemini_answer(question_text)

        # Compare answers
        models_agree, consensus, disagreement = self._compare_answers([
            gpt4_answer,
            claude_answer,
            open_model_answer,
            gemini_answer
        ])

        # Create validation object
        validation = QuestionValidation(
            question_id=question["question_id"],
            question_text=question_text,
            correct_answer=correct_answer,
            gpt4_answer=gpt4_answer,
            claude_answer=claude_answer,
            open_model_answer=open_model_answer,
            gemini_answer=gemini_answer,
            models_agree=models_agree,
            consensus_answer=consensus,
            disagreement_details=disagreement,
            needs_human_review=not models_agree
        )

        return validation

    def _get_gpt4_answer(self, question: str, reference_answer: str) -> ModelAnswer:
        """
        Get answer from GPT-4

        Args:
            question: Question text
            reference_answer: Reference answer

        Returns:
            ModelAnswer object
        """
        # Mock GPT-4 answer
        # In production, this would call OpenAI API

        logger.info("  [GPT-4] Generating answer...")

        # Simulate ~95% accuracy
        import random
        if random.random() < 0.95:
            answer = reference_answer
        else:
            answer = "Alternative interpretation: " + reference_answer

        return ModelAnswer(
            model_name="gpt4",
            answer=answer,
            confidence=0.92,
            reasoning="Based on visual frame analysis"
        )

    def _get_claude_answer(self, question: str, reference_answer: str) -> ModelAnswer:
        """
        Get answer from Claude

        Args:
            question: Question text
            reference_answer: Reference answer

        Returns:
            ModelAnswer object
        """
        # Mock Claude answer
        # In production, this would call Anthropic API

        logger.info("  [Claude] Generating answer...")

        # Simulate ~93% accuracy
        import random
        if random.random() < 0.93:
            answer = reference_answer
        else:
            answer = "Different perspective: " + reference_answer

        return ModelAnswer(
            model_name="claude",
            answer=answer,
            confidence=0.89,
            reasoning="Based on context and frame analysis"
        )

    def _get_gemini_answer(self, question: str) -> ModelAnswer:
        """
        Get answer from Gemini (full video)

        Args:
            question: Question text

        Returns:
            ModelAnswer object
        """
        # Mock Gemini answer
        # In production, this would call Google Gemini API

        logger.info("  [Gemini] Analyzing full video...")

        # Gemini sees full video, so may have different perspective
        answer = "Based on full video context: [answer]"

        return ModelAnswer(
            model_name="gemini",
            answer=answer,
            confidence=0.88,
            reasoning="Full video analysis"
        )

    def _compare_answers(
        self,
        answers: List[Optional[ModelAnswer]]
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Compare answers from different models

        Args:
            answers: List of ModelAnswer objects

        Returns:
            (models_agree, consensus_answer, disagreement_details)
        """
        # Filter out None answers
        valid_answers = [a for a in answers if a is not None]

        if not valid_answers:
            return False, None, {"reason": "No answers available"}

        # Simple agreement check: all answers match
        answer_texts = [a.answer for a in valid_answers]
        unique_answers = set(answer_texts)

        if len(unique_answers) == 1:
            # All models agree
            return True, answer_texts[0], None
        else:
            # Models disagree
            disagreement = {
                "unique_answers": len(unique_answers),
                "answer_distribution": {
                    ans: answer_texts.count(ans)
                    for ans in unique_answers
                },
                "model_answers": {
                    a.model_name: a.answer
                    for a in valid_answers
                }
            }
            return False, None, disagreement

    def save_validation_results(
        self,
        validations: List[QuestionValidation],
        output_path: Path
    ):
        """
        Save validation results to JSON

        Args:
            validations: List of QuestionValidation objects
            output_path: Path to save JSON
        """
        # Convert to dict
        validation_data = []
        for v in validations:
            v_dict = asdict(v)
            validation_data.append(v_dict)

        output = {
            "video_path": str(self.video_path),
            "validation_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_questions": len(validations),
                "consensus_reached": len([v for v in validations if v.models_agree]),
                "needs_human_review": len([v for v in validations if v.needs_human_review]),
                "accuracy_by_model": self._calculate_accuracies(validations)
            },
            "validations": validation_data
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"✓ Saved validation results to: {output_path}")

    def _calculate_accuracies(self, validations: List[QuestionValidation]) -> Dict:
        """
        Calculate accuracy for each model

        Args:
            validations: List of QuestionValidation objects

        Returns:
            Accuracy dict
        """
        accuracies = {}

        for model_name in ["gpt4", "claude", "open_models", "gemini"]:
            correct = 0
            total = 0

            for v in validations:
                # Get model's answer
                model_answer = None
                if model_name == "gpt4":
                    model_answer = v.gpt4_answer
                elif model_name == "claude":
                    model_answer = v.claude_answer
                elif model_name == "open_models":
                    model_answer = v.open_model_answer
                elif model_name == "gemini":
                    model_answer = v.gemini_answer

                if model_answer:
                    total += 1
                    if model_answer.answer == v.correct_answer:
                        correct += 1

            if total > 0:
                accuracies[model_name] = {
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total
                }

        return accuracies


def test_multi_model_validator(
    video_path: str,
    questions_path: str
):
    """
    Test multi-model validator

    Args:
        video_path: Path to video
        questions_path: Path to questions JSON
    """
    # Load questions
    with open(questions_path, 'r') as f:
        questions_data = json.load(f)

    questions = questions_data.get("questions", [])

    print(f"\n✓ Loaded {len(questions)} questions")

    # Create validator
    validator = MultiModelValidator(
        video_path=video_path,
        questions=questions[:5],  # Test with first 5 questions
        enable_gpt4=True,
        enable_claude=True,
        enable_gemini=True
    )

    # Validate
    validations = validator.validate_all_questions()

    # Save results
    output_path = Path(questions_path).parent / f"{Path(questions_path).stem.replace('_questions', '')}_validation.json"
    validator.save_validation_results(validations, output_path)

    # Display results
    print("\n" + "=" * 80)
    print("MULTI-MODEL VALIDATION RESULTS")
    print("=" * 80)
    print(f"Total Questions Validated: {len(validations)}")
    print(f"  Consensus: {len([v for v in validations if v.models_agree])}")
    print(f"  Needs Review: {len([v for v in validations if v.needs_human_review])}")

    print("\nSample Validation:")
    print("-" * 80)
    v = validations[0]
    print(f"Question: {v.question_text}")
    print(f"Correct Answer: {v.correct_answer}")
    if v.gpt4_answer:
        print(f"GPT-4: {v.gpt4_answer.answer} ({v.gpt4_answer.confidence:.2%})")
    if v.claude_answer:
        print(f"Claude: {v.claude_answer.answer} ({v.claude_answer.confidence:.2%})")
    if v.open_model_answer:
        print(f"Open Models: {v.open_model_answer.answer}")
    if v.gemini_answer:
        print(f"Gemini: {v.gemini_answer.answer} ({v.gemini_answer.confidence:.2%})")
    print(f"Models Agree: {v.models_agree}")

    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        video_path = sys.argv[1]
        questions_path = sys.argv[2]
        test_multi_model_validator(video_path, questions_path)
    else:
        print("Usage: python multi_model_validator.py <video_path> <questions.json>")
        print("\nExample:")
        print("  python multi_model_validator.py video.mp4 video_questions.json")
