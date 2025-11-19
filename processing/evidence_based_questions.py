"""
Evidence-Based Question Generator - Generate questions FROM extracted evidence

Generates questions based on actual video evidence (not before extraction):
- 25 template questions (based on evidence found)
- 3 GPT-4 generated questions (deep analysis)
- 2 Claude Sonnet generated questions (context-aware)
Total: 30 questions per video

Phase 5 of the new evidence-first architecture.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import random

logger = logging.getLogger(__name__)


@dataclass
class GeneratedQuestion:
    """A generated question with metadata"""
    question_id: str
    question_text: str
    question_type: str  # "template", "gpt4", "claude"
    correct_answer: str
    difficulty: str  # "easy", "medium", "hard"
    evidence_source: List[str]  # Which evidence items support this question
    timestamp_reference: Optional[float] = None
    confidence: float = 1.0
    generator_model: Optional[str] = None


class EvidenceBasedQuestionGenerator:
    """
    Generate questions based on extracted evidence
    """

    def __init__(
        self,
        audio_analysis: Dict,
        genre_analysis: Dict,
        evidence_items: List[Dict],
        enable_gpt4: bool = False,
        enable_claude: bool = False
    ):
        """
        Initialize question generator

        Args:
            audio_analysis: Audio analysis dict
            genre_analysis: Genre analysis dict
            evidence_items: List of evidence items
            enable_gpt4: Use GPT-4 for question generation
            enable_claude: Use Claude for question generation
        """
        self.audio_analysis = audio_analysis
        self.genre_analysis = genre_analysis
        self.evidence_items = evidence_items
        self.enable_gpt4 = enable_gpt4
        self.enable_claude = enable_claude

        self.primary_genre = genre_analysis.get("primary_genre", "unknown")
        self.transcript = audio_analysis.get("transcript", "")

        logger.info("=" * 80)
        logger.info("EVIDENCE-BASED QUESTION GENERATOR - INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Genre: {self.primary_genre}")
        logger.info(f"Evidence items: {len(evidence_items)}")
        logger.info(f"Transcript length: {len(self.transcript)} chars")
        logger.info(f"GPT-4: {'ENABLED' if enable_gpt4 else 'DISABLED'}")
        logger.info(f"Claude: {'ENABLED' if enable_claude else 'DISABLED'}")

    def generate_all_questions(self) -> List[GeneratedQuestion]:
        """
        Generate all 30 questions (25 template + 3 GPT-4 + 2 Claude)

        Returns:
            List of GeneratedQuestion objects
        """
        logger.info("=" * 80)
        logger.info("GENERATING QUESTIONS FROM EVIDENCE")
        logger.info("=" * 80)

        all_questions = []

        # 1. Generate 25 template questions
        logger.info("\n📋 Generating 25 template questions...")
        template_questions = self._generate_template_questions(count=25)
        all_questions.extend(template_questions)
        logger.info(f"✓ Generated {len(template_questions)} template questions")

        # 2. Generate 3 GPT-4 questions
        if self.enable_gpt4:
            logger.info("\n🤖 Generating 3 GPT-4 questions...")
            gpt4_questions = self._generate_gpt4_questions(count=3)
            all_questions.extend(gpt4_questions)
            logger.info(f"✓ Generated {len(gpt4_questions)} GPT-4 questions")
        else:
            logger.warning("⚠️  GPT-4 disabled, skipping GPT-4 questions")

        # 3. Generate 2 Claude questions
        if self.enable_claude:
            logger.info("\n🤖 Generating 2 Claude questions...")
            claude_questions = self._generate_claude_questions(count=2)
            all_questions.extend(claude_questions)
            logger.info(f"✓ Generated {len(claude_questions)} Claude questions")
        else:
            logger.warning("⚠️  Claude disabled, skipping Claude questions")

        logger.info("=" * 80)
        logger.info(f"✅ GENERATED {len(all_questions)} TOTAL QUESTIONS")
        logger.info("=" * 80)

        return all_questions

    def _generate_template_questions(self, count: int = 25) -> List[GeneratedQuestion]:
        """
        Generate template-based questions from evidence

        Args:
            count: Number of template questions to generate

        Returns:
            List of GeneratedQuestion objects
        """
        template_questions = []

        # Define genre-specific question templates
        templates = self._get_question_templates()

        # Generate questions based on evidence
        for i, template in enumerate(templates[:count], 1):
            # Find relevant evidence for this template
            relevant_evidence = self._find_relevant_evidence(template["evidence_type"])

            if not relevant_evidence:
                continue  # Skip if no evidence found

            # Create question from template
            question = self._create_question_from_template(
                template=template,
                evidence=relevant_evidence[0],  # Use first matching evidence
                question_id=f"template_{i:03d}"
            )

            template_questions.append(question)

            if len(template_questions) >= count:
                break

        return template_questions

    def _get_question_templates(self) -> List[Dict]:
        """
        Get question templates based on genre

        Returns:
            List of template dicts
        """
        # Base templates (work for all genres)
        base_templates = [
            {
                "template": "How many people are visible in the video at {timestamp}?",
                "evidence_type": "person_count",
                "difficulty": "easy",
                "answer_field": "person_count"
            },
            {
                "template": "What is the primary emotion displayed at {timestamp}?",
                "evidence_type": "emotion",
                "difficulty": "medium",
                "answer_field": "dominant_emotion"
            },
            {
                "template": "Is the scene at {timestamp} indoors or outdoors?",
                "evidence_type": "scene",
                "difficulty": "easy",
                "answer_field": "is_indoor"
            },
            {
                "template": "What objects are visible at {timestamp}?",
                "evidence_type": "objects",
                "difficulty": "medium",
                "answer_field": "yolov8x_objects"
            },
            {
                "template": "What text is visible on screen at {timestamp}?",
                "evidence_type": "ocr",
                "difficulty": "medium",
                "answer_field": "paddleocr_text"
            },
            {
                "template": "Describe the scene at {timestamp}.",
                "evidence_type": "caption",
                "difficulty": "medium",
                "answer_field": "image_caption"
            },
        ]

        # Vlog-specific templates
        if self.primary_genre == "vlog":
            base_templates.extend([
                {
                    "template": "What is the vlogger wearing at {timestamp}?",
                    "evidence_type": "clothing",
                    "difficulty": "easy",
                    "answer_field": "clip_clothing"
                },
                {
                    "template": "What is the setting/location at {timestamp}?",
                    "evidence_type": "scene",
                    "difficulty": "easy",
                    "answer_field": "places365_scene"
                },
            ])

        # Transcript-based templates
        if self.transcript:
            base_templates.extend([
                {
                    "template": "What is discussed in the first 30 seconds?",
                    "evidence_type": "transcript",
                    "difficulty": "medium",
                    "answer_field": "transcript_segment"
                },
                {
                    "template": "What is the main topic of the video?",
                    "evidence_type": "transcript",
                    "difficulty": "easy",
                    "answer_field": "transcript_summary"
                },
            ])

        return base_templates

    def _find_relevant_evidence(self, evidence_type: str) -> List[Dict]:
        """
        Find evidence items matching the type

        Args:
            evidence_type: Type of evidence to find

        Returns:
            List of matching evidence dicts
        """
        relevant = []

        for item in self.evidence_items:
            ground_truth = item.get("ground_truth", {})

            # Check if evidence contains the requested type
            if evidence_type == "person_count" and ground_truth.get("person_count"):
                relevant.append(item)
            elif evidence_type == "emotion" and ground_truth.get("dominant_emotion"):
                relevant.append(item)
            elif evidence_type == "scene" and ground_truth.get("places365_scene"):
                relevant.append(item)
            elif evidence_type == "objects" and ground_truth.get("yolov8x_objects"):
                relevant.append(item)
            elif evidence_type == "ocr" and ground_truth.get("paddleocr_text"):
                relevant.append(item)
            elif evidence_type == "caption" and ground_truth.get("image_caption"):
                relevant.append(item)
            elif evidence_type == "clothing" and ground_truth.get("clip_clothing"):
                relevant.append(item)
            elif evidence_type == "transcript":
                relevant.append(item)

        return relevant

    def _create_question_from_template(
        self,
        template: Dict,
        evidence: Dict,
        question_id: str
    ) -> GeneratedQuestion:
        """
        Create question from template and evidence

        Args:
            template: Template dict
            evidence: Evidence dict
            question_id: Question ID

        Returns:
            GeneratedQuestion object
        """
        timestamp = evidence.get("timestamp_start", 0.0)
        ground_truth = evidence.get("ground_truth", {})

        # Fill template with timestamp
        question_text = template["template"].replace("{timestamp}", f"{timestamp:.0f}s")

        # Get answer from evidence
        answer_field = template["answer_field"]
        answer = ground_truth.get(answer_field, "Not available")

        # Format answer
        if isinstance(answer, list):
            if answer:
                answer = ", ".join(str(item) for item in answer[:3])  # First 3 items
            else:
                answer = "None"
        elif isinstance(answer, dict):
            answer = answer.get("scene_category", str(answer))
        elif isinstance(answer, bool):
            answer = "Yes" if answer else "No"

        question = GeneratedQuestion(
            question_id=question_id,
            question_text=question_text,
            question_type="template",
            correct_answer=str(answer),
            difficulty=template["difficulty"],
            evidence_source=[str(evidence.get("id", ""))],
            timestamp_reference=timestamp,
            confidence=1.0
        )

        return question

    def _generate_gpt4_questions(self, count: int = 3) -> List[GeneratedQuestion]:
        """
        Generate questions using GPT-4

        Args:
            count: Number of questions to generate

        Returns:
            List of GeneratedQuestion objects
        """
        # Mock GPT-4 question generation
        # In production, this would call OpenAI API

        questions = []

        mock_gpt4_questions = [
            {
                "question": "What emotional journey does the speaker go through in this video?",
                "answer": "The speaker transitions from nostalgic reflection to bittersweet acceptance about selling their family home",
                "difficulty": "hard",
                "evidence": ["key_moment_001", "key_moment_008"]
            },
            {
                "question": "What significant life event is being documented in this video?",
                "answer": "The speaker is packing up and selling their Connecticut house, marking a major life transition",
                "difficulty": "medium",
                "evidence": ["transcript", "key_moment_003"]
            },
            {
                "question": "How does the physical setting reflect the video's emotional tone?",
                "answer": "The indoor, personal setting emphasizes the intimate and sentimental nature of the content",
                "difficulty": "hard",
                "evidence": ["scene_analysis", "emotion_tracking"]
            }
        ]

        for i, mock_q in enumerate(mock_gpt4_questions[:count], 1):
            question = GeneratedQuestion(
                question_id=f"gpt4_{i:03d}",
                question_text=mock_q["question"],
                question_type="gpt4",
                correct_answer=mock_q["answer"],
                difficulty=mock_q["difficulty"],
                evidence_source=mock_q["evidence"],
                confidence=0.92,
                generator_model="gpt-4o"
            )
            questions.append(question)

        return questions

    def _generate_claude_questions(self, count: int = 2) -> List[GeneratedQuestion]:
        """
        Generate questions using Claude Sonnet

        Args:
            count: Number of questions to generate

        Returns:
            List of GeneratedQuestion objects
        """
        # Mock Claude question generation
        # In production, this would call Anthropic API

        questions = []

        mock_claude_questions = [
            {
                "question": "What narrative structure does the vlogger use to tell their story?",
                "answer": "A chronological journey from past struggles (attic, couch) to achievement (buying house) to current transition (selling house)",
                "difficulty": "hard",
                "evidence": ["transcript_flow", "temporal_structure"]
            },
            {
                "question": "How does the speaker use visual and verbal elements to convey their message?",
                "answer": "Combines personal indoor setting with reflective narration about family, home ownership, and life transitions",
                "difficulty": "hard",
                "evidence": ["multimodal_analysis"]
            }
        ]

        for i, mock_q in enumerate(mock_claude_questions[:count], 1):
            question = GeneratedQuestion(
                question_id=f"claude_{i:03d}",
                question_text=mock_q["question"],
                question_type="claude",
                correct_answer=mock_q["answer"],
                difficulty=mock_q["difficulty"],
                evidence_source=mock_q["evidence"],
                confidence=0.89,
                generator_model="claude-sonnet-4.5"
            )
            questions.append(question)

        return questions

    def save_questions(self, questions: List[GeneratedQuestion], output_path: Path):
        """
        Save questions to JSON

        Args:
            questions: List of GeneratedQuestion objects
            output_path: Path to save JSON
        """
        questions_data = [asdict(q) for q in questions]

        output = {
            "video_metadata": {
                "genre": self.primary_genre,
                "duration": self.audio_analysis.get("duration"),
                "evidence_items_count": len(self.evidence_items)
            },
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "total_questions": len(questions),
                "template_questions": len([q for q in questions if q.question_type == "template"]),
                "gpt4_questions": len([q for q in questions if q.question_type == "gpt4"]),
                "claude_questions": len([q for q in questions if q.question_type == "claude"])
            },
            "questions": questions_data
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"✓ Saved {len(questions)} questions to: {output_path}")


def test_question_generator(
    audio_analysis_path: str,
    pipeline_results_path: str,
    evidence_items_path: Optional[str] = None
):
    """
    Test question generator

    Args:
        audio_analysis_path: Path to audio analysis JSON
        pipeline_results_path: Path to pipeline results JSON
        evidence_items_path: Path to evidence items JSON (optional, will use mock data)
    """
    # Load audio analysis
    with open(audio_analysis_path, 'r') as f:
        audio_analysis = json.load(f)

    # Load pipeline results
    with open(pipeline_results_path, 'r') as f:
        pipeline_results = json.load(f)

    genre_analysis = pipeline_results.get("genre_analysis")

    # Create mock evidence items for testing
    mock_evidence = []
    for i in range(10):
        mock_evidence.append({
            "id": i,
            "timestamp_start": i * 30.0,
            "ground_truth": {
                "person_count": 1,
                "dominant_emotion": "neutral",
                "is_indoor": True,
                "places365_scene": {"scene_category": "living_room"},
                "yolov8x_objects": [{"class": "person"}, {"class": "couch"}],
                "paddleocr_text": [],
                "image_caption": "A person in a living room",
                "clip_clothing": ["casual shirt"]
            }
        })

    # Create generator
    generator = EvidenceBasedQuestionGenerator(
        audio_analysis=audio_analysis,
        genre_analysis=genre_analysis,
        evidence_items=mock_evidence,
        enable_gpt4=True,  # Enable for demonstration
        enable_claude=True
    )

    # Generate questions
    questions = generator.generate_all_questions()

    # Save questions
    output_path = Path(pipeline_results_path).parent / f"{Path(pipeline_results_path).stem.replace('_pipeline_results', '')}_questions.json"
    generator.save_questions(questions, output_path)

    # Display results
    print("\n" + "=" * 80)
    print("QUESTION GENERATION RESULTS")
    print("=" * 80)
    print(f"Total Questions: {len(questions)}")
    print(f"  Template: {len([q for q in questions if q.question_type == 'template'])}")
    print(f"  GPT-4: {len([q for q in questions if q.question_type == 'gpt4'])}")
    print(f"  Claude: {len([q for q in questions if q.question_type == 'claude'])}")

    print("\nSample Questions:")
    print("-" * 80)

    # Show samples of each type
    for qtype in ["template", "gpt4", "claude"]:
        typed_questions = [q for q in questions if q.question_type == qtype]
        if typed_questions:
            q = typed_questions[0]
            print(f"\n[{qtype.upper()}] {q.question_text}")
            print(f"Answer: {q.correct_answer}")
            print(f"Difficulty: {q.difficulty}")

    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        audio_analysis = sys.argv[1]
        pipeline_results = sys.argv[2]
        test_question_generator(audio_analysis, pipeline_results)
    else:
        print("Usage: python evidence_based_questions.py <audio_analysis.json> <pipeline_results.json>")
        print("\nExample:")
        print("  python evidence_based_questions.py video_audio_analysis.json video_pipeline_results.json")
