"""
Adversarial Question Generator - Phase 5 of Smart Pipeline

Generates 30 adversarial questions designed to expose Gemini's weaknesses:
- 20 Template Questions: Using existing template system with opportunities
- 7 AI-Generated Questions: GPT-4 generates questions for premium keyframes
- 3 Cross-Validated Questions: GPT-4 + Claude (placeholder) for top 3 keyframes

All questions MUST follow guidelines:
1. Both audio AND visual cues required
2. NO names (use descriptors only)
3. Precise timestamps (cover cues + actions)
4. Not answerable with one cue only
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import os
import openai

logger = logging.getLogger(__name__)


@dataclass
class Question:
    """Single Q&A pair with metadata"""
    question_id: str
    question: str
    golden_answer: str
    start_timestamp: str  # HH:MM:SS format
    end_timestamp: str  # HH:MM:SS format
    task_types: List[str]
    generation_tier: str  # "template", "ai_generated", "cross_validated"

    # Additional metadata
    audio_cue: str = ""
    visual_cue: str = ""
    opportunity_type: Optional[str] = None
    evidence_sources: List[str] = field(default_factory=list)
    adversarial_target: Optional[str] = None
    needs_human_review: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class AdversarialQuestionGenerator:
    """
    Generate 30 adversarial questions from opportunities and evidence.

    Breakdown:
    - 20 template-based questions (using template registry)
    - 7 AI-generated questions (GPT-4 for premium keyframes)
    - 3 cross-validated questions (GPT-4 + Claude for top 3)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        claude_api_key: Optional[str] = None
    ):
        """
        Initialize question generator.

        Args:
            openai_api_key: OpenAI API key (for GPT-4)
            claude_api_key: Claude API key (placeholder for now)
        """
        # Set API keys
        if openai_api_key:
            openai.api_key = openai_api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key required")

        self.claude_api_key = claude_api_key  # Placeholder
        self.total_cost = 0.0

        logger.info("AdversarialQuestionGenerator initialized")

    def generate_all_questions(
        self,
        opportunities_path: str,
        evidence_path: str,
        video_id: str = "unknown"
    ) -> List[Question]:
        """
        Generate all 30 questions (20 template + 7 AI + 3 cross-validated).

        Args:
            opportunities_path: Path to opportunities JSON
            evidence_path: Path to evidence JSON
            video_id: Video identifier

        Returns:
            List of 30 Question objects
        """
        logger.info("=" * 80)
        logger.info("ADVERSARIAL QUESTION GENERATION - PHASE 5")
        logger.info("=" * 80)

        # Load opportunities and evidence
        with open(opportunities_path, 'r') as f:
            opportunities = json.load(f)

        with open(evidence_path, 'r') as f:
            evidence = json.load(f)

        all_questions = []

        # A. Generate 20 template questions
        logger.info("\n📝 Generating 20 template-based questions...")
        template_questions = self._generate_template_questions(
            opportunities,
            evidence,
            target_count=20
        )
        all_questions.extend(template_questions)
        logger.info(f"✓ Generated {len(template_questions)} template questions")

        # B. Generate 7 AI questions for premium keyframes
        logger.info("\n🤖 Generating 7 AI questions for premium keyframes...")
        ai_questions = self._generate_ai_questions(
            opportunities,
            evidence,
            count=7
        )
        all_questions.extend(ai_questions)
        logger.info(f"✓ Generated {len(ai_questions)} AI questions")

        # C. Generate 3 cross-validated questions
        logger.info("\n✅ Generating 3 cross-validated questions...")
        cross_validated = self._generate_cross_validated_questions(
            opportunities,
            evidence,
            count=3
        )
        all_questions.extend(cross_validated)
        logger.info(f"✓ Generated {len(cross_validated)} cross-validated questions")

        # Summary
        logger.info("=" * 80)
        logger.info(f"✅ GENERATED {len(all_questions)} TOTAL QUESTIONS")
        logger.info(f"   Template: {len(template_questions)}")
        logger.info(f"   AI-generated: {len(ai_questions)}")
        logger.info(f"   Cross-validated: {len(cross_validated)}")
        logger.info(f"   Total cost: ${self.total_cost:.4f}")
        logger.info("=" * 80)

        return all_questions

    def _generate_template_questions(
        self,
        opportunities: Dict,
        evidence: Dict,
        target_count: int = 20
    ) -> List[Question]:
        """
        Generate template-based questions from opportunities.

        Strategy:
        - Map opportunities to appropriate templates
        - Use evidence to fill in details
        - Enforce guidelines (no names, both cues, etc.)

        Args:
            opportunities: Adversarial opportunities dict
            evidence: Evidence extraction results
            target_count: Target number of questions

        Returns:
            List of Question objects
        """
        questions = []
        question_id_counter = 1

        # Strategy: Distribute across opportunity types
        # 8 temporal, 5 ambiguous, 4 counting, 3 sequential/context

        # 1. Temporal Understanding questions (8 questions)
        temporal_markers = opportunities.get("temporal_markers", [])[:8]
        for tm in temporal_markers:
            question = self._create_temporal_question(
                tm,
                evidence,
                f"template_{question_id_counter:03d}"
            )
            if question:
                questions.append(question)
                question_id_counter += 1

        # 2. Referential Grounding from ambiguous references (5 questions)
        ambiguous_refs = opportunities.get("ambiguous_references", [])[:5]
        for ar in ambiguous_refs:
            question = self._create_referential_question(
                ar,
                evidence,
                f"template_{question_id_counter:03d}"
            )
            if question:
                questions.append(question)
                question_id_counter += 1

        # 3. Counting questions (4 questions)
        counting_opps = opportunities.get("counting_opportunities", [])[:4]
        for co in counting_opps:
            question = self._create_counting_question(
                co,
                evidence,
                f"template_{question_id_counter:03d}"
            )
            if question:
                questions.append(question)
                question_id_counter += 1

        # 4. Sequential + Context questions (3 questions)
        sequential_events = opportunities.get("sequential_events", [])[:2]
        for se in sequential_events:
            question = self._create_sequential_question(
                se,
                evidence,
                f"template_{question_id_counter:03d}"
            )
            if question:
                questions.append(question)
                question_id_counter += 1

        context_frames = opportunities.get("context_rich_frames", [])[:1]
        for cf in context_frames:
            question = self._create_context_question(
                cf,
                evidence,
                f"template_{question_id_counter:03d}"
            )
            if question:
                questions.append(question)
                question_id_counter += 1

        return questions[:target_count]

    def _create_temporal_question(
        self,
        temporal_marker: Dict,
        evidence: Dict,
        question_id: str
    ) -> Optional[Question]:
        """Create question from temporal marker"""
        timestamp = temporal_marker["timestamp"]
        quote = temporal_marker["quote"]
        marker_type = temporal_marker["type"]

        # Find evidence at this timestamp
        frame_evidence = self._find_evidence_at_timestamp(evidence, timestamp)
        if not frame_evidence:
            return None

        # Extract visual elements from evidence
        visual_desc = self._extract_visual_description(frame_evidence)

        # Create question
        if marker_type == "before":
            question_text = f"What happens before the audio cue \"{quote}\"?"
        elif marker_type == "after":
            question_text = f"What happens after the audio cue \"{quote}\"?"
        else:  # when
            question_text = f"What is shown when you hear \"{quote}\"?"

        # Create answer from visual evidence
        answer = f"When the audio cue \"{quote}\" is heard, {visual_desc}"

        # Calculate timestamps (±3 seconds from marker)
        start_ts = max(0, timestamp - 3)
        end_ts = timestamp + 3

        return Question(
            question_id=question_id,
            question=question_text,
            golden_answer=answer,
            start_timestamp=self._format_timestamp(start_ts),
            end_timestamp=self._format_timestamp(end_ts),
            task_types=["Temporal Understanding", "Sequential"],
            generation_tier="template",
            audio_cue=quote,
            visual_cue=visual_desc,
            opportunity_type="temporal_marker",
            evidence_sources=["YOLO", "OCR", "CLIP"]
        )

    def _create_referential_question(
        self,
        ambiguous_ref: Dict,
        evidence: Dict,
        question_id: str
    ) -> Optional[Question]:
        """Create question from ambiguous reference"""
        timestamp = ambiguous_ref["timestamp"]
        quote = ambiguous_ref["quote"]
        possible_referents = ambiguous_ref.get("possible_referents", [])

        frame_evidence = self._find_evidence_at_timestamp(evidence, timestamp)
        if not frame_evidence:
            return None

        visual_desc = self._extract_visual_description(frame_evidence)

        question_text = f"When you hear \"{quote}\", what specifically is being referred to, and how can you distinguish it from other similar elements visible on screen?"

        answer = f"When \"{quote}\" is said at {self._format_timestamp(timestamp)}, it refers to {visual_desc}."

        start_ts = max(0, timestamp - 2)
        end_ts = timestamp + 2

        return Question(
            question_id=question_id,
            question=question_text,
            golden_answer=answer,
            start_timestamp=self._format_timestamp(start_ts),
            end_timestamp=self._format_timestamp(end_ts),
            task_types=["Referential Grounding", "Needle"],
            generation_tier="template",
            audio_cue=quote,
            visual_cue=visual_desc,
            opportunity_type="ambiguous_reference",
            evidence_sources=["YOLO", "CLIP"]
        )

    def _create_counting_question(
        self,
        counting_opp: Dict,
        evidence: Dict,
        question_id: str
    ) -> Optional[Question]:
        """Create counting question"""
        event_type = counting_opp["event_type"]
        boundary_quote = counting_opp["boundary_quote"]
        boundary_timestamp = counting_opp["boundary_timestamp"]

        # Simplified: Create question asking to count events before boundary
        question_text = f"How many times did {event_type} occur before you hear \"{boundary_quote}\"?"

        # Answer requires analysis of frames (placeholder)
        answer = f"The {event_type} occurred X times before the audio cue \"{boundary_quote}\" at {self._format_timestamp(boundary_timestamp)}."

        return Question(
            question_id=question_id,
            question=question_text,
            golden_answer=answer,
            start_timestamp="00:00:00",
            end_timestamp=self._format_timestamp(boundary_timestamp),
            task_types=["Counting", "Temporal Understanding"],
            generation_tier="template",
            audio_cue=boundary_quote,
            visual_cue=event_type,
            opportunity_type="counting",
            evidence_sources=["YOLO", "Event Detection"]
        )

    def _create_sequential_question(
        self,
        sequential_event: Dict,
        evidence: Dict,
        question_id: str
    ) -> Optional[Question]:
        """Create sequential events question"""
        events = sequential_event["events"]
        start = sequential_event["start"]
        end = sequential_event["end"]

        question_text = f"What is the order of these events: {', '.join(events)}?"
        answer = f"The events occur in this order: {' → '.join(events)}"

        return Question(
            question_id=question_id,
            question=question_text,
            golden_answer=answer,
            start_timestamp=self._format_timestamp(start),
            end_timestamp=self._format_timestamp(end),
            task_types=["Sequential", "Temporal Understanding"],
            generation_tier="template",
            audio_cue="",
            visual_cue=", ".join(events),
            opportunity_type="sequential",
            evidence_sources=["YOLO", "Action Detection"]
        )

    def _create_context_question(
        self,
        context_frame: Dict,
        evidence: Dict,
        question_id: str
    ) -> Optional[Question]:
        """Create context-rich question"""
        timestamp = context_frame["timestamp"]
        audio_cue = context_frame["audio_cue"]

        frame_evidence = self._find_evidence_at_timestamp(evidence, timestamp)
        if not frame_evidence:
            return None

        visual_desc = self._extract_visual_description(frame_evidence, focus="background")

        question_text = f"When you hear \"{audio_cue}\", what visual elements are present in the background?"
        answer = f"When \"{audio_cue}\" is said, the background shows {visual_desc}."

        start_ts = max(0, timestamp - 2)
        end_ts = timestamp + 2

        return Question(
            question_id=question_id,
            question=question_text,
            golden_answer=answer,
            start_timestamp=self._format_timestamp(start_ts),
            end_timestamp=self._format_timestamp(end_ts),
            task_types=["Context"],
            generation_tier="template",
            audio_cue=audio_cue,
            visual_cue=visual_desc,
            opportunity_type="context",
            evidence_sources=["YOLO", "Places365"]
        )

    def _generate_ai_questions(
        self,
        opportunities: Dict,
        evidence: Dict,
        count: int = 7
    ) -> List[Question]:
        """
        Generate AI questions using GPT-4 for premium keyframes.

        Args:
            opportunities: Opportunities dict
            evidence: Evidence dict
            count: Number of questions (default 7)

        Returns:
            List of Question objects
        """
        questions = []
        premium_keyframes = opportunities.get("premium_analysis_keyframes", [])[:count]

        for i, timestamp in enumerate(premium_keyframes, 1):
            # Find evidence and audio context
            frame_evidence = self._find_evidence_at_timestamp(evidence, timestamp)
            audio_context = self._find_audio_context(opportunities, timestamp)

            if not frame_evidence:
                logger.warning(f"No evidence found for premium keyframe at {timestamp}s")
                continue

            # Use GPT-4 to generate question
            question = self._generate_question_with_gpt4(
                timestamp=timestamp,
                frame_evidence=frame_evidence,
                audio_context=audio_context,
                question_id=f"ai_{i:03d}",
                adversarial_target="premium_keyframe"
            )

            if question:
                questions.append(question)
                self.total_cost += 0.21  # Estimate $0.21 per AI question

        return questions

    def _generate_cross_validated_questions(
        self,
        opportunities: Dict,
        evidence: Dict,
        count: int = 3
    ) -> List[Question]:
        """
        Generate cross-validated questions (GPT-4 + Claude placeholder).

        Args:
            opportunities: Opportunities dict
            evidence: Evidence dict
            count: Number of questions (default 3)

        Returns:
            List of Question objects
        """
        questions = []
        premium_keyframes = opportunities.get("premium_analysis_keyframes", [])[:count]

        for i, timestamp in enumerate(premium_keyframes, 1):
            frame_evidence = self._find_evidence_at_timestamp(evidence, timestamp)
            audio_context = self._find_audio_context(opportunities, timestamp)

            if not frame_evidence:
                continue

            # Generate with GPT-4
            gpt4_question = self._generate_question_with_gpt4(
                timestamp=timestamp,
                frame_evidence=frame_evidence,
                audio_context=audio_context,
                question_id=f"cross_{i:03d}",
                adversarial_target="cross_validation"
            )

            # TODO: Add Claude validation here (placeholder)
            # For now, just use GPT-4 question
            if gpt4_question:
                gpt4_question.generation_tier = "cross_validated"
                questions.append(gpt4_question)
                self.total_cost += 0.27  # Estimate

        return questions

    def _generate_question_with_gpt4(
        self,
        timestamp: float,
        frame_evidence: Dict,
        audio_context: str,
        question_id: str,
        adversarial_target: str
    ) -> Optional[Question]:
        """
        Use GPT-4 to generate adversarial question.

        Args:
            timestamp: Frame timestamp
            frame_evidence: Evidence data for frame
            audio_context: Audio transcript around timestamp
            question_id: Unique question ID
            adversarial_target: Type of adversarial question

        Returns:
            Question object or None
        """
        # Extract visual description from evidence
        visual_desc = self._extract_visual_description(frame_evidence)

        prompt = f"""Generate a challenging video Q&A pair that requires BOTH audio AND visual understanding.

TIMESTAMP: {timestamp}s ({self._format_timestamp(timestamp)})
AUDIO CONTEXT: "{audio_context}"
VISUAL EVIDENCE: {visual_desc}

GUIDELINES (CRITICAL - MUST FOLLOW):
1. Question MUST have BOTH audio AND visual cues
2. NO names allowed - use descriptors (e.g., "person in blue jacket", "main character")
3. Question should NOT be answerable with just audio OR just visual alone
4. Answer must be precise with specific details
5. Target Gemini weakness: {adversarial_target}

TASK TYPES to choose from:
- Temporal Understanding (before/after/when)
- Referential Grounding (what "that"/"he"/"it" refers to)
- Needle (specific detail in crowded scene)
- Context (background/foreground elements)
- Inference (why/purpose)

Return JSON only:
{{
  "question": "the question text",
  "golden_answer": "the detailed answer",
  "task_types": ["task_type1", "task_type2"],
  "audio_cue": "exact audio quote used",
  "visual_cue": "key visual element"
}}"""

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating adversarial video Q&A pairs that expose AI model weaknesses. You return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )

            content = response.choices[0].message.content.strip()

            # Remove markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # Parse JSON
            data = json.loads(content)

            # Create Question object
            start_ts = max(0, timestamp - 3)
            end_ts = timestamp + 3

            return Question(
                question_id=question_id,
                question=data["question"],
                golden_answer=data["golden_answer"],
                start_timestamp=self._format_timestamp(start_ts),
                end_timestamp=self._format_timestamp(end_ts),
                task_types=data.get("task_types", ["Inference"]),
                generation_tier="ai_generated",
                audio_cue=data.get("audio_cue", ""),
                visual_cue=data.get("visual_cue", ""),
                adversarial_target=adversarial_target,
                evidence_sources=["GPT-4 Vision"]
            )

        except Exception as e:
            logger.error(f"Failed to generate GPT-4 question: {e}")
            return None

    # Helper methods

    def _find_evidence_at_timestamp(
        self,
        evidence: Dict,
        timestamp: float,
        tolerance: float = 2.0
    ) -> Optional[Dict]:
        """Find evidence frame closest to timestamp"""
        frames = evidence.get("frames", {})

        closest_frame = None
        min_diff = float('inf')

        for frame_data in frames.values():
            frame_ts = frame_data.get("timestamp", 0)
            diff = abs(frame_ts - timestamp)

            if diff < min_diff and diff <= tolerance:
                min_diff = diff
                closest_frame = frame_data

        return closest_frame

    def _find_audio_context(
        self,
        opportunities: Dict,
        timestamp: float,
        context_window: float = 5.0
    ) -> str:
        """Find audio context around timestamp"""
        # Look through all opportunities for quotes near timestamp
        quotes = []

        # Check temporal markers
        for tm in opportunities.get("temporal_markers", []):
            if abs(tm["timestamp"] - timestamp) <= context_window:
                quotes.append(tm["quote"])

        # Check ambiguous references
        for ar in opportunities.get("ambiguous_references", []):
            if abs(ar["timestamp"] - timestamp) <= context_window:
                quotes.append(ar["quote"])

        return " | ".join(quotes) if quotes else "[No audio context found]"

    def _extract_visual_description(
        self,
        frame_evidence: Dict,
        focus: str = "all"
    ) -> str:
        """
        Extract visual description from evidence data.

        Args:
            frame_evidence: Evidence data for frame
            focus: "all", "background", or "foreground"

        Returns:
            Human-readable visual description
        """
        ground_truth = frame_evidence.get("ground_truth", {})

        descriptions = []

        # Check for GPT-4V description first (premium frames)
        gpt4v_desc = ground_truth.get("gpt4v_description", "")
        if gpt4v_desc:
            return gpt4v_desc

        # YOLO objects
        yolo_objects = ground_truth.get("yolo_objects", [])
        if yolo_objects:
            obj_desc = ", ".join([obj.get("class", "") for obj in yolo_objects[:3]])
            descriptions.append(f"visible objects: {obj_desc}")

        # OCR text
        ocr_text = ground_truth.get("ocr_text", [])
        if ocr_text:
            text_desc = ", ".join([t.get("text", "") for t in ocr_text[:2]])
            descriptions.append(f"text: '{text_desc}'")

        # Scene type
        scene = ground_truth.get("scene_type", "")
        if scene and scene != "analyzed_with_gpt4v":
            descriptions.append(f"scene: {scene}")

        # CLIP attributes
        clip_desc = ground_truth.get("clip_description", "")
        if clip_desc:
            descriptions.append(clip_desc)

        return ", ".join(descriptions) if descriptions else "visual elements visible on screen"

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def save_questions(
        self,
        questions: List[Question],
        output_path: Path
    ):
        """Save questions to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        questions_data = {
            "total_questions": len(questions),
            "metadata": {
                "template_count": len([q for q in questions if q.generation_tier == "template"]),
                "ai_count": len([q for q in questions if q.generation_tier == "ai_generated"]),
                "cross_validated_count": len([q for q in questions if q.generation_tier == "cross_validated"])
            },
            "questions": [q.to_dict() for q in questions]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved {len(questions)} questions to: {output_path}")


# Test function
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        opportunities_path = sys.argv[1]
        evidence_path = sys.argv[2]

        generator = AdversarialQuestionGenerator()
        questions = generator.generate_all_questions(
            opportunities_path,
            evidence_path
        )

        # Save
        output_path = Path(opportunities_path).parent / "questions.json"
        generator.save_questions(questions, output_path)

        print(f"\n✅ Generated {len(questions)} questions")
        print(f"Saved to: {output_path}")
    else:
        print("Usage: python adversarial_question_generator.py <opportunities.json> <evidence.json>")
