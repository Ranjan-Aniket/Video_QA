"""
Question Regenerator - Regenerate Rejected Questions with Feedback

Handles question regeneration with max 3 attempts per question.
Uses feedback to improve next generation attempt.
"""
import logging
from typing import Optional, Dict, Any
from database.operations import QuestionOperations, VideoOperations
from database.schema import Question
from generation.tier1_deterministic import Tier1Generator
from generation.tier2_llama_api import Tier2LlamaGenerator
from generation.tier3_creative import Tier3Generator

logger = logging.getLogger(__name__)


class QuestionRegenerator:
    """Regenerate rejected questions with feedback"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.tier1_gen = Tier1Generator()
        self.tier2_gen = Tier2LlamaGenerator(api_key=openai_api_key) if openai_api_key else None
        self.tier3_gen = Tier3Generator(api_key=openai_api_key) if openai_api_key else None
    
    async def regenerate_question(
        self,
        question_id: int,
        feedback: Optional[str] = None
    ) -> Question:
        """
        Regenerate a rejected question
        
        Args:
            question_id: Original question database ID
            feedback: Rejection feedback to incorporate
            
        Returns:
            Newly generated Question object
        """
        # Get original question
        original = QuestionOperations.get_question_by_id(question_id)
        if not original:
            raise ValueError(f"Question {question_id} not found")
        
        # Get video for evidence
        video = VideoOperations.get_video_by_id(original.video_id)
        
        # Extract evidence (lightweight)
        evidence = self._extract_evidence(video)
        
        # Determine which tier generator to use
        tier = original.generation_tier
        
        logger.info(
            f"Regenerating question {question_id} (tier={tier}, "
            f"attempt={original.generation_attempt + 1}, feedback={feedback[:50] if feedback else 'None'})"
        )
        
        # Generate new question with feedback
        if tier == 'template':
            new_questions = self.tier1_gen.generate(
                evidence=evidence,
                target_count=1,
                feedback=feedback,
                avoid_similar_to=original.question_text
            )
        elif tier == 'llama':
            if not self.tier2_gen:
                raise ValueError("Tier 2 generator not initialized (missing API key)")
            new_questions = self.tier2_gen.generate(
                evidence=evidence,
                target_count=1,
                feedback=feedback,
                avoid_similar_to=original.question_text
            )
        elif tier == 'gpt4mini':
            if not self.tier3_gen:
                raise ValueError("Tier 3 generator not initialized (missing API key)")
            new_questions = self.tier3_gen.generate(
                evidence=evidence,
                target_count=1,
                feedback=feedback,
                avoid_similar_to=original.question_text
            )
        else:
            raise ValueError(f"Unknown generation tier: {tier}")
        
        if not new_questions:
            raise ValueError("Failed to generate new question")
        
        new_q = new_questions[0]
        
        # Save to database
        saved_question = QuestionOperations.create_question(
            question_id=f"{original.question_id}_regen_{original.generation_attempt + 1}",
            video_id=original.video_id,
            question_text=new_q.question_text,
            golden_answer=new_q.golden_answer,
            generation_tier=tier,
            task_type=new_q.question_types[0].value if hasattr(new_q, 'question_types') and new_q.question_types else original.task_type,
            template_name=getattr(new_q, 'template_name', original.template_name),
            start_seconds=getattr(new_q, 'start_timestamp', original.start_seconds),
            end_seconds=getattr(new_q, 'end_timestamp', original.end_seconds),
            audio_cues=getattr(new_q, 'audio_cues', original.audio_cues),
            visual_cues=getattr(new_q, 'visual_cues', original.visual_cues),
            evidence_refs=getattr(new_q, 'evidence_refs', original.evidence_refs),
            confidence_score=getattr(new_q, 'complexity_score', original.confidence_score),
            generation_attempt=original.generation_attempt + 1,
            parent_question_id=original.id,
            is_regeneration=True
        )
        
        logger.info(
            f"Regenerated question saved: {saved_question.question_id} "
            f"(attempt {saved_question.generation_attempt})"
        )
        
        return saved_question
    
    def _extract_evidence(self, video) -> Any:
        """Extract lightweight evidence for generation"""
        from templates.base import EvidenceDatabase
        
        # TODO: Implement proper evidence extraction
        return EvidenceDatabase(
            video_id=video.video_id,
            duration=video.duration or 60.0,
            transcript_segments=[],
            music_segments=[],
            sound_effects=[],
            ambient_sounds=[],
            tone_changes=[],
            person_detections=[],
            object_detections=[],
            scene_detections=[],
            ocr_detections=[],
            action_detections=[],
            scene_changes=[],
            event_timeline=[],
            character_names=[],
            team_names=[],
            media_names=[],
            brand_names=[]
        )