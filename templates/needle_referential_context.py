"""
Needle, Referential Grounding, and Context Templates

Needle: Find specific details at precise moments
Referential Grounding: Connect audio to visual at specific time
Context: Background/foreground elements during events

Examples from taxonomy:

Needle:
- "Describe the graphic that pops up when the man says 'Definitely check it out'"
- "Who is the player that hits his first three when a minute and ten seconds left?"

Referential Grounding:
- "Who are the two people visually present when the man says 'What is the protocol'?"
- "What creates the distinct humming and buzzing sound in the video?"

Context:
- "What does the black and white billboard say at the top left when person discusses takeaway?"
- "What visual elements are present in background when person says 'That's how we're gonna win'?"
"""

from typing import Optional, List, Dict
from templates.base import (
    QuestionTemplate, GeneratedQuestion, EvidenceDatabase,
    QuestionType, Cue, CueType
)
from templates.descriptor_generator import DescriptorGenerator, VisualEvidence


class NeedleTemplate(QuestionTemplate):
    """Find specific detail at specific moment"""
    
    def get_question_types(self) -> List[QuestionType]:
        return [QuestionType.NEEDLE]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        """Need OCR or specific visual details"""
        return (
            len(evidence.ocr_detections) >= 2 and
            len(evidence.transcript_segments) >= 3
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate needle question
        
        Strategy:
        1. Find OCR text or specific visual detail
        2. Find audio cue near that moment
        3. Ask what visual detail appears during audio
        """
        # Filter OCR not in intro/outro
        valid_ocr = [
            ocr for ocr in evidence.ocr_detections
            if not self.is_in_intro_outro(ocr['timestamp'], evidence)
        ]
        
        if not valid_ocr:
            return None
        
        # Select OCR with nearby audio
        for ocr in valid_ocr:
            audio_segment = self.find_audio_near_timestamp(
                ocr['timestamp'],
                evidence,
                max_distance=3.0
            )
            if not audio_segment:
                continue
            
            # Get audio text (no truncation per guideline 12)
            audio_text = self.get_audio_quote_for_question(audio_segment, max_length=70)

            # Get OCR location descriptor
            location = ocr.get('location', 'screen')

            # Generate question
            question = (
                f'What text appears on the {location} when the audio cue "{audio_text}" is heard?'
            )
            
            answer = f'The text "{ocr["text"]}" appears on the {location}.'
            
            # Create cues
            audio_cue = Cue(
                cue_type=CueType.AUDIO_SPEECH,
                content=audio_text,
                timestamp=audio_segment['start']
            )
            
            visual_cue = Cue(
                cue_type=CueType.VISUAL_TEXT,
                content=ocr['text'],
                timestamp=ocr['timestamp'],
                confidence=ocr.get('confidence', 1.0)
            )
            
            # Timestamps
            start_ts = min(audio_segment['start'], ocr['timestamp'])
            end_ts = max(audio_segment['start'] + 2.0, ocr['timestamp'] + 1.0)
            
            # Validate
            if not self.validate_no_names(question, evidence):
                continue
            if not self.validate_no_names(answer, evidence):
                continue
            
            return GeneratedQuestion(
                question_text=question,
                golden_answer=answer,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                audio_cues=[audio_cue],
                visual_cues=[visual_cue],
                question_types=[QuestionType.NEEDLE],
                generation_tier=1,
                template_name=self.name,
                complexity_score=0.7,
                evidence_refs=[
                    f"audio:{audio_segment['start']}",
                    f"ocr:{ocr['timestamp']}"
                ]
            )
        
        return None


class ReferentialGroundingTemplate(QuestionTemplate):
    """Connect audio to visual at specific time"""
    
    def get_question_types(self) -> List[QuestionType]:
        return [QuestionType.REFERENTIAL_GROUNDING]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        """Need people/objects and audio"""
        return (
            len(evidence.person_detections) >= 2 and
            len(evidence.transcript_segments) >= 3
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate referential grounding question
        
        Strategy:
        1. Find audio segment
        2. Find who/what is visible at that moment
        3. Ask who is present or what creates sound
        """
        # Find audio with multiple people visible
        for audio_segment in evidence.transcript_segments:
            # Skip intro/outro
            if self.is_in_intro_outro(audio_segment['start'], evidence):
                continue
            
            # Find people visible during audio
            people_at_time = [
                p for p in evidence.person_detections
                if abs(p['timestamp'] - audio_segment['start']) < 2.0
            ]
            
            if len(people_at_time) >= 2:
                # Generate "who is present" question
                return self._generate_who_is_present(
                    audio_segment,
                    people_at_time[:2],  # Use first 2
                    evidence
                )
        
        # Try "what creates sound" questions
        for sound_effect in evidence.sound_effects:
            # Skip intro/outro
            if self.is_in_intro_outro(sound_effect['timestamp'], evidence):
                continue
            
            # Find what object is visible
            obj = self.find_visual_near_timestamp(
                sound_effect['timestamp'],
                evidence,
                visual_type="object"
            )
            if obj:
                return self._generate_what_creates_sound(
                    sound_effect,
                    obj,
                    evidence
                )
        
        return None
    
    def _generate_who_is_present(
        self,
        audio_segment: Dict,
        people: List[Dict],
        evidence: EvidenceDatabase
    ) -> Optional[GeneratedQuestion]:
        """Generate 'who is present' question"""
        
        # Generate descriptors for people
        descriptors = []
        for person in people:
            visual_evidence = VisualEvidence(
                timestamp=person['timestamp'],
                bbox=person['bbox'],
                **person.get('attributes', {})
            )
            desc = DescriptorGenerator.generate_person_descriptor(visual_evidence)
            descriptors.append(desc)
        
        # Get audio text (no truncation per guideline 12)
        audio_text = self.get_audio_quote_for_question(audio_segment, max_length=70)

        # Generate question - Use "which people" instead of "who"
        question = (
            f'Which people are visible on screen when the audio cue "{audio_text}" is heard?'
        )

        # Answer - Avoid pronoun patterns
        answer = f'The {descriptors[0]} and the {descriptors[1]} are visible on screen.'
        
        # Create cues
        audio_cue = Cue(
            cue_type=CueType.AUDIO_SPEECH,
            content=audio_text,
            timestamp=audio_segment['start']
        )
        
        visual_cues = [
            Cue(
                cue_type=CueType.VISUAL_PERSON,
                content=desc,
                timestamp=person['timestamp']
            )
            for desc, person in zip(descriptors, people)
        ]
        
        # Timestamps
        start_ts = audio_segment['start']
        end_ts = audio_segment['start'] + 3.0
        
        # Validate
        if not self.validate_no_names(question, evidence):
            return None
        if not self.validate_no_names(answer, evidence):
            return None
        
        return GeneratedQuestion(
            question_text=question,
            golden_answer=answer,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            audio_cues=[audio_cue],
            visual_cues=visual_cues,
            question_types=[QuestionType.REFERENTIAL_GROUNDING],
            generation_tier=1,
            template_name=self.name,
            complexity_score=0.6,
            evidence_refs=[
                f"audio:{audio_segment['start']}",
                *[f"person:{p['timestamp']}" for p in people]
            ]
        )
    
    def _generate_what_creates_sound(
        self,
        sound_effect: Dict,
        obj: Dict,
        evidence: EvidenceDatabase
    ) -> Optional[GeneratedQuestion]:
        """Generate 'what creates sound' question"""
        
        # Generate object descriptor
        visual_evidence = VisualEvidence(
            timestamp=obj['timestamp'],
            bbox=obj['bbox'],
            object_class=obj['object_class'],
            color=obj.get('color')
        )
        obj_desc = DescriptorGenerator.generate_object_descriptor(visual_evidence)
        
        sound_type = sound_effect['sound_type']
        
        # Generate question
        question = (
            f'What object creates the {sound_type} sound at this moment in the video?'
        )
        
        answer = f'The {obj_desc} creates the {sound_type} sound.'
        
        # Create cues
        audio_cue = Cue(
            cue_type=CueType.AUDIO_SOUND_EFFECT,
            content=sound_type,
            timestamp=sound_effect['timestamp']
        )
        
        visual_cue = Cue(
            cue_type=CueType.VISUAL_OBJECT,
            content=obj_desc,
            timestamp=obj['timestamp']
        )
        
        # Timestamps
        start_ts = sound_effect['timestamp'] - 1.0
        end_ts = sound_effect['timestamp'] + 1.0
        
        # Validate
        if not self.validate_no_names(question, evidence):
            return None
        if not self.validate_no_names(answer, evidence):
            return None
        
        return GeneratedQuestion(
            question_text=question,
            golden_answer=answer,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            audio_cues=[audio_cue],
            visual_cues=[visual_cue],
            question_types=[QuestionType.REFERENTIAL_GROUNDING],
            generation_tier=1,
            template_name=self.name,
            complexity_score=0.7,
            evidence_refs=[
                f"sound:{sound_effect['timestamp']}",
                f"object:{obj['timestamp']}"
            ]
        )


class ContextTemplate(QuestionTemplate):
    """Background/foreground elements during events"""
    
    def get_question_types(self) -> List[QuestionType]:
        return [QuestionType.CONTEXT]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        """Need scene detections and audio"""
        return (
            len(evidence.scene_detections) >= 2 and
            len(evidence.transcript_segments) >= 3
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate context question
        
        Strategy:
        1. Find audio segment
        2. Identify background/foreground elements at that time
        3. Ask what visual elements are present
        """
        for audio_segment in evidence.transcript_segments:
            # Skip intro/outro
            if self.is_in_intro_outro(audio_segment['start'], evidence):
                continue
            
            # Find scene at this time
            scene = self.find_visual_near_timestamp(
                audio_segment['start'],
                evidence,
                visual_type="scene"
            )
            if not scene:
                continue
            
            # Find objects in background
            objects_at_time = [
                obj for obj in evidence.object_detections
                if abs(obj['timestamp'] - audio_segment['start']) < 2.0
            ]
            
            if len(objects_at_time) < 2:
                continue
            
            # Get audio text (no truncation per guideline 12)
            audio_text = self.get_audio_quote_for_question(audio_segment, max_length=70)

            # Generate object descriptors
            obj_descriptors = []
            for obj in objects_at_time[:3]:  # Max 3 objects
                visual_evidence = VisualEvidence(
                    timestamp=obj['timestamp'],
                    bbox=obj['bbox'],
                    object_class=obj['object_class'],
                    color=obj.get('color')
                )
                desc = DescriptorGenerator.generate_object_descriptor(visual_evidence)
                obj_descriptors.append(desc)

            # Generate question
            question = (
                f'What visual elements are present in the background when '
                f'the audio cue "{audio_text}" is heard?'
            )
            
            answer = (
                f'In the background, we can see {", ".join(obj_descriptors[:-1])}, '
                f'and {obj_descriptors[-1]}.'
            )
            
            # Create cues
            audio_cue = Cue(
                cue_type=CueType.AUDIO_SPEECH,
                content=audio_text,
                timestamp=audio_segment['start']
            )
            
            visual_cues = [
                Cue(
                    cue_type=CueType.VISUAL_OBJECT,
                    content=desc,
                    timestamp=obj['timestamp']
                )
                for desc, obj in zip(obj_descriptors, objects_at_time[:3])
            ]
            
            # Timestamps
            start_ts = audio_segment['start']
            end_ts = audio_segment['start'] + 3.0
            
            # Validate
            if not self.validate_no_names(question, evidence):
                continue
            if not self.validate_no_names(answer, evidence):
                continue
            
            return GeneratedQuestion(
                question_text=question,
                golden_answer=answer,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                audio_cues=[audio_cue],
                visual_cues=visual_cues,
                question_types=[QuestionType.CONTEXT],
                generation_tier=1,
                template_name=self.name,
                complexity_score=0.6,
                evidence_refs=[
                    f"audio:{audio_segment['start']}",
                    *[f"object:{o['timestamp']}" for o in objects_at_time[:3]]
                ]
            )
        
        return None