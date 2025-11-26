"""
Counting Template

Generates counting questions that require precise enumeration with both
audio and visual cues.

Examples from taxonomy:
- "How many 'teaspoons of sugar' does person add into the small metal pitcher, 
   after placing the moka pot onto the stove?"
- "Throughout the video, how many counts does the dance instructor use to 
   choreograph the structure for the line dancing?"
- "How many times did team score from the start of the game till when player 
   was mentioned by an announcer?"

CRITICAL GUIDELINES:
- Must have BOTH audio and visual cues
- Count must be exact (from evidence, not estimated)
- Must use descriptors, not names
- Challenging counting (not obvious, requires attention)
"""

from typing import Optional, List, Dict
from templates.base import (
    QuestionTemplate, GeneratedQuestion, EvidenceDatabase,
    QuestionType, Cue, CueType
)
from templates.descriptor_generator import DescriptorGenerator, VisualEvidence


class CountingTemplate(QuestionTemplate):
    """Generate counting questions"""
    
    def get_question_types(self) -> List[QuestionType]:
        return [QuestionType.COUNTING]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        """
        Can apply if:
        - Multiple occurrences of same object/action (at least 3)
        - Has both audio and visual cues
        - Not in intro/outro
        """
        # Need multiple objects or actions to count
        if len(evidence.object_detections) < 3:
            return False
        
        # Need audio cues
        if len(evidence.transcript_segments) < 2:
            return False
        
        return True
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate counting question
        
        Strategy:
        1. Find repeated object/action with audio cue
        2. Count occurrences in specific time window
        3. Generate question with both cues
        """
        # Group objects by class
        object_groups: Dict[str, List[Dict]] = {}
        for obj in evidence.object_detections:
            obj_class = obj['object_class']
            if obj_class not in object_groups:
                object_groups[obj_class] = []
            object_groups[obj_class].append(obj)
        
        # Find objects with multiple occurrences
        for obj_class, detections in object_groups.items():
            if len(detections) < 3:  # Need at least 3 for interesting count
                continue
            
            # Try to find audio cue that creates time boundary
            for audio_segment in evidence.transcript_segments:
                # Skip intro/outro
                if self.is_in_intro_outro(audio_segment['start'], evidence):
                    continue
                
                # Count objects after this audio cue
                count_after = sum(
                    1 for obj in detections 
                    if obj['timestamp'] > audio_segment['start']
                    and not self.is_in_intro_outro(obj['timestamp'], evidence)
                )
                
                if count_after >= 2:  # At least 2 to count
                    return self._generate_count_after_audio(
                        obj_class=obj_class,
                        count=count_after,
                        audio_segment=audio_segment,
                        detections=[d for d in detections if d['timestamp'] > audio_segment['start']],
                        evidence=evidence
                    )
        
        # Try action counting
        return self._generate_action_count(evidence)
    
    def _generate_count_after_audio(
        self,
        obj_class: str,
        count: int,
        audio_segment: Dict,
        detections: List[Dict],
        evidence: EvidenceDatabase
    ) -> Optional[GeneratedQuestion]:
        """Generate question counting objects after audio cue"""
        
        # Generate descriptor for object
        first_detection = detections[0]
        visual_evidence = VisualEvidence(
            timestamp=first_detection['timestamp'],
            bbox=first_detection['bbox'],
            object_class=obj_class,
            color=first_detection.get('color'),
            size=first_detection.get('size')
        )
        object_descriptor = DescriptorGenerator.generate_object_descriptor(visual_evidence)
        
        # Get audio cue text (no truncation per guideline 12)
        audio_text = self.get_audio_quote_for_question(audio_segment, max_length=60)

        # Generate question
        question = (
            f'How many {object_descriptor}s appear on screen after the audio cue '
            f'"{audio_text}"?'
        )
        
        # Generate answer
        answer = f'After the audio cue, {count} {object_descriptor}s appear on screen.'
        
        # Create cues
        audio_cue = Cue(
            cue_type=CueType.AUDIO_SPEECH,
            content=audio_text,
            timestamp=audio_segment['start'],
            confidence=audio_segment.get('confidence', 1.0)
        )
        
        visual_cue = Cue(
            cue_type=CueType.VISUAL_OBJECT,
            content=f"{count} {object_descriptor}s",
            timestamp=detections[0]['timestamp'],
            confidence=min(d.get('confidence', 1.0) for d in detections)
        )
        
        # Calculate timestamps
        start_ts = audio_segment['start']
        end_ts = detections[-1]['timestamp'] + 1.0  # Include last object
        
        # Validate no names
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
            question_types=[QuestionType.COUNTING],
            generation_tier=1,
            template_name=self.name,
            complexity_score=0.6,
            evidence_refs=[
                f"audio:{audio_segment['start']}",
                *[f"object:{d['timestamp']}" for d in detections]
            ]
        )
    
    def _generate_action_count(
        self,
        evidence: EvidenceDatabase
    ) -> Optional[GeneratedQuestion]:
        """Generate question counting actions"""
        
        # Group actions by type
        action_groups: Dict[str, List[Dict]] = {}
        for action in evidence.action_detections:
            action_type = action['action']
            if action_type not in action_groups:
                action_groups[action_type] = []
            action_groups[action_type].append(action)
        
        # Find actions with multiple occurrences
        for action_type, detections in action_groups.items():
            if len(detections) < 3:
                continue
            
            # Find person doing action
            person_id = detections[0].get('person_id')
            if not person_id:
                continue
            
            # Get person descriptor
            person_detections = [
                p for p in evidence.person_detections 
                if p['person_id'] == person_id
            ]
            if not person_detections:
                continue
            
            person_det = person_detections[0]
            visual_evidence = VisualEvidence(
                timestamp=person_det['timestamp'],
                bbox=person_det['bbox'],
                **person_det.get('attributes', {})
            )
            person_descriptor = DescriptorGenerator.generate_person_descriptor(visual_evidence)
            
            # Find audio cue near first action
            audio_segment = self.find_audio_near_timestamp(
                detections[0]['timestamp'],
                evidence
            )
            if not audio_segment:
                continue
            
            # Skip intro/outro
            if self.is_in_intro_outro(audio_segment['start'], evidence):
                continue

            # Get audio text (no truncation per guideline 12)
            audio_text = self.get_audio_quote_for_question(audio_segment, max_length=60)

            # Generate question
            question = (
                f'How many times does the {person_descriptor} perform the action of '
                f'{action_type} after the audio cue "{audio_text}"?'
            )
            
            # Count actions after audio
            count = sum(
                1 for action in detections
                if action['timestamp'] > audio_segment['start']
            )
            
            answer = f'The {person_descriptor} performs {action_type} {count} times.'
            
            # Create cues
            audio_cue = Cue(
                cue_type=CueType.AUDIO_SPEECH,
                content=audio_text,
                timestamp=audio_segment['start']
            )
            
            visual_cue = Cue(
                cue_type=CueType.VISUAL_ACTION,
                content=f"{person_descriptor} {action_type}",
                timestamp=detections[0]['timestamp']
            )
            
            # Calculate timestamps
            start_ts = audio_segment['start']
            end_ts = detections[-1]['timestamp'] + 1.0
            
            # Validate no names
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
                question_types=[QuestionType.COUNTING],
                generation_tier=1,
                template_name=self.name,
                complexity_score=0.7,
                evidence_refs=[
                    f"audio:{audio_segment['start']}",
                    *[f"action:{a['timestamp']}" for a in detections]
                ]
            )
        
        return None