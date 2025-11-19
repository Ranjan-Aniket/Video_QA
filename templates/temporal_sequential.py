"""
Temporal Understanding & Sequential Templates

These templates generate questions about:
- What happens before/after events
- Order of events
- Temporal relationships

Examples from taxonomy:
Temporal:
- "What happens before the woman in blonde hair says to Captain America, 
   'This is going to work Steve'?"
- "What happens after Casey introduces Ollie?"

Sequential:
- "What is the order of events in this video? A) X B) Y C) Z D) W"
- "Which happens first in the video: the interviewer asking about defense 
   or the Raptors scoring 66 points?"

CRITICAL GUIDELINES:
- Must have BOTH audio and visual cues
- Use before/after/when with CAUTION (per guidelines)
- Must use descriptors, not names
- Timestamps must cover both cues and actions
"""

from typing import Optional, List, Dict, Tuple
from templates.base import (
    QuestionTemplate, GeneratedQuestion, EvidenceDatabase,
    QuestionType, Cue, CueType
)
from templates.descriptor_generator import DescriptorGenerator, VisualEvidence


class TemporalUnderstandingTemplate(QuestionTemplate):
    """Generate before/after questions"""
    
    def get_question_types(self) -> List[QuestionType]:
        return [QuestionType.TEMPORAL_UNDERSTANDING]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        """Need multiple events in timeline"""
        return (
            len(evidence.event_timeline) >= 3 and
            len(evidence.transcript_segments) >= 3
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate temporal question
        
        Strategy:
        1. Find event with audio cue
        2. Identify what happens before/after
        3. Ensure both cues present
        """
        # Try "what happens after" pattern
        for i, event in enumerate(evidence.event_timeline[:-1]):  # Not last event
            # Skip intro/outro
            if self.is_in_intro_outro(event['timestamp'], evidence):
                continue
            
            # Find audio cue near this event
            audio_segment = self.find_audio_near_timestamp(
                event['timestamp'],
                evidence,
                max_distance=2.0
            )
            if not audio_segment:
                continue
            
            # Get next event
            next_event = evidence.event_timeline[i + 1]
            
            # Generate descriptors for events
            event_descriptor = self._get_event_descriptor(event, evidence)
            next_event_descriptor = self._get_event_descriptor(next_event, evidence)
            
            if not event_descriptor or not next_event_descriptor:
                continue
            
            # Get audio text
            audio_text = audio_segment['text'][:60]
            if len(audio_segment['text']) > 60:
                audio_text += "..."
            
            # Generate question
            question = (
                f'What happens after someone says "{audio_text}" and we see '
                f'{event_descriptor}?'
            )
            
            answer = f'{next_event_descriptor.capitalize()} happens next.'
            
            # Create cues
            audio_cue = Cue(
                cue_type=CueType.AUDIO_SPEECH,
                content=audio_text,
                timestamp=audio_segment['start']
            )
            
            visual_cue1 = Cue(
                cue_type=CueType.VISUAL_ACTION,
                content=event_descriptor,
                timestamp=event['timestamp']
            )
            
            visual_cue2 = Cue(
                cue_type=CueType.VISUAL_ACTION,
                content=next_event_descriptor,
                timestamp=next_event['timestamp']
            )
            
            # Timestamps: from audio to next event completion
            start_ts = min(audio_segment['start'], event['timestamp'])
            end_ts = next_event['timestamp'] + 2.0  # Include next event
            
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
                visual_cues=[visual_cue1, visual_cue2],
                question_types=[QuestionType.TEMPORAL_UNDERSTANDING],
                generation_tier=1,
                template_name=self.name,
                complexity_score=0.6,
                evidence_refs=[
                    f"audio:{audio_segment['start']}",
                    f"event:{event['timestamp']}",
                    f"event:{next_event['timestamp']}"
                ]
            )
        
        return None
    
    def _get_event_descriptor(
        self,
        event: Dict,
        evidence: EvidenceDatabase
    ) -> Optional[str]:
        """Get natural language descriptor for event"""
        event_type = event.get('event_type')
        
        if event_type == 'person_action':
            # Get person descriptor
            person_id = event.get('person_id')
            if not person_id:
                return None
            
            person_detections = [
                p for p in evidence.person_detections
                if p['person_id'] == person_id
                and abs(p['timestamp'] - event['timestamp']) < 1.0
            ]
            if not person_detections:
                return None
            
            person_det = person_detections[0]
            visual_evidence = VisualEvidence(
                timestamp=person_det['timestamp'],
                bbox=person_det['bbox'],
                **person_det.get('attributes', {})
            )
            person_desc = DescriptorGenerator.generate_person_descriptor(visual_evidence)
            
            action = event.get('description', 'acting')
            return f"the {person_desc} {action}"
        
        elif event_type == 'scene_change':
            scene = event.get('description', 'scene change')
            return f"the scene changes to {scene}"
        
        elif event_type == 'object_appearance':
            obj_class = event.get('object_class', 'object')
            return f"a {obj_class} appears"
        
        else:
            return event.get('description')


class SequentialTemplate(QuestionTemplate):
    """Generate order-of-events questions"""
    
    def get_question_types(self) -> List[QuestionType]:
        return [QuestionType.SEQUENTIAL]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        """Need at least 3 distinct events"""
        return len(evidence.event_timeline) >= 3
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate sequential question
        
        Strategy:
        1. Select 3-4 events
        2. Create multiple choice order question
        3. Ensure both audio and visual cues
        """
        # Filter events not in intro/outro
        valid_events = [
            e for e in evidence.event_timeline
            if not self.is_in_intro_outro(e['timestamp'], evidence)
        ]
        
        if len(valid_events) < 3:
            return None
        
        # Select 3 events with some temporal distance
        selected_events = self._select_spaced_events(valid_events, count=3)
        if len(selected_events) < 3:
            return None
        
        # Get descriptors for each event
        descriptors = []
        for event in selected_events:
            desc = self._get_event_descriptor(event, evidence)
            if not desc:
                return None
            descriptors.append(desc)
        
        # Find audio cue near first event
        audio_segment = self.find_audio_near_timestamp(
            selected_events[0]['timestamp'],
            evidence
        )
        if not audio_segment:
            return None
        
        audio_text = audio_segment['text'][:50]
        if len(audio_segment['text']) > 50:
            audio_text += "..."
        
        # Generate question with multiple choice
        question = (
            f'After someone says "{audio_text}", what is the order of these events?\n'
            f'(A) {descriptors[0]}\n'
            f'(B) {descriptors[1]}\n'
            f'(C) {descriptors[2]}'
        )
        
        answer = f'The correct order is (A)(B)(C).'
        
        # Create cues
        audio_cue = Cue(
            cue_type=CueType.AUDIO_SPEECH,
            content=audio_text,
            timestamp=audio_segment['start']
        )
        
        visual_cues = [
            Cue(
                cue_type=CueType.VISUAL_ACTION,
                content=desc,
                timestamp=event['timestamp']
            )
            for desc, event in zip(descriptors, selected_events)
        ]
        
        # Timestamps: from audio to last event
        start_ts = audio_segment['start']
        end_ts = selected_events[-1]['timestamp'] + 2.0
        
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
            question_types=[QuestionType.SEQUENTIAL],
            generation_tier=1,
            template_name=self.name,
            complexity_score=0.7,
            evidence_refs=[
                f"audio:{audio_segment['start']}",
                *[f"event:{e['timestamp']}" for e in selected_events]
            ]
        )
    
    def _select_spaced_events(
        self,
        events: List[Dict],
        count: int,
        min_gap: float = 5.0
    ) -> List[Dict]:
        """Select events with minimum temporal gap"""
        if len(events) < count:
            return events
        
        selected = [events[0]]
        
        for event in events[1:]:
            if len(selected) >= count:
                break
            
            # Check gap with last selected
            if event['timestamp'] - selected[-1]['timestamp'] >= min_gap:
                selected.append(event)
        
        return selected
    
    def _get_event_descriptor(
        self,
        event: Dict,
        evidence: EvidenceDatabase
    ) -> Optional[str]:
        """Get natural language descriptor for event"""
        # Reuse from TemporalUnderstandingTemplate
        event_type = event.get('event_type')
        
        if event_type == 'person_action':
            person_id = event.get('person_id')
            if not person_id:
                return None
            
            person_detections = [
                p for p in evidence.person_detections
                if p['person_id'] == person_id
                and abs(p['timestamp'] - event['timestamp']) < 1.0
            ]
            if not person_detections:
                return None
            
            person_det = person_detections[0]
            visual_evidence = VisualEvidence(
                timestamp=person_det['timestamp'],
                bbox=person_det['bbox'],
                **person_det.get('attributes', {})
            )
            person_desc = DescriptorGenerator.generate_person_descriptor(visual_evidence)
            
            action = event.get('description', 'acting')
            return f"the {person_desc} {action}"
        
        elif event_type == 'scene_change':
            scene = event.get('description', 'scene change')
            return f"the scene changes to {scene}"
        
        else:
            return event.get('description')