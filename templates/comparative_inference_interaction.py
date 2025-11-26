"""
Comparative, Inference, and Object Interaction Reasoning Templates

Comparative: Compare differences/similarities between states or elements
Inference: Understand intentions, purposes, causal relationships
Object Interaction: How objects change/transform through interactions

Examples from taxonomy:

Comparative:
- "What is the difference between the man's shirt before and after he says X?"
- "What are the two biggest distinctions in person's appearance when performing vs interviewing?"

Inference:
- "Why do teams switch after announcer says 'She just added emphasis'?"
- "Why did wicketkeeper scream when score is 145-2?"
- "Based on audio and visual, why is Dom Cobb feeling 'guilt'?"

Object Interaction:
- "How does the clay change after the man talks about 'pour over vs coffee dripper'?"
- "What is the visual effect created when man places second prism in spectrum path?"
"""

from typing import Optional, List, Dict
from templates.base import (
    QuestionTemplate, GeneratedQuestion, EvidenceDatabase,
    QuestionType, Cue, CueType
)
from templates.descriptor_generator import DescriptorGenerator, VisualEvidence


class ComparativeTemplate(QuestionTemplate):
    """Compare differences between states"""
    
    def get_question_types(self) -> List[QuestionType]:
        return [QuestionType.COMPARATIVE]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        """Need same object/person at different times"""
        return (
            len(evidence.person_detections) >= 4 and
            len(evidence.transcript_segments) >= 2
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate comparative question
        
        Strategy:
        1. Find same person at two different times
        2. Identify visual difference
        3. Use audio cue to mark the transition
        """
        # Group person detections by person_id
        person_groups: Dict[str, List[Dict]] = {}
        for person in evidence.person_detections:
            person_id = person.get('person_id')
            if not person_id:
                continue
            if person_id not in person_groups:
                person_groups[person_id] = []
            person_groups[person_id].append(person)
        
        # Find person with detections at different times
        for person_id, detections in person_groups.items():
            if len(detections) < 2:
                continue
            
            # Sort by time
            detections = sorted(detections, key=lambda d: d['timestamp'])
            
            # Get first and last detection
            first = detections[0]
            last = detections[-1]
            
            # Skip if too close in time
            if last['timestamp'] - first['timestamp'] < 10.0:
                continue
            
            # Skip intro/outro
            if self.is_in_intro_outro(first['timestamp'], evidence):
                continue
            if self.is_in_intro_outro(last['timestamp'], evidence):
                continue
            
            # Check for visual difference
            first_attrs = first.get('attributes', {})
            last_attrs = last.get('attributes', {})
            
            # Compare clothing color (common change)
            first_color = first_attrs.get('clothing_color')
            last_color = last_attrs.get('clothing_color')
            
            if first_color and last_color and first_color != last_color:
                # Found difference! Find audio cue between them
                audio_segment = None
                for segment in evidence.transcript_segments:
                    if first['timestamp'] < segment['start'] < last['timestamp']:
                        audio_segment = segment
                        break
                
                if not audio_segment:
                    continue
                
                # Generate descriptors
                visual_evidence_first = VisualEvidence(
                    timestamp=first['timestamp'],
                    bbox=first['bbox'],
                    **first_attrs
                )
                person_desc_first = DescriptorGenerator.generate_person_descriptor(visual_evidence_first)
                
                visual_evidence_last = VisualEvidence(
                    timestamp=last['timestamp'],
                    bbox=last['bbox'],
                    **last_attrs
                )
                person_desc_last = DescriptorGenerator.generate_person_descriptor(visual_evidence_last)
                
                # Get audio text (no truncation per guideline 12)
                audio_text = self.get_audio_quote_for_question(audio_segment, max_length=70)

                # Generate question
                question = (
                    f'What is the difference in the person\'s appearance before and '
                    f'after the audio cue "{audio_text}"?'
                )
                
                answer = (
                    f'Before the audio cue, the person is wearing {first_color}, '
                    f'but after the audio cue, the person is wearing {last_color}.'
                )
                
                # Create cues
                audio_cue = Cue(
                    cue_type=CueType.AUDIO_SPEECH,
                    content=audio_text,
                    timestamp=audio_segment['start']
                )
                
                visual_cue1 = Cue(
                    cue_type=CueType.VISUAL_PERSON,
                    content=person_desc_first,
                    timestamp=first['timestamp']
                )
                
                visual_cue2 = Cue(
                    cue_type=CueType.VISUAL_PERSON,
                    content=person_desc_last,
                    timestamp=last['timestamp']
                )
                
                # Timestamps: from first appearance to last
                start_ts = first['timestamp']
                end_ts = last['timestamp'] + 1.0
                
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
                    question_types=[QuestionType.COMPARATIVE],
                    generation_tier=1,
                    template_name=self.name,
                    complexity_score=0.7,
                    evidence_refs=[
                        f"audio:{audio_segment['start']}",
                        f"person:{first['timestamp']}",
                        f"person:{last['timestamp']}"
                    ]
                )
        
        return None


class InferenceTemplate(QuestionTemplate):
    """Understand intentions, purposes, causal relationships"""
    
    def get_question_types(self) -> List[QuestionType]:
        return [QuestionType.INFERENCE]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        """Need events with potential causal relationships"""
        return (
            len(evidence.event_timeline) >= 3 and
            len(evidence.transcript_segments) >= 3
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate inference question
        
        Strategy:
        1. Find event B that follows event A
        2. Ask "why" B happened
        3. Answer requires understanding A caused B
        """
        # Look for causal event pairs
        for i, event_a in enumerate(evidence.event_timeline[:-1]):
            event_b = evidence.event_timeline[i + 1]
            
            # Skip intro/outro
            if self.is_in_intro_outro(event_a['timestamp'], evidence):
                continue
            if self.is_in_intro_outro(event_b['timestamp'], evidence):
                continue
            
            # Check if events are close enough to be causal
            if event_b['timestamp'] - event_a['timestamp'] > 10.0:
                continue
            
            # Get event descriptors
            desc_a = self._get_event_descriptor(event_a, evidence)
            desc_b = self._get_event_descriptor(event_b, evidence)
            
            if not desc_a or not desc_b:
                continue
            
            # Find audio cue near event_a
            audio_segment = self.find_audio_near_timestamp(
                event_a['timestamp'],
                evidence,
                max_distance=2.0
            )
            if not audio_segment:
                continue
            
            audio_text = self.get_audio_quote_for_question(audio_segment, max_length=70)

            # Generate question
            question = (
                f'Why does {desc_b} happen after the audio cue "{audio_text}" '
                f'and we see {desc_a}?'
            )
            
            # Answer requires inference
            answer = (
                f'{desc_b.capitalize()} happens as a result of {desc_a}.'
            )
            
            # Create cues
            audio_cue = Cue(
                cue_type=CueType.AUDIO_SPEECH,
                content=audio_text,
                timestamp=audio_segment['start']
            )
            
            visual_cue1 = Cue(
                cue_type=CueType.VISUAL_ACTION,
                content=desc_a,
                timestamp=event_a['timestamp']
            )
            
            visual_cue2 = Cue(
                cue_type=CueType.VISUAL_ACTION,
                content=desc_b,
                timestamp=event_b['timestamp']
            )
            
            # Timestamps
            start_ts = min(audio_segment['start'], event_a['timestamp'])
            end_ts = event_b['timestamp'] + 2.0
            
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
                question_types=[QuestionType.INFERENCE],
                generation_tier=1,
                template_name=self.name,
                complexity_score=0.8,
                evidence_refs=[
                    f"audio:{audio_segment['start']}",
                    f"event:{event_a['timestamp']}",
                    f"event:{event_b['timestamp']}"
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
        
        else:
            return event.get('description')


class ObjectInteractionTemplate(QuestionTemplate):
    """How objects change through interactions"""
    
    def get_question_types(self) -> List[QuestionType]:
        return [QuestionType.OBJECT_INTERACTION_REASONING]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        """Need object detections and actions"""
        return (
            len(evidence.object_detections) >= 3 and
            len(evidence.action_detections) >= 2
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate object interaction question
        
        Strategy:
        1. Find object before and after an action
        2. Identify transformation
        3. Ask how object changed
        """
        # Group objects by class
        object_groups: Dict[str, List[Dict]] = {}
        for obj in evidence.object_detections:
            obj_class = obj['object_class']
            if obj_class not in object_groups:
                object_groups[obj_class] = []
            object_groups[obj_class].append(obj)
        
        # Find objects that appear at different times
        for obj_class, detections in object_groups.items():
            if len(detections) < 2:
                continue
            
            # Sort by time
            detections = sorted(detections, key=lambda d: d['timestamp'])
            
            first = detections[0]
            last = detections[-1]
            
            # Find action between them
            actions_between = [
                action for action in evidence.action_detections
                if first['timestamp'] < action['timestamp'] < last['timestamp']
            ]
            
            if not actions_between:
                continue
            
            action = actions_between[0]
            
            # Skip intro/outro
            if self.is_in_intro_outro(first['timestamp'], evidence):
                continue
            
            # Find audio near action
            audio_segment = self.find_audio_near_timestamp(
                action['timestamp'],
                evidence
            )
            if not audio_segment:
                continue
            
            audio_text = self.get_audio_quote_for_question(audio_segment, max_length=70)

            # Generate object descriptors
            visual_evidence_first = VisualEvidence(
                timestamp=first['timestamp'],
                bbox=first['bbox'],
                object_class=obj_class,
                color=first.get('color')
            )
            obj_desc_first = DescriptorGenerator.generate_object_descriptor(visual_evidence_first)

            # Generate question
            question = (
                f'How does the {obj_class} change after the audio cue "{audio_text}" '
                f'and an action is performed?'
            )
            
            answer = (
                f'The {obj_class} transforms or changes state during this segment.'
            )
            
            # Create cues
            audio_cue = Cue(
                cue_type=CueType.AUDIO_SPEECH,
                content=audio_text,
                timestamp=audio_segment['start']
            )
            
            visual_cue = Cue(
                cue_type=CueType.VISUAL_OBJECT,
                content=f"{obj_class} transformation",
                timestamp=first['timestamp']
            )
            
            # Timestamps
            start_ts = first['timestamp']
            end_ts = last['timestamp'] + 1.0
            
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
                question_types=[QuestionType.OBJECT_INTERACTION_REASONING],
                generation_tier=1,
                template_name=self.name,
                complexity_score=0.7,
                evidence_refs=[
                    f"audio:{audio_segment['start']}",
                    f"object:{first['timestamp']}",
                    f"action:{action['timestamp']}"
                ]
            )
        
        return None