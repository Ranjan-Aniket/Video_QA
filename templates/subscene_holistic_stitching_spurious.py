"""
Subscene, Holistic Reasoning, Audio-Visual Stitching, and Spurious Correlations Templates

Subscene: Caption a segment based on conditioning in question
Holistic Reasoning: Understanding entire video, overall patterns
Audio-Visual Stitching: Editing choices, transitions, clip composition
Spurious Correlations: Unexpected/counter-intuitive events

Examples from taxonomy:

Subscene:
- "Describe what happens in the match when the score is 118-2 in favor of UGA?"
- "In third quarter with 9:49 on clock, how does bench react when player dunks?"

Holistic Reasoning:
- "What was the point of the video breaking up the game into clips?"
- "Describe how colored bars relate to the song being played"

Audio-Visual Stitching:
- "When Steven refers to AeroPress inventor, is inventor in same room or separate clip?"
- "How do spliced musical clips pace the interview?"

Spurious Correlations:
- "A group has meeting, man says 'don't wave red cape at charging bull.' Who are they referring to?"
  (Answer: Superman shown holographically)
"""

from typing import Optional, List, Dict
from templates.base import (
    QuestionTemplate, GeneratedQuestion, EvidenceDatabase,
    QuestionType, Cue, CueType
)
from templates.descriptor_generator import DescriptorGenerator, VisualEvidence


class SubsceneTemplate(QuestionTemplate):
    """Caption specific video segment"""
    
    def get_question_types(self) -> List[QuestionType]:
        return [QuestionType.SUBSCENE]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        """Need temporal markers (scores, time codes, etc.)"""
        return (
            len(evidence.event_timeline) >= 5 and
            len(evidence.transcript_segments) >= 3
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate subscene question
        
        Strategy:
        1. Find temporal marker (score, time, etc.)
        2. Ask what happens during that segment
        3. Provide detailed caption
        """
        # Look for events with specific markers
        for event in evidence.event_timeline:
            # Skip intro/outro
            if self.is_in_intro_outro(event['timestamp'], evidence):
                continue
            
            # Check if event has temporal marker
            marker = event.get('marker')  # Could be score, time code, etc.
            if not marker:
                continue
            
            # Find what happens around this marker
            events_at_marker = [
                e for e in evidence.event_timeline
                if abs(e['timestamp'] - event['timestamp']) < 5.0
                and e != event
            ]
            
            if len(events_at_marker) < 2:
                continue
            
            # Find audio around marker
            audio_segment = self.find_audio_near_timestamp(
                event['timestamp'],
                evidence,
                max_distance=3.0
            )
            if not audio_segment:
                continue
            
            audio_text = audio_segment['text'][:60]
            if len(audio_segment['text']) > 60:
                audio_text += "..."
            
            # Generate question
            question = (
                f'Describe what happens when {marker} is shown and someone says '
                f'"{audio_text}"?'
            )
            
            # Generate answer from events
            event_descriptions = []
            for e in events_at_marker[:3]:  # Max 3 events
                desc = self._get_event_descriptor(e, evidence)
                if desc:
                    event_descriptions.append(desc)
            
            answer = (
                f'When {marker} is shown, {", ".join(event_descriptions[:-1])}, '
                f'and {event_descriptions[-1]}.'
            )
            
            # Create cues
            audio_cue = Cue(
                cue_type=CueType.AUDIO_SPEECH,
                content=audio_text,
                timestamp=audio_segment['start']
            )
            
            visual_cues = [
                Cue(
                    cue_type=CueType.VISUAL_ACTION,
                    content=self._get_event_descriptor(e, evidence),
                    timestamp=e['timestamp']
                )
                for e in events_at_marker[:3]
                if self._get_event_descriptor(e, evidence)
            ]
            
            # Timestamps
            start_ts = min(event['timestamp'], audio_segment['start'])
            end_ts = max(e['timestamp'] for e in events_at_marker) + 2.0
            
            # Validate
            if not self.validate_no_names(question, evidence):
                continue
            if not self.validate_no_names(answer, evidence):
                continue
            
            if not visual_cues:
                continue
            
            return GeneratedQuestion(
                question_text=question,
                golden_answer=answer,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                audio_cues=[audio_cue],
                visual_cues=visual_cues,
                question_types=[QuestionType.SUBSCENE],
                generation_tier=1,
                template_name=self.name,
                complexity_score=0.8,
                evidence_refs=[
                    f"audio:{audio_segment['start']}",
                    f"event:{event['timestamp']}",
                    *[f"event:{e['timestamp']}" for e in events_at_marker[:3]]
                ]
            )
        
        return None
    
    def _get_event_descriptor(
        self,
        event: Dict,
        evidence: EvidenceDatabase
    ) -> Optional[str]:
        """Get event descriptor"""
        return event.get('description')


class HolisticReasoningTemplate(QuestionTemplate):
    """Understanding overall video patterns"""
    
    def get_question_types(self) -> List[QuestionType]:
        return [QuestionType.GENERAL_HOLISTIC_REASONING]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        """Need overall video structure"""
        return (
            len(evidence.scene_changes) >= 3 and
            len(evidence.event_timeline) >= 5
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate holistic reasoning question
        
        Strategy:
        1. Identify overall pattern (editing, music, theme)
        2. Ask about purpose or relationship
        3. Requires understanding whole video
        """
        # Check for editing patterns (scene changes)
        if len(evidence.scene_changes) >= 5:
            # Rapid scene changes might indicate montage
            avg_gap = sum(
                evidence.scene_changes[i+1] - evidence.scene_changes[i]
                for i in range(len(evidence.scene_changes)-1)
            ) / (len(evidence.scene_changes) - 1)
            
            if avg_gap < 3.0:  # Rapid cuts
                # Find audio that spans multiple scenes
                for audio_segment in evidence.transcript_segments:
                    # Count scene changes during this audio
                    changes_during = sum(
                        1 for sc in evidence.scene_changes
                        if audio_segment['start'] <= sc <= audio_segment.get('end', audio_segment['start'] + 5.0)
                    )
                    
                    if changes_during >= 2:
                        audio_text = audio_segment['text'][:60]
                        if len(audio_segment['text']) > 60:
                            audio_text += "..."
                        
                        # Generate question
                        question = (
                            f'What is the purpose of the rapid scene changes while '
                            f'someone says "{audio_text}"?'
                        )
                        
                        answer = (
                            'The rapid scene changes create a montage effect that '
                            'compresses time and shows multiple related events.'
                        )
                        
                        # Create cues
                        audio_cue = Cue(
                            cue_type=CueType.AUDIO_SPEECH,
                            content=audio_text,
                            timestamp=audio_segment['start']
                        )
                        
                        visual_cue = Cue(
                            cue_type=CueType.VISUAL_SCENE,
                            content="rapid scene transitions",
                            timestamp=evidence.scene_changes[0]
                        )
                        
                        # Timestamps: span multiple scenes
                        start_ts = audio_segment['start']
                        end_ts = audio_segment.get('end', audio_segment['start'] + 5.0)
                        
                        # Validate
                        if not self.validate_no_names(question, evidence):
                            continue
                        
                        return GeneratedQuestion(
                            question_text=question,
                            golden_answer=answer,
                            start_timestamp=start_ts,
                            end_timestamp=end_ts,
                            audio_cues=[audio_cue],
                            visual_cues=[visual_cue],
                            question_types=[QuestionType.GENERAL_HOLISTIC_REASONING],
                            generation_tier=1,
                            template_name=self.name,
                            complexity_score=0.8,
                            evidence_refs=[
                                f"audio:{audio_segment['start']}",
                                *[f"scene_change:{sc}" for sc in evidence.scene_changes[:5]]
                            ]
                        )
        
        return None


class AudioVisualStitchingTemplate(QuestionTemplate):
    """Editing choices and transitions"""
    
    def get_question_types(self) -> List[QuestionType]:
        return [QuestionType.AUDIO_VISUAL_STITCHING]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        """Need scene changes with continuous audio"""
        return (
            len(evidence.scene_changes) >= 2 and
            len(evidence.transcript_segments) >= 2
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate audio-visual stitching question
        
        Strategy:
        1. Find scene change during continuous audio
        2. Ask if clip is spliced or continuous
        """
        for audio_segment in evidence.transcript_segments:
            # Find scene changes during this audio
            changes_during = [
                sc for sc in evidence.scene_changes
                if audio_segment['start'] <= sc <= audio_segment.get('end', audio_segment['start'] + 5.0)
            ]
            
            if changes_during:
                audio_text = audio_segment['text'][:60]
                if len(audio_segment['text']) > 60:
                    audio_text += "..."
                
                # Generate question
                question = (
                    f'When someone says "{audio_text}", does the scene change '
                    f'or remain continuous?'
                )
                
                answer = (
                    'The scene changes while the audio continues, indicating '
                    'the audio and video are from separate clips spliced together.'
                )
                
                # Create cues
                audio_cue = Cue(
                    cue_type=CueType.AUDIO_SPEECH,
                    content=audio_text,
                    timestamp=audio_segment['start']
                )
                
                visual_cue = Cue(
                    cue_type=CueType.VISUAL_SCENE,
                    content="scene transition",
                    timestamp=changes_during[0]
                )
                
                # Timestamps
                start_ts = audio_segment['start']
                end_ts = changes_during[-1] + 2.0
                
                # Validate
                if not self.validate_no_names(question, evidence):
                    continue
                
                return GeneratedQuestion(
                    question_text=question,
                    golden_answer=answer,
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                    audio_cues=[audio_cue],
                    visual_cues=[visual_cue],
                    question_types=[QuestionType.AUDIO_VISUAL_STITCHING],
                    generation_tier=1,
                    template_name=self.name,
                    complexity_score=0.8,
                    evidence_refs=[
                        f"audio:{audio_segment['start']}",
                        *[f"scene_change:{sc}" for sc in changes_during]
                    ]
                )
        
        return None


class SpuriousCorrelationsTemplate(QuestionTemplate):
    """Unexpected/counter-intuitive events"""
    
    def get_question_types(self) -> List[QuestionType]:
        return [QuestionType.TACKLING_SPURIOUS_CORRELATIONS]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        """Need events where audio might refer to different visual"""
        return (
            len(evidence.transcript_segments) >= 3 and
            len(evidence.object_detections) >= 3
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate spurious correlation question
        
        Strategy:
        1. Find audio with pronouns (they, it, that)
        2. Find what's actually visible
        3. Ask what audio refers to (not what you'd expect)
        """
        # Look for pronouns in transcript
        pronoun_words = ['they', 'it', 'that', 'this', 'referring', 'he', 'she']
        
        for audio_segment in evidence.transcript_segments:
            # Skip intro/outro
            if self.is_in_intro_outro(audio_segment['start'], evidence):
                continue
            
            # Check for pronouns
            text_lower = audio_segment['text'].lower()
            has_pronoun = any(word in text_lower for word in pronoun_words)
            
            if not has_pronoun:
                continue
            
            # Find objects visible at this time
            objects_at_time = [
                obj for obj in evidence.object_detections
                if abs(obj['timestamp'] - audio_segment['start']) < 2.0
            ]
            
            if not objects_at_time:
                continue
            
            obj = objects_at_time[0]
            
            # Generate object descriptor
            visual_evidence = VisualEvidence(
                timestamp=obj['timestamp'],
                bbox=obj['bbox'],
                object_class=obj['object_class'],
                color=obj.get('color')
            )
            obj_desc = DescriptorGenerator.generate_object_descriptor(visual_evidence)
            
            audio_text = audio_segment['text'][:80]
            if len(audio_segment['text']) > 80:
                audio_text += "..."
            
            # Generate question
            question = (
                f'When someone says "{audio_text}", what are they referring to '
                f'based on what is shown on screen?'
            )
            
            answer = (
                f'They are referring to the {obj_desc} that is shown on screen '
                f'at this moment.'
            )
            
            # Create cues
            audio_cue = Cue(
                cue_type=CueType.AUDIO_SPEECH,
                content=audio_text,
                timestamp=audio_segment['start']
            )
            
            visual_cue = Cue(
                cue_type=CueType.VISUAL_OBJECT,
                content=obj_desc,
                timestamp=obj['timestamp']
            )
            
            # Timestamps
            start_ts = min(audio_segment['start'], obj['timestamp'])
            end_ts = max(audio_segment['start'] + 2.0, obj['timestamp'] + 1.0)
            
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
                question_types=[QuestionType.TACKLING_SPURIOUS_CORRELATIONS],
                generation_tier=1,
                template_name=self.name,
                complexity_score=0.9,
                evidence_refs=[
                    f"audio:{audio_segment['start']}",
                    f"object:{obj['timestamp']}"
                ]
            )
        
        return None