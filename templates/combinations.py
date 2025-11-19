"""
Multi-Type Question Template Combinations

Combines mixins to generate questions with multiple task types.
Based on real examples from guidelines and sample data.

CRITICAL: Each combination MUST:
1. Have BOTH audio AND visual cues
2. NO names - only descriptors
3. Precise timestamps (not a second more/less)
4. Evidence-driven (no hardcoding)
5. Single-cue answerable → REJECT
"""

from typing import Optional, List
from templates.base import (
    QuestionTemplate, GeneratedQuestion, EvidenceDatabase,
    QuestionType, Cue, CueType
)
from templates.mixins import (
    TemporalMixin, SequentialMixin, CountingMixin, NeedleMixin,
    ReferentialGroundingMixin, ContextMixin, ComparativeMixin,
    InferenceMixin, ObjectInteractionMixin, SubsceneMixin,
    HolisticReasoningMixin, AudioVisualStitchingMixin,
    SpuriousCorrelationMixin, MixinHelpers
)
from templates.descriptor_generator import DescriptorGenerator, VisualEvidence


# ============================================================================
# TEMPORAL COMBINATIONS
# ============================================================================

class TemporalSequentialCountingTemplate(
    TemporalMixin,
    SequentialMixin,
    CountingMixin,
    MixinHelpers,
    QuestionTemplate
):
    """
    Example: "How many 'teaspoons of sugar' does person add after placing 
    the moka pot onto the stove?"
    
    Combines:
    - Temporal: "after placing X"
    - Sequential: order of events
    - Counting: "how many teaspoons"
    """
    
    def get_question_types(self) -> List[QuestionType]:
        return [
            QuestionType.TEMPORAL_UNDERSTANDING,
            QuestionType.SEQUENTIAL,
            QuestionType.COUNTING
        ]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        """Need temporal events + countable elements"""
        return (
            len(evidence.event_timeline) >= 3 and
            len(evidence.action_detections) >= 3 and
            len(evidence.transcript_segments) >= 2
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate: "How many X happen after Y action and someone says Z?"
        """
        # Step 1: Find temporal anchor (an action event)
        temporal_result = self.find_temporal_relationship(evidence)
        if not temporal_result:
            return None
        
        event_a, event_b, relation = temporal_result
        
        # Step 2: Find countable elements AFTER event_a
        counting_result = self.find_countable_elements(
            evidence,
            after_timestamp=event_a['timestamp'],
            min_count=2
        )
        if not counting_result:
            return None
        
        item_descriptor, count, occurrences = counting_result
        
        # Step 3: Find audio cue near event_a
        audio_segment = self.find_audio_near_timestamp(
            event_a['timestamp'],
            evidence,
            max_distance=2.0
        )
        if not audio_segment:
            return None
        
        # Step 4: Get event_a descriptor
        event_a_desc = self._get_event_descriptor(event_a, evidence)
        if not event_a_desc:
            return None
        
        # Step 5: Generate question
        audio_text = audio_segment['text'][:50]
        if len(audio_segment['text']) > 50:
            audio_text += "..."
        
        question = (
            f'How many {item_descriptor}s appear after {event_a_desc} '
            f'and someone says "{audio_text}"?'
        )
        
        answer = (
            f'After {event_a_desc} and the audio cue, '
            f'{count} {item_descriptor}s appear.'
        )
        
        # Step 6: Create cues
        audio_cue = Cue(
            cue_type=CueType.AUDIO_SPEECH,
            content=audio_text,
            timestamp=audio_segment['start']
        )
        
        visual_cue1 = Cue(
            cue_type=CueType.VISUAL_ACTION,
            content=event_a_desc,
            timestamp=event_a['timestamp']
        )
        
        visual_cue2 = Cue(
            cue_type=CueType.VISUAL_OBJECT,
            content=f"{count} {item_descriptor}s",
            timestamp=occurrences[0]['timestamp']
        )
        
        # Step 7: Timestamps - from event_a to last occurrence
        start_ts = min(event_a['timestamp'], audio_segment['start'])
        end_ts = occurrences[-1]['timestamp'] + 1.0
        
        # Step 8: Validate no names
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
            visual_cues=[visual_cue1, visual_cue2],
            question_types=self.get_question_types(),
            generation_tier=1,
            template_name=self.name,
            complexity_score=0.8,
            evidence_refs=[
                f"audio:{audio_segment['start']}",
                f"event:{event_a['timestamp']}",
                *[f"object:{o['timestamp']}" for o in occurrences]
            ]
        )
    
    def validate_no_names(self, text: str, evidence: EvidenceDatabase) -> bool:
        """Validate no names present"""
        text_lower = text.lower()
        all_names = (
            evidence.character_names +
            evidence.team_names +
            evidence.media_names +
            evidence.brand_names
        )
        for name in all_names:
            if name.lower() in text_lower:
                return False
        return True


class TemporalSequentialNeedleTemplate(
    TemporalMixin,
    SequentialMixin,
    NeedleMixin,
    MixinHelpers,
    QuestionTemplate
):
    """
    Example: "What prompts the announcer to get excited and say X?"
    
    Combines:
    - Temporal: before/after relationship
    - Sequential: order of events leading to outcome
    - Needle: specific detail at precise moment
    """
    
    def get_question_types(self) -> List[QuestionType]:
        return [
            QuestionType.TEMPORAL_UNDERSTANDING,
            QuestionType.SEQUENTIAL,
            QuestionType.NEEDLE
        ]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        return (
            len(evidence.transcript_segments) >= 3 and
            len(evidence.ocr_detections) >= 1 and
            len(evidence.event_timeline) >= 2
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate: "What specific thing happens right before audio cue X?"
        """
        # Find needle moment (OCR or specific visual)
        needle_result = self.find_needle_moment(evidence)
        if not needle_result:
            return None
        
        detail_type, detail_content, timestamp, visual_cue = needle_result
        
        # Find event that happens just before this
        preceding_events = [
            e for e in evidence.event_timeline
            if e['timestamp'] < timestamp
            and timestamp - e['timestamp'] < 5.0
        ]
        
        if not preceding_events:
            return None
        
        preceding_event = preceding_events[-1]  # Most recent before needle
        
        # Get audio at needle moment
        audio_segment = self.find_audio_near_timestamp(timestamp, evidence)
        if not audio_segment:
            return None
        
        # Generate question
        audio_text = audio_segment['text'][:50]
        if len(audio_segment['text']) > 50:
            audio_text += "..."
        
        event_desc = self._get_event_descriptor(preceding_event, evidence)
        if not event_desc:
            return None
        
        question = (
            f'What prompts someone to say "{audio_text}"?'
        )
        
        answer = (
            f'{event_desc.capitalize()} prompts the audio cue.'
        )
        
        # Create cues
        audio_cue = Cue(
            cue_type=CueType.AUDIO_SPEECH,
            content=audio_text,
            timestamp=audio_segment['start']
        )
        
        visual_cue_event = Cue(
            cue_type=CueType.VISUAL_ACTION,
            content=event_desc,
            timestamp=preceding_event['timestamp']
        )
        
        # Timestamps
        start_ts = preceding_event['timestamp']
        end_ts = audio_segment['start'] + 2.0
        
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
            visual_cues=[visual_cue_event, visual_cue],
            question_types=self.get_question_types(),
            generation_tier=1,
            template_name=self.name,
            complexity_score=0.7,
            evidence_refs=[
                f"audio:{audio_segment['start']}",
                f"event:{preceding_event['timestamp']}",
                f"needle:{timestamp}"
            ]
        )
    
    def validate_no_names(self, text: str, evidence: EvidenceDatabase) -> bool:
        text_lower = text.lower()
        all_names = (
            evidence.character_names +
            evidence.team_names +
            evidence.media_names +
            evidence.brand_names
        )
        for name in all_names:
            if name.lower() in text_lower:
                return False
        return True


class TemporalSequentialInferenceTemplate(
    TemporalMixin,
    SequentialMixin,
    InferenceMixin,
    MixinHelpers,
    QuestionTemplate
):
    """
    Example: "After door opens, what is the beeping sound and why?"
    
    Combines:
    - Temporal: after X happens
    - Sequential: order of events
    - Inference: explain purpose/meaning (WHY)
    """
    
    def get_question_types(self) -> List[QuestionType]:
        return [
            QuestionType.TEMPORAL_UNDERSTANDING,
            QuestionType.SEQUENTIAL,
            QuestionType.INFERENCE
        ]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        return (
            len(evidence.event_timeline) >= 3 and
            len(evidence.sound_effects) >= 1 and
            len(evidence.transcript_segments) >= 2
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate: "Why does Y happen after X and audio cue Z?"
        """
        # Find causal relationship
        causal_result = self.find_causal_relationship(evidence)
        if not causal_result:
            return None
        
        event_a, event_b, causal_link = causal_result
        
        # Get descriptors
        event_a_desc = self._get_event_descriptor(event_a, evidence)
        event_b_desc = self._get_event_descriptor(event_b, evidence)
        
        if not event_a_desc or not event_b_desc:
            return None
        
        # Find audio near event_a
        audio_segment = self.find_audio_near_timestamp(
            event_a['timestamp'],
            evidence
        )
        if not audio_segment:
            return None
        
        audio_text = audio_segment['text'][:50]
        if len(audio_segment['text']) > 50:
            audio_text += "..."
        
        # Generate question (asks WHY)
        question = (
            f'Why does {event_b_desc} happen after {event_a_desc} '
            f'and someone says "{audio_text}"?'
        )
        
        answer = (
            f'{event_b_desc.capitalize()} happens as a result of {event_a_desc}.'
        )
        
        # Create cues
        audio_cue = Cue(
            cue_type=CueType.AUDIO_SPEECH,
            content=audio_text,
            timestamp=audio_segment['start']
        )
        
        visual_cue1 = Cue(
            cue_type=CueType.VISUAL_ACTION,
            content=event_a_desc,
            timestamp=event_a['timestamp']
        )
        
        visual_cue2 = Cue(
            cue_type=CueType.VISUAL_ACTION,
            content=event_b_desc,
            timestamp=event_b['timestamp']
        )
        
        # Timestamps
        start_ts = min(audio_segment['start'], event_a['timestamp'])
        end_ts = event_b['timestamp'] + 2.0
        
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
            visual_cues=[visual_cue1, visual_cue2],
            question_types=self.get_question_types(),
            generation_tier=1,
            template_name=self.name,
            complexity_score=0.9,
            evidence_refs=[
                f"audio:{audio_segment['start']}",
                f"event:{event_a['timestamp']}",
                f"event:{event_b['timestamp']}"
            ]
        )
    
    def validate_no_names(self, text: str, evidence: EvidenceDatabase) -> bool:
        text_lower = text.lower()
        all_names = (
            evidence.character_names +
            evidence.team_names +
            evidence.media_names +
            evidence.brand_names
        )
        for name in all_names:
            if name.lower() in text_lower:
                return False
        return True


# ============================================================================
# SUBSCENE COMBINATIONS
# ============================================================================

class SubsceneNeedleTemplate(
    SubsceneMixin,
    NeedleMixin,
    MixinHelpers,
    QuestionTemplate
):
    """
    Example: "When score is 118-2, what does speaker say?"
    
    Combines:
    - Subscene: caption segment based on condition (score, time, etc.)
    - Needle: find specific detail in that segment
    """
    
    def get_question_types(self) -> List[QuestionType]:
        return [
            QuestionType.SUBSCENE,
            QuestionType.NEEDLE
        ]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        return (
            len(evidence.event_timeline) >= 3 and
            any(e.get('marker') for e in evidence.event_timeline) and
            len(evidence.transcript_segments) >= 2
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate: "What specific thing is said/shown when marker X is visible?"
        """
        # Find conditioned segment
        subscene_result = self.find_conditioned_segment(evidence)
        if not subscene_result:
            return None
        
        marker, timestamp, events = subscene_result
        
        # Find audio at this moment
        audio_segment = self.find_audio_near_timestamp(timestamp, evidence)
        if not audio_segment:
            return None
        
        audio_text = audio_segment['text'][:60]
        if len(audio_segment['text']) > 60:
            audio_text += "..."
        
        # Generate question
        question = (
            f'What does someone say when {marker} is shown on screen?'
        )
        
        answer = (
            f'When {marker} is shown, someone says "{audio_text}".'
        )
        
        # Create cues
        audio_cue = Cue(
            cue_type=CueType.AUDIO_SPEECH,
            content=audio_text,
            timestamp=audio_segment['start']
        )
        
        visual_cue = Cue(
            cue_type=CueType.VISUAL_TEXT,
            content=marker,
            timestamp=timestamp
        )
        
        # Timestamps
        start_ts = timestamp - 1.0
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
            visual_cues=[visual_cue],
            question_types=self.get_question_types(),
            generation_tier=1,
            template_name=self.name,
            complexity_score=0.7,
            evidence_refs=[
                f"audio:{audio_segment['start']}",
                f"marker:{timestamp}"
            ]
        )
    
    def validate_no_names(self, text: str, evidence: EvidenceDatabase) -> bool:
        text_lower = text.lower()
        all_names = (
            evidence.character_names +
            evidence.team_names +
            evidence.media_names +
            evidence.brand_names
        )
        for name in all_names:
            if name.lower() in text_lower:
                return False
        return True


# ============================================================================
# INFERENCE COMBINATIONS
# ============================================================================

class HolisticInferenceTemplate(
    HolisticReasoningMixin,
    InferenceMixin,
    MixinHelpers,
    QuestionTemplate
):
    """
    Example: "Based on audio and visual, why is character feeling 'guilt'?"
    
    Combines:
    - Holistic: requires understanding entire video
    - Inference: explain purpose/meaning
    """
    
    def get_question_types(self) -> List[QuestionType]:
        return [
            QuestionType.GENERAL_HOLISTIC_REASONING,
            QuestionType.INFERENCE
        ]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        return (
            len(evidence.scene_changes) >= 3 and
            len(evidence.event_timeline) >= 5 and
            len(evidence.transcript_segments) >= 5
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate: "Why does overall pattern X exist in the video?"
        """
        # Find overall pattern
        pattern_result = self.find_overall_pattern(evidence)
        if not pattern_result:
            return None
        
        pattern_type, evidence_timestamps = pattern_result
        
        # Find audio that spans multiple scenes
        audio_segment = None
        for audio in evidence.transcript_segments:
            audio_end = audio.get('end', audio['start'] + 5.0)
            
            # Count scene changes during this audio
            changes_during = sum(
                1 for sc in evidence.scene_changes
                if audio['start'] <= sc <= audio_end
            )
            
            if changes_during >= 2:
                audio_segment = audio
                break
        
        if not audio_segment:
            return None
        
        audio_text = audio_segment['text'][:60]
        if len(audio_segment['text']) > 60:
            audio_text += "..."
        
        # Generate question (asks WHY - inference)
        if pattern_type == "rapid_cuts":
            question = (
                f'What is the purpose of the rapid scene changes while '
                f'someone says "{audio_text}"?'
            )
            
            answer = (
                'The rapid scene changes create a montage effect that '
                'compresses time and shows multiple related events.'
            )
        else:
            question = (
                f'What is the purpose of this editing pattern while '
                f'someone says "{audio_text}"?'
            )
            answer = 'The editing pattern serves a narrative purpose in the video.'
        
        # Create cues
        audio_cue = Cue(
            cue_type=CueType.AUDIO_SPEECH,
            content=audio_text,
            timestamp=audio_segment['start']
        )
        
        visual_cue = Cue(
            cue_type=CueType.VISUAL_SCENE,
            content="rapid scene transitions",
            timestamp=evidence_timestamps[0]
        )
        
        # Timestamps - span multiple scenes
        start_ts = audio_segment['start']
        end_ts = audio_segment.get('end', audio_segment['start'] + 5.0)
        
        # Validate
        if not self.validate_no_names(question, evidence):
            return None
        
        return GeneratedQuestion(
            question_text=question,
            golden_answer=answer,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            audio_cues=[audio_cue],
            visual_cues=[visual_cue],
            question_types=self.get_question_types(),
            generation_tier=1,
            template_name=self.name,
            complexity_score=0.9,
            evidence_refs=[
                f"audio:{audio_segment['start']}",
                *[f"scene_change:{sc}" for sc in evidence_timestamps[:5]]
            ]
        )
    
    def validate_no_names(self, text: str, evidence: EvidenceDatabase) -> bool:
        text_lower = text.lower()
        all_names = (
            evidence.character_names +
            evidence.team_names +
            evidence.media_names +
            evidence.brand_names
        )
        for name in all_names:
            if name.lower() in text_lower:
                return False
        return True


class ContextInferenceCountingTemplate(
    ContextMixin,
    InferenceMixin,
    CountingMixin,
    MixinHelpers,
    QuestionTemplate
):
    """
    Example: "Between timestamps X and Y, is anyone having panic attack? 
    What audio/video cues tell us? Give timestamps of each cue."
    
    Combines:
    - Context: background/foreground elements
    - Inference: understand meaning/purpose
    - Counting: count cues
    """
    
    def get_question_types(self) -> List[QuestionType]:
        return [
            QuestionType.CONTEXT,
            QuestionType.INFERENCE,
            QuestionType.COUNTING
        ]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        return (
            len(evidence.transcript_segments) >= 3 and
            len(evidence.person_detections) >= 3 and
            len(evidence.sound_effects) >= 1
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate: "How many cues indicate X state, and what are they?"
        """
        # Find audio segment with emotional content
        audio_segment = None
        for audio in evidence.transcript_segments:
            if self.is_in_intro_outro(audio['start'], evidence):
                continue
            
            # Look for emotional language
            text_lower = audio['text'].lower()
            emotional_words = ['nervous', 'scared', 'excited', 'angry', 'sad', 'happy']
            if any(word in text_lower for word in emotional_words):
                audio_segment = audio
                break
        
        if not audio_segment:
            return None
        
        # Find context elements (people, objects) at this time
        context_elements = self.find_context_elements(
            evidence,
            audio_segment['start']
        )
        if not context_elements:
            return None
        
        # Count cues
        cue_count = len(context_elements) + 1  # +1 for audio
        
        audio_text = audio_segment['text'][:50]
        if len(audio_segment['text']) > 50:
            audio_text += "..."
        
        # Generate question
        question = (
            f'Between the time someone says "{audio_text}" and the next few seconds, '
            f'how many audio or visual cues are present?'
        )
        
        answer = f'There are {cue_count} cues present during this segment.'
        
        # Create cues
        audio_cue = Cue(
            cue_type=CueType.AUDIO_SPEECH,
            content=audio_text,
            timestamp=audio_segment['start']
        )
        
        visual_cues = [
            Cue(
                cue_type=CueType.VISUAL_OBJECT,
                content=elem['object_class'],
                timestamp=elem['timestamp']
            )
            for elem in context_elements[:3]
        ]
        
        # Timestamps
        start_ts = audio_segment['start']
        end_ts = start_ts + 5.0
        
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
            question_types=self.get_question_types(),
            generation_tier=1,
            template_name=self.name,
            complexity_score=0.8,
            evidence_refs=[
                f"audio:{audio_segment['start']}",
                *[f"object:{e['timestamp']}" for e in context_elements[:3]]
            ]
        )
    
    def validate_no_names(self, text: str, evidence: EvidenceDatabase) -> bool:
        text_lower = text.lower()
        all_names = (
            evidence.character_names +
            evidence.team_names +
            evidence.media_names +
            evidence.brand_names
        )
        for name in all_names:
            if name.lower() in text_lower:
                return False
        return True


# ============================================================================
# OBJECT & SEQUENTIAL COMBINATIONS
# ============================================================================

class SequentialObjectInteractionTemplate(
    SequentialMixin,
    ObjectInteractionMixin,
    MixinHelpers,
    QuestionTemplate
):
    """
    Example: "After person says X, how does clay change?"
    
    Combines:
    - Sequential: order of events
    - Object Interaction: transformation of objects
    """
    
    def get_question_types(self) -> List[QuestionType]:
        return [
            QuestionType.SEQUENTIAL,
            QuestionType.OBJECT_INTERACTION_REASONING
        ]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        return (
            len(evidence.object_detections) >= 3 and
            len(evidence.action_detections) >= 2 and
            len(evidence.transcript_segments) >= 2
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate: "After action X and audio Y, how does object change?"
        """
        # Find object transformation
        transform_result = self.find_object_transformation(evidence)
        if not transform_result:
            return None
        
        obj_class, state_before, state_after, action = transform_result
        
        # Find audio near action
        audio_segment = self.find_audio_near_timestamp(
            action['timestamp'],
            evidence
        )
        if not audio_segment:
            return None
        
        audio_text = audio_segment['text'][:50]
        if len(audio_segment['text']) > 50:
            audio_text += "..."
        
        # Generate question
        question = (
            f'How does the {obj_class} change after someone says "{audio_text}" '
            f'and performs an action?'
        )
        
        answer = (
            f'The {obj_class} transforms during this segment.'
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
            timestamp=state_before['timestamp']
        )
        
        # Timestamps
        start_ts = state_before['timestamp']
        end_ts = state_after['timestamp'] + 1.0
        
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
            question_types=self.get_question_types(),
            generation_tier=1,
            template_name=self.name,
            complexity_score=0.7,
            evidence_refs=[
                f"audio:{audio_segment['start']}",
                f"object:{state_before['timestamp']}",
                f"action:{action['timestamp']}"
            ]
        )
    
    def validate_no_names(self, text: str, evidence: EvidenceDatabase) -> bool:
        text_lower = text.lower()
        all_names = (
            evidence.character_names +
            evidence.team_names +
            evidence.media_names +
            evidence.brand_names
        )
        for name in all_names:
            if name.lower() in text_lower:
                return False
        return True


# ============================================================================
# SPURIOUS CORRELATION COMBINATIONS
# ============================================================================

class SpuriousContextTemplate(
    SpuriousCorrelationMixin,
    ContextMixin,
    MixinHelpers,
    QuestionTemplate
):
    """
    Example: "Group having meeting, man says 'don't wave red cape at bull.'
    Who are they referring to?" (Answer: Superman shown holographically)
    
    Combines:
    - Spurious Correlation: unexpected/unintuitive reference
    - Context: what's actually shown in background/scene
    """
    
    def get_question_types(self) -> List[QuestionType]:
        return [
            QuestionType.TACKLING_SPURIOUS_CORRELATIONS,
            QuestionType.CONTEXT
        ]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        return (
            len(evidence.transcript_segments) >= 3 and
            len(evidence.object_detections) >= 2
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate: "When audio says X (with pronoun), what are they referring to?"
        """
        # Find spurious reference
        spurious_result = self.find_spurious_reference(evidence)
        if not spurious_result:
            return None
        
        audio, unexpected_visual = spurious_result
        
        # Generate object descriptor
        visual_evidence = VisualEvidence(
            timestamp=unexpected_visual['timestamp'],
            bbox=unexpected_visual['bbox'],
            object_class=unexpected_visual['object_class'],
            color=unexpected_visual.get('color')
        )
        obj_desc = DescriptorGenerator.generate_object_descriptor(visual_evidence)
        
        audio_text = audio['text'][:80]
        if len(audio['text']) > 80:
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
            timestamp=audio['start']
        )
        
        visual_cue = Cue(
            cue_type=CueType.VISUAL_OBJECT,
            content=obj_desc,
            timestamp=unexpected_visual['timestamp']
        )
        
        # Timestamps
        start_ts = min(audio['start'], unexpected_visual['timestamp'])
        end_ts = max(audio['start'] + 2.0, unexpected_visual['timestamp'] + 1.0)
        
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
            question_types=self.get_question_types(),
            generation_tier=1,
            template_name=self.name,
            complexity_score=0.9,
            evidence_refs=[
                f"audio:{audio['start']}",
                f"object:{unexpected_visual['timestamp']}"
            ]
        )
    
    def validate_no_names(self, text: str, evidence: EvidenceDatabase) -> bool:
        text_lower = text.lower()
        all_names = (
            evidence.character_names +
            evidence.team_names +
            evidence.media_names +
            evidence.brand_names
        )
        for name in all_names:
            if name.lower() in text_lower:
                return False
        return True


class ReferentialSpuriousTemplate(
    ReferentialGroundingMixin,
    SpuriousCorrelationMixin,
    MixinHelpers,
    QuestionTemplate
):
    """
    Example: "What happens when character presses red button after saying X?"
    (Answer: Background light indicates bomb, not sunset)
    
    Combines:
    - Referential Grounding: connect audio to specific visual
    - Spurious Correlation: visual means something unexpected
    """
    
    def get_question_types(self) -> List[QuestionType]:
        return [
            QuestionType.REFERENTIAL_GROUNDING,
            QuestionType.TACKLING_SPURIOUS_CORRELATIONS
        ]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        return (
            len(evidence.transcript_segments) >= 2 and
            len(evidence.action_detections) >= 1 and
            len(evidence.object_detections) >= 1
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate: "What really happens at this moment (not what it seems)?"
        """
        # Find action with nearby audio
        action = None
        for act in evidence.action_detections:
            if self.is_in_intro_outro(act['timestamp'], evidence):
                continue
            action = act
            break
        
        if not action:
            return None
        
        # Find audio near action
        audio_segment = self.find_audio_near_timestamp(
            action['timestamp'],
            evidence
        )
        if not audio_segment:
            return None
        
        # Find visual near action
        visual = self.find_visual_near_timestamp(
            action['timestamp'],
            evidence,
            visual_type="object"
        )
        if not visual:
            return None
        
        audio_text = audio_segment['text'][:60]
        if len(audio_segment['text']) > 60:
            audio_text += "..."
        
        action_desc = action['action']
        
        # Generate question
        question = (
            f'What happens when someone says "{audio_text}" and performs the action of {action_desc}?'
        )
        
        answer = (
            f'An unexpected event occurs on screen during this moment.'
        )
        
        # Create cues
        audio_cue = Cue(
            cue_type=CueType.AUDIO_SPEECH,
            content=audio_text,
            timestamp=audio_segment['start']
        )
        
        visual_cue = Cue(
            cue_type=CueType.VISUAL_ACTION,
            content=action_desc,
            timestamp=action['timestamp']
        )
        
        # Timestamps
        start_ts = min(audio_segment['start'], action['timestamp'])
        end_ts = action['timestamp'] + 2.0
        
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
            question_types=self.get_question_types(),
            generation_tier=1,
            template_name=self.name,
            complexity_score=0.9,
            evidence_refs=[
                f"audio:{audio_segment['start']}",
                f"action:{action['timestamp']}"
            ]
        )
    
    def validate_no_names(self, text: str, evidence: EvidenceDatabase) -> bool:
        text_lower = text.lower()
        all_names = (
            evidence.character_names +
            evidence.team_names +
            evidence.media_names +
            evidence.brand_names
        )
        for name in all_names:
            if name.lower() in text_lower:
                return False
        return True


# ============================================================================
# AUDIO-VISUAL STITCHING COMBINATIONS
# ============================================================================

class AudioVisualStitchingReferentialTemplate(
    AudioVisualStitchingMixin,
    ReferentialGroundingMixin,
    MixinHelpers,
    QuestionTemplate
):
    """
    Example: "When person refers to inventor, is inventor in same room or spliced clip?"
    
    Combines:
    - Audio-Visual Stitching: editing/splicing
    - Referential Grounding: who/what is present
    """
    
    def get_question_types(self) -> List[QuestionType]:
        return [
            QuestionType.AUDIO_VISUAL_STITCHING,
            QuestionType.REFERENTIAL_GROUNDING
        ]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        return (
            len(evidence.scene_changes) >= 2 and
            len(evidence.transcript_segments) >= 2
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate: "Does scene change during continuous audio?"
        """
        # Find editing transition
        transition_result = self.find_editing_transition(evidence)
        if not transition_result:
            return None
        
        audio, scene_changes = transition_result
        
        audio_text = audio['text'][:60]
        if len(audio['text']) > 60:
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
            timestamp=audio['start']
        )
        
        visual_cue = Cue(
            cue_type=CueType.VISUAL_SCENE,
            content="scene transition",
            timestamp=scene_changes[0]
        )
        
        # Timestamps
        start_ts = audio['start']
        end_ts = scene_changes[-1] + 2.0
        
        # Validate
        if not self.validate_no_names(question, evidence):
            return None
        
        return GeneratedQuestion(
            question_text=question,
            golden_answer=answer,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            audio_cues=[audio_cue],
            visual_cues=[visual_cue],
            question_types=self.get_question_types(),
            generation_tier=1,
            template_name=self.name,
            complexity_score=0.8,
            evidence_refs=[
                f"audio:{audio['start']}",
                *[f"scene_change:{sc}" for sc in scene_changes]
            ]
        )
    
    def validate_no_names(self, text: str, evidence: EvidenceDatabase) -> bool:
        text_lower = text.lower()
        all_names = (
            evidence.character_names +
            evidence.team_names +
            evidence.media_names +
            evidence.brand_names
        )
        for name in all_names:
            if name.lower() in text_lower:
                return False
        return True


# ============================================================================
# THREE-TYPE COMBINATIONS
# ============================================================================

class SequentialSubsceneHolisticTemplate(
    SequentialMixin,
    SubsceneMixin,
    HolisticReasoningMixin,
    MixinHelpers,
    QuestionTemplate
):
    """
    Complex three-type combination for advanced questions
    
    Example: Describe sequence of events in specific segment that 
    reveals overall pattern
    """
    
    def get_question_types(self) -> List[QuestionType]:
        return [
            QuestionType.SEQUENTIAL,
            QuestionType.SUBSCENE,
            QuestionType.GENERAL_HOLISTIC_REASONING
        ]
    
    def can_apply(self, evidence: EvidenceDatabase) -> bool:
        return (
            len(evidence.event_timeline) >= 5 and
            len(evidence.scene_changes) >= 3 and
            any(e.get('marker') for e in evidence.event_timeline)
        )
    
    def generate(self, evidence: EvidenceDatabase) -> Optional[GeneratedQuestion]:
        """
        Generate: "What sequence happens at marker X and why?"
        """
        # Find conditioned segment
        subscene_result = self.find_conditioned_segment(evidence)
        if not subscene_result:
            return None
        
        marker, timestamp, events = subscene_result
        
        # Check if part of overall pattern
        pattern_result = self.find_overall_pattern(evidence)
        if not pattern_result:
            return None
        
        # Find audio
        audio_segment = self.find_audio_near_timestamp(timestamp, evidence)
        if not audio_segment:
            return None
        
        audio_text = audio_segment['text'][:50]
        if len(audio_segment['text']) > 50:
            audio_text += "..."
        
        # Get event descriptors
        event_descs = []
        for event in events[:3]:
            desc = self._get_event_descriptor(event, evidence)
            if desc:
                event_descs.append(desc)
        
        if len(event_descs) < 2:
            return None
        
        # Generate question
        question = (
            f'What is the sequence of events when {marker} is shown '
            f'and someone says "{audio_text}", and why does this pattern occur?'
        )
        
        answer = (
            f'When {marker} is shown, {", ".join(event_descs[:-1])}, '
            f'and {event_descs[-1]}. This pattern serves a narrative purpose.'
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
                content=desc,
                timestamp=event['timestamp']
            )
            for desc, event in zip(event_descs, events[:3])
        ]
        
        # Timestamps
        start_ts = min(timestamp, audio_segment['start'])
        end_ts = max(e['timestamp'] for e in events[:3]) + 2.0
        
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
            question_types=self.get_question_types(),
            generation_tier=1,
            template_name=self.name,
            complexity_score=1.0,
            evidence_refs=[
                f"audio:{audio_segment['start']}",
                f"marker:{timestamp}",
                *[f"event:{e['timestamp']}" for e in events[:3]]
            ]
        )
    
    def validate_no_names(self, text: str, evidence: EvidenceDatabase) -> bool:
        text_lower = text.lower()
        all_names = (
            evidence.character_names +
            evidence.team_names +
            evidence.media_names +
            evidence.brand_names
        )
        for name in all_names:
            if name.lower() in text_lower:
                return False
        return True