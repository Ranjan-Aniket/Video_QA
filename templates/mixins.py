"""
Mixins for Question Type Components

Each mixin provides reusable logic for ONE question type.
Templates combine mixins to create multi-type questions.

CRITICAL: All mixins MUST enforce:
1. Both audio AND visual cues required
2. NO names - only descriptors
3. Precise timestamps
4. Evidence-first (no hardcoding)
"""

from typing import Optional, List, Dict, Tuple
from templates.base import EvidenceDatabase, Cue, CueType
from templates.descriptor_generator import DescriptorGenerator, VisualEvidence


class TemporalMixin:
    """
    Adds temporal understanding (before/after/when)
    
    GUIDELINE: "Use before/after/when with extra caution"
    """
    
    def find_temporal_relationship(
        self,
        evidence: EvidenceDatabase,
        min_gap: float = 5.0
    ) -> Optional[Tuple[Dict, Dict, str]]:
        """
        Find two events with temporal relationship
        
        Returns: (event_a, event_b, relation)
        relation = "after" or "before"
        """
        for i, event_a in enumerate(evidence.event_timeline[:-1]):
            # Skip intro/outro (GUIDELINE)
            if self.is_in_intro_outro(event_a['timestamp'], evidence):
                continue
            
            event_b = evidence.event_timeline[i + 1]
            if self.is_in_intro_outro(event_b['timestamp'], evidence):
                continue
            
            # Check minimum temporal gap
            gap = event_b['timestamp'] - event_a['timestamp']
            if gap < min_gap:
                continue
            
            return event_a, event_b, "after"
        
        return None
    
    def format_temporal_question(
        self,
        base_question: str,
        temporal_relation: str,
        audio_cue: str
    ) -> str:
        """
        Format question with temporal marker
        
        GUIDELINE: "after X was said" (not "when X is said" if after visual)
        """
        return f'{base_question} {temporal_relation} someone says "{audio_cue}"?'


class SequentialMixin:
    """
    Adds sequential/order logic
    
    GUIDELINE: "Sequential always go hand in hand with Temporal Understanding"
    """
    
    def find_event_sequence(
        self,
        evidence: EvidenceDatabase,
        count: int = 3,
        min_gap: float = 5.0
    ) -> Optional[List[Dict]]:
        """
        Find sequence of events with temporal spacing
        
        Returns: List of events in order
        """
        valid_events = [
            e for e in evidence.event_timeline
            if not self.is_in_intro_outro(e['timestamp'], evidence)
        ]
        
        if len(valid_events) < count:
            return None
        
        # Select events with minimum gap
        selected = [valid_events[0]]
        for event in valid_events[1:]:
            if len(selected) >= count:
                break
            if event['timestamp'] - selected[-1]['timestamp'] >= min_gap:
                selected.append(event)
        
        return selected if len(selected) >= count else None
    
    def format_sequential_question(
        self,
        events: List[Dict],
        evidence: EvidenceDatabase
    ) -> str:
        """
        Format as "What is the order of events?" question
        """
        descriptors = []
        for event in events:
            desc = self._get_event_descriptor(event, evidence)
            if desc:
                descriptors.append(desc)
        
        options = '\n'.join([
            f'({chr(65+i)}) {desc}'
            for i, desc in enumerate(descriptors)
        ])
        
        return f'What is the order of these events?\n{options}'


class CountingMixin:
    """
    Adds counting logic
    
    GUIDELINE: "Challenging counting questions" - not obvious counts
    """
    
    def find_countable_elements(
        self,
        evidence: EvidenceDatabase,
        after_timestamp: Optional[float] = None,
        min_count: int = 3
    ) -> Optional[Tuple[str, int, List[Dict]]]:
        """
        Find elements that appear multiple times
        
        Returns: (item_description, count, occurrences)
        """
        # Group objects by class
        object_groups = {}
        for obj in evidence.object_detections:
            if after_timestamp and obj['timestamp'] <= after_timestamp:
                continue
            
            if self.is_in_intro_outro(obj['timestamp'], evidence):
                continue
            
            obj_class = obj['object_class']
            if obj_class not in object_groups:
                object_groups[obj_class] = []
            object_groups[obj_class].append(obj)
        
        # Find groups with enough occurrences
        for obj_class, occurrences in object_groups.items():
            if len(occurrences) >= min_count:
                # Generate descriptor for first occurrence
                first = occurrences[0]
                visual_evidence = VisualEvidence(
                    timestamp=first['timestamp'],
                    bbox=first['bbox'],
                    object_class=obj_class,
                    color=first.get('color'),
                    size=first.get('size')
                )
                descriptor = DescriptorGenerator.generate_object_descriptor(visual_evidence)
                
                return descriptor, len(occurrences), occurrences
        
        return None
    
    def format_counting_question(
        self,
        item_descriptor: str,
        context: str
    ) -> str:
        """Format counting question"""
        return f'How many {item_descriptor}s {context}?'


class NeedleMixin:
    """
    Adds needle-in-haystack logic (find specific detail at precise moment)
    
    GUIDELINE: Look for OCR text, specific visual details
    """
    
    def find_needle_moment(
        self,
        evidence: EvidenceDatabase
    ) -> Optional[Tuple[str, str, float, Cue]]:
        """
        Find specific detail (OCR, object) at a moment with audio
        
        Returns: (detail_type, detail_content, timestamp, visual_cue)
        """
        # Try OCR first (most common needle type)
        for ocr in evidence.ocr_detections:
            if self.is_in_intro_outro(ocr['timestamp'], evidence):
                continue
            
            # Find audio near this OCR
            audio = self.find_audio_near_timestamp(
                ocr['timestamp'],
                evidence,
                max_distance=2.0
            )
            if audio:
                cue = Cue(
                    cue_type=CueType.VISUAL_TEXT,
                    content=ocr['text'],
                    timestamp=ocr['timestamp'],
                    confidence=ocr.get('confidence', 1.0)
                )
                return "text", ocr['text'], ocr['timestamp'], cue
        
        return None
    
    def format_needle_question(
        self,
        detail_type: str,
        location: str,
        audio_cue: str
    ) -> str:
        """Format needle question"""
        if detail_type == "text":
            return f'What text appears on the {location} when someone says "{audio_cue}"?'
        else:
            return f'What {detail_type} appears when someone says "{audio_cue}"?'


class ReferentialGroundingMixin:
    """
    Connects audio to visual at specific time
    
    GUIDELINE: "Connects the audio and visual elements at a specific time"
    """
    
    def find_audio_visual_connection(
        self,
        evidence: EvidenceDatabase
    ) -> Optional[Tuple[Dict, List[Dict], str]]:
        """
        Find audio with multiple people/objects visible
        
        Returns: (audio_segment, visual_elements, connection_type)
        """
        for audio in evidence.transcript_segments:
            if self.is_in_intro_outro(audio['start'], evidence):
                continue
            
            # Find people visible during audio
            people = [
                p for p in evidence.person_detections
                if abs(p['timestamp'] - audio['start']) < 2.0
            ]
            
            if len(people) >= 2:
                return audio, people[:2], "people_present"
            
            # Or find what creates a sound
            for sound in evidence.sound_effects:
                if abs(sound['timestamp'] - audio['start']) < 1.0:
                    # Find object creating sound
                    obj = self.find_visual_near_timestamp(
                        sound['timestamp'],
                        evidence,
                        visual_type="object"
                    )
                    if obj:
                        return audio, [obj], "sound_source"
        
        return None
    
    def format_referential_question(
        self,
        audio_cue: str,
        connection_type: str
    ) -> str:
        """Format referential grounding question"""
        if connection_type == "people_present":
            return f'Who is visible on screen when someone says "{audio_cue}"?'
        elif connection_type == "sound_source":
            return f'What creates the sound when someone says "{audio_cue}"?'
        else:
            return f'What is shown when someone says "{audio_cue}"?'


class ContextMixin:
    """
    Background/foreground elements during events
    
    GUIDELINE: "asks about the setting or background"
    """
    
    def find_context_elements(
        self,
        evidence: EvidenceDatabase,
        timestamp: float
    ) -> Optional[List[Dict]]:
        """
        Find background/foreground objects at timestamp
        
        Returns: List of objects (2-3)
        """
        objects = [
            obj for obj in evidence.object_detections
            if abs(obj['timestamp'] - timestamp) < 2.0
        ]
        
        if len(objects) >= 2:
            return objects[:3]
        
        return None
    
    def format_context_question(
        self,
        audio_cue: str,
        element_type: str = "background"
    ) -> str:
        """Format context question"""
        return f'What visual elements are present in the {element_type} when someone says "{audio_cue}"?'


class ComparativeMixin:
    """
    Compare differences/similarities between states
    
    GUIDELINE: "direct asks about differences/similarities"
    """
    
    def find_comparable_states(
        self,
        evidence: EvidenceDatabase
    ) -> Optional[Tuple[Dict, Dict, str, str]]:
        """
        Find same person/object at different times with difference
        
        Returns: (state_a, state_b, difference_type, difference_description)
        """
        # Group person detections by person_id
        person_groups = {}
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
            
            detections = sorted(detections, key=lambda d: d['timestamp'])
            first = detections[0]
            last = detections[-1]
            
            # Skip if too close
            if last['timestamp'] - first['timestamp'] < 10.0:
                continue
            
            # Check for visual difference
            first_attrs = first.get('attributes', {})
            last_attrs = last.get('attributes', {})
            
            # Compare clothing color
            first_color = first_attrs.get('clothing_color')
            last_color = last_attrs.get('clothing_color')
            
            if first_color and last_color and first_color != last_color:
                return first, last, "clothing_color", f"{first_color} to {last_color}"
        
        return None
    
    def format_comparative_question(
        self,
        audio_cue: str,
        comparison_aspect: str
    ) -> str:
        """Format comparative question"""
        return f'What is the difference in the person\'s {comparison_aspect} before and after someone says "{audio_cue}"?'


class InferenceMixin:
    """
    Understanding intentions, purposes, causal relationships
    
    GUIDELINE: "explain the PURPOSE/MEANING of something" - answers "why"
    """
    
    def find_causal_relationship(
        self,
        evidence: EvidenceDatabase
    ) -> Optional[Tuple[Dict, Dict, str]]:
        """
        Find event B caused by event A
        
        Returns: (event_a, event_b, causal_link)
        """
        for i, event_a in enumerate(evidence.event_timeline[:-1]):
            event_b = evidence.event_timeline[i + 1]
            
            # Skip intro/outro
            if self.is_in_intro_outro(event_a['timestamp'], evidence):
                continue
            if self.is_in_intro_outro(event_b['timestamp'], evidence):
                continue
            
            # Check if close enough to be causal
            gap = event_b['timestamp'] - event_a['timestamp']
            if 1.0 <= gap <= 10.0:  # Causal window
                return event_a, event_b, "caused_by"
        
        return None
    
    def format_inference_question(
        self,
        event_descriptor: str,
        audio_cue: str
    ) -> str:
        """Format inference question (asks "why")"""
        return f'Why does {event_descriptor} happen after someone says "{audio_cue}"?'


class ObjectInteractionMixin:
    """
    How objects change/transform through interactions
    
    GUIDELINE: "track transformations across context"
    """
    
    def find_object_transformation(
        self,
        evidence: EvidenceDatabase
    ) -> Optional[Tuple[str, Dict, Dict, Dict]]:
        """
        Find object before/after action
        
        Returns: (object_class, state_before, state_after, action)
        """
        # Group objects by class
        object_groups = {}
        for obj in evidence.object_detections:
            obj_class = obj['object_class']
            if obj_class not in object_groups:
                object_groups[obj_class] = []
            object_groups[obj_class].append(obj)
        
        # Find objects at different times with action between
        for obj_class, detections in object_groups.items():
            if len(detections) < 2:
                continue
            
            detections = sorted(detections, key=lambda d: d['timestamp'])
            first = detections[0]
            last = detections[-1]
            
            # Find action between them
            actions = [
                a for a in evidence.action_detections
                if first['timestamp'] < a['timestamp'] < last['timestamp']
            ]
            
            if actions:
                return obj_class, first, last, actions[0]
        
        return None
    
    def format_object_interaction_question(
        self,
        object_class: str,
        audio_cue: str
    ) -> str:
        """Format object interaction question"""
        return f'How does the {object_class} change after someone says "{audio_cue}" and performs an action?'


class SubsceneMixin:
    """
    Caption a segment based on conditioning in question
    
    GUIDELINE: "caption a relevant and important part of a long video"
    """
    
    def find_conditioned_segment(
        self,
        evidence: EvidenceDatabase
    ) -> Optional[Tuple[str, float, List[Dict]]]:
        """
        Find segment with temporal marker (score, time code, etc.)
        
        Returns: (marker, timestamp, events_in_segment)
        """
        for event in evidence.event_timeline:
            marker = event.get('marker')
            if not marker:
                continue
            
            if self.is_in_intro_outro(event['timestamp'], evidence):
                continue
            
            # Find events around this marker (±5 seconds)
            events_at_marker = [
                e for e in evidence.event_timeline
                if abs(e['timestamp'] - event['timestamp']) < 5.0
                and e != event
            ]
            
            if len(events_at_marker) >= 2:
                return marker, event['timestamp'], events_at_marker
        
        return None
    
    def format_subscene_question(
        self,
        marker: str,
        audio_cue: str
    ) -> str:
        """Format subscene question"""
        return f'Describe what happens when {marker} is shown and someone says "{audio_cue}"?'


class HolisticReasoningMixin:
    """
    Understanding entire video, overall patterns
    
    GUIDELINE: "knowledge for the entire video is required"
    """
    
    def find_overall_pattern(
        self,
        evidence: EvidenceDatabase
    ) -> Optional[Tuple[str, List[float]]]:
        """
        Identify patterns across entire video
        
        Returns: (pattern_type, evidence_timestamps)
        """
        # Check for rapid scene changes (montage)
        if len(evidence.scene_changes) >= 5:
            gaps = [
                evidence.scene_changes[i+1] - evidence.scene_changes[i]
                for i in range(len(evidence.scene_changes) - 1)
            ]
            avg_gap = sum(gaps) / len(gaps)
            
            if avg_gap < 3.0:  # Rapid cuts
                return "rapid_cuts", evidence.scene_changes[:5]
        
        return None
    
    def format_holistic_question(
        self,
        pattern_type: str,
        audio_cue: str
    ) -> str:
        """Format holistic reasoning question"""
        if pattern_type == "rapid_cuts":
            return f'What is the purpose of the rapid scene changes while someone says "{audio_cue}"?'
        else:
            return f'What is the purpose of this pattern in the video?'


class AudioVisualStitchingMixin:
    """
    Editing choices, transitions, clip composition
    
    GUIDELINE: "reasoning about editing choices, transitions"
    """
    
    def find_editing_transition(
        self,
        evidence: EvidenceDatabase
    ) -> Optional[Tuple[Dict, List[float]]]:
        """
        Find scene change during continuous audio
        
        Returns: (audio_segment, scene_change_timestamps)
        """
        for audio in evidence.transcript_segments:
            audio_end = audio.get('end', audio['start'] + 5.0)
            
            # Find scene changes during this audio
            changes = [
                sc for sc in evidence.scene_changes
                if audio['start'] <= sc <= audio_end
            ]
            
            if changes:
                return audio, changes
        
        return None
    
    def format_stitching_question(
        self,
        audio_cue: str
    ) -> str:
        """Format audio-visual stitching question"""
        return f'When someone says "{audio_cue}", does the scene change or remain continuous?'


class SpuriousCorrelationMixin:
    """
    Unexpected/counter-intuitive events
    
    GUIDELINE: "unexpected, unnatural, or un-intuitive events"
    """
    
    def find_spurious_reference(
        self,
        evidence: EvidenceDatabase
    ) -> Optional[Tuple[Dict, Dict]]:
        """
        Find audio with pronouns referring to unexpected visual
        
        Returns: (audio_with_pronoun, unexpected_visual)
        """
        pronouns = ['they', 'it', 'that', 'this', 'referring', 'he', 'she']
        
        for audio in evidence.transcript_segments:
            text_lower = audio['text'].lower()
            
            # Check for pronouns
            if not any(word in text_lower for word in pronouns):
                continue
            
            if self.is_in_intro_outro(audio['start'], evidence):
                continue
            
            # Find objects visible at this time
            objects = [
                obj for obj in evidence.object_detections
                if abs(obj['timestamp'] - audio['start']) < 2.0
            ]
            
            if objects:
                return audio, objects[0]
        
        return None
    
    def format_spurious_question(
        self,
        audio_cue: str
    ) -> str:
        """Format spurious correlation question"""
        return f'When someone says "{audio_cue}", what are they referring to based on what is shown on screen?'


# Helper methods that all mixins share
class MixinHelpers:
    """Shared helper methods for all mixins"""
    
    def is_in_intro_outro(
        self,
        timestamp: float,
        evidence: EvidenceDatabase
    ) -> bool:
        """
        GUIDELINE: "Do not use intro and outro of video for reference points"
        """
        if evidence.intro_end and timestamp < evidence.intro_end:
            return True
        if evidence.outro_start and timestamp > evidence.outro_start:
            return True
        return False
    
    def find_audio_near_timestamp(
        self,
        timestamp: float,
        evidence: EvidenceDatabase,
        max_distance: float = 3.0
    ) -> Optional[Dict]:
        """Find audio segment near timestamp"""
        closest = None
        min_dist = float('inf')
        
        for segment in evidence.transcript_segments:
            dist = abs(segment['start'] - timestamp)
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                closest = segment
        
        return closest
    
    def find_visual_near_timestamp(
        self,
        timestamp: float,
        evidence: EvidenceDatabase,
        visual_type: str = "any",
        max_distance: float = 2.0
    ) -> Optional[Dict]:
        """Find visual detection near timestamp"""
        candidates = []
        
        if visual_type in ["any", "person"]:
            candidates.extend(evidence.person_detections)
        if visual_type in ["any", "object"]:
            candidates.extend(evidence.object_detections)
        if visual_type in ["any", "scene"]:
            candidates.extend(evidence.scene_detections)
        
        closest = None
        min_dist = float('inf')
        
        for item in candidates:
            dist = abs(item['timestamp'] - timestamp)
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                closest = item
        
        return closest
    
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