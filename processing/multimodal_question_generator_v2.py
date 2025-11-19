"""
Multimodal Question Generator V2 - Complete Implementation

Generates 30 questions from Phase 4 evidence:
- 3 GPT-4V questions (from agreed frames)
- 7 Claude questions (from disagreed frames)
- 20 Template questions (from all frames)

Architecture:
1. AIDescriptionParser: Extract person attributes from AI descriptions
2. Phase4EvidenceConverter: Convert frame-based → timeline-based evidence
3. PremiumFrameAnalyzer: Analyze premium frames with consensus
4. AIQuestionGenerator: Generate GPT-4V + Claude questions
5. TemplateIntegrator: Use template registry for 20 questions
6. UnifiedValidator: Validate all 30 questions

Based on:
- Question Types_ Skills.pdf
- Guidelines_ Prompt Creation.docx
- Sample work sheet - MSPO 557.xlsx
"""

import logging
import json
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
import os
import openai
from anthropic import Anthropic

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MultimodalQuestion:
    """Question requiring BOTH audio AND visual understanding"""
    question_id: str
    question: str
    golden_answer: str

    # Timestamps (HH:MM:SS format)
    start_timestamp: str
    end_timestamp: str

    # Required cues
    audio_cue: str  # Exact quote from transcript
    visual_cue: str  # Visual elements from frame

    # Metadata
    task_types: List[str]
    generation_tier: str  # "template", "ai_gpt4v", "ai_claude"
    complexity: str  # "low", "medium", "high"

    # Validation
    requires_both_modalities: bool = True
    validated: bool = False
    validation_notes: str = ""


@dataclass
class QuestionGenerationResult:
    """Complete question generation results"""
    video_id: str
    total_questions: int = 0
    validated_questions: int = 0

    questions: List[MultimodalQuestion] = field(default_factory=list)

    metadata: Dict = field(default_factory=dict)
    generation_cost: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "total_questions": self.total_questions,
            "validated_questions": self.validated_questions,
            "metadata": {
                "video_id": self.video_id,
                **self.metadata
            },
            "questions": [asdict(q) for q in self.questions],
            "generation_cost": self.generation_cost
        }


# ============================================================================
# COMPONENT 1: AI DESCRIPTION PARSER
# ============================================================================

class AIDescriptionParser:
    """Extract person attributes from GPT-4V/Claude descriptions"""
    
    # Color keywords
    COLORS = [
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
        'black', 'white', 'gray', 'grey', 'brown', 'beige', 'navy',
        'maroon', 'teal', 'cyan', 'magenta', 'lime', 'olive', 'tan'
    ]
    
    # Clothing types
    CLOTHING = [
        'jacket', 'shirt', 'dress', 'pants', 'shoes', 'hat', 'cap',
        'sweater', 'hoodie', 'coat', 'jersey', 'uniform', 'suit'
    ]
    
    # Positions
    POSITIONS = ['left', 'right', 'center', 'middle', 'foreground', 'background']
    
    @staticmethod
    def parse_person_attributes(description: str, bbox: List[float]) -> Dict:
        """
        Extract person attributes from AI description
        
        Args:
            description: AI-generated description (GPT-4V/Claude/BLIP-2)
            bbox: Bounding box [x, y, w, h] normalized
            
        Returns:
            Dict with attributes: clothing_color, clothing_type, gender, position, etc.
        """
        if not description:
            return AIDescriptionParser._get_fallback_attributes(bbox)
        
        description_lower = description.lower()
        attributes = {}
        
        # Extract gender
        if any(word in description_lower for word in ['man', 'male', 'boy', 'gentleman']):
            attributes['gender'] = 'man'
        elif any(word in description_lower for word in ['woman', 'female', 'girl', 'lady']):
            attributes['gender'] = 'woman'
        elif any(word in description_lower for word in ['child', 'kid']):
            attributes['age_group'] = 'child'
        
        # Extract clothing color + type
        # Pattern 1: "wearing [color] [clothing]"
        clothing_match = re.search(
            r'wearing\s+(' + '|'.join(AIDescriptionParser.COLORS) + r')\s+(' + '|'.join(AIDescriptionParser.CLOTHING) + r')',
            description_lower
        )
        if clothing_match:
            attributes['clothing_color'] = clothing_match.group(1)
            attributes['clothing_type'] = clothing_match.group(2)
        
        # Pattern 2: "in [color] [clothing]"
        if 'clothing_color' not in attributes:
            in_match = re.search(
                r'in\s+(' + '|'.join(AIDescriptionParser.COLORS) + r')\s+(' + '|'.join(AIDescriptionParser.CLOTHING) + r')',
                description_lower
            )
            if in_match:
                attributes['clothing_color'] = in_match.group(1)
                attributes['clothing_type'] = in_match.group(2)
        
        # Pattern 3: Just "[color] [clothing]"
        if 'clothing_color' not in attributes:
            color_clothing_match = re.search(
                r'(' + '|'.join(AIDescriptionParser.COLORS) + r')\s+(' + '|'.join(AIDescriptionParser.CLOTHING) + r')',
                description_lower
            )
            if color_clothing_match:
                attributes['clothing_color'] = color_clothing_match.group(1)
                attributes['clothing_type'] = color_clothing_match.group(2)
        
        # Extract position from description
        for pos in AIDescriptionParser.POSITIONS:
            if pos in description_lower:
                attributes['position'] = pos
                break
        
        # Fallback: Calculate position from bbox
        if 'position' not in attributes:
            attributes['position'] = AIDescriptionParser._calculate_position_from_bbox(bbox)
        
        # Extract action
        action_patterns = [
            r'(standing|sitting|walking|running|jumping|pointing|holding|looking)',
            r'(speaking|talking|gesturing|smiling|laughing)'
        ]
        for pattern in action_patterns:
            action_match = re.search(pattern, description_lower)
            if action_match:
                attributes['action'] = action_match.group(1)
                break
        
        return attributes
    
    @staticmethod
    def _calculate_position_from_bbox(bbox: List[float]) -> str:
        """Calculate position descriptor from bbox"""
        if not bbox or len(bbox) < 4:
            return "center"
        
        x, y, w, h = bbox
        center_x = x + w / 2
        
        if center_x < 0.33:
            return "left"
        elif center_x > 0.66:
            return "right"
        else:
            return "center"
    
    @staticmethod
    def _get_fallback_attributes(bbox: List[float]) -> Dict:
        """Fallback attributes when no AI description available"""
        return {
            'position': AIDescriptionParser._calculate_position_from_bbox(bbox)
        }


# ============================================================================
# COMPONENT 2: PHASE 4 EVIDENCE CONVERTER
# ============================================================================

class Phase4EvidenceConverter:
    """Convert Phase 4 frame-based evidence to timeline-based EvidenceDatabase"""
    
    def __init__(self):
        self.person_counter = 0
        self.object_counter = 0
        logger.info("Phase4EvidenceConverter initialized")
    
    def convert(
        self, 
        phase4_evidence: Dict, 
        audio_analysis: Dict
    ):
        """
        Convert Phase 4 evidence to EvidenceDatabase format
        
        Args:
            phase4_evidence: Output from smart_pipeline Phase 4
            audio_analysis: Audio analysis with transcript segments
            
        Returns:
            EvidenceDatabase object for templates
        """
        from templates.base import EvidenceDatabase
        
        logger.info("=" * 80)
        logger.info("CONVERTING PHASE 4 EVIDENCE TO TIMELINE FORMAT")
        logger.info("=" * 80)
        
        video_id = phase4_evidence.get("video_id", "unknown")
        frames = phase4_evidence.get("frames", {})
        
        logger.info(f"Video ID: {video_id}")
        logger.info(f"Total frames: {len(frames)}")
        
        # Initialize aggregated lists
        person_detections = []
        object_detections = []
        scene_detections = []
        ocr_detections = []
        action_detections = []
        body_poses = []
        hand_gestures = []
        face_landmarks = []
        facial_expressions = []
        jersey_numbers = []
        text_orientation = []
        blip2_captions = []
        clip_embeddings = []
        
        # Process each frame
        for frame_id, frame_data in frames.items():
            timestamp = frame_data.get("timestamp", 0.0)
            gt = frame_data.get("ground_truth", {})
            
            # Get AI descriptions (premium frames only)
            gpt4v_desc = gt.get("gpt4v_description")
            claude_desc = gt.get("claude_description")
            blip2_caption = gt.get("blip2_caption")
            
            # Primary AI description (priority: GPT-4V > Claude > BLIP-2)
            ai_description = gpt4v_desc or claude_desc or blip2_caption
            
            # Process YOLO objects (with person attribute extraction)
            yolo_objects = gt.get("yolo_objects", [])
            person_idx = 0
            obj_idx = 0
            
            for yolo_obj in yolo_objects:
                obj_class = yolo_obj.get("class", "unknown")
                bbox = yolo_obj.get("bbox", [0, 0, 0, 0])
                confidence = yolo_obj.get("confidence", 0.0)
                
                if obj_class == "person":
                    # Extract person attributes from AI description
                    attributes = AIDescriptionParser.parse_person_attributes(
                        ai_description if ai_description else "",
                        bbox
                    )
                    
                    # Try to enhance with pose data
                    pose_data = self._find_matching_pose(
                        gt.get("body_poses", []),
                        bbox,
                        timestamp
                    )
                    if pose_data and 'action' not in attributes:
                        attributes['action'] = pose_data.get('pose_type', 'standing')
                    
                    person_detections.append({
                        "person_id": f"{frame_id}_person_{person_idx}",
                        "timestamp": timestamp,
                        "bbox": bbox,
                        "attributes": attributes,
                        "confidence": confidence,
                        "frame_id": frame_id
                    })
                    person_idx += 1
                else:
                    # Regular object
                    object_detections.append({
                        "object_class": obj_class,
                        "timestamp": timestamp,
                        "bbox": bbox,
                        "confidence": confidence,
                        "frame_id": frame_id
                    })
                    obj_idx += 1
            
            # Process OCR detections
            for ocr_item in gt.get("ocr_text", []):
                ocr_detections.append({
                    "text": ocr_item.get("text", ""),
                    "location": self._get_ocr_location(ocr_item.get("bbox", [])),
                    "timestamp": timestamp,
                    "confidence": ocr_item.get("confidence", 0.0),
                    "frame_id": frame_id
                })
            
            # Process scene detection
            scene_type = gt.get("scene_type")
            scene_confidence = gt.get("scene_confidence", 0.0)
            if scene_type and scene_type != "unknown" and scene_type != "analyzed_with_gpt4v":
                scene_detections.append({
                    "scene_type": scene_type,
                    "timestamp": timestamp,
                    "confidence": scene_confidence,
                    "frame_id": frame_id
                })
            
            # Process advanced fields (NEW)
            # Body poses
            for pose in gt.get("body_poses", []):
                body_poses.append({
                    **pose,
                    "timestamp": timestamp,
                    "frame_id": frame_id
                })
            
            # Hand gestures
            for gesture in gt.get("hand_gestures", []):
                hand_gestures.append({
                    **gesture,
                    "timestamp": timestamp,
                    "frame_id": frame_id
                })
            
            # Face landmarks
            face_landmark = gt.get("face_landmarks")
            if face_landmark:
                face_landmarks.append({
                    **face_landmark,
                    "timestamp": timestamp,
                    "frame_id": frame_id
                })
            
            # Facial expressions
            for expr in gt.get("facial_expressions", []):
                facial_expressions.append({
                    **expr,
                    "timestamp": timestamp,
                    "frame_id": frame_id
                })
            
            # Jersey numbers
            for jersey in gt.get("jersey_numbers", []):
                jersey_numbers.append({
                    **jersey,
                    "timestamp": timestamp,
                    "frame_id": frame_id
                })
            
            # Text orientation
            text_orient = gt.get("text_orientation")
            if text_orient:
                text_orientation.append({
                    "orientation": text_orient,
                    "timestamp": timestamp,
                    "frame_id": frame_id
                })
            
            # BLIP-2 captions
            if blip2_caption:
                blip2_captions.append({
                    "caption": blip2_caption,
                    "confidence": gt.get("blip2_confidence", 0.0),
                    "timestamp": timestamp,
                    "frame_id": frame_id
                })
            
            # CLIP embeddings
            clip_emb = gt.get("clip_embeddings")
            if clip_emb:
                clip_embeddings.append({
                    "embedding_vector": clip_emb,
                    "timestamp": timestamp,
                    "frame_id": frame_id
                })
        
        # NEW: Build action_detections from body_poses
        logger.info("\n📊 Building action detections from body poses...")
        action_detections = self._build_action_detections(body_poses, person_detections)
        logger.info(f"   Built {len(action_detections)} action detections")
        
        # NEW: Build event_timeline from all evidence
        logger.info("\n📊 Building event timeline...")
        event_timeline = self._build_event_timeline(
            action_detections,
            scene_detections,
            object_detections,
            ocr_detections,
            audio_analysis.get("segments", [])
        )
        logger.info(f"   Built timeline with {len(event_timeline)} events")
        
        # Create EvidenceDatabase
        evidence_db = EvidenceDatabase(
            video_id=video_id,
            duration=audio_analysis.get("duration", 0.0),
            transcript_segments=audio_analysis.get("segments", []),
            person_detections=person_detections,
            object_detections=object_detections,
            scene_detections=scene_detections,
            ocr_detections=ocr_detections,
            action_detections=action_detections,
            body_poses=body_poses,
            hand_gestures=hand_gestures,
            face_landmarks=face_landmarks,
            facial_expressions=facial_expressions,
            jersey_numbers=jersey_numbers,
            text_orientation=text_orientation,
            blip2_captions=blip2_captions,
            clip_embeddings=clip_embeddings,
            event_timeline=event_timeline
        )
        
        logger.info("✅ Conversion Complete!")
        logger.info(f"   Person detections: {len(person_detections)}")
        logger.info(f"   Object detections: {len(object_detections)}")
        logger.info(f"   OCR detections: {len(ocr_detections)}")
        logger.info(f"   Scene detections: {len(scene_detections)}")
        logger.info(f"   Body poses: {len(body_poses)}")
        logger.info(f"   Hand gestures: {len(hand_gestures)}")
        logger.info(f"   Facial expressions: {len(facial_expressions)}")
        logger.info(f"   Action detections: {len(action_detections)} (NEW)")
        logger.info(f"   Event timeline: {len(event_timeline)} events (NEW)")
        logger.info(f"   Transcript segments: {len(evidence_db.transcript_segments)}")
        logger.info("=" * 80)
        
        return evidence_db
    
    def _build_action_detections(
        self, 
        body_poses: List[Dict],
        person_detections: List[Dict]
    ) -> List[Dict]:
        """
        Build action_detections from body_poses
        
        Aggregates actions across frames and tracks action changes.
        Supports Type B questions that require counting actions.
        
        Args:
            body_poses: List of body pose detections from all frames
            person_detections: List of person detections with person_id
            
        Returns:
            List of action detections with structure:
            [{
                "action": "dribbling",
                "person_id": "frame_001_person_0",
                "timestamp": 65.0,
                "confidence": 0.9,
                "frame_id": "frame_001"
            }]
        """
        action_detections = []
        
        # Group poses by timestamp to match with persons
        poses_by_timestamp = {}
        for pose in body_poses:
            ts = pose.get("timestamp")
            if ts not in poses_by_timestamp:
                poses_by_timestamp[ts] = []
            poses_by_timestamp[ts].append(pose)
        
        # For each person detection, try to find matching pose
        for person in person_detections:
            person_id = person.get("person_id")
            timestamp = person.get("timestamp")
            bbox = person.get("bbox", [])
            
            # Find poses at this timestamp
            frame_poses = poses_by_timestamp.get(timestamp, [])
            
            # Find best matching pose for this person
            best_pose = None
            best_overlap = 0.0
            
            for pose in frame_poses:
                pose_bbox = pose.get("bbox", [])
                overlap = self._bbox_overlap(bbox, pose_bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pose = pose
            
            # If we found a matching pose, create action detection
            if best_pose and best_overlap > 0.3:  # Require 30% overlap
                action_detections.append({
                    "action": best_pose.get("pose_type", "unknown"),
                    "person_id": person_id,
                    "timestamp": timestamp,
                    "confidence": best_pose.get("confidence", 0.0),
                    "frame_id": person.get("frame_id"),
                    "bbox": pose_bbox
                })
        
        # Sort by timestamp
        action_detections.sort(key=lambda x: x["timestamp"])
        
        logger.info(f"      Built {len(action_detections)} action detections from {len(body_poses)} poses")
        return action_detections
    
    def _build_event_timeline(
        self,
        action_detections: List[Dict],
        scene_detections: List[Dict],
        object_detections: List[Dict],
        ocr_detections: List[Dict],
        transcript_segments: List[Dict]
    ) -> List[Dict]:
        """
        Build comprehensive event timeline from all evidence sources
        
        Creates a sorted timeline of events including:
        - Action start/stop events (person starts/stops action)
        - Scene changes (different scene_type)
        - Object first appearances
        - OCR text appearances
        - Audio events (speech segments)
        
        Args:
            action_detections: List of action detections
            scene_detections: List of scene detections
            object_detections: List of object detections
            ocr_detections: List of OCR detections
            transcript_segments: List of transcript segments
            
        Returns:
            List of timeline events sorted by timestamp:
            [{
                "event_type": "person_action_start",
                "person_id": "frame_001_person_0",
                "description": "started running",
                "timestamp": 65.0,
                "frame_id": "frame_001"
            }]
        """
        timeline = []
        
        # 1. Add action change events
        prev_actions = {}  # person_id -> last action
        for action in action_detections:
            person_id = action.get("person_id")
            current_action = action.get("action")
            timestamp = action.get("timestamp")
            
            prev_action = prev_actions.get(person_id)
            
            if prev_action and prev_action != current_action:
                # Action changed
                timeline.append({
                    "event_type": "person_action_change",
                    "person_id": person_id,
                    "description": f"changed from {prev_action} to {current_action}",
                    "timestamp": timestamp,
                    "frame_id": action.get("frame_id"),
                    "previous_action": prev_action,
                    "new_action": current_action
                })
            elif not prev_action:
                # First action detected
                timeline.append({
                    "event_type": "person_action_start",
                    "person_id": person_id,
                    "description": f"started {current_action}",
                    "timestamp": timestamp,
                    "frame_id": action.get("frame_id"),
                    "action": current_action
                })
            
            prev_actions[person_id] = current_action
        
        # 2. Add scene change events
        prev_scene = None
        for scene in scene_detections:
            scene_type = scene.get("scene_type")
            timestamp = scene.get("timestamp")
            
            if prev_scene and prev_scene != scene_type:
                timeline.append({
                    "event_type": "scene_change",
                    "description": f"scene changed from {prev_scene} to {scene_type}",
                    "timestamp": timestamp,
                    "frame_id": scene.get("frame_id"),
                    "previous_scene": prev_scene,
                    "new_scene": scene_type
                })
            
            prev_scene = scene_type
        
        # 3. Add object first appearances
        seen_objects = set()
        for obj in object_detections:
            obj_class = obj.get("object_class")
            timestamp = obj.get("timestamp")
            
            if obj_class not in seen_objects:
                timeline.append({
                    "event_type": "object_first_appearance",
                    "object_class": obj_class,
                    "description": f"{obj_class} first appeared",
                    "timestamp": timestamp,
                    "frame_id": obj.get("frame_id")
                })
                seen_objects.add(obj_class)
        
        # 4. Add OCR text appearances
        seen_text = set()
        for ocr in ocr_detections:
            text = ocr.get("text", "").strip()
            timestamp = ocr.get("timestamp")
            
            if text and text not in seen_text and len(text) > 2:  # Skip very short text
                timeline.append({
                    "event_type": "text_appearance",
                    "text": text,
                    "description": f"text '{text}' appeared",
                    "timestamp": timestamp,
                    "frame_id": ocr.get("frame_id"),
                    "location": ocr.get("location", "screen")
                })
                seen_text.add(text)
        
        # 5. Add audio events (speech segments)
        for segment in transcript_segments:
            text = segment.get("text", "").strip()
            timestamp = segment.get("start", 0.0)
            
            if text:
                timeline.append({
                    "event_type": "speech_segment",
                    "description": f"speech: '{text[:50]}{'...' if len(text) > 50 else ''}'",
                    "timestamp": timestamp,
                    "text": text,
                    "speaker": segment.get("speaker")
                })
        
        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x["timestamp"])
        
        logger.info(f"      Built timeline with {len(timeline)} events:")
        logger.info(f"        Action changes: {len([e for e in timeline if 'action' in e.get('event_type', '')])}")
        logger.info(f"        Scene changes: {len([e for e in timeline if e.get('event_type') == 'scene_change'])}")
        logger.info(f"        Object appearances: {len([e for e in timeline if e.get('event_type') == 'object_first_appearance'])}")
        logger.info(f"        Text appearances: {len([e for e in timeline if e.get('event_type') == 'text_appearance'])}")
        logger.info(f"        Speech segments: {len([e for e in timeline if e.get('event_type') == 'speech_segment'])}")
        
        return timeline
    
    def _find_matching_pose(self, poses: List[Dict], bbox: List[float], timestamp: float) -> Optional[Dict]:
        """Find pose matching person bbox"""
        if not poses or not bbox:
            return None
        
        # Find pose with overlapping bbox
        for pose in poses:
            pose_bbox = pose.get("bbox", [])
            if self._bbox_overlap(bbox, pose_bbox) > 0.5:
                return pose
        
        return None
    
    def _bbox_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU overlap between two bboxes"""
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_ocr_location(self, bbox: List) -> str:
        """Get natural language location from OCR bbox"""
        if not bbox or len(bbox) < 4:
            return "screen"
        
        # OCR bbox format is typically [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        if isinstance(bbox[0], list):
            # Get center
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
        else:
            # Standard [x, y, w, h] format
            center_x = bbox[0] + bbox[2] / 2
            center_y = bbox[1] + bbox[3] / 2
        
        # Determine location
        h_pos = "left" if center_x < 0.33 else ("right" if center_x > 0.66 else "center")
        v_pos = "top" if center_y < 0.33 else ("bottom" if center_y > 0.66 else "middle")
        
        if v_pos == "middle":
            return h_pos
        else:
            return f"{v_pos} {h_pos}"


# ============================================================================
# COMPONENT 3: PREMIUM FRAME ANALYZER
# ============================================================================

class PremiumFrameAnalyzer:
    """Analyze premium frames with AI consensus"""
    
    def __init__(self):
        logger.info("PremiumFrameAnalyzer initialized")
    
    def analyze_premium_frames(self, phase4_evidence: Dict) -> Dict:
        """
        Analyze premium frames and separate by consensus
        
        Returns:
            {
                "agreed_frames": [...],     # consensus_reached=True, similarity>0.7
                "disagreed_frames": [...],  # consensus_reached=False
                "ai_descriptions": {...}    # frame_id -> descriptions
            }
        """
        logger.info("\n📊 Analyzing Premium Frames with AI Consensus")
        
        frames = phase4_evidence.get("frames", {})
        
        agreed_frames = []
        disagreed_frames = []
        ai_descriptions = {}
        
        for frame_id, frame_data in frames.items():
            gt = frame_data.get("ground_truth", {})
            ai_consensus = gt.get("ai_consensus")
            
            # Only premium frames have AI consensus
            if not ai_consensus:
                continue
            
            gpt4v_desc = gt.get("gpt4v_description", "")
            claude_desc = gt.get("claude_description", "")
            
            # Store descriptions
            ai_descriptions[frame_id] = {
                "gpt4v": gpt4v_desc,
                "claude": claude_desc,
                "consensus": ai_consensus
            }
            
            # Categorize by consensus
            if ai_consensus.get("consensus_reached", False) and ai_consensus.get("similarity_score", 0) > 0.7:
                agreed_frames.append(frame_data)
            else:
                disagreed_frames.append(frame_data)

        # Sort frames by complexity (high → medium → low) and priority
        # Priority: is_key_frame, then complexity, then timestamp
        def get_frame_priority(frame):
            complexity_map = {"high": 3, "medium": 2, "low": 1}
            complexity = frame.get("complexity", "medium")
            is_key = 1 if frame.get("is_key_frame", False) else 0
            timestamp = frame.get("timestamp", 0)
            return (is_key, complexity_map.get(complexity, 2), -timestamp)

        agreed_frames.sort(key=get_frame_priority, reverse=True)
        disagreed_frames.sort(key=get_frame_priority, reverse=True)

        logger.info(f"   Premium frames with AI consensus: {len(ai_descriptions)}")
        logger.info(f"   Agreed frames (GPT-4V & Claude agree): {len(agreed_frames)}")
        logger.info(f"   Disagreed frames (potential adversarial): {len(disagreed_frames)}")
        logger.info(f"   ✓ Frames sorted by: key_frame > complexity > recency")

        return {
            "agreed_frames": agreed_frames,
            "disagreed_frames": disagreed_frames,
            "ai_descriptions": ai_descriptions
        }


# ============================================================================
# COMPONENT 4: AI QUESTION GENERATOR
# ============================================================================

class AIQuestionGenerator:
    """Generate questions using GPT-4V/Claude descriptions"""
    
    def __init__(self, openai_api_key: str, claude_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.claude_api_key = claude_api_key
        openai.api_key = openai_api_key
        
        if claude_api_key:
            self.claude_client = Anthropic(api_key=claude_api_key)
        else:
            self.claude_client = None
        
        self.total_cost = 0.0
        logger.info("AIQuestionGenerator initialized")
    
    def generate_gpt4v_questions(
        self,
        agreed_frames: List[Dict],
        audio_analysis: Dict,
        count: int = 3
    ) -> List[MultimodalQuestion]:
        """
        Generate questions from frames where GPT-4V & Claude agreed
        
        Strategy: Use GPT-4V description as ground truth for complex questions
        """
        logger.info(f"\n🤖 Generating {count} GPT-4V Questions (Agreed Frames)")
        
        if not agreed_frames:
            logger.warning("   No agreed frames available")
            return []
        
        questions = []
        
        for i, frame_data in enumerate(agreed_frames[:count]):
            logger.info(f"   [{i+1}/{min(count, len(agreed_frames))}] Processing frame {frame_data.get('frame_id')}...")
            
            question = self._generate_single_gpt4v_question(
                frame_data,
                audio_analysis,
                question_id=f"gpt4v_{i+1:03d}"
            )
            
            if question:
                questions.append(question)
                logger.info(f"      ✓ Generated question")
            else:
                logger.warning(f"      ✗ Failed to generate question")
        
        logger.info(f"   Generated {len(questions)} GPT-4V questions")
        return questions
    
    def generate_claude_questions(
        self,
        disagreed_frames: List[Dict],
        audio_analysis: Dict,
        count: int = 7
    ) -> List[MultimodalQuestion]:
        """
        Generate HARDER questions from disagreement frames
        
        Strategy: Use disagreement to create adversarial questions
        """
        logger.info(f"\n🤖 Generating {count} Claude Questions (Disagreed Frames)")
        
        if not disagreed_frames:
            logger.warning("   No disagreed frames available")
            return []
        
        questions = []
        
        for i, frame_data in enumerate(disagreed_frames[:count]):
            logger.info(f"   [{i+1}/{min(count, len(disagreed_frames))}] Processing frame {frame_data.get('frame_id')}...")
            
            question = self._generate_single_claude_question(
                frame_data,
                audio_analysis,
                question_id=f"claude_{i+1:03d}"
            )
            
            if question:
                questions.append(question)
                logger.info(f"      ✓ Generated question")
            else:
                logger.warning(f"      ✗ Failed to generate question")
        
        logger.info(f"   Generated {len(questions)} Claude questions")
        return questions
    
    def _generate_single_gpt4v_question(
        self,
        frame_data: Dict,
        audio_analysis: Dict,
        question_id: str
    ) -> Optional[MultimodalQuestion]:
        """Generate single question using GPT-4V description"""
        
        gt = frame_data.get("ground_truth", {})
        timestamp = frame_data.get("timestamp", 0.0)
        
        gpt4v_desc = gt.get("gpt4v_description", "")
        if not gpt4v_desc:
            return None
        
        # Find audio cue near this timestamp
        audio_cue = self._find_audio_near_timestamp(timestamp, audio_analysis)
        if not audio_cue:
            return None
        
        audio_text = audio_cue["text"][:60]
        if len(audio_cue["text"]) > 60:
            audio_text += "..."
        
        # Select random question type
        question_type = random.choice([
            'temporal', 'inference', 'counting', 'comparative', 'needle',
            'object_interaction', 'subscene', 'holistic', 'context'
        ])

        # Generate question using GPT-4
        try:
            prompt = f"""Generate a challenging multimodal video question that requires BOTH audio and visual information to answer.

AUDIO CUE: Someone says "{audio_text}" at timestamp {timestamp:.1f}s

VISUAL CONTEXT (from GPT-4 Vision analysis): {gpt4v_desc}

QUESTION TYPE: {question_type.upper()}
Generate a {question_type} question following these patterns:
- TEMPORAL: "What is happening on screen during the moment when someone says X?"
- INFERENCE: "Based on what's visible when someone says X, what can you infer?"
- COUNTING: "How many [objects/people] are visible when someone says X?"
- COMPARATIVE: "What visual elements change during the segment when someone says X?"
- NEEDLE: "What specific visual details are present when someone says X?"
- OBJECT_INTERACTION: "Describe the interactions or movements visible when someone says X."
- SUBSCENE: "Describe the complete scene visible when you hear X."
- HOLISTIC: "What is the overall context and situation when someone says X?"
- CONTEXT: "What background elements or environmental context are visible when someone says X?"

REQUIREMENTS:
1. Question MUST require BOTH the audio cue AND visual context to answer
2. Use descriptors only (NO names): "person in blue shirt", "player in jersey #10"
3. Follow the specific {question_type} question pattern above
4. Question should be challenging and require both modalities
5. Answer must reference BOTH audio cue and visual details

FORMAT:
Question: [your {question_type} question here]
Answer: [golden answer referencing both audio and visual]"""

            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.8
            )

            self.total_cost += 0.02  # Estimate

            content = response.choices[0].message.content

            # Parse response
            question_text, answer_text = self._parse_qa_response(content)
            if not question_text or not answer_text:
                return None

            # Determine task_types based on question_type
            task_type_mapping = {
                'temporal': ['temporal', 'referential', 'context'],
                'inference': ['inference', 'context', 'holistic'],
                'counting': ['counting', 'referential', 'context'],
                'comparative': ['comparative', 'temporal', 'context'],
                'needle': ['needle', 'referential', 'context'],
                'object_interaction': ['object_interaction', 'referential', 'sequential'],
                'subscene': ['subscene', 'context', 'holistic'],
                'holistic': ['holistic', 'inference', 'context'],
                'context': ['context', 'referential']
            }

            # Calculate timestamps
            start_ts = self._format_timestamp(max(0, audio_cue["start"] - 1))
            end_ts = self._format_timestamp(audio_cue["end"] + 2)

            return MultimodalQuestion(
                question_id=question_id,
                question=question_text,
                golden_answer=answer_text,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                audio_cue=audio_text,
                visual_cue=gpt4v_desc[:100],
                task_types=task_type_mapping.get(question_type, ["temporal", "referential"]),
                generation_tier="ai_gpt4v",
                complexity="high",
                requires_both_modalities=True
            )
            
        except Exception as e:
            logger.error(f"GPT-4 generation failed: {e}")
            return None
    
    def _generate_single_claude_question(
        self,
        frame_data: Dict,
        audio_analysis: Dict,
        question_id: str
    ) -> Optional[MultimodalQuestion]:
        """Generate diverse adversarial question from disagreement

        Uses 13 task types with weighted distribution:
        - Temporal/Sequential (30%)
        - Inference (20%)
        - Counting (10%)
        - Comparative (10%)
        - Needle (10%)
        - Object Interaction (5%)
        - Subscene (5%)
        - Spurious Correlation (5%)
        - Context (5%)
        """

        gt = frame_data.get("ground_truth", {})
        timestamp = frame_data.get("timestamp", 0.0)

        gpt4v_desc = gt.get("gpt4v_description", "")
        claude_desc = gt.get("claude_description", "")

        if not gpt4v_desc or not claude_desc:
            return None

        # Find audio cue
        audio_cue = self._find_audio_near_timestamp(timestamp, audio_analysis)
        if not audio_cue:
            return None

        audio_text = audio_cue["text"][:60]
        if len(audio_cue["text"]) > 60:
            audio_text += "..."

        # Select question pattern based on weighted distribution
        pattern_type = random.choices(
            ['temporal', 'inference', 'counting', 'comparative', 'needle',
             'object_interaction', 'subscene', 'spurious', 'context'],
            weights=[30, 20, 10, 10, 10, 5, 5, 5, 5]
        )[0]

        # Generate question based on selected pattern
        question_data = self._generate_pattern_question(
            pattern_type, audio_text, claude_desc, gpt4v_desc
        )

        start_ts = self._format_timestamp(max(0, audio_cue["start"] - 1))
        end_ts = self._format_timestamp(audio_cue["end"] + 2)

        return MultimodalQuestion(
            question_id=question_id,
            question=question_data["question"],
            golden_answer=question_data["answer"],
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            audio_cue=audio_text,
            visual_cue=claude_desc[:100],
            task_types=question_data["task_types"],
            generation_tier="ai_claude",
            complexity="high",
            requires_both_modalities=True
        )

    def _generate_pattern_question(
        self,
        pattern_type: str,
        audio_text: str,
        claude_desc: str,
        gpt4v_desc: str
    ) -> Dict:
        """Generate question based on specific pattern type"""

        # Extract key visual elements from descriptions
        visual_summary = self._extract_visual_summary(claude_desc)

        patterns = {
            'temporal': {
                'question': f'What is happening on screen during the moment when someone says "{audio_text}"?',
                'answer': f'During "{audio_text}", {claude_desc[:200]}',
                'task_types': ['temporal', 'referential', 'context']
            },
            'inference': {
                'question': f'Based on what\'s visible when someone says "{audio_text}", what can you infer about the situation?',
                'answer': f'When "{audio_text}" is said, {claude_desc[:150]} This suggests the scene context and activity at that moment.',
                'task_types': ['inference', 'context', 'holistic']
            },
            'counting': {
                'question': f'How many distinct {visual_summary} are visible when someone says "{audio_text}"?',
                'answer': f'When "{audio_text}" is heard, {claude_desc[:150]}',
                'task_types': ['counting', 'referential', 'context']
            },
            'comparative': {
                'question': f'What visual elements change or remain constant during the segment when someone says "{audio_text}"?',
                'answer': f'At "{audio_text}", {claude_desc[:180]}',
                'task_types': ['comparative', 'temporal', 'context']
            },
            'needle': {
                'question': f'What specific visual details are present on screen when someone says "{audio_text}"?',
                'answer': f'When "{audio_text}" is said, specific details include: {claude_desc[:180]}',
                'task_types': ['needle', 'referential', 'context']
            },
            'object_interaction': {
                'question': f'Describe the interactions or movements visible when someone says "{audio_text}".',
                'answer': f'During "{audio_text}", {claude_desc[:180]}',
                'task_types': ['object_interaction', 'referential', 'sequential']
            },
            'subscene': {
                'question': f'Describe the complete scene visible when you hear "{audio_text}".',
                'answer': f'When "{audio_text}" is heard, the scene shows: {claude_desc[:200]}',
                'task_types': ['subscene', 'context', 'holistic']
            },
            'spurious': {
                'question': f'What unexpected or notable visual element appears when someone says "{audio_text}"?',
                'answer': f'Notably, when "{audio_text}" is said, {claude_desc[:180]} (This represents a potential disagreement between visual models)',
                'task_types': ['spurious_correlation', 'context', 'inference']
            },
            'context': {
                'question': f'What background elements or environmental context are visible when someone says "{audio_text}"?',
                'answer': f'When "{audio_text}" is heard, the background shows: {claude_desc[:180]}',
                'task_types': ['context', 'referential']
            }
        }

        return patterns.get(pattern_type, patterns['temporal'])

    def _extract_visual_summary(self, description: str) -> str:
        """Extract key visual elements for counting questions"""

        # Common countable elements
        if "player" in description.lower():
            return "players"
        elif "person" in description.lower() or "people" in description.lower():
            return "people"
        elif "object" in description.lower():
            return "objects"
        elif "uniform" in description.lower():
            return "uniformed individuals"
        else:
            return "visual elements"
    
    def _find_audio_near_timestamp(
        self,
        timestamp: float,
        audio_analysis: Dict,
        max_distance: float = 3.0
    ) -> Optional[Dict]:
        """Find audio segment near timestamp"""
        segments = audio_analysis.get("segments", [])
        
        closest = None
        min_dist = float('inf')
        
        for segment in segments:
            dist = abs(segment["start"] - timestamp)
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                closest = segment
        
        return closest
    
    def _parse_qa_response(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse Q&A from GPT-4 response"""
        lines = content.strip().split('\n')
        
        question = None
        answer = None
        
        for line in lines:
            if line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
            elif line.startswith("Answer:"):
                answer = line.replace("Answer:", "").strip()
        
        return question, answer
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ============================================================================
# COMPONENT 5: TEMPLATE INTEGRATOR
# ============================================================================

class TemplateIntegrator:
    """Integrate with template registry for generating questions"""
    
    def __init__(self):
        logger.info("TemplateIntegrator initialized")
    
    def generate_template_questions(
        self,
        evidence_db,
        target_count: int = 40,
        keep_best_n: Optional[int] = 20
    ) -> List[MultimodalQuestion]:
        """
        Generate questions using template registry
        
        Args:
            evidence_db: EvidenceDatabase object
            target_count: Target number of questions to generate (default 40)
            keep_best_n: Keep only the best N questions (default 20, None = keep all)
            
        Returns:
            List of MultimodalQuestion objects (best keep_best_n if specified)
        """
        logger.info(f"\n📋 Generating {target_count} Template Questions")
        if keep_best_n:
            logger.info(f"   Will select best {keep_best_n} questions")
        
        try:
            from templates import get_registry
            
            registry = get_registry()
            logger.info(f"   Loaded template registry with {len(registry.templates)} templates")
            
            # Generate questions using registry
            generated_questions = registry.generate_tier1_questions(
                evidence_db,
                target_count=target_count
            )
            
            logger.info(f"   Generated {len(generated_questions)} template questions")
            
            # Convert to MultimodalQuestion format
            multimodal_questions = []
            
            for i, gq in enumerate(generated_questions):
                # Convert GeneratedQuestion to MultimodalQuestion
                question_dict = gq.to_dict()
                
                # Extract audio and visual cues
                audio_cue_content = ""
                visual_cue_content = ""
                
                if question_dict.get("audio_cues"):
                    audio_cue_content = question_dict["audio_cues"][0]["content"]
                
                if question_dict.get("visual_cues"):
                    visual_cue_content = question_dict["visual_cues"][0]["content"]
                
                # Format timestamps
                start_ts = self._format_timestamp(question_dict["start_timestamp"])
                end_ts = self._format_timestamp(question_dict["end_timestamp"])
                
                multimodal_questions.append(MultimodalQuestion(
                    question_id=f"template_{i+1:03d}",
                    question=question_dict["question"],
                    golden_answer=question_dict["golden_answer"],
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                    audio_cue=audio_cue_content,
                    visual_cue=visual_cue_content,
                    task_types=[qt for qt in question_dict.get("question_types", [])],
                    generation_tier="template",
                    complexity=question_dict.get("complexity_score", 0.5)  # Use actual complexity
                ))
            
            # Select best N questions if requested
            if keep_best_n and len(multimodal_questions) > keep_best_n:
                logger.info(f"\n   Selecting best {keep_best_n} from {len(multimodal_questions)} questions...")
                multimodal_questions = self._select_best_questions(
                    multimodal_questions,
                    keep_n=keep_best_n
                )
                logger.info(f"   Selected {len(multimodal_questions)} best questions")
            
            return multimodal_questions
            
        except ImportError as e:
            logger.error(f"Failed to import template registry: {e}")
            logger.error("Make sure templates package is in Python path")
            return []
        except Exception as e:
            logger.error(f"Template generation failed: {e}", exc_info=True)
            return []
    
    def _select_best_questions(
        self,
        questions: List[MultimodalQuestion],
        keep_n: int
    ) -> List[MultimodalQuestion]:
        """
        Select best N questions based on complexity and diversity
        
        Selection criteria:
        1. Complexity score (higher = better)
        2. Diversity of task types
        3. Evidence quality (more cues = better)
        
        Args:
            questions: List of all questions
            keep_n: Number of questions to keep
            
        Returns:
            List of best N questions
        """
        # Score each question
        scored_questions = []
        
        for q in questions:
            score = 0.0
            
            # Factor 1: Complexity (if stored as float, use it; otherwise parse string)
            if isinstance(q.complexity, (int, float)):
                score += q.complexity * 0.5
            elif q.complexity == "high":
                score += 0.5
            elif q.complexity == "medium":
                score += 0.3
            else:  # low
                score += 0.1
            
            # Factor 2: Number of task types (diversity)
            score += min(len(q.task_types), 3) * 0.15
            
            # Factor 3: Evidence quality (longer cues = more specific)
            audio_quality = min(len(q.audio_cue), 100) / 100.0 * 0.15
            visual_quality = min(len(q.visual_cue), 100) / 100.0 * 0.15
            score += audio_quality + visual_quality
            
            # Factor 4: Question length (longer = more complex, up to a point)
            optimal_length = 80
            length_ratio = min(len(q.question), optimal_length) / optimal_length
            score += length_ratio * 0.05
            
            scored_questions.append((score, q))
        
        # Sort by score (descending)
        scored_questions.sort(key=lambda x: x[0], reverse=True)
        
        # Take top N
        selected = [q for score, q in scored_questions[:keep_n]]
        
        # Re-assign question IDs
        for i, q in enumerate(selected):
            q.question_id = f"template_{i+1:03d}"
        
        logger.info(f"      Score range: {scored_questions[0][0]:.3f} (best) to {scored_questions[keep_n-1][0]:.3f} (kept) to {scored_questions[-1][0]:.3f} (worst)")
        
        return selected
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ============================================================================
# COMPONENT 6: UNIFIED VALIDATOR
# ============================================================================

class UnifiedValidator:
    """Validate all questions (AI + template)"""
    
    def __init__(self):
        logger.info("UnifiedValidator initialized")
    
    def validate_all_questions(
        self,
        questions: List[MultimodalQuestion],
        evidence_db
    ) -> List[MultimodalQuestion]:
        """
        Validate all questions
        
        Checks:
        1. Dual cue requirement
        2. No names
        3. Precise timestamps
        4. Evidence grounding
        """
        logger.info(f"\n✓ Validating {len(questions)} Questions")
        
        validated = []
        
        for i, question in enumerate(questions):
            is_valid, reason = self._validate_question(question, evidence_db)
            
            question.validated = is_valid
            question.validation_notes = reason
            
            if is_valid:
                validated.append(question)
            else:
                logger.warning(f"   Question {i+1} invalid: {reason}")
        
        logger.info(f"   ✅ {len(validated)}/{len(questions)} questions passed validation")
        
        return validated
    
    def _validate_question(
        self,
        question: MultimodalQuestion,
        evidence_db
    ) -> Tuple[bool, str]:
        """Validate single question"""
        
        # Check 1: Has both audio and visual cues
        if not question.audio_cue or not question.visual_cue:
            return False, "Missing audio or visual cue"
        
        # Check 2: Check for names in question and answer
        all_names = (
            evidence_db.character_names +
            evidence_db.team_names +
            evidence_db.media_names +
            evidence_db.brand_names
        )
        
        text_to_check = question.question + " " + question.golden_answer
        text_lower = text_to_check.lower()
        
        for name in all_names:
            if name.lower() in text_lower:
                return False, f"Contains name: {name}"
        
        # Check 3: Question references audio
        if not any(word in question.question.lower() for word in ["says", "said", "hear", "audio", "spoken", "when someone"]):
            return False, "Question doesn't reference audio cue"
        
        # Check 4: Question references visual
        if not any(word in question.question.lower() for word in ["see", "visible", "screen", "appears", "shown", "wearing", "on screen"]):
            return False, "Question doesn't reference visual cue"
        
        return True, "Valid"


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class MultimodalQuestionGeneratorV2:
    """
    Complete multimodal question generator
    
    Orchestrates all components to generate 30 questions:
    - 3 GPT-4V questions (agreed frames)
    - 7 Claude questions (disagreed frames)
    - 20 Template questions (all frames)
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        claude_api_key: Optional[str] = None
    ):
        """Initialize question generator"""
        if openai_api_key:
            self.openai_api_key = openai_api_key
        elif os.getenv("OPENAI_API_KEY"):
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key required")
        
        self.claude_api_key = claude_api_key or os.getenv("CLAUDE_API_KEY")
        
        # Initialize components
        self.converter = Phase4EvidenceConverter()
        self.premium_analyzer = PremiumFrameAnalyzer()
        self.ai_generator = AIQuestionGenerator(
            self.openai_api_key,
            self.claude_api_key
        )
        self.template_integrator = TemplateIntegrator()
        self.validator = UnifiedValidator()
        
        self.total_cost = 0.0
        
        logger.info("=" * 80)
        logger.info("MULTIMODAL QUESTION GENERATOR V2 - INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"OpenAI API: {'✓' if self.openai_api_key else '✗'}")
        logger.info(f"Claude API: {'✓' if self.claude_api_key else '✗'}")
        logger.info("=" * 80)
    
    def generate_questions(
        self,
        phase4_evidence: Dict,
        audio_analysis: Dict,
        video_id: str = "unknown",
        target_gpt4v: int = None,
        target_claude: int = None,
        target_template: int = 40,
        keep_best_template: int = None
    ) -> QuestionGenerationResult:
        """
        Generate all questions with dynamic allocation based on premium frames

        ALLOCATION STRATEGY:
        - Total: Always 30 questions
        - If premium_frames >= 20: 20 AI (10 GPT-4V + 10 Claude) + 10 template
        - If premium_frames < 20: min(premium, 20) AI + remaining template
        - AI split: 50/50 between GPT-4V and Claude

        Args:
            phase4_evidence: Phase 4 evidence from smart_pipeline
            audio_analysis: Audio analysis with transcript
            video_id: Video identifier
            target_gpt4v: Target GPT-4V questions (auto-calculated if None)
            target_claude: Target Claude questions (auto-calculated if None)
            target_template: Generate this many template questions (default 40)
            keep_best_template: Keep best N template questions (auto-calculated if None)

        Returns:
            QuestionGenerationResult with 30 total questions
        """
        logger.info("=" * 80)
        logger.info("STARTING QUESTION GENERATION")
        logger.info("=" * 80)

        result = QuestionGenerationResult(video_id=video_id)

        # Step 1: Convert Phase 4 evidence to EvidenceDatabase
        logger.info("\n📦 Step 1: Converting Evidence Format")
        evidence_db = self.converter.convert(phase4_evidence, audio_analysis)

        # Step 2: Analyze premium frames
        logger.info("\n📊 Step 2: Analyzing Premium Frames")
        premium_analysis = self.premium_analyzer.analyze_premium_frames(phase4_evidence)

        # Calculate dynamic allocation if not specified
        premium_count = len(premium_analysis["agreed_frames"]) + len(premium_analysis["disagreed_frames"])

        if target_gpt4v is None or target_claude is None or keep_best_template is None:
            logger.info(f"\n🎯 Dynamic Allocation (Premium Frames: {premium_count})")

            # Dynamic allocation logic
            if premium_count >= 20:
                ai_questions = 20
                template_questions = 10
            else:
                ai_questions = min(premium_count, 20)
                template_questions = 30 - ai_questions

            # Split AI questions 50/50
            if target_gpt4v is None:
                target_gpt4v = ai_questions // 2
            if target_claude is None:
                target_claude = ai_questions - target_gpt4v
            if keep_best_template is None:
                keep_best_template = template_questions

            logger.info(f"   AI Questions: {ai_questions} ({target_gpt4v} GPT-4V + {target_claude} Claude)")
            logger.info(f"   Template Questions: {keep_best_template} (from {target_template} generated)")
            logger.info(f"   Total: {target_gpt4v + target_claude + keep_best_template} questions")

        logger.info(f"\n📋 Final Allocation: {target_gpt4v} GPT-4V + {target_claude} Claude + {keep_best_template} template")
        logger.info(f"   Premium frames available: {premium_count}")
        logger.info(f"     - Agreed (consensus): {len(premium_analysis['agreed_frames'])}")
        logger.info(f"     - Disagreed (adversarial): {len(premium_analysis['disagreed_frames'])}")
        
        # Step 3: Generate AI questions (GPT-4V + Claude)
        logger.info("\n🤖 Step 3: Generating AI Questions")
        
        # GPT-4V questions (agreed frames)
        gpt4v_questions = self.ai_generator.generate_gpt4v_questions(
            premium_analysis["agreed_frames"],
            audio_analysis,
            count=target_gpt4v
        )
        result.questions.extend(gpt4v_questions)
        
        # Claude questions (disagreed frames)
        claude_questions = self.ai_generator.generate_claude_questions(
            premium_analysis["disagreed_frames"],
            audio_analysis,
            count=target_claude
        )
        result.questions.extend(claude_questions)
        
        # Step 4: Generate template questions (40) and keep best (20)
        logger.info("\n📋 Step 4: Generating Template Questions")
        template_questions = self.template_integrator.generate_template_questions(
            evidence_db,
            target_count=target_template,
            keep_best_n=keep_best_template
        )
        result.questions.extend(template_questions)
        
        # Step 5: Validate all questions
        logger.info("\n✓ Step 5: Validating All Questions")
        result.questions = self.validator.validate_all_questions(
            result.questions,
            evidence_db
        )
        
        # Calculate final statistics
        result.total_questions = len(result.questions)
        result.validated_questions = len([q for q in result.questions if q.validated])
        result.generation_cost = self.ai_generator.total_cost
        
        result.metadata = {
            "gpt4v_questions": len(gpt4v_questions),
            "claude_questions": len(claude_questions),
            "template_questions": len(template_questions),
            "premium_frames_available": len(premium_analysis["agreed_frames"]) + len(premium_analysis["disagreed_frames"])
        }
        
        logger.info("=" * 80)
        logger.info("✅ QUESTION GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total questions: {result.total_questions}")
        logger.info(f"  GPT-4V: {len(gpt4v_questions)}")
        logger.info(f"  Claude: {len(claude_questions)}")
        logger.info(f"  Template: {len(template_questions)}")
        logger.info(f"Validated: {result.validated_questions}/{result.total_questions}")
        logger.info(f"Generation cost: ${result.generation_cost:.4f}")
        logger.info("=" * 80)
        
        return result
    
    def save_questions(
        self,
        result: QuestionGenerationResult,
        output_path: Path
    ):
        """Save questions to JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved {result.total_questions} questions to: {output_path}")


# ============================================================================
# TEST FUNCTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        phase4_evidence_path = sys.argv[1]
        audio_analysis_path = sys.argv[2]
        
        with open(phase4_evidence_path, 'r') as f:
            phase4_evidence = json.load(f)
        
        with open(audio_analysis_path, 'r') as f:
            audio_analysis = json.load(f)
        
        generator = MultimodalQuestionGeneratorV2()
        
        result = generator.generate_questions(
            phase4_evidence,
            audio_analysis,
            video_id="test"
        )
        
        output_path = Path(phase4_evidence_path).parent / "questions_v2_complete.json"
        generator.save_questions(result, output_path)
        
        print(f"\n✅ Generated {result.validated_questions}/{result.total_questions} valid questions")
        print(f"Saved to: {output_path}")
    else:
        print("Usage: python multimodal_question_generator_v2.py <phase4_evidence.json> <audio_analysis.json>")
        print("\nExample:")
        print("  python multimodal_question_generator_v2.py outputs/video_evidence.json outputs/video_audio_analysis.json")