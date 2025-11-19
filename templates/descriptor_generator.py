"""
Descriptor Generator - Generate descriptors instead of names

CRITICAL GUIDELINE COMPLIANCE:
- "Never use any names in prompt or responses related to the video"
- "Always qualify it with a character wearing an orange shirt, main character, 
  female lead, white puppy etc."
- "Must avoid names across the board including but not limited to sports teams, 
  company/band, movies/books/songs"
- "Avoid using he/she in the question. Always qualify the character with more 
  description for clarity: a man in black jacket, woman with white sports shoes, 
  main female lead etc."

This module generates evidence-based descriptors for:
- People (based on clothing, position, actions)
- Objects (based on color, size, position)
- Teams (based on jersey color, not names)
- Locations (based on visual features, not names)
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import random


class DescriptorType(Enum):
    """Types of descriptors we can generate"""
    PERSON = "person"
    OBJECT = "object"
    TEAM = "team"
    LOCATION = "location"
    ANIMAL = "animal"


@dataclass
class VisualEvidence:
    """Evidence about a visual element from multimodal context"""
    timestamp: float
    bbox: Tuple[float, float, float, float]  # x, y, w, h normalized
    
    # Person attributes
    clothing_color: Optional[str] = None
    clothing_type: Optional[str] = None  # shirt, jacket, dress, etc.
    gender: Optional[str] = None  # male/female/person if unclear
    age_group: Optional[str] = None  # child, teenager, adult, elderly
    hair_color: Optional[str] = None
    accessories: List[str] = None  # glasses, hat, etc.
    action: Optional[str] = None  # standing, sitting, running, etc.
    
    # Object attributes
    object_class: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None  # small, medium, large
    material: Optional[str] = None
    
    # Team attributes
    jersey_color: Optional[str] = None
    jersey_number: Optional[int] = None
    
    # Location attributes
    location_type: Optional[str] = None  # kitchen, park, stadium, etc.
    
    # Position in frame
    position: Optional[str] = None  # left, right, center, foreground, background
    
    def __post_init__(self):
        if self.accessories is None:
            self.accessories = []


class DescriptorGenerator:
    """
    Generates descriptors from visual evidence
    
    NO HARDCODING - all descriptors built from actual evidence
    """
    
    # Common clothing colors (for validation, not generation)
    VALID_COLORS = {
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
        'black', 'white', 'gray', 'grey', 'brown', 'beige', 'navy',
        'maroon', 'teal', 'cyan', 'magenta', 'lime', 'olive'
    }
    
    # Position descriptors
    POSITION_MAP = {
        (0.0, 0.33): "left",
        (0.33, 0.66): "center",
        (0.66, 1.0): "right"
    }
    
    @staticmethod
    def generate_person_descriptor(evidence: VisualEvidence) -> str:
        """
        Generate person descriptor from evidence
        
        Priority order:
        1. Gender + clothing (most specific)
        2. Gender + action
        3. Gender + position
        4. Just "person" if nothing else
        
        Examples:
        - "man in the blue jacket"
        - "woman wearing red shirt"
        - "person in the striped dress"
        - "child with glasses"
        """
        parts = []
        
        # Start with age group or gender
        if evidence.age_group:
            parts.append(evidence.age_group)
        elif evidence.gender and evidence.gender != "person":
            parts.append(evidence.gender)
        else:
            parts.append("person")
        
        # Add most distinctive feature
        if evidence.clothing_color and evidence.clothing_type:
            if evidence.accessories:
                # "man in blue jacket with glasses"
                descriptor = f"{parts[0]} in {evidence.clothing_color} {evidence.clothing_type} with {', '.join(evidence.accessories)}"
            else:
                # "woman wearing red shirt"
                descriptor = f"{parts[0]} wearing {evidence.clothing_color} {evidence.clothing_type}"
        elif evidence.clothing_color:
            descriptor = f"{parts[0]} in {evidence.clothing_color}"
        elif evidence.action:
            descriptor = f"{parts[0]} {evidence.action}"
        elif evidence.position:
            descriptor = f"{parts[0]} on the {evidence.position}"
        else:
            descriptor = parts[0]
        
        return descriptor
    
    @staticmethod
    def generate_object_descriptor(evidence: VisualEvidence) -> str:
        """
        Generate object descriptor from evidence
        
        Examples:
        - "small red ball"
        - "large wooden table"
        - "metal pitcher"
        """
        parts = []
        
        if evidence.size:
            parts.append(evidence.size)
        if evidence.color:
            parts.append(evidence.color)
        if evidence.material:
            parts.append(evidence.material)
        if evidence.object_class:
            parts.append(evidence.object_class)
        
        return " ".join(parts) if parts else "object"
    
    @staticmethod
    def generate_team_descriptor(evidence: VisualEvidence) -> str:
        """
        Generate team descriptor from evidence (NO TEAM NAMES)
        
        Examples:
        - "team in white jerseys"
        - "players wearing blue"
        - "home team" / "away team" (if determinable from evidence)
        """
        if evidence.jersey_color:
            return f"team in {evidence.jersey_color} jerseys"
        elif evidence.position:
            # If we can determine home/away from position
            return f"{evidence.position} team"
        else:
            return "team"
    
    @staticmethod
    def generate_player_descriptor(evidence: VisualEvidence) -> str:
        """
        Generate player descriptor (jersey color + number, NO NAMES)
        
        Examples:
        - "player number 23 in red jersey"
        - "player wearing blue number 10"
        """
        if evidence.jersey_number and evidence.jersey_color:
            return f"player number {evidence.jersey_number} in {evidence.jersey_color} jersey"
        elif evidence.jersey_number:
            return f"player number {evidence.jersey_number}"
        elif evidence.jersey_color:
            return f"player in {evidence.jersey_color} jersey"
        else:
            return "player"
    
    @staticmethod
    def generate_location_descriptor(evidence: VisualEvidence) -> str:
        """
        Generate location descriptor from evidence
        
        Examples:
        - "outdoor basketball court"
        - "inside a warehouse"
        - "park with trees in background"
        """
        if evidence.location_type:
            return evidence.location_type
        else:
            return "location"
    
    @staticmethod
    def generate_animal_descriptor(evidence: VisualEvidence) -> str:
        """
        Generate animal descriptor from evidence
        
        Examples:
        - "small white puppy"
        - "large brown dog"
        """
        parts = []
        
        if evidence.size:
            parts.append(evidence.size)
        if evidence.color:
            parts.append(evidence.color)
        if evidence.object_class:
            parts.append(evidence.object_class)
        
        return " ".join(parts) if parts else "animal"
    
    @staticmethod
    def get_position_descriptor(bbox: Tuple[float, float, float, float]) -> str:
        """
        Get position descriptor from bounding box
        
        Args:
            bbox: (x, y, w, h) normalized coordinates
            
        Returns:
            Position descriptor: "left", "center", "right", "foreground", "background"
        """
        x, y, w, h = bbox
        center_x = x + w / 2
        
        # Horizontal position
        for (min_x, max_x), pos in DescriptorGenerator.POSITION_MAP.items():
            if min_x <= center_x < max_x:
                return pos
        
        return "center"
    
    @staticmethod
    def disambiguate_people(people: List[VisualEvidence]) -> List[str]:
        """
        Generate unique descriptors for multiple people
        
        If multiple people have same descriptor, add position or other features
        
        Args:
            people: List of person evidence
            
        Returns:
            List of unique descriptors
        """
        descriptors = []
        used_descriptors = set()
        
        for person in people:
            base_descriptor = DescriptorGenerator.generate_person_descriptor(person)
            
            # If descriptor already used, make it unique
            if base_descriptor in used_descriptors:
                # Add position
                if person.position:
                    descriptor = f"{base_descriptor} on the {person.position}"
                else:
                    # Add position from bbox
                    position = DescriptorGenerator.get_position_descriptor(person.bbox)
                    descriptor = f"{base_descriptor} on the {position}"
            else:
                descriptor = base_descriptor
            
            descriptors.append(descriptor)
            used_descriptors.add(descriptor)
        
        return descriptors


class DescriptorTracker:
    """
    Track descriptors used in a video to ensure consistency
    
    If we refer to "man in blue jacket" at 1:30, we should use same descriptor
    at 2:00 if it's the same person
    """
    
    def __init__(self):
        self.person_descriptors: Dict[str, str] = {}  # person_id -> descriptor
        self.object_descriptors: Dict[str, str] = {}  # object_id -> descriptor
        self.team_descriptors: Dict[str, str] = {}    # team_id -> descriptor
    
    def get_or_create_person_descriptor(
        self, 
        person_id: str, 
        evidence: VisualEvidence
    ) -> str:
        """
        Get existing descriptor for person or create new one
        
        Ensures consistency across video
        """
        if person_id in self.person_descriptors:
            return self.person_descriptors[person_id]
        
        descriptor = DescriptorGenerator.generate_person_descriptor(evidence)
        self.person_descriptors[person_id] = descriptor
        return descriptor
    
    def get_or_create_object_descriptor(
        self, 
        object_id: str, 
        evidence: VisualEvidence
    ) -> str:
        """Get or create object descriptor"""
        if object_id in self.object_descriptors:
            return self.object_descriptors[object_id]
        
        descriptor = DescriptorGenerator.generate_object_descriptor(evidence)
        self.object_descriptors[object_id] = descriptor
        return descriptor
    
    def get_or_create_team_descriptor(
        self, 
        team_id: str, 
        evidence: VisualEvidence
    ) -> str:
        """Get or create team descriptor"""
        if team_id in self.team_descriptors:
            return self.team_descriptors[team_id]
        
        descriptor = DescriptorGenerator.generate_team_descriptor(evidence)
        self.team_descriptors[team_id] = descriptor
        return descriptor


def validate_no_names(text: str, video_context: Dict) -> bool:
    """
    Validate that text contains no names
    
    Checks against:
    - Character names from video metadata
    - Sports team names
    - Movie/show names
    - Song names
    - Company names
    
    Args:
        text: Text to validate
        video_context: Video metadata with detected names
        
    Returns:
        True if no names found, False otherwise
    """
    text_lower = text.lower()
    
    # Check character names
    if 'character_names' in video_context:
        for name in video_context['character_names']:
            if name.lower() in text_lower:
                return False
    
    # Check team names
    if 'team_names' in video_context:
        for name in video_context['team_names']:
            if name.lower() in text_lower:
                return False
    
    # Check movie/show names
    if 'media_names' in video_context:
        for name in video_context['media_names']:
            if name.lower() in text_lower:
                return False
    
    # Check song names
    if 'song_names' in video_context:
        for name in video_context['song_names']:
            if name.lower() in text_lower:
                return False
    
    # Check company/brand names
    if 'brand_names' in video_context:
        for name in video_context['brand_names']:
            if name.lower() in text_lower:
                return False
    
    return True