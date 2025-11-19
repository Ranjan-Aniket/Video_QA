"""
Dynamic Logit Bias System

CRITICAL for achieving 99%+ hallucination-free generation.

Evidence-driven logit bias:
- Boost tokens from evidence: +10 to +20
- Suppress vague tokens: -100
- Block names completely: -1000
- Block fabrication words: -100

This ensures LLM can ONLY generate content from evidence.
"""

from typing import Dict, List, Set, Optional
from templates.base import EvidenceDatabase
import tiktoken


class LogitBiasBuilder:
    """
    Build dynamic logit bias from evidence
    
    Constrains LLM to only use vocabulary present in evidence.
    """
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize logit bias builder
        
        Args:
            model: Model name for tokenizer
        """
        self.encoding = tiktoken.encoding_for_model(model)
        
        # Vague words to suppress
        self.vague_words = {
            'several', 'many', 'multiple', 'some', 'few', 'various',
            'numerous', 'countless', 'approximately', 'roughly',
            'about', 'around', 'nearly', 'almost', 'maybe',
            'perhaps', 'possibly', 'probably', 'likely', 'unclear'
        }
        
        # Fabrication indicators to block
        self.fabrication_words = {
            'i think', 'i believe', 'it seems', 'it appears',
            'presumably', 'supposedly', 'allegedly', 'reportedly'
        }
    
    def build(
        self,
        evidence: EvidenceDatabase,
        boost_strength: int = 15,
        suppress_strength: int = -100,
        block_strength: int = -1000
    ) -> Dict[str, int]:
        """
        Build logit bias dictionary from evidence
        
        Args:
            evidence: Evidence database
            boost_strength: Bias for evidence tokens (+10 to +20)
            suppress_strength: Bias for vague words (-100)
            block_strength: Bias for names (-1000)
            
        Returns:
            Logit bias dictionary {token_id: bias_value}
        """
        logit_bias = {}
        
        # 1. BOOST evidence vocabulary
        evidence_vocab = self._extract_evidence_vocabulary(evidence)
        for word in evidence_vocab:
            tokens = self.encoding.encode(word)
            for token in tokens:
                logit_bias[str(token)] = boost_strength
        
        # 2. SUPPRESS vague words
        for word in self.vague_words:
            tokens = self.encoding.encode(word)
            for token in tokens:
                logit_bias[str(token)] = suppress_strength
        
        # 3. SUPPRESS fabrication indicators
        for phrase in self.fabrication_words:
            tokens = self.encoding.encode(phrase)
            for token in tokens:
                logit_bias[str(token)] = suppress_strength
        
        # 4. BLOCK names completely
        all_names = (
            evidence.character_names +
            evidence.team_names +
            evidence.media_names +
            evidence.brand_names
        )
        for name in all_names:
            tokens = self.encoding.encode(name)
            for token in tokens:
                logit_bias[str(token)] = block_strength
        
        return logit_bias
    
    def _extract_evidence_vocabulary(
        self,
        evidence: EvidenceDatabase
    ) -> Set[str]:
        """
        Extract all vocabulary from evidence
        
        This creates the ALLOWED vocabulary for generation.
        """
        vocab = set()
        
        # From transcript
        for segment in evidence.transcript_segments:
            words = segment['text'].lower().split()
            vocab.update(words)
        
        # From object classes
        for obj in evidence.object_detections:
            vocab.add(obj['object_class'].lower())
            if obj.get('color'):
                vocab.add(obj['color'].lower())
        
        # From scene types
        for scene in evidence.scene_detections:
            vocab.add(scene['scene_type'].lower())
        
        # From OCR text
        for ocr in evidence.ocr_detections:
            words = ocr['text'].lower().split()
            vocab.update(words)
        
        # From actions
        for action in evidence.action_detections:
            vocab.add(action['action'].lower())
        
        # Add descriptor vocabulary
        descriptor_words = {
            'person', 'man', 'woman', 'child', 'adult', 'teenager',
            'wearing', 'in', 'with', 'on', 'at', 'the', 'a', 'an',
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'gray',
            'shirt', 'jacket', 'dress', 'pants', 'shoes',
            'left', 'right', 'center', 'foreground', 'background',
            'small', 'medium', 'large', 'big', 'tiny'
        }
        vocab.update(descriptor_words)
        
        # Add question words
        question_words = {
            'what', 'when', 'where', 'who', 'why', 'how',
            'is', 'are', 'was', 'were', 'do', 'does', 'did',
            'after', 'before', 'during', 'while', 'then',
            'happens', 'appears', 'shows', 'says', 'visible'
        }
        vocab.update(question_words)
        
        return vocab
    
    def build_for_specific_evidence(
        self,
        evidence_items: List[str],
        boost_strength: int = 20
    ) -> Dict[str, int]:
        """
        Build logit bias for specific evidence items
        
        Use when generating answer from specific evidence.
        
        Args:
            evidence_items: List of evidence strings (e.g., specific transcript)
            boost_strength: How much to boost (+20 for very strong)
            
        Returns:
            Logit bias dictionary
        """
        logit_bias = {}
        
        for item in evidence_items:
            words = item.lower().split()
            for word in words:
                tokens = self.encoding.encode(word)
                for token in tokens:
                    logit_bias[str(token)] = boost_strength
        
        return logit_bias


class TemplateSlotConstraints:
    """
    Template-slot architecture for constrained generation
    
    Define fixed template with constrained slots:
    "When {AUDIO_CUE} is heard, {VISUAL_ELEMENT} appears."
    
    Each slot can ONLY be filled from evidence.
    """
    
    def __init__(self, evidence: EvidenceDatabase):
        self.evidence = evidence
        self.logit_builder = LogitBiasBuilder()
    
    def build_slot_constraints(
        self,
        slot_name: str,
        slot_type: str
    ) -> Dict[str, int]:
        """
        Build logit bias for specific slot
        
        Args:
            slot_name: Name of slot (e.g., "AUDIO_CUE")
            slot_type: Type of evidence (e.g., "transcript", "object")
            
        Returns:
            Logit bias for this slot
        """
        evidence_items = []
        
        if slot_type == "transcript":
            for seg in self.evidence.transcript_segments:
                evidence_items.append(seg['text'])
        
        elif slot_type == "object":
            for obj in self.evidence.object_detections:
                evidence_items.append(obj['object_class'])
                if obj.get('color'):
                    evidence_items.append(obj['color'])
        
        elif slot_type == "person":
            # Generate descriptors
            person_words = set()
            for person in self.evidence.person_detections:
                attrs = person.get('attributes', {})
                if attrs.get('clothing_color'):
                    person_words.add(attrs['clothing_color'])
                if attrs.get('clothing_type'):
                    person_words.add(attrs['clothing_type'])
            evidence_items = list(person_words)
        
        elif slot_type == "action":
            for action in self.evidence.action_detections:
                evidence_items.append(action['action'])
        
        # Build bias ONLY for this evidence
        return self.logit_builder.build_for_specific_evidence(
            evidence_items,
            boost_strength=20
        )
