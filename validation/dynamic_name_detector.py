"""
Dynamic Name Detector - NER-Based Name Detection

Uses spaCy Named Entity Recognition to detect names dynamically.
NO HARDCODED LISTS - works for ANY domain.

Detects:
- PERSON (any person name)
- ORG (organizations, teams, companies)
- PRODUCT (brands, products)
- WORK_OF_ART (movies, books, songs)
- Pronouns (he, she, they, him, her)
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_trf")


@dataclass
class DetectedEntity:
    """Detected name/entity"""
    text: str
    type: str  # PERSON, ORG, PRODUCT, WORK_OF_ART, PRONOUN
    start: int
    end: int
    confidence: float = 1.0


class DynamicNameDetector:
    """
    Detect names using NER instead of hardcoded lists.
    Works for ANY domain (sports, cooking, interviews, etc.)
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize with spaCy model.
        
        Args:
            model_name: spaCy model ('en_core_web_sm', 'en_core_web_md', 'en_core_web_trf')
                       'trf' = transformer-based (best accuracy)
                       'sm'/'md' = smaller/faster models
        """
        if not SPACY_AVAILABLE:
            raise ImportError(
                "spaCy is required for DynamicNameDetector. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )
        
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            # Model not found, try to download
            logger.warning(f"Model {model_name} not found. Trying 'en_core_web_sm'...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded fallback model: en_core_web_sm")
            except OSError:
                raise ImportError(
                    f"spaCy model not found. Download with: python -m spacy download {model_name}"
                )
        
        # Pronoun list for additional detection
        self.pronouns = {
            'he', 'she', 'they', 'him', 'her', 'his', 'hers', 'their', 'theirs',
            'himself', 'herself', 'themselves', 'them'
        }
    
    def detect_all_entities(self, text: str) -> List[DetectedEntity]:
        """
        Detect ALL named entities dynamically.
        
        Returns entities of type:
        - PERSON (any person name)
        - ORG (any organization, team, company)
        - PRODUCT (any product, brand)
        - WORK_OF_ART (movie, book, song)
        - PRONOUN (he, she, they, etc.)
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected entities
        """
        doc = self.nlp(text)
        
        entities = []
        
        # 1. Named entities from spaCy
        for ent in doc.ents:
            # Only care about entity types that are names
            if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'WORK_OF_ART', 'GPE']:
                entities.append(DetectedEntity(
                    text=ent.text,
                    type=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.9  # High confidence for NER detections
                ))
        
        # 2. Pronouns (explicit detection)
        pronoun_entities = self._detect_pronouns(doc)
        entities.extend(pronoun_entities)
        
        logger.debug(f"Detected {len(entities)} entities in text")
        return entities
    
    def _detect_pronouns(self, doc) -> List[DetectedEntity]:
        """Detect pronouns in text"""
        pronouns = []
        
        for token in doc:
            # Check POS tag for pronoun OR explicit list
            is_pronoun = (
                token.pos_ == 'PRON' and 
                token.text.lower() in self.pronouns
            )
            
            if is_pronoun:
                pronouns.append(DetectedEntity(
                    text=token.text,
                    type='PRONOUN',
                    start=token.idx,
                    end=token.idx + len(token.text),
                    confidence=1.0
                ))
        
        return pronouns
    
    def has_names_or_pronouns(self, text: str) -> bool:
        """Quick check if text contains any names or pronouns"""
        entities = self.detect_all_entities(text)
        return len(entities) > 0
    
    def get_entity_summary(self, text: str) -> Dict[str, int]:
        """Get count of each entity type"""
        entities = self.detect_all_entities(text)
        summary = {
            'PERSON': 0,
            'ORG': 0,
            'PRODUCT': 0,
            'WORK_OF_ART': 0,
            'GPE': 0,
            'PRONOUN': 0
        }
        
        for entity in entities:
            if entity.type in summary:
                summary[entity.type] += 1
        
        return summary


def detect_names_quick(text: str) -> List[str]:
    """
    Convenience function for quick name detection.
    Returns list of detected name strings.
    """
    detector = DynamicNameDetector()
    entities = detector.detect_all_entities(text)
    return [entity.text for entity in entities]


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test examples across domains
    test_cases = [
        # Sports
        "LeBron James passes to Lakers teammate",
        
        # Cooking
        "Gordon Ramsay adds olive oil to the pan",
        
        # Interview
        "Elon Musk discusses Tesla innovation",
        
        # Nature
        "The African lion stalks the gazelle",
        
        # Pronouns
        "He picks up the ball and she passes it to them"
    ]
    
    detector = DynamicNameDetector()
    
    for text in test_cases:
        print(f"\nText: {text}")
        entities = detector.detect_all_entities(text)
        for entity in entities:
            print(f"  {entity.type}: '{entity.text}'")
        print(f"  Summary: {detector.get_entity_summary(text)}")
