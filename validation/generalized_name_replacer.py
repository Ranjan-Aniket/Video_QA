"""
Generalized Name Replacer - Domain-Agnostic Name → Descriptor Replacement

Orchestrates:
1. Dynamic name detection (spaCy NER)
2. Dynamic descriptor extraction (Claude)
3. Text replacement

Works for ANY domain: sports, cooking, interviews, nature, tutorials, etc.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from .dynamic_name_detector import DynamicNameDetector, DetectedEntity
from .dynamic_descriptor_extractor import DynamicDescriptorExtractor, DescriptorResult

logger = logging.getLogger(__name__)


@dataclass
class ReplacementResult:
    """Result of name replacement operation"""
    original: str
    cleaned: str
    replacements: List[Dict]
    entities_found: int
    has_names: bool


class GeneralizedNameReplacer:
    """
    FULLY GENERALIZED name replacer.
    Works for ANY video domain with ZERO hardcoding.
    """
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        """
        Initialize name replacer.
        
        Args:
            anthropic_api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
        """
        self.detector = DynamicNameDetector()
        self.extractor = DynamicDescriptorExtractor(anthropic_api_key)
        
        logger.info("GeneralizedNameReplacer initialized")
    
    def replace_names(
        self,
        text: str,
        evidence: Dict
    ) -> ReplacementResult:
        """
        Replace ALL names/pronouns with visual descriptors.
        
        Works for:
        - Sports (jersey numbers)
        - Cooking (chef attire)
        - Interviews (clothing)
        - Nature (animal features)
        - ANY domain
        
        Args:
            text: Text to process
            evidence: Evidence dictionary from frame analysis
            
        Returns:
            ReplacementResult with cleaned text and replacement details
        """
        # Step 1: Detect entities dynamically using NER
        entities = self.detector.detect_all_entities(text)
        
        if not entities:
            # No names/pronouns found
            return ReplacementResult(
                original=text,
                cleaned=text,
                replacements=[],
                entities_found=0,
                has_names=False
            )
        
        logger.info(f"Found {len(entities)} entities in text")
        
        # Step 2: Extract descriptors from evidence using Claude
        descriptors = self.extractor.extract_batch(entities, evidence)
        
        # Step 3: Build replacement list
        replacements = []
        for entity in entities:
            descriptor_result = descriptors.get(entity.text)
            if descriptor_result:
                replacements.append({
                    'original': entity.text,
                    'replacement': descriptor_result.descriptor,
                    'type': entity.type,
                    'confidence': descriptor_result.confidence,
                    'source': descriptor_result.source
                })
        
        # Step 4: Apply replacements to text
        clean_text = self._apply_replacements(text, replacements)
        
        return ReplacementResult(
            original=text,
            cleaned=clean_text,
            replacements=replacements,
            entities_found=len(entities),
            has_names=True
        )
    
    def _apply_replacements(
        self,
        text: str,
        replacements: List[Dict]
    ) -> str:
        """
        Apply replacements to text.
        
        Strategy: Sort by position (reverse) to preserve indices.
        """
        result = text
        
        # Sort by original text length (longest first)
        # This prevents partial replacements
        sorted_repls = sorted(
            replacements,
            key=lambda x: len(x['original']),
            reverse=True
        )
        
        for repl in sorted_repls:
            original = repl['original']
            replacement = repl['replacement']
            
            # Case-sensitive replacement (first occurrence)
            if original in result:
                result = result.replace(original, replacement, 1)
            # Try case-insensitive if exact match not found
            elif original.lower() in result.lower():
                # Find and replace case-insensitively
                import re
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                result = pattern.sub(replacement, result, count=1)
        
        return result
    
    def has_names(self, text: str) -> bool:
        """Quick check if text contains names/pronouns"""
        return self.detector.has_names_or_pronouns(text)
    
    def get_entity_summary(self, text: str) -> Dict[str, int]:
        """Get count of each entity type in text"""
        return self.detector.get_entity_summary(text)


def replace_names_in_text(
    text: str,
    evidence: Dict,
    anthropic_api_key: Optional[str] = None
) -> str:
    """
    Convenience function for quick name replacement.
    
    Args:
        text: Text to process
        evidence: Evidence dictionary
        anthropic_api_key: Optional API key
        
    Returns:
        Cleaned text with names replaced
    """
    replacer = GeneralizedNameReplacer(anthropic_api_key)
    result = replacer.replace_names(text, evidence)
    return result.cleaned


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example evidence
    evidence = {
        'gpt4v_description': "LeBron James, tall man in purple jersey #23, passes to teammate in yellow jersey #3",
        'claude_description': "Chef Gordon Ramsay in white apron adds ingredients"
    }
    
    # Test cases
    test_cases = [
        "LeBron James passes to Lakers teammate. He scores!",
        "Gordon Ramsay adds olive oil. He stirs the pan.",
        "When Elon Musk speaks, they listen carefully."
    ]
    
    replacer = GeneralizedNameReplacer()
    
    for text in test_cases:
        print(f"\nOriginal: {text}")
        result = replacer.replace_names(text, evidence)
        print(f"Cleaned:  {result.cleaned}")
        print(f"Replaced: {result.entities_found} entities")
        for repl in result.replacements:
            print(f"  '{repl['original']}' → '{repl['replacement']}' ({repl['type']})")
