"""
Dynamic Descriptor Extractor - Claude-Powered Descriptor Generation

Extracts visual descriptors from AI descriptions (GPT-4V, Claude).
NO HARDCODED ATTRIBUTES - extracts from actual evidence.

Examples:
- "LeBron James" + GPT-4V description → "tall man in purple jersey #23"
- "Gordon Ramsay" + Claude description → "chef in white apron"
- "Elon Musk" + evidence → "man in black turtleneck"
"""

import os
import logging
import re
from typing import Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    logger.warning("Anthropic SDK not available. Install with: pip install anthropic")


@dataclass
class DescriptorResult:
    """Result of descriptor extraction"""
    descriptor: str
    confidence: float
    source: str  # "gpt4v", "claude", "fallback"
    original_text: Optional[str] = None


class DynamicDescriptorExtractor:
    """
    Extract visual descriptors from AI descriptions.
    Works for ANY entity in ANY video type.
    """
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        """
        Initialize with Claude API key.
        
        Args:
            anthropic_api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
        """
        if not CLAUDE_AVAILABLE:
            raise ImportError(
                "Anthropic SDK required. Install with: pip install anthropic"
            )
        
        # Get API key
        api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass anthropic_api_key parameter."
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-5-20250929"
    
    def extract_descriptor(
        self,
        entity_name: str,
        entity_type: str,
        gpt4v_description: str = "",
        claude_description: str = ""
    ) -> DescriptorResult:
        """
        Extract visual descriptor for entity from AI descriptions.
        
        Args:
            entity_name: Name to replace (e.g., "LeBron James")
            entity_type: Entity type (PERSON, ORG, etc.)
            gpt4v_description: GPT-4V visual description
            claude_description: Claude visual description
            
        Returns:
            DescriptorResult with extracted descriptor
        """
        # Combine AI descriptions
        full_text = f"{gpt4v_description}\n{claude_description}".strip()
        
        if not full_text:
            # No AI descriptions available, use generic fallback
            return self._generate_generic_descriptor(entity_type)
        
        # Try to find descriptor in AI text
        descriptor = self._find_descriptor_in_ai_text(
            entity_name,
            full_text
        )
        
        if descriptor:
            return DescriptorResult(
                descriptor=descriptor,
                confidence=0.9,
                source="ai_extraction",
                original_text=full_text[:200]
            )
        
        # Fallback to generic descriptor
        return self._generate_generic_descriptor(entity_type)
    
    def _find_descriptor_in_ai_text(
        self,
        entity_name: str,
        ai_text: str
    ) -> Optional[str]:
        """
        Use Claude to extract visual descriptor from AI descriptions.
        
        This is the KEY method that makes the system generalized!
        """
        prompt = f"""Extract the VISUAL DESCRIPTOR for "{entity_name}" from this description.

Description:
{ai_text[:1000]}  

Return ONLY the visual descriptor (appearance, clothing, role), NOT the name.

Examples:
- "LeBron James in purple jersey #23 dribbling" → "the tall man in purple jersey #23"
- "Chef Gordon preparing dish" → "the chef in white apron"
- "Elon Musk wearing black shirt speaking" → "the man in black turtleneck"
- "Woman with glasses using laptop" → "the woman with glasses"

Rules:
1. Start with "the" (e.g., "the man", "the woman", "the person")
2. Include visible attributes (clothing color, numbers, objects)
3. Do NOT include the name
4. Be specific and concise (5-10 words max)

Descriptor for "{entity_name}":"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            descriptor = response.content[0].text.strip()
            
            # Validate: Must not contain the entity name
            if entity_name.lower() in descriptor.lower():
                logger.warning(f"Descriptor contains name: {descriptor}")
                return None
            
            # Validate: Must start with article
            if not descriptor.lower().startswith(('the ', 'a ', 'an ')):
                descriptor = f"the {descriptor}"
            
            logger.debug(f"Extracted descriptor: '{entity_name}' → '{descriptor}'")
            return descriptor
            
        except Exception as e:
            logger.error(f"Error extracting descriptor: {e}")
            return None
    
    def _generate_generic_descriptor(self, entity_type: str) -> DescriptorResult:
        """Fallback generic descriptor when AI extraction fails"""
        mapping = {
            'PERSON': 'the person',
            'ORG': 'the organization',
            'PRODUCT': 'the product',
            'WORK_OF_ART': 'the content',
            'GPE': 'the location',
            'PRONOUN': 'the person'  # Pronouns also map to person
        }
        
        descriptor = mapping.get(entity_type, 'the entity')
        
        return DescriptorResult(
            descriptor=descriptor,
            confidence=0.5,  # Low confidence for generic
            source="fallback"
        )
    
    def extract_batch(
        self,
        entities: list,
        evidence: Dict
    ) -> Dict[str, DescriptorResult]:
        """
        Extract descriptors for multiple entities at once.
        
        Args:
            entities: List of entities to process
            evidence: Evidence dictionary with AI descriptions
            
        Returns:
            Dictionary mapping entity text to DescriptorResult
        """
        descriptors = {}
        
        # Extract AI descriptions from evidence
        gpt4v = evidence.get('gpt4v_description', '')
        claude_desc = evidence.get('claude_description', '')
        
        # Convert to string if dict
        if isinstance(gpt4v, dict):
            gpt4v = str(gpt4v)
        if isinstance(claude_desc, dict):
            claude_desc = str(claude_desc)
        
        for entity in entities:
            result = self.extract_descriptor(
                entity_name=entity.text,
                entity_type=entity.type,
                gpt4v_description=gpt4v,
                claude_description=claude_desc
            )
            descriptors[entity.text] = result
        
        return descriptors


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test extractor
    extractor = DynamicDescriptorExtractor()
    
    # Test case 1: Sports
    result = extractor.extract_descriptor(
        entity_name="LeBron James",
        entity_type="PERSON",
        gpt4v_description="LeBron James, the tall man in purple Lakers jersey #23, passes the ball to teammate",
        claude_description=""
    )
    print(f"Sports: {result.descriptor} (confidence: {result.confidence})")
    
    # Test case 2: Cooking
    result = extractor.extract_descriptor(
        entity_name="Gordon Ramsay",
        entity_type="PERSON",
        gpt4v_description="",
        claude_description="Chef Gordon Ramsay, wearing a white chef's apron and black shirt, adds olive oil to pan"
    )
    print(f"Cooking: {result.descriptor} (confidence: {result.confidence})")
    
    # Test case 3: No description (fallback)
    result = extractor.extract_descriptor(
        entity_name="Unknown Person",
        entity_type="PERSON",
        gpt4v_description="",
        claude_description=""
    )
    print(f"Fallback: {result.descriptor} (confidence: {result.confidence})")
