"""
Extended Name Blocker - Enforces Name-Free Generation

CRITICAL GUIDELINE FROM DOCUMENT:
"Never use any names in prompt or responses related to the video.
Always qualify it with a character wearing an orange shirt, main character,
female lead, white puppy etc."

"Must avoid names across the board including but not limited to:
sports teams, company/band, movies/books/songs"

This module blocks ALL name types and enforces descriptor usage.
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class NameDetectionResult:
    """Result of name detection"""
    has_names: bool
    detected_names: List[Dict[str, str]]  # [{"type": "person", "name": "John", "context": "..."}]
    suggested_replacements: List[Dict[str, str]]  # Descriptor suggestions
    confidence: float  # 0.0 to 1.0


class ExtendedNameBlocker:
    """
    Extended name blocker that blocks ALL name categories:
    1. Character/Person names
    2. Sports team names
    3. Company/Band names
    4. Movie/Book/Song names
    5. Location names (when referring to people/things from that location)
    
    Enforces descriptor usage per Guidelines.
    """
    
    def __init__(self):
        """Initialize with comprehensive name databases"""
        self._load_name_databases()
        self._compile_patterns()
    
    def _load_name_databases(self):
        """Load comprehensive name databases"""
        
        # Common first names (both male and female)
        self.first_names = {
            # Male names (sample - production should have comprehensive list)
            'john', 'james', 'robert', 'michael', 'william', 'david', 'richard',
            'joseph', 'thomas', 'charles', 'daniel', 'matthew', 'anthony', 'mark',
            'donald', 'steven', 'paul', 'andrew', 'joshua', 'kenneth', 'kevin',
            'brian', 'george', 'edward', 'ronald', 'timothy', 'jason', 'jeffrey',
            'ryan', 'jacob', 'gary', 'nicholas', 'eric', 'jonathan', 'stephen',
            
            # Female names (sample - production should have comprehensive list)
            'mary', 'patricia', 'jennifer', 'linda', 'barbara', 'elizabeth',
            'susan', 'jessica', 'sarah', 'karen', 'nancy', 'lisa', 'betty',
            'margaret', 'sandra', 'ashley', 'kimberly', 'emily', 'donna',
            'michelle', 'dorothy', 'carol', 'amanda', 'melissa', 'deborah',
            'stephanie', 'rebecca', 'sharon', 'laura', 'cynthia', 'kathleen',
            
            # International names (sample)
            'jose', 'maria', 'juan', 'carlos', 'luis', 'ana', 'wei', 'chen',
            'kumar', 'singh', 'ahmed', 'mohammed', 'ali', 'yuki', 'akira',
        }
        
        # Sports team names (comprehensive sample)
        self.sports_teams = {
            # NBA
            'lakers', 'celtics', 'warriors', 'bulls', 'heat', 'nets', 'knicks',
            'sixers', 'raptors', 'bucks', 'clippers', 'mavericks', 'nuggets',
            
            # NFL
            'patriots', 'cowboys', 'packers', 'steelers', '49ers', 'chiefs',
            'eagles', 'ravens', 'seahawks', 'saints', 'broncos', 'raiders',
            
            # MLB
            'yankees', 'red sox', 'dodgers', 'cubs', 'astros', 'braves',
            
            # Soccer
            'barcelona', 'real madrid', 'manchester united', 'liverpool',
            'bayern munich', 'juventus', 'arsenal', 'chelsea',
            
            # College teams
            'crimson tide', 'buckeyes', 'wolverines', 'longhorns', 'sooners',
        }
        
        # Company/Band names (comprehensive sample)
        self.companies = {
            # Tech companies
            'apple', 'google', 'microsoft', 'amazon', 'meta', 'facebook',
            'tesla', 'twitter', 'netflix', 'uber', 'airbnb', 'spotify',
            
            # Other companies
            'nike', 'adidas', 'coca-cola', 'pepsi', 'mcdonalds', 'starbucks',
            'walmart', 'target', 'costco', 'bmw', 'toyota', 'ford',
            
            # Bands/Musicians
            'beatles', 'rolling stones', 'led zeppelin', 'pink floyd',
            'queen', 'metallica', 'nirvana', 'radiohead', 'coldplay',
            'beyonce', 'taylor swift', 'drake', 'kanye west', 'eminem',
        }
        
        # Movie/Book/Song titles (comprehensive sample)
        self.media_titles = {
            # Movies
            'titanic', 'avatar', 'inception', 'interstellar', 'godfather',
            'star wars', 'harry potter', 'lord of the rings', 'matrix',
            'pulp fiction', 'fight club', 'forrest gump', 'gladiator',
            
            # Books
            'moby dick', 'pride and prejudice', 'to kill a mockingbird',
            '1984', 'brave new world', 'catch-22', 'great gatsby',
            
            # Songs
            'bohemian rhapsody', 'imagine', 'smells like teen spirit',
            'billie jean', 'hey jude', 'stairway to heaven', 'hotel california',
        }
        
        # Character names from popular media
        self.character_names = {
            'harry', 'hermione', 'ron', 'frodo', 'gandalf', 'luke', 'leia',
            'darth vader', 'spiderman', 'batman', 'superman', 'iron man',
            'captain america', 'thor', 'hulk', 'wolverine', 'deadpool',
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for name detection"""
        
        # Capitalized word pattern (potential proper noun)
        self.capitalized_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
        
        # Possessive pattern (suggests name usage)
        self.possessive_pattern = re.compile(r'\b[A-Z][a-z]+\'s\b')
        
        # Title pattern (Mr., Mrs., Dr., etc.)
        self.title_pattern = re.compile(
            r'\b(Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Lord|Lady|Captain|Coach)\.?\s+[A-Z][a-z]+',
            re.IGNORECASE
        )
        
        # Descriptor patterns (what we WANT to see)
        self.descriptor_patterns = [
            r'character wearing',
            r'person (in|with|wearing)',
            r'(man|woman|boy|girl|child) (in|with|wearing)',
            r'main character',
            r'female lead',
            r'male lead',
            r'protagonist',
            r'player number \d+',
            r'(first|second|third) person',
        ]
    
    def detect_names(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> NameDetectionResult:
        """
        Detect ALL types of names in text.
        
        Args:
            text: Text to check for names
            context: Optional context (video metadata, transcript, etc.)
            
        Returns:
            NameDetectionResult with detected names and suggestions
        """
        detected_names = []
        suggested_replacements = []
        
        # 1. Check for character/person names
        person_names = self._detect_person_names(text)
        detected_names.extend(person_names)
        
        # 2. Check for sports team names
        team_names = self._detect_sports_teams(text)
        detected_names.extend(team_names)
        
        # 3. Check for company/band names
        company_names = self._detect_companies(text)
        detected_names.extend(company_names)
        
        # 4. Check for movie/book/song names
        media_names = self._detect_media_titles(text)
        detected_names.extend(media_names)
        
        # 5. Check for character names from media
        char_names = self._detect_character_names(text)
        detected_names.extend(char_names)
        
        # 6. Check for capitalized words (potential names)
        potential_names = self._detect_capitalized_words(text)
        detected_names.extend(potential_names)
        
        # Generate descriptor suggestions
        if detected_names and context:
            suggested_replacements = self._generate_descriptor_replacements(
                detected_names, context
            )
        
        # Calculate confidence
        confidence = self._calculate_confidence(detected_names, text)
        
        return NameDetectionResult(
            has_names=len(detected_names) > 0,
            detected_names=detected_names,
            suggested_replacements=suggested_replacements,
            confidence=confidence
        )
    
    def _detect_person_names(self, text: str) -> List[Dict]:
        """Detect person/character names"""
        detected = []
        words = text.split()
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            
            # Check against first name database
            if word_lower in self.first_names:
                detected.append({
                    'type': 'person_name',
                    'name': word,
                    'position': i,
                    'context': ' '.join(words[max(0, i-2):min(len(words), i+3)])
                })
        
        # Check for titles (Mr. Smith, Dr. Jones, etc.)
        title_matches = self.title_pattern.findall(text)
        for match in title_matches:
            detected.append({
                'type': 'titled_name',
                'name': match,
                'position': -1,
                'context': text
            })
        
        return detected
    
    def _detect_sports_teams(self, text: str) -> List[Dict]:
        """Detect sports team names"""
        detected = []
        text_lower = text.lower()
        
        for team in self.sports_teams:
            if team in text_lower:
                detected.append({
                    'type': 'sports_team',
                    'name': team,
                    'position': text_lower.index(team),
                    'context': text
                })
        
        return detected
    
    def _detect_companies(self, text: str) -> List[Dict]:
        """Detect company/band names"""
        detected = []
        text_lower = text.lower()
        
        for company in self.companies:
            if company in text_lower:
                detected.append({
                    'type': 'company_or_band',
                    'name': company,
                    'position': text_lower.index(company),
                    'context': text
                })
        
        return detected
    
    def _detect_media_titles(self, text: str) -> List[Dict]:
        """Detect movie/book/song titles"""
        detected = []
        text_lower = text.lower()
        
        for title in self.media_titles:
            if title in text_lower:
                detected.append({
                    'type': 'media_title',
                    'name': title,
                    'position': text_lower.index(title),
                    'context': text
                })
        
        return detected
    
    def _detect_character_names(self, text: str) -> List[Dict]:
        """Detect character names from popular media"""
        detected = []
        text_lower = text.lower()
        
        for char_name in self.character_names:
            if char_name in text_lower:
                detected.append({
                    'type': 'character_name',
                    'name': char_name,
                    'position': text_lower.index(char_name),
                    'context': text
                })
        
        return detected
    
    def _detect_capitalized_words(self, text: str) -> List[Dict]:
        """
        Detect capitalized words that might be names.
        
        This catches names not in our database.
        """
        detected = []
        
        # Find all capitalized words/phrases
        matches = self.capitalized_pattern.findall(text)
        
        for match in matches:
            # Skip common words that are capitalized (sentence start, etc.)
            if match.lower() in {'the', 'a', 'an', 'this', 'that', 'what', 'when', 'where', 'how', 'why'}:
                continue
            
            # Skip if it's a descriptor pattern
            is_descriptor = False
            for pattern in self.descriptor_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    is_descriptor = True
                    break
            
            if not is_descriptor:
                detected.append({
                    'type': 'potential_name',
                    'name': match,
                    'position': text.index(match),
                    'context': text
                })
        
        return detected
    
    def _generate_descriptor_replacements(
        self,
        detected_names: List[Dict],
        context: Dict
    ) -> List[Dict]:
        """
        Generate descriptor-based replacements for detected names.
        
        Per Guidelines:
        "Always qualify it with a character wearing an orange shirt,
        main character, female lead, white puppy etc."
        """
        replacements = []
        
        # Extract visual attributes from context
        visual_attrs = context.get('visual_attributes', {})
        
        for name_info in detected_names:
            name = name_info['name']
            name_type = name_info['type']
            
            if name_type in ['person_name', 'character_name', 'titled_name']:
                # Generate person descriptors
                descriptor = self._generate_person_descriptor(name, visual_attrs)
                replacements.append({
                    'original': name,
                    'type': name_type,
                    'replacement': descriptor,
                    'confidence': 0.8
                })
            
            elif name_type == 'sports_team':
                # Generate team descriptors
                descriptor = self._generate_team_descriptor(name, context)
                replacements.append({
                    'original': name,
                    'type': name_type,
                    'replacement': descriptor,
                    'confidence': 0.9
                })
            
            elif name_type == 'company_or_band':
                # Generate company/band descriptors
                descriptor = self._generate_company_descriptor(name, context)
                replacements.append({
                    'original': name,
                    'type': name_type,
                    'replacement': descriptor,
                    'confidence': 0.7
                })
            
            elif name_type == 'media_title':
                # Generate media descriptors
                descriptor = self._generate_media_descriptor(name, context)
                replacements.append({
                    'original': name,
                    'type': name_type,
                    'replacement': descriptor,
                    'confidence': 0.8
                })
        
        return replacements
    
    def _generate_person_descriptor(
        self,
        name: str,
        visual_attrs: Dict
    ) -> str:
        """
        Generate descriptor for person.
        
        Examples from Guidelines:
        - "character wearing an orange shirt"
        - "main character"
        - "female lead"
        - "man in black jacket"
        - "woman with white sports shoes"
        """
        # Try to get attributes
        clothing = visual_attrs.get('clothing', {}).get(name, '')
        role = visual_attrs.get('role', {}).get(name, '')
        gender = visual_attrs.get('gender', {}).get(name, '')
        
        # Build descriptor
        if clothing:
            if gender:
                return f"{gender} wearing {clothing}"
            else:
                return f"person wearing {clothing}"
        elif role:
            return role
        elif gender:
            return f"the {gender}"
        else:
            return "the person"
    
    def _generate_team_descriptor(self, team_name: str, context: Dict) -> str:
        """Generate descriptor for sports team"""
        # Example: "Lakers" → "the team in yellow jerseys"
        team_attrs = context.get('team_attributes', {}).get(team_name, {})
        
        jersey_color = team_attrs.get('jersey_color', '')
        if jersey_color:
            return f"the team in {jersey_color} jerseys"
        else:
            return "the team"
    
    def _generate_company_descriptor(self, company_name: str, context: Dict) -> str:
        """Generate descriptor for company/band"""
        # Example: "Apple" → "the tech company"
        # Example: "Beatles" → "the band"
        
        company_type = context.get('entity_type', {}).get(company_name, '')
        if company_type:
            return f"the {company_type}"
        else:
            return "the organization"
    
    def _generate_media_descriptor(self, media_title: str, context: Dict) -> str:
        """Generate descriptor for movie/book/song"""
        # Example: "Inception" → "the movie"
        # Example: "Bohemian Rhapsody" → "the song"
        
        media_type = context.get('media_type', {}).get(media_title, '')
        if media_type:
            return f"the {media_type}"
        else:
            return "the content"
    
    def _calculate_confidence(self, detected_names: List[Dict], text: str) -> float:
        """Calculate confidence in name detection"""
        if not detected_names:
            return 1.0  # High confidence: no names found
        
        # Lower confidence if we found many potential names
        # (might be false positives)
        num_detected = len(detected_names)
        
        # Check if descriptors are already present
        has_descriptors = any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in self.descriptor_patterns
        )
        
        if has_descriptors:
            # Lower confidence - might be false positives
            return max(0.5, 1.0 - (num_detected * 0.1))
        else:
            # High confidence - likely real names
            return max(0.7, 1.0 - (num_detected * 0.05))
    
    def apply_replacements(
        self,
        text: str,
        replacements: List[Dict]
    ) -> str:
        """
        Apply descriptor replacements to text.
        
        Args:
            text: Original text
            replacements: List of replacement dictionaries
            
        Returns:
            Text with names replaced by descriptors
        """
        result = text
        
        # Sort by position (replace from end to start to preserve indices)
        sorted_replacements = sorted(
            replacements,
            key=lambda x: x.get('position', 0),
            reverse=True
        )
        
        for repl in sorted_replacements:
            original = repl['original']
            replacement = repl['replacement']
            
            # Case-sensitive replacement
            result = result.replace(original, replacement)
            
            # Also try case-insensitive
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            result = pattern.sub(replacement, result)
        
        return result
    
    def block_names_in_logit_bias(
        self,
        detected_names: List[Dict],
        tokenizer
    ) -> Dict[int, float]:
        """
        Create logit bias dictionary to block name tokens.
        
        Per architecture: Block names completely with -1000 bias.
        
        Args:
            detected_names: List of detected names
            tokenizer: Tokenizer to convert names to token IDs
            
        Returns:
            Dictionary mapping token_id -> bias_value
        """
        logit_bias = {}
        
        for name_info in detected_names:
            name = name_info['name']
            
            # Tokenize the name
            tokens = tokenizer.encode(name)
            
            # Block all tokens from this name
            for token_id in tokens:
                logit_bias[token_id] = -1000  # Complete block
        
        return logit_bias


def detect_and_block_names(
    text: str,
    context: Optional[Dict] = None
) -> Tuple[bool, List[str], str]:
    """
    Convenience function to detect names and get replacements.
    
    Returns:
        (has_names, violation_messages, corrected_text)
    """
    blocker = ExtendedNameBlocker()
    result = blocker.detect_names(text, context)
    
    violations = []
    if result.has_names:
        for name_info in result.detected_names:
            name_type = name_info['type']
            name = name_info['name']
            violations.append(
                f"Found {name_type}: '{name}'. "
                f"Use descriptors instead (e.g., 'person wearing orange shirt')."
            )
    
    # Generate corrected text
    corrected_text = text
    if result.suggested_replacements and context:
        corrected_text = blocker.apply_replacements(text, result.suggested_replacements)
    
    return (result.has_names, violations, corrected_text)
