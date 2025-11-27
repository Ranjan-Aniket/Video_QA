"""
Quality Fixer - Post-processing module for Phase 8

Uses Claude Haiku 4 to fix quality issues in generated questions:
- Hedging language ("appears to", "seems to", "might be", etc.)
- Pronoun usage ("he", "she", "they", "him", "her", etc.)

Without wasting the cluster or regenerating all questions.

Cost: ~$0.0003 per question fix (vs $0.019 to regenerate)
Speed: ~500ms per fix
"""

import logging
import json
import re
from typing import Dict, List, Tuple
from anthropic import Anthropic

# ✅ FIXED: Import centralized validation patterns
from .validation_patterns import HEDGING_PATTERNS, PRONOUN_PATTERNS, NAME_PATTERNS

logger = logging.getLogger(__name__)


class HedgingFixer:
    """Fix hedging language and pronoun usage in questions using Claude Haiku"""

    # ✅ FIXED: Use centralized validation patterns from validation_patterns.py
    # (imported at module level above)

    def __init__(self, anthropic_api_key: str):
        """Initialize with Claude API key"""
        self.client = Anthropic(api_key=anthropic_api_key)
        self.model = "claude-haiku-4-5-20251001"  # Latest Haiku
        self.fix_count = 0
        self.total_cost = 0.0

        # ✅ PROMPT CACHING: Build cacheable system prompt once (reused across ALL questions)
        self._cached_system_prompt = self._build_cached_system_prompt()

    def _build_cached_system_prompt(self) -> str:
        """
        Build CACHEABLE system prompt with all static guidelines.
        This will be reused across ALL questions.

        Returns:
            System prompt string
        """
        return """You are fixing quality issues in video Q&A questions.

**YOUR TASK:**
Rewrite the question and answer to fix ALL quality issues.

**FIXING RULES:**

1. **Remove ALL hedging language** - Use DEFINITIVE statements:
   ❌ "appears to be" → ✅ "is"
   ❌ "seems to show" → ✅ "shows"
   ❌ "could be" → ✅ "is"
   ❌ "suggests that" → ✅ "shows that" or "demonstrates that"

2. **Remove ALL pronouns** - Use specific descriptors:
   ❌ "he is wearing" → ✅ "the person is wearing"
   ❌ "she picks up" → ✅ "the individual picks up"
   ❌ "they are positioned" → ✅ "the objects are positioned"
   ❌ "his hand" → ✅ "the person's hand"
   ❌ "their position" → ✅ "the figures' position"

3. **Remove ALL proper names** - Use generic descriptors:
   ❌ "John throws" → ✅ "the person throws"
   ❌ "Sarah's action" → ✅ "the individual's action"
   ❌ "Lakers jersey" → ✅ "the basketball team's jersey"
   ❌ "Nike logo" → ✅ "the brand logo"
   ❌ "LEGO model" → ✅ "the construction toy model"
   ❌ "Mario jumps" → ✅ "the character jumps"

4. **Preserve everything else:**
   - Keep the same meaning and information
   - Use the same question type
   - Maintain similar length
   - DO NOT change: timestamps, audio_cue, visual_cue, confidence, question_type
   - ONLY fix the "question" and "golden_answer" fields

**EXAMPLES:**

❌ BAD: "The visual shows he appears to be holding a tool"
✅ GOOD: "The visual shows the person is holding a tool"

❌ BAD: "What does her action suggest about the scene?"
✅ GOOD: "What does the person's action demonstrate about the scene?"

❌ BAD: "They seem to be positioned near the edge"
✅ GOOD: "The objects are positioned near the edge"

❌ BAD: "When does Mario jump in the video?"
✅ GOOD: "When does the character jump in the video?"

❌ BAD: "What is John wearing in this scene?"
✅ GOOD: "What is the person wearing in this scene?"

**OUTPUT FORMAT:**
Return ONLY a JSON object with the fixed question and answer:
{{
  "question": "Fixed question text here",
  "golden_answer": "Fixed answer text here"
}}

Return valid JSON only, no other text."""

    def fix_quality_issues(
        self,
        questions: List[Dict],
        has_hedging: bool = False,
        has_pronouns: bool = False,
        has_names: bool = False,
        hedging_warnings: List[str] = None,
        pronoun_warnings: List[str] = None,
        name_warnings: List[str] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Fix quality issues (hedging, pronouns, and/or names) in questions.

        Args:
            questions: List of question dicts
            has_hedging: Whether hedging was detected
            has_pronouns: Whether pronouns were detected
            has_names: Whether names were detected
            hedging_warnings: List of hedging warning messages
            pronoun_warnings: List of pronoun warning messages
            name_warnings: List of name warning messages

        Returns:
            (fixed_questions, fix_stats)
        """
        if not has_hedging and not has_pronouns and not has_names:
            return questions, {"fixes_applied": 0, "hedging_fixes": 0, "pronoun_fixes": 0, "name_fixes": 0, "cost": 0.0}

        hedging_warnings = hedging_warnings or []
        pronoun_warnings = pronoun_warnings or []
        name_warnings = name_warnings or []

        issue_types = []
        if has_hedging:
            issue_types.append("hedging")
        if has_pronouns:
            issue_types.append("pronouns")
        if has_names:
            issue_types.append("names")

        logger.info(f"🔧 Quality issues detected ({', '.join(issue_types)}), fixing affected questions...")

        fixed_questions = []
        fixes_applied = 0
        hedging_fixes = 0
        pronoun_fixes = 0
        name_fixes = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for i, question in enumerate(questions):
            question_id = question.get('question_id', f'Q{i+1}')

            # Check if this specific question has issues
            question_has_hedging = any(question_id in warning for warning in hedging_warnings)
            question_has_pronouns = any(question_id in warning for warning in pronoun_warnings)
            question_has_names = any(question_id in warning for warning in name_warnings)

            if question_has_hedging or question_has_pronouns or question_has_names:
                issues = []
                if question_has_hedging:
                    issues.append("hedging")
                    hedging_fixes += 1
                if question_has_pronouns:
                    issues.append("pronouns")
                    pronoun_fixes += 1
                if question_has_names:
                    issues.append("names")
                    name_fixes += 1

                issue_type = "multiple" if len(issues) > 1 else issues[0]
                logger.info(f"   Fixing {question_id} ({', '.join(issues)})...")

                try:
                    fixed_q, tokens = self._fix_single_question(question, issue_type=issue_type)
                    total_input_tokens += tokens['input']
                    total_output_tokens += tokens['output']
                    fixes_applied += 1
                    fixed_questions.append(fixed_q)
                    logger.info(f"   ✅ Fixed {question_id}")
                except Exception as e:
                    logger.error(f"   ❌ Failed to fix {question_id}: {e}")
                    # Keep original question if fix fails
                    fixed_questions.append(question)
            else:
                # No issues, keep as-is
                fixed_questions.append(question)

        # Calculate cost
        input_cost = (total_input_tokens / 1_000_000) * 0.25
        output_cost = (total_output_tokens / 1_000_000) * 1.25
        fix_cost = input_cost + output_cost

        self.fix_count += fixes_applied
        self.total_cost += fix_cost

        fix_stats = {
            "fixes_applied": fixes_applied,
            "hedging_fixes": hedging_fixes,
            "pronoun_fixes": pronoun_fixes,
            "name_fixes": name_fixes,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cost": fix_cost
        }

        logger.info(f"   Fixed {fixes_applied} questions (hedging: {hedging_fixes}, pronouns: {pronoun_fixes}, names: {name_fixes}, cost: ${fix_cost:.4f})")

        return fixed_questions, fix_stats

    def fix_hedging_in_questions(
        self,
        questions: List[Dict],
        has_hedging: bool,
        hedging_warnings: List[str]
    ) -> Tuple[List[Dict], Dict]:
        """
        Fix hedging language in questions that have violations.
        (Backward compatibility wrapper for fix_quality_issues)

        Args:
            questions: List of question dicts
            has_hedging: Whether hedging was detected
            hedging_warnings: List of warning messages

        Returns:
            (fixed_questions, fix_stats)
        """
        return self.fix_quality_issues(
            questions=questions,
            has_hedging=has_hedging,
            has_pronouns=False,
            hedging_warnings=hedging_warnings,
            pronoun_warnings=[]
        )

    def _fix_single_question(self, question: Dict, issue_type: str = "hedging") -> Tuple[Dict, Dict]:
        """
        Fix quality issues in a single question using Claude Haiku.

        Args:
            question: Question dict with quality issues
            issue_type: Type of issue ("hedging", "pronouns", "names", or "multiple")

        Returns:
            (fixed_question, token_usage)
        """
        # Detect what issues exist
        combined_text = question.get('question', '') + ' ' + question.get('golden_answer', '')
        has_hedging = self._detect_in_text(combined_text, HEDGING_PATTERNS)
        has_pronouns = self._detect_in_text(combined_text, PRONOUN_PATTERNS)
        # Use LLM-based name detection (more comprehensive than regex)
        has_names, detected_names = self.detect_names_llm(combined_text)

        # Build problem description
        problems = []
        if has_hedging or issue_type in ["hedging", "multiple"]:
            problems.append("""**HEDGING LANGUAGE:**
- "suggests", "indicates", "appears to", "seems to"
- "looks like", "could be", "may be", "might be"
- "likely", "probably", "possibly" """)

        if has_pronouns or issue_type in ["pronouns", "multiple"]:
            problems.append("""**PRONOUN USAGE:**
- "he", "she", "him", "her", "his", "hers"
- "they", "them", "their", "theirs" """)

        if has_names or issue_type in ["names", "multiple"]:
            name_list = f" (detected: {', '.join(detected_names)})" if has_names and detected_names else ""
            problems.append(f"""**PROPER NAMES{name_list}:**
- People names (e.g., "John", "Sarah", "Mr. Smith")
- Team/Brand names (e.g., "Lakers", "Nike", "LEGO")
- Character names (e.g., "Mario", "Pikachu") """)

        problem_text = "\n\n".join(problems)

        # Build dynamic user prompt (question-specific data only)
        user_prompt = f"""**ORIGINAL QUESTION:**
{question.get('question', '')}

**ORIGINAL ANSWER:**
{question.get('golden_answer', '')}

**PROBLEMS DETECTED:**
{problem_text}

**QUESTION TYPE:** {question.get('question_type', '')}"""

        # ✅ PROMPT CACHING: Use system messages with cache_control
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            temperature=0,
            system=[{
                "type": "text",
                "text": self._cached_system_prompt,
                "cache_control": {"type": "ephemeral"}
            }],
            messages=[{"role": "user", "content": user_prompt}]
        )

        # Parse response
        response_text = response.content[0].text.strip()

        # Remove markdown code fences if present
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response_text = '\n'.join(lines).strip()

        # Parse JSON
        fixed_data = json.loads(response_text)

        # Create fixed question by updating only question and answer
        fixed_question = question.copy()
        fixed_question['question'] = fixed_data['question']
        fixed_question['golden_answer'] = fixed_data['golden_answer']

        # ✅ GAP #5 FIX: Validate answer length post-hedging (250-400 chars required)
        answer_length = len(fixed_question['golden_answer'])
        if answer_length < 250:
            logger.warning(f"   ⚠️  Answer too short after hedging fix ({answer_length} chars < 250 min)")
            # Keep original answer if fix made it too short
            fixed_question['golden_answer'] = question['golden_answer']
            fixed_question['length_validation_failed'] = True
        elif answer_length > 400:
            logger.warning(f"   ⚠️  Answer too long after hedging fix ({answer_length} chars > 400 max)")
            # Truncate to 400 chars at sentence boundary
            truncated = fixed_question['golden_answer'][:400]
            # Find last period before 400 chars
            last_period = truncated.rfind('.')
            if last_period > 250:  # Only truncate if we still have 250+ chars
                fixed_question['golden_answer'] = truncated[:last_period + 1]
                fixed_question['length_truncated'] = True
        else:
            # Length is valid
            fixed_question['length_validated'] = True

        # Add fix marker
        fixed_question['hedging_fixed'] = True

        # Token usage
        tokens = {
            'input': response.usage.input_tokens,
            'output': response.usage.output_tokens
        }

        return fixed_question, tokens

    def _detect_in_text(self, text: str, patterns: List[str]) -> bool:
        """
        Helper method to detect patterns in text.

        Args:
            text: Text to check
            patterns: List of regex patterns

        Returns:
            True if any pattern matches
        """
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def detect_hedging(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect hedging language in text.

        Args:
            text: Text to check

        Returns:
            (has_hedging, list of matched patterns)
        """
        matches = []
        for pattern in HEDGING_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)

        return len(matches) > 0, matches

    def detect_pronouns(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect pronoun usage in text.

        Args:
            text: Text to check

        Returns:
            (has_pronouns, list of matched patterns)
        """
        matches = []
        for pattern in PRONOUN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)

        return len(matches) > 0, matches

    def detect_names(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect proper names (people, brands, teams) in text using regex (legacy).

        Args:
            text: Text to check

        Returns:
            (has_names, list of matched patterns)
        """
        matches = []
        for pattern in NAME_PATTERNS:
            match = re.search(pattern, text)
            if match:
                matches.append(match.group(0))

        return len(matches) > 0, matches

    def detect_names_llm(self, text: str) -> Tuple[bool, List[str]]:
        """
        Detect ALL proper names using Claude Haiku LLM.
        More comprehensive than regex - catches contextual names.

        Args:
            text: Text to check

        Returns:
            (has_names, list of detected proper nouns)
        """
        prompt = f"""Scan this text and identify ALL proper nouns (people, brands, teams, characters, places):

TEXT: "{text}"

PROPER NOUNS TO DETECT:
- Person names: "John", "Sarah", "Mr. Smith", "Dr. Jones"
- Character names: "Mario", "Pikachu", "Batman", "Elsa"
- Brand names: "Nike", "LEGO", "Adidas", "Apple", "Tesla"
- Team names: "Lakers", "Warriors", "Manchester United"
- Place names: "Paris", "New York", "Eiffel Tower"
- Product names: "iPhone", "PlayStation", "Ferrari"

IMPORTANT RULES:
1. Detect names in ANY context (possessive, object, subject, etc.):
   - "Mario's action" → DETECT "Mario"
   - "watching Mario" → DETECT "Mario"
   - "LEGO Ninjago figure" → DETECT "LEGO" and "Ninjago"
   - "the Lakers jersey" → DETECT "Lakers"

2. DO NOT flag generic words even if capitalized:
   - "The person moves" → NO DETECTION (generic)
   - "A tool is visible" → NO DETECTION (generic)

3. Return ONLY the proper nouns found, one per line.
4. If NO proper nouns found, return exactly: "NONE"

OUTPUT (one name per line or "NONE"):"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.content[0].text.strip()

            if result == "NONE" or not result:
                return False, []

            # Parse names from response (one per line)
            detected_names = [
                name.strip()
                for name in result.split('\n')
                if name.strip() and name.strip() != "NONE"
            ]

            return len(detected_names) > 0, detected_names

        except Exception as e:
            logger.warning(f"LLM name detection failed, falling back to regex: {e}")
            # Fallback to regex if LLM fails
            return self.detect_names(text)

    def get_stats(self) -> Dict:
        """Get statistics about fixes applied"""
        return {
            "total_fixes": self.fix_count,
            "total_cost": self.total_cost,
            "avg_cost_per_fix": self.total_cost / self.fix_count if self.fix_count > 0 else 0
        }


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)

    # Test with mock question
    fixer = HedgingFixer(anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY'))

    mock_question = {
        'question_id': 'Q1',
        'question': 'What visual change suggests that the animation has started?',
        'golden_answer': 'The screen appears to transition from static to animated.',
        'question_type': 'visual_change',
        'confidence': 0.95
    }

    # Detect hedging
    has_hedging_q, _ = fixer.detect_hedging(mock_question['question'])
    has_hedging_a, _ = fixer.detect_hedging(mock_question['golden_answer'])

    print(f"\nHedging in question: {has_hedging_q}")
    print(f"Hedging in answer: {has_hedging_a}")

    if has_hedging_q or has_hedging_a:
        print("\nFixing question...")
        fixed_questions, stats = fixer.fix_hedging_in_questions(
            [mock_question],
            has_hedging=True,
            hedging_warnings=['Q1']
        )

        print(f"\nOriginal question: {mock_question['question']}")
        print(f"Fixed question: {fixed_questions[0]['question']}")
        print(f"\nOriginal answer: {mock_question['golden_answer']}")
        print(f"Fixed answer: {fixed_questions[0]['golden_answer']}")
        print(f"\nCost: ${stats['cost']:.4f}")
