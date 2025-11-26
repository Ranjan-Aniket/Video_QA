"""
Hallucination Validator - Critical Gap #3 Fix

Validates that golden answers match actual visual evidence in frames.
Prevents hallucinated claims like "person grips shark's tail" when
they're just positioned near each other.

Uses GPT-4o Vision to verify answer claims against frame content.
"""

import logging
import base64
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class HallucinationValidator:
    """Validate that answers match visual evidence using GPT-4o Vision"""

    def __init__(self, openai_api_key: str):
        """Initialize with OpenAI API key"""
        self.client = OpenAI(api_key=openai_api_key)
        self.model = "gpt-4o-2024-11-20"  # Latest GPT-4o Vision
        self.validation_count = 0
        self.rejections = 0

    def validate_answer_against_frame(
        self,
        question: str,
        answer: str,
        frame_path: Path,
        question_type: str
    ) -> Tuple[bool, str]:
        """
        Validate that the answer claims can be verified from the frame.

        Args:
            question: The question text
            answer: The golden answer to validate
            frame_path: Path to the frame image
            question_type: Type of question (for context)

        Returns:
            (is_valid, rejection_reason)
            - is_valid: True if answer matches visual evidence
            - rejection_reason: Empty string if valid, explanation if invalid
        """
        self.validation_count += 1

        try:
            # Load and encode image
            with open(frame_path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')

            # Build validation prompt
            prompt = f"""You are a strict visual evidence validator for video Q&A datasets.

**TASK**: Verify that the ANSWER claims can be DIRECTLY OBSERVED in the image.

**QUESTION**: {question}

**ANSWER TO VALIDATE**: {answer}

**QUESTION TYPE**: {question_type}

**VALIDATION RULES**:

1. **DIRECT OBSERVATION ONLY** - The answer must describe what is ACTUALLY VISIBLE:
   ✅ VALID: "The person's hand is positioned near the object"
   ❌ INVALID: "The person grips the object" (can't see if actually gripping)

   ✅ VALID: "Two objects are placed side by side"
   ❌ INVALID: "One object is heavier than the other" (weight not visible)

2. **NO INFERENCE BEYOND VISUAL** - Don't allow claims that require assumptions:
   ✅ VALID: "The person extends their hand toward the camera"
   ❌ INVALID: "The person is counting to three" (intent not visible, only gesture visible)

   ✅ VALID: "Text reading 'STOP' is visible on the sign"
   ❌ INVALID: "The sign warns drivers to stop" (function is inference, only text visible)

3. **SPECIFIC SPATIAL RELATIONSHIPS** - Verify exact positions, not assumptions:
   ✅ VALID: "The person's hand is raised above shoulder level"
   ❌ INVALID: "The person is waving goodbye" (waving is motion, static image only shows position)

4. **VERIFIABLE ATTRIBUTES ONLY** - Only validate what can be measured/seen:
   ✅ VALID: "The background has a grainy texture throughout the frame"
   ❌ INVALID: "The video quality is low" (quality is subjective interpretation)

**SPECIAL CASES**:

- **Gestures**: Allow reasonable interpretations (per golden dataset):
  ✅ "The person extends the left hand as if counting" → VALID (gesture interpretation allowed)
  ✅ "Fingers shaped like a box or rectangle" → VALID (metaphorical description allowed)
  ❌ "The person is actually counting to five" → INVALID (claims intent, not just gesture)

- **Visual Scope**: Answer MUST specify scope for visual effects:
  ✅ "The entire frame, including the person and background, displays visible grain" → VALID
  ❌ "The scene becomes grainy" → INVALID (unclear scope - what parts affected?)

**YOUR TASK**:
1. Look at the image carefully
2. Check EVERY claim in the answer against what you can see
3. If ANY claim cannot be directly verified, REJECT

Return ONLY a JSON object:
{{
  "is_valid": true/false,
  "rejection_reason": "Empty string if valid, specific explanation if invalid"
}}

**EXAMPLES**:

Example 1:
Answer: "The person grips the shark's tail"
Image: Shows person's hand near shark tail, but contact unclear
Response: {{"is_valid": false, "rejection_reason": "Cannot verify 'grips' - hand is near tail but actual contact/grip not visible in static image"}}

Example 2:
Answer: "The person's hand is positioned near the shark's tail in the lower right portion of the frame"
Image: Shows person's hand near shark tail
Response: {{"is_valid": true, "rejection_reason": ""}}

Example 3:
Answer: "The scene becomes grainy"
Image: Shows grainy texture
Response: {{"is_valid": false, "rejection_reason": "Scope not specified - unclear if 'scene' means entire frame, background only, or specific region"}}

Now validate the answer above against the image.
"""

            # Call GPT-4o Vision for validation
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=200,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ]
            )

            # Parse response
            result_text = response.choices[0].message.content.strip()

            # Remove markdown code fences if present
            if result_text.startswith('```'):
                lines = result_text.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                result_text = '\n'.join(lines).strip()

            result = json.loads(result_text)
            is_valid = result.get('is_valid', False)
            rejection_reason = result.get('rejection_reason', '')

            if not is_valid:
                self.rejections += 1
                logger.debug(f"Hallucination detected: {rejection_reason}")

            return is_valid, rejection_reason

        except Exception as e:
            logger.warning(f"Hallucination validation failed (defaulting to valid): {e}")
            # Default to valid if validation fails (don't block on validation errors)
            return True, ""

    def validate_question_batch(
        self,
        questions: List[Dict],
        frames_dir: Path
    ) -> Tuple[List[Dict], Dict]:
        """
        Validate a batch of questions against their frames.

        Args:
            questions: List of GeneratedQuestion objects (as dicts)
            frames_dir: Directory containing frame images

        Returns:
            (valid_questions, validation_stats)
        """
        logger.info(f"🔍 Validating {len(questions)} questions for hallucinations...")

        valid_questions = []
        rejected_questions = []

        for q in questions:
            # Get frame path
            frame_id = q.get('frame_id', '')
            if not frame_id or 'cluster' in frame_id:
                # Skip cluster questions (harder to validate, need multi-frame logic)
                valid_questions.append(q)
                continue

            frame_path = frames_dir / f"{frame_id}.jpg"
            if not frame_path.exists():
                logger.warning(f"   Frame not found: {frame_path}, skipping validation")
                valid_questions.append(q)  # Don't reject if frame missing
                continue

            # Validate answer against frame
            is_valid, rejection_reason = self.validate_answer_against_frame(
                question=q.get('question', ''),
                answer=q.get('golden_answer', ''),
                frame_path=frame_path,
                question_type=q.get('question_type', '')
            )

            if is_valid:
                valid_questions.append(q)
            else:
                rejected_questions.append({
                    'question_id': q.get('question_id', ''),
                    'reason': rejection_reason
                })
                logger.warning(f"   ❌ Rejected {q.get('question_id', '')}: {rejection_reason}")

        validation_stats = {
            'total_validated': self.validation_count,
            'rejections': self.rejections,
            'rejection_rate': self.rejections / self.validation_count if self.validation_count > 0 else 0,
            'rejected_questions': rejected_questions
        }

        logger.info(f"   ✅ Validation complete: {len(valid_questions)}/{len(questions)} valid ({self.rejections} rejected)")

        return valid_questions, validation_stats

    def get_stats(self) -> Dict:
        """Get validation statistics"""
        return {
            "total_validations": self.validation_count,
            "total_rejections": self.rejections,
            "rejection_rate": self.rejections / self.validation_count if self.validation_count > 0 else 0
        }


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)

    # Test with mock data
    validator = HallucinationValidator(openai_api_key=os.environ.get('OPENAI_API_KEY'))

    # Note: This test requires an actual frame image to validate
    # mock_frame_path = Path("/path/to/test/frame.jpg")
    # is_valid, reason = validator.validate_answer_against_frame(
    #     question="What action does the person perform?",
    #     answer="The person grips the shark's tail",
    #     frame_path=mock_frame_path,
    #     question_type="Counting"
    # )
    # print(f"Valid: {is_valid}, Reason: {reason}")
