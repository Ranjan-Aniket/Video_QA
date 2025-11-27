"""
Pass 3: Sonnet 4.5 - Batched Q&A Generation

Generates adversarial Q&A pairs from validated moments.

Features:
- Batch processing (10-12 moments per API call)
- Strict validation (both cues, no names, not single-cue answerable)
- 1-2 questions per moment
- Uses cue triad (visual/audio/correspondence) for context
- ✅ Prompt caching enabled (guidelines cached, 28% cost reduction)

Cost: ~$0.43 first video, ~$0.31 subsequent (Sonnet 4.5 with caching)
Output: 35-50 QA pairs
"""

import json
from typing import Dict, List
from pathlib import Path
from loguru import logger
import anthropic
import os

# Quality improvement modules
from processing.hedging_fixer import HedgingFixer
from processing.hallucination_validator import HallucinationValidator


class Pass3QAGenerator:
    """
    Batched Q&A generator using Sonnet 4.5
    """

    def __init__(self):
        """Initialize QA generator"""
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-5-20250929"
        self.batch_size = 12  # Process 10-12 moments per API call

        # Quality improvement tools
        self.hedging_fixer = HedgingFixer(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.hallucination_validator = HallucinationValidator(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Build cacheable system prompt (reused across all videos)
        self._cached_system_prompt = self._build_cached_system_prompt()

    def _build_cached_system_prompt(self) -> str:
        """
        Build CACHEABLE system prompt with all static guidelines.
        This will be reused across ALL videos (90% cost reduction on this part).

        Returns:
            System prompt string
        """
        return """=== QA GENERATION GUIDELINES ===

For each moment, generate 1-2 questions that:

1. EXPLOIT ADVERSARIAL FEATURES
   - Each question must target specific adversarial features listed above
   - Make questions that are easy to get wrong

2. REQUIRE BOTH CUES
   - Question must reference audio AND visual elements
   - Cannot be answered from audio alone
   - Cannot be answered from video alone
   - Use the correspondence to understand how they relate

3. NOT ANSWERABLE SINGLE-CUE
   - If question can be answered with just audio → REJECT
   - If question can be answered with just visual → REJECT
   - Both modalities must be ESSENTIAL to answer

4. NO NAMES
   - Use descriptors: "man in blue shirt", "woman with glasses"
   - Never use proper names (John, Mary, etc.)

5. QUESTION DIRECTION DIVERSITY
   - Mix "When you hear X, what do you see?" style
   - Mix "When you see X, what do you hear?" style
   - Mix "What happens when..." style

6. PRECISE TIMESTAMPS
   - start_time = when visual cue first appears
   - end_time = when action/audio in answer completes

7. AUDIO CUE DIVERSITY
   - Not just speech - also music, ambient sounds, sound effects
   - Reference audio events explicitly

8. CONCISE CUES
   - Visual cue: 1-2 sentences max
   - Audio cue: 1-2 sentences max
   - No verbose descriptions

=== CRITICAL QUALITY GUIDELINES ===

❌ NO HEDGING LANGUAGE in questions, answers, or cues:
- FORBIDDEN: "appears to", "seems to", "looks like", "could be", "may be", "might be"
- FORBIDDEN: "suggests", "likely", "probably", "possibly", "indicates"
- USE DEFINITIVE LANGUAGE: "shows", "displays", "contains", "is", "are", "does"

Examples:
✗ BAD Q: "What does the person appear to be doing?"
✓ GOOD Q: "What is the person doing?"
✗ BAD A: "The speaker likely means X"
✓ GOOD A: "The speaker means X"

❌ NO PRONOUNS in questions, answers, or cues:
- FORBIDDEN: "he", "she", "him", "her", "his", "hers", "they", "them", "their"
- USE DESCRIPTORS: "person", "speaker", "individual", "man", "woman", "player", "character"

Examples:
✗ BAD: "What does he pick up?"
✓ GOOD: "What does the person pick up?"
✗ BAD: "When she points at the screen..."
✓ GOOD: "When the speaker points at the screen..."

❌ NO PROPER NAMES:
- FORBIDDEN: Person names, brand names, team names, character names
- USE GENERIC DESCRIPTORS

=== ANSWER LENGTH REQUIREMENTS ===

STRICT REQUIREMENTS:
- Minimum: 50 words (250 characters)
- Maximum: 80 words (400 characters)
- Target: 60-70 words (300-350 characters)

Answers must be:
- RICH: Provide full context and details
- COMPLETE: Answer the full question, not just part of it
- CONCRETE: Use specific details from audio and visual cues
- GROUNDED: Only describe what is observable in the provided cues

Examples of GOOD answers (60-70 words):
✓ "The speaker says 'look at the union label' while pointing at the person's shirt. The text 'TO THE NUIUN 9 LABLL 2,6080' is visible on the shirt label. The text is partially occluded and contains unusual characters like '9' which could be misread as 'g'. This requires matching the speech timing with when the text becomes visible."

Examples of BAD answers:
✗ TOO SHORT (15 words): "The speaker mentions the label and points at the shirt text."
✗ TOO VAGUE: "The person talks about something on the shirt while gesturing."

=== MINIMUM TEMPORAL WINDOWS (by type) ===

Enforce these MINIMUM durations for each type:
- Sequential: 30.0 seconds minimum
- Comparative: 30.0 seconds minimum
- Temporal Understanding: 20.0 seconds minimum
- Inference: 15.0 seconds minimum
- Audio-Visual Stitching: 10.0 seconds minimum

For other types, use moment duration as provided.

If a moment's duration is LESS than the minimum for its type:
- Either extend end_time to meet minimum
- Or skip generating questions for that moment

=== AUDIO MODALITY DIVERSITY ===

Generated questions should collectively use at least 2 different audio modalities:
1. SPEECH: Dialogue, narration, words, phrases
2. MUSIC: Tempo, tone, melody, starts/stops
3. SOUND EFFECTS: Impacts, whooshes, mechanical sounds, clicks
4. SILENCE: Pauses, gaps, scene boundaries

DON'T generate all questions about speech - vary the audio modality.

=== HALLUCINATION PREVENTION ===

FORBIDDEN HALLUCINATIONS (these will be REJECTED):
✗ NO inventing details not in the provided cues
✗ NO inventing interactions not described in correspondence
✗ NO inventing object states not mentioned in visual_cues
✗ NO inventing audio content not mentioned in audio_cues
✗ NO assuming actions before/after the temporal window
✗ NO inferring emotions/intentions without explicit cues

ONLY use information from:
- visual_cues (what you can see)
- audio_cues (what you can hear)
- correspondence (how they relate)

VERIFICATION CHECKLIST before finalizing each answer:
☐ Is every detail in the answer mentioned in the cues?
☐ Am I inventing any information not provided?
☐ Am I assuming actions outside the temporal window?
☐ Is the answer 50-80 words (250-400 chars)?
☐ Does the answer have NO hedging language?
☐ Does the answer have NO pronouns?
☐ Does the answer have NO proper names?

=== OUTPUT FORMAT ===

For each moment, output:

{
  "moment_id": 1,
  "questions": [
    {
      "question": "When the speaker says 'look at the label', what specific text is visible on the shirt?",
      "answer": "TO THE NUIUN 9 LABLL 2,6080",
      "visual_cue": "Text 'TO THE NUIUN 9 LABLL 2,6080' visible on shirt label",
      "audio_cue": "Speaker says 'look at the label'",
      "start_time": 56.0,
      "end_time": 58.5,
      "primary_task_type": "Needle",
      "sub_task_types": ["Referential"],
      "validation": {
        "audio_only_answerable": false,
        "visual_only_answerable": false,
        "both_cues_in_question": true,
        "names_used": false,
        "exploits_adversarial_features": true
      },
      "adversarial_rationale": "Text is partially occluded and easy to misread. Requires matching speech timing to text visibility. Audio-only would miss the exact text."
    }
  ]
}

=== CONSTRAINTS ===

- 1-2 questions per moment (prioritize quality over quantity)
- Each question must pass ALL validation checks
- If a moment doesn't yield good adversarial questions, skip it
- Focus on truly challenging questions that exploit the adversarial features

Return JSON array:

[
  { "moment_id": 1, "questions": [...] },
  { "moment_id": 2, "questions": [...] },
  ...
]

Generate Q&A pairs now. Be ruthless about quality - only generate questions that will truly challenge Gemini."""

    def build_batch_prompt(
        self,
        moments: List[Dict],
        video_id: str
    ) -> str:
        """
        Build FRESH user prompt with video-specific moment data.
        This changes per video/batch.

        Args:
            moments: List of validated moments (up to 12)
            video_id: Video identifier

        Returns:
            User prompt string
        """
        prompt = f"""Generate adversarial Q&A pairs for video: {video_id}

TASK: Generate 1-2 questions per moment that will STUMP Gemini.

MOMENTS PROVIDED: {len(moments)}

"""

        # Add each moment
        for i, moment in enumerate(moments, 1):
            prompt += f"""
=== MOMENT {i} ===

Ontology: {moment.get('primary_ontology')} (secondary: {', '.join(moment.get('secondary_ontologies', []))})
Mode: {moment.get('mode')}
Duration: {moment.get('duration')}s
Timestamps: {moment.get('timestamps')}

VISUAL CUES:
{chr(10).join(f"- {vc}" for vc in moment.get('visual_cues', []))}

AUDIO CUES:
{chr(10).join(f"- {ac}" for ac in moment.get('audio_cues', []))}

CORRESPONDENCE:
{moment.get('correspondence', '')}

ADVERSARIAL FEATURES:
{chr(10).join(f"- {af}" for af in moment.get('adversarial_features', []))}

"""

        return prompt

    def call_sonnet_45(
        self,
        moments: List[Dict],
        video_id: str
    ) -> Dict:
        """
        Call Sonnet 4.5 to generate QA pairs for a batch WITH PROMPT CACHING.

        System prompt (guidelines) is cached after first use - 90% cost reduction!

        Args:
            moments: Batch of moments
            video_id: Video identifier

        Returns:
            Generated QA pairs with cost breakdown
        """
        logger.info(f"Generating QA pairs for batch of {len(moments)} moments...")

        # Build fresh user prompt (video-specific data)
        user_prompt = self.build_batch_prompt(moments, video_id)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                system=[
                    {
                        "type": "text",
                        "text": self._cached_system_prompt,
                        "cache_control": {"type": "ephemeral"}  # 🔥 CACHE THIS!
                    }
                ],
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            # Parse response
            response_text = response.content[0].text

            # Extract JSON
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                logger.error("No JSON array found in Sonnet 4.5 response")
                return {"qa_pairs": [], "error": "No JSON found"}

            result = json.loads(json_match.group(0))

            # Track cost WITH CACHING (Sonnet 4.5 pricing)
            usage = response.usage
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens

            # Check cache usage
            cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0)
            cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0)

            # Calculate cost
            # Regular input: $3.00/MTok, Cache write: $3.75/MTok, Cache read: $0.30/MTok, Output: $15/MTok
            cost = (
                (input_tokens * 3.00 / 1_000_000) +
                (cache_creation_tokens * 3.75 / 1_000_000) +
                (cache_read_tokens * 0.30 / 1_000_000) +
                (output_tokens * 15.00 / 1_000_000)
            )

            # Log cache performance
            if cache_read_tokens > 0:
                cache_savings = (cache_read_tokens * 2.70 / 1_000_000)  # Saved $2.70/MTok
                logger.info(f"  💰 Cache HIT! Read {cache_read_tokens} tokens (saved ${cache_savings:.4f})")
            elif cache_creation_tokens > 0:
                logger.info(f"  📝 Cache WRITE: {cache_creation_tokens} tokens written to cache")

            logger.info(f"  Cost: ${cost:.4f} (input: ${input_tokens * 3.00 / 1_000_000:.4f}, output: ${output_tokens * 15.00 / 1_000_000:.4f})")

            return {
                'qa_pairs': result,
                'cost': cost,
                'tokens': {
                    'input': input_tokens,
                    'output': output_tokens,
                    'cache_creation': cache_creation_tokens,
                    'cache_read': cache_read_tokens
                }
            }

        except Exception as e:
            logger.error(f"Sonnet 4.5 call failed: {e}")
            import traceback
            traceback.print_exc()
            return {"qa_pairs": [], "error": str(e), "cost": 0}

    def generate_qa_pairs(
        self,
        validated_moments: List[Dict],
        video_id: str
    ) -> Dict:
        """
        Generate QA pairs for all validated moments in batches

        Args:
            validated_moments: List of validated moments
            video_id: Video identifier

        Returns:
            {
                'qa_pairs': [...],
                'generation_summary': {...},
                'cost': float
            }
        """
        logger.info("=" * 60)
        logger.info(f"PASS 3: Batched QA Generation ({len(validated_moments)} moments)")
        logger.info("=" * 60)

        all_qa_pairs = []
        total_cost = 0

        # Process in batches
        for i in range(0, len(validated_moments), self.batch_size):
            batch = validated_moments[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(validated_moments) + self.batch_size - 1) // self.batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} moments)...")

            # Generate QA for batch
            batch_result = self.call_sonnet_45(batch, video_id)

            # Collect QA pairs
            for moment_qa in batch_result.get('qa_pairs', []):
                for qa in moment_qa.get('questions', []):
                    all_qa_pairs.append(qa)

            total_cost += batch_result.get('cost', 0)

            logger.info(f"  Batch {batch_num} generated {len(batch_result.get('qa_pairs', []))} moment QAs")

        # Summary
        total_questions = len(all_qa_pairs)
        avg_questions_per_moment = total_questions / len(validated_moments) if validated_moments else 0

        # Validation summary
        validation_passed = sum(1 for qa in all_qa_pairs if self._validate_qa(qa))
        validation_failed = total_questions - validation_passed

        # Ontology distribution
        ontology_dist = {}
        for qa in all_qa_pairs:
            primary = qa.get('primary_task_type', 'Unknown')
            ontology_dist[primary] = ontology_dist.get(primary, 0) + 1

        logger.info("=" * 60)
        logger.info(f"Pass 3 Complete:")
        logger.info(f"  Total QA pairs: {total_questions}")
        logger.info(f"  Avg per moment: {avg_questions_per_moment:.2f}")
        logger.info(f"  Validation passed: {validation_passed}")
        logger.info(f"  Validation failed: {validation_failed}")
        logger.info(f"  Total cost: ${total_cost:.4f}")
        logger.info("")
        logger.info(f"Ontology distribution:")
        for otype, count in sorted(ontology_dist.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {otype}: {count}")
        logger.info("=" * 60)

        return {
            'qa_pairs': all_qa_pairs,
            'generation_summary': {
                'total_questions': total_questions,
                'moments_processed': len(validated_moments),
                'avg_questions_per_moment': avg_questions_per_moment,
                'validation_passed': validation_passed,
                'validation_failed': validation_failed,
                'ontology_distribution': ontology_dist
            },
            'cost': total_cost
        }

    def _validate_qa(self, qa: Dict) -> bool:
        """
        Validate a single QA pair

        Args:
            qa: QA pair dict

        Returns:
            True if valid
        """
        validation = qa.get('validation', {})

        # All validation checks must pass
        required_checks = [
            not validation.get('audio_only_answerable', True),
            not validation.get('visual_only_answerable', True),
            validation.get('both_cues_in_question', False),
            not validation.get('names_used', True),
            validation.get('exploits_adversarial_features', False)
        ]

        return all(required_checks)

    def _apply_hedging_fixes(
        self,
        qa_pairs: List[Dict],
        frames_dir: Path
    ) -> tuple[List[Dict], float]:
        """
        Apply hedging and pronoun fixes to QA pairs

        Args:
            qa_pairs: List of QA pairs
            frames_dir: Directory with frame images

        Returns:
            (fixed_qa_pairs, cost)
        """
        logger.info("Checking QA pairs for quality issues...")

        # Detect issues in all QA pairs
        has_hedging = False
        has_pronouns = False
        has_names = False
        hedging_warnings = []
        pronoun_warnings = []
        name_warnings = []

        for i, qa in enumerate(qa_pairs):
            question_id = qa.get('question_id', f'Q{i+1}')
            text = qa.get('question', '') + ' ' + qa.get('answer', '')

            # Detect hedging
            hedging_found, hedging_matches = self.hedging_fixer.detect_hedging(text)
            if hedging_found:
                has_hedging = True
                hedging_warnings.append(f"{question_id}: {', '.join(hedging_matches[:3])}")

            # Detect pronouns
            pronouns_found, pronoun_matches = self.hedging_fixer.detect_pronouns(text)
            if pronouns_found:
                has_pronouns = True
                pronoun_warnings.append(f"{question_id}: {', '.join(pronoun_matches[:3])}")

            # Detect names (using LLM for better accuracy)
            names_found, name_matches = self.hedging_fixer.detect_names_llm(text)
            if names_found:
                has_names = True
                name_warnings.append(f"{question_id}: {', '.join(name_matches[:3])}")

        # If no issues found, return original pairs
        if not has_hedging and not has_pronouns and not has_names:
            logger.info("  ✅ No quality issues detected")
            return qa_pairs, 0.0

        # Apply fixes using HedgingFixer
        fixed_pairs, fix_stats = self.hedging_fixer.fix_quality_issues(
            questions=qa_pairs,
            has_hedging=has_hedging,
            has_pronouns=has_pronouns,
            has_names=has_names,
            hedging_warnings=hedging_warnings,
            pronoun_warnings=pronoun_warnings,
            name_warnings=name_warnings
        )

        logger.info(f"  Fixed {fix_stats['fixes_applied']}/{len(qa_pairs)} QA pairs (${fix_stats['cost']:.4f})")

        return fixed_pairs, fix_stats['cost']

    def _validate_hallucinations(
        self,
        qa_pairs: List[Dict],
        frames_dir: Path
    ) -> tuple[List[Dict], List[Dict], float]:
        """
        Validate QA pairs for hallucinations

        Args:
            qa_pairs: List of QA pairs
            frames_dir: Directory with frame images

        Returns:
            (validated_pairs, rejected_pairs, cost)
        """
        logger.info("Validating answers against visual evidence...")

        validated_pairs = []
        rejected_pairs = []
        total_cost = 0

        for qa in qa_pairs:
            # Get frame path
            frame_ids = qa.get('frame_ids', [])
            if not frame_ids:
                validated_pairs.append(qa)
                continue

            frame_path = frames_dir / f"frame_{frame_ids[0]:06d}.jpg"

            if not frame_path.exists():
                logger.warning(f"Frame not found for validation: {frame_path}")
                validated_pairs.append(qa)
                continue

            # Validate answer
            is_valid, rejection_reason = self.hallucination_validator.validate_answer_against_frame(
                question=qa.get('question', ''),
                answer=qa.get('answer', ''),
                frame_path=frame_path,
                question_type=qa.get('primary_task_type', '')
            )

            # Track cost (~$0.002 per validation)
            total_cost += 0.002

            if is_valid:
                validated_pairs.append(qa)
            else:
                qa['rejection_reason'] = rejection_reason
                rejected_pairs.append(qa)

        logger.info(f"  Validated: {len(validated_pairs)}/{len(qa_pairs)} (${total_cost:.4f})")
        if rejected_pairs:
            logger.warning(f"  Rejected {len(rejected_pairs)} hallucinated answers")

        return validated_pairs, rejected_pairs, total_cost


def run_qa_generation(
    validated_moments: List[Dict],
    video_id: str,
    frames_dir: str,
    output_path: str,
    apply_quality_fixes: bool = True
) -> Dict:
    """
    Run Pass 3 QA generation with quality improvements and save results

    Args:
        validated_moments: Validated moments from validator
        video_id: Video identifier
        frames_dir: Directory with frame images
        output_path: Path to save results
        apply_quality_fixes: Whether to apply hedging fixes and hallucination validation

    Returns:
        QA generation results
    """
    generator = Pass3QAGenerator()

    # Generate QA pairs
    results = generator.generate_qa_pairs(validated_moments, video_id)

    if apply_quality_fixes:
        # Apply hedging and pronoun fixes
        fixed_pairs, hedging_cost = generator._apply_hedging_fixes(
            results['qa_pairs'],
            Path(frames_dir)
        )

        # Validate for hallucinations
        validated_pairs, rejected_pairs, hallucination_cost = generator._validate_hallucinations(
            fixed_pairs,
            Path(frames_dir)
        )

        # Update results
        results['qa_pairs'] = validated_pairs
        results['rejected_hallucinations'] = rejected_pairs
        results['cost'] += hedging_cost + hallucination_cost
        results['generation_summary']['hedging_fixes_applied'] = len(fixed_pairs) - len([qa for qa in results['qa_pairs'] if qa == fixed_pairs[results['qa_pairs'].index(qa)]])
        results['generation_summary']['hallucinations_rejected'] = len(rejected_pairs)
        results['generation_summary']['final_qa_count'] = len(validated_pairs)

        logger.info(f"Quality improvements: +${hedging_cost + hallucination_cost:.4f}")
        logger.info(f"Final QA count: {len(validated_pairs)} (after quality filtering)")

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Pass 3 results saved to {output_path}")

    return results
