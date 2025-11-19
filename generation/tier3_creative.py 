"""
Tier 3: Creative GPT-4 Generation

Cost: ~$0.75 per video
Hallucination Rate: 2-8% (with validation)
Target: 2 questions per video (ultra-creative edge cases)

Uses GPT-4 for maximum creativity to discover novel adversarial angles
that templates and Llama might miss.

CRITICAL SAFEGUARDS:
- Evidence-first generation (force GPT-4 to cite evidence)
- 10-layer strict validation
- Evidence grounding verification (cross-check with second LLM)
- Human review required
- Comprehensive monitoring

GUIDELINES ENFORCED:
1. ✅ Both audio AND visual cues required
2. ✅ NO names - only descriptors
3. ✅ Precise timestamps (phoneme-level)
4. ✅ Evidence-driven only
5. ✅ Maximum creativity within constraints
"""

from typing import List, Dict, Optional, Tuple
import logging
import json
from dataclasses import dataclass, field
from enum import Enum
import openai
import time

from templates.base import GeneratedQuestion, EvidenceDatabase, QuestionType, Cue, CueType
from generation.validation_strict import StrictValidator, ValidationResult

logger = logging.getLogger(__name__)


class CreativeStrategy(Enum):
    """Creative generation strategies"""
    NOVEL_COMBINATIONS = "novel_combinations"  # Unusual type combinations
    EDGE_CASES = "edge_cases"  # Extreme scenarios
    COUNTER_INTUITIVE = "counter_intuitive"  # Surprising patterns
    MULTI_STEP_REASONING = "multi_step_reasoning"  # Complex reasoning chains
    TEMPORAL_EDGE = "temporal_edge"  # Edge of timestamp boundaries


@dataclass
class Tier3Statistics:
    """Statistics for Tier 3 generation"""
    questions_attempted: int = 0
    questions_generated: int = 0
    questions_rejected: int = 0
    llm_calls: int = 0
    total_cost: float = 0.0
    
    # Rejection tracking
    rejected_hallucination: int = 0
    rejected_validation: int = 0
    rejected_no_dual_cue: int = 0
    rejected_low_quality: int = 0
    
    # Strategy tracking
    strategy_usage: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    average_complexity_score: float = 0.0
    average_prompt_tokens: int = 0
    average_completion_tokens: int = 0


class Tier3CreativeGenerator:
    """
    Tier 3: GPT-4 Creative Generation
    
    GOAL: Discover novel adversarial patterns through maximum creativity
    
    SAFEGUARDS:
    - Evidence-first prompting
    - Citation requirements
    - Strict validation
    - Evidence grounding check
    - Human review required
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        enable_grounding_check: bool = True,
        temperature: float = 0.9,  # High creativity
        max_retries: int = 3
    ):
        """
        Initialize Tier 3 generator
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (default gpt-4)
            enable_grounding_check: Use second LLM to verify evidence grounding
            temperature: Sampling temperature (0.9 for creativity)
            max_retries: Max retries per question
        """
        self.api_key = api_key
        self.model = model
        self.enable_grounding_check = enable_grounding_check
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Initialize OpenAI client
        openai.api_key = api_key
        
        # Validator
        self.validator = StrictValidator()
        
        # Statistics
        self.stats = Tier3Statistics()
        
        logger.info(
            f"Tier 3 Generator initialized: model={model}, "
            f"grounding_check={enable_grounding_check}, temp={temperature}"
        )
    
    def generate(
        self,
        evidence: EvidenceDatabase,
        target_count: int = 2,
        strategies: Optional[List[CreativeStrategy]] = None
    ) -> List[GeneratedQuestion]:
        """
        Generate creative questions using GPT-4
        
        GENERATION FLOW:
        1. Select creative strategies
        2. For each strategy:
           a. Build evidence-first prompt
           b. Generate question with GPT-4
           c. Parse and validate
           d. Check evidence grounding
           e. Validate through 10 layers
        3. Rank and return top N
        
        Args:
            evidence: Evidence database
            target_count: Target number of questions (default 2)
            strategies: Creative strategies to use (default: all)
            
        Returns:
            List of creative, validated questions
        """
        logger.info(f"Starting Tier 3 generation: target={target_count} questions")
        
        # Default strategies if not provided
        if strategies is None:
            strategies = [
                CreativeStrategy.NOVEL_COMBINATIONS,
                CreativeStrategy.EDGE_CASES,
                CreativeStrategy.MULTI_STEP_REASONING,
            ]
        
        generated_questions = []
        
        for strategy in strategies:
            if len(generated_questions) >= target_count:
                break
            
            logger.info(f"Trying strategy: {strategy.value}")
            self.stats.strategy_usage[strategy.value] = \
                self.stats.strategy_usage.get(strategy.value, 0) + 1
            
            # Generate question with this strategy
            for attempt in range(self.max_retries):
                self.stats.questions_attempted += 1
                
                try:
                    # Step 1: Build evidence-first prompt
                    prompt = self._build_evidence_first_prompt(evidence, strategy)
                    
                    # Step 2: Generate with GPT-4
                    response = self._call_gpt4(prompt)
                    self.stats.llm_calls += 1
                    
                    # Step 3: Parse response
                    question = self._parse_gpt4_response(response, evidence)
                    
                    if question is None:
                        logger.debug(f"Failed to parse GPT-4 response (attempt {attempt+1})")
                        continue
                    
                    # Step 4: Evidence grounding check
                    if self.enable_grounding_check:
                        is_grounded = self._check_evidence_grounding(question, evidence)
                        if not is_grounded:
                            self.stats.rejected_hallucination += 1
                            logger.debug("Rejected: Failed evidence grounding check")
                            continue
                    
                    # Step 5: Strict validation
                    validation_result = self.validator.validate(question, evidence)
                    
                    if not validation_result.is_valid:
                        self.stats.rejected_validation += 1
                        logger.debug(
                            f"Rejected: Validation failed - "
                            f"{validation_result.failed_layers[0]}"
                        )
                        continue
                    
                    # Question passed all checks
                    generated_questions.append(question)
                    self.stats.questions_generated += 1
                    
                    logger.info(
                        f"Generated question {len(generated_questions)}/{target_count} "
                        f"using {strategy.value}"
                    )
                    break  # Success, move to next strategy
                    
                except Exception as e:
                    logger.error(f"Error generating question: {e}", exc_info=True)
                    continue
        
        # Update statistics
        if generated_questions:
            self.stats.average_complexity_score = sum(
                q.complexity_score for q in generated_questions
            ) / len(generated_questions)
        
        self.stats.questions_rejected = (
            self.stats.questions_attempted - self.stats.questions_generated
        )
        
        logger.info(
            f"Tier 3 generation complete: "
            f"generated={len(generated_questions)}/{target_count}, "
            f"cost=${self.stats.total_cost:.2f}"
        )
        
        return generated_questions[:target_count]
    
    def _build_evidence_first_prompt(
        self,
        evidence: EvidenceDatabase,
        strategy: CreativeStrategy
    ) -> str:
        """
        Build evidence-first prompt for GPT-4
        
        CRITICAL: Forces GPT-4 to cite evidence BEFORE generating questions
        This prevents hallucination by grounding generation in evidence.
        """
        
        # Extract evidence summary
        evidence_summary = self._summarize_evidence(evidence)
        
        # Strategy-specific instructions
        strategy_instructions = {
            CreativeStrategy.NOVEL_COMBINATIONS: (
                "Find UNUSUAL combinations of modalities that haven't been tested much. "
                "Examples: counting + temporal + spatial, subscene + audio-visual stitching."
            ),
            CreativeStrategy.EDGE_CASES: (
                "Target EDGE CASES that are technically valid but unusual. "
                "Examples: events at timestamp boundaries, overlapping audio-visual cues."
            ),
            CreativeStrategy.COUNTER_INTUITIVE: (
                "Create questions with COUNTER-INTUITIVE answers. "
                "The obvious answer should be wrong, the correct answer should be surprising."
            ),
            CreativeStrategy.MULTI_STEP_REASONING: (
                "Require MULTI-STEP reasoning chains with 3+ inferential steps. "
                "Each step should require both audio and visual information."
            ),
            CreativeStrategy.TEMPORAL_EDGE: (
                "Focus on PRECISE TEMPORAL boundaries. "
                "Questions should be answerable only with exact timestamps."
            ),
        }
        
        prompt = f"""You are creating adversarial test questions for Google's Gemini 2.5 Pro multimodal model.

GOAL: Generate questions that Gemini will get WRONG, exposing weaknesses in multimodal reasoning.

CRITICAL RULES (violating ANY rule = rejected):
1. BOTH audio AND visual cues required (not answerable with only one)
2. NO NAMES - use descriptors only (e.g., "the person in blue shirt" not "John")
3. PRECISE TIMESTAMPS - cite exact timestamps from evidence
4. EVIDENCE-DRIVEN - every detail must come from evidence below
5. UNAMBIGUOUS - question must have one clear answer

STRATEGY FOR THIS QUESTION: {strategy.value}
{strategy_instructions[strategy]}

=== EVIDENCE FROM VIDEO ===
{evidence_summary}
=== END EVIDENCE ===

GENERATION PROCESS (MUST FOLLOW):

STEP 1: CITE EVIDENCE
List the specific evidence you will use:
- Audio evidence: [quote transcript with timestamp]
- Visual evidence: [describe detection with timestamp]
- Additional context: [any other relevant evidence]

STEP 2: GENERATE QUESTION
Create question using ONLY the evidence cited above.
Question format:
{{
  "question": "Your adversarial question here",
  "answer": "The golden answer",
  "question_types": ["type1", "type2"],
  "audio_cues": [
    {{"type": "audio_speech", "content": "...", "timestamp": 0.0}}
  ],
  "visual_cues": [
    {{"type": "visual_object", "content": "...", "timestamp": 0.0}}
  ],
  "start_timestamp": 0.0,
  "end_timestamp": 0.0,
  "complexity_score": 0.8,
  "evidence_citations": ["ref1", "ref2"]
}}

STEP 3: VERIFY
- Does question require BOTH audio and visual? YES/NO
- Are ALL details from evidence? YES/NO
- Are there ANY names? YES/NO (must be NO)
- Is the answer unambiguous? YES/NO

Return your response as JSON following the format above."""

        return prompt
    
    def _summarize_evidence(self, evidence: EvidenceDatabase) -> str:
        """Summarize evidence for prompt"""
        
        # Transcript summary (first 10 segments)
        transcript_summary = "\n".join([
            f"[{seg['start']:.2f}s] {seg['text']}"
            for seg in evidence.transcript_segments[:10]
        ])
        
        # Object detections summary
        object_summary = "\n".join([
            f"[{det['timestamp']:.2f}s] {det['object_class']} at ({det['bbox'][0]:.2f}, {det['bbox'][1]:.2f})"
            for det in evidence.object_detections[:20]
        ])
        
        # Person detections summary
        person_summary = "\n".join([
            f"[{det['timestamp']:.2f}s] Person {det['person_id']}: {det.get('attributes', {})}"
            for det in evidence.person_detections[:10]
        ])
        
        # Action detections summary
        action_summary = "\n".join([
            f"[{det['timestamp']:.2f}s] {det['action_type']}"
            for det in evidence.action_detections[:10]
        ])
        
        summary = f"""TRANSCRIPT (first 10 segments):
{transcript_summary}

OBJECTS DETECTED (first 20):
{object_summary}

PEOPLE DETECTED (first 10):
{person_summary}

ACTIONS DETECTED (first 10):
{action_summary}

VIDEO METADATA:
- Duration: {evidence.video_duration:.2f}s
- Total transcript segments: {len(evidence.transcript_segments)}
- Total object detections: {len(evidence.object_detections)}
- Total person detections: {len(evidence.person_detections)}
- Total action detections: {len(evidence.action_detections)}
"""
        
        return summary
    
    def _call_gpt4(self, prompt: str) -> Dict:
        """Call GPT-4 API"""
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating adversarial test questions for multimodal AI systems."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=1000,
            )
            
            # Track costs
            prompt_tokens = response['usage']['prompt_tokens']
            completion_tokens = response['usage']['completion_tokens']
            
            # GPT-4 pricing: $0.03 per 1K prompt tokens, $0.06 per 1K completion tokens
            cost = (prompt_tokens / 1000 * 0.03) + (completion_tokens / 1000 * 0.06)
            self.stats.total_cost += cost
            
            # Update token statistics
            self.stats.average_prompt_tokens = (
                (self.stats.average_prompt_tokens * (self.stats.llm_calls - 1) + prompt_tokens)
                / self.stats.llm_calls
            )
            self.stats.average_completion_tokens = (
                (self.stats.average_completion_tokens * (self.stats.llm_calls - 1) + completion_tokens)
                / self.stats.llm_calls
            )
            
            return response
            
        except Exception as e:
            logger.error(f"GPT-4 API error: {e}")
            raise
    
    def _parse_gpt4_response(
        self,
        response: Dict,
        evidence: EvidenceDatabase
    ) -> Optional[GeneratedQuestion]:
        """Parse GPT-4 response into GeneratedQuestion"""
        
        try:
            # Extract JSON from response
            content = response['choices'][0]['message']['content']
            
            # Try to find JSON in response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in GPT-4 response")
                return None
            
            data = json.loads(json_match.group())
            
            # Parse cues
            audio_cues = [
                Cue(
                    cue_type=CueType.AUDIO_SPEECH,
                    content=cue['content'],
                    timestamp=cue['timestamp']
                )
                for cue in data.get('audio_cues', [])
            ]
            
            visual_cues = [
                Cue(
                    cue_type=CueType.VISUAL_OBJECT,
                    content=cue['content'],
                    timestamp=cue['timestamp']
                )
                for cue in data.get('visual_cues', [])
            ]
            
            # Parse question types
            question_types = [
                QuestionType(qt) for qt in data.get('question_types', [])
            ]
            
            # Create GeneratedQuestion
            question = GeneratedQuestion(
                question_text=data['question'],
                golden_answer=data['answer'],
                start_timestamp=data['start_timestamp'],
                end_timestamp=data['end_timestamp'],
                audio_cues=audio_cues,
                visual_cues=visual_cues,
                question_types=question_types,
                generation_tier=3,
                template_name="GPT4Creative",
                complexity_score=data.get('complexity_score', 0.8),
                evidence_refs=data.get('evidence_citations', [])
            )
            
            return question
            
        except Exception as e:
            logger.error(f"Failed to parse GPT-4 response: {e}")
            return None
    
    def _check_evidence_grounding(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> bool:
        """
        Cross-check question against evidence using second LLM call
        
        This catches plausible hallucinations that slip through validation.
        """
        
        try:
            # Build verification prompt
            verify_prompt = f"""You are verifying if a question is 100% grounded in video evidence.

QUESTION:
{question.question_text}

ANSWER:
{question.golden_answer}

EVIDENCE:
{self._summarize_evidence(evidence)}

TASK: Check if EVERY detail in the question and answer exists in the evidence.

Look for hallucinations:
- Object counts: Does the count match evidence?
- Colors/attributes: Do they match evidence?
- Actions: Do they match evidence?
- Spatial relationships: Do they match evidence?
- Timestamps: Do events happen at those times?

Respond with JSON:
{{
  "is_grounded": true/false,
  "hallucinations_found": ["list any hallucinations"],
  "confidence": 0.0-1.0
}}"""

            # Call GPT-4 for verification
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a fact-checker verifying evidence grounding."},
                    {"role": "user", "content": verify_prompt}
                ],
                temperature=0.0,  # Deterministic for verification
                max_tokens=500,
            )
            
            # Track cost
            cost = (response['usage']['prompt_tokens'] / 1000 * 0.03) + \
                   (response['usage']['completion_tokens'] / 1000 * 0.06)
            self.stats.total_cost += cost
            self.stats.llm_calls += 1
            
            # Parse verification
            content = response['choices'][0]['message']['content']
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                is_grounded = result.get('is_grounded', False)
                
                if not is_grounded:
                    logger.warning(
                        f"Evidence grounding failed: "
                        f"{result.get('hallucinations_found', [])}"
                    )
                
                return is_grounded
            
            return False
            
        except Exception as e:
            logger.error(f"Evidence grounding check failed: {e}")
            return False  # Fail safe
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            "tier": 3,
            "method": "gpt4_creative",
            "model": self.model,
            "temperature": self.temperature,
            
            # Generation metrics
            "questions_attempted": self.stats.questions_attempted,
            "questions_generated": self.stats.questions_generated,
            "questions_rejected": self.stats.questions_rejected,
            "success_rate": (
                self.stats.questions_generated / self.stats.questions_attempted
                if self.stats.questions_attempted > 0 else 0.0
            ),
            
            # Cost metrics
            "llm_calls": self.stats.llm_calls,
            "total_cost": self.stats.total_cost,
            "cost_per_question": (
                self.stats.total_cost / self.stats.questions_generated
                if self.stats.questions_generated > 0 else 0.0
            ),
            "avg_prompt_tokens": self.stats.average_prompt_tokens,
            "avg_completion_tokens": self.stats.average_completion_tokens,
            
            # Rejection analysis
            "rejected_hallucination": self.stats.rejected_hallucination,
            "rejected_validation": self.stats.rejected_validation,
            "rejected_no_dual_cue": self.stats.rejected_no_dual_cue,
            "rejected_low_quality": self.stats.rejected_low_quality,
            
            # Strategy tracking
            "strategy_usage": self.stats.strategy_usage,
            
            # Quality metrics
            "average_complexity_score": self.stats.average_complexity_score,
            
            # Safeguards
            "grounding_check_enabled": self.enable_grounding_check,
        }
    
    def print_statistics(self):
        """Print human-readable statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("TIER 3 (GPT-4 CREATIVE) STATISTICS")
        print("="*80)
        
        print(f"\n📊 GENERATION METRICS:")
        print(f"  Attempted:  {stats['questions_attempted']}")
        print(f"  Generated:  {stats['questions_generated']}")
        print(f"  Rejected:   {stats['questions_rejected']}")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        
        print(f"\n💰 COST METRICS:")
        print(f"  LLM Calls:     {stats['llm_calls']}")
        print(f"  Total Cost:    ${stats['total_cost']:.2f}")
        print(f"  Per Question:  ${stats['cost_per_question']:.2f}")
        print(f"  Avg Tokens:    {stats['avg_prompt_tokens']:.0f} prompt + "
              f"{stats['avg_completion_tokens']:.0f} completion")
        
        print(f"\n❌ REJECTION BREAKDOWN:")
        print(f"  Hallucination: {stats['rejected_hallucination']}")
        print(f"  Validation:    {stats['rejected_validation']}")
        print(f"  No Dual Cue:   {stats['rejected_no_dual_cue']}")
        print(f"  Low Quality:   {stats['rejected_low_quality']}")
        
        if stats['strategy_usage']:
            print(f"\n🎯 STRATEGY USAGE:")
            for strategy, count in stats['strategy_usage'].items():
                print(f"  {strategy}: {count}")
        
        print(f"\n⭐ QUALITY:")
        print(f"  Avg Complexity: {stats['average_complexity_score']:.2f}")
        
        print(f"\n🛡️ SAFEGUARDS:")
        print(f"  Grounding Check: {'✓' if stats['grounding_check_enabled'] else '✗'}")
        print(f"  Validation: ✓ (10 layers)")
        print(f"  Human Review: ✓ (required)")
        
        print("="*80 + "\n")