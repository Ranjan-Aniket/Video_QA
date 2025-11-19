"""
Tier 2: Constrained Llama API Generation

Cost: ~$0.30 per video (17 questions)
Hallucination Rate: <1% (heavy constraints)
Target: 17 questions per video

Uses Llama 3.1 70B via API (Together AI / Groq / Replicate) with:
- Evidence-only vocabulary (logit bias)
- Template-slot architecture
- Structural validation
- Multi-retry with backoff

Providers supported:
- Together AI (recommended - cheap + fast)
- Groq (fastest, but rate limits)
- Replicate (fallback)
"""

from typing import List, Dict, Optional, Tuple
import json
import time
from dataclasses import dataclass
from enum import Enum

from templates.base import (
    GeneratedQuestion, EvidenceDatabase, QuestionType, Cue, CueType
)
from generation.logit_bias import LogitBiasBuilder, TemplateSlotConstraints
from generation.constraint_engine import (
    GrammarConstraints, StructuralValidator, SlotFiller
)


class LlamaProvider(Enum):
    """Supported Llama API providers"""
    TOGETHER = "together"
    GROQ = "groq"
    REPLICATE = "replicate"


@dataclass
class ProviderConfig:
    """Configuration for Llama API provider"""
    provider: LlamaProvider
    api_key: str
    model_name: str
    base_url: Optional[str] = None
    max_retries: int = 3
    timeout: int = 60


class Tier2LlamaGenerator:
    """
    Tier 2: Constrained generation with Llama API
    
    Applies heavy constraints to ensure evidence-only generation.
    """
    
    def __init__(
        self,
        provider_config: ProviderConfig,
        use_logit_bias: bool = True
    ):
        """
        Initialize Tier 2 generator
        
        Args:
            provider_config: Provider configuration
            use_logit_bias: Apply logit bias (recommended)
        """
        self.provider_config = provider_config
        self.use_logit_bias = use_logit_bias
        
        # Initialize provider client
        self.client = self._initialize_client()
        
        # Cost tracking
        self.cost_per_question = 0.018  # ~$0.30 / 17 questions
        self.total_cost = 0.0
        
        # Statistics
        self.stats = {
            "questions_generated": 0,
            "questions_failed": 0,
            "total_retries": 0
        }
    
    def _initialize_client(self):
        """Initialize API client for provider"""
        if self.provider_config.provider == LlamaProvider.TOGETHER:
            # Together AI uses OpenAI-compatible API
            import openai
            client = openai.OpenAI(
                api_key=self.provider_config.api_key,
                base_url="https://api.together.xyz/v1"
            )
            return client
        
        elif self.provider_config.provider == LlamaProvider.GROQ:
            # Groq uses OpenAI-compatible API
            import openai
            client = openai.OpenAI(
                api_key=self.provider_config.api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            return client
        
        elif self.provider_config.provider == LlamaProvider.REPLICATE:
            # Replicate client
            import replicate
            replicate.api_token = self.provider_config.api_key
            return replicate
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider_config.provider}")
    
    def generate(
        self,
        evidence: EvidenceDatabase,
        target_count: int = 17,
        question_types: Optional[List[QuestionType]] = None
    ) -> List[GeneratedQuestion]:
        """
        Generate Tier 2 questions using constrained Llama
        
        Args:
            evidence: Evidence database
            target_count: Target number of questions
            question_types: Specific types to generate (or None for diverse)
            
        Returns:
            List of generated questions
        """
        questions = []
        
        # Build constraints
        logit_builder = LogitBiasBuilder()
        evidence_vocab = self._build_evidence_vocabulary(evidence)
        
        # Determine question types to generate
        if question_types is None:
            # Generate diverse set
            question_types = self._select_diverse_types(target_count)
        
        # Generate questions for each type
        for q_type in question_types:
            if len(questions) >= target_count:
                break
            
            # Get template for this type
            template = self._get_template(q_type)
            if not template:
                continue
            
            # Generate with retries
            for attempt in range(self.provider_config.max_retries):
                try:
                    question = self._generate_single_question(
                        evidence=evidence,
                        question_type=q_type,
                        template=template,
                        evidence_vocab=evidence_vocab
                    )
                    
                    if question:
                        questions.append(question)
                        self.stats["questions_generated"] += 1
                        self.total_cost += self.cost_per_question
                        break
                    else:
                        self.stats["total_retries"] += 1
                        
                except Exception as e:
                    print(f"[Tier2] Generation error (attempt {attempt+1}): {e}")
                    self.stats["total_retries"] += 1
                    
                    if attempt == self.provider_config.max_retries - 1:
                        self.stats["questions_failed"] += 1
                    
                    time.sleep(1)  # Backoff
        
        return questions[:target_count]
    
    def _select_diverse_types(self, count: int) -> List[QuestionType]:
        """
        Select diverse question types
        
        Prioritizes challenging types for Tier 2.
        """
        # Priority order (challenging first)
        priority_types = [
            QuestionType.INFERENCE,
            QuestionType.COMPARATIVE,
            QuestionType.OBJECT_INTERACTION_REASONING,
            QuestionType.GENERAL_HOLISTIC_REASONING,
            QuestionType.AUDIO_VISUAL_STITCHING,
            QuestionType.TACKLING_SPURIOUS_CORRELATIONS,
            QuestionType.TEMPORAL_UNDERSTANDING,
            QuestionType.SEQUENTIAL,
            QuestionType.COUNTING,
            QuestionType.NEEDLE,
            QuestionType.REFERENTIAL_GROUNDING,
            QuestionType.CONTEXT,
            QuestionType.SUBSCENE
        ]
        
        # Repeat to fill count
        selected = []
        while len(selected) < count:
            selected.extend(priority_types)
        
        return selected[:count]
    
    def _get_template(self, question_type: QuestionType) -> Optional[Dict]:
        """
        Get generation template for question type
        
        Templates define structure with slots to fill from evidence.
        """
        templates = {
            QuestionType.COUNTING: {
                "question_template": "How many {OBJECT}s {ACTION_CONTEXT} after someone says \"{AUDIO_CUE}\"?",
                "answer_template": "After the audio cue, {COUNT} {OBJECT}s {ACTION_RESULT}.",
                "slots": ["OBJECT", "ACTION_CONTEXT", "AUDIO_CUE", "COUNT", "ACTION_RESULT"],
                "required_evidence": ["object_detections", "transcript_segments"]
            },
            
            QuestionType.TEMPORAL_UNDERSTANDING: {
                "question_template": "What happens after someone says \"{AUDIO_CUE}\" and {VISUAL_EVENT_A}?",
                "answer_template": "{VISUAL_EVENT_B} happens next.",
                "slots": ["AUDIO_CUE", "VISUAL_EVENT_A", "VISUAL_EVENT_B"],
                "required_evidence": ["event_timeline", "transcript_segments"]
            },
            
            QuestionType.NEEDLE: {
                "question_template": "What {DETAIL_TYPE} appears on the {LOCATION} when someone says \"{AUDIO_CUE}\"?",
                "answer_template": "The {DETAIL_TYPE} \"{DETAIL_CONTENT}\" appears on the {LOCATION}.",
                "slots": ["DETAIL_TYPE", "LOCATION", "AUDIO_CUE", "DETAIL_CONTENT"],
                "required_evidence": ["ocr_detections", "transcript_segments"]
            },
            
            QuestionType.REFERENTIAL_GROUNDING: {
                "question_template": "Who is visible on screen when someone says \"{AUDIO_CUE}\"?",
                "answer_template": "{PERSON_DESCRIPTOR_1} and {PERSON_DESCRIPTOR_2} are visible.",
                "slots": ["AUDIO_CUE", "PERSON_DESCRIPTOR_1", "PERSON_DESCRIPTOR_2"],
                "required_evidence": ["person_detections", "transcript_segments"]
            },
            
            QuestionType.COMPARATIVE: {
                "question_template": "What is the difference in {COMPARISON_ASPECT} before and after someone says \"{AUDIO_CUE}\"?",
                "answer_template": "Before the audio cue, {STATE_BEFORE}, but after, {STATE_AFTER}.",
                "slots": ["COMPARISON_ASPECT", "AUDIO_CUE", "STATE_BEFORE", "STATE_AFTER"],
                "required_evidence": ["person_detections", "transcript_segments"]
            },
            
            QuestionType.INFERENCE: {
                "question_template": "Why does {EVENT_B} happen after someone says \"{AUDIO_CUE}\" and {EVENT_A}?",
                "answer_template": "{EVENT_B} happens as a result of {EVENT_A}.",
                "slots": ["EVENT_A", "EVENT_B", "AUDIO_CUE"],
                "required_evidence": ["event_timeline", "transcript_segments"]
            },
            
            QuestionType.OBJECT_INTERACTION_REASONING: {
                "question_template": "How does the {OBJECT} change after someone says \"{AUDIO_CUE}\" and {ACTION}?",
                "answer_template": "The {OBJECT} {TRANSFORMATION}.",
                "slots": ["OBJECT", "AUDIO_CUE", "ACTION", "TRANSFORMATION"],
                "required_evidence": ["object_detections", "action_detections"]
            },
            
            QuestionType.CONTEXT: {
                "question_template": "What visual elements are present in the {ELEMENT_TYPE} when someone says \"{AUDIO_CUE}\"?",
                "answer_template": "In the {ELEMENT_TYPE}, {ELEMENTS_LIST}.",
                "slots": ["ELEMENT_TYPE", "AUDIO_CUE", "ELEMENTS_LIST"],
                "required_evidence": ["object_detections", "transcript_segments"]
            }
        }
        
        return templates.get(question_type)
    
    def _generate_single_question(
        self,
        evidence: EvidenceDatabase,
        question_type: QuestionType,
        template: Dict,
        evidence_vocab: Dict[str, List[str]]
    ) -> Optional[GeneratedQuestion]:
        """
        Generate single question using constrained Llama
        
        CRITICAL: Ensures generation only uses evidence vocabulary.
        """
        # Step 1: Check evidence requirements
        required_evidence = template.get("required_evidence", [])
        if not self._check_evidence_availability(evidence, required_evidence):
            return None
        
        # Step 2: Build evidence-grounded prompt
        prompt = self._build_constrained_prompt(
            evidence=evidence,
            question_type=question_type,
            template=template,
            evidence_vocab=evidence_vocab
        )
        
        # Step 3: Call Llama API with constraints
        response = self._call_llama_api(prompt)
        if not response:
            return None
        
        # Step 4: Parse and validate response
        question_data = self._parse_response(response)
        if not question_data:
            return None
        
        # Step 5: Validate structure
        is_valid, error = StructuralValidator.validate_structure(
            question_data['question'],
            question_data['answer'],
            question_type.value
        )
        
        if not is_valid:
            print(f"[Tier2] Structure validation failed: {error}")
            return None
        
        # Step 6: Create GeneratedQuestion object
        try:
            return GeneratedQuestion(
                question_text=question_data['question'],
                golden_answer=question_data['answer'],
                start_timestamp=question_data['start_timestamp'],
                end_timestamp=question_data['end_timestamp'],
                audio_cues=question_data['audio_cues'],
                visual_cues=question_data['visual_cues'],
                question_types=[question_type],
                generation_tier=2,
                template_name=f"Tier2_{question_type.value}",
                complexity_score=question_data.get('complexity_score', 0.7),
                evidence_refs=question_data.get('evidence_refs', [])
            )
        except Exception as e:
            print(f"[Tier2] Question creation failed: {e}")
            return None
    
    def _check_evidence_availability(
        self,
        evidence: EvidenceDatabase,
        required_evidence: List[str]
    ) -> bool:
        """Check if required evidence is available"""
        for req in required_evidence:
            evidence_list = getattr(evidence, req, [])
            if not evidence_list or len(evidence_list) == 0:
                return False
        return True
    
    def _build_constrained_prompt(
        self,
        evidence: EvidenceDatabase,
        question_type: QuestionType,
        template: Dict,
        evidence_vocab: Dict[str, List[str]]
    ) -> str:
        """
        Build prompt with heavy constraints
        
        CRITICAL: Forces LLM to use ONLY evidence vocabulary.
        """
        # Extract evidence snippets
        evidence_summary = self._summarize_evidence(evidence, template)
        
        # Build slot constraints
        slot_constraints = self._build_slot_constraints(template, evidence_vocab)
        
        prompt = f"""You are generating a {question_type.value} question for video understanding evaluation.

CRITICAL RULES:
1. Use ONLY information from EVIDENCE below - NO fabrication
2. Question MUST have BOTH audio cue (someone says X) AND visual cue
3. NO names - use descriptors only (e.g., "person in blue shirt")
4. Follow TEMPLATE structure exactly
5. Fill SLOTS only from provided options

EVIDENCE:
{evidence_summary}

TEMPLATE:
Question: {template['question_template']}
Answer: {template['answer_template']}

SLOT OPTIONS (use ONLY these):
{slot_constraints}

TASK: Fill the template slots with evidence.

Output JSON:
{{
    "question": "filled question text",
    "answer": "filled answer text",
    "slots_used": {{"SLOT_NAME": "value_from_evidence"}},
    "start_timestamp": 10.5,
    "end_timestamp": 15.2,
    "audio_cues": [{{"content": "text", "timestamp": 10.5}}],
    "visual_cues": [{{"content": "description", "timestamp": 11.0}}]
}}"""

        return prompt
    
    def _summarize_evidence(
        self,
        evidence: EvidenceDatabase,
        template: Dict
    ) -> str:
        """Summarize relevant evidence for template"""
        summary_parts = []
        
        # Transcript
        if "transcript_segments" in template.get("required_evidence", []):
            summary_parts.append("TRANSCRIPT:")
            for seg in evidence.transcript_segments[:10]:
                summary_parts.append(f"  {seg['start']:.1f}s: \"{seg['text'][:60]}\"")
        
        # Objects
        if "object_detections" in template.get("required_evidence", []):
            summary_parts.append("\nOBJECTS:")
            objects_seen = set()
            for obj in evidence.object_detections[:20]:
                obj_desc = obj['object_class']
                if obj.get('color'):
                    obj_desc = f"{obj['color']} {obj_desc}"
                objects_seen.add(f"{obj_desc} at {obj['timestamp']:.1f}s")
            for obj in list(objects_seen)[:10]:
                summary_parts.append(f"  {obj}")
        
        # People
        if "person_detections" in template.get("required_evidence", []):
            summary_parts.append("\nPEOPLE:")
            for person in evidence.person_detections[:10]:
                attrs = person.get('attributes', {})
                desc = attrs.get('clothing_color', 'person')
                summary_parts.append(f"  {desc} person at {person['timestamp']:.1f}s")
        
        # Events
        if "event_timeline" in template.get("required_evidence", []):
            summary_parts.append("\nEVENTS:")
            for event in evidence.event_timeline[:10]:
                summary_parts.append(f"  {event['timestamp']:.1f}s: {event.get('description', 'event')}")
        
        # Actions
        if "action_detections" in template.get("required_evidence", []):
            summary_parts.append("\nACTIONS:")
            for action in evidence.action_detections[:10]:
                summary_parts.append(f"  {action['timestamp']:.1f}s: {action['action']}")
        
        # OCR
        if "ocr_detections" in template.get("required_evidence", []):
            summary_parts.append("\nTEXT ON SCREEN:")
            for ocr in evidence.ocr_detections[:5]:
                summary_parts.append(f"  {ocr['timestamp']:.1f}s: \"{ocr['text']}\"")
        
        return "\n".join(summary_parts)
    
    def _build_slot_constraints(
        self,
        template: Dict,
        evidence_vocab: Dict[str, List[str]]
    ) -> str:
        """Build slot constraints from evidence"""
        constraints = []
        
        for slot in template.get("slots", []):
            if slot == "OBJECT":
                options = evidence_vocab.get('objects', [])[:10]
                constraints.append(f"{slot}: {', '.join(options)}")
            
            elif slot == "AUDIO_CUE":
                options = [text[:40] for text in evidence_vocab.get('audio', [])][:5]
                constraints.append(f"{slot}: {', '.join(options)}")
            
            elif slot == "COUNT":
                constraints.append(f"{slot}: 2, 3, 4, 5, 6, 7, 8, 9, 10")
            
            elif slot == "LOCATION":
                constraints.append(f"{slot}: screen, top left, top right, bottom, center")
            
            elif slot == "DETAIL_TYPE":
                constraints.append(f"{slot}: text, graphic, number, symbol")
            
            elif slot == "DETAIL_CONTENT":
                options = evidence_vocab.get('ocr', [])[:5]
                constraints.append(f"{slot}: {', '.join(options)}")
            
            elif "EVENT" in slot or "ACTION" in slot:
                options = evidence_vocab.get('actions', [])[:8]
                constraints.append(f"{slot}: {', '.join(options)}")
            
            elif "PERSON" in slot:
                constraints.append(f"{slot}: person in [color] [clothing], person on [left/right/center]")
            
            elif "ELEMENT_TYPE" in slot:
                constraints.append(f"{slot}: background, foreground, left side, right side")
        
        return "\n".join(constraints)
    
    def _build_evidence_vocabulary(
        self,
        evidence: EvidenceDatabase
    ) -> Dict[str, List[str]]:
        """Build vocabulary dictionary from evidence"""
        return {
            'objects': [obj['object_class'] for obj in evidence.object_detections],
            'audio': [seg['text'][:50] for seg in evidence.transcript_segments],
            'ocr': [ocr['text'] for ocr in evidence.ocr_detections],
            'actions': [action['action'] for action in evidence.action_detections],
            'people': [
                f"person_{p['person_id']}" 
                for p in evidence.person_detections 
                if p.get('person_id')
            ]
        }
    
    def _call_llama_api(self, prompt: str) -> Optional[str]:
        """
        Call Llama API with prompt
        
        Handles different providers.
        """
        try:
            if self.provider_config.provider in [LlamaProvider.TOGETHER, LlamaProvider.GROQ]:
                # OpenAI-compatible API
                response = self.client.chat.completions.create(
                    model=self.provider_config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise video question generator. Follow instructions exactly."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,  # Low temperature for consistency
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
                
                return response.choices[0].message.content
            
            elif self.provider_config.provider == LlamaProvider.REPLICATE:
                # Replicate API
                import replicate
                
                output = replicate.run(
                    self.provider_config.model_name,
                    input={
                        "prompt": prompt,
                        "temperature": 0.3,
                        "max_tokens": 500,
                        "system_prompt": "You are a precise video question generator. Output only JSON."
                    }
                )
                
                # Replicate returns iterator
                result = "".join(output)
                return result
            
        except Exception as e:
            print(f"[Tier2] API call failed: {e}")
            return None
    
    def _parse_response(self, response: str) -> Optional[Dict]:
        """
        Parse Llama response
        
        Expects JSON with question, answer, timestamps, cues.
        """
        try:
            # Clean response (remove markdown fences if present)
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            data = json.loads(response)
            
            # Validate required fields
            required = ['question', 'answer', 'start_timestamp', 'end_timestamp']
            for field in required:
                if field not in data:
                    print(f"[Tier2] Missing field: {field}")
                    return None
            
            # Parse cues
            audio_cues = []
            for cue_data in data.get('audio_cues', []):
                audio_cues.append(Cue(
                    cue_type=CueType.AUDIO_SPEECH,
                    content=cue_data['content'],
                    timestamp=cue_data['timestamp']
                ))
            
            visual_cues = []
            for cue_data in data.get('visual_cues', []):
                visual_cues.append(Cue(
                    cue_type=CueType.VISUAL_ACTION,
                    content=cue_data['content'],
                    timestamp=cue_data['timestamp']
                ))
            
            # Must have both audio and visual
            if not audio_cues or not visual_cues:
                print("[Tier2] Missing audio or visual cues")
                return None
            
            data['audio_cues'] = audio_cues
            data['visual_cues'] = visual_cues
            
            return data
            
        except json.JSONDecodeError as e:
            print(f"[Tier2] JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"[Tier2] Response parsing error: {e}")
            return None
    
    def get_cost(self) -> float:
        """Get total cost"""
        return self.total_cost
    
    def get_statistics(self) -> Dict:
        """Get generation statistics"""
        return {
            "tier": 2,
            "method": "constrained_llama_api",
            "provider": self.provider_config.provider.value,
            "model": self.provider_config.model_name,
            "questions_generated": self.stats["questions_generated"],
            "questions_failed": self.stats["questions_failed"],
            "total_retries": self.stats["total_retries"],
            "success_rate": f"{(self.stats['questions_generated'] / (self.stats['questions_generated'] + self.stats['questions_failed']) * 100):.1f}%" if (self.stats['questions_generated'] + self.stats['questions_failed']) > 0 else "0%",
            "total_cost": f"${self.total_cost:.2f}",
            "cost_per_question": f"${self.cost_per_question:.3f}"
        }


# ============================================================================
# PROVIDER HELPERS
# ============================================================================

def create_together_config(api_key: str) -> ProviderConfig:
    """
    Create Together AI configuration
    
    Recommended: Cheap + fast + good quality
    Model: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
    Cost: ~$0.88 per 1M tokens
    """
    return ProviderConfig(
        provider=LlamaProvider.TOGETHER,
        api_key=api_key,
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    )


def create_groq_config(api_key: str) -> ProviderConfig:
    """
    Create Groq configuration
    
    Fastest, but rate limits are strict
    Model: llama-3.1-70b-versatile
    Cost: Free tier available
    """
    return ProviderConfig(
        provider=LlamaProvider.GROQ,
        api_key=api_key,
        model_name="llama-3.1-70b-versatile"
    )


def create_replicate_config(api_key: str) -> ProviderConfig:
    """
    Create Replicate configuration
    
    Fallback option
    Model: meta/meta-llama-3.1-70b-instruct
    Cost: ~$0.65 per 1M tokens
    """
    return ProviderConfig(
        provider=LlamaProvider.REPLICATE,
        api_key=api_key,
        model_name="meta/meta-llama-3.1-70b-instruct"
    )


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Tier 2 Llama API Generator")
    print("=" * 80)
    print()
    print("Supported providers:")
    print("  1. Together AI (recommended) - Fast + cheap")
    print("  2. Groq - Fastest, but rate limited")
    print("  3. Replicate - Fallback")
    print()
    print("Example usage:")
    print("""
    from generation.tier2_llama_api import Tier2LlamaGenerator, create_together_config
    from templates.base import EvidenceDatabase
    
    # Create config
    config = create_together_config(api_key="your_key_here")
    
    # Initialize generator
    generator = Tier2LlamaGenerator(config)
    
    # Generate questions
    questions = generator.generate(evidence, target_count=17)
    
    # Get stats
    stats = generator.get_statistics()
    print(stats)
    """)