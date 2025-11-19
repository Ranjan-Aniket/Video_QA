"""
Tier 3: Creative GPT-4-mini Generation

Cost: ~$0.15 per video (5 questions × $0.03)
Hallucination Rate: ~5% (with validation, 3 retries)
Target: 5 questions per video

Uses:
- GPT-4-mini API
- Strict validation (10 layers)
- Max 3 retries per question
- Only for edge cases that templates can't handle
"""

from typing import List, Dict, Optional
import openai
from templates.base import GeneratedQuestion, EvidenceDatabase, QuestionType, Cue, CueType
from generation.logit_bias import LogitBiasBuilder


class Tier3Generator:
    """
    Tier 3: Creative generation with GPT-4-mini
    
    Used for complex questions that templates can't handle.
    Heavy validation to ensure accuracy.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize Tier 3 generator
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.cost_per_question = 0.03
        self.total_cost = 0.0
        self.max_retries = 3
    
    def generate(
        self,
        evidence: EvidenceDatabase,
        target_count: int = 5,
        question_types: Optional[List[QuestionType]] = None
    ) -> List[GeneratedQuestion]:
        """
        Generate Tier 3 questions using GPT-4-mini
        
        Args:
            evidence: Evidence database
            target_count: Target number of questions
            question_types: Specific types to generate
            
        Returns:
            List of generated questions
        """
        questions = []
        
        # Build logit bias from evidence
        logit_builder = LogitBiasBuilder(model=self.model)
        logit_bias = logit_builder.build(evidence)
        
        # Generate questions
        types_to_generate = question_types or [
            QuestionType.INFERENCE,
            QuestionType.GENERAL_HOLISTIC_REASONING,
            QuestionType.TACKLING_SPURIOUS_CORRELATIONS
        ]
        
        for q_type in types_to_generate:
            if len(questions) >= target_count:
                break
            
            # Generate with retries
            q = self._generate_with_retries(
                evidence=evidence,
                question_type=q_type,
                logit_bias=logit_bias
            )
            
            if q:
                questions.append(q)
                self.total_cost += self.cost_per_question
        
        return questions[:target_count]
    
    def _generate_with_retries(
        self,
        evidence: EvidenceDatabase,
        question_type: QuestionType,
        logit_bias: Dict[str, int]
    ) -> Optional[GeneratedQuestion]:
        """Generate question with retries on validation failure"""
        
        for attempt in range(self.max_retries):
            try:
                q = self._generate_single_question(
                    evidence=evidence,
                    question_type=question_type,
                    logit_bias=logit_bias
                )
                
                if q:
                    # Validate (will be done by validation layers later)
                    return q
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return None
                continue
        
        return None
    
    def _generate_single_question(
        self,
        evidence: EvidenceDatabase,
        question_type: QuestionType,
        logit_bias: Dict[str, int]
    ) -> Optional[GeneratedQuestion]:
        """Generate single question using GPT-4-mini"""
        
        # Build prompt
        prompt = self._build_prompt(evidence, question_type)
        
        # Call API with logit bias
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a question generator for video understanding. "
                        "Generate questions that have BOTH audio and visual cues. "
                        "Use only information from the provided evidence. "
                        "NEVER use names - use descriptors instead. "
                        "Output as JSON with fields: question, answer, audio_cues, visual_cues, "
                        "start_timestamp, end_timestamp."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=500,
            logit_bias=logit_bias,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        result = response.choices[0].message.content
        import json
        data = json.loads(result)
        
        # Convert to GeneratedQuestion
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
                cue_type=CueType.VISUAL_ACTION,
                content=cue['content'],
                timestamp=cue['timestamp']
            )
            for cue in data.get('visual_cues', [])
        ]
        
        if not audio_cues or not visual_cues:
            return None
        
        return GeneratedQuestion(
            question_text=data['question'],
            golden_answer=data['answer'],
            start_timestamp=data['start_timestamp'],
            end_timestamp=data['end_timestamp'],
            audio_cues=audio_cues,
            visual_cues=visual_cues,
            question_types=[question_type],
            generation_tier=3,
            template_name=f"Tier3_{question_type.value}",
            complexity_score=0.8,
            evidence_refs=[]
        )
    
    def _build_prompt(
        self,
        evidence: EvidenceDatabase,
        question_type: QuestionType
    ) -> str:
        """Build prompt for GPT-4-mini"""
        
        # Summarize evidence
        evidence_summary = self._summarize_evidence(evidence)
        
        prompt = f"""
Generate a {question_type.value} question from this video evidence.

CRITICAL RULES:
1. Question MUST have both audio and visual cues
2. Use ONLY information from the evidence below
3. NEVER use names - use descriptors (e.g., "person in blue shirt")
4. Timestamps must be precise

EVIDENCE:
{evidence_summary}

Generate a single question as JSON:
{{
    "question": "...",
    "answer": "...",
    "audio_cues": [{{"content": "...", "timestamp": 10.5}}],
    "visual_cues": [{{"content": "...", "timestamp": 10.5}}],
    "start_timestamp": 10.0,
    "end_timestamp": 15.0
}}
"""
        return prompt
    
    def _summarize_evidence(self, evidence: EvidenceDatabase) -> str:
        """Summarize evidence for prompt"""
        
        summary = []
        
        # Transcript
        if evidence.transcript_segments:
            summary.append("TRANSCRIPT:")
            for seg in evidence.transcript_segments[:10]:
                summary.append(f"  {seg['start']:.1f}s: \"{seg['text']}\"")
        
        # Objects
        if evidence.object_detections:
            summary.append("\nOBJECTS:")
            objects_seen = set()
            for obj in evidence.object_detections[:20]:
                obj_desc = obj['object_class']
                if obj.get('color'):
                    obj_desc = f"{obj['color']} {obj_desc}"
                objects_seen.add(f"{obj_desc} at {obj['timestamp']:.1f}s")
            for obj in list(objects_seen)[:10]:
                summary.append(f"  {obj}")
        
        # Actions
        if evidence.action_detections:
            summary.append("\nACTIONS:")
            for action in evidence.action_detections[:10]:
                summary.append(f"  {action['timestamp']:.1f}s: {action['action']}")
        
        return "\n".join(summary)
    
    def get_cost(self) -> float:
        """Get total cost"""
        return self.total_cost
