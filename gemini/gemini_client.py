"""
Gemini Client - Google Gemini API Wrapper

Purpose: Interface with Google's Gemini AI models for adversarial testing
Compliance: Track API costs, handle rate limits, support multimodal input
Architecture: Clean abstraction over Gemini API

Cost Model:
- Gemini 1.5 Flash: $0.075/1M input tokens, $0.30/1M output tokens
- Gemini 1.5 Pro: $1.25/1M input tokens, $5.00/1M output tokens
- Target: ~$0.30 per video for Gemini testing (30 questions @ $0.01 each)
"""

# Standard library imports
import logging
import time
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import base64

# Third-party imports
import google.generativeai as genai  # pip install google-generativeai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)


class GeminiModel(Enum):
    """Available Gemini models"""
    FLASH_1_5 = "gemini-1.5-flash"  # Fast, cheap
    PRO_1_5 = "gemini-1.5-pro"      # Accurate, expensive
    FLASH_2_0 = "gemini-2.0-flash-exp"  # Experimental


@dataclass
class GeminiConfig:
    """Configuration for Gemini API"""
    model: GeminiModel = GeminiModel.FLASH_1_5
    temperature: float = 0.1  # Low for consistency
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 1024
    
    # Safety settings (permissive for testing)
    safety_settings: Dict[HarmCategory, HarmBlockThreshold] = field(default_factory=dict)
    
    # Rate limiting
    requests_per_minute: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        """Initialize default safety settings"""
        if not self.safety_settings:
            # Permissive settings for adversarial testing
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }


@dataclass
class GeminiResponse:
    """Response from Gemini API"""
    answer: str
    model: str
    finish_reason: str
    
    # Token usage
    input_tokens: int
    output_tokens: int
    total_tokens: int
    
    # Cost tracking
    input_cost: float
    output_cost: float
    total_cost: float
    
    # Timing
    response_time: float  # seconds
    
    # Metadata
    safety_ratings: Optional[List[Dict]] = None
    blocked: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "answer": self.answer,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "response_time": self.response_time,
            "blocked": self.blocked
        }


class GeminiClient:
    """
    Client for Google Gemini API.
    
    Provides clean interface for adversarial testing with:
    - Multimodal input support (text + images + video)
    - Cost tracking
    - Rate limiting
    - Retry logic
    - Safety controls
    """
    
    def __init__(
        self,
        api_key: str,
        config: Optional[GeminiConfig] = None
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google AI API key
            config: Gemini configuration
        """
        self.api_key = api_key
        self.config = config or GeminiConfig()
        
        # Configure API
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.config.model.value,
            generation_config={
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "max_output_tokens": self.config.max_output_tokens,
            },
            safety_settings=self.config.safety_settings
        )
        
        # Rate limiting
        self.last_request_time = 0.0
        self.request_count = 0
        self.minute_start = time.time()
        
        # Cost tracking
        self.total_cost = 0.0
        self.total_requests = 0
        
        logger.info(
            f"GeminiClient initialized (model: {self.config.model.value})"
        )
    
    def ask_question(
        self,
        question: str,
        context: Optional[str] = None,
        image_path: Optional[Path] = None,
        video_frame: Optional[Any] = None
    ) -> GeminiResponse:
        """
        Ask Gemini a question with optional context and media.
        
        Args:
            question: Question text
            context: Optional context/evidence
            image_path: Optional path to image
            video_frame: Optional video frame (numpy array or base64)
        
        Returns:
            GeminiResponse with answer and metadata
        """
        # Rate limiting
        self._enforce_rate_limit()
        
        # Build prompt
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Context: {context}\n\n")
        
        prompt_parts.append(f"Question: {question}")
        
        # Add media if provided
        if image_path and image_path.exists():
            with open(image_path, 'rb') as f:
                image_data = f.read()
            prompt_parts.append({
                'mime_type': 'image/jpeg',
                'data': image_data
            })
        
        if video_frame is not None:
            # Convert video frame to image if needed
            # TODO: Implement frame conversion
            pass
        
        # Make request with retry logic
        start_time = time.time()
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.model.generate_content(prompt_parts)
                response_time = time.time() - start_time
                
                # Check if blocked
                if not response.candidates:
                    logger.warning(f"Response blocked by safety filters")
                    return self._create_blocked_response(response_time)
                
                # Extract answer
                candidate = response.candidates[0]
                answer = candidate.content.parts[0].text
                finish_reason = candidate.finish_reason.name
                
                # Get token counts
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
                
                # Calculate costs
                input_cost, output_cost = self._calculate_cost(
                    input_tokens, output_tokens
                )
                total_cost = input_cost + output_cost
                
                # Update tracking
                self.total_cost += total_cost
                self.total_requests += 1
                
                # Create response
                gemini_response = GeminiResponse(
                    answer=answer,
                    model=self.config.model.value,
                    finish_reason=finish_reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    input_cost=input_cost,
                    output_cost=output_cost,
                    total_cost=total_cost,
                    response_time=response_time,
                    safety_ratings=[
                        {"category": r.category.name, "probability": r.probability.name}
                        for r in candidate.safety_ratings
                    ],
                    blocked=False
                )
                
                logger.debug(
                    f"Gemini response: {len(answer)} chars, "
                    f"{total_tokens} tokens, ${total_cost:.4f}"
                )
                
                return gemini_response
                
            except Exception as e:
                logger.warning(
                    f"Gemini API error (attempt {attempt+1}/{self.config.retry_attempts}): {e}"
                )
                
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise
        
        raise RuntimeError("Failed to get Gemini response after all retries")
    
    def test_qa_pair(
        self,
        question: str,
        golden_answer: str,
        evidence: Optional[str] = None,
        image_path: Optional[Path] = None
    ) -> GeminiResponse:
        """
        Test a Q&A pair against Gemini.
        
        Args:
            question: Question to test
            golden_answer: Expected correct answer
            evidence: Optional evidence context
            image_path: Optional image evidence
        
        Returns:
            GeminiResponse with Gemini's answer
        """
        # For adversarial testing, we don't provide the golden answer
        # We want to see if Gemini hallucinates
        
        return self.ask_question(
            question=question,
            context=evidence,
            image_path=image_path
        )
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.minute_start >= 60:
            self.request_count = 0
            self.minute_start = current_time
        
        # Check rate limit
        if self.request_count >= self.config.requests_per_minute:
            sleep_time = 60 - (current_time - self.minute_start)
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.request_count = 0
                self.minute_start = time.time()
        
        # Increment counter
        self.request_count += 1
        self.last_request_time = current_time
    
    def _calculate_cost(
        self, input_tokens: int, output_tokens: int
    ) -> tuple[float, float]:
        """
        Calculate cost based on token usage.
        
        Returns:
            (input_cost, output_cost)
        """
        # Cost per 1M tokens
        if self.config.model == GeminiModel.FLASH_1_5:
            input_rate = 0.075
            output_rate = 0.30
        elif self.config.model == GeminiModel.PRO_1_5:
            input_rate = 1.25
            output_rate = 5.00
        else:  # FLASH_2_0 or others
            input_rate = 0.075
            output_rate = 0.30
        
        input_cost = (input_tokens / 1_000_000) * input_rate
        output_cost = (output_tokens / 1_000_000) * output_rate
        
        return input_cost, output_cost
    
    def _create_blocked_response(self, response_time: float) -> GeminiResponse:
        """Create response for blocked content"""
        return GeminiResponse(
            answer="[BLOCKED BY SAFETY FILTERS]",
            model=self.config.model.value,
            finish_reason="SAFETY",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
            response_time=response_time,
            blocked=True
        )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_requests": self.total_requests,
            "total_cost": self.total_cost,
            "avg_cost_per_request": (
                self.total_cost / self.total_requests
                if self.total_requests > 0 else 0.0
            ),
            "model": self.config.model.value
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize client
    client = GeminiClient(
        api_key="YOUR_API_KEY",
        config=GeminiConfig(
            model=GeminiModel.FLASH_1_5,
            temperature=0.1
        )
    )
    
    # Test question
    response = client.ask_question(
        question="What color is the person's shirt in this video?",
        context="A person wearing a red shirt walks across the frame."
    )
    
    print(f"✅ Gemini response:")
    print(f"   Answer: {response.answer}")
    print(f"   Tokens: {response.total_tokens}")
    print(f"   Cost: ${response.total_cost:.6f}")
    print(f"   Time: {response.response_time:.2f}s")
    
    # Get stats
    stats = client.get_usage_stats()
    print(f"\n✅ Usage stats:")
    print(f"   Requests: {stats['total_requests']}")
    print(f"   Total cost: ${stats['total_cost']:.6f}")
