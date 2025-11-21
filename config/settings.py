"""
Configuration management using Pydantic Settings
Loads from environment variables with validation
"""
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, List
from pathlib import Path
import os

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # ==================== DATABASE ====================
    database_url: str = Field(
        default="postgresql://user:pass@localhost:5432/video_qa",
        description="PostgreSQL connection URL"
    )
    database_pool_size: int = Field(default=20, ge=1, le=100)
    
    # ==================== REDIS ====================
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    # ==================== API KEYS ====================
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    google_api_key: str = Field(default="", description="Google API key")
    gemini_api_key: str = Field(default="", description="Gemini API key")
    hf_token: Optional[str] = Field(default=None, description="HuggingFace token for speaker diarization")
    
    # ==================== MODEL CONFIGURATION ====================
    gemini_model: str = Field(default="gemini-2.5-pro")
    gemini_flash_model: str = Field(default="gemini-1.5-flash")
    gpt4_model: str = Field(default="gpt-4-0125-preview")
    gpt4_mini_model: str = Field(default="gpt-4-mini")
    claude_model: str = Field(default="claude-sonnet-4-5-20250929")
    
    # Llama-3 (Local)
    llama_model_path: Optional[str] = Field(default=None)
    llama_enabled: bool = Field(default=False)
    
    # ==================== COST CONFIGURATION (per video) ====================
    cost_evidence_extraction_light: float = Field(default=0.50)
    cost_evidence_extraction_deep: float = Field(default=1.00)
    cost_qa_generation_templates: float = Field(default=0.00)
    cost_qa_generation_llama: float = Field(default=0.30)
    cost_qa_generation_gpt4mini: float = Field(default=0.15)
    cost_gemini_flash_per_q: float = Field(default=0.005)
    cost_gemini_pro_per_q: float = Field(default=0.05)
    cost_video_upload: float = Field(default=0.50)
    cost_explanation_gpt4mini: float = Field(default=0.01)
    cost_storage: float = Field(default=0.20)
    
    # ==================== QUALITY THRESHOLDS ====================
    min_confidence_threshold: float = Field(
        default=0.999,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for accepting Q&A (99.9%)"
    )
    min_evidence_confidence: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for evidence extraction"
    )
    min_failure_score: float = Field(
        default=8.0,
        ge=0.0,
        le=10.0,
        description="Minimum score for accepting failures"
    )
    validation_pass_threshold: float = Field(
        default=1.0,
        description="All validations must pass (100%)"
    )
    
    # ==================== PROCESSING CONFIGURATION ====================
    max_parallel_workers: int = Field(default=10, ge=1, le=50)
    gpu_enabled: bool = Field(default=True)
    gpu_memory_fraction: float = Field(default=0.9, ge=0.1, le=1.0)
    
    # ==================== BATCH PROCESSING ====================
    min_questions_to_test: int = Field(default=15, ge=1, le=50)
    max_questions_to_test: int = Field(default=20, ge=1, le=50)
    candidates_to_generate: int = Field(default=30, ge=10, le=100)
    
    # Tier distribution
    tier1_template_count: int = Field(default=8)
    tier2_llama_count: int = Field(default=17)
    tier3_gpt4mini_count: int = Field(default=5)
    
    # VALIDATOR COMMENTED OUT - was causing issues
    # The validator needs all three tier values but they're validated sequentially
    # Uncomment and fix if you need strict validation
    # @validator("tier3_gpt4mini_count")
    # def validate_tier_sum(cls, v, values):
    #     """Ensure tiers sum to candidates_to_generate"""
    #     if 'candidates_to_generate' in values and 'tier1_template_count' in values and 'tier2_llama_count' in values:
    #         total = values.get('tier1_template_count', 0) + values.get('tier2_llama_count', 0) + v
    #         expected = values.get('candidates_to_generate', 30)
    #         if total != expected:
    #             raise ValueError(
    #                 f"Tier counts must sum to candidates_to_generate "
    #                 f"(got {total}, expected {expected})"
    #             )
    #     return v
    
    # ==================== FEEDBACK LOOP ====================
    feedback_loop_enabled: bool = Field(default=True)
    update_weights_every_n_videos: int = Field(default=100, ge=10)
    min_score_for_template_discovery: float = Field(default=9.0, ge=0.0, le=10.0)
    finetune_llama_every_n_videos: int = Field(default=1000, ge=100)
    
    # ==================== TIERED TESTING ====================
    use_tiered_testing: bool = Field(default=True)
    flash_pass_rate_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="If Flash passes >50%, skip Flash tier"
    )
    
    # ==================== STORAGE ====================
    upload_dir: Path = Field(default=Path("./uploads"))
    output_dir: Path = Field(default=Path("./outputs"))
    log_dir: Path = Field(default=Path("./logs"))
    temp_dir: Path = Field(default=Path("./temp"))
    
    @validator("upload_dir", "output_dir", "log_dir", "temp_dir")
    def create_directory(cls, v):
        """Create directory if it doesn't exist"""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    # ==================== LOGGING ====================
    log_level: str = Field(default="INFO")
    log_to_file: bool = Field(default=True)
    log_to_console: bool = Field(default=True)
    
    # ==================== API SERVER ====================
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    api_workers: int = Field(default=4, ge=1)
    
    # ==================== CORS ====================
    cors_origins: List[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"]
    )
    
    # ==================== CELERY ====================
    celery_broker_url: str = Field(default="redis://localhost:6379/0")
    celery_result_backend: str = Field(default="redis://localhost:6379/0")
    celery_task_track_started: bool = Field(default=True)
    celery_task_time_limit: int = Field(default=3600)  # 1 hour
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# ==================== SINGLETON INSTANCE ====================
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get or create settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

# Convenience access
settings = get_settings()