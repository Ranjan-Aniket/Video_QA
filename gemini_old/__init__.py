"""
Gemini Package - Block 5: Gemini Testing Integration

This package provides adversarial testing against Google's Gemini AI model
to identify hallucination-inducing questions.
"""

from .gemini_client import (
    GeminiClient,
    GeminiModel,
    GeminiResponse,
    GeminiConfig
)

from .adversarial_tester import (
    AdversarialTester,
    TestResult,
    TestStatus,
    TestConfig
)

from .hallucination_detector import (
    HallucinationDetector,
    HallucinationType,
    HallucinationResult,
    DetectionConfig
)

from .benchmark_analyzer import (
    BenchmarkAnalyzer,
    BenchmarkMetrics,
    PerformanceReport,
    QuestionTypeStats
)

__all__ = [
    # Gemini Client
    'GeminiClient',
    'GeminiModel',
    'GeminiResponse',
    'GeminiConfig',
    
    # Adversarial Tester
    'AdversarialTester',
    'TestResult',
    'TestStatus',
    'TestConfig',
    
    # Hallucination Detector
    'HallucinationDetector',
    'HallucinationType',
    'HallucinationResult',
    'DetectionConfig',
    
    # Benchmark Analyzer
    'BenchmarkAnalyzer',
    'BenchmarkMetrics',
    'PerformanceReport',
    'QuestionTypeStats',
]

__version__ = '1.0.0'
