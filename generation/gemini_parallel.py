"""
Parallel Gemini Testing System

Tests generated questions against Gemini 2.5 Pro in parallel.

GOAL: Questions should be adversarial - Gemini should get them WRONG.
If Gemini answers correctly, the question is TOO EASY.

Features:
- Parallel API calls (30 questions in ~15 seconds)
- Answer comparison (exact, semantic, partial)
- Detailed failure analysis
- Export to Excel format (matching sample worksheet)
- Cost tracking

Expected results:
- Good questions: Gemini gets 60-80% WRONG
- Excellent questions: Gemini gets 80-90% WRONG
- Too easy: Gemini gets >50% RIGHT (need harder questions)
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import time
from pathlib import Path
import json

from templates.base import GeneratedQuestion, EvidenceDatabase
import google.generativeai as genai


class AnswerCorrectness(Enum):
    """Answer correctness assessment"""
    CORRECT = "correct"              # Gemini got it right (bad for us)
    INCORRECT = "incorrect"          # Gemini got it wrong (good!)
    PARTIALLY_CORRECT = "partial"    # Partially correct
    UNABLE_TO_ANSWER = "unable"      # Gemini couldn't answer


@dataclass
class GeminiTestResult:
    """Result of testing one question with Gemini"""
    question: GeneratedQuestion
    gemini_answer: str
    golden_answer: str
    correctness: AnswerCorrectness
    similarity_score: float  # 0.0 to 1.0
    explanation: str  # Why Gemini was right/wrong
    response_time: float  # Seconds
    cost: float  # Estimated cost


class GeminiParallelTester:
    """
    Test questions against Gemini 2.5 Pro in parallel
    
    Implements async parallel testing for speed.
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash-exp",
        max_concurrent: int = 10
    ):
        """
        Initialize Gemini tester
        
        Args:
            api_key: Google API key
            model_name: Gemini model to test against
            max_concurrent: Maximum concurrent API calls
        """
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        self.max_concurrent = max_concurrent
        
        # Cost tracking (approximate)
        # Gemini 2.0 Flash: $0.075 per 1M input tokens, $0.30 per 1M output
        self.input_cost_per_token = 0.075 / 1_000_000
        self.output_cost_per_token = 0.30 / 1_000_000
        
        self.total_cost = 0.0
        
        # Statistics
        self.stats = {
            "total_tested": 0,
            "correct": 0,
            "incorrect": 0,
            "partial": 0,
            "unable": 0,
            "total_time": 0.0
        }
    
    async def test_batch_parallel(
        self,
        questions: List[GeneratedQuestion],
        video_path: str,
        evidence: EvidenceDatabase
    ) -> List[GeminiTestResult]:
        """
        Test batch of questions in parallel
        
        Args:
            questions: List of questions to test
            video_path: Path to video file
            evidence: Evidence database
            
        Returns:
            List of test results
        """
        print(f"[GeminiTest] Testing {len(questions)} questions in parallel...")
        start_time = time.time()
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Create tasks
        tasks = [
            self._test_single_question_async(q, video_path, evidence, semaphore)
            for q in questions
        ]
        
        # Execute in parallel
        results = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        self.stats["total_time"] += elapsed
        
        print(f"[GeminiTest] Completed in {elapsed:.1f}s ({elapsed/len(questions):.2f}s per question)")
        
        return results
    
    async def _test_single_question_async(
        self,
        question: GeneratedQuestion,
        video_path: str,
        evidence: EvidenceDatabase,
        semaphore: asyncio.Semaphore
    ) -> GeminiTestResult:
        """Test single question (async)"""
        async with semaphore:
            # Run sync Gemini call in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._test_single_question_sync,
                question,
                video_path,
                evidence
            )
            return result
    
    def _test_single_question_sync(
        self,
        question: GeneratedQuestion,
        video_path: str,
        evidence: EvidenceDatabase
    ) -> GeminiTestResult:
        """Test single question (sync)"""
        start_time = time.time()
        
        try:
            # Build prompt for Gemini
            prompt = self._build_gemini_prompt(question, evidence)
            
            # Upload video and get Gemini's answer
            gemini_answer = self._query_gemini(video_path, prompt, question)
            
            response_time = time.time() - start_time
            
            # Compare answers
            correctness, similarity, explanation = self._compare_answers(
                gemini_answer,
                question.golden_answer,
                question.question_text
            )
            
            # Estimate cost
            cost = self._estimate_cost(prompt, gemini_answer)
            self.total_cost += cost
            
            # Update stats
            self.stats["total_tested"] += 1
            if correctness == AnswerCorrectness.CORRECT:
                self.stats["correct"] += 1
            elif correctness == AnswerCorrectness.INCORRECT:
                self.stats["incorrect"] += 1
            elif correctness == AnswerCorrectness.PARTIALLY_CORRECT:
                self.stats["partial"] += 1
            else:
                self.stats["unable"] += 1
            
            return GeminiTestResult(
                question=question,
                gemini_answer=gemini_answer,
                golden_answer=question.golden_answer,
                correctness=correctness,
                similarity_score=similarity,
                explanation=explanation,
                response_time=response_time,
                cost=cost
            )
            
        except Exception as e:
            print(f"[GeminiTest] Error testing question: {e}")
            
            return GeminiTestResult(
                question=question,
                gemini_answer=f"ERROR: {str(e)}",
                golden_answer=question.golden_answer,
                correctness=AnswerCorrectness.UNABLE_TO_ANSWER,
                similarity_score=0.0,
                explanation=f"API error: {str(e)}",
                response_time=time.time() - start_time,
                cost=0.0
            )
    
    def _build_gemini_prompt(
        self,
        question: GeneratedQuestion,
        evidence: EvidenceDatabase
    ) -> str:
        """
        Build prompt for Gemini
        
        CRITICAL: Includes timestamp range to focus Gemini's attention
        """
        prompt = f"""Answer this question about the video.

QUESTION: {question.question_text}

IMPORTANT: Focus on the video segment from {question.start_timestamp:.1f}s to {question.end_timestamp:.1f}s.

Provide a concise, direct answer. Do not include preamble or explanation unless asked."""

        return prompt
    
    def _query_gemini(
        self,
        video_path: str,
        prompt: str,
        question: GeneratedQuestion
    ) -> str:
        """
        Query Gemini with video and prompt
        
        Uploads video and sends prompt.
        """
        # Upload video file
        print(f"[GeminiTest] Uploading video: {video_path}")
        video_file = genai.upload_file(path=video_path)
        
        # Wait for video to be processed
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise Exception(f"Video processing failed: {video_file.state.name}")
        
        # Generate response
        response = self.model.generate_content(
            [video_file, prompt],
            request_options={"timeout": 60}
        )
        
        # Clean up uploaded file
        genai.delete_file(video_file.name)
        
        return response.text
    
    def _compare_answers(
        self,
        gemini_answer: str,
        golden_answer: str,
        question_text: str
    ) -> Tuple[AnswerCorrectness, float, str]:
        """
        Compare Gemini's answer with golden answer
        
        Returns:
            (correctness, similarity_score, explanation)
        """
        # Clean answers
        gemini_clean = gemini_answer.strip().lower()
        golden_clean = golden_answer.strip().lower()
        
        # Check for "unable to answer" patterns
        unable_patterns = [
            "i cannot",
            "i can't",
            "unable to",
            "don't have access",
            "cannot determine",
            "not enough information"
        ]
        
        if any(pattern in gemini_clean for pattern in unable_patterns):
            return (
                AnswerCorrectness.UNABLE_TO_ANSWER,
                0.0,
                "Gemini was unable to answer the question"
            )
        
        # Exact match
        if gemini_clean == golden_clean:
            return (
                AnswerCorrectness.CORRECT,
                1.0,
                "Gemini's answer matches exactly"
            )
        
        # Semantic similarity using simple word overlap
        gemini_words = set(gemini_clean.split())
        golden_words = set(golden_clean.split())
        
        if not golden_words:
            similarity = 0.0
        else:
            overlap = len(gemini_words & golden_words)
            similarity = overlap / len(golden_words)
        
        # Determine correctness based on similarity
        if similarity >= 0.8:
            return (
                AnswerCorrectness.CORRECT,
                similarity,
                f"Gemini's answer is substantially correct (similarity: {similarity:.2f})"
            )
        elif similarity >= 0.5:
            return (
                AnswerCorrectness.PARTIALLY_CORRECT,
                similarity,
                f"Gemini's answer is partially correct (similarity: {similarity:.2f})"
            )
        else:
            return (
                AnswerCorrectness.INCORRECT,
                similarity,
                f"Gemini's answer is incorrect (similarity: {similarity:.2f})"
            )
    
    def _estimate_cost(self, prompt: str, response: str) -> float:
        """Estimate API cost"""
        # Rough token estimate (4 chars ≈ 1 token)
        input_tokens = len(prompt) // 4
        output_tokens = len(response) // 4
        
        cost = (
            input_tokens * self.input_cost_per_token +
            output_tokens * self.output_cost_per_token
        )
        
        return cost
    
    def get_statistics(self) -> Dict:
        """Get testing statistics"""
        total = self.stats["total_tested"]
        
        if total == 0:
            return {
                "total_tested": 0,
                "gemini_wrong_rate": "0%",
                "question_quality": "N/A"
            }
        
        # Calculate rates
        correct_rate = self.stats["correct"] / total * 100
        incorrect_rate = self.stats["incorrect"] / total * 100
        partial_rate = self.stats["partial"] / total * 100
        unable_rate = self.stats["unable"] / total * 100
        
        # Assess question quality
        # GOAL: High incorrect rate = good adversarial questions
        if incorrect_rate >= 80:
            quality = "Excellent - Very challenging for Gemini"
        elif incorrect_rate >= 60:
            quality = "Good - Appropriately challenging"
        elif incorrect_rate >= 40:
            quality = "Fair - Moderately challenging"
        else:
            quality = "Poor - Too easy for Gemini"
        
        return {
            "total_tested": total,
            "gemini_correct": self.stats["correct"],
            "gemini_incorrect": self.stats["incorrect"],
            "gemini_partial": self.stats["partial"],
            "gemini_unable": self.stats["unable"],
            "correct_rate": f"{correct_rate:.1f}%",
            "incorrect_rate": f"{incorrect_rate:.1f}%",
            "partial_rate": f"{partial_rate:.1f}%",
            "unable_rate": f"{unable_rate:.1f}%",
            "gemini_wrong_rate": f"{incorrect_rate:.1f}%",
            "question_quality": quality,
            "total_cost": f"${self.total_cost:.2f}",
            "avg_response_time": f"{self.stats['total_time'] / total:.2f}s" if total > 0 else "0s"
        }
    
    def print_statistics(self):
        """Print testing statistics"""
        stats = self.get_statistics()
        
        print("=" * 80)
        print("GEMINI TESTING STATISTICS")
        print("=" * 80)
        print(f"Total Questions Tested: {stats['total_tested']}")
        print()
        print("GEMINI PERFORMANCE:")
        print(f"  Correct: {stats['gemini_correct']} ({stats['correct_rate']})")
        print(f"  Incorrect: {stats['gemini_incorrect']} ({stats['incorrect_rate']}) ← GOAL")
        print(f"  Partial: {stats['gemini_partial']} ({stats['partial_rate']})")
        print(f"  Unable: {stats['gemini_unable']} ({stats['unable_rate']})")
        print()
        print(f"QUESTION QUALITY: {stats['question_quality']}")
        print()
        print(f"Total Cost: {stats['total_cost']}")
        print(f"Avg Response Time: {stats['avg_response_time']}")
        print("=" * 80)
    
    def export_to_excel(
        self,
        results: List[GeminiTestResult],
        output_path: str,
        video_info: Dict
    ):
        """
        Export results to Excel format matching sample worksheet
        
        Matches the format from Sample_work_sheet_-_MSPO_557.xlsx
        """
        try:
            import pandas as pd
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment
            
            # Prepare data rows
            data = []
            
            for i, result in enumerate(results, 1):
                row = {
                    # Video info
                    'Drive Link': video_info.get('url', ''),
                    'Video Duration - format (h:mm:ss)': video_info.get('duration', ''),
                    
                    # Question info
                    f'Question {i}': result.question.question_text,
                    f'Golden Answer {i}': result.golden_answer,
                    f'Gemini 2.5 Pro Answer {i}': result.gemini_answer,
                    
                    # Timestamps
                    f'Start Time-Stamp (hh:mm:ss) {i}': self._format_timestamp(result.question.start_timestamp),
                    f'End Time-Stamp (hh:mm:ss) {i}': self._format_timestamp(result.question.end_timestamp),
                    
                    # Question type
                    f'Question Type {i}': ', '.join([qt.value for qt in result.question.question_types]),
                    
                    # Correctness assessment
                    f"Explain why the model's response is not correct {i}": result.explanation if result.correctness != AnswerCorrectness.CORRECT else '',
                    
                    # Validation flags
                    f'Can Q{i} Be Used?': 'Pass' if result.correctness == AnswerCorrectness.INCORRECT else 'Needs Review',
                    
                    # Similarity score
                    f'Similarity Score {i}': f"{result.similarity_score:.2f}"
                }
                
                data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Write to Excel
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Results')
                
                # Get workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets['Results']
                
                # Style header row
                header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                header_font = Font(color="FFFFFF", bold=True)
                
                for cell in worksheet[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(cell.value)
                        except:
                            pass
                    
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"[GeminiTest] Results exported to: {output_path}")
            
        except ImportError:
            print("[GeminiTest] pandas/openpyxl not installed - saving as JSON instead")
            self._export_to_json(results, output_path.replace('.xlsx', '.json'), video_info)
    
    def _export_to_json(
        self,
        results: List[GeminiTestResult],
        output_path: str,
        video_info: Dict
    ):
        """Export results to JSON (fallback)"""
        data = {
            "video_info": video_info,
            "results": [
                {
                    "question": result.question.question_text,
                    "golden_answer": result.golden_answer,
                    "gemini_answer": result.gemini_answer,
                    "correctness": result.correctness.value,
                    "similarity_score": result.similarity_score,
                    "explanation": result.explanation,
                    "start_timestamp": result.question.start_timestamp,
                    "end_timestamp": result.question.end_timestamp,
                    "question_types": [qt.value for qt in result.question.question_types],
                    "response_time": result.response_time,
                    "cost": result.cost
                }
                for result in results
            ],
            "statistics": self.get_statistics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[GeminiTest] Results exported to: {output_path}")
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp as hh:mm:ss"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ============================================================================
# SYNCHRONOUS WRAPPER (for non-async code)
# ============================================================================

def test_questions_with_gemini(
    questions: List[GeneratedQuestion],
    video_path: str,
    evidence: EvidenceDatabase,
    api_key: str,
    output_path: Optional[str] = None
) -> List[GeminiTestResult]:
    """
    Synchronous wrapper for testing questions
    
    Args:
        questions: Questions to test
        video_path: Path to video file
        evidence: Evidence database
        api_key: Google API key
        output_path: Path to save results (optional)
        
    Returns:
        List of test results
    """
    # Create tester
    tester = GeminiParallelTester(api_key=api_key)
    
    # Run async test
    results = asyncio.run(
        tester.test_batch_parallel(questions, video_path, evidence)
    )
    
    # Print statistics
    tester.print_statistics()
    
    # Export if requested
    if output_path:
        video_info = {
            'url': f"file://{video_path}",
            'duration': evidence.duration
        }
        tester.export_to_excel(results, output_path, video_info)
    
    return results


# ============================================================================
# QUALITY ASSESSMENT
# ============================================================================

def assess_question_quality(results: List[GeminiTestResult]) -> Dict:
    """
    Assess overall question quality based on Gemini performance
    
    Returns:
        Quality assessment with recommendations
    """
    total = len(results)
    if total == 0:
        return {"quality": "N/A", "recommendations": []}
    
    # Count by correctness
    counts = {
        AnswerCorrectness.CORRECT: 0,
        AnswerCorrectness.INCORRECT: 0,
        AnswerCorrectness.PARTIALLY_CORRECT: 0,
        AnswerCorrectness.UNABLE_TO_ANSWER: 0
    }
    
    for result in results:
        counts[result.correctness] += 1
    
    incorrect_rate = counts[AnswerCorrectness.INCORRECT] / total
    correct_rate = counts[AnswerCorrectness.CORRECT] / total
    
    # Assess quality
    if incorrect_rate >= 0.8:
        quality = "Excellent"
        grade = "A"
        recommendations = [
            "Questions are highly adversarial - perfect for exposing Gemini weaknesses",
            "Maintain current question generation strategy"
        ]
    elif incorrect_rate >= 0.6:
        quality = "Good"
        grade = "B"
        recommendations = [
            "Questions are appropriately challenging",
            "Consider increasing complexity slightly for more adversarial questions"
        ]
    elif incorrect_rate >= 0.4:
        quality = "Fair"
        grade = "C"
        recommendations = [
            "Questions are moderately challenging but could be harder",
            "Focus on inference, spurious correlations, and multi-step reasoning",
            "Increase temporal complexity and use more subtle cues"
        ]
    else:
        quality = "Poor"
        grade = "D"
        recommendations = [
            "Questions are too easy for Gemini",
            "CRITICAL: Increase question complexity significantly",
            "Use multi-type combinations (Temporal + Counting + Needle)",
            "Add spurious correlations and counter-intuitive scenarios",
            "Ensure questions truly require both audio AND visual cues"
        ]
    
    return {
        "quality": quality,
        "grade": grade,
        "gemini_correct_rate": f"{correct_rate * 100:.1f}%",
        "gemini_incorrect_rate": f"{incorrect_rate * 100:.1f}%",
        "total_questions": total,
        "recommendations": recommendations,
        "breakdown": {
            "correct": counts[AnswerCorrectness.CORRECT],
            "incorrect": counts[AnswerCorrectness.INCORRECT],
            "partial": counts[AnswerCorrectness.PARTIALLY_CORRECT],
            "unable": counts[AnswerCorrectness.UNABLE_TO_ANSWER]
        }
    }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Gemini Parallel Testing System")
    print("=" * 80)
    print()
    print("This module tests generated questions against Gemini 2.5 Pro.")
    print()
    print("GOAL: Questions should be adversarial - Gemini should get them WRONG.")
    print()
    print("Quality Grades:")
    print("  A (Excellent): 80%+ incorrect - Highly adversarial")
    print("  B (Good): 60-80% incorrect - Appropriately challenging")
    print("  C (Fair): 40-60% incorrect - Moderately challenging")
    print("  D (Poor): <40% incorrect - Too easy, need harder questions")
    print()
    print("Example usage:")
    print("""
    from generation.gemini_parallel import test_questions_with_gemini
    
    results = test_questions_with_gemini(
        questions=generated_questions,
        video_path="path/to/video.mp4",
        evidence=evidence_db,
        api_key="your_google_api_key",
        output_path="results.xlsx"
    )
    
    # Assessment
    from generation.gemini_parallel import assess_question_quality
    assessment = assess_question_quality(results)
    print(assessment)
    """)