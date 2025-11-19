/**
 * TypeScript Types for Review System
 * 
 * All type definitions for the human review workflow
 */

// ============================================================================
// QUESTION TYPES
// ============================================================================

export interface Question {
  id: number;
  question_id: string;
  question_text: string;
  golden_answer: string;
  generation_tier: 'template' | 'llama' | 'gpt4mini';
  task_type: string | null;
  start_seconds: number | null;
  end_seconds: number | null;
  audio_cues: any[] | null;
  visual_cues: any[] | null;
  confidence_score: number | null;
  
  // Stage 1 Review
  stage1_review_status: 'pending' | 'approved' | 'rejected';
  stage1_reviewer: string | null;
  stage1_feedback: string | null;
  stage1_reviewed_at: string | null;
  
  // Regeneration
  generation_attempt: number;
  is_regeneration: boolean;
  parent_question_id: number | null;
  
  // Stage 2 Selection
  stage2_shown_for_selection: boolean;
  stage2_manually_selected: boolean;
  stage2_selection_rank: number | null;
  stage2_selected_at: string | null;
  
  created_at: string;
}

// ============================================================================
// PROGRESS TYPES
// ============================================================================

export interface ReviewProgress {
  approved: number;
  rejected: number;
  pending: number;
  total: number;
  progress_percent: number;
}

// ============================================================================
// FAILURE TYPES
// ============================================================================

export interface GeminiFailure {
  failure_id: string;
  question_id: string;
  question_text: string;
  golden_answer: string;
  gemini_answer: string;
  failure_type: string;
  failure_score: number;
  severity_score: number;
  clarity_score: number;
  combined_score: number;
  rank: number;
}

// ============================================================================
// VIDEO STATUS TYPES
// ============================================================================

export type PipelineStage = 
  | 'generating'
  | 'awaiting_stage1_review'
  | 'validating'
  | 'testing_gemini'
  | 'awaiting_stage2_selection'
  | 'completed'
  | 'failed';

export interface VideoStatus {
  video_id: string;
  pipeline_stage: PipelineStage;
  status: string;
  stage1_progress?: ReviewProgress;
  stage2_failures_count?: number;
  stage2_selected_count?: number;
  final_selected?: number;
  completed_at?: string;
}

// ============================================================================
// WEBSOCKET EVENT TYPES
// ============================================================================

export type WebSocketEventType =
  | 'connection_established'
  | 'stage1_ready'
  | 'stage1_progress'
  | 'stage1_complete'
  | 'validation_progress'
  | 'gemini_testing_progress'
  | 'stage2_ready'
  | 'pipeline_complete'
  | 'pipeline_error'
  | 'regeneration_complete'
  | 'pong';

export interface WebSocketEvent {
  type: WebSocketEventType;
  video_id: string;
  timestamp: string;
  [key: string]: any;
}

export interface Stage1ReadyEvent extends WebSocketEvent {
  type: 'stage1_ready';
  total_questions: number;
  message: string;
}

export interface Stage1ProgressEvent extends WebSocketEvent {
  type: 'stage1_progress';
  approved: number;
  rejected: number;
  pending: number;
  total: number;
  progress_percent: number;
}

export interface Stage2ReadyEvent extends WebSocketEvent {
  type: 'stage2_ready';
  failures_count: number;
  message: string;
}

export interface PipelineCompleteEvent extends WebSocketEvent {
  type: 'pipeline_complete';
  excel_path: string;
  message: string;
}

export interface RegenerationCompleteEvent extends WebSocketEvent {
  type: 'regeneration_complete';
  original_question_id: string;
  new_question_id: string;
  message: string;
}

// ============================================================================
// API REQUEST TYPES
// ============================================================================

export interface ApproveQuestionRequest {
  reviewer: string;
}

export interface RejectQuestionRequest {
  reviewer: string;
  feedback: string;
}

export interface BulkReviewRequest {
  question_ids: number[];
  reviewer: string;
  feedback?: string;
}

export interface RegenerateQuestionRequest {
  reviewer: string;
}

export interface Stage2SelectionRequest {
  selected_question_ids: number[];
  reviewer: string;
}

// ============================================================================
// API RESPONSE TYPES
// ============================================================================

export interface ApproveQuestionResponse {
  success: boolean;
  question_id: number;
  status: string;
  progress: ReviewProgress;
}

export interface RejectQuestionResponse {
  success: boolean;
  question_id: number;
  status: string;
  feedback: string;
  can_regenerate: boolean;
  regeneration_info: string;
  generation_attempt: number;
  progress: ReviewProgress;
}

export interface RegenerateQuestionResponse {
  success: boolean;
  original_question_id: number;
  new_question_id: number;
  new_question: Question;
  generation_attempt: number;
}

export interface Stage2SelectionResponse {
  success: boolean;
  video_id: string;
  selected_count: number;
  selected_question_ids: number[];
  excel_path: string;
  pipeline_stage: string;
}

// ============================================================================
// UI STATE TYPES
// ============================================================================

export interface ReviewFilters {
  status: 'all' | 'pending' | 'approved' | 'rejected';
  tier: 'all' | 'template' | 'llama' | 'gpt4mini';
  taskType: 'all' | string;
  searchTerm: string;
}

export interface SortConfig {
  field: keyof Question | 'none';
  direction: 'asc' | 'desc';
}