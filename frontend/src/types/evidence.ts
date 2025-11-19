/**
 * TypeScript Types for HITL Evidence Review System
 *
 * All type definitions for the Human-in-the-Loop evidence review workflow
 */

// ============================================================================
// EVIDENCE ITEM TYPES
// ============================================================================

export interface EvidenceItem {
  id: number
  video_id: string
  evidence_type: 'audio_transcript' | 'visual_text' | 'visual_action'
  timestamp_start: number
  timestamp_end: number
  gpt4_prediction: any
  claude_prediction: any
  open_model_prediction: any
  consensus_answer: any
  confidence_score: number
  needs_human_review: boolean
  priority_level: 'high' | 'medium' | 'low'
  flag_reason: string
  human_review_status: 'pending' | 'approved' | 'corrected' | 'rejected' | 'skipped'
  human_reviewer_id?: string
  human_answer?: any
  human_confidence?: string
  human_notes?: string
  review_timestamp?: string
  review_duration_seconds?: number
  created_at?: string
  updated_at?: string
}

// ============================================================================
// REVIEW SUBMISSION TYPES
// ============================================================================

export interface ReviewSubmission {
  decision: 'approved' | 'corrected' | 'rejected' | 'skipped'
  corrected_answer?: any
  confidence?: 'high' | 'medium' | 'low'
  notes?: string
  review_duration_seconds?: number
}

// ============================================================================
// PROGRESS TRACKING TYPES
// ============================================================================

export interface ReviewProgress {
  total: number
  needs_review: number
  pending: number
  approved: number
  corrected: number
  rejected: number
  skipped: number
  percent_complete: number
  percent_approved: number
}

// ============================================================================
// REVIEWER STATS TYPES
// ============================================================================

export interface ReviewerStats {
  items_reviewed: number
  avg_review_time: number
  approved: number
  corrected: number
  rejected: number
  approval_rate: number
  correction_rate: number
}

// ============================================================================
// SESSION TYPES
// ============================================================================

export interface ReviewSession {
  session_id: number
  reviewer_id: string
  session_type: 'batch' | 'sequential'
  started_at: string
  ended_at?: string
  items_reviewed?: number
}

// ============================================================================
// UI STATE TYPES
// ============================================================================

export interface EvidenceFilters {
  videoId?: string
  priority?: 'high' | 'medium' | 'low'
  evidenceType?: 'audio_transcript' | 'visual_text' | 'visual_action'
  status?: 'pending' | 'approved' | 'corrected' | 'rejected' | 'skipped'
}

// ============================================================================
// MODEL PREDICTION TYPES
// ============================================================================

export type ModelType = 'gpt4' | 'claude' | 'open'

export interface ModelPrediction {
  model: ModelType
  prediction: any
  confidence?: number
  isCorrect?: boolean
}

// ============================================================================
// DISAGREEMENT TYPES
// ============================================================================

export interface DisagreementInfo {
  hasDisagreement: boolean
  level: 'none' | 'low' | 'medium' | 'high'
  disagreementCount: number
  agreeingModels: ModelType[]
  disagreeingModels: ModelType[]
}
