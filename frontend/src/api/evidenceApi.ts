import axios from 'axios'
import type {
  EvidenceItem,
  ReviewSubmission,
  ReviewProgress,
  ReviewerStats,
  ReviewSession,
} from '../types/evidence'

const API_BASE = 'http://localhost:8000/api'

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
})

// ============================================================================
// EVIDENCE REVIEW QUEUE
// ============================================================================

export interface GetReviewQueueParams {
  video_id?: string
  priority?: 'high' | 'medium' | 'low'
  limit?: number
}

export const getReviewQueue = async (params?: GetReviewQueueParams): Promise<EvidenceItem[]> => {
  const { data } = await api.get('/evidence/review/queue', { params })
  return data.items || []
}

// ============================================================================
// EVIDENCE ITEM OPERATIONS
// ============================================================================

export const getEvidence = async (evidenceId: number): Promise<EvidenceItem> => {
  const { data } = await api.get(`/evidence/review/${evidenceId}`)
  return data.evidence || data
}

export const submitReview = async (
  evidenceId: number,
  reviewData: ReviewSubmission,
  reviewerId: string
): Promise<EvidenceItem> => {
  const { data } = await api.post(
    `/evidence/review/${evidenceId}/review`,
    reviewData,
    {
      params: { reviewer_id: reviewerId },
    }
  )
  return data
}

export const skipEvidence = async (
  evidenceId: number,
  reviewerId: string,
  reason: string = 'Too ambiguous to review'
): Promise<EvidenceItem> => {
  const { data } = await api.post(
    `/evidence/review/${evidenceId}/skip`,
    null,
    {
      params: { reviewer_id: reviewerId, reason },
    }
  )
  return data
}

// ============================================================================
// BATCH OPERATIONS
// ============================================================================

export interface BatchReviewItem {
  evidence_id: number
  decision: 'approved' | 'corrected' | 'rejected' | 'skipped'
  notes?: string
}

export interface BatchReviewRequest {
  reviewer_id: string
  reviews: BatchReviewItem[]
}

export const submitBatchReview = async (
  batchReview: BatchReviewRequest
): Promise<{ success: boolean; results: EvidenceItem[] }> => {
  const { data } = await api.post('/evidence/review/batch', batchReview)
  return data
}

// ============================================================================
// PROGRESS TRACKING
// ============================================================================

export const getProgress = async (videoId: string): Promise<ReviewProgress> => {
  const { data } = await api.get(`/evidence/review/progress/${videoId}`)
  return data.progress || data
}

// ============================================================================
// REVIEWER STATS
// ============================================================================

export interface GetStatsParams {
  date?: string // YYYY-MM-DD format
}

export const getReviewerStats = async (
  reviewerId: string,
  params?: GetStatsParams
): Promise<ReviewerStats> => {
  const { data } = await api.get(`/evidence/review/stats/${reviewerId}`, { params })
  return data.stats || data
}

// ============================================================================
// SESSION MANAGEMENT
// ============================================================================

export const startSession = async (
  reviewerId: string,
  sessionType: 'batch' | 'sequential'
): Promise<ReviewSession> => {
  const { data } = await api.post('/evidence/review/session/start', null, {
    params: { reviewer_id: reviewerId, session_type: sessionType },
  })
  return data
}

export const endSession = async (
  sessionId: number,
  itemsReviewed: number
): Promise<ReviewSession> => {
  const { data } = await api.post(`/evidence/review/session/${sessionId}/end`, null, {
    params: { items_reviewed: itemsReviewed },
  })
  return data
}

export default {
  getReviewQueue,
  getEvidence,
  submitReview,
  skipEvidence,
  submitBatchReview,
  getProgress,
  getReviewerStats,
  startSession,
  endSession,
}
