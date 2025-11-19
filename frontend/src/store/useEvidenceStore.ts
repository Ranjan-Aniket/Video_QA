import { create } from 'zustand'
import type {
  EvidenceItem,
  ReviewProgress,
  ReviewerStats,
  ReviewSession,
  EvidenceFilters,
} from '../types/evidence'

interface EvidenceState {
  // Evidence Queue
  evidenceQueue: EvidenceItem[]
  currentEvidence: EvidenceItem | null
  currentIndex: number

  // Progress & Stats
  reviewProgress: ReviewProgress | null
  reviewerStats: ReviewerStats | null

  // Session
  currentSession: ReviewSession | null

  // Filters
  filters: EvidenceFilters

  // Loading States
  isLoading: boolean
  error: string | null

  // Reviewer ID
  reviewerId: string

  // Actions
  setEvidenceQueue: (queue: EvidenceItem[]) => void
  setCurrentEvidence: (evidence: EvidenceItem | null) => void
  setCurrentIndex: (index: number) => void
  nextEvidence: () => void
  previousEvidence: () => void
  setReviewProgress: (progress: ReviewProgress | null) => void
  setReviewerStats: (stats: ReviewerStats | null) => void
  setCurrentSession: (session: ReviewSession | null) => void
  setFilters: (filters: EvidenceFilters) => void
  setIsLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  setReviewerId: (id: string) => void
  updateEvidenceInQueue: (evidenceId: number, updates: Partial<EvidenceItem>) => void
  removeFromQueue: (evidenceId: number) => void
  reset: () => void
}

const initialState = {
  evidenceQueue: [],
  currentEvidence: null,
  currentIndex: 0,
  reviewProgress: null,
  reviewerStats: null,
  currentSession: null,
  filters: {},
  isLoading: false,
  error: null,
  reviewerId: 'test_reviewer_001', // Default reviewer ID
}

export const useEvidenceStore = create<EvidenceState>((set, get) => ({
  ...initialState,

  setEvidenceQueue: (queue) => set({ evidenceQueue: queue }),

  setCurrentEvidence: (evidence) => set({ currentEvidence: evidence }),

  setCurrentIndex: (index) => {
    const { evidenceQueue } = get()
    if (index >= 0 && index < evidenceQueue.length) {
      set({
        currentIndex: index,
        currentEvidence: evidenceQueue[index],
      })
    }
  },

  nextEvidence: () => {
    const { currentIndex, evidenceQueue } = get()
    if (currentIndex < evidenceQueue.length - 1) {
      const newIndex = currentIndex + 1
      set({
        currentIndex: newIndex,
        currentEvidence: evidenceQueue[newIndex],
      })
    }
  },

  previousEvidence: () => {
    const { currentIndex, evidenceQueue } = get()
    if (currentIndex > 0) {
      const newIndex = currentIndex - 1
      set({
        currentIndex: newIndex,
        currentEvidence: evidenceQueue[newIndex],
      })
    }
  },

  setReviewProgress: (progress) => set({ reviewProgress: progress }),

  setReviewerStats: (stats) => set({ reviewerStats: stats }),

  setCurrentSession: (session) => set({ currentSession: session }),

  setFilters: (filters) => set({ filters }),

  setIsLoading: (loading) => set({ isLoading: loading }),

  setError: (error) => set({ error }),

  setReviewerId: (id) => set({ reviewerId: id }),

  updateEvidenceInQueue: (evidenceId, updates) => {
    const { evidenceQueue, currentEvidence } = get()
    const updatedQueue = evidenceQueue.map((item) =>
      item.id === evidenceId ? { ...item, ...updates } : item
    )
    set({
      evidenceQueue: updatedQueue,
      currentEvidence:
        currentEvidence?.id === evidenceId
          ? { ...currentEvidence, ...updates }
          : currentEvidence,
    })
  },

  removeFromQueue: (evidenceId) => {
    const { evidenceQueue } = get()
    const filteredQueue = evidenceQueue.filter((item) => item.id !== evidenceId)
    set({ evidenceQueue: filteredQueue })
  },

  reset: () => set(initialState),
}))
