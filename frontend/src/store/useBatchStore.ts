import { create } from 'zustand'

interface BatchState {
  currentBatch: any | null
  batches: any[]
  setCurrentBatch: (batch: any) => void
  setBatches: (batches: any[]) => void
}

export const useBatchStore = create<BatchState>((set) => ({
  currentBatch: null,
  batches: [],
  setCurrentBatch: (batch) => set({ currentBatch: batch }),
  setBatches: (batches) => set({ batches }),
}))