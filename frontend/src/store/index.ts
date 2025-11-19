import { create } from 'zustand'

// Batch Store
interface BatchState {
  currentBatch: any | null
  batches: any[]
  setCurrentBatch: (batch: any) => void
  setBatches: (batches: any[]) => void
}

export const useBatchStore = create<BatchState>((set) => ({
  currentBatch: null,
  batches: [],
  setCurrentBatch: (batch: any) => set({ currentBatch: batch }),
  setBatches: (batches: any[]) => set({ batches }),
}))

// Video Store
interface VideoState {
  processingVideos: any[]
  completedVideos: any[]
  setProcessingVideos: (videos: any[]) => void
  addCompletedVideo: (video: any) => void
}

export const useVideoStore = create<VideoState>((set) => ({
  processingVideos: [],
  completedVideos: [],
  setProcessingVideos: (videos) => set({ processingVideos: videos }),
  addCompletedVideo: (video) => set((state) => ({
    completedVideos: [...state.completedVideos, video],
  })),
}))

// Settings Store
interface SettingsState {
  settings: any
  setSettings: (settings: any) => void
}

export const useSettingsStore = create<SettingsState>((set) => ({
  settings: {},
  setSettings: (settings) => set({ settings }),
}))