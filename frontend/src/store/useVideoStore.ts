import { create } from 'zustand'

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