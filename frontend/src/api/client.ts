import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : 'http://localhost:8000/api'

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Dashboard
export const getDashboardStats = async () => {
  const { data } = await api.get('/dashboard/stats')
  return data
}

// Batch Management
export const uploadBatch = async (formData: FormData) => {
  const { data } = await api.post('/batches/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export const getBatchDetails = async (batchId: number) => {
  const { data } = await api.get(`/batches/${batchId}`)
  return data
}

export const pauseBatch = async (batchId: number) => {
  const { data } = await api.post(`/batches/${batchId}/pause`)
  return data
}

export const retryFailedVideos = async (batchId: number) => {
  const { data } = await api.post(`/batches/${batchId}/retry`)
  return data
}

export const deleteBatch = async (batchId: number) => {
  await api.delete(`/batches/${batchId}`)
}

// Processing
export const getProcessingStatus = async (batchId: number) => {
  const { data } = await api.get(`/processing/status/${batchId}`)
  return data
}

// Video Results
export const getVideoResults = async (videoId: number) => {
  const { data } = await api.get(`/videos/${videoId}/results`)
  return data
}

// Analytics
export const getAnalytics = async (dateRange: { start: string; end: string }) => {
  const { data } = await api.get('/analytics', { params: dateRange })
  return data
}

// Settings
export const getSettings = async () => {
  const { data } = await api.get('/settings')
  return data
}

export const updateSettings = async (settings: any) => {
  const { data } = await api.put('/settings', settings)
  return data
}

// Export
export const exportBatchExcel = async (batchId: number, format: string) => {
  const { data } = await api.get(`/batches/${batchId}/export`, {
    params: { format },
    responseType: 'blob',
  })

  // Trigger download
  const url = window.URL.createObjectURL(new Blob([data]))
  const link = document.createElement('a')
  link.href = url
  link.setAttribute('download', `batch_${batchId}.${format}`)
  document.body.appendChild(link)
  link.click()
  link.remove()
}

// Video Upload
export const uploadVideo = async (
  file: File,
  title?: string,
  description?: string,
  autoStart: boolean = true,
  onProgress?: (progress: number) => void
) => {
  const formData = new FormData()
  formData.append('file', file)
  if (title) formData.append('title', title)
  if (description) formData.append('description', description)
  formData.append('auto_start', autoStart.toString())

  const { data } = await api.post('/upload/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        onProgress(progress)
      }
    },
  })
  return data
}

export const listUploadedVideos = async (stage?: string) => {
  const { data } = await api.get('/upload/list', {
    params: stage ? { stage } : {},
  })
  return data
}

export const startVideoProcessing = async (videoId: number) => {
  const { data } = await api.post(`/upload/start/${videoId}`)
  return data
}

export const deleteVideo = async (videoId: number) => {
  await api.delete(`/upload/${videoId}`)
}

export const getVideoStatus = async (videoId: number) => {
  const { data } = await api.get(`/upload/status/${videoId}`)
  return data
}

// Smart Pipeline
export const runSmartPipeline = async (
  videoId: string,
  enableGpt4: boolean = false,
  enableClaude: boolean = false,
  enableGemini: boolean = false
) => {
  const { data } = await api.post('/smart-pipeline/run', {
    video_id: videoId,
    enable_gpt4: enableGpt4,
    enable_claude: enableClaude,
    enable_gemini: enableGemini,
  })
  return data
}

export const getSmartPipelineStatus = async (videoId: string) => {
  const { data } = await api.get(`/smart-pipeline/status/${videoId}`)
  return data
}

export const getAudioAnalysis = async (videoId: string) => {
  const { data } = await api.get(`/smart-pipeline/audio/${videoId}`)
  return data
}

export const getOpportunities = async (videoId: string) => {
  const { data } = await api.get(`/smart-pipeline/opportunities/${videoId}`)
  return data
}

// Legacy function (deprecated - use getOpportunities instead)
export const getGenreAnalysis = async (videoId: string) => {
  return getOpportunities(videoId)
}

export const getFrameExtraction = async (videoId: string) => {
  const { data } = await api.get(`/smart-pipeline/frames/${videoId}`)
  return data
}

export const getSmartPipelineQuestions = async (videoId: string) => {
  const { data } = await api.get(`/smart-pipeline/questions/${videoId}`)
  return data
}

export const getGeminiResults = async (videoId: string) => {
  const { data } = await api.get(`/smart-pipeline/gemini-results/${videoId}`)
  return data
}

// Legacy function (deprecated - use getGeminiResults instead)
export const getSmartPipelineValidation = async (videoId: string) => {
  return getGeminiResults(videoId)
}

export const getFullTranscript = async (videoId: string) => {
  const { data } = await api.get(`/smart-pipeline/transcript/${videoId}`)
  return data
}

export const deleteSmartPipelineResults = async (videoId: string) => {
  const { data } = await api.delete(`/smart-pipeline/${videoId}`)
  return data
}

export default api