import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import { uploadVideo, listUploadedVideos, deleteVideo, startVideoProcessing } from '../api/client'
import { toast } from 'sonner'

/**
 * Single Video Upload Page
 *
 * Features:
 * - Drag-and-drop video upload
 * - Upload progress tracking
 * - Title and description metadata
 * - Auto-start pipeline option
 * - List of previously uploaded videos
 * - Video management (start processing, delete)
 */
export default function VideoUpload() {
  const navigate = useNavigate()
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [autoStart, setAutoStart] = useState(true)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)

  // Fetch list of uploaded videos
  const { data: videosData, refetch: refetchVideos } = useQuery({
    queryKey: ['uploaded-videos'],
    queryFn: () => listUploadedVideos(),
    refetchInterval: 5000, // Refresh every 5 seconds
  })

  const uploadMutation = useMutation({
    mutationFn: async () => {
      if (!videoFile) throw new Error('No file selected')

      setIsUploading(true)
      setUploadProgress(0)

      return uploadVideo(
        videoFile,
        title || undefined,
        description || undefined,
        autoStart,
        setUploadProgress
      )
    },
    onSuccess: (data) => {
      toast.success(data.message || 'Video uploaded successfully!')

      // Reset form
      setVideoFile(null)
      setTitle('')
      setDescription('')
      setUploadProgress(0)
      setIsUploading(false)

      // Refresh video list
      refetchVideos()

      // Navigate to processing monitor if auto-start is enabled
      if (autoStart) {
        navigate(`/monitor/video/${data.video_id}`)
      }
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || error.message || 'Upload failed')
      setIsUploading(false)
      setUploadProgress(0)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteVideo,
    onSuccess: () => {
      toast.success('Video deleted successfully')
      refetchVideos()
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to delete video')
    },
  })

  const startProcessingMutation = useMutation({
    mutationFn: startVideoProcessing,
    onSuccess: (data) => {
      toast.success('Processing started')
      refetchVideos()
      navigate(`/monitor/video/${data.video_id}`)
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to start processing')
    },
  })

  const handleFileSelect = (file: File) => {
    // Validate file type
    const validTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/webm']
    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv|webm)$/i)) {
      toast.error('Invalid file type. Please upload a video file (mp4, avi, mov, mkv, webm)')
      return
    }

    // Check file size (500MB max)
    const maxSize = 500 * 1024 * 1024
    if (file.size > maxSize) {
      toast.error('File too large. Maximum size is 500MB')
      return
    }

    setVideoFile(file)

    // Auto-populate title from filename if not set
    if (!title) {
      const filename = file.name.replace(/\.[^/.]+$/, '') // Remove extension
      setTitle(filename)
    }

    toast.success(`File "${file.name}" selected`)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleSubmit = () => {
    if (!videoFile) {
      toast.error('Please select a video file')
      return
    }
    uploadMutation.mutate()
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
  }

  const getStatusColor = (stage: string) => {
    const colors: Record<string, string> = {
      'uploaded': 'bg-gray-500',
      'generating': 'bg-blue-500',
      'awaiting_stage1_review': 'bg-yellow-500',
      'validating': 'bg-purple-500',
      'testing_gemini': 'bg-orange-500',
      'awaiting_stage2_selection': 'bg-yellow-600',
      'completed': 'bg-green-500',
      'failed': 'bg-red-500',
    }
    return colors[stage] || 'bg-gray-400'
  }

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-8">
      <h1 className="text-3xl font-bold">Upload Video</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div className="space-y-6">
          {/* Drag & Drop Zone */}
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">Select Video File</h2>

            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors cursor-pointer"
            >
              <input
                type="file"
                accept="video/mp4,video/avi,video/mov,video/mkv,video/webm"
                onChange={handleFileInput}
                className="hidden"
                id="video-upload"
                disabled={isUploading}
              />
              <label htmlFor="video-upload" className="cursor-pointer">
                {videoFile ? (
                  <div>
                    <div className="text-green-600 text-lg mb-2">✓ {videoFile.name}</div>
                    <div className="text-gray-500 text-sm">{formatFileSize(videoFile.size)}</div>
                    {!isUploading && (
                      <div className="text-gray-500 text-xs mt-2">Click or drag to replace</div>
                    )}
                  </div>
                ) : (
                  <div>
                    <div className="text-gray-600 text-lg mb-2">🎥 Drop video file here or click to upload</div>
                    <div className="text-gray-500 text-sm">Supported: MP4, AVI, MOV, MKV, WebM</div>
                    <div className="text-gray-500 text-xs mt-1">Max size: 500MB</div>
                  </div>
                )}
              </label>
            </div>

            {/* Upload Progress */}
            {isUploading && (
              <div className="mt-4">
                <div className="flex justify-between text-sm mb-1">
                  <span>Uploading...</span>
                  <span>{uploadProgress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Video Metadata */}
          <div className="bg-white p-6 rounded-lg shadow space-y-4">
            <h2 className="text-xl font-semibold">Video Details</h2>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Title
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g., Product Demo Video"
                className="w-full border border-gray-300 rounded-lg p-2"
                disabled={isUploading}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description (Optional)
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Add a description..."
                className="w-full h-24 border border-gray-300 rounded-lg p-2"
                disabled={isUploading}
              />
            </div>

            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="auto-start"
                checked={autoStart}
                onChange={(e) => setAutoStart(e.target.checked)}
                className="w-4 h-4"
                disabled={isUploading}
              />
              <label htmlFor="auto-start" className="text-sm text-gray-700">
                Auto-start processing after upload
              </label>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-4">
            <button
              onClick={() => navigate('/')}
              disabled={isUploading}
              className="flex-1 bg-gray-200 hover:bg-gray-300 text-gray-700 px-6 py-3 rounded-lg disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={!videoFile || isUploading}
              className="flex-1 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isUploading ? `⏳ Uploading... ${uploadProgress}%` : '🚀 Upload Video'}
            </button>
          </div>
        </div>

        {/* Recent Uploads */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Recent Uploads</h2>

          <div className="space-y-3 max-h-[600px] overflow-y-auto">
            {videosData?.videos?.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                No videos uploaded yet
              </div>
            ) : (
              videosData?.videos?.map((video: any) => (
                <div
                  key={video.id}
                  className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
                >
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex-1">
                      <div className="font-medium text-gray-900">{video.title}</div>
                      <div className="text-sm text-gray-500">{video.filename}</div>
                    </div>
                    <span className={`px-2 py-1 text-xs text-white rounded ${getStatusColor(video.pipeline_stage)}`}>
                      {video.pipeline_stage}
                    </span>
                  </div>

                  <div className="flex justify-between items-center text-xs text-gray-500 mb-3">
                    <span>{video.file_size_mb} MB</span>
                    <span>{new Date(video.created_at).toLocaleString()}</span>
                  </div>

                  <div className="flex gap-2">
                    {video.pipeline_stage === 'uploaded' && (
                      <button
                        onClick={() => startProcessingMutation.mutate(video.id)}
                        className="flex-1 bg-green-600 hover:bg-green-700 text-white text-sm px-3 py-1 rounded"
                      >
                        Start Processing
                      </button>
                    )}
                    {['generating', 'validating', 'testing_gemini'].includes(video.pipeline_stage) && (
                      <button
                        onClick={() => navigate(`/monitor/video/${video.id}`)}
                        className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm px-3 py-1 rounded"
                      >
                        View Progress
                      </button>
                    )}
                    {video.pipeline_stage === 'completed' && (
                      <button
                        onClick={() => navigate(`/videos/${video.id}/results`)}
                        className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm px-3 py-1 rounded"
                      >
                        View Results
                      </button>
                    )}
                    <button
                      onClick={() => {
                        if (window.confirm('Are you sure you want to delete this video?')) {
                          deleteMutation.mutate(video.id)
                        }
                      }}
                      className="bg-red-600 hover:bg-red-700 text-white text-sm px-3 py-1 rounded"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
