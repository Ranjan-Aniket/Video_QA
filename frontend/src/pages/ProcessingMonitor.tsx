import { useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getProcessingStatus, pauseBatch } from '../api/client'
import { useWebSocket } from '../hooks/useWebSocket'
import { useVideoStore } from '../store/useVideoStore'
import { toast } from 'sonner'

/**
 * Processing Monitor Page
 * 
 * Following EXACT design from architecture:
 * - Overall progress bar (73/100 videos)
 * - Estimated completion time
 * - Current processing grid (10 parallel workers)
 * - Real-time WebSocket updates
 * - Recent completions list
 * - Live stats
 */
export default function ProcessingMonitor() {
  const { batchId } = useParams<{ batchId: string }>()
  const navigate = useNavigate()
  const processingVideos = useVideoStore(state => state.processingVideos)
  const setProcessingVideos = useVideoStore(state => state.setProcessingVideos)

  // WebSocket connection for real-time updates
  const { isConnected } = useWebSocket({
    videoId: batchId || '',
  })

  // Fetch processing status
  const { data: status, refetch } = useQuery({
    queryKey: ['processing-status', batchId],
    queryFn: () => getProcessingStatus(parseInt(batchId!)),
    enabled: !!batchId,
    refetchInterval: 5000, // Fallback polling every 5s if WebSocket fails
  })

  useEffect(() => {
    if (status?.videos_processing) {
      setProcessingVideos(status.videos_processing)
    }
  }, [status, setProcessingVideos])

  const handlePause = async () => {
    try {
      await pauseBatch(parseInt(batchId!))
      toast.success('Batch paused')
      refetch()
    } catch (error: any) {
      toast.error(error.message || 'Failed to pause batch')
    }
  }

  const progressPercent = status ? (status.completed / status.total) * 100 : 0

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">{status?.batch_name || 'Processing Monitor'}</h1>
          <div className="flex items-center gap-2 mt-2">
            <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-sm text-gray-600">
              {isConnected ? 'Live updates active' : 'Reconnecting...'}
            </span>
          </div>
        </div>
        <div className="flex gap-3">
          <button
            onClick={handlePause}
            className="bg-yellow-600 hover:bg-yellow-700 text-white px-4 py-2 rounded-lg"
          >
            ⏸️ Pause
          </button>
          <button
            onClick={() => navigate('/')}
            className="bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded-lg"
          >
            ← Back
          </button>
        </div>
      </div>

      {/* Overall Progress */}
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-xl font-semibold">Overall Progress</h2>
          <span className="text-gray-600">
            {status?.completed || 0} / {status?.total || 0} videos
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-4">
          <div 
            className="bg-blue-600 h-4 rounded-full transition-all duration-300"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
        <div className="flex justify-between text-sm text-gray-600 mt-2">
          <span>{progressPercent.toFixed(1)}% complete</span>
          <span>Est. {status?.estimated_time_remaining || 'calculating...'}</span>
        </div>
      </div>

      {/* Live Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg shadow">
          <div className="text-gray-500 text-sm">Success</div>
          <div className="text-2xl font-bold text-green-600">{status?.successful || 0}</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <div className="text-gray-500 text-sm">Failed</div>
          <div className="text-2xl font-bold text-red-600">{status?.failed || 0}</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <div className="text-gray-500 text-sm">Avg Time</div>
          <div className="text-2xl font-bold">{status?.avg_time_per_video || '-'}</div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <div className="text-gray-500 text-sm">Active Workers</div>
          <div className="text-2xl font-bold">{processingVideos?.length || 0}/{status?.max_workers || 10}</div>
        </div>
      </div>

      {/* Current Processing Grid */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Processing Now ({processingVideos?.length || 0})</h2>
        <div className="grid grid-cols-2 gap-4">
          {processingVideos?.map((video: any) => (
            <div key={video.id} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
              <div className="flex gap-3">
                <img 
                  src={video.thumbnail || '/placeholder.jpg'} 
                  alt={video.title}
                  className="w-24 h-16 object-cover rounded"
                />
                <div className="flex-1">
                  <div className="font-medium text-sm truncate">{video.title}</div>
                  <div className="text-xs text-gray-500 mt-1">
                    Stage: {video.current_stage || 'Starting...'}
                  </div>
                  <div className="text-xs text-gray-600 mt-1">
                    {video.progress_text || 'Initializing...'}
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${video.progress_percent || 0}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          ))}
          {(!processingVideos || processingVideos.length === 0) && (
            <div className="col-span-2 text-center text-gray-500 py-8">
              No videos currently processing
            </div>
          )}
        </div>
      </div>

      {/* Recent Completions */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Recent Completions</h2>
        <table className="w-full">
          <thead className="border-b">
            <tr className="text-left text-gray-600 text-sm">
              <th className="pb-2">Video</th>
              <th className="pb-2">Status</th>
              <th className="pb-2">Questions</th>
              <th className="pb-2">Time</th>
              <th className="pb-2">Cost</th>
            </tr>
          </thead>
          <tbody>
            {status?.recent_completions?.map((video: any) => (
              <tr key={video.id} className="border-b hover:bg-gray-50">
                <td className="py-3 text-sm">{video.title}</td>
                <td className="py-3">
                  <span className={`px-2 py-1 rounded text-xs ${
                    video.status === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                  }`}>
                    {video.status}
                  </span>
                </td>
                <td className="py-3 text-sm">{video.questions_generated || 0}</td>
                <td className="py-3 text-sm">{video.processing_time || '-'}</td>
                <td className="py-3 text-sm">${(video.cost || 0).toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}