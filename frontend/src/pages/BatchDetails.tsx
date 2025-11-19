import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { getBatchDetails, retryFailedVideos, deleteBatch, exportBatchExcel } from '../api/client'
import { toast } from 'sonner'

/**
 * Batch Details Page
 * 
 * Following EXACT design from architecture:
 * - Batch metadata
 * - Video list with status
 * - Bulk actions (retry failed, re-export, delete batch)
 * - Batch-level analytics
 * - Export options (Excel, CSV, JSON, Google Sheets)
 */
export default function BatchDetails() {
  const { batchId } = useParams<{ batchId: string }>()
  
  const { data: batch, refetch } = useQuery({
    queryKey: ['batch-details', batchId],
    queryFn: () => getBatchDetails(parseInt(batchId!)),
    enabled: !!batchId,
  })

  const retryMutation = useMutation({
    mutationFn: () => retryFailedVideos(parseInt(batchId!)),
    onSuccess: () => {
      toast.success('Retrying failed videos')
      refetch()
    },
  })

  const deleteMutation = useMutation({
    mutationFn: () => deleteBatch(parseInt(batchId!)),
    onSuccess: () => {
      toast.success('Batch deleted')
      window.location.href = '/'
    },
  })

  const handleExport = async (format: 'excel' | 'csv' | 'json') => {
    try {
      await exportBatchExcel(parseInt(batchId!), format)
      toast.success(`Exported as ${format.toUpperCase()}`)
    } catch (error) {
      toast.error('Export failed')
    }
  }

  if (!batch) {
    return <div className="p-8">Loading...</div>
  }

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">{batch.name}</h1>
          <div className="text-gray-600 mt-1">Batch ID: {batchId}</div>
        </div>
        <Link to="/">
          <button className="bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded-lg">
            ← Back
          </button>
        </Link>
      </div>

      {/* Batch Metadata */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Batch Information</h2>
        <div className="grid grid-cols-4 gap-4">
          <div>
            <div className="text-gray-500 text-sm">Status</div>
            <div className="text-lg font-medium">
              <span className={`px-3 py-1 rounded ${
                batch.status === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                batch.status === 'completed' ? 'bg-green-100 text-green-800' :
                'bg-red-100 text-red-800'
              }`}>
                {batch.status}
              </span>
            </div>
          </div>
          <div>
            <div className="text-gray-500 text-sm">Created</div>
            <div className="text-lg font-medium">{new Date(batch.created_at).toLocaleDateString()}</div>
          </div>
          <div>
            <div className="text-gray-500 text-sm">Completed</div>
            <div className="text-lg font-medium">{batch.completed_at ? new Date(batch.completed_at).toLocaleDateString() : 'In progress'}</div>
          </div>
          <div>
            <div className="text-gray-500 text-sm">Workers</div>
            <div className="text-lg font-medium">{batch.parallel_workers}</div>
          </div>
        </div>
      </div>

      {/* Batch Analytics */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Analytics</h2>
        <div className="grid grid-cols-5 gap-4">
          <div>
            <div className="text-gray-500 text-sm">Total Videos</div>
            <div className="text-2xl font-bold">{batch.total_videos}</div>
          </div>
          <div>
            <div className="text-gray-500 text-sm">Completed</div>
            <div className="text-2xl font-bold text-green-600">{batch.completed_count}</div>
          </div>
          <div>
            <div className="text-gray-500 text-sm">Failed</div>
            <div className="text-2xl font-bold text-red-600">{batch.failed_count}</div>
          </div>
          <div>
            <div className="text-gray-500 text-sm">Total Cost</div>
            <div className="text-2xl font-bold">${batch.total_cost?.toFixed(2)}</div>
          </div>
          <div>
            <div className="text-gray-500 text-sm">Success Rate</div>
            <div className="text-2xl font-bold">{batch.success_rate?.toFixed(1)}%</div>
          </div>
        </div>
      </div>

      {/* Bulk Actions */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Bulk Actions</h2>
        <div className="flex gap-3">
          <button
            onClick={() => retryMutation.mutate()}
            disabled={batch.failed_count === 0}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg disabled:opacity-50"
          >
            Retry Failed ({batch.failed_count})
          </button>
          <button
            onClick={() => handleExport('excel')}
            className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg"
          >
            Export to Excel
          </button>
          <button
            onClick={() => handleExport('csv')}
            className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg"
          >
            Export to CSV
          </button>
          <button
            onClick={() => handleExport('json')}
            className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg"
          >
            Export to JSON
          </button>
          <button
            onClick={() => deleteMutation.mutate()}
            className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg ml-auto"
          >
            Delete Batch
          </button>
        </div>
      </div>

      {/* Video List */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Videos ({batch.videos?.length})</h2>
        <table className="w-full">
          <thead className="border-b">
            <tr className="text-left text-gray-600 text-sm">
              <th className="pb-3">Video</th>
              <th className="pb-3">Status</th>
              <th className="pb-3">Questions</th>
              <th className="pb-3">Success Rate</th>
              <th className="pb-3">Cost</th>
              <th className="pb-3">Time</th>
              <th className="pb-3">Actions</th>
            </tr>
          </thead>
          <tbody>
            {batch.videos?.map((video: any) => (
              <tr key={video.id} className="border-b hover:bg-gray-50">
                <td className="py-3 text-sm max-w-xs truncate">{video.title}</td>
                <td className="py-3">
                  <span className={`px-2 py-1 rounded text-xs ${
                    video.status === 'completed' ? 'bg-green-100 text-green-800' :
                    video.status === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {video.status}
                  </span>
                </td>
                <td className="py-3 text-sm">{video.questions_generated || 0}</td>
                <td className="py-3 text-sm">{video.success_rate?.toFixed(1)}%</td>
                <td className="py-3 text-sm">${video.cost?.toFixed(2)}</td>
                <td className="py-3 text-sm">{video.processing_time || '-'}</td>
                <td className="py-3">
                  {video.status === 'completed' && (
                    <Link to={`/video/${video.id}`} className="text-blue-600 hover:underline text-sm">
                      View →
                    </Link>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}