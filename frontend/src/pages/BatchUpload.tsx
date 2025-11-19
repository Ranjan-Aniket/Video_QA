import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { uploadBatch } from '../api/client'
import { toast } from 'sonner'

/**
 * Batch Upload Page
 * 
 * Following EXACT design from architecture:
 * - CSV upload with drag-and-drop
 * - Google Drive link input
 * - Batch configuration (name, priority, workers, auto-start, quality threshold)
 * - Validation preview
 * - Cost estimate
 * - [Start Processing] button
 */
export default function BatchUpload() {
  const navigate = useNavigate()
  const [csvFile, setCsvFile] = useState<File | null>(null)
  const [driveUrls, setDriveUrls] = useState('')
  const [batchName, setBatchName] = useState('')
  const [parallelWorkers, setParallelWorkers] = useState(10)
  const [autoStart, setAutoStart] = useState(true)
  const [qualityThreshold, setQualityThreshold] = useState(0.81)
  const [uploading, setUploading] = useState(false)

  const uploadMutation = useMutation({
    mutationFn: uploadBatch,
    onSuccess: (data) => {
      toast.success('Batch created successfully!')
      navigate(`/monitor/${data.batch_id}`)
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to create batch')
    },
  })

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
        toast.error('Please upload a CSV file')
        return
      }
      setCsvFile(file)
      toast.success(`File "${file.name}" loaded`)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) {
      if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
        toast.error('Please upload a CSV file')
        return
      }
      setCsvFile(file)
      toast.success(`File "${file.name}" loaded`)
    }
  }

  const handleSubmit = async () => {
    if (!csvFile && !driveUrls.trim()) {
      toast.error('Please provide either a CSV file or Google Drive URLs')
      return
    }

    setUploading(true)
    const formData = new FormData()
    
    if (csvFile) {
      formData.append('file', csvFile)
    }
    
    formData.append('batch_name', batchName || `Batch ${new Date().toISOString()}`)
    formData.append('drive_urls', driveUrls)
    formData.append('parallel_workers', parallelWorkers.toString())
    formData.append('auto_start', autoStart.toString())
    formData.append('quality_threshold', qualityThreshold.toString())

    uploadMutation.mutate(formData as any)
    setUploading(false)
  }

  const estimatedVideos = csvFile ? 'Analyzing...' : driveUrls.split('\n').filter(u => u.trim()).length
  const estimatedCost = typeof estimatedVideos === 'number' ? `$${(estimatedVideos * 5.5).toFixed(2)}` : '-'
  const estimatedRevenue = typeof estimatedVideos === 'number' ? `$${(estimatedVideos * 8).toFixed(2)}` : '-'

  return (
    <div className="p-8 max-w-4xl mx-auto space-y-8">
      <h1 className="text-3xl font-bold">Create New Batch</h1>

      {/* CSV Upload */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Upload Video List</h2>
        
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors cursor-pointer"
        >
          <input
            type="file"
            accept=".csv"
            onChange={handleFileUpload}
            className="hidden"
            id="csv-upload"
          />
          <label htmlFor="csv-upload" className="cursor-pointer">
            {csvFile ? (
              <div>
                <div className="text-green-600 text-lg">✓ {csvFile.name}</div>
                <div className="text-gray-500 text-sm mt-2">Click or drag to replace</div>
              </div>
            ) : (
              <div>
                <div className="text-gray-600 text-lg">📄 Drop CSV file here or click to upload</div>
                <div className="text-gray-500 text-sm mt-2">CSV must have columns: video_url, title</div>
              </div>
            )}
          </label>
        </div>

        <div className="mt-4 text-center text-gray-500">— OR —</div>

        {/* Google Drive URLs */}
        <div className="mt-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Google Drive URLs (one per line)
          </label>
          <textarea
            value={driveUrls}
            onChange={(e) => setDriveUrls(e.target.value)}
            placeholder="https://drive.google.com/file/d/...&#10;https://drive.google.com/file/d/..."
            className="w-full h-32 border border-gray-300 rounded-lg p-3 text-sm"
          />
        </div>
      </div>

      {/* Batch Configuration */}
      <div className="bg-white p-6 rounded-lg shadow space-y-4">
        <h2 className="text-xl font-semibold">Batch Configuration</h2>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Batch Name</label>
          <input
            type="text"
            value={batchName}
            onChange={(e) => setBatchName(e.target.value)}
            placeholder="e.g., Healthcare Videos - Nov 2024"
            className="w-full border border-gray-300 rounded-lg p-2"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Parallel Workers: {parallelWorkers}
          </label>
          <input
            type="range"
            min="1"
            max="10"
            value={parallelWorkers}
            onChange={(e) => setParallelWorkers(parseInt(e.target.value))}
            className="w-full"
          />
          <div className="text-xs text-gray-500 mt-1">More workers = faster but higher costs</div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Quality Threshold: {qualityThreshold}
          </label>
          <input
            type="range"
            min="0.5"
            max="0.99"
            step="0.01"
            value={qualityThreshold}
            onChange={(e) => setQualityThreshold(parseFloat(e.target.value))}
            className="w-full"
          />
          <div className="text-xs text-gray-500 mt-1">Minimum validation pass rate per question</div>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="auto-start"
            checked={autoStart}
            onChange={(e) => setAutoStart(e.target.checked)}
            className="w-4 h-4"
          />
          <label htmlFor="auto-start" className="text-sm text-gray-700">
            Auto-start processing after upload
          </label>
        </div>
      </div>

      {/* Cost Estimate */}
      <div className="bg-blue-50 p-6 rounded-lg">
        <h2 className="text-xl font-semibold mb-3">Estimate</h2>
        <div className="grid grid-cols-3 gap-4 text-sm">
          <div>
            <div className="text-gray-600">Videos</div>
            <div className="text-lg font-semibold">{estimatedVideos}</div>
          </div>
          <div>
            <div className="text-gray-600">Estimated Cost</div>
            <div className="text-lg font-semibold text-red-600">{estimatedCost}</div>
          </div>
          <div>
            <div className="text-gray-600">Estimated Revenue</div>
            <div className="text-lg font-semibold text-green-600">{estimatedRevenue}</div>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-4">
        <button
          onClick={() => navigate('/')}
          className="flex-1 bg-gray-200 hover:bg-gray-300 text-gray-700 px-6 py-3 rounded-lg"
        >
          Cancel
        </button>
        <button
          onClick={handleSubmit}
          disabled={uploading || (!csvFile && !driveUrls.trim())}
          className="flex-1 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {uploading ? '⏳ Creating...' : '🚀 Create Batch'}
        </button>
      </div>
    </div>
  )
}