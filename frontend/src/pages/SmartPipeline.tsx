import { useState, useEffect } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { toast } from 'sonner'
import {
  listUploadedVideos,
  runSmartPipeline,
  getSmartPipelineStatus,
  getAudioAnalysis,
  getFrameExtraction,
  getSmartPipelineQuestions,
  getGeminiResults,
  getFullTranscript,
} from '../api/client'

/**
 * Smart Pipeline Page - Intelligent Video Q&A Generation
 *
 * Features:
 * - Run the 9-phase smart pipeline
 * - View audio analysis with word timestamps
 * - View highlight detection and frame selection
 * - View frame extraction plan (premium + template + bulk)
 * - See generated questions with full evidence
 * - View Gemini testing results
 */
export default function SmartPipeline() {
  const [selectedVideoId, setSelectedVideoId] = useState<string>('')
  const [enableGpt4, setEnableGpt4] = useState(false)
  const [enableClaude, setEnableClaude] = useState(false)
  const [enableGemini, setEnableGemini] = useState(false)
  const [activeTab, setActiveTab] = useState<'overview' | 'audio' | 'questions' | 'gemini'>('overview')

  // Fetch list of uploaded videos
  // Poll only if we don't have a selected video yet (to detect new uploads/processing)
  const { data: videosData, isLoading: videosLoading } = useQuery({
    queryKey: ['uploaded-videos'],
    queryFn: () => listUploadedVideos(),
    refetchInterval: !selectedVideoId ? 5000 : false, // Stop polling once video selected
  })

  // Auto-select video with ongoing pipeline on mount/refresh
  useEffect(() => {
    if (!selectedVideoId && videosData?.videos?.length > 0) {
      // Find first video with 'generating' stage (processing)
      const processingVideo = videosData.videos.find(
        (v: any) => v.pipeline_stage === 'generating'
      )
      if (processingVideo) {
        setSelectedVideoId(processingVideo.video_id)
        toast.info('Resuming pipeline monitoring')
      }
    }
  }, [videosData, selectedVideoId])

  // Fetch pipeline status for selected video
  const { data: statusData, refetch: refetchStatus, isLoading: statusLoading, isFetching: statusFetching } = useQuery({
    queryKey: ['smart-pipeline-status', selectedVideoId],
    queryFn: () => getSmartPipelineStatus(selectedVideoId),
    enabled: !!selectedVideoId,
    // ✅ COST OPTIMIZATION: Only poll while pipeline is actively processing
    refetchInterval: (query) => {
      const data = query.state.data as any
      const isProcessing = data?.status === 'processing' || data?.status === 'running'
      // Poll every 1s while processing, stop when complete/error/idle
      return selectedVideoId && isProcessing ? 1000 : false
    },
  })

  // Fetch audio analysis
  const { data: audioData, isLoading: audioLoading, error: audioError } = useQuery({
    queryKey: ['audio-analysis', selectedVideoId],
    queryFn: () => getAudioAnalysis(selectedVideoId),
    enabled: !!selectedVideoId && statusData?.phases_complete?.includes('audio_analysis'),
    retry: false, // Don't retry 404s
  })

  // Fetch frame extraction
  const { data: framesData, isLoading: framesLoading, error: framesError } = useQuery({
    queryKey: ['frame-extraction', selectedVideoId],
    queryFn: () => getFrameExtraction(selectedVideoId),
    enabled: !!selectedVideoId && statusData?.phases_complete?.includes('frame_extraction'),
    retry: false,
  })

  // Fetch questions
  const { data: questionsData, isLoading: questionsLoading, error: questionsError } = useQuery({
    queryKey: ['smart-questions', selectedVideoId],
    queryFn: () => getSmartPipelineQuestions(selectedVideoId),
    enabled: !!selectedVideoId && statusData?.phases_complete?.includes('question_generation'),
    retry: false,
  })

  // Fetch Gemini testing results (optional)
  const { data: geminiData, isLoading: geminiLoading, error: geminiError } = useQuery({
    queryKey: ['gemini-results', selectedVideoId],
    queryFn: () => getGeminiResults(selectedVideoId),
    enabled: !!selectedVideoId && statusData?.phases_complete?.includes('question_generation'),
    retry: false,
  })

  // Fetch full transcript
  const { data: transcriptData, isLoading: transcriptLoading, error: transcriptError } = useQuery({
    queryKey: ['full-transcript', selectedVideoId],
    queryFn: () => getFullTranscript(selectedVideoId),
    enabled: !!selectedVideoId && statusData?.phases_complete?.includes('audio_analysis'),
    retry: false,
  })

  // Run pipeline mutation
  const runPipelineMutation = useMutation({
    mutationFn: () => runSmartPipeline(selectedVideoId, enableGpt4, enableClaude, enableGemini),
    onSuccess: () => {
      toast.success('Smart pipeline started!')
      refetchStatus()
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to start pipeline')
    },
  })

  const handleRunPipeline = () => {
    if (!selectedVideoId) {
      toast.error('Please select a video')
      return
    }
    runPipelineMutation.mutate()
  }

  const renderProgressBar = () => {
    // Show loading state while fetching initial status
    if (statusLoading && !statusData) {
      return (
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center justify-center py-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mr-3"></div>
            <span className="text-gray-600">Loading pipeline status...</span>
          </div>
        </div>
      )
    }

    if (!statusData) return null

    // Calculate progress from phases_complete (Pass 1-2B Architecture: 9 phases)
    const totalPhases = 9  // Pass 1-2B: Audio, Visual, CLIP, Pass1, Pass2A, Pass2B, Validation, Pass3, Gemini
    const phasesCompleted = statusData.phases_complete?.length || 0
    const progress = (phasesCompleted / totalPhases) * 100

    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold">Pipeline Progress</h3>
            {statusFetching && (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
            )}
          </div>
          <span className={`px-3 py-1 text-sm rounded font-medium ${
            statusData.status === 'completed' ? 'bg-green-100 text-green-800 border border-green-300' :
            statusData.status === 'processing' ? 'bg-blue-100 text-blue-800 border border-blue-300' :
            statusData.status === 'pending' ? 'bg-gray-100 text-gray-800 border border-gray-300' :
            'bg-red-100 text-red-800 border border-red-300'
          }`}>
            {statusData.status?.toUpperCase()}
          </span>
        </div>

        <div className="mb-2">
          <div className="flex justify-between text-sm text-gray-700 mb-1">
            <span className="font-medium">{statusData.current_phase || 'Initializing...'}</span>
            <span className="font-semibold text-blue-600">{progress.toFixed(0)}% ({phasesCompleted}/{totalPhases} phases)</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div
              className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${Math.max(5, progress)}%` }}
            />
          </div>
        </div>

        {statusData.phases_complete && (
          <div className="mt-6">
            <h4 className="text-sm font-semibold text-gray-700 mb-2">Phase Status (Pass 1-2B Architecture):</h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {[
                { key: 'audio_analysis', label: '1. Audio Analysis' },
                { key: 'visual_sampling', label: '2. Visual Sampling' },
                { key: 'clip_analysis', label: '3. CLIP Analysis' },
                { key: 'pass1_filter', label: '4. Pass 1 Filter' },
                { key: 'pass2a_moments', label: '5. Pass 2A (Sonnet)' },
                { key: 'pass2b_moments', label: '6. Pass 2B (Opus)' },
                { key: 'validation', label: '7. Validation' },
                { key: 'question_generation', label: '8. Pass 3 (QA)' },
                { key: 'gemini_testing', label: '9. Gemini Test' }
              ].map((phase) => {
                const isComplete = statusData.phases_complete.includes(phase.key)
                return (
                  <div
                    key={phase.key}
                    className={`p-2 rounded text-center text-xs font-medium transition-colors ${
                      isComplete
                        ? 'bg-green-100 text-green-800 border border-green-300'
                        : 'bg-gray-50 text-gray-500 border border-gray-200'
                    }`}
                  >
                    {isComplete ? '✓' : '○'} {phase.label}
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>
    )
  }

  const renderDataCard = (title: string, isLoading: boolean, error: any, data: any, content: JSX.Element) => {
    if (isLoading) {
      return (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">{title}</h3>
          <div className="flex items-center justify-center py-4">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          </div>
        </div>
      )
    }

    if (error && error.response?.status !== 404) {
      return (
        <div className="bg-white p-6 rounded-lg shadow border-l-4 border-red-500">
          <h3 className="text-lg font-semibold mb-2 text-red-600">{title}</h3>
          <p className="text-sm text-red-600">Error loading data</p>
        </div>
      )
    }

    if (!data) return null

    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-2">{title}</h3>
        {content}
      </div>
    )
  }

  const renderOverview = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4">
        {renderDataCard(
          "Audio Analysis",
          audioLoading,
          audioError,
          audioData,
          <div className="space-y-2 text-sm">
            <div><span className="font-medium">Duration:</span> {audioData?.duration.toFixed(1)}s</div>
            <div><span className="font-medium">Segments:</span> {audioData?.segments_count}</div>
            <div><span className="font-medium">Speakers:</span> {audioData?.speaker_count}</div>
            <div><span className="font-medium">Language:</span> {audioData?.language}</div>
          </div>
        )}
      </div>

      {/* Show pipeline stats from checkpoint data OR Pass 3 results */}
      {(audioData || framesData || questionsData) && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">
            {questionsData ? 'Pass 3: Question Generation Complete' : 'Pipeline Checkpoint Data'}
          </h3>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-3xl font-bold text-blue-600">
                {questionsData?.total_questions || '—'}
              </div>
              <div className="text-sm text-gray-600">Questions</div>
              <div className="text-xs text-gray-400 mt-1">
                {questionsData ? '✓ Complete' : 'Pending Pass 3'}
              </div>
            </div>
            <div>
              <div className="text-3xl font-bold text-orange-600">
                {audioData?.segments_count || '—'}
              </div>
              <div className="text-sm text-gray-600">Audio Segments</div>
              <div className="text-xs text-gray-400 mt-1">
                {audioData ? `✓ ${audioData.duration?.toFixed(0)}s` : 'Pending'}
              </div>
            </div>
            <div>
              <div className="text-3xl font-bold text-green-600">
                {questionsData?.metadata?.model_used || 'GPT-4o'}
              </div>
              <div className="text-sm text-gray-600">Model</div>
              <div className="text-xs text-gray-400 mt-1">
                {questionsData ? '✓ Used' : 'Ready'}
              </div>
            </div>
          </div>
          {questionsData?.cost_summary && (
            <div className="mt-4 text-center space-y-1 pt-4 border-t border-gray-200">
              <div className="text-sm text-gray-600">
                Total Cost: <span className="font-semibold">${questionsData.cost_summary.total_cost?.toFixed(4)}</span>
              </div>
              <div className="text-xs text-gray-500">
                Tokens: {questionsData.cost_summary.input_tokens?.toLocaleString()} in + {questionsData.cost_summary.output_tokens?.toLocaleString()} out = {questionsData.cost_summary.total_tokens?.toLocaleString()} total
              </div>
              <div className="text-xs text-gray-500">
                Generated {questionsData.cost_summary.questions_generated} questions → Selected {questionsData.total_questions}
              </div>
            </div>
          )}
        </div>
      )}

      {geminiData?.tested && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Gemini Testing Results</h3>
          <div className="text-sm text-gray-600">
            Gemini 2.0 Flash testing completed. View full results in the Gemini tab.
          </div>
        </div>
      )}
    </div>
  )

  const renderAudioTab = () => (
    <div className="space-y-6">
      {audioData && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Audio Analysis Summary</h3>
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div><span className="font-medium">Duration:</span> {audioData.duration.toFixed(1)}s</div>
            <div><span className="font-medium">Segments:</span> {audioData.segments_count}</div>
            <div><span className="font-medium">Speakers:</span> {audioData.speaker_count}</div>
            <div><span className="font-medium">Language:</span> {audioData.language}</div>
          </div>
          <div>
            <h4 className="font-medium mb-2">Transcript Preview:</h4>
            <div className="bg-gray-50 p-4 rounded text-sm max-h-60 overflow-y-auto">
              {audioData.transcript}
            </div>
          </div>
        </div>
      )}

      {transcriptData && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Full Transcript with Timestamps</h3>
          <div className="bg-gray-50 p-4 rounded max-h-96 overflow-y-auto space-y-2">
            {transcriptData.segments?.map((segment: any, idx: number) => (
              <div key={idx} className="text-sm">
                <span className="font-mono text-blue-600">[{segment.start.toFixed(1)}s - {segment.end.toFixed(1)}s]</span>
                <span className="ml-2">{segment.text}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )

  const renderFramesTab = () => (
    <div className="space-y-6">
      {framesData && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Frame Extraction Strategy</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="text-center p-4 bg-blue-50 rounded">
              <div className="text-3xl font-bold text-blue-600">{framesData.total_frames}</div>
              <div className="text-sm text-gray-600">Total Frames</div>
            </div>
            <div className="text-center p-4 bg-orange-50 rounded">
              <div className="text-3xl font-bold text-orange-600">{framesData.premium_frames}</div>
              <div className="text-sm text-gray-600">Premium (GPT-4V/Claude)</div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded">
              <div className="text-3xl font-bold text-purple-600">{framesData.template_frames}</div>
              <div className="text-sm text-gray-600">Template</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded">
              <div className="text-3xl font-bold text-green-600">{framesData.bulk_frames}</div>
              <div className="text-sm text-gray-600">Bulk (every 5s)</div>
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded">
            <h4 className="font-medium mb-2">Frame Extraction Strategy:</h4>
            <ul className="text-sm space-y-2">
              <li>🎯 <strong>{framesData.premium_frames} premium frames</strong> for GPT-4 Vision & Claude Vision (key moments)</li>
              <li>📋 <strong>{framesData.template_frames} template frames</strong> for template-based questions</li>
              <li>🔄 <strong>{framesData.bulk_frames} bulk frames</strong> extracted every 5 seconds for comprehensive coverage</li>
              <li>💰 Optimized strategy: expensive models only on critical frames</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  )

  const exportToCSV = () => {
    if (!questionsData?.questions) return

    // CSV header
    const headers = ['Question ID', 'Question', 'Answer', 'Type', 'Audio Cue', 'Visual Cue', 'Start Time', 'End Time']

    // CSV rows
    const rows = questionsData.questions.map((q: any) => [
      q.question_id || '',
      q.question || '',
      q.golden_answer || '',
      q.question_type || '',
      q.audio_cue || '',
      q.visual_cue || '',
      q.start_timestamp || '',
      q.end_timestamp || ''
    ])

    // Combine into CSV string
    const csvContent = [
      headers.join(','),
      ...rows.map((row: any[]) => row.map((cell: any) => `"${String(cell).replace(/"/g, '""')}"`).join(','))
    ].join('\n')

    // Download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `${selectedVideoId}_questions_${new Date().toISOString().split('T')[0]}.csv`
    link.click()

    toast.success('Questions exported to CSV!')
  }

  const renderQuestionsTab = () => (
    <div className="space-y-6">
      {!questionsData && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-3">
            <div className="text-2xl">⏳</div>
            <h3 className="text-lg font-semibold text-yellow-800">Pass 3: Question Generation Pending</h3>
          </div>
          <p className="text-sm text-yellow-700 mb-3">
            Questions will appear here after Pass 3 completes. The pipeline uses validated moments from Pass 2A/2B to generate adversarial multimodal questions with GPT-4o Vision.
          </p>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="bg-white p-3 rounded">
              <div className="font-medium text-gray-700 mb-1">✓ Completed Phases:</div>
              <ul className="text-gray-600 space-y-1">
                {audioData && <li>• Phase 1: Audio Analysis</li>}
                {statusData?.phases_complete?.includes('visual_sampling') && <li>• Phase 2: Visual Sampling</li>}
                {statusData?.phases_complete?.includes('pass2a_moments') && <li>• Pass 2A: Sonnet Moments</li>}
                {statusData?.phases_complete?.includes('pass2b_moments') && <li>• Pass 2B: Opus Moments</li>}
                {statusData?.phases_complete?.includes('validation') && <li>• Validation Complete</li>}
              </ul>
            </div>
            <div className="bg-white p-3 rounded">
              <div className="font-medium text-gray-700 mb-1">⏭️ Next Steps:</div>
              <ul className="text-gray-600 space-y-1">
                <li>• Run Smart Pipeline</li>
                <li>• Wait for Pass 3 (~30s-2min)</li>
                <li>• Export questions to CSV</li>
              </ul>
            </div>
          </div>
        </div>
      )}
      {questionsData && (
        <>
          <div className="bg-white p-6 rounded-lg shadow mb-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Pass 3: Question Generation Summary</h3>
              <button
                onClick={exportToCSV}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
                Export CSV (Google Sheets)
              </button>
            </div>
            <div className="grid grid-cols-3 gap-4 text-center mb-4">
              <div>
                <div className="text-2xl font-bold text-blue-600">{questionsData.total_questions || 0}</div>
                <div className="text-xs text-gray-600">Final Questions</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-orange-600">{questionsData.cost_summary?.questions_generated || 0}</div>
                <div className="text-xs text-gray-600">Generated (Before Selection)</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-600">${questionsData.cost_summary?.total_cost?.toFixed(4) || '0.00'}</div>
                <div className="text-xs text-gray-600">Total Cost</div>
              </div>
            </div>
            <div className="text-center text-xs text-gray-500">
              Model: {questionsData.metadata?.model_used || 'GPT-4o'} |
              Tokens: {questionsData.cost_summary?.input_tokens?.toLocaleString() || 0} in + {questionsData.cost_summary?.output_tokens?.toLocaleString() || 0} out
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">All Questions ({questionsData.questions?.length || 0})</h3>
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {questionsData.questions?.map((q: any, idx: number) => (
                <div key={idx} className="border border-gray-200 rounded-lg p-4 hover:border-blue-300 transition-colors">
                  <div className="flex justify-between items-start mb-2">
                    <div className="font-medium text-gray-800">Q{idx + 1}: {q.question}</div>
                    <div className="flex gap-2 items-center">
                      <span className="px-2 py-1 text-xs rounded bg-blue-100 text-blue-800">
                        {q.question_type || 'Unknown'}
                      </span>
                      <span className="px-2 py-1 text-xs rounded bg-green-100 text-green-800">
                        {q.confidence ? `${(q.confidence * 100).toFixed(0)}%` : 'N/A'}
                      </span>
                    </div>
                  </div>
                  <div className="text-sm text-gray-700 mb-2 bg-gray-50 p-2 rounded">
                    <span className="font-medium text-gray-900">Answer:</span> {q.golden_answer}
                  </div>
                  {q.audio_cue && (
                    <div className="text-xs text-blue-700 mb-1 bg-blue-50 p-2 rounded">
                      <span className="font-medium">🎵 Audio Cue:</span> "{q.audio_cue}"
                    </div>
                  )}
                  {q.visual_cue && (
                    <div className="text-xs text-orange-700 mb-1 bg-orange-50 p-2 rounded">
                      <span className="font-medium">👁️ Visual Cue:</span> {q.visual_cue}
                    </div>
                  )}
                  <div className="flex justify-between items-center text-xs text-gray-500 mt-2 pt-2 border-t border-gray-200">
                    <div>
                      <span className="font-medium">ID:</span> {q.question_id || 'N/A'}
                    </div>
                    <div>
                      <span className="font-medium">Time:</span> {q.start_timestamp || 'N/A'} - {q.end_timestamp || 'N/A'}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )

  const renderGeminiTab = () => (
    <div className="space-y-6">
      {geminiData?.tested ? (
        <>
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">Gemini 2.0 Flash Testing Results</h3>
            <div className="bg-blue-50 p-4 rounded mb-4">
              <p className="text-sm text-gray-700">
                These questions were tested against Gemini 2.0 Flash to measure the adversarial effectiveness
                of the generated questions.
              </p>
            </div>

            {geminiData.results && (
              <div className="space-y-4">
                <h4 className="font-medium">Test Results:</h4>
                <pre className="bg-gray-50 p-4 rounded text-sm overflow-x-auto">
                  {JSON.stringify(geminiData.results, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </>
      ) : (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Gemini Testing Not Run</h3>
          <div className="bg-gray-50 p-4 rounded">
            <p className="text-sm text-gray-600 mb-4">
              Gemini 2.0 Flash testing has not been run for this video yet.
            </p>
            <p className="text-sm text-gray-600">
              To enable Gemini testing, check the "Enable Gemini" checkbox when running the pipeline.
            </p>
          </div>
        </div>
      )}
    </div>
  )

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Adversarial Smart Pipeline</h1>
          <p className="text-gray-600">Pass 1-2B Architecture: Scene/FPS sampling → CLIP analysis → Sonnet 4.5 + Opus 4 moment selection → GPT-4o Q&A generation</p>
        </div>
      </div>

      {/* Video Selection & Pipeline Controls */}
      <div className="bg-white p-6 rounded-lg shadow space-y-4">
        <h2 className="text-xl font-semibold">Run Pipeline</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Video Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Video
            </label>
            <select
              value={selectedVideoId}
              onChange={(e) => setSelectedVideoId(e.target.value)}
              className="w-full border border-gray-300 rounded-lg p-2"
            >
              <option value="">-- Select a video --</option>
              {videosData?.videos?.map((video: any) => (
                <option key={video.video_id} value={video.video_id}>
                  {video.title} ({video.filename})
                </option>
              ))}
            </select>
          </div>

          {/* Model Options */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              AI Models (Required: GPT-4, Optional: Claude & Gemini)
            </label>
            <div className="space-y-2">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={enableGpt4}
                  onChange={(e) => setEnableGpt4(e.target.checked)}
                  className="mr-2"
                />
                <span className="text-sm">Enable GPT-4 (semantic analysis + premium frames + questions)</span>
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={enableClaude}
                  onChange={(e) => setEnableClaude(e.target.checked)}
                  className="mr-2"
                />
                <span className="text-sm">Enable Claude Sonnet 4.5 (cross-validation for top 3 premium frames)</span>
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={enableGemini}
                  onChange={(e) => setEnableGemini(e.target.checked)}
                  className="mr-2"
                />
                <span className="text-sm">Enable Gemini 2.0 Flash (test generated questions)</span>
              </label>
            </div>
          </div>
        </div>

        <button
          onClick={handleRunPipeline}
          disabled={!selectedVideoId || runPipelineMutation.isPending}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {runPipelineMutation.isPending ? 'Starting Pipeline...' : '🎯 Run Adversarial Pipeline'}
        </button>
      </div>

      {/* Progress Bar */}
      {selectedVideoId && renderProgressBar()}

      {/* Results Tabs */}
      {selectedVideoId && statusData?.status !== 'pending' && (
        <div className="bg-white rounded-lg shadow">
          {/* Tab Navigation */}
          <div className="border-b border-gray-200 px-6">
            <div className="flex space-x-8">
              {[
                { id: 'overview', label: 'Overview', icon: '📊' },
                { id: 'audio', label: 'Audio', icon: '🎵' },
                { id: 'questions', label: 'Questions', icon: '❓' },
                { id: 'gemini', label: 'Gemini', icon: '🧪' },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`py-4 px-2 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {tab.icon} {tab.label}
                </button>
              ))}
            </div>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {activeTab === 'overview' && renderOverview()}
            {activeTab === 'audio' && renderAudioTab()}
            {activeTab === 'questions' && renderQuestionsTab()}
            {activeTab === 'gemini' && renderGeminiTab()}
          </div>
        </div>
      )}
    </div>
  )
}
