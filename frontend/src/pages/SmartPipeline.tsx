import { useState, useEffect } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { toast } from 'sonner'
import {
  listUploadedVideos,
  runSmartPipeline,
  getSmartPipelineStatus,
  getAudioAnalysis,
  getOpportunities,
  getFrameExtraction,
  getSmartPipelineQuestions,
  getGeminiResults,
  getFullTranscript,
} from '../api/client'

/**
 * Adversarial Smart Pipeline Page - Opportunity-Based Video Q&A Generation
 *
 * Features:
 * - Run the adversarial evidence pipeline
 * - View audio analysis with word timestamps
 * - See adversarial opportunities (temporal markers, ambiguous references, etc.)
 * - View frame extraction plan (premium + template + bulk)
 * - See generated questions (20 template + 7 AI + 3 cross-validated)
 * - View Gemini testing results
 */
export default function SmartPipeline() {
  const [selectedVideoId, setSelectedVideoId] = useState<string>('')
  const [enableGpt4, setEnableGpt4] = useState(false)
  const [enableClaude, setEnableClaude] = useState(false)
  const [enableGemini, setEnableGemini] = useState(false)
  const [activeTab, setActiveTab] = useState<'overview' | 'audio' | 'opportunities' | 'frames' | 'questions' | 'gemini'>('overview')

  // Fetch list of uploaded videos
  const { data: videosData, isLoading: videosLoading } = useQuery({
    queryKey: ['uploaded-videos'],
    queryFn: () => listUploadedVideos(),
    refetchInterval: 5000,
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
    refetchInterval: selectedVideoId ? 3000 : false,
  })

  // Fetch audio analysis
  const { data: audioData, isLoading: audioLoading, error: audioError } = useQuery({
    queryKey: ['audio-analysis', selectedVideoId],
    queryFn: () => getAudioAnalysis(selectedVideoId),
    enabled: !!selectedVideoId && statusData?.phases_complete?.includes('audio_analysis'),
    retry: false, // Don't retry 404s
  })

  // Fetch adversarial opportunities
  const { data: opportunitiesData, isLoading: opportunitiesLoading, error: opportunitiesError } = useQuery({
    queryKey: ['opportunities', selectedVideoId],
    queryFn: () => getOpportunities(selectedVideoId),
    enabled: !!selectedVideoId && statusData?.phases_complete?.includes('opportunity_mining'),
    retry: false,
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

    const progress = statusData.progress * 100

    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="flex justify-between items-center mb-2">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold">Pipeline Progress</h3>
            {statusFetching && (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
            )}
          </div>
          <span className={`px-3 py-1 text-sm rounded ${
            statusData.status === 'completed' ? 'bg-green-500 text-white' :
            statusData.status === 'processing' ? 'bg-blue-500 text-white' :
            statusData.status === 'pending' ? 'bg-gray-500 text-white' :
            'bg-red-500 text-white'
          }`}>
            {statusData.status}
          </span>
        </div>

        <div className="w-full bg-gray-200 rounded-full h-4 mb-2">
          <div
            className="bg-blue-600 h-4 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>

        <div className="flex justify-between text-sm text-gray-600">
          <span>{statusData.current_phase}</span>
          <span>{progress.toFixed(0)}%</span>
        </div>

        {statusData.phases_complete && (
          <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
            {['audio_analysis', 'opportunity_mining', 'frame_extraction', 'evidence_extraction', 'question_generation'].map((phase) => (
              <div
                key={phase}
                className={`p-2 rounded text-center ${
                  statusData.phases_complete.includes(phase)
                    ? 'bg-green-100 text-green-800'
                    : 'bg-gray-100 text-gray-500'
                }`}
              >
                {statusData.phases_complete.includes(phase) ? '✓' : '○'} {phase.replace(/_/g, ' ')}
              </div>
            ))}
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
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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

        {renderDataCard(
          "Adversarial Opportunities",
          opportunitiesLoading,
          opportunitiesError,
          opportunitiesData,
          <div className="space-y-2 text-sm">
            <div><span className="font-medium">Temporal Markers:</span> {opportunitiesData?.temporal_markers_count}</div>
            <div><span className="font-medium">Ambiguous Refs:</span> {opportunitiesData?.ambiguous_references_count}</div>
            <div><span className="font-medium">Premium Keyframes:</span> {opportunitiesData?.premium_keyframes_count}</div>
            <div><span className="font-medium">Detection Cost:</span> ${opportunitiesData?.detection_cost.toFixed(4)}</div>
          </div>
        )}

        {renderDataCard(
          "Frame Extraction",
          framesLoading,
          framesError,
          framesData,
          <div className="space-y-2 text-sm">
            <div><span className="font-medium">Total Frames:</span> {framesData?.total_frames}</div>
            <div><span className="font-medium">Premium (GPT-4V/Claude):</span> {framesData?.premium_frames}</div>
            <div><span className="font-medium">Template:</span> {framesData?.template_frames}</div>
            <div><span className="font-medium">Bulk (every 5s):</span> {framesData?.bulk_frames}</div>
          </div>
        )}
      </div>

      {questionsData && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Adversarial Questions (30 Total)</h3>
          <div className="grid grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-3xl font-bold text-blue-600">{questionsData.total_questions}</div>
              <div className="text-sm text-gray-600">Total</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-purple-600">{questionsData.template_count}</div>
              <div className="text-sm text-gray-600">Template</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-orange-600">{questionsData.ai_count}</div>
              <div className="text-sm text-gray-600">AI-Generated</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-green-600">{questionsData.cross_validated_count}</div>
              <div className="text-sm text-gray-600">Cross-Validated</div>
            </div>
          </div>
          {questionsData.generation_cost && (
            <div className="mt-4 text-center text-sm text-gray-600">
              Generation Cost: ${questionsData.generation_cost.toFixed(4)}
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

  const renderOpportunitiesTab = () => (
    <div className="space-y-6">
      {opportunitiesData && (
        <>
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">Adversarial Opportunities Detected</h3>
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div><span className="font-medium">Temporal Markers:</span> {opportunitiesData.temporal_markers_count}</div>
              <div><span className="font-medium">Ambiguous References:</span> {opportunitiesData.ambiguous_references_count}</div>
              <div><span className="font-medium">Counting Opportunities:</span> {opportunitiesData.counting_opportunities_count}</div>
              <div><span className="font-medium">Sequential Events:</span> {opportunitiesData.sequential_events_count}</div>
              <div><span className="font-medium">Context-Rich Frames:</span> {opportunitiesData.context_rich_frames_count}</div>
              <div><span className="font-medium">Premium Keyframes:</span> {opportunitiesData.premium_keyframes_count}</div>
            </div>

            <div className="mb-4">
              <h4 className="font-medium mb-2">Premium Keyframe Timestamps:</h4>
              <div className="flex gap-2 flex-wrap">
                {opportunitiesData.premium_keyframes?.map((timestamp: number, idx: number) => (
                  <span key={idx} className="px-3 py-1 bg-orange-100 text-orange-800 rounded-full text-sm font-mono">
                    {timestamp.toFixed(1)}s
                  </span>
                ))}
              </div>
            </div>

            <div className="bg-gray-50 p-4 rounded">
              <h4 className="font-medium mb-2">Detection Details:</h4>
              <ul className="text-sm space-y-2">
                <li>🎯 GPT-4 analyzed transcript to find adversarial opportunities</li>
                <li>⏱️ Identified moments where temporal understanding is critical</li>
                <li>❓ Found ambiguous references that require visual grounding</li>
                <li>💰 Detection cost: ${opportunitiesData.detection_cost.toFixed(4)}</li>
              </ul>
            </div>
          </div>
        </>
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
              <div className="text-sm text-gray-600">Template (opportunities)</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded">
              <div className="text-3xl font-bold text-green-600">{framesData.bulk_frames}</div>
              <div className="text-sm text-gray-600">Bulk (every 5s)</div>
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded">
            <h4 className="font-medium mb-2">Adversarial Frame Extraction:</h4>
            <ul className="text-sm space-y-2">
              <li>🎯 <strong>{framesData.premium_frames} premium frames</strong> for GPT-4 Vision & Claude Vision (most challenging moments)</li>
              <li>📋 <strong>{framesData.template_frames} template frames</strong> extracted at adversarial opportunities</li>
              <li>🔄 <strong>{framesData.bulk_frames} bulk frames</strong> extracted every 5 seconds for comprehensive coverage</li>
              <li>💰 Optimized strategy: expensive models only on critical frames</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  )

  const renderQuestionsTab = () => (
    <div className="space-y-6">
      {questionsData && (
        <>
          <div className="bg-white p-6 rounded-lg shadow mb-6">
            <h3 className="text-lg font-semibold mb-4">Adversarial Question Generation (30 Total)</h3>
            <div className="grid grid-cols-4 gap-4 text-center mb-4">
              <div>
                <div className="text-2xl font-bold text-blue-600">{questionsData.total_questions}</div>
                <div className="text-xs text-gray-600">Total Questions</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-purple-600">{questionsData.template_count}</div>
                <div className="text-xs text-gray-600">Template (20)</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-orange-600">{questionsData.ai_count}</div>
                <div className="text-xs text-gray-600">AI-Generated (7)</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-600">{questionsData.cross_validated_count}</div>
                <div className="text-xs text-gray-600">Cross-Validated (3)</div>
              </div>
            </div>
            {questionsData.generation_cost && (
              <div className="text-center text-sm text-gray-600">
                Generation Cost: <span className="font-semibold">${questionsData.generation_cost.toFixed(4)}</span>
              </div>
            )}
          </div>

          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">All Questions</h3>
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {questionsData.questions?.map((q: any, idx: number) => (
                <div key={idx} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex justify-between items-start mb-2">
                    <div className="font-medium">Q{idx + 1}: {q.question}</div>
                    <span className={`px-2 py-1 text-xs rounded ${
                      q.generation_tier === 'template' ? 'bg-purple-100 text-purple-800' :
                      q.generation_tier === 'ai' ? 'bg-orange-100 text-orange-800' :
                      q.generation_tier === 'cross_validated' ? 'bg-green-100 text-green-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {q.generation_tier === 'template' ? 'Template' :
                       q.generation_tier === 'ai' ? 'AI-Generated' :
                       q.generation_tier === 'cross_validated' ? 'Cross-Validated' :
                       q.generation_tier}
                    </span>
                  </div>
                  <div className="text-sm text-gray-600 mb-2">
                    <span className="font-medium">Answer:</span> {q.golden_answer}
                  </div>
                  {q.audio_cue && (
                    <div className="text-xs text-blue-600 mb-1">
                      <span className="font-medium">🎵 Audio:</span> "{q.audio_cue}"
                    </div>
                  )}
                  {q.visual_cue && (
                    <div className="text-xs text-orange-600 mb-1">
                      <span className="font-medium">👁️ Visual:</span> {q.visual_cue}
                    </div>
                  )}
                  <div className="text-xs text-gray-500 mt-2">
                    Type: {q.task_types?.join(', ') || 'N/A'} | {q.opportunity_type || 'N/A'}
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
          <p className="text-gray-600">Opportunity-based video Q&A generation designed to challenge Gemini 2.0 Flash</p>
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
                <span className="text-sm">Enable GPT-4 (opportunity mining + premium frames + questions)</span>
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
                { id: 'opportunities', label: 'Opportunities', icon: '🎯' },
                { id: 'frames', label: 'Frames', icon: '🎞️' },
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
            {activeTab === 'opportunities' && renderOpportunitiesTab()}
            {activeTab === 'frames' && renderFramesTab()}
            {activeTab === 'questions' && renderQuestionsTab()}
            {activeTab === 'gemini' && renderGeminiTab()}
          </div>
        </div>
      )}
    </div>
  )
}
