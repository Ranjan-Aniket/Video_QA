import { useEffect, useState, useMemo } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { useEvidenceStore } from '../store/useEvidenceStore'
import * as evidenceApi from '../api/evidenceApi'
import ModelPredictionDisplay from '../components/evidence/ModelPredictionDisplay'
import DisagreementIndicator from '../components/evidence/DisagreementIndicator'
import ReviewForm from '../components/evidence/ReviewForm'
import ReviewProgressBar from '../components/evidence/ReviewProgressBar'
import EvidenceTimeline from '../components/evidence/EvidenceTimeline'
import type { ReviewSubmission, DisagreementInfo, ModelType } from '../types/evidence'

export default function EvidenceReview() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const videoId = searchParams.get('video_id') || 'video_20251117_050110_Copy of w-A-4ckmFJo'

  const {
    evidenceQueue,
    currentEvidence,
    currentIndex,
    reviewProgress,
    reviewerId,
    isLoading,
    error,
    setEvidenceQueue,
    setCurrentIndex,
    setReviewProgress,
    setIsLoading,
    setError,
    nextEvidence,
    previousEvidence,
    updateEvidenceInQueue,
  } = useEvidenceStore()

  const [isSubmitting, setIsSubmitting] = useState(false)

  // Load review queue on mount
  useEffect(() => {
    loadReviewQueue()
    loadProgress()
  }, [videoId])

  const loadReviewQueue = async () => {
    try {
      setIsLoading(true)
      setError(null)
      const queue = await evidenceApi.getReviewQueue({ video_id: videoId })
      setEvidenceQueue(queue)
      if (queue.length > 0) {
        setCurrentIndex(0)
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load review queue')
      toast.error('Failed to load review queue')
    } finally {
      setIsLoading(false)
    }
  }

  const loadProgress = async () => {
    try {
      const progress = await evidenceApi.getProgress(videoId)
      setReviewProgress(progress)
    } catch (err: any) {
      console.error('Failed to load progress:', err)
    }
  }

  const handleSubmitReview = async (review: ReviewSubmission) => {
    if (!currentEvidence) return

    try {
      setIsSubmitting(true)
      const updated = await evidenceApi.submitReview(
        currentEvidence.id,
        review,
        reviewerId
      )

      updateEvidenceInQueue(currentEvidence.id, updated)
      await loadProgress()

      toast.success(`Review ${review.decision} submitted successfully`)

      // Move to next evidence
      if (currentIndex < evidenceQueue.length - 1) {
        nextEvidence()
      } else {
        toast.info('All evidence items reviewed!')
      }
    } catch (err: any) {
      toast.error(err.message || 'Failed to submit review')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleSkip = async () => {
    if (!currentEvidence) return

    try {
      setIsSubmitting(true)
      const updated = await evidenceApi.skipEvidence(currentEvidence.id, reviewerId)

      updateEvidenceInQueue(currentEvidence.id, updated)
      await loadProgress()

      toast.info('Evidence item skipped')

      // Move to next evidence
      if (currentIndex < evidenceQueue.length - 1) {
        nextEvidence()
      }
    } catch (err: any) {
      toast.error(err.message || 'Failed to skip evidence')
    } finally {
      setIsSubmitting(false)
    }
  }

  // Calculate disagreement info
  const disagreementInfo: DisagreementInfo = useMemo(() => {
    if (!currentEvidence) {
      return {
        hasDisagreement: false,
        level: 'none',
        disagreementCount: 0,
        agreeingModels: [],
        disagreeingModels: [],
      }
    }

    const predictions = [
      { model: 'gpt4' as ModelType, value: currentEvidence.gpt4_prediction },
      { model: 'claude' as ModelType, value: currentEvidence.claude_prediction },
      { model: 'open' as ModelType, value: currentEvidence.open_model_prediction },
    ]

    // Simple disagreement check (can be enhanced)
    const uniquePredictions = new Set(
      predictions.map((p) => JSON.stringify(p.value))
    ).size

    const disagreementCount = uniquePredictions - 1
    const hasDisagreement = disagreementCount > 0

    let level: DisagreementInfo['level'] = 'none'
    if (disagreementCount === 0) level = 'none'
    else if (disagreementCount === 1) level = 'low'
    else if (disagreementCount === 2) level = 'high'

    return {
      hasDisagreement,
      level,
      disagreementCount,
      agreeingModels: hasDisagreement ? [] : predictions.map((p) => p.model),
      disagreeingModels: hasDisagreement ? predictions.map((p) => p.model) : [],
    }
  }, [currentEvidence])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mb-4"></div>
          <p className="text-xl font-semibold">Loading evidence queue...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center bg-red-50 border-2 border-red-500 rounded-lg p-8 max-w-md">
          <p className="text-2xl font-bold text-red-700 mb-4">Error</p>
          <p className="text-red-600 mb-4">{error}</p>
          <button
            onClick={loadReviewQueue}
            className="bg-red-600 text-white px-6 py-2 rounded-lg hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (evidenceQueue.length === 0) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center bg-gray-50 border-2 border-gray-300 rounded-lg p-8 max-w-md">
          <p className="text-2xl font-bold text-gray-700 mb-4">No Evidence to Review</p>
          <p className="text-gray-600 mb-4">
            There are no evidence items in the queue for this video.
          </p>
          <button
            onClick={() => navigate('/')}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
          >
            Go to Dashboard
          </button>
        </div>
      </div>
    )
  }

  if (!currentEvidence) {
    return null
  }

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">Evidence Review</h1>
          <p className="text-gray-600">
            Video: {videoId} • Reviewing {currentIndex + 1} of {evidenceQueue.length}
          </p>
        </div>

        {/* Progress Bar */}
        {reviewProgress && (
          <div className="mb-6">
            <ReviewProgressBar progress={reviewProgress} />
          </div>
        )}

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Left Column - Timeline */}
          <div className="lg:col-span-1">
            <EvidenceTimeline
              videoId={videoId}
              evidenceItems={evidenceQueue}
              currentItem={currentEvidence}
              onSelect={(item) => {
                const index = evidenceQueue.findIndex((e) => e.id === item.id)
                if (index !== -1) setCurrentIndex(index)
              }}
            />
          </div>

          {/* Right Column - Review Area */}
          <div className="lg:col-span-2 space-y-6">
            {/* Evidence Info */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h2 className="text-2xl font-bold">
                    Evidence #{currentEvidence.id}
                  </h2>
                  <p className="text-gray-600">
                    {currentEvidence.evidence_type.replace('_', ' ')} • {' '}
                    {Math.floor(currentEvidence.timestamp_start / 60)}:
                    {Math.floor(currentEvidence.timestamp_start % 60)
                      .toString()
                      .padStart(2, '0')}{' '}
                    -{' '}
                    {Math.floor(currentEvidence.timestamp_end / 60)}:
                    {Math.floor(currentEvidence.timestamp_end % 60)
                      .toString()
                      .padStart(2, '0')}
                  </p>
                </div>
                <span
                  className={`px-3 py-1 rounded text-sm font-semibold ${
                    currentEvidence.priority_level === 'high'
                      ? 'bg-red-100 text-red-800'
                      : currentEvidence.priority_level === 'medium'
                      ? 'bg-yellow-100 text-yellow-800'
                      : 'bg-green-100 text-green-800'
                  }`}
                >
                  {currentEvidence.priority_level.toUpperCase()} PRIORITY
                </span>
              </div>

              {currentEvidence.flag_reason && (
                <div className="mb-4 bg-yellow-50 border-l-4 border-yellow-500 p-4">
                  <p className="font-semibold text-yellow-800">Flag Reason:</p>
                  <p className="text-yellow-700">{currentEvidence.flag_reason}</p>
                </div>
              )}
            </div>

            {/* Disagreement Indicator */}
            <DisagreementIndicator disagreementInfo={disagreementInfo} />

            {/* Model Predictions */}
            <div className="space-y-4">
              <h3 className="text-xl font-bold">AI Model Predictions</h3>
              <div className="grid grid-cols-1 gap-4">
                <ModelPredictionDisplay
                  model="gpt4"
                  prediction={currentEvidence.gpt4_prediction}
                  confidence={currentEvidence.confidence_score}
                />
                <ModelPredictionDisplay
                  model="claude"
                  prediction={currentEvidence.claude_prediction}
                  confidence={currentEvidence.confidence_score}
                />
                <ModelPredictionDisplay
                  model="open"
                  prediction={currentEvidence.open_model_prediction}
                  confidence={currentEvidence.confidence_score}
                />
              </div>
            </div>

            {/* Review Form */}
            <ReviewForm
              evidenceId={currentEvidence.id}
              consensusAnswer={currentEvidence.consensus_answer}
              onSubmit={handleSubmitReview}
              onSkip={handleSkip}
              isSubmitting={isSubmitting}
            />

            {/* Navigation Buttons */}
            <div className="flex gap-4">
              <button
                onClick={previousEvidence}
                disabled={currentIndex === 0}
                className="flex-1 bg-gray-200 text-gray-700 py-3 px-6 rounded-lg font-semibold hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                ← Previous
              </button>
              <button
                onClick={nextEvidence}
                disabled={currentIndex === evidenceQueue.length - 1}
                className="flex-1 bg-gray-200 text-gray-700 py-3 px-6 rounded-lg font-semibold hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Next →
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
