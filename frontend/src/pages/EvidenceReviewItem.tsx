import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import * as evidenceApi from '../api/evidenceApi'
import { useEvidenceStore } from '../store/useEvidenceStore'
import ModelPredictionDisplay from '../components/evidence/ModelPredictionDisplay'
import ReviewForm from '../components/evidence/ReviewForm'
import type { EvidenceItem, ReviewSubmission } from '../types/evidence'

export default function EvidenceReviewItem() {
  const { evidenceId } = useParams<{ evidenceId: string }>()
  const navigate = useNavigate()
  const { reviewerId } = useEvidenceStore()

  const [evidence, setEvidence] = useState<EvidenceItem | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (evidenceId) {
      loadEvidence(parseInt(evidenceId))
    }
  }, [evidenceId])

  const loadEvidence = async (id: number) => {
    try {
      setIsLoading(true)
      setError(null)
      const data = await evidenceApi.getEvidence(id)
      setEvidence(data)
    } catch (err: any) {
      setError(err.message || 'Failed to load evidence')
      toast.error('Failed to load evidence item')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmitReview = async (review: ReviewSubmission) => {
    if (!evidence) return

    try {
      setIsSubmitting(true)
      await evidenceApi.submitReview(evidence.id, review, reviewerId)

      toast.success(`Review ${review.decision} submitted successfully`)

      // Reload to see updated status
      await loadEvidence(evidence.id)
    } catch (err: any) {
      toast.error(err.message || 'Failed to submit review')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleSkip = async () => {
    if (!evidence) return

    try {
      setIsSubmitting(true)
      await evidenceApi.skipEvidence(evidence.id, reviewerId)

      toast.info('Evidence item skipped')
      await loadEvidence(evidence.id)
    } catch (err: any) {
      toast.error(err.message || 'Failed to skip evidence')
    } finally {
      setIsSubmitting(false)
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mb-4"></div>
          <p className="text-xl font-semibold">Loading evidence...</p>
        </div>
      </div>
    )
  }

  if (error || !evidence) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center bg-red-50 border-2 border-red-500 rounded-lg p-8 max-w-md">
          <p className="text-2xl font-bold text-red-700 mb-4">Error</p>
          <p className="text-red-600 mb-4">{error || 'Evidence not found'}</p>
          <button
            onClick={() => navigate('/evidence/review')}
            className="bg-red-600 text-white px-6 py-2 rounded-lg hover:bg-red-700"
          >
            Back to Review Queue
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <button
            onClick={() => navigate('/evidence/review')}
            className="text-blue-600 hover:text-blue-700 mb-4 flex items-center gap-2"
          >
            ← Back to Review Queue
          </button>
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Evidence Item #{evidence.id}
          </h1>
          <p className="text-gray-600">Video: {evidence.video_id}</p>
        </div>

        {/* Evidence Details */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <p className="text-sm font-semibold text-gray-600">Evidence Type</p>
              <p className="text-lg capitalize">
                {evidence.evidence_type.replace('_', ' ')}
              </p>
            </div>
            <div>
              <p className="text-sm font-semibold text-gray-600">Priority</p>
              <p className="text-lg capitalize">{evidence.priority_level}</p>
            </div>
            <div>
              <p className="text-sm font-semibold text-gray-600">Timestamp</p>
              <p className="text-lg">
                {Math.floor(evidence.timestamp_start / 60)}:
                {Math.floor(evidence.timestamp_start % 60)
                  .toString()
                  .padStart(2, '0')}{' '}
                -{' '}
                {Math.floor(evidence.timestamp_end / 60)}:
                {Math.floor(evidence.timestamp_end % 60)
                  .toString()
                  .padStart(2, '0')}
              </p>
            </div>
            <div>
              <p className="text-sm font-semibold text-gray-600">Review Status</p>
              <p
                className={`text-lg capitalize font-semibold ${
                  evidence.human_review_status === 'pending'
                    ? 'text-gray-600'
                    : evidence.human_review_status === 'approved'
                    ? 'text-green-600'
                    : evidence.human_review_status === 'corrected'
                    ? 'text-blue-600'
                    : evidence.human_review_status === 'rejected'
                    ? 'text-red-600'
                    : 'text-yellow-600'
                }`}
              >
                {evidence.human_review_status}
              </p>
            </div>
          </div>

          {evidence.flag_reason && (
            <div className="mt-4 bg-yellow-50 border-l-4 border-yellow-500 p-4">
              <p className="font-semibold text-yellow-800">Flag Reason:</p>
              <p className="text-yellow-700">{evidence.flag_reason}</p>
            </div>
          )}

          <div className="mt-4">
            <p className="text-sm font-semibold text-gray-600 mb-2">
              Confidence Score
            </p>
            <div className="flex items-center gap-3">
              <div className="flex-1 bg-gray-200 rounded-full h-4">
                <div
                  className={`h-4 rounded-full ${
                    evidence.confidence_score >= 0.7
                      ? 'bg-green-500'
                      : evidence.confidence_score >= 0.4
                      ? 'bg-yellow-500'
                      : 'bg-red-500'
                  }`}
                  style={{ width: `${evidence.confidence_score * 100}%` }}
                />
              </div>
              <span className="text-lg font-semibold">
                {(evidence.confidence_score * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>

        {/* Model Predictions */}
        <div className="mb-6">
          <h2 className="text-2xl font-bold mb-4">AI Model Predictions</h2>
          <div className="grid grid-cols-1 gap-4">
            <ModelPredictionDisplay
              model="gpt4"
              prediction={evidence.gpt4_prediction}
              confidence={evidence.confidence_score}
            />
            <ModelPredictionDisplay
              model="claude"
              prediction={evidence.claude_prediction}
              confidence={evidence.confidence_score}
            />
            <ModelPredictionDisplay
              model="open"
              prediction={evidence.open_model_prediction}
              confidence={evidence.confidence_score}
            />
          </div>
        </div>

        {/* Consensus Answer */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h3 className="text-xl font-bold mb-3">Consensus Answer</h3>
          <pre className="bg-gray-50 p-4 rounded overflow-auto max-h-60 text-sm">
            {typeof evidence.consensus_answer === 'string'
              ? evidence.consensus_answer
              : JSON.stringify(evidence.consensus_answer, null, 2)}
          </pre>
        </div>

        {/* Review History */}
        {evidence.human_review_status !== 'pending' && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-xl font-bold mb-3">Review Details</h3>
            <div className="space-y-2">
              <div>
                <span className="font-semibold text-gray-700">Reviewed by:</span>{' '}
                {evidence.human_reviewer_id || 'N/A'}
              </div>
              <div>
                <span className="font-semibold text-gray-700">Review time:</span>{' '}
                {evidence.review_timestamp
                  ? new Date(evidence.review_timestamp).toLocaleString()
                  : 'N/A'}
              </div>
              <div>
                <span className="font-semibold text-gray-700">Duration:</span>{' '}
                {evidence.review_duration_seconds
                  ? `${Math.floor(evidence.review_duration_seconds / 60)}m ${
                      evidence.review_duration_seconds % 60
                    }s`
                  : 'N/A'}
              </div>
              {evidence.human_confidence && (
                <div>
                  <span className="font-semibold text-gray-700">Confidence:</span>{' '}
                  <span className="capitalize">{evidence.human_confidence}</span>
                </div>
              )}
              {evidence.human_notes && (
                <div>
                  <span className="font-semibold text-gray-700">Notes:</span>
                  <p className="mt-1 bg-gray-50 p-3 rounded">{evidence.human_notes}</p>
                </div>
              )}
              {evidence.human_answer && (
                <div>
                  <span className="font-semibold text-gray-700">Human Answer:</span>
                  <pre className="mt-1 bg-gray-50 p-3 rounded overflow-auto max-h-40 text-sm">
                    {typeof evidence.human_answer === 'string'
                      ? evidence.human_answer
                      : JSON.stringify(evidence.human_answer, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Review Form (only if pending or allowing re-review) */}
        {evidence.human_review_status === 'pending' && (
          <ReviewForm
            evidenceId={evidence.id}
            consensusAnswer={evidence.consensus_answer}
            onSubmit={handleSubmitReview}
            onSkip={handleSkip}
            isSubmitting={isSubmitting}
          />
        )}
      </div>
    </div>
  )
}
