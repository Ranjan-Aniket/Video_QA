import { useState, useEffect } from 'react'
import type { ReviewSubmission } from '../../types/evidence'

interface ReviewFormProps {
  evidenceId: number
  consensusAnswer?: any
  onSubmit: (review: ReviewSubmission) => void
  onSkip: () => void
  isSubmitting?: boolean
}

export default function ReviewForm({
  evidenceId,
  consensusAnswer,
  onSubmit,
  onSkip,
  isSubmitting = false,
}: ReviewFormProps) {
  const [decision, setDecision] = useState<'approved' | 'corrected' | 'rejected'>('approved')
  const [correctedAnswer, setCorrectedAnswer] = useState('')
  const [confidence, setConfidence] = useState<'high' | 'medium' | 'low'>('high')
  const [notes, setNotes] = useState('')
  const [startTime] = useState(Date.now())

  // Reset form when evidence changes
  useEffect(() => {
    setDecision('approved')
    setCorrectedAnswer('')
    setConfidence('high')
    setNotes('')
  }, [evidenceId])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Don't trigger if typing in textarea
      if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement) {
        return
      }

      switch (e.key.toLowerCase()) {
        case 'a':
          setDecision('approved')
          break
        case 'c':
          setDecision('corrected')
          break
        case 'r':
          setDecision('rejected')
          break
        case 's':
          onSkip()
          break
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [onSkip])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    const reviewDurationSeconds = Math.floor((Date.now() - startTime) / 1000)

    const review: ReviewSubmission = {
      decision,
      confidence,
      notes: notes.trim() || undefined,
      review_duration_seconds: reviewDurationSeconds,
    }

    if (decision === 'corrected' && correctedAnswer.trim()) {
      try {
        review.corrected_answer = JSON.parse(correctedAnswer)
      } catch {
        review.corrected_answer = correctedAnswer
      }
    }

    onSubmit(review)
  }

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow-lg p-6 space-y-4">
      <h2 className="text-2xl font-bold mb-4">Review Decision</h2>

      {/* Decision Radio Buttons */}
      <div className="space-y-2">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Decision (Keyboard: A/C/R/S)
        </label>

        <div className="space-y-2">
          <label className="flex items-center gap-3 p-3 border-2 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors">
            <input
              type="radio"
              name="decision"
              value="approved"
              checked={decision === 'approved'}
              onChange={(e) => setDecision(e.target.value as any)}
              className="w-5 h-5"
            />
            <div className="flex-1">
              <span className="font-semibold text-green-700">Approve</span>
              <span className="text-sm text-gray-600 ml-2">(A)</span>
              <p className="text-sm text-gray-500">Consensus answer is correct</p>
            </div>
          </label>

          <label className="flex items-center gap-3 p-3 border-2 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors">
            <input
              type="radio"
              name="decision"
              value="corrected"
              checked={decision === 'corrected'}
              onChange={(e) => setDecision(e.target.value as any)}
              className="w-5 h-5"
            />
            <div className="flex-1">
              <span className="font-semibold text-blue-700">Correct</span>
              <span className="text-sm text-gray-600 ml-2">(C)</span>
              <p className="text-sm text-gray-500">Provide corrected answer</p>
            </div>
          </label>

          <label className="flex items-center gap-3 p-3 border-2 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors">
            <input
              type="radio"
              name="decision"
              value="rejected"
              checked={decision === 'rejected'}
              onChange={(e) => setDecision(e.target.value as any)}
              className="w-5 h-5"
            />
            <div className="flex-1">
              <span className="font-semibold text-red-700">Reject</span>
              <span className="text-sm text-gray-600 ml-2">(R)</span>
              <p className="text-sm text-gray-500">All predictions are incorrect</p>
            </div>
          </label>
        </div>
      </div>

      {/* Corrected Answer Field (shown only when corrected is selected) */}
      {decision === 'corrected' && (
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Corrected Answer *
          </label>
          <textarea
            value={correctedAnswer}
            onChange={(e) => setCorrectedAnswer(e.target.value)}
            placeholder="Enter the corrected answer (JSON or text)"
            className="w-full px-4 py-2 border-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={4}
            required={decision === 'corrected'}
          />
          {consensusAnswer && (
            <button
              type="button"
              onClick={() => setCorrectedAnswer(JSON.stringify(consensusAnswer, null, 2))}
              className="mt-2 text-sm text-blue-600 hover:text-blue-700 underline"
            >
              Use consensus answer as template
            </button>
          )}
        </div>
      )}

      {/* Confidence Level */}
      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Confidence Level
        </label>
        <select
          value={confidence}
          onChange={(e) => setConfidence(e.target.value as any)}
          className="w-full px-4 py-2 border-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="high">High - Very confident in this review</option>
          <option value="medium">Medium - Moderately confident</option>
          <option value="low">Low - Should be double-checked</option>
        </select>
      </div>

      {/* Notes */}
      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Notes (optional)
        </label>
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Add any notes or comments about this review..."
          className="w-full px-4 py-2 border-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={3}
        />
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3 pt-4">
        <button
          type="submit"
          disabled={isSubmitting || (decision === 'corrected' && !correctedAnswer.trim())}
          className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          {isSubmitting ? 'Submitting...' : 'Submit Review'}
        </button>

        <button
          type="button"
          onClick={onSkip}
          disabled={isSubmitting}
          className="px-6 py-3 border-2 border-gray-300 rounded-lg font-semibold hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Skip (S)
        </button>
      </div>
    </form>
  )
}
