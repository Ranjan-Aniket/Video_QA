import type { EvidenceItem } from '../../types/evidence'

interface ReviewHistoryListProps {
  reviews: EvidenceItem[]
  onEdit?: (evidence: EvidenceItem) => void
}

const statusIcons = {
  approved: '✓',
  corrected: '✎',
  rejected: '✗',
  skipped: '⊘',
  pending: '?',
}

const statusColors = {
  approved: 'text-green-600 bg-green-50',
  corrected: 'text-blue-600 bg-blue-50',
  rejected: 'text-red-600 bg-red-50',
  skipped: 'text-yellow-600 bg-yellow-50',
  pending: 'text-gray-600 bg-gray-50',
}

export default function ReviewHistoryList({
  reviews,
  onEdit,
}: ReviewHistoryListProps) {
  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A'
    const date = new Date(dateString)
    return date.toLocaleString()
  }

  const formatDuration = (seconds?: number) => {
    if (!seconds) return 'N/A'
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}m ${secs}s`
  }

  const sortedReviews = [...reviews]
    .filter((r) => r.human_review_status !== 'pending')
    .sort((a, b) => {
      if (!a.review_timestamp || !b.review_timestamp) return 0
      return new Date(b.review_timestamp).getTime() - new Date(a.review_timestamp).getTime()
    })

  if (sortedReviews.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-bold mb-4">Review History</h3>
        <p className="text-gray-500 text-center py-8">No reviews yet</p>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-xl font-bold mb-4">
        Review History ({sortedReviews.length})
      </h3>

      <div className="space-y-3 max-h-[600px] overflow-y-auto">
        {sortedReviews.map((review) => (
          <div
            key={review.id}
            className={`border-2 rounded-lg p-4 transition-all hover:shadow-md ${
              statusColors[review.human_review_status]
            }`}
          >
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-3">
                <span className="text-2xl">
                  {statusIcons[review.human_review_status]}
                </span>
                <div>
                  <p className="font-semibold">
                    Evidence #{review.id} - {review.evidence_type.replace('_', ' ')}
                  </p>
                  <p className="text-sm text-gray-600">
                    {formatDate(review.review_timestamp)}
                  </p>
                </div>
              </div>

              {onEdit && (
                <button
                  onClick={() => onEdit(review)}
                  className="text-sm text-blue-600 hover:text-blue-700 underline"
                >
                  Edit
                </button>
              )}
            </div>

            {/* Review Details */}
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-gray-700">Decision:</span>
                <span className="capitalize font-medium">
                  {review.human_review_status}
                </span>
              </div>

              {review.human_confidence && (
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-gray-700">Confidence:</span>
                  <span className="capitalize">{review.human_confidence}</span>
                </div>
              )}

              {review.review_duration_seconds && (
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-gray-700">Duration:</span>
                  <span>{formatDuration(review.review_duration_seconds)}</span>
                </div>
              )}

              {review.human_reviewer_id && (
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-gray-700">Reviewer:</span>
                  <span>{review.human_reviewer_id}</span>
                </div>
              )}

              {review.human_notes && (
                <div className="mt-3 pt-3 border-t border-current border-opacity-20">
                  <p className="font-semibold text-gray-700 mb-1">Notes:</p>
                  <p className="text-gray-700">{review.human_notes}</p>
                </div>
              )}

              {review.human_answer && review.human_review_status === 'corrected' && (
                <div className="mt-3 pt-3 border-t border-current border-opacity-20">
                  <p className="font-semibold text-gray-700 mb-1">Corrected Answer:</p>
                  <pre className="text-xs bg-white bg-opacity-50 p-2 rounded overflow-auto max-h-32">
                    {typeof review.human_answer === 'string'
                      ? review.human_answer
                      : JSON.stringify(review.human_answer, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
