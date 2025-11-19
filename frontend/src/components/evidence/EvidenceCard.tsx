import type { EvidenceItem } from '../../types/evidence'

interface EvidenceCardProps {
  evidence: EvidenceItem
  onSelect?: () => void
  isSelected?: boolean
}

const evidenceTypeLabels = {
  audio_transcript: 'Audio Transcript',
  visual_text: 'Visual Text',
  visual_action: 'Visual Action',
}

const evidenceTypeIcons = {
  audio_transcript: '🎤',
  visual_text: '📝',
  visual_action: '🎬',
}

const priorityColors = {
  high: 'bg-red-100 border-red-500 text-red-800',
  medium: 'bg-yellow-100 border-yellow-500 text-yellow-800',
  low: 'bg-green-100 border-green-500 text-green-800',
}

const statusColors = {
  pending: 'bg-gray-100 text-gray-800',
  approved: 'bg-green-100 text-green-800',
  corrected: 'bg-blue-100 text-blue-800',
  rejected: 'bg-red-100 text-red-800',
  skipped: 'bg-yellow-100 text-yellow-800',
}

export default function EvidenceCard({
  evidence,
  onSelect,
  isSelected = false,
}: EvidenceCardProps) {
  const formatTimestamp = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div
      className={`border-2 rounded-lg p-4 transition-all cursor-pointer hover:shadow-lg ${
        isSelected ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-300'
      }`}
      onClick={onSelect}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-2xl">
            {evidenceTypeIcons[evidence.evidence_type]}
          </span>
          <div>
            <h3 className="font-semibold text-lg">
              {evidenceTypeLabels[evidence.evidence_type]}
            </h3>
            <p className="text-sm text-gray-600">
              {formatTimestamp(evidence.timestamp_start)} - {formatTimestamp(evidence.timestamp_end)}
            </p>
          </div>
        </div>

        <div className="flex flex-col gap-1 items-end">
          <span
            className={`text-xs font-semibold px-2 py-1 rounded ${
              priorityColors[evidence.priority_level]
            } border`}
          >
            {evidence.priority_level.toUpperCase()}
          </span>
          <span
            className={`text-xs font-semibold px-2 py-1 rounded ${
              statusColors[evidence.human_review_status]
            }`}
          >
            {evidence.human_review_status.toUpperCase()}
          </span>
        </div>
      </div>

      {/* Flag Reason */}
      {evidence.flag_reason && (
        <div className="mb-3 bg-yellow-50 border-l-4 border-yellow-500 p-2">
          <p className="text-sm font-semibold text-yellow-800">
            Flag Reason:
          </p>
          <p className="text-sm text-yellow-700">{evidence.flag_reason}</p>
        </div>
      )}

      {/* Consensus Answer Preview */}
      <div className="bg-gray-50 rounded p-3 mb-3">
        <p className="text-xs font-semibold text-gray-600 mb-1">Consensus Answer:</p>
        <p className="text-sm line-clamp-2">
          {typeof evidence.consensus_answer === 'string'
            ? evidence.consensus_answer
            : JSON.stringify(evidence.consensus_answer)}
        </p>
      </div>

      {/* Confidence Score */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-gray-600">Confidence:</span>
          <div className="flex-1 bg-gray-200 rounded-full h-2 w-24">
            <div
              className={`h-2 rounded-full ${
                evidence.confidence_score >= 0.7
                  ? 'bg-green-500'
                  : evidence.confidence_score >= 0.4
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
              }`}
              style={{ width: `${evidence.confidence_score * 100}%` }}
            />
          </div>
          <span className="text-sm font-semibold">
            {(evidence.confidence_score * 100).toFixed(0)}%
          </span>
        </div>

        {evidence.human_reviewer_id && (
          <span className="text-xs text-gray-500">
            Reviewed by: {evidence.human_reviewer_id}
          </span>
        )}
      </div>

      {/* Human Notes (if reviewed) */}
      {evidence.human_notes && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <p className="text-xs font-semibold text-gray-600 mb-1">Review Notes:</p>
          <p className="text-sm text-gray-700">{evidence.human_notes}</p>
        </div>
      )}
    </div>
  )
}
