import type { EvidenceItem } from '../../types/evidence'

interface EvidenceTimelineProps {
  videoId?: string
  evidenceItems: EvidenceItem[]
  currentItem?: EvidenceItem
  onSelect: (item: EvidenceItem) => void
}

const statusColors = {
  pending: 'bg-gray-400',
  approved: 'bg-green-500',
  corrected: 'bg-blue-500',
  rejected: 'bg-red-500',
  skipped: 'bg-yellow-500',
}

export default function EvidenceTimeline({
  evidenceItems,
  currentItem,
  onSelect,
}: EvidenceTimelineProps) {
  const sortedItems = [...evidenceItems].sort(
    (a, b) => a.timestamp_start - b.timestamp_start
  )

  const maxTimestamp = Math.max(...sortedItems.map((item) => item.timestamp_end))

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-xl font-bold mb-4">Evidence Timeline</h3>

      <div className="relative">
        {/* Timeline Bar */}
        <div className="relative h-12 bg-gray-200 rounded-lg overflow-hidden mb-4">
          {sortedItems.map((item) => {
            const left = (item.timestamp_start / maxTimestamp) * 100
            const width =
              ((item.timestamp_end - item.timestamp_start) / maxTimestamp) * 100
            const isActive = currentItem?.id === item.id

            return (
              <div
                key={item.id}
                className={`absolute h-full ${
                  statusColors[item.human_review_status]
                } ${
                  isActive ? 'ring-4 ring-blue-400 z-10' : 'hover:ring-2 hover:ring-blue-300'
                } cursor-pointer transition-all`}
                style={{
                  left: `${left}%`,
                  width: `${width}%`,
                }}
                onClick={() => onSelect(item)}
                title={`${formatTime(item.timestamp_start)} - ${formatTime(
                  item.timestamp_end
                )}`}
              />
            )
          })}
        </div>

        {/* Time Markers */}
        <div className="flex justify-between text-xs text-gray-600 mb-4">
          <span>0:00</span>
          <span>{formatTime(maxTimestamp / 2)}</span>
          <span>{formatTime(maxTimestamp)}</span>
        </div>
      </div>

      {/* Evidence List */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {sortedItems.map((item) => {
          const isActive = currentItem?.id === item.id

          return (
            <div
              key={item.id}
              className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all ${
                isActive
                  ? 'bg-blue-100 border-2 border-blue-500'
                  : 'bg-gray-50 hover:bg-gray-100 border-2 border-transparent'
              }`}
              onClick={() => onSelect(item)}
            >
              <div
                className={`w-3 h-3 rounded-full ${
                  statusColors[item.human_review_status]
                }`}
              />
              <div className="flex-1">
                <p className="text-sm font-semibold">
                  {formatTime(item.timestamp_start)} - {formatTime(item.timestamp_end)}
                </p>
                <p className="text-xs text-gray-600">
                  {item.evidence_type.replace('_', ' ')} • {item.priority_level} priority
                </p>
              </div>
              <span
                className={`text-xs font-semibold px-2 py-1 rounded ${
                  item.human_review_status === 'pending'
                    ? 'bg-gray-200 text-gray-700'
                    : item.human_review_status === 'approved'
                    ? 'bg-green-100 text-green-700'
                    : item.human_review_status === 'corrected'
                    ? 'bg-blue-100 text-blue-700'
                    : item.human_review_status === 'rejected'
                    ? 'bg-red-100 text-red-700'
                    : 'bg-yellow-100 text-yellow-700'
                }`}
              >
                {item.human_review_status}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
