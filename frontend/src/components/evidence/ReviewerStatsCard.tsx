import type { ReviewerStats } from '../../types/evidence'

interface ReviewerStatsCardProps {
  stats: ReviewerStats
  reviewerId: string
  dateRange?: string
}

export default function ReviewerStatsCard({
  stats,
  reviewerId,
  dateRange = 'Today',
}: ReviewerStatsCardProps) {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}m ${secs}s`
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800">Reviewer Statistics</h2>
        <p className="text-sm text-gray-600">
          {reviewerId} • {dateRange}
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
          <p className="text-3xl font-bold text-blue-700">{stats.items_reviewed}</p>
          <p className="text-sm text-gray-600 mt-1">Items Reviewed</p>
        </div>

        <div className="bg-purple-50 rounded-lg p-4 border-2 border-purple-200">
          <p className="text-3xl font-bold text-purple-700">
            {formatTime(stats.avg_review_time)}
          </p>
          <p className="text-sm text-gray-600 mt-1">Avg Review Time</p>
        </div>

        <div className="bg-green-50 rounded-lg p-4 border-2 border-green-200">
          <p className="text-3xl font-bold text-green-700">
            {stats.approval_rate.toFixed(1)}%
          </p>
          <p className="text-sm text-gray-600 mt-1">Approval Rate</p>
        </div>
      </div>

      {/* Breakdown */}
      <div className="space-y-3">
        <h3 className="font-semibold text-gray-700 mb-3">Review Breakdown</h3>

        <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 bg-green-500 rounded-full" />
            <span className="font-medium text-gray-700">Approved</span>
          </div>
          <div className="text-right">
            <p className="text-xl font-bold text-green-700">{stats.approved}</p>
            <p className="text-xs text-gray-600">
              {((stats.approved / stats.items_reviewed) * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 bg-blue-500 rounded-full" />
            <span className="font-medium text-gray-700">Corrected</span>
          </div>
          <div className="text-right">
            <p className="text-xl font-bold text-blue-700">{stats.corrected}</p>
            <p className="text-xs text-gray-600">
              {stats.correction_rate.toFixed(1)}%
            </p>
          </div>
        </div>

        <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 bg-red-500 rounded-full" />
            <span className="font-medium text-gray-700">Rejected</span>
          </div>
          <div className="text-right">
            <p className="text-xl font-bold text-red-700">{stats.rejected}</p>
            <p className="text-xs text-gray-600">
              {((stats.rejected / stats.items_reviewed) * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>

      {/* Performance Indicators */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">Efficiency</span>
          <span className="text-sm font-bold text-gray-900">
            {stats.avg_review_time < 60
              ? 'Excellent'
              : stats.avg_review_time < 120
              ? 'Good'
              : 'Needs Improvement'}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700">Accuracy</span>
          <span className="text-sm font-bold text-gray-900">
            {stats.approval_rate > 70
              ? 'High'
              : stats.approval_rate > 50
              ? 'Medium'
              : 'Low'}
          </span>
        </div>
      </div>
    </div>
  )
}
