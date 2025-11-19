import type { ReviewProgress } from '../../types/evidence'

interface ReviewProgressBarProps {
  progress: ReviewProgress
  showDetails?: boolean
}

export default function ReviewProgressBar({
  progress,
  showDetails = true,
}: ReviewProgressBarProps) {
  const { total, approved, corrected, rejected, pending, percent_complete, percent_approved } = progress

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold">Review Progress</h3>
        <span className="text-2xl font-bold text-blue-600">
          {percent_complete.toFixed(0)}%
        </span>
      </div>

      {/* Main Progress Bar */}
      <div className="relative w-full h-8 bg-gray-200 rounded-full overflow-hidden mb-4">
        <div
          className="absolute h-full bg-green-500 transition-all duration-300"
          style={{ width: `${(approved / total) * 100}%` }}
          title={`Approved: ${approved}`}
        />
        <div
          className="absolute h-full bg-blue-500 transition-all duration-300"
          style={{
            width: `${(corrected / total) * 100}%`,
            left: `${(approved / total) * 100}%`,
          }}
          title={`Corrected: ${corrected}`}
        />
        <div
          className="absolute h-full bg-red-500 transition-all duration-300"
          style={{
            width: `${(rejected / total) * 100}%`,
            left: `${((approved + corrected) / total) * 100}%`,
          }}
          title={`Rejected: ${rejected}`}
        />
      </div>

      {/* Legend */}
      <div className="flex gap-4 mb-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-500 rounded" />
          <span>Approved</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-blue-500 rounded" />
          <span>Corrected</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-500 rounded" />
          <span>Rejected</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-gray-300 rounded" />
          <span>Pending</span>
        </div>
      </div>

      {/* Detailed Stats */}
      {showDetails && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 pt-4 border-t border-gray-200">
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-700">{total}</p>
            <p className="text-sm text-gray-600">Total</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">{approved}</p>
            <p className="text-sm text-gray-600">Approved</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">{corrected}</p>
            <p className="text-sm text-gray-600">Corrected</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-red-600">{rejected}</p>
            <p className="text-sm text-gray-600">Rejected</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-600">{pending}</p>
            <p className="text-sm text-gray-600">Pending</p>
          </div>
        </div>
      )}

      {/* Approval Rate */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="flex items-center justify-between">
          <span className="text-sm font-semibold text-gray-700">Approval Rate:</span>
          <span className="text-lg font-bold text-green-600">
            {percent_approved.toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  )
}
