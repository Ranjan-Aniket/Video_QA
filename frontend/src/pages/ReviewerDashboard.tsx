import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import * as evidenceApi from '../api/evidenceApi'
import { useEvidenceStore } from '../store/useEvidenceStore'
import ReviewerStatsCard from '../components/evidence/ReviewerStatsCard'
import ReviewHistoryList from '../components/evidence/ReviewHistoryList'
import type { ReviewerStats, EvidenceItem } from '../types/evidence'

type DateRange = 'today' | 'week' | 'month' | 'all'

export default function ReviewerDashboard() {
  const navigate = useNavigate()
  const { reviewerId } = useEvidenceStore()

  const [stats, setStats] = useState<ReviewerStats | null>(null)
  const [reviewHistory, setReviewHistory] = useState<EvidenceItem[]>([])
  const [dateRange, setDateRange] = useState<DateRange>('today')
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadDashboardData()
  }, [dateRange, reviewerId])

  const loadDashboardData = async () => {
    try {
      setIsLoading(true)
      setError(null)

      // Calculate date parameter based on range
      let dateParam: string | undefined
      const today = new Date()

      if (dateRange === 'today') {
        dateParam = today.toISOString().split('T')[0]
      } else if (dateRange === 'week') {
        const weekAgo = new Date(today)
        weekAgo.setDate(weekAgo.getDate() - 7)
        dateParam = weekAgo.toISOString().split('T')[0]
      } else if (dateRange === 'month') {
        const monthAgo = new Date(today)
        monthAgo.setMonth(monthAgo.getMonth() - 1)
        dateParam = monthAgo.toISOString().split('T')[0]
      }

      // Load stats
      const statsData = await evidenceApi.getReviewerStats(reviewerId, {
        date: dateParam,
      })
      setStats(statsData)

      // Load review history
      // Note: We'd need a separate endpoint for this, but for now we'll use the queue
      // and filter reviewed items
      const queue = await evidenceApi.getReviewQueue({})
      const reviewed = queue.filter(
        (item) =>
          item.human_reviewer_id === reviewerId &&
          item.human_review_status !== 'pending'
      )
      setReviewHistory(reviewed)
    } catch (err: any) {
      setError(err.message || 'Failed to load dashboard data')
      toast.error('Failed to load dashboard data')
    } finally {
      setIsLoading(false)
    }
  }

  const handleEditReview = (evidence: EvidenceItem) => {
    navigate(`/evidence/review/${evidence.id}`)
  }

  const dateRangeLabels: Record<DateRange, string> = {
    today: 'Today',
    week: 'Last 7 Days',
    month: 'Last 30 Days',
    all: 'All Time',
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mb-4"></div>
          <p className="text-xl font-semibold">Loading dashboard...</p>
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
            onClick={loadDashboardData}
            className="bg-red-600 text-white px-6 py-2 rounded-lg hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Reviewer Dashboard
          </h1>
          <p className="text-gray-600">Reviewer ID: {reviewerId}</p>
        </div>

        {/* Date Range Selector */}
        <div className="mb-6 bg-white rounded-lg shadow-lg p-4">
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Time Period
          </label>
          <div className="flex gap-2">
            {(['today', 'week', 'month', 'all'] as DateRange[]).map((range) => (
              <button
                key={range}
                onClick={() => setDateRange(range)}
                className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                  dateRange === range
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {dateRangeLabels[range]}
              </button>
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Stats */}
          <div className="lg:col-span-1">
            {stats ? (
              <ReviewerStatsCard
                stats={stats}
                reviewerId={reviewerId}
                dateRange={dateRangeLabels[dateRange]}
              />
            ) : (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <p className="text-gray-600 text-center">No statistics available</p>
              </div>
            )}

            {/* Quick Actions */}
            <div className="mt-6 bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-bold mb-4">Quick Actions</h3>
              <div className="space-y-3">
                <button
                  onClick={() => navigate('/evidence/review')}
                  className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
                >
                  Start Reviewing
                </button>
                <button
                  onClick={loadDashboardData}
                  className="w-full bg-gray-200 text-gray-700 py-3 px-4 rounded-lg font-semibold hover:bg-gray-300 transition-colors"
                >
                  Refresh Dashboard
                </button>
              </div>
            </div>

            {/* Achievement Badges (placeholder for future enhancement) */}
            <div className="mt-6 bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-bold mb-4">Achievements</h3>
              <div className="space-y-2">
                {stats && stats.items_reviewed >= 10 && (
                  <div className="flex items-center gap-3 p-3 bg-yellow-50 rounded-lg">
                    <span className="text-2xl">🏆</span>
                    <div>
                      <p className="font-semibold text-yellow-800">
                        Review Master
                      </p>
                      <p className="text-sm text-yellow-600">
                        Reviewed 10+ items
                      </p>
                    </div>
                  </div>
                )}
                {stats && stats.approval_rate >= 70 && (
                  <div className="flex items-center gap-3 p-3 bg-green-50 rounded-lg">
                    <span className="text-2xl">⭐</span>
                    <div>
                      <p className="font-semibold text-green-800">
                        Quality Reviewer
                      </p>
                      <p className="text-sm text-green-600">
                        70%+ approval rate
                      </p>
                    </div>
                  </div>
                )}
                {stats && stats.avg_review_time < 60 && stats.items_reviewed > 5 && (
                  <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg">
                    <span className="text-2xl">⚡</span>
                    <div>
                      <p className="font-semibold text-blue-800">Speed Demon</p>
                      <p className="text-sm text-blue-600">
                        Under 60s avg review time
                      </p>
                    </div>
                  </div>
                )}
                {stats && stats.items_reviewed === 0 && (
                  <p className="text-gray-500 text-center py-4">
                    Complete reviews to earn achievements!
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Right Column - Review History */}
          <div className="lg:col-span-2">
            <ReviewHistoryList
              reviews={reviewHistory}
              onEdit={handleEditReview}
            />

            {/* Summary Stats */}
            {stats && stats.items_reviewed > 0 && (
              <div className="mt-6 bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-xl font-bold mb-4">Performance Summary</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <p className="text-3xl font-bold text-gray-700">
                      {stats.items_reviewed}
                    </p>
                    <p className="text-sm text-gray-600">Total Reviews</p>
                  </div>
                  <div className="text-center">
                    <p className="text-3xl font-bold text-green-600">
                      {stats.approval_rate.toFixed(0)}%
                    </p>
                    <p className="text-sm text-gray-600">Approval Rate</p>
                  </div>
                  <div className="text-center">
                    <p className="text-3xl font-bold text-blue-600">
                      {stats.correction_rate.toFixed(0)}%
                    </p>
                    <p className="text-sm text-gray-600">Correction Rate</p>
                  </div>
                  <div className="text-center">
                    <p className="text-3xl font-bold text-purple-600">
                      {stats.avg_review_time}s
                    </p>
                    <p className="text-sm text-gray-600">Avg Time</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
