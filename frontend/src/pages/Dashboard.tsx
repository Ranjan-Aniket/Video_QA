import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { getDashboardStats } from '../api/client'

/**
 * Dashboard - Home Page
 * 
 * Following EXACT design from architecture:
 * - Quick stats cards (Processed, In Progress, Success Rate, Profit)
 * - Progress bar to 1M videos
 * - Financial overview
 * - Recent batches list
 * - System learning visualization
 * - Quick action buttons
 */
export default function Dashboard() {
  const { data: stats } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: getDashboardStats,
    refetchInterval: 30000, // Refresh every 30s
  })

  const progressTo1M = ((stats?.total_processed || 0) / 1000000) * 100

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">Video Q&A Dashboard</h1>
        <Link to="/upload">
          <button className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg">
            + New Batch
          </button>
        </Link>
      </div>

      {/* Quick Stats Cards */}
      <div className="grid grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-gray-500 text-sm">Total Processed</div>
          <div className="text-3xl font-bold mt-2">{stats?.total_processed?.toLocaleString() || 0}</div>
          <div className="text-green-600 text-sm mt-2">+{stats?.processed_today || 0} today</div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-gray-500 text-sm">In Progress</div>
          <div className="text-3xl font-bold mt-2">{stats?.in_progress || 0}</div>
          <div className="text-gray-600 text-sm mt-2">{stats?.parallel_workers || 0} workers active</div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-gray-500 text-sm">Success Rate</div>
          <div className="text-3xl font-bold mt-2">{(stats?.success_rate || 0).toFixed(1)}%</div>
          <div className="text-gray-600 text-sm mt-2">Target: 99.9%</div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-gray-500 text-sm">Profit Margin</div>
          <div className="text-3xl font-bold mt-2">{(stats?.profit_margin || 0).toFixed(1)}%</div>
          <div className="text-green-600 text-sm mt-2">${(stats?.total_profit || 0).toFixed(2)} earned</div>
        </div>
      </div>

      {/* Progress to 1M Videos */}
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-xl font-semibold">Progress to 1 Million Videos</h2>
          <span className="text-gray-600">{(stats?.total_processed || 0).toLocaleString()} / 1,000,000</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-4">
          <div 
            className="bg-blue-600 h-4 rounded-full transition-all duration-300"
            style={{ width: `${progressTo1M}%` }}
          />
        </div>
        <div className="text-sm text-gray-600 mt-2">
          {progressTo1M.toFixed(2)}% complete • Est. {stats?.days_to_1m || 0} days remaining
        </div>
      </div>

      {/* Financial Overview */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Financial Overview</h2>
        <div className="grid grid-cols-3 gap-6">
          <div>
            <div className="text-gray-500 text-sm">Revenue</div>
            <div className="text-2xl font-bold text-green-600">${(stats?.total_revenue || 0).toFixed(2)}</div>
            <div className="text-sm text-gray-600">@ $8/video</div>
          </div>
          <div>
            <div className="text-gray-500 text-sm">Costs</div>
            <div className="text-2xl font-bold text-red-600">${(stats?.total_cost || 0).toFixed(2)}</div>
            <div className="text-sm text-gray-600">Avg: ${(stats?.avg_cost_per_video || 0).toFixed(2)}/video</div>
          </div>
          <div>
            <div className="text-gray-500 text-sm">Net Profit</div>
            <div className="text-2xl font-bold text-blue-600">${(stats?.total_profit || 0).toFixed(2)}</div>
            <div className="text-sm text-gray-600">{(stats?.profit_margin || 0).toFixed(1)}% margin</div>
          </div>
        </div>
      </div>

      {/* Recent Batches */}
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold">Recent Batches</h2>
          <Link to="/batches" className="text-blue-600 hover:underline text-sm">View All</Link>
        </div>
        <table className="w-full">
          <thead className="border-b">
            <tr className="text-left text-gray-600 text-sm">
              <th className="pb-3">Batch Name</th>
              <th className="pb-3">Status</th>
              <th className="pb-3">Progress</th>
              <th className="pb-3">Success Rate</th>
              <th className="pb-3">Actions</th>
            </tr>
          </thead>
          <tbody>
            {stats?.recent_batches?.map((batch: any) => (
              <tr key={batch.id} className="border-b hover:bg-gray-50">
                <td className="py-3">{batch.name}</td>
                <td className="py-3">
                  <span className={`px-2 py-1 rounded text-xs ${
                    batch.status === 'processing' ? 'bg-yellow-100 text-yellow-800' :
                    batch.status === 'completed' ? 'bg-green-100 text-green-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {batch.status}
                  </span>
                </td>
                <td className="py-3">{batch.completed}/{batch.total}</td>
                <td className="py-3">{batch.success_rate.toFixed(1)}%</td>
                <td className="py-3">
                  <Link to={`/batch/${batch.id}`} className="text-blue-600 hover:underline text-sm">
                    View →
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* System Learning Visualization */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Failure Pattern Learning</h2>
        <div className="text-gray-600 text-sm">
          Most common Gemini failures: {stats?.top_failure_types?.join(', ') || 'Loading...'}
        </div>
      </div>
    </div>
  )
}