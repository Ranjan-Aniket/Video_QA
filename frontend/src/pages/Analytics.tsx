import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getAnalytics } from '../api/client'
import { PieChart, Pie, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts'

/**
 * Analytics Page
 * 
 * Following EXACT design from architecture:
 * - Date range selector
 * - Key metrics grid
 * - Charts: failure types (pie), task types (bar), processing time (histogram), cost breakdown
 * - Quality score trend over time
 */
export default function Analytics() {
  const [dateRange, setDateRange] = useState({ start: '', end: '' })

  const { data: analytics } = useQuery({
    queryKey: ['analytics', dateRange],
    queryFn: () => getAnalytics(dateRange),
  })

  const COLORS = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899']

  return (
    <div className="p-8 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">Analytics</h1>
        
        {/* Date Range Selector */}
        <div className="flex gap-3 items-center">
          <input
            type="date"
            value={dateRange.start}
            onChange={(e) => setDateRange({...dateRange, start: e.target.value})}
            className="border border-gray-300 rounded px-3 py-2"
          />
          <span>to</span>
          <input
            type="date"
            value={dateRange.end}
            onChange={(e) => setDateRange({...dateRange, end: e.target.value})}
            className="border border-gray-300 rounded px-3 py-2"
          />
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-gray-500 text-sm">Total Processed</div>
          <div className="text-3xl font-bold">{analytics?.total_processed || 0}</div>
          <div className="text-green-600 text-sm mt-1">
            {analytics?.percent_change_processed > 0 && '↑'} {analytics?.percent_change_processed}% vs last period
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-gray-500 text-sm">Success Rate</div>
          <div className="text-3xl font-bold">{analytics?.success_rate?.toFixed(1)}%</div>
          <div className="text-gray-600 text-sm mt-1">Target: 99.9%</div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-gray-500 text-sm">Avg Cost/Video</div>
          <div className="text-3xl font-bold">${analytics?.avg_cost?.toFixed(2)}</div>
          <div className={`text-sm mt-1 ${analytics?.cost_trend === 'down' ? 'text-green-600' : 'text-red-600'}`}>
            {analytics?.cost_trend === 'down' ? '↓' : '↑'} {analytics?.cost_change}% trend
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <div className="text-gray-500 text-sm">Profit Margin</div>
          <div className="text-3xl font-bold">{analytics?.profit_margin?.toFixed(1)}%</div>
          <div className="text-green-600 text-sm mt-1">${analytics?.total_profit?.toFixed(2)} earned</div>
        </div>
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-2 gap-6">
        {/* Gemini Failure Types Pie Chart */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Gemini Failure Types</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={analytics?.failure_types || []}
                dataKey="count"
                nameKey="type"
                cx="50%"
                cy="50%"
                outerRadius={100}
                label
              >
                {analytics?.failure_types?.map((_entry: any, index: number) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Questions by Task Type Bar Chart */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Questions by Task Type</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={analytics?.task_types || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="type" angle={-45} textAnchor="end" height={100} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#3B82F6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-2 gap-6">
        {/* Processing Time Distribution */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Processing Time Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={analytics?.time_distribution || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="range" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#10B981" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Cost Breakdown */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Cost Breakdown</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={analytics?.cost_breakdown || []} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="component" type="category" width={100} />
              <Tooltip />
              <Bar dataKey="cost" fill="#F59E0B" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Quality Score Trend Over Time */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Quality Score Trend Over Time</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={analytics?.quality_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="score" stroke="#3B82F6" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Learning Insights */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Learning Insights</h2>
        <div className="space-y-3">
          <div>
            <div className="text-sm text-gray-600">Most Effective Failure Patterns</div>
            <div className="text-lg font-medium">{analytics?.effective_patterns?.join(', ') || 'N/A'}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Improved Areas</div>
            <div className="text-lg font-medium text-green-600">{analytics?.improved_areas?.join(', ') || 'N/A'}</div>
          </div>
          <div>
            <div className="text-sm text-gray-600">Problematic Areas</div>
            <div className="text-lg font-medium text-red-600">{analytics?.problematic_areas?.join(', ') || 'N/A'}</div>
          </div>
        </div>
      </div>
    </div>
  )
}