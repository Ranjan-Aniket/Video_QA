import { useState, useEffect } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { getSettings, updateSettings } from '../api/client'
import { toast } from 'sonner'

/**
 * Settings Page
 *
 * Following EXACT design from architecture:
 * - API Keys configuration
 * - Processing settings (max workers, GPU, auto-retry, quality threshold)
 * - Cost settings (budget alerts, limits)
 * - Notification settings
 * - Database management
 */
export default function Settings() {
  const { data: settings, refetch } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })

  const [formData, setFormData] = useState(settings || {})

  // Sync formData when settings change (e.g., after refetch or initial load)
  useEffect(() => {
    if (settings) {
      setFormData(settings)
    }
  }, [settings])

  const updateMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: () => {
      toast.success('Settings saved successfully')
      refetch()
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to save settings')
    },
  })

  const handleSave = () => {
    updateMutation.mutate(formData)
  }

  return (
    <div className="p-8 max-w-4xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold">Settings</h1>

      {/* API Keys */}
      <div className="bg-white p-6 rounded-lg shadow space-y-4">
        <h2 className="text-xl font-semibold">API Keys</h2>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">OpenAI API Key</label>
          <input
            type="password"
            value={formData?.openai_api_key || ''}
            onChange={(e) => setFormData({...formData, openai_api_key: e.target.value})}
            placeholder="sk-..."
            className="w-full border border-gray-300 rounded-lg p-2"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Gemini API Key</label>
          <input
            type="password"
            value={formData?.gemini_api_key || ''}
            onChange={(e) => setFormData({...formData, gemini_api_key: e.target.value})}
            placeholder="..."
            className="w-full border border-gray-300 rounded-lg p-2"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">HuggingFace Token (Optional)</label>
          <input
            type="password"
            value={formData?.huggingface_token || ''}
            onChange={(e) => setFormData({...formData, huggingface_token: e.target.value})}
            placeholder="hf_..."
            className="w-full border border-gray-300 rounded-lg p-2"
          />
        </div>
      </div>

      {/* Processing Settings */}
      <div className="bg-white p-6 rounded-lg shadow space-y-4">
        <h2 className="text-xl font-semibold">Processing Settings</h2>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Max Parallel Workers: {formData?.max_workers || 10}
          </label>
          <input
            type="range"
            min="1"
            max="20"
            value={formData?.max_workers || 10}
            onChange={(e) => setFormData({...formData, max_workers: parseInt(e.target.value)})}
            className="w-full"
          />
          <div className="text-xs text-gray-500 mt-1">Higher = faster processing but more cost</div>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="gpu-enabled"
            checked={formData?.gpu_enabled || false}
            onChange={(e) => setFormData({...formData, gpu_enabled: e.target.checked})}
            className="w-4 h-4"
          />
          <label htmlFor="gpu-enabled" className="text-sm">Enable GPU Acceleration (if available)</label>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="auto-retry"
            checked={formData?.auto_retry || false}
            onChange={(e) => setFormData({...formData, auto_retry: e.target.checked})}
            className="w-4 h-4"
          />
          <label htmlFor="auto-retry" className="text-sm">Auto-retry on errors</label>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Quality Threshold: {formData?.quality_threshold || 0.81}
          </label>
          <input
            type="range"
            min="0.5"
            max="0.99"
            step="0.01"
            value={formData?.quality_threshold || 0.81}
            onChange={(e) => setFormData({...formData, quality_threshold: parseFloat(e.target.value)})}
            className="w-full"
          />
          <div className="text-xs text-gray-500 mt-1">Minimum validation pass rate</div>
        </div>
      </div>

      {/* Cost Settings */}
      <div className="bg-white p-6 rounded-lg shadow space-y-4">
        <h2 className="text-xl font-semibold">Cost Settings</h2>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Per-Video Budget Alert ($)</label>
          <input
            type="number"
            step="0.1"
            value={formData?.per_video_budget || 6}
            onChange={(e) => setFormData({...formData, per_video_budget: parseFloat(e.target.value)})}
            className="w-full border border-gray-300 rounded-lg p-2"
          />
          <div className="text-xs text-gray-500 mt-1">Alert if video processing exceeds this cost</div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Total Budget Limit ($)</label>
          <input
            type="number"
            step="100"
            value={formData?.total_budget_limit || 0}
            onChange={(e) => setFormData({...formData, total_budget_limit: parseFloat(e.target.value)})}
            className="w-full border border-gray-300 rounded-lg p-2"
          />
          <div className="text-xs text-gray-500 mt-1">Pause all processing if total cost exceeds (0 = no limit)</div>
        </div>
      </div>

      {/* Notification Settings */}
      <div className="bg-white p-6 rounded-lg shadow space-y-4">
        <h2 className="text-xl font-semibold">Notifications</h2>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="email-completion"
            checked={formData?.email_on_completion || false}
            onChange={(e) => setFormData({...formData, email_on_completion: e.target.checked})}
            className="w-4 h-4"
          />
          <label htmlFor="email-completion" className="text-sm">Email on batch completion</label>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="error-alerts"
            checked={formData?.error_alerts || false}
            onChange={(e) => setFormData({...formData, error_alerts: e.target.checked})}
            className="w-4 h-4"
          />
          <label htmlFor="error-alerts" className="text-sm">Error alerts</label>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="daily-summary"
            checked={formData?.daily_summary || false}
            onChange={(e) => setFormData({...formData, daily_summary: e.target.checked})}
            className="w-4 h-4"
          />
          <label htmlFor="daily-summary" className="text-sm">Daily summary report</label>
        </div>
      </div>

      {/* Database Management */}
      <div className="bg-white p-6 rounded-lg shadow space-y-4">
        <h2 className="text-xl font-semibold">Database Management</h2>

        <div className="flex gap-3">
          <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg">
            Backup Database
          </button>
          <button className="bg-yellow-600 hover:bg-yellow-700 text-white px-4 py-2 rounded-lg">
            Clear Old Data ({'>'}90 days)
          </button>
          <button className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg">
            Export All Data
          </button>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-end gap-3">
        <button
          onClick={() => setFormData(settings)}
          className="bg-gray-200 hover:bg-gray-300 text-gray-700 px-6 py-3 rounded-lg"
        >
          Reset
        </button>
        <button
          onClick={handleSave}
          disabled={updateMutation.isPending}
          className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg disabled:opacity-50"
        >
          {updateMutation.isPending ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </div>
  )
}