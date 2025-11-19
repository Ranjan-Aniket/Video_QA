// Footer.tsx
export function Footer() {
  return (
    <footer className="bg-gray-800 text-white text-center py-4 text-sm">
      © 2024 Video Q&A Generator | Target: 1M Videos | 99.9% Hallucination-Free
    </footer>
  )
}

// ProgressBar.tsx
interface ProgressBarProps {
  current: number
  total: number
  label?: string
}

export function ProgressBar({ current, total, label }: ProgressBarProps) {
  const percent = (current / total) * 100
  
  return (
    <div className="w-full">
      {label && <div className="text-sm text-gray-600 mb-2">{label}</div>}
      <div className="w-full bg-gray-200 rounded-full h-3">
        <div 
          className="bg-blue-600 h-3 rounded-full transition-all duration-300"
          style={{ width: `${percent}%` }}
        />
      </div>
      <div className="text-xs text-gray-500 mt-1">{current} / {total} ({percent.toFixed(1)}%)</div>
    </div>
  )
}

// ValidationBadge.tsx
interface ValidationBadgeProps {
  passed: number
  total: number
}

export function ValidationBadge({ passed, total }: ValidationBadgeProps) {
  const percent = (passed / total) * 100
  const isPass = percent >= 81
  
  return (
    <span className={`px-2 py-1 rounded text-xs font-semibold ${
      isPass ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
    }`}>
      {passed}/{total} layers ({percent.toFixed(0)}%)
    </span>
  )
}

// FailureTypeBadge.tsx
interface FailureTypeBadgeProps {
  type: string
}

export function FailureTypeBadge({ type }: FailureTypeBadgeProps) {
  const colors: Record<string, string> = {
    'Counting': 'bg-red-100 text-red-800',
    'Temporal': 'bg-orange-100 text-orange-800',
    'Context': 'bg-yellow-100 text-yellow-800',
    'Needle': 'bg-purple-100 text-purple-800',
    'Spurious': 'bg-pink-100 text-pink-800',
  }
  
  return (
    <span className={`px-2 py-1 rounded text-xs ${colors[type] || 'bg-gray-100 text-gray-800'}`}>
      {type}
    </span>
  )
}

// EvidenceTrail.tsx
interface EvidenceTrailProps {
  audioCues: string[]
  visualCues: string[]
  onCueClick?: (cue: string, type: 'audio' | 'visual') => void
}

export function EvidenceTrail({ audioCues, visualCues, onCueClick }: EvidenceTrailProps) {
  return (
    <div className="space-y-2 text-sm">
      {audioCues?.map((cue, i) => (
        <button
          key={`audio-${i}`}
          onClick={() => onCueClick?.(cue, 'audio')}
          className="flex items-start gap-2 hover:bg-blue-50 p-2 rounded w-full text-left"
        >
          <span className="text-blue-600">🎤</span>
          <span className="text-gray-700">"{cue}"</span>
        </button>
      ))}
      {visualCues?.map((cue, i) => (
        <button
          key={`visual-${i}`}
          onClick={() => onCueClick?.(cue, 'visual')}
          className="flex items-start gap-2 hover:bg-purple-50 p-2 rounded w-full text-left"
        >
          <span className="text-purple-600">👁️</span>
          <span className="text-gray-700">{cue}</span>
        </button>
      ))}
    </div>
  )
}

// CostTracker.tsx
interface CostTrackerProps {
  currentCost: number
  budgetLimit: number
}

export function CostTracker({ currentCost, budgetLimit }: CostTrackerProps) {
  const percent = (currentCost / budgetLimit) * 100
  const isWarning = percent > 80
  
  return (
    <div className={`p-4 rounded-lg ${isWarning ? 'bg-red-50' : 'bg-green-50'}`}>
      <div className="text-sm text-gray-600">Cost Tracker</div>
      <div className={`text-2xl font-bold ${isWarning ? 'text-red-600' : 'text-green-600'}`}>
        ${currentCost.toFixed(2)} / ${budgetLimit.toFixed(2)}
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
        <div 
          className={`h-2 rounded-full ${isWarning ? 'bg-red-600' : 'bg-green-600'}`}
          style={{ width: `${Math.min(percent, 100)}%` }}
        />
      </div>
    </div>
  )
}

// LogsViewer.tsx
interface Log {
  timestamp: string
  level: 'info' | 'error' | 'warning'
  message: string
}

interface LogsViewerProps {
  logs: Log[]
}

export function LogsViewer({ logs }: LogsViewerProps) {
  const levelColors = {
    info: 'text-blue-600',
    warning: 'text-yellow-600',
    error: 'text-red-600',
  }
  
  return (
    <div className="bg-gray-900 text-white p-4 rounded-lg h-64 overflow-y-auto font-mono text-xs">
      {logs?.map((log, i) => (
        <div key={i} className="mb-1">
          <span className="text-gray-400">[{log.timestamp}]</span>
          <span className={`ml-2 ${levelColors[log.level]}`}>{log.level.toUpperCase()}</span>
          <span className="ml-2">{log.message}</span>
        </div>
      ))}
    </div>
  )
}

// ExportButton.tsx
interface ExportButtonProps {
  onExport: (format: 'excel' | 'csv' | 'json') => void
  label?: string
}

export function ExportButton({ onExport, label = 'Export' }: ExportButtonProps) {
  return (
    <div className="relative group">
      <button className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg">
        📥 {label}
      </button>
      <div className="absolute right-0 mt-2 w-32 bg-white rounded-lg shadow-lg hidden group-hover:block">
        <button
          onClick={() => onExport('excel')}
          className="block w-full text-left px-4 py-2 hover:bg-gray-100"
        >
          Excel
        </button>
        <button
          onClick={() => onExport('csv')}
          className="block w-full text-left px-4 py-2 hover:bg-gray-100"
        >
          CSV
        </button>
        <button
          onClick={() => onExport('json')}
          className="block w-full text-left px-4 py-2 hover:bg-gray-100"
        >
          JSON
        </button>
      </div>
    </div>
  )
}