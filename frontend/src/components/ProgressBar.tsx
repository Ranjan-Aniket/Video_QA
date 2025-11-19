interface ProgressBarProps {
  current: number
  total: number
  label?: string
}

export default function ProgressBar({ current, total, label }: ProgressBarProps) {
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