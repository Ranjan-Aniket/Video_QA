interface ValidationBadgeProps {
  passed: number
  total: number
}

export default function ValidationBadge({ passed, total }: ValidationBadgeProps) {
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