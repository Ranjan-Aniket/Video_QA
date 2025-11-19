interface FailureTypeBadgeProps {
  type: string
}

export default function FailureTypeBadge({ type }: FailureTypeBadgeProps) {
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