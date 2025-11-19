interface CostTrackerProps {
  currentCost: number
  budgetLimit: number
}

export default function CostTracker({ currentCost, budgetLimit }: CostTrackerProps) {
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