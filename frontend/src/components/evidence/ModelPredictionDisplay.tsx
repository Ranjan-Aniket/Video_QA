import type { ModelType } from '../../types/evidence'

interface ModelPredictionDisplayProps {
  model: ModelType
  prediction: any
  confidence?: number
  isCorrect?: boolean
  isSelected?: boolean
}

const modelNames: Record<ModelType, string> = {
  gpt4: 'GPT-4',
  claude: 'Claude',
  open: 'Open Model',
}

const modelColors: Record<ModelType, string> = {
  gpt4: 'bg-blue-100 border-blue-400 text-blue-900',
  claude: 'bg-purple-100 border-purple-400 text-purple-900',
  open: 'bg-green-100 border-green-400 text-green-900',
}

export default function ModelPredictionDisplay({
  model,
  prediction,
  confidence,
  isCorrect,
  isSelected,
}: ModelPredictionDisplayProps) {
  const colorClass = modelColors[model]
  const borderClass = isSelected ? 'border-4' : 'border-2'

  return (
    <div
      className={`${colorClass} ${borderClass} rounded-lg p-4 relative transition-all hover:shadow-md`}
    >
      {/* Model Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h3 className="font-bold text-lg">{modelNames[model]}</h3>
          {isCorrect !== undefined && (
            <span className="text-xl">
              {isCorrect ? '✓' : '✗'}
            </span>
          )}
        </div>
        {confidence !== undefined && (
          <span className="text-sm font-semibold px-2 py-1 rounded bg-white bg-opacity-50">
            {(confidence * 100).toFixed(0)}% confident
          </span>
        )}
      </div>

      {/* Prediction Content */}
      <div className="bg-white bg-opacity-50 rounded p-3">
        {typeof prediction === 'string' ? (
          <p className="text-sm whitespace-pre-wrap">{prediction}</p>
        ) : prediction ? (
          <pre className="text-sm overflow-auto max-h-40 whitespace-pre-wrap">
            {JSON.stringify(prediction, null, 2)}
          </pre>
        ) : (
          <p className="text-sm italic text-gray-600">No prediction available</p>
        )}
      </div>

      {/* Selected Indicator */}
      {isSelected && (
        <div className="absolute -top-2 -right-2 bg-green-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold shadow-lg">
          ✓
        </div>
      )}
    </div>
  )
}
