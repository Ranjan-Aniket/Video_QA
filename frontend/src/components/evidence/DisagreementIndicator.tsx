import type { DisagreementInfo } from '../../types/evidence'

interface DisagreementIndicatorProps {
  disagreementInfo: DisagreementInfo
}

const levelColors = {
  none: 'bg-green-100 border-green-500 text-green-800',
  low: 'bg-yellow-100 border-yellow-500 text-yellow-800',
  medium: 'bg-orange-100 border-orange-500 text-orange-800',
  high: 'bg-red-100 border-red-500 text-red-800',
}

const levelIcons = {
  none: '✓',
  low: '⚠',
  medium: '⚠⚠',
  high: '⚠⚠⚠',
}

const levelLabels = {
  none: 'All Models Agree',
  low: 'Minor Disagreement',
  medium: 'Moderate Disagreement',
  high: 'Major Disagreement',
}

export default function DisagreementIndicator({
  disagreementInfo,
}: DisagreementIndicatorProps) {
  const { level, disagreementCount, agreeingModels, disagreeingModels } = disagreementInfo
  const colorClass = levelColors[level]
  const icon = levelIcons[level]
  const label = levelLabels[level]

  return (
    <div className={`${colorClass} border-2 rounded-lg p-4`}>
      <div className="flex items-center gap-3 mb-2">
        <span className="text-2xl">{icon}</span>
        <div>
          <h3 className="font-bold text-lg">{label}</h3>
          <p className="text-sm">
            {disagreementCount === 0
              ? 'All 3 models agree'
              : `${disagreementCount} model${disagreementCount > 1 ? 's' : ''} disagree`}
          </p>
        </div>
      </div>

      {disagreementCount > 0 && (
        <div className="mt-3 pt-3 border-t border-current border-opacity-30">
          <div className="grid grid-cols-2 gap-2 text-sm">
            {agreeingModels.length > 0 && (
              <div>
                <p className="font-semibold mb-1">Agreeing:</p>
                <ul className="list-disc list-inside">
                  {agreeingModels.map((model) => (
                    <li key={model} className="capitalize">
                      {model === 'gpt4' ? 'GPT-4' : model}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {disagreeingModels.length > 0 && (
              <div>
                <p className="font-semibold mb-1">Disagreeing:</p>
                <ul className="list-disc list-inside">
                  {disagreeingModels.map((model) => (
                    <li key={model} className="capitalize">
                      {model === 'gpt4' ? 'GPT-4' : model}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
