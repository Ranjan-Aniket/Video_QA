interface EvidenceTrailProps {
  audioCues: string[]
  visualCues: string[]
  onCueClick?: (cue: string, type: 'audio' | 'visual') => void
}

export default function EvidenceTrail({ audioCues, visualCues, onCueClick }: EvidenceTrailProps) {
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