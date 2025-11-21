import { useState } from 'react'

interface QuestionCardProps {
  question: any
  index: number
  videoUrl: string
  onSeekTo?: (timestamp: string) => void  // Callback to seek video to timestamp
}

/**
 * QuestionCard Component
 *
 * Following EXACT design from architecture:
 * - Question text
 * - Gemini answer vs Golden answer comparison
 * - Failure type badge
 * - Score (0-10)
 * - Timestamp range (NOW VISIBLE IN COLLAPSED VIEW!)
 * - [▶️ Play Segment] button
 * - [📝 View Full] modal
 * - Evidence trail (clickable)
 */
export default function QuestionCard({ question, index, videoUrl: _videoUrl, onSeekTo }: QuestionCardProps) {
  const [expanded, setExpanded] = useState(false)

  // Convert HH:MM:SS to seconds
  const parseTimestamp = (timestamp: string): number => {
    const parts = timestamp.split(':')
    if (parts.length === 3) {
      return parseInt(parts[0]) * 3600 + parseInt(parts[1]) * 60 + parseInt(parts[2])
    }
    return 0
  }

  const handlePlaySegment = () => {
    const timestampStart = question.start_timestamp || question.timestamp_start || '00:00:00'

    // If callback provided, use it
    if (onSeekTo) {
      onSeekTo(timestampStart)
    } else {
      // Otherwise, try to find video player on page
      const videoElement = document.querySelector('video') as HTMLVideoElement
      if (videoElement) {
        videoElement.currentTime = parseTimestamp(timestampStart)
        videoElement.play()
      }
    }

    console.log('Playing from:', timestampStart)
  }

  return (
    <div className="bg-white border rounded-lg shadow hover:shadow-lg transition-shadow">
      {/* Header */}
      <div
        onClick={() => setExpanded(!expanded)}
        className="p-4 cursor-pointer hover:bg-gray-50"
      >
        <div className="flex justify-between items-start">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-semibold">
                Q{index}
              </span>
              {/* Timestamp Badge - Clickable */}
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  handlePlaySegment()
                }}
                className="bg-purple-100 text-purple-800 hover:bg-purple-200 px-2 py-1 rounded text-xs font-mono font-semibold transition-colors flex items-center gap-1"
                title="Click to play video at this timestamp"
              >
                <span>🕐</span>
                <span>{question.start_timestamp || question.timestamp_start || '00:00:00'}</span>
              </button>
              <span className="text-xs text-gray-500">{question.task_type}</span>
              {question.failure_type && (
                <span className="bg-red-100 text-red-800 px-2 py-1 rounded text-xs">
                  {question.failure_type}
                </span>
              )}
              <span className="ml-auto text-sm font-semibold">
                Score: {question.score}/10
              </span>
            </div>
            <div className="text-gray-900 font-medium">{question.question_text}</div>
          </div>
          <button className="ml-4 text-gray-400 hover:text-gray-600">
            {expanded ? '▼' : '▶'}
          </button>
        </div>
      </div>

      {/* Expanded Content */}
      {expanded && (
        <div className="border-t p-4 space-y-4 bg-gray-50">
          {/* Answers Comparison */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-red-50 p-3 rounded">
              <div className="text-sm font-semibold text-red-800 mb-2">❌ Gemini Answer (Incorrect)</div>
              <div className="text-sm text-gray-700">{question.gemini_answer}</div>
            </div>
            <div className="bg-green-50 p-3 rounded">
              <div className="text-sm font-semibold text-green-800 mb-2">✓ Golden Answer (Correct)</div>
              <div className="text-sm text-gray-700">{question.golden_answer}</div>
            </div>
          </div>

          {/* Why Gemini Failed */}
          {question.failure_explanation && (
            <div className="bg-yellow-50 p-3 rounded">
              <div className="text-sm font-semibold text-yellow-800 mb-1">Why Gemini Failed:</div>
              <div className="text-sm text-gray-700">{question.failure_explanation}</div>
            </div>
          )}

          {/* Evidence Trail */}
          <div className="bg-white p-3 rounded border">
            <div className="text-sm font-semibold mb-2">Evidence Trail:</div>
            <div className="space-y-2 text-sm">
              {question.audio_cues?.map((cue: string, i: number) => (
                <div key={`audio-${i}`} className="flex items-start gap-2">
                  <span className="text-blue-600">🎤</span>
                  <span className="text-gray-700">"{cue}"</span>
                </div>
              ))}
              {question.visual_cues?.map((cue: string, i: number) => (
                <div key={`visual-${i}`} className="flex items-start gap-2">
                  <span className="text-purple-600">👁️</span>
                  <span className="text-gray-700">{cue}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Metadata */}
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="bg-purple-50 p-3 rounded border border-purple-200">
              <div className="text-purple-700 font-semibold mb-1 flex items-center gap-1">
                <span>🕐</span>
                <span>Timestamp Range</span>
              </div>
              <div className="font-mono text-purple-900">
                {question.start_timestamp || question.timestamp_start || '00:00:00'}
                <span className="mx-2 text-purple-400">→</span>
                {question.end_timestamp || question.timestamp_end || '00:00:08'}
              </div>
            </div>
            <div>
              <div className="text-gray-500">Validation Layers Passed</div>
              <div>{question.validation_layers_passed}/15</div>
            </div>
            <div>
              <div className="text-gray-500">Complexity Score</div>
              <div>{question.complexity_score}</div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <button
              onClick={handlePlaySegment}
              className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded text-sm font-semibold transition-colors flex items-center gap-2"
            >
              <span>▶️</span>
              <span>Play at {question.start_timestamp || question.timestamp_start || '00:00:00'}</span>
            </button>
            <button className="bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded text-sm">
              📝 View Full Details
            </button>
          </div>
        </div>
      )}
    </div>
  )
}