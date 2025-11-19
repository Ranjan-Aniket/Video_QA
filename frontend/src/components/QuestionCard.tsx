import { useState } from 'react'

interface QuestionCardProps {
  question: any
  index: number
  videoUrl: string
}

/**
 * QuestionCard Component
 * 
 * Following EXACT design from architecture:
 * - Question text
 * - Gemini answer vs Golden answer comparison
 * - Failure type badge
 * - Score (0-10)
 * - Timestamp range
 * - [▶️ Play Segment] button
 * - [📝 View Full] modal
 * - Evidence trail (clickable)
 */
export default function QuestionCard({ question, index, videoUrl: _videoUrl }: QuestionCardProps) {
  const [expanded, setExpanded] = useState(false)

  const handlePlaySegment = () => {
    // Would trigger video player to jump to timestamp
    const timestampStart = question.timestamp_start
    console.log('Play from:', timestampStart)
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
            <div>
              <div className="text-gray-500">Timestamp</div>
              <div className="font-mono">{question.timestamp_start} - {question.timestamp_end}</div>
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
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded text-sm"
            >
              ▶️ Play Segment
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