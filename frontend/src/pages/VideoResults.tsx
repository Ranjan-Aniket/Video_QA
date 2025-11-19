import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { getVideoResults } from '../api/client'
import VideoPlayer from '../components/VideoPlayer'
import QuestionCard from '../components/QuestionCard'

/**
 * Video Results Page
 * 
 * Following EXACT design from architecture:
 * - Video player with frame-level control
 * - Summary panel (candidates, tested, failures, selected, time, cost)
 * - Selected Questions (4) with expandable cards
 * - All candidates table (sortable, filterable)
 * - Validation results panel
 */
export default function VideoResults() {
  const { videoId } = useParams<{ videoId: string }>()
  const [selectedTab, setSelectedTab] = useState<'selected' | 'all'>('selected')
  
  const { data: results, isLoading } = useQuery({
    queryKey: ['video-results', videoId],
    queryFn: () => getVideoResults(parseInt(videoId!)),
    enabled: !!videoId,
  })

  if (isLoading) {
    return <div className="p-8">Loading results...</div>
  }

  if (!results) {
    return <div className="p-8">Video not found</div>
  }

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">{results.title}</h1>
        <div className="text-gray-600 mt-1">Video ID: {videoId}</div>
      </div>

      {/* Summary Panel */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-xl font-semibold mb-4">Processing Summary</h2>
        <div className="grid grid-cols-6 gap-4 text-sm">
          <div>
            <div className="text-gray-500">Candidates Generated</div>
            <div className="text-2xl font-bold">{results.candidates_generated}</div>
          </div>
          <div>
            <div className="text-gray-500">Gemini Tested</div>
            <div className="text-2xl font-bold">{results.gemini_tested}</div>
          </div>
          <div>
            <div className="text-gray-500">Failures Found</div>
            <div className="text-2xl font-bold text-red-600">{results.failures_found}</div>
          </div>
          <div>
            <div className="text-gray-500">Final Selected</div>
            <div className="text-2xl font-bold text-green-600">{results.final_selected}</div>
          </div>
          <div>
            <div className="text-gray-500">Processing Time</div>
            <div className="text-2xl font-bold">{results.processing_time}</div>
          </div>
          <div>
            <div className="text-gray-500">API Cost</div>
            <div className="text-2xl font-bold">${results.api_cost.toFixed(2)}</div>
          </div>
        </div>
      </div>

      {/* Video Player */}
      <div className="bg-white p-6 rounded-lg shadow">
        <VideoPlayer 
          videoUrl={results.video_url}
          duration={results.duration}
        />
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b">
        <button
          onClick={() => setSelectedTab('selected')}
          className={`px-4 py-2 font-medium ${
            selectedTab === 'selected' 
              ? 'border-b-2 border-blue-600 text-blue-600' 
              : 'text-gray-600'
          }`}
        >
          Selected Questions ({results.selected_questions.length})
        </button>
        <button
          onClick={() => setSelectedTab('all')}
          className={`px-4 py-2 font-medium ${
            selectedTab === 'all' 
              ? 'border-b-2 border-blue-600 text-blue-600' 
              : 'text-gray-600'
          }`}
        >
          All Candidates ({results.all_candidates.length})
        </button>
      </div>

      {/* Selected Questions */}
      {selectedTab === 'selected' && (
        <div className="space-y-4">
          {results.selected_questions.map((question: any, index: number) => (
            <QuestionCard 
              key={question.id}
              question={question}
              index={index + 1}
              videoUrl={results.video_url}
            />
          ))}
        </div>
      )}

      {/* All Candidates Table */}
      {selectedTab === 'all' && (
        <div className="bg-white p-6 rounded-lg shadow">
          <table className="w-full">
            <thead className="border-b">
              <tr className="text-left text-gray-600 text-sm">
                <th className="pb-3">Question</th>
                <th className="pb-3">Task Type</th>
                <th className="pb-3">Failure Type</th>
                <th className="pb-3">Score</th>
                <th className="pb-3">Validation</th>
                <th className="pb-3">Selected</th>
              </tr>
            </thead>
            <tbody>
              {results.all_candidates.map((candidate: any) => (
                <tr key={candidate.id} className="border-b hover:bg-gray-50">
                  <td className="py-3 text-sm max-w-md truncate">{candidate.question}</td>
                  <td className="py-3 text-sm">{candidate.task_type}</td>
                  <td className="py-3">
                    {candidate.failure_type && (
                      <span className="px-2 py-1 bg-red-100 text-red-800 rounded text-xs">
                        {candidate.failure_type}
                      </span>
                    )}
                  </td>
                  <td className="py-3 text-sm">{candidate.score}/10</td>
                  <td className="py-3">
                    <span className={`px-2 py-1 rounded text-xs ${
                      candidate.validation_passed 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {candidate.validation_passed ? 'Pass' : 'Fail'}
                    </span>
                  </td>
                  <td className="py-3">
                    {candidate.selected && <span className="text-green-600">✓</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}