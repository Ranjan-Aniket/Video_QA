import { useRef, useEffect, useState } from 'react'
import videojs from 'video.js'
import 'video.js/dist/video-js.css'

interface VideoPlayerProps {
  videoUrl: string
  duration: number
  onTimeUpdate?: (currentTime: number) => void
}

/**
 * VideoPlayer Component
 * 
 * Following EXACT design from architecture:
 * - Video.js based player
 * - Frame-level timestamp navigation
 * - Jump to timestamp functionality
 * - Playback controls
 */
export default function VideoPlayer({ videoUrl, duration, onTimeUpdate }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const playerRef = useRef<any>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [jumpToTime, setJumpToTime] = useState('')

  useEffect(() => {
    if (!videoRef.current) return

    // Initialize Video.js player
    const player = videojs(videoRef.current, {
      controls: true,
      responsive: true,
      fluid: true,
      playbackRates: [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2],
    })

    playerRef.current = player

    // Time update listener
    player.on('timeupdate', () => {
      const time = player.currentTime()
      if (time !== undefined) {
        setCurrentTime(time)
        onTimeUpdate?.(time)
      }
    })

    return () => {
      if (playerRef.current) {
        playerRef.current.dispose()
      }
    }
  }, [videoUrl, onTimeUpdate])

  const handleJumpTo = () => {
    if (!playerRef.current || !jumpToTime) return

    // Parse timestamp HH:MM:SS or MM:SS
    const parts = jumpToTime.split(':').map(p => parseInt(p) || 0)
    let seconds = 0
    
    if (parts.length === 3) {
      seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
    } else if (parts.length === 2) {
      seconds = parts[0] * 60 + parts[1]
    } else {
      seconds = parts[0]
    }

    playerRef.current.currentTime(seconds)
  }

  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    const s = Math.floor(seconds % 60)
    
    if (h > 0) {
      return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
    }
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-4">
      <div data-vjs-player>
        <video
          ref={videoRef}
          className="video-js vjs-big-play-centered"
        >
          <source src={videoUrl} type="video/mp4" />
        </video>
      </div>

      {/* Jump to Timestamp */}
      <div className="flex gap-3 items-center bg-gray-50 p-4 rounded-lg">
        <div className="text-sm text-gray-600">
          Current: <span className="font-mono">{formatTime(currentTime)}</span> / {formatTime(duration)}
        </div>
        <div className="flex gap-2 ml-auto">
          <input
            type="text"
            value={jumpToTime}
            onChange={(e) => setJumpToTime(e.target.value)}
            placeholder="MM:SS or HH:MM:SS"
            className="border border-gray-300 rounded px-3 py-1 text-sm font-mono w-32"
          />
          <button
            onClick={handleJumpTo}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-1 rounded text-sm"
          >
            Jump To
          </button>
        </div>
      </div>
    </div>
  )
}