interface Log {
  timestamp: string
  level: 'info' | 'error' | 'warning'
  message: string
}

interface LogsViewerProps {
  logs: Log[]
}

export default function LogsViewer({ logs }: LogsViewerProps) {
  const levelColors = {
    info: 'text-blue-600',
    warning: 'text-yellow-600',
    error: 'text-red-600',
  }
  
  return (
    <div className="bg-gray-900 text-white p-4 rounded-lg h-64 overflow-y-auto font-mono text-xs">
      {logs?.map((log, i) => (
        <div key={i} className="mb-1">
          <span className="text-gray-400">[{log.timestamp}]</span>
          <span className={`ml-2 ${levelColors[log.level]}`}>{log.level.toUpperCase()}</span>
          <span className="ml-2">{log.message}</span>
        </div>
      ))}
    </div>
  )
}