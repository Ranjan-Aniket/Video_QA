interface ExportButtonProps {
  onExport: (format: 'excel' | 'csv' | 'json') => void
  label?: string
}

export default function ExportButton({ onExport, label = 'Export' }: ExportButtonProps) {
  return (
    <div className="relative group">
      <button className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg">
        {label}
      </button>
      <div className="absolute right-0 mt-2 w-32 bg-white rounded-lg shadow-lg hidden group-hover:block z-10">
        <button
          onClick={() => onExport('excel')}
          className="block w-full text-left px-4 py-2 hover:bg-gray-100 rounded-t-lg"
        >
          Excel
        </button>
        <button
          onClick={() => onExport('csv')}
          className="block w-full text-left px-4 py-2 hover:bg-gray-100"
        >
          CSV
        </button>
        <button
          onClick={() => onExport('json')}
          className="block w-full text-left px-4 py-2 hover:bg-gray-100 rounded-b-lg"
        >
          JSON
        </button>
      </div>
    </div>
  )
}