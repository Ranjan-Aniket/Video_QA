import { Link } from 'react-router-dom'

export default function Navbar() {
  return (
    <nav className="bg-white border-b px-6 py-4 flex justify-between items-center">
      <div className="text-xl font-bold text-gray-800">Video Q&A Generator</div>
      
      <div className="flex items-center gap-4">
        <Link to="/upload">
          <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm">
            + New Batch
          </button>
        </Link>
        
        <Link to="/settings">
          <button className="text-gray-600 hover:text-gray-800">
            ⚙️
          </button>
        </Link>
        
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white text-sm">
            U
          </div>
          <span className="text-sm text-gray-700">User</span>
        </div>
      </div>
    </nav>
  )
}