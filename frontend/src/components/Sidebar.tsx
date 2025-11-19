import { Link, useLocation } from 'react-router-dom'

export default function Sidebar() {
  const location = useLocation()

  const links = [
    { path: '/', label: 'Smart Pipeline', icon: '🚀' },
    { path: '/upload/video', label: 'Upload Video', icon: '🎥' },
    { path: '/analytics', label: 'Analytics', icon: '📈' },
    { path: '/settings', label: 'Settings', icon: '⚙️' },
  ]

  return (
    <aside className="w-64 bg-gray-900 text-white p-6 space-y-2">
      {links.map(link => (
        <Link
          key={link.path}
          to={link.path}
          className={`block px-4 py-3 rounded-lg hover:bg-gray-800 transition-colors ${
            location.pathname === link.path ? 'bg-gray-800' : ''
          }`}
        >
          <span className="mr-3">{link.icon}</span>
          {link.label}
        </Link>
      ))}
    </aside>
  )
}