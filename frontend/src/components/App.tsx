import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'sonner'
import Layout from './Layout'
import Dashboard from '../pages/Dashboard'
import BatchUpload from '../pages/BatchUpload'
import VideoUpload from '../pages/VideoUpload'
import ProcessingMonitor from '../pages/ProcessingMonitor'
import VideoResults from '../pages/VideoResults'
import Analytics from '../pages/Analytics'
import Settings from '../pages/Settings'
import BatchDetails from '../pages/BatchDetails'
import EvidenceReview from '../pages/EvidenceReview'
import EvidenceReviewItem from '../pages/EvidenceReviewItem'
import ReviewerDashboard from '../pages/ReviewerDashboard'
import SmartPipeline from '../pages/SmartPipeline'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<SmartPipeline />} />
            <Route path="/upload/video" element={<VideoUpload />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </Layout>
      </BrowserRouter>
      <Toaster position="top-right" />
    </QueryClientProvider>
  )
}

export default App