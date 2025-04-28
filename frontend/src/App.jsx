import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import PredictionPage from './pages/PredictionPage'
import BacktestPage from './pages/BacktestPage'
import ModelComparisonPage from './pages/ModelComparisonPage'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <Router>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <Navbar onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
        <div className="flex">
          <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
          <main className="flex-1 p-4 sm:p-6 md:p-8 max-w-full overflow-x-hidden">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/prediction" element={<PredictionPage />} />
              <Route path="/backtest" element={<BacktestPage />} />
              <Route path="/model-comparison" element={<ModelComparisonPage />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  )
}

export default App
