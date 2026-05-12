import { BrowserRouter, Routes, Route } from 'react-router-dom'
import AppLayout from './layout/AppLayout'
import Dashboard from './pages/Dashboard'
import Dataset from './pages/Dataset'
import Training from './pages/Training'
import TrainingLive from './pages/TrainingLive'
import Monitor from './pages/Monitor'
import Models from './pages/Models'
import Inference from './pages/Inference'
import Export from './pages/Export'
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dataset" element={<Dataset />} />
          <Route path="/training" element={<Training />} />
          <Route path="/training/live/:runId" element={<TrainingLive />} />
          <Route path="/monitor" element={<Monitor />} />
          <Route path="/models" element={<Models />} />
          <Route path="/inference" element={<Inference />} />
          <Route path="/export" element={<Export />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}