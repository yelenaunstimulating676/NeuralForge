import { BrowserRouter, Routes, Route } from 'react-router-dom'
import AppLayout from './layout/AppLayout'
import Dashboard from './pages/Dashboard'
import Dataset from './pages/Dataset'
import Training from './pages/Training'
import Monitor from './pages/Monitor'
import Models from './pages/Models'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dataset" element={<Dataset />} />
          <Route path="/training" element={<Training />} />
          <Route path="/monitor" element={<Monitor />} />
          <Route path="/models" element={<Models />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}