/**
 * Axios client configurato per il backend NeuralForge.
 * Tutti i moduli che chiamano REST passano da qui.
 */

import axios from 'axios'

const baseURL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

export const api = axios.create({
  baseURL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error.response?.status
    const url = error.config?.url
    if (status) {
      console.error(`[API] ${status} ${url}:`, error.response?.data)
    } else {
      console.error(`[API] Network error on ${url}:`, error.message)
    }
    return Promise.reject(error)
  }
)

// ===== Health =====
export const fetchHealth = () => api.get('/api/health').then((r) => r.data)

// ===== System =====
export const fetchSystemInfo = () =>
  api.get('/api/system/info').then((r) => r.data)

export const fetchGpus = () =>
  api.get('/api/system/gpus').then((r) => r.data)

export const fetchVram = (index = 0) =>
  api.get('/api/system/vram', { params: { index } }).then((r) => r.data)

export const fetchTrainingSuggestion = (index = 0, targetEffectiveBatch = 16) =>
  api
    .get('/api/system/suggest', {
      params: { index, target_effective_batch: targetEffectiveBatch },
    })
    .then((r) => r.data)