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


    // ===== Models =====

export const fetchWhitelist = () =>
  api.get('/api/models/whitelist').then((r) => r.data)

export const fetchBaseModels = () =>
  api.get('/api/models/base').then((r) => r.data)

export const fetchBaseModel = (id) =>
  api.get(`/api/models/base/${id}`).then((r) => r.data)

export const deleteBaseModel = (id, removeFiles = true) =>
  api
    .delete(`/api/models/base/${id}`, { params: { remove_files: removeFiles } })
    .then((r) => r.data)

export const startModelDownload = (hfRepo, token = null) =>
  api
    .post('/api/models/base/download', { hf_repo: hfRepo, token })
    .then((r) => r.data)

export const validateHfRepo = (hfRepo, token = null) =>
  api
    .post('/api/models/validate-repo', { hf_repo: hfRepo, token })
    .then((r) => r.data)

export const fetchJobs = (kind = null) =>
  api
    .get('/api/models/jobs', { params: kind ? { kind } : undefined })
    .then((r) => r.data)

export const fetchJob = (jobId) =>
  api.get(`/api/models/jobs/${jobId}`).then((r) => r.data)

export const cancelJob = (jobId) =>
  api.delete(`/api/models/jobs/${jobId}`).then((r) => r.data)