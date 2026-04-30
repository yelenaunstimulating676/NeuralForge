/**
 * Dashboard — landing page.
 * Mostra stato backend, sistema, GPU rilevate (con VRAM live)
 * e configurazione di training suggerita.
 */

import { useEffect, useState } from 'react'
import { Loader2, AlertTriangle } from 'lucide-react'
import {
  fetchHealth,
  fetchSystemInfo,
  fetchTrainingSuggestion,
} from '../api/client'
import SystemCard from '../components/SystemCard'
import GPUCard from '../components/GPUCard'
import ConfigSuggestion from '../components/ConfigSuggestion'

export default function Dashboard() {
  const [health, setHealth] = useState(null)
  const [systemInfo, setSystemInfo] = useState(null)
  const [config, setConfig] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      try {
        const [h, s] = await Promise.all([fetchHealth(), fetchSystemInfo()])
        if (cancelled) return
        setHealth(h)
        setSystemInfo(s)

        // Config suggerita solo se c'è almeno una GPU
        if (s.gpu_count > 0) {
          try {
            const cfg = await fetchTrainingSuggestion(0, 16)
            if (!cancelled) setConfig(cfg)
          } catch (e) {
            console.warn('Config suggestion failed:', e.message)
          }
        }
      } catch (err) {
        if (!cancelled) setError(err.message ?? 'Errore di caricamento')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [])

  // ===== Loading =====
  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="flex items-center gap-2 text-[var(--color-text-muted)]">
          <Loader2 size={18} className="animate-spin" />
          <span className="text-sm">Caricamento sistema…</span>
        </div>
      </div>
    )
  }

  // ===== Error =====
  if (error) {
    return (
      <div className="space-y-4">
        <header>
          <h1 className="text-2xl font-semibold text-[var(--color-text)]">
            Dashboard
          </h1>
        </header>
        <div className="flex items-start gap-3 rounded-lg border border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 p-4">
          <AlertTriangle
            size={18}
            className="mt-0.5 shrink-0 text-[var(--color-danger)]"
          />
          <div>
            <p className="text-sm font-medium text-[var(--color-danger)]">
              Errore di connessione al backend
            </p>
            <p className="mt-1 text-xs text-[var(--color-text-muted)]">
              {error}
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold text-[var(--color-text)]">
          Dashboard
        </h1>
        <p className="mt-1 text-sm text-[var(--color-text-muted)]">
          Stato del sistema NeuralForge · Backend v{health?.version}
        </p>
      </header>

      {/* Sistema + GPU */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {systemInfo && (
          <div className="lg:col-span-1">
            <SystemCard info={systemInfo} />
          </div>
        )}

        {systemInfo && systemInfo.gpus.length > 0 && (
          <div className="lg:col-span-2 grid grid-cols-1 gap-4 md:grid-cols-1">
            {systemInfo.gpus.map((g) => (
              <GPUCard key={g.index} gpu={g} />
            ))}
          </div>
        )}

        {systemInfo && systemInfo.gpus.length === 0 && (
          <div className="lg:col-span-2 flex items-center justify-center rounded-lg border border-dashed border-[var(--color-border)] bg-[var(--color-surface)] p-8">
            <div className="flex flex-col items-center gap-2 text-center">
              <AlertTriangle
                size={24}
                className="text-[var(--color-warning)]"
              />
              <p className="text-sm font-medium text-[var(--color-text)]">
                Nessuna GPU NVIDIA rilevata
              </p>
              <p className="text-xs text-[var(--color-text-muted)]">
                NeuralForge richiede una GPU CUDA per il training.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Config suggerita */}
      {config && <ConfigSuggestion config={config} />}
    </div>
  )
}