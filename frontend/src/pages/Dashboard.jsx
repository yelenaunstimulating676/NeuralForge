/**
 * Dashboard — landing page.
 * In M0 mostra solo il check di connessione al backend.
 * In M1 verrà arricchita con info GPU + VRAM live.
 */

import { useEffect, useState } from 'react'
import { CheckCircle2, XCircle, Loader2 } from 'lucide-react'
import { fetchHealth } from '../api/client'

export default function Dashboard() {
  const [health, setHealth] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchHealth()
      .then((data) => {
        setHealth(data)
        setError(null)
      })
      .catch((err) => {
        setError(err.message ?? 'Errore sconosciuto')
      })
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold text-[var(--color-text)]">
          Dashboard
        </h1>
        <p className="mt-1 text-sm text-[var(--color-text-muted)]">
          Stato del sistema NeuralForge
        </p>
      </header>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {/* Card: Backend health */}
        <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-[var(--color-text-muted)]">
              Backend
            </h3>
            {loading && (
              <Loader2 size={16} className="animate-spin text-[var(--color-text-muted)]" />
            )}
            {!loading && !error && (
              <CheckCircle2 size={16} className="text-[var(--color-success)]" />
            )}
            {!loading && error && (
              <XCircle size={16} className="text-[var(--color-danger)]" />
            )}
          </div>
          <div className="mt-3">
            {loading && (
              <p className="text-sm text-[var(--color-text-muted)]">Connecting…</p>
            )}
            {!loading && health && (
              <>
                <p className="text-2xl font-semibold text-[var(--color-success)]">
                  Online
                </p>
                <p className="mt-1 text-xs text-[var(--color-text-muted)]">
                  Version {health.version} · {new Date(health.timestamp).toLocaleString('it-IT')}
                </p>
              </>
            )}
            {!loading && error && (
              <>
                <p className="text-2xl font-semibold text-[var(--color-danger)]">
                  Offline
                </p>
                <p className="mt-1 text-xs text-[var(--color-text-muted)]">
                  {error}
                </p>
              </>
            )}
          </div>
        </div>

        {/* Card: GPU placeholder (M1) */}
        <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5 opacity-60">
          <h3 className="text-sm font-medium text-[var(--color-text-muted)]">
            GPU
          </h3>
          <p className="mt-3 text-sm text-[var(--color-text-muted)] italic">
            Disponibile in M1 — System Detector
          </p>
        </div>

        {/* Card: Datasets placeholder */}
        <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5 opacity-60">
          <h3 className="text-sm font-medium text-[var(--color-text-muted)]">
            Datasets
          </h3>
          <p className="mt-3 text-sm text-[var(--color-text-muted)] italic">
            Disponibile in M3 — Dataset Engine
          </p>
        </div>
      </div>
    </div>
  )
}