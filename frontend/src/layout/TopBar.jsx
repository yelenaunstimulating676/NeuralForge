/**
 * Barra superiore con stato connessione backend.
 * Polling /api/health ogni 5 secondi.
 */

import { useEffect, useState } from 'react'
import { CheckCircle2, XCircle, Loader2 } from 'lucide-react'
import { fetchHealth } from '../api/client'

export default function TopBar() {
  const [status, setStatus] = useState('loading') // loading | ok | error
  const [version, setVersion] = useState(null)

  useEffect(() => {
    let cancelled = false

    const check = async () => {
      try {
        const data = await fetchHealth()
        if (cancelled) return
        setStatus('ok')
        setVersion(data.version)
      } catch {
        if (cancelled) return
        setStatus('error')
        setVersion(null)
      }
    }

    check()
    const interval = setInterval(check, 5000)

    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  return (
    <div className="flex h-14 items-center justify-end border-b border-[var(--color-border)] bg-[var(--color-surface)] px-6">
      <div className="flex items-center gap-2 text-sm">
        {status === 'loading' && (
          <>
            <Loader2 size={16} className="animate-spin text-[var(--color-text-muted)]" />
            <span className="text-[var(--color-text-muted)]">Connecting…</span>
          </>
        )}
        {status === 'ok' && (
          <>
            <CheckCircle2 size={16} className="text-[var(--color-success)]" />
            <span className="text-[var(--color-text-muted)]">
              Backend connected
              {version && <span className="ml-1 text-[var(--color-text)]">v{version}</span>}
            </span>
          </>
        )}
        {status === 'error' && (
          <>
            <XCircle size={16} className="text-[var(--color-danger)]" />
            <span className="text-[var(--color-danger)]">Backend offline</span>
          </>
        )}
      </div>
    </div>
  )
}