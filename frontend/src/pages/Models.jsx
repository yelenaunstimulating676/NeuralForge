/**
 * Pagina Models: gestione modelli base scaricati e download da HF.
 *
 * Layout:
 *   - Tab "Whitelist" / "Custom" / "Locali (N)"
 *   - Sezione job attivi sempre visibile in alto se ci sono download in corso
 */

import { useEffect, useState, useCallback, useRef } from 'react'
import { Loader2, AlertTriangle, Inbox } from 'lucide-react'
import {
  fetchWhitelist,
  fetchBaseModels,
  fetchJobs,
  startModelDownload,
  cancelJob,
  deleteBaseModel,
} from '../api/client'
import WhitelistEntry from '../components/WhitelistEntry'
import ModelCard from '../components/ModelCard'
import CustomRepoForm from '../components/CustomRepoForm'
import DownloadJobCard from '../components/DownloadJobCard'

const POLL_JOBS_MS = 1500
const POLL_MODELS_MS = 5000

export default function Models() {
  const [tab, setTab] = useState('whitelist') // whitelist | custom | local

  const [whitelist, setWhitelist] = useState([])
  const [models, setModels] = useState([])
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Per evitare race condition tra polling e refresh manuale
  const pollingRef = useRef(false)

  const loadAll = useCallback(async () => {
    try {
      const [w, m, j] = await Promise.all([
        fetchWhitelist(),
        fetchBaseModels(),
        fetchJobs('download'),
      ])
      setWhitelist(w)
      setModels(m)
      setJobs(j)
      setError(null)
    } catch (err) {
      setError(err.message ?? 'Errore di caricamento')
    }
  }, [])

  // Initial load
  useEffect(() => {
    setLoading(true)
    loadAll().finally(() => setLoading(false))
  }, [loadAll])

  // Poll jobs (frequente) — aggiorna anche models quando un job completa
  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      if (pollingRef.current) return
      pollingRef.current = true
      try {
        const newJobs = await fetchJobs('download')
        if (cancelled) return

        // Se c'è un job appena passato a "completed", ricarichiamo i modelli
        const justCompleted = newJobs.some(
          (nj) =>
            nj.status === 'completed' &&
            jobs.some((oj) => oj.id === nj.id && oj.status !== 'completed')
        )
        setJobs(newJobs)

        if (justCompleted) {
          const newModels = await fetchBaseModels()
          if (!cancelled) setModels(newModels)
        }
      } catch (err) {
        // Non rumoreggiamo per errori transient
        console.warn('Poll jobs failed:', err.message)
      } finally {
        pollingRef.current = false
      }
    }

    const id = setInterval(poll, POLL_JOBS_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [jobs])

  // Poll più lento dei modelli (safety net)
  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const newModels = await fetchBaseModels()
        setModels(newModels)
      } catch {
        // ignore
      }
    }, POLL_MODELS_MS)
    return () => clearInterval(id)
  }, [])

  // ===== Handlers =====

  const handleDownload = async (entry) => {
    try {
      await startModelDownload(entry.hf_repo)
      await loadAll()
    } catch (err) {
      const msg = err.response?.data?.detail ?? err.message
      alert(`Errore avvio download: ${msg}`)
    }
  }

  const handleDownloadCustom = async (hfRepo, token) => {
    try {
      await startModelDownload(hfRepo, token)
      await loadAll()
    } catch (err) {
      const msg = err.response?.data?.detail ?? err.message
      alert(`Errore avvio download: ${msg}`)
    }
  }

  const handleCancelJob = async (jobId) => {
    try {
      await cancelJob(jobId)
    } catch (err) {
      console.warn('Cancel job failed:', err.message)
    }
  }

  const handleDeleteModel = async (model) => {
    const ok = window.confirm(
      `Cancellare ${model.display_name}?\n\nVerranno rimossi anche i file su disco.`
    )
    if (!ok) return
    try {
      await deleteBaseModel(model.id, true)
      const newModels = await fetchBaseModels()
      setModels(newModels)
    } catch (err) {
      const msg = err.response?.data?.detail ?? err.message
      alert(`Errore cancellazione: ${msg}`)
    }
  }

  // ===== Derived data =====

  const downloadedRepos = new Set(models.map((m) => m.hf_repo))
  const activeJobRepos = new Set(
    jobs
      .filter((j) => j.status === 'running' || j.status === 'pending')
      .map((j) => j.result?.hf_repo)
      .filter(Boolean)
  )
  const visibleJobs = jobs.filter(
    (j) =>
      j.status === 'running' ||
      j.status === 'pending' ||
      (j.status === 'failed' && j.finished_at) ||
      (j.status === 'cancelled' && j.finished_at)
  )

  // ===== Render =====

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="flex items-center gap-2 text-[var(--color-text-muted)]">
          <Loader2 size={18} className="animate-spin" />
          <span className="text-sm">Caricamento modelli…</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-4">
        <header>
          <h1 className="text-2xl font-semibold text-[var(--color-text)]">
            Models
          </h1>
        </header>
        <div className="flex items-start gap-3 rounded-lg border border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 p-4">
          <AlertTriangle
            size={18}
            className="mt-0.5 shrink-0 text-[var(--color-danger)]"
          />
          <div>
            <p className="text-sm font-medium text-[var(--color-danger)]">
              Errore
            </p>
            <p className="mt-1 text-xs text-[var(--color-text-muted)]">{error}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold text-[var(--color-text)]">
          Models
        </h1>
        <p className="mt-1 text-sm text-[var(--color-text-muted)]">
          Scarica e gestisci modelli base da HuggingFace.
        </p>
      </header>

      {/* Active jobs (sempre visibili se ce ne sono) */}
      {visibleJobs.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-sm font-medium text-[var(--color-text-muted)]">
            Download in corso ({visibleJobs.length})
          </h2>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {visibleJobs.map((j) => (
              <DownloadJobCard key={j.id} job={j} onCancel={handleCancelJob} />
            ))}
          </div>
        </section>
      )}

      {/* Tabs */}
      <div className="flex border-b border-[var(--color-border)]">
        <TabButton active={tab === 'whitelist'} onClick={() => setTab('whitelist')}>
          Raccomandati ({whitelist.length})
        </TabButton>
        <TabButton active={tab === 'custom'} onClick={() => setTab('custom')}>
          Custom HF
        </TabButton>
        <TabButton active={tab === 'local'} onClick={() => setTab('local')}>
          Locali ({models.length})
        </TabButton>
      </div>

      {/* Tab: Whitelist */}
      {tab === 'whitelist' && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {whitelist.map((entry) => (
            <WhitelistEntry
              key={entry.hf_repo}
              entry={entry}
              alreadyDownloaded={downloadedRepos.has(entry.hf_repo)}
              downloading={activeJobRepos.has(entry.hf_repo)}
              onDownload={handleDownload}
            />
          ))}
        </div>
      )}

      {/* Tab: Custom */}
      {tab === 'custom' && (
        <div className="max-w-2xl">
          <CustomRepoForm
            onDownload={handleDownloadCustom}
            downloading={visibleJobs.some(
              (j) => j.status === 'running' || j.status === 'pending'
            )}
          />
        </div>
      )}

      {/* Tab: Locali */}
      {tab === 'local' && (
        <>
          {models.length === 0 ? (
            <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-[var(--color-border)] bg-[var(--color-surface)] py-16 px-6 text-center">
              <Inbox size={36} className="text-[var(--color-text-muted)]" />
              <p className="mt-4 text-sm text-[var(--color-text-muted)]">
                Nessun modello scaricato.
              </p>
              <p className="mt-1 text-xs text-[var(--color-text-muted)]/70">
                Vai sui tab Raccomandati o Custom per iniziare.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {models.map((m) => (
                <ModelCard key={m.id} model={m} onDelete={handleDeleteModel} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab button helper
// ---------------------------------------------------------------------------

function TabButton({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className={`relative px-4 py-2.5 text-sm font-medium transition-colors ${
        active
          ? 'text-[var(--color-text)]'
          : 'text-[var(--color-text-muted)] hover:text-[var(--color-text)]'
      }`}
    >
      {children}
      {active && (
        <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-[var(--color-accent)]" />
      )}
    </button>
  )
}