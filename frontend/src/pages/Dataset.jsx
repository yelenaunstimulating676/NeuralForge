/**
 * Pagina Dataset: wizard di creazione + lista dataset salvati.
 *
 * State machine:
 *   'home'      → lista dataset + bottone Nuovo
 *   'wizard-1'  → upload file
 *   'wizard-2'  → configure (detection + chunker + preview live)
 *   'wizard-3'  → validate & save
 */

import { useEffect, useState, useCallback } from 'react'
import { Plus, ArrowLeft, Loader2, Inbox, AlertTriangle } from 'lucide-react'
import {
  fetchDatasets,
  deleteDataset as apiDeleteDataset,
} from '../api/client'
import DatasetUploader from '../components/DatasetUploader'
import DatasetConfigure from '../components/DatasetConfigure'
import DatasetReview from '../components/DatasetReview'
import DatasetListCard from '../components/DatasetListCard'
import useDocumentTitle from "../hooks/useDocumentTitle";

export default function Dataset() {
  useDocumentTitle("Dataset");
  const [view, setView] = useState('home')
  const [datasets, setDatasets] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Stato wizard (passato tra step)
  const [uploadInfo, setUploadInfo] = useState(null)
  const [configFromStep2, setConfigFromStep2] = useState(null)

  const loadDatasets = useCallback(async () => {
    try {
      const list = await fetchDatasets()
      setDatasets(list)
      setError(null)
    } catch (err) {
      setError(err.response?.data?.detail ?? err.message)
    }
  }, [])

  useEffect(() => {
    setLoading(true)
    loadDatasets().finally(() => setLoading(false))
  }, [loadDatasets])

  const resetWizard = () => {
    setUploadInfo(null)
    setConfigFromStep2(null)
    setView('home')
  }

  const handleUploaded = (info) => {
    setUploadInfo(info)
    setView('wizard-2')
  }

  const handleConfigured = (config) => {
    setConfigFromStep2(config)
    setView('wizard-3')
  }

  const handleSaved = async (dataset) => {
    await loadDatasets()
    resetWizard()
  }

  const handleDelete = async (dataset) => {
    const ok = window.confirm(
      `Cancellare "${dataset.name}"?\n\nVerranno rimossi anche i file su disco.`
    )
    if (!ok) return
    try {
      await apiDeleteDataset(dataset.id, true)
      await loadDatasets()
    } catch (err) {
      alert(`Errore: ${err.response?.data?.detail ?? err.message}`)
    }
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="flex items-center gap-2 text-[var(--color-text-muted)]">
          <Loader2 size={18} className="animate-spin" />
          <span className="text-sm">Caricamento dataset…</span>
        </div>
      </div>
    )
  }

  // ===== WIZARD =====
  if (view.startsWith('wizard-')) {
    const stepNum = view.split('-')[1]

    return (
      <div className="space-y-6">
        <header>
          <button
            onClick={resetWizard}
            className="mb-2 inline-flex items-center gap-1 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
          >
            <ArrowLeft size={12} />
            Esci dal wizard
          </button>
          <h1 className="text-2xl font-semibold text-[var(--color-text)]">
            Nuovo dataset · Step {stepNum} di 3
          </h1>
          <p className="mt-1 text-sm text-[var(--color-text-muted)]">
            {stepNum === '1' && 'Carica il file da convertire.'}
            {stepNum === '2' && 'Configura i parametri di chunking e conversione.'}
            {stepNum === '3' && 'Applica i filtri di validazione e salva.'}
          </p>
        </header>

        <div>
          {view === 'wizard-1' && <DatasetUploader onUploaded={handleUploaded} />}

          {view === 'wizard-2' && uploadInfo && (
            <DatasetConfigure
              uploadInfo={uploadInfo}
              onBack={() => setView('wizard-1')}
              onNext={handleConfigured}
            />
          )}

          {view === 'wizard-3' && uploadInfo && configFromStep2 && (
            <DatasetReview
              uploadInfo={uploadInfo}
              configFromStep2={configFromStep2}
              onBack={() => setView('wizard-2')}
              onSaved={handleSaved}
            />
          )}
        </div>
      </div>
    )
  }

  // ===== HOME =====
  return (
    <div className="space-y-6">
      <header className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-[var(--color-text)]">
            Dataset
          </h1>
          <p className="mt-1 text-sm text-[var(--color-text-muted)]">
            Trasforma documenti raw in dataset di instruction tuning.
          </p>
        </div>

        <button
          onClick={() => setView('wizard-1')}
          className="inline-flex items-center gap-1.5 rounded-md bg-[var(--color-accent)] px-3 py-2 text-xs font-medium text-white transition-colors hover:bg-[var(--color-accent-hover)]"
        >
          <Plus size={14} />
          Nuovo dataset
        </button>
      </header>

      {error && (
        <div className="flex items-start gap-3 rounded-lg border border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 p-4">
          <AlertTriangle
            size={18}
            className="mt-0.5 shrink-0 text-[var(--color-danger)]"
          />
          <div>
            <p className="text-sm font-medium text-[var(--color-danger)]">Errore</p>
            <p className="mt-1 text-xs text-[var(--color-text-muted)]">{error}</p>
          </div>
        </div>
      )}

      {datasets.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-[var(--color-border)] bg-[var(--color-surface)] py-16 px-6 text-center">
          <Inbox size={36} className="text-[var(--color-text-muted)]" />
          <p className="mt-4 text-sm text-[var(--color-text-muted)]">
            Nessun dataset creato.
          </p>
          <p className="mt-1 text-xs text-[var(--color-text-muted)]/70 mb-4">
            Carica un file per iniziare il fine-tuning.
          </p>
          <button
            onClick={() => setView('wizard-1')}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm rounded-lg inline-flex items-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Crea il primo dataset
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {datasets.map((d) => (
            <DatasetListCard
              key={d.id}
              dataset={d}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}
    </div>
  )
}