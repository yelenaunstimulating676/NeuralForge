/**
 * Step 1 del wizard: drag & drop + upload del file.
 * Notifica il parent con `onUploaded(uploadInfo)` quando l'upload finisce.
 */

import { useCallback, useRef, useState } from 'react'
import { Upload, FileUp, AlertTriangle, Loader2 } from 'lucide-react'
import { uploadDatasetFile } from '../api/client'
import { formatBytes } from '../utils/format'

const SUPPORTED_EXTS = ['.pdf', '.txt', '.md', '.csv', '.tsv', '.json', '.jsonl', '.docx']
const MAX_SIZE_MB = 100

export default function DatasetUploader({ onUploaded }) {
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const handleFile = useCallback(
    async (file) => {
      setError(null)

      // Client-side validation
      const ext = '.' + (file.name.split('.').pop() || '').toLowerCase()
      if (!SUPPORTED_EXTS.includes(ext)) {
        setError(
          `Estensione ${ext} non supportata. Formati ammessi: ${SUPPORTED_EXTS.join(', ')}`
        )
        return
      }
      if (file.size > MAX_SIZE_MB * 1024 * 1024) {
        setError(`File troppo grande (${formatBytes(file.size)}). Max ${MAX_SIZE_MB} MB.`)
        return
      }
      if (file.size === 0) {
        setError('File vuoto.')
        return
      }

      setUploading(true)
      try {
        const result = await uploadDatasetFile(file)
        onUploaded(result)
      } catch (err) {
        const msg = err.response?.data?.detail ?? err.message ?? 'Errore di upload'
        setError(msg)
      } finally {
        setUploading(false)
      }
    },
    [onUploaded]
  )

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  const handleSelect = (e) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
    // reset input per permettere re-upload stesso file
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed py-16 px-6 text-center cursor-pointer transition-colors ${
          dragOver
            ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/5'
            : 'border-[var(--color-border)] bg-[var(--color-surface)] hover:border-[var(--color-accent)]/40'
        }`}
      >
        {uploading ? (
          <>
            <Loader2
              size={36}
              className="text-[var(--color-accent)] animate-spin"
            />
            <p className="mt-4 text-sm text-[var(--color-text)]">
              Upload in corso…
            </p>
          </>
        ) : (
          <>
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-[var(--color-accent)]/15">
              <Upload size={26} className="text-[var(--color-accent)]" />
            </div>
            <p className="mt-4 text-sm font-medium text-[var(--color-text)]">
              Trascina qui un file o{' '}
              <span className="text-[var(--color-accent)] underline">
                seleziona dal disco
              </span>
            </p>
            <p className="mt-1 text-xs text-[var(--color-text-muted)]">
              Formati: PDF, TXT, MD, CSV, TSV, JSON, JSONL, DOCX · Max {MAX_SIZE_MB} MB
            </p>
          </>
        )}

        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept={SUPPORTED_EXTS.join(',')}
          onChange={handleSelect}
        />
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-md border border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 p-3 text-xs">
          <AlertTriangle
            size={14}
            className="mt-0.5 shrink-0 text-[var(--color-danger)]"
          />
          <p className="text-[var(--color-danger)]">{error}</p>
        </div>
      )}

      <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
        <h3 className="flex items-center gap-2 text-sm font-medium text-[var(--color-text)]">
          <FileUp size={14} className="text-[var(--color-accent)]" />
          Cosa succede dopo l'upload?
        </h3>
        <ol className="mt-2 ml-5 list-decimal space-y-1 text-xs text-[var(--color-text-muted)]">
          <li>NeuralForge analizza il file e rileva il tipo di contenuto</li>
          <li>Configuri parametri di chunking e conversione (con preview live)</li>
          <li>Salvi il dataset finale pronto per il training</li>
        </ol>
      </div>
    </div>
  )
}