/**
 * Step 3 del wizard: applica filtri di validazione, mostra stats finali,
 * permette di salvare il dataset con un nome.
 *
 * Per le stats useremo l'output del save endpoint (che fa anche validate).
 * Per evitare 2 chiamate, in questo step facciamo "save in dry-run":
 *   1. User imposta nome + validator config
 *   2. Click "Salva" → chiamata save → ritorna stats reali
 *
 * (Non implementiamo dry-run server-side per semplicità: in M3.6 il
 *  pulsante "Salva" è il commit definitivo. Per stats preview-only,
 *  in futuro aggiungiamo /api/dataset/upload/{id}/validate.)
 */

import { useState } from 'react'
import {
  ArrowLeft,
  Save,
  Loader2,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react'
import { saveUploadAsDataset } from '../api/client'

export default function DatasetReview({ uploadInfo, configFromStep2, onBack, onSaved }) {
  const [name, setName] = useState('')
  const [enableFuzzyDedup, setEnableFuzzyDedup] = useState(false)
  const [minOutputChars, setMinOutputChars] = useState(20)
  const [maxOutputChars, setMaxOutputChars] = useState(8192)

  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)

  const handleSave = async () => {
    if (!name.trim()) {
      setError('Inserisci un nome per il dataset.')
      return
    }
    setSaving(true)
    setError(null)
    try {
      const body = {
        name: name.trim(),
        content_type_override: configFromStep2.contentTypeOverride,
        chunker_config: configFromStep2.chunkerConfig,
        converter_config: configFromStep2.converterConfig,
        validator_config: {
          min_output_chars: minOutputChars,
          max_output_chars: maxOutputChars,
          enable_fuzzy_dedup: enableFuzzyDedup,
        },
      }
      const result = await saveUploadAsDataset(uploadInfo.upload_id, body)
      onSaved(result.dataset)
    } catch (err) {
      setError(err.response?.data?.detail ?? err.message)
    } finally {
      setSaving(false)
    }
  }

  // Snapshot dalla preview di Step 2 (è solo informativa)
  const preview = configFromStep2.previewSnapshot

  return (
    <div className="space-y-6">
      {/* Riepilogo da Step 2 */}
      {preview && (
        <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
          <div className="flex items-start gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-[var(--color-accent)]/15">
              <CheckCircle2 size={18} className="text-[var(--color-accent)]" />
            </div>
            <div>
              <h3 className="text-sm font-medium text-[var(--color-text)]">
                Pronto per il salvataggio
              </h3>
              <p className="mt-0.5 text-xs text-[var(--color-text-muted)]">
                Stima:{' '}
                <span className="font-mono text-[var(--color-text)]">
                  {preview.total_examples_estimated}
                </span>{' '}
                esempi · tipo{' '}
                <span className="font-mono text-[var(--color-text)]">
                  {preview.content_type}
                </span>
              </p>
            </div>
          </div>
          <p className="mt-3 text-[11px] text-[var(--color-text-muted)]">
            Il numero finale potrà essere inferiore: il Validator filtrerà esempi
            duplicati o troppo corti/lunghi secondo le regole sotto.
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1.2fr_1fr]">
        {/* Validator config */}
        <div className="space-y-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
          <h3 className="text-sm font-medium text-[var(--color-text)]">
            Filtri di validazione
          </h3>

          <div>
            <label className="text-xs font-medium text-[var(--color-text)]">
              Output minimo: <span className="font-mono text-[var(--color-text-muted)]">{minOutputChars} caratteri</span>
            </label>
            <input
              type="range"
              min={10}
              max={200}
              step={5}
              value={minOutputChars}
              onChange={(e) => setMinOutputChars(Number(e.target.value))}
              className="mt-2 w-full accent-[var(--color-accent)]"
            />
            <p className="mt-1 text-[10px] text-[var(--color-text-muted)]/80">
              Esempi con output più corto vengono scartati.
            </p>
          </div>

          <div>
            <label className="text-xs font-medium text-[var(--color-text)]">
              Output massimo: <span className="font-mono text-[var(--color-text-muted)]">{maxOutputChars} caratteri</span>
            </label>
            <input
              type="range"
              min={1000}
              max={32000}
              step={500}
              value={maxOutputChars}
              onChange={(e) => setMaxOutputChars(Number(e.target.value))}
              className="mt-2 w-full accent-[var(--color-accent)]"
            />
          </div>

          <label className="flex items-start gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={enableFuzzyDedup}
              onChange={(e) => setEnableFuzzyDedup(e.target.checked)}
              className="mt-0.5 accent-[var(--color-accent)]"
            />
            <div>
              <p className="text-xs font-medium text-[var(--color-text)]">
                Dedup fuzzy
              </p>
              <p className="text-[10px] text-[var(--color-text-muted)]/80">
                Rimuove anche esempi simili (non solo duplicati esatti). Più lento.
              </p>
            </div>
          </label>
        </div>

        {/* Save form */}
        <div className="space-y-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
          <h3 className="text-sm font-medium text-[var(--color-text)]">
            Salva dataset
          </h3>

          <div>
            <label className="mb-1 block text-xs font-medium text-[var(--color-text-muted)]">
              Nome dataset
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="es. Manuale Tecnico Q&A"
              className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm text-[var(--color-text)] placeholder-[var(--color-text-muted)]/60 focus:border-[var(--color-accent)] focus:outline-none"
            />
          </div>

          {error && (
            <div className="flex items-start gap-2 rounded-md border border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 p-3 text-xs">
              <AlertTriangle
                size={12}
                className="mt-0.5 shrink-0 text-[var(--color-danger)]"
              />
              <p className="text-[var(--color-danger)]">{error}</p>
            </div>
          )}

          <button
            onClick={handleSave}
            disabled={saving || !name.trim()}
            className="inline-flex w-full items-center justify-center gap-1.5 rounded-md bg-[var(--color-accent)] px-3 py-2.5 text-sm font-medium text-white hover:bg-[var(--color-accent-hover)] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {saving ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                Salvataggio in corso…
              </>
            ) : (
              <>
                <Save size={14} />
                Salva dataset
              </>
            )}
          </button>
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between border-t border-[var(--color-border)] pt-4">
        <button
          onClick={onBack}
          disabled={saving}
          className="inline-flex items-center gap-1.5 rounded-md border border-[var(--color-border)] px-3 py-2 text-xs font-medium text-[var(--color-text)] hover:bg-[var(--color-surface-2)] disabled:opacity-50"
        >
          <ArrowLeft size={14} />
          Indietro
        </button>
      </div>
    </div>
  )
}