/**
 * Form per scaricare un repo HF custom (fuori whitelist).
 * Include validazione preventiva via /validate-repo prima del download.
 *
 * Distingue 4 esiti di validazione:
 *   1. Errore (non esiste / malformato / network)         → box rosso
 *   2. Gated, manca token                                  → box arancione warning
 *   3. Gated MA token fornito                              → box verde
 *   4. Pubblico                                            → box verde
 */

import { useState } from 'react'
import {
  Search,
  Download,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
} from 'lucide-react'
import { validateHfRepo } from '../api/client'

export default function CustomRepoForm({ onDownload, downloading }) {
  const [hfRepo, setHfRepo] = useState('')
  const [token, setToken] = useState('')
  const [showToken, setShowToken] = useState(false)
  const [validating, setValidating] = useState(false)
  const [validation, setValidation] = useState(null)

  const handleValidate = async () => {
    if (!hfRepo.trim()) return
    setValidating(true)
    setValidation(null)
    try {
      const result = await validateHfRepo(hfRepo.trim(), token.trim() || null)
      setValidation(result)
    } catch (err) {
      setValidation({
        accessible: false,
        message: err.response?.data?.detail ?? err.message,
      })
    } finally {
      setValidating(false)
    }
  }

  const handleDownload = () => {
    if (!validation?.accessible || validation?.requires_token) return
    onDownload(hfRepo.trim(), token.trim() || null)
  }

  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
      <h3 className="text-sm font-medium text-[var(--color-text)]">
        Repository custom
      </h3>
      <p className="mt-1 text-xs text-[var(--color-text-muted)]">
        Scarica qualsiasi modello da HuggingFace. Per repo gated (es. Gemma,
        Llama) serve un token di accesso HF.
      </p>

      <div className="mt-4 space-y-3">
        <div>
          <label className="mb-1 block text-xs font-medium text-[var(--color-text-muted)]">
            HuggingFace repo
          </label>
          <input
            type="text"
            value={hfRepo}
            onChange={(e) => {
              setHfRepo(e.target.value)
              setValidation(null)
            }}
            placeholder="es. mistralai/Mistral-7B-Instruct-v0.3"
            className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm font-mono text-[var(--color-text)] placeholder-[var(--color-text-muted)]/60 focus:border-[var(--color-accent)] focus:outline-none"
          />
        </div>

        <div>
          <button
            type="button"
            onClick={() => setShowToken((s) => !s)}
            className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
          >
            {showToken ? '− Nascondi token HF' : '+ Token HF (per repo gated)'}
          </button>
          {showToken && (
            <input
              type="password"
              value={token}
              onChange={(e) => {
                setToken(e.target.value)
                setValidation(null)
              }}
              placeholder="hf_..."
              className="mt-2 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm font-mono text-[var(--color-text)] placeholder-[var(--color-text-muted)]/60 focus:border-[var(--color-accent)] focus:outline-none"
            />
          )}
        </div>

        <div className="flex gap-2">
          <button
            onClick={handleValidate}
            disabled={!hfRepo.trim() || validating}
            className="inline-flex items-center gap-1.5 rounded-md border border-[var(--color-border)] px-3 py-2 text-xs font-medium text-[var(--color-text)] transition-colors hover:bg-[var(--color-surface-2)] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {validating ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Search size={14} />
            )}
            Verifica
          </button>

          <button
            onClick={handleDownload}
            disabled={
              !validation?.accessible ||
              validation?.requires_token ||
              downloading
            }
            className="inline-flex items-center gap-1.5 rounded-md bg-[var(--color-accent)] px-3 py-2 text-xs font-medium text-white transition-colors hover:bg-[var(--color-accent-hover)] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Download size={14} />
            Scarica
          </button>
        </div>

        {validation && <ValidationResult validation={validation} />}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Validation result helper
// ---------------------------------------------------------------------------

function ValidationResult({ validation }) {
  // Caso 1: errore (non esiste, malformato, network)
  if (!validation.accessible) {
    return (
      <div className="flex items-start gap-2 rounded-md border border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 p-3 text-xs">
        <XCircle
          size={14}
          className="mt-0.5 shrink-0 text-[var(--color-danger)]"
        />
        <div className="min-w-0">
          <p className="text-[var(--color-danger)]">{validation.message}</p>
        </div>
      </div>
    )
  }

  // Caso 2: gated, serve token (non fornito)
  if (validation.requires_token) {
    return (
      <div className="flex items-start gap-2 rounded-md border border-[var(--color-warning)]/40 bg-[var(--color-warning)]/10 p-3 text-xs">
        <AlertTriangle
          size={14}
          className="mt-0.5 shrink-0 text-[var(--color-warning)]"
        />
        <div className="min-w-0 space-y-1">
          <p className="font-medium text-[var(--color-warning)]">
            Repo gated · token HF richiesto
          </p>
          <p className="text-[var(--color-text-muted)]">
            {validation.message}
          </p>
        </div>
      </div>
    )
  }

  // Caso 3: gated MA con token fornito → ok
  if (validation.gated) {
    return (
      <div className="flex items-start gap-2 rounded-md border border-[var(--color-success)]/40 bg-[var(--color-success)]/10 p-3 text-xs">
        <CheckCircle2
          size={14}
          className="mt-0.5 shrink-0 text-[var(--color-success)]"
        />
        <div className="min-w-0">
          <p className="text-[var(--color-success)]">
            Repo accessibile (gated, token fornito) ·{' '}
            {validation.siblings_count} file
          </p>
        </div>
      </div>
    )
  }

  // Caso 4: pubblico, tutto ok
  return (
    <div className="flex items-start gap-2 rounded-md border border-[var(--color-success)]/40 bg-[var(--color-success)]/10 p-3 text-xs">
      <CheckCircle2
        size={14}
        className="mt-0.5 shrink-0 text-[var(--color-success)]"
      />
      <div className="min-w-0">
        <p className="text-[var(--color-success)]">
          Repo accessibile · {validation.siblings_count} file
        </p>
      </div>
    </div>
  )
}