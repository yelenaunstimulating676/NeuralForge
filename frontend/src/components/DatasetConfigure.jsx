/**
 * Step 2 del wizard: configura parametri di chunking e conversione,
 * vedi preview live degli esempi generati.
 *
 * Flow:
 *   1. Al mount → chiama analyze (extract + detect)
 *   2. Al cambio config (debounced 500ms) → chiama preview
 *   3. Click "Avanti" → passa al parent {contentType, chunkerConfig, converterConfig}
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import {
  ArrowLeft,
  ArrowRight,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  Info,
} from 'lucide-react'
import { analyzeUpload, previewUpload } from '../api/client'
import { formatPercent, formatInt } from '../utils/format'
import Slider from './Slider'
import DatasetExampleCard from './DatasetExampleCard'

const CONTENT_TYPES = [
  { value: 'qa_pairs', label: 'Q&A pairs', desc: 'Domande e risposte estratte' },
  { value: 'narrative', label: 'Narrative', desc: 'Prosa, libri, articoli' },
  { value: 'code', label: 'Code', desc: 'Codice sorgente' },
  { value: 'dialogue', label: 'Dialogue', desc: 'Conversazioni multi-turn' },
  { value: 'tabular', label: 'Tabular', desc: 'Dati strutturati (CSV)' },
  { value: 'mixed', label: 'Mixed', desc: 'Contenuto misto' },
]

export default function DatasetConfigure({ uploadInfo, onBack, onNext }) {
  // Detection
  const [analyzing, setAnalyzing] = useState(true)
  const [analysis, setAnalysis] = useState(null)
  const [analysisError, setAnalysisError] = useState(null)

  // Config
  const [contentTypeOverride, setContentTypeOverride] = useState(null)
  const [targetChars, setTargetChars] = useState(2048)
  const [overlapChars, setOverlapChars] = useState(200)
  const [minChunkChars, setMinChunkChars] = useState(200)
  const [examplesPerChunk, setExamplesPerChunk] = useState(1)
  const [language, setLanguage] = useState('it')

  // Preview
  const [previewing, setPreviewing] = useState(false)
  const [preview, setPreview] = useState(null)
  const [previewError, setPreviewError] = useState(null)
  const debounceRef = useRef(null)

  // ===== Initial analyze =====
  useEffect(() => {
    let cancelled = false
    setAnalyzing(true)
    analyzeUpload(uploadInfo.upload_id)
      .then((data) => {
        if (cancelled) return
        setAnalysis(data)
        setAnalysisError(null)
      })
      .catch((err) => {
        if (cancelled) return
        setAnalysisError(err.response?.data?.detail ?? err.message)
      })
      .finally(() => {
        if (!cancelled) setAnalyzing(false)
      })
    return () => {
      cancelled = true
    }
  }, [uploadInfo.upload_id])

  // ===== Preview debounced =====
  const fetchPreview = useCallback(async () => {
    setPreviewing(true)
    try {
      const body = {
        content_type_override: contentTypeOverride,
        chunker_config: {
          target_chars: targetChars,
          overlap_chars: overlapChars,
          min_chunk_chars: minChunkChars,
        },
        converter_config: {
          examples_per_narrative_chunk: examplesPerChunk,
          template_language: language,
        },
        max_examples: 5,
      }
      const result = await previewUpload(uploadInfo.upload_id, body)
      setPreview(result)
      setPreviewError(null)
    } catch (err) {
      setPreviewError(err.response?.data?.detail ?? err.message)
    } finally {
      setPreviewing(false)
    }
  }, [
    uploadInfo.upload_id,
    contentTypeOverride,
    targetChars,
    overlapChars,
    minChunkChars,
    examplesPerChunk,
    language,
  ])

  // Trigger preview alla prima analyze E ad ogni cambio config (debounced)
  useEffect(() => {
    if (!analysis) return
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      fetchPreview()
    }, 500)
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [analysis, fetchPreview])

  // ===== Render =====

  if (analyzing) {
    return (
      <div className="flex flex-col items-center justify-center py-16">
        <Loader2 size={28} className="text-[var(--color-accent)] animate-spin" />
        <p className="mt-4 text-sm text-[var(--color-text-muted)]">
          Analisi del file in corso…
        </p>
      </div>
    )
  }

  if (analysisError) {
    return (
      <div className="space-y-4">
        <div className="flex items-start gap-3 rounded-lg border border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 p-4">
          <AlertTriangle size={18} className="mt-0.5 shrink-0 text-[var(--color-danger)]" />
          <div>
            <p className="text-sm font-medium text-[var(--color-danger)]">
              Errore analisi
            </p>
            <p className="mt-1 text-xs text-[var(--color-text-muted)]">
              {analysisError}
            </p>
          </div>
        </div>
        <button onClick={onBack} className="text-xs text-[var(--color-accent)] hover:underline">
          ← Torna allo Step 1
        </button>
      </div>
    )
  }

  const detected = analysis.detection
  const extracted = analysis.extracted
  const effectiveType = contentTypeOverride ?? detected.content_type

  return (
    <div className="space-y-6">
      {/* Detection panel */}
      <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
        <div className="flex items-start gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-[var(--color-success)]/15">
            <CheckCircle2 size={18} className="text-[var(--color-success)]" />
          </div>
          <div className="flex-1">
            <h3 className="text-sm font-medium text-[var(--color-text)]">
              File analizzato
            </h3>
            <p className="mt-0.5 text-xs text-[var(--color-text-muted)]">
              {extracted.source_format.toUpperCase()} · {formatInt(extracted.char_count)} caratteri ·{' '}
              {extracted.section_count} sezion{extracted.section_count === 1 ? 'e' : 'i'}
            </p>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <p className="text-[10px] font-medium uppercase tracking-wide text-[var(--color-text-muted)]">
              Tipo rilevato
            </p>
            <p className="mt-1 font-mono text-sm text-[var(--color-accent)]">
              {detected.content_type}
            </p>
            <p className="mt-1 text-[10px] text-[var(--color-text-muted)]">
              Confidence {formatPercent(detected.confidence * 100)}
            </p>
          </div>

          <div>
            <p className="text-[10px] font-medium uppercase tracking-wide text-[var(--color-text-muted)]">
              Override manuale
            </p>
            <select
              value={contentTypeOverride ?? detected.content_type}
              onChange={(e) => {
                const v = e.target.value
                setContentTypeOverride(v === detected.content_type ? null : v)
              }}
              className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-bg)] px-2 py-1.5 text-xs text-[var(--color-text)] focus:border-[var(--color-accent)] focus:outline-none"
            >
              {CONTENT_TYPES.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label} — {t.desc}
                </option>
              ))}
            </select>
          </div>
        </div>

        {detected.indicators.length > 0 && (
          <div className="mt-4 flex items-start gap-2 rounded-md bg-[var(--color-surface-2)]/50 p-3">
            <Info size={12} className="mt-0.5 shrink-0 text-[var(--color-text-muted)]" />
            <ul className="space-y-1 text-[11px] text-[var(--color-text-muted)]">
              {detected.indicators.map((ind, i) => (
                <li key={i}>{ind}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Config + Preview */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_1.3fr]">
        {/* Config */}
        <div className="space-y-5 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
          <h3 className="text-sm font-medium text-[var(--color-text)]">Parametri</h3>

          <Slider
            label="Dimensione chunk"
            value={targetChars}
            onChange={setTargetChars}
            min={500}
            max={8000}
            step={100}
            unit=" chars"
            hint={`≈ ${Math.round(targetChars / 4)} token`}
          />
          <Slider
            label="Overlap tra chunk"
            value={overlapChars}
            onChange={setOverlapChars}
            min={0}
            max={Math.min(2000, targetChars - 100)}
            step={50}
            unit=" chars"
            hint="Sovrapposizione per non perdere contesto al bordo"
          />
          <Slider
            label="Chunk minimo"
            value={minChunkChars}
            onChange={setMinChunkChars}
            min={50}
            max={Math.min(2000, targetChars)}
            step={50}
            unit=" chars"
            hint="Chunk più piccoli vengono scartati"
          />

          {(effectiveType === 'narrative' || effectiveType === 'mixed') && (
            <>
              <Slider
                label="Esempi per chunk"
                value={examplesPerChunk}
                onChange={setExamplesPerChunk}
                min={1}
                max={3}
                hint="Numero di varianti generate per ogni chunk narrative"
              />

              <div>
                <label className="text-xs font-medium text-[var(--color-text)]">
                  Lingua dei template
                </label>
                <div className="mt-2 flex gap-2">
                  {['it', 'en'].map((lang) => (
                    <button
                      key={lang}
                      onClick={() => setLanguage(lang)}
                      className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                        language === lang
                          ? 'bg-[var(--color-accent)] text-white'
                          : 'bg-[var(--color-surface-2)] text-[var(--color-text-muted)] hover:text-[var(--color-text)]'
                      }`}
                    >
                      {lang.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>

        {/* Preview */}
        <div className="space-y-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-[var(--color-text)]">Preview esempi</h3>
            {previewing && (
              <Loader2 size={14} className="text-[var(--color-accent)] animate-spin" />
            )}
          </div>

          {preview && (
            <p className="text-xs text-[var(--color-text-muted)]">
              {formatInt(preview.total_examples_estimated)} esempi totali stimati ·{' '}
              {formatInt(preview.total_chunks)} chunk
            </p>
          )}

          {previewError && (
            <div className="flex items-start gap-2 rounded-md border border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10 p-3 text-xs">
              <AlertTriangle
                size={12}
                className="mt-0.5 shrink-0 text-[var(--color-danger)]"
              />
              <p className="text-[var(--color-danger)]">{previewError}</p>
            </div>
          )}

          {preview && preview.examples.length === 0 && !previewError && (
            <p className="rounded-md border border-dashed border-[var(--color-border)] bg-[var(--color-bg)] p-4 text-xs text-[var(--color-text-muted)]">
              Nessun esempio generato con questi parametri. Prova a ridurre la
              dimensione del chunk o cambia il tipo override.
            </p>
          )}

          {preview && preview.examples.length > 0 && (
            <div className="space-y-2">
              {preview.examples.map((ex, i) => (
                <DatasetExampleCard
                  key={i}
                  example={ex}
                  index={i}
                  defaultOpen={i === 0}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Footer actions */}
      <div className="flex items-center justify-between border-t border-[var(--color-border)] pt-4">
        <button
          onClick={onBack}
          className="inline-flex items-center gap-1.5 rounded-md border border-[var(--color-border)] px-3 py-2 text-xs font-medium text-[var(--color-text)] hover:bg-[var(--color-surface-2)]"
        >
          <ArrowLeft size={14} />
          Indietro
        </button>

        <button
          onClick={() =>
            onNext({
              contentTypeOverride,
              chunkerConfig: {
                target_chars: targetChars,
                overlap_chars: overlapChars,
                min_chunk_chars: minChunkChars,
              },
              converterConfig: {
                examples_per_narrative_chunk: examplesPerChunk,
                template_language: language,
              },
              previewSnapshot: preview,
            })
          }
          disabled={!preview || preview.examples.length === 0}
          className="inline-flex items-center gap-1.5 rounded-md bg-[var(--color-accent)] px-3 py-2 text-xs font-medium text-white hover:bg-[var(--color-accent-hover)] disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Avanti
          <ArrowRight size={14} />
        </button>
      </div>
    </div>
  )
}