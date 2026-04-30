/**
 * Singola entry della whitelist con bottone Download.
 */

import { Download, Check, Loader2 } from 'lucide-react'
import { formatFloat } from '../utils/format'

export default function WhitelistEntry({
  entry,
  alreadyDownloaded,
  downloading,
  onDownload,
}) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h4 className="font-medium text-[var(--color-text)]">
            {entry.display_name}
          </h4>
          <p className="truncate text-xs font-mono text-[var(--color-text-muted)]">
            {entry.hf_repo}
          </p>
          {entry.description && (
            <p className="mt-2 text-xs text-[var(--color-text-muted)]">
              {entry.description}
            </p>
          )}
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[var(--color-text-muted)]">
            <span>
              <span className="font-mono text-[var(--color-text)]">
                {entry.params_billions}B
              </span>{' '}
              params
            </span>
            <span>
              ~
              <span className="font-mono text-[var(--color-text)]">
                {formatFloat(entry.size_gb, 1)}
              </span>{' '}
              GB
            </span>
            <span className="rounded bg-[var(--color-surface-2)] px-1.5 py-0.5 font-mono text-[10px]">
              {entry.tag}
            </span>
          </div>
        </div>

        <div className="shrink-0">
          {alreadyDownloaded ? (
            <span className="inline-flex items-center gap-1.5 rounded-md bg-[var(--color-success)]/15 px-3 py-2 text-xs font-medium text-[var(--color-success)]">
              <Check size={14} />
              Scaricato
            </span>
          ) : downloading ? (
            <span className="inline-flex items-center gap-1.5 rounded-md bg-[var(--color-accent)]/15 px-3 py-2 text-xs font-medium text-[var(--color-accent)]">
              <Loader2 size={14} className="animate-spin" />
              In corso
            </span>
          ) : (
            <button
              onClick={() => onDownload(entry)}
              className="inline-flex items-center gap-1.5 rounded-md bg-[var(--color-accent)] px-3 py-2 text-xs font-medium text-white transition-colors hover:bg-[var(--color-accent-hover)]"
            >
              <Download size={14} />
              Download
            </button>
          )}
        </div>
      </div>
    </div>
  )
}