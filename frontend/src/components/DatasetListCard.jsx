/**
 * Card di un dataset salvato (per la lista home Dataset).
 */

import { Database, Trash2, FileText } from 'lucide-react'
import { formatDateTime, formatInt } from '../utils/format'

export default function DatasetListCard({ dataset, onDelete, onView }) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4 transition-colors hover:border-[var(--color-accent)]/40">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 min-w-0 flex-1">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-[var(--color-accent)]/15">
            <Database size={18} className="text-[var(--color-accent)]" />
          </div>
          <div className="min-w-0 flex-1">
            <h4 className="truncate font-medium text-[var(--color-text)]">
              {dataset.name}
            </h4>
            {dataset.source_file && (
              <p className="truncate text-xs font-mono text-[var(--color-text-muted)]">
                <FileText size={10} className="inline mr-1" />
                {dataset.source_file}
              </p>
            )}
          </div>
        </div>

        <button
          onClick={() => onDelete(dataset)}
          className="shrink-0 rounded-md p-1.5 text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-danger)]/15 hover:text-[var(--color-danger)]"
          title="Cancella dataset"
        >
          <Trash2 size={16} />
        </button>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
        <span className="text-[var(--color-text-muted)]">
          <span className="font-mono text-[var(--color-text)]">
            {formatInt(dataset.num_examples)}
          </span>{' '}
          esempi
        </span>
        <span className="rounded bg-[var(--color-surface-2)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--color-text-muted)]">
          {dataset.format}
        </span>
        <span className="text-[var(--color-text-muted)]">
          {formatDateTime(dataset.created_at)}
        </span>
      </div>

      {onView && (
        <button
          onClick={() => onView(dataset)}
          className="mt-3 text-xs text-[var(--color-accent)] hover:underline"
        >
          Mostra esempi →
        </button>
      )}
    </div>
  )
}