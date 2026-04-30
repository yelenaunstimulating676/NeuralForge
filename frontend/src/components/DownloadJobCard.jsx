/**
 * Card di un download job in corso o terminato.
 */

import { X, Loader2, CheckCircle2, XCircle, AlertCircle } from 'lucide-react'
import ProgressBar from './ProgressBar'

const STATUS_COLORS = {
  pending: 'text-[var(--color-text-muted)]',
  running: 'text-[var(--color-accent)]',
  completed: 'text-[var(--color-success)]',
  failed: 'text-[var(--color-danger)]',
  cancelled: 'text-[var(--color-warning)]',
}

const STATUS_ICONS = {
  pending: Loader2,
  running: Loader2,
  completed: CheckCircle2,
  failed: XCircle,
  cancelled: AlertCircle,
}

export default function DownloadJobCard({ job, onCancel }) {
  const StatusIcon = STATUS_ICONS[job.status] || Loader2
  const colorClass = STATUS_COLORS[job.status] || ''
  const isActive = job.status === 'running' || job.status === 'pending'
  const repo = job.result?.hf_repo ?? '—'

  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <StatusIcon
            size={16}
            className={`${colorClass} ${isActive ? 'animate-spin' : ''}`}
          />
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-[var(--color-text)]">
              {repo}
            </p>
            <p className={`text-xs ${colorClass} capitalize`}>{job.status}</p>
          </div>
        </div>

        {isActive && (
          <button
            onClick={() => onCancel(job.id)}
            className="shrink-0 rounded-md p-1 text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-danger)]/15 hover:text-[var(--color-danger)]"
            title="Annulla download"
          >
            <X size={16} />
          </button>
        )}
      </div>

      {(isActive || job.status === 'completed') && (
        <div className="mt-3">
          <ProgressBar
            value={job.progress}
            message={job.progress_message}
            color={job.status === 'completed' ? 'success' : 'accent'}
          />
        </div>
      )}

      {job.status === 'failed' && job.error && (
        <p className="mt-2 text-xs text-[var(--color-danger)]">{job.error}</p>
      )}
    </div>
  )
}