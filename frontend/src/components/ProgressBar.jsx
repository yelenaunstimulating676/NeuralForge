/**
 * Progress bar generica con percentuale e messaggio opzionale.
 */

import { formatPercent } from '../utils/format'

export default function ProgressBar({ value, message, color = 'accent' }) {
  const pct = Math.min(100, Math.max(0, (value ?? 0) * 100))
  const bgColor =
    color === 'success'
      ? 'bg-[var(--color-success)]'
      : color === 'warning'
      ? 'bg-[var(--color-warning)]'
      : color === 'danger'
      ? 'bg-[var(--color-danger)]'
      : 'bg-[var(--color-accent)]'

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-[var(--color-text-muted)]">{message ?? ''}</span>
        <span className="font-mono text-[var(--color-text)]">
          {formatPercent(pct)}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--color-surface-2)]">
        <div
          className={`h-full transition-all duration-500 ease-out ${bgColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}