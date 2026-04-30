/**
 * Barra di riempimento VRAM colorata.
 * Verde < 70%, giallo 70-90%, rosso > 90%.
 */

export default function GPUBar({ usedMb, totalMb, label = 'VRAM' }) {
  const pct = totalMb > 0 ? (usedMb / totalMb) * 100 : 0

  let color = 'bg-[var(--color-success)]'
  if (pct >= 90) color = 'bg-[var(--color-danger)]'
  else if (pct >= 70) color = 'bg-[var(--color-warning)]'

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-[var(--color-text-muted)]">{label}</span>
        <span className="font-mono text-[var(--color-text)]">
          {usedMb.toLocaleString()} / {totalMb.toLocaleString()} MB
          <span className="ml-2 text-[var(--color-text-muted)]">
            ({pct.toFixed(1)}%)
          </span>
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--color-surface-2)]">
        <div
          className={`h-full transition-all duration-500 ease-out ${color}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
    </div>
  )
}