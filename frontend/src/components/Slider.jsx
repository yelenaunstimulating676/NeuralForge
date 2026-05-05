/**
 * Slider HTML range con etichetta e valore corrente affiancato.
 */

export default function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  unit = '',
  hint,
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label className="text-xs font-medium text-[var(--color-text)]">
          {label}
        </label>
        <span className="font-mono text-xs text-[var(--color-text-muted)]">
          {value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-2 w-full accent-[var(--color-accent)]"
      />
      {hint && (
        <p className="mt-1 text-[10px] text-[var(--color-text-muted)]/80">{hint}</p>
      )}
    </div>
  )
}