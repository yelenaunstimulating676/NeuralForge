/**
 * Card di un singolo InstructionExample con instruction, input, output.
 * Riusato in preview (Step 2) e review (Step 3).
 */

import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'

export default function DatasetExampleCard({ example, index, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  const strategy = example.metadata?.strategy

  return (
    <div className="rounded-md border border-[var(--color-border)] bg-[var(--color-surface)]">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-start gap-2 px-3 py-2 text-left hover:bg-[var(--color-surface-2)]/50"
      >
        {open ? (
          <ChevronDown size={14} className="mt-0.5 shrink-0 text-[var(--color-text-muted)]" />
        ) : (
          <ChevronRight size={14} className="mt-0.5 shrink-0 text-[var(--color-text-muted)]" />
        )}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-mono text-[10px] text-[var(--color-text-muted)]">
              #{index + 1}
            </span>
            {strategy && (
              <span className="rounded bg-[var(--color-surface-2)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--color-text-muted)]">
                {strategy}
              </span>
            )}
          </div>
          <p className="mt-1 truncate text-sm text-[var(--color-text)]">
            {example.instruction}
          </p>
        </div>
      </button>

      {open && (
        <div className="space-y-3 border-t border-[var(--color-border)] px-3 py-3">
          <Field label="Instruction" value={example.instruction} />
          {example.input && <Field label="Input" value={example.input} />}
          <Field label="Output" value={example.output} accent />
        </div>
      )}
    </div>
  )
}

function Field({ label, value, accent = false }) {
  return (
    <div>
      <p className="mb-1 text-[10px] font-medium uppercase tracking-wide text-[var(--color-text-muted)]">
        {label}
      </p>
      <p
        className={`whitespace-pre-wrap text-sm ${
          accent ? 'text-[var(--color-text)]' : 'text-[var(--color-text-muted)]'
        }`}
      >
        {value}
      </p>
    </div>
  )
}