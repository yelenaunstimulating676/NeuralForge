/**
 * Card che mostra la configurazione di training suggerita per la GPU
 * rilevata. È read-only per ora; in M5 diventerà editabile.
 */

import { Settings2, Info } from 'lucide-react'

export default function ConfigSuggestion({ config }) {
  const params = [
    { label: 'Strategy', value: config.strategy.toUpperCase(), highlight: true },
    { label: 'Precision', value: config.mixed_precision_dtype },
    { label: 'Batch size', value: config.batch_size },
    { label: 'Grad accum', value: config.gradient_accumulation_steps },
    { label: 'Effective batch', value: config.effective_batch_size },
    { label: 'Max seq length', value: config.max_seq_length },
    { label: 'LoRA rank', value: config.lora_rank },
    { label: 'LoRA alpha', value: config.lora_alpha },
    { label: '4-bit quant', value: config.use_4bit ? 'On' : 'Off' },
    { label: '8-bit optim', value: config.use_8bit_optimizer ? 'On' : 'Off' },
    {
      label: 'Grad checkpoint',
      value: config.gradient_checkpointing ? 'On' : 'Off',
    },
  ]

  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
      <div className="flex items-center gap-2">
        <Settings2 size={18} className="text-[var(--color-accent)]" />
        <h3 className="text-sm font-medium text-[var(--color-text-muted)]">
          Configurazione di training suggerita
        </h3>
      </div>

      <p className="mt-1 text-xs text-[var(--color-text-muted)]">
        Calcolata automaticamente in base alla tua GPU. Modificabile prima
        di avviare il training (M5).
      </p>

      <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3 lg:grid-cols-4">
        {params.map(({ label, value, highlight }) => (
          <div key={label}>
            <dt className="text-xs text-[var(--color-text-muted)]">{label}</dt>
            <dd
              className={`mt-0.5 font-mono text-sm ${
                highlight
                  ? 'font-semibold text-[var(--color-accent)]'
                  : 'text-[var(--color-text)]'
              }`}
            >
              {value}
            </dd>
          </div>
        ))}
      </div>

      {config.notes && config.notes.length > 0 && (
        <div className="mt-4 flex gap-2 rounded-md border border-[var(--color-border)] bg-[var(--color-surface-2)] p-3">
          <Info
            size={14}
            className="mt-0.5 shrink-0 text-[var(--color-text-muted)]"
          />
          <ul className="space-y-1 text-xs text-[var(--color-text-muted)]">
            {config.notes.map((note, i) => (
              <li key={i}>{note}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}