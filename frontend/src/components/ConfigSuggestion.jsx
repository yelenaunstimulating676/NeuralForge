/**
 * Card che mostra la configurazione di training suggerita per la GPU.
 * Read-only per ora; in M5 diventerà editabile.
 */

import { Settings2, Info, HelpCircle } from 'lucide-react'
import Tooltip from './Tooltip'

const TOOLTIPS = {
  Strategy:
    'Tecnica di fine-tuning. QLORA quantizza il modello base a 4-bit per risparmiare VRAM, LORA usa fp16/bf16, FULL aggiorna tutti i pesi (richiede molta VRAM).',
  Precision:
    'Tipo di dato per il training. bf16 (Ampere+) è il miglior compromesso tra range numerico e memoria. fp16 è più veloce ma soggetto a overflow. fp32 è il più stabile ma il doppio della memoria.',
  'Batch size':
    'Numero di esempi processati in parallelo per ogni step di training. Più grande = più stabile ma più VRAM.',
  'Grad accum':
    'Step di accumulazione gradient. Permette di simulare batch più grandi senza aumentare la VRAM. Esempio: batch=4 + grad_accum=4 simula un batch=16.',
  'Effective batch':
    'Batch size effettivo = batch_size × gradient_accumulation_steps. È il batch "vero" su cui si calcola lo step optimizer.',
  'Max seq length':
    'Lunghezza massima della sequenza in token. Esempi più lunghi vengono troncati. Più alto = più contesto ma più VRAM.',
  'LoRA rank':
    'Dimensione delle matrici di adattamento LoRA. Rank più alto = più capacità di apprendere, più parametri trainabili. Tipico: 8-32.',
  'LoRA alpha':
    'Fattore di scala LoRA. Convenzione comune: alpha = 2 × rank. Influenza l\'intensità degli aggiornamenti LoRA.',
  '4-bit quant':
    'Quantizzazione del modello base a 4-bit (NF4). Riduce la VRAM occupata dal modello di ~4× con perdita di qualità minima. Necessaria su GPU consumer.',
  '8-bit optim':
    'Stati dell\'optimizer (AdamW) salvati a 8-bit invece che 32-bit. Dimezza la VRAM occupata da gradients momentum/variance.',
  'Grad checkpoint':
    'Ricalcola le activations durante il backward invece di tenerle in memoria. Riduce VRAM ~30% al costo di ~20% in più di tempo per step.',
}

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
    { label: 'Grad checkpoint', value: config.gradient_checkpointing ? 'On' : 'Off' },
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
        di avviare il training (M5).{' '}
        <span className="text-[var(--color-text-muted)]/70">
          Hover sui nomi per i dettagli.
        </span>
      </p>

      <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3 lg:grid-cols-4">
        {params.map(({ label, value, highlight }) => (
          <div key={label}>
            <Tooltip content={TOOLTIPS[label] ?? ''}>
              <dt className="inline-flex items-center gap-1 text-xs text-[var(--color-text-muted)] cursor-help">
                {label}
                <HelpCircle size={11} className="opacity-60" />
              </dt>
            </Tooltip>
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