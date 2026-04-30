/**
 * Card di un modello locale scaricato.
 */

import { Boxes, Trash2, HardDrive } from 'lucide-react'
import { formatBytes, formatDateTime } from '../utils/format'

const TAG_COLORS = {
  'qwen2.5': 'bg-purple-500/20 text-purple-300',
  'phi3.5': 'bg-blue-500/20 text-blue-300',
  phi2: 'bg-blue-500/20 text-blue-300',
  smollm2: 'bg-emerald-500/20 text-emerald-300',
  smollm3: 'bg-teal-500/20 text-teal-300',
  mistral: 'bg-orange-500/20 text-orange-300',
  tinyllama: 'bg-pink-500/20 text-pink-300',
}

function getTagClass(tag) {
  if (!tag) return 'bg-[var(--color-surface-2)] text-[var(--color-text-muted)]'
  return TAG_COLORS[tag] || 'bg-[var(--color-surface-2)] text-[var(--color-text-muted)]'
}

export default function ModelCard({ model, onDelete }) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4 transition-colors hover:border-[var(--color-accent)]/40">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 min-w-0 flex-1">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-[var(--color-accent)]/15">
            <Boxes size={18} className="text-[var(--color-accent)]" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <h4 className="truncate font-medium text-[var(--color-text)]">
                {model.display_name}
              </h4>
              {model.is_custom && (
                <span className="rounded bg-[var(--color-warning)]/20 px-1.5 py-0.5 text-[10px] font-medium text-[var(--color-warning)]">
                  custom
                </span>
              )}
            </div>
            <p className="truncate text-xs font-mono text-[var(--color-text-muted)]">
              {model.hf_repo}
            </p>
          </div>
        </div>

        <button
          onClick={() => onDelete(model)}
          className="shrink-0 rounded-md p-1.5 text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-danger)]/15 hover:text-[var(--color-danger)]"
          title="Cancella modello"
        >
          <Trash2 size={16} />
        </button>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
        {model.tag && (
          <span
            className={`rounded px-2 py-0.5 font-mono text-[10px] ${getTagClass(
              model.tag
            )}`}
          >
            {model.tag}
          </span>
        )}
        {model.params_billions != null && (
          <span className="text-[var(--color-text-muted)]">
            <span className="font-mono text-[var(--color-text)]">
              {model.params_billions}B
            </span>{' '}
            params
          </span>
        )}
        <span className="flex items-center gap-1 text-[var(--color-text-muted)]">
          <HardDrive size={12} />
          <span className="font-mono text-[var(--color-text)]">
            {formatBytes(model.size_bytes)}
          </span>
        </span>
        <span className="text-[var(--color-text-muted)]">
          {formatDateTime(model.downloaded_at)}
        </span>
      </div>
    </div>
  )
}