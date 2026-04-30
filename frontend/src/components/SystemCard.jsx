/**
 * Card sistema: OS, Python, PyTorch, CUDA availability.
 */

import { Server, CheckCircle2, XCircle } from 'lucide-react'
import { formatInt } from '../utils/format'

export default function SystemCard({ info }) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
      <div className="flex items-center gap-2">
        <Server size={18} className="text-[var(--color-accent)]" />
        <h3 className="text-sm font-medium text-[var(--color-text-muted)]">
          Sistema
        </h3>
      </div>

      <dl className="mt-4 space-y-2 text-sm">
        <Row label="OS" value={info.os} />
        <Row label="Python" value={info.python_version} />
        <Row label="PyTorch" value={info.torch_version} />
        <Row
          label="CUDA"
          value={
            <span
              className={`inline-flex items-center gap-1 ${
                info.cuda_available
                  ? 'text-[var(--color-success)]'
                  : 'text-[var(--color-danger)]'
              }`}
            >
              {info.cuda_available ? (
                <>
                  <CheckCircle2 size={14} /> Disponibile
                </>
              ) : (
                <>
                  <XCircle size={14} /> Non disponibile
                </>
              )}
            </span>
          }
        />
        <Row label="GPU rilevate" value={formatInt(info.gpu_count)} />
      </dl>
    </div>
  )
}

function Row({ label, value }) {
  return (
    <div className="flex items-center justify-between">
      <dt className="text-[var(--color-text-muted)]">{label}</dt>
      <dd className="font-mono text-[var(--color-text)]">{value}</dd>
    </div>
  )
}