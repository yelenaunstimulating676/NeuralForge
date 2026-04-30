/**
 * Card riassuntiva GPU.
 * Mostra nome, compute capability, driver, CUDA, supporti dtype, VRAM live.
 */

import { useEffect, useState } from 'react'
import { Cpu, AlertTriangle } from 'lucide-react'
import { fetchVram } from '../api/client'
import GPUBar from './GPUBar'

const POLL_INTERVAL_MS = 2000

export default function GPUCard({ gpu }) {
  const [vram, setVram] = useState({
    total_mb: gpu.vram_total_mb,
    used_mb: gpu.vram_used_mb,
    free_mb: gpu.vram_free_mb,
  })
  const [pollError, setPollError] = useState(null)

  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      try {
        const data = await fetchVram(gpu.index)
        if (!cancelled) {
          setVram(data)
          setPollError(null)
        }
      } catch (err) {
        if (!cancelled) setPollError(err.message)
      }
    }

    poll()
    const id = setInterval(poll, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [gpu.index])

  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <Cpu size={18} className="text-[var(--color-accent)]" />
          <h3 className="text-sm font-medium text-[var(--color-text-muted)]">
            GPU {gpu.index}
          </h3>
        </div>
        {pollError && (
          <span title={pollError}>
            <AlertTriangle size={14} className="text-[var(--color-warning)]" />
          </span>
        )}
      </div>

      <p className="mt-2 text-lg font-semibold text-[var(--color-text)]">
        {gpu.name}
      </p>

      <div className="mt-4">
        <GPUBar usedMb={vram.used_mb} totalMb={vram.total_mb} />
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
        <Detail label="Compute" value={gpu.compute_capability} />
        <Detail label="Driver" value={gpu.driver_version ?? 'n/a'} />
        <Detail label="CUDA" value={gpu.cuda_runtime_version ?? 'n/a'} />
        <Detail
          label="Precision"
          value={
            <>
              {gpu.bf16_supported && (
                <span className="mr-1 inline-block rounded bg-[var(--color-surface-2)] px-1.5 py-0.5 font-mono text-[10px]">
                  bf16
                </span>
              )}
              {gpu.fp16_supported && (
                <span className="inline-block rounded bg-[var(--color-surface-2)] px-1.5 py-0.5 font-mono text-[10px]">
                  fp16
                </span>
              )}
            </>
          }
        />
      </dl>
    </div>
  )
}

function Detail({ label, value }) {
  return (
    <div>
      <dt className="text-[var(--color-text-muted)]">{label}</dt>
      <dd className="mt-0.5 font-mono text-[var(--color-text)]">{value}</dd>
    </div>
  )
}