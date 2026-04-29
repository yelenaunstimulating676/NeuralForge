/**
 * Componente placeholder per pagine non ancora implementate.
 * Mostra titolo, descrizione, e in quale milestone sarà costruita.
 */

import { Construction } from 'lucide-react'

export default function PlaceholderPage({ title, description, milestone }) {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold text-[var(--color-text)]">
          {title}
        </h1>
        <p className="mt-1 text-sm text-[var(--color-text-muted)]">
          {description}
        </p>
      </header>

      <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-[var(--color-border)] bg-[var(--color-surface)] py-16 px-6 text-center">
        <Construction size={36} className="text-[var(--color-text-muted)]" />
        <p className="mt-4 text-sm text-[var(--color-text-muted)]">
          Questa sezione sarà disponibile in:
        </p>
        <p className="mt-2 text-base font-medium text-[var(--color-accent)]">
          {milestone}
        </p>
      </div>
    </div>
  )
}