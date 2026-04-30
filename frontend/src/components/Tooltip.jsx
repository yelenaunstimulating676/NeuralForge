/**
 * Tooltip CSS-only che appare al hover.
 *
 * Uso:
 *   <Tooltip content="Spiegazione...">
 *     <span>Label</span>
 *   </Tooltip>
 */

export default function Tooltip({ content, children, side = 'top' }) {
  const positions = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
  }

  return (
    <span className="group relative inline-flex">
      {children}
      <span
        role="tooltip"
        className={`pointer-events-none absolute z-50 whitespace-normal rounded-md border border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-2 text-xs text-[var(--color-text)] opacity-0 shadow-lg transition-opacity duration-150 group-hover:opacity-100 ${positions[side]}`}
        style={{ minWidth: '180px', maxWidth: '280px' }}
      >
        {content}
      </span>
    </span>
  )
}