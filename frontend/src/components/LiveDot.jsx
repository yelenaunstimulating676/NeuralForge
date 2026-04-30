/**
 * Pallino pulsante che lampeggia ad ogni "tick" passato come prop.
 *
 * Uso:
 *   <LiveDot tick={vramReading} />
 *
 * Ogni volta che `tick` cambia (anche se è lo stesso oggetto con valori
 * uguali, ma riferimento diverso), il pallino lampeggia per 600ms.
 */

import { useEffect, useState } from 'react'

export default function LiveDot({ tick, color = 'var(--color-success)' }) {
  const [pulse, setPulse] = useState(false)

  useEffect(() => {
    setPulse(true)
    const t = setTimeout(() => setPulse(false), 600)
    return () => clearTimeout(t)
  }, [tick])

  return (
    <span className="relative inline-flex h-2.5 w-2.5 items-center justify-center">
      {/* Halo */}
      <span
        className={`absolute inline-flex h-full w-full rounded-full opacity-75 transition-opacity ${
          pulse ? 'animate-ping' : 'opacity-0'
        }`}
        style={{ backgroundColor: color }}
      />
      {/* Core */}
      <span
        className="relative inline-flex h-2 w-2 rounded-full"
        style={{ backgroundColor: color }}
      />
    </span>
  )
}