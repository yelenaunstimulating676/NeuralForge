/**
 * Helper di formattazione numerica per NeuralForge.
 * Tutti i numeri visibili all'utente passano da qui.
 *
 * Locale forzato a 'it-IT' per consistenza visiva indipendentemente
 * dalla locale del browser.
 */

const LOCALE = 'it-IT'

/** Formatta un intero con separatori migliaia. Es: 12282 → "12.282". */
export function formatInt(n) {
  if (n == null || Number.isNaN(n)) return '—'
  return Number(n).toLocaleString(LOCALE)
}

/** Formatta un float a N decimali con locale italiana. */
export function formatFloat(n, decimals = 1) {
  if (n == null || Number.isNaN(n)) return '—'
  return Number(n).toLocaleString(LOCALE, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

/** Formatta una percentuale 0-100. Es: 14.32 → "14,3%". */
export function formatPercent(n, decimals = 1) {
  if (n == null || Number.isNaN(n)) return '—'
  return `${formatFloat(n, decimals)}%`
}

/** Formatta MB con unità. Es: 12282 → "12.282 MB". */
export function formatMB(mb) {
  return `${formatInt(mb)} MB`
}

/** Formatta MB come GB se > 1024. Es: 12282 → "12,0 GB". */
export function formatMBasGB(mb, decimals = 1) {
  if (mb == null) return '—'
  if (mb < 1024) return formatMB(mb)
  return `${formatFloat(mb / 1024, decimals)} GB`
}

/** Formatta byte come MB/GB human-readable. */
export function formatBytes(bytes, decimals = 1) {
  if (bytes == null || Number.isNaN(bytes)) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${formatFloat(bytes / 1024, decimals)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${formatFloat(bytes / (1024 * 1024), decimals)} MB`
  return `${formatFloat(bytes / (1024 * 1024 * 1024), decimals)} GB`
}

/** Formatta una data ISO come dd/mm/yyyy hh:mm in locale italiana. */
export function formatDateTime(isoString) {
  if (!isoString) return '—'
  try {
    const d = new Date(isoString)
    return d.toLocaleString('it-IT', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return isoString
  }
}