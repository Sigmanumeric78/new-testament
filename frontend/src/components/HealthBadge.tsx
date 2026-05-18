import type { HealthResponse } from '../lib/types'
import { healthLabel } from '../lib/format'

interface HealthBadgeProps {
  health: HealthResponse | null
  isLoading: boolean
}

export default function HealthBadge({ health, isLoading }: HealthBadgeProps) {
  const label = isLoading ? 'Checking...' : health ? healthLabel(health.status) : 'Offline'
  const cls =
    label === 'Healthy'
      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
      : label === 'Degraded'
        ? 'bg-amber-50 text-amber-700 border-amber-200'
        : 'bg-rose-50 text-rose-700 border-rose-200'

  const degradedDetails =
    health?.status === 'degraded'
      ? Object.entries(health.components)
          .filter(([, value]) => !value.ok)
          .map(([key, value]) => `${key}: ${value.detail}`)
      : []

  return (
    <div className="space-y-2">
      <span
        className={`inline-flex rounded-full border px-3 py-1 text-xs font-semibold ${cls}`}
        title={degradedDetails.join('\n')}
        aria-label={`System health: ${label}`}
      >
        {label}
      </span>
      {degradedDetails.length > 0 && <p className="text-xs text-slate-500">Some backend services are degraded.</p>}
    </div>
  )
}
