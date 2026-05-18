import { formatBac, formatHours } from '../lib/format'

interface EstimateCardsProps {
  peakBac: number | null
  timeToSober: number | null
  timeToPeak: number | null
}

export default function EstimateCards({ peakBac, timeToSober, timeToPeak }: EstimateCardsProps) {
  const card = (label: string, value: string) => (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-2 text-lg font-semibold text-slate-900">{value}</p>
    </div>
  )

  return (
    <div className="grid gap-3 md:grid-cols-3">
      {card('Estimated peak BAC', formatBac(peakBac))}
      {card('Estimated time to sober', formatHours(timeToSober))}
      {card('Estimated time to peak', formatHours(timeToPeak))}
    </div>
  )
}
