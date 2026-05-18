interface RiskCardProps {
  riskLevel: string
  summary: string
}

const riskStyles: Record<string, string> = {
  unknown: 'border-slate-200 bg-slate-50 text-slate-700',
  low: 'border-emerald-200 bg-emerald-50 text-emerald-700',
  moderate: 'border-amber-200 bg-amber-50 text-amber-700',
  high: 'border-orange-200 bg-orange-50 text-orange-700',
  very_high: 'border-rose-200 bg-rose-50 text-rose-700',
  possible_medical_emergency: 'border-red-300 bg-red-100 text-red-800',
}

export default function RiskCard({ riskLevel, summary }: RiskCardProps) {
  const style = riskStyles[riskLevel] ?? riskStyles.unknown
  const label = riskLevel.split('_').join(' ')
  return (
    <section className={`rounded-xl border p-4 ${style}`} aria-live="polite">
      <h3 className="text-sm font-semibold uppercase tracking-wide">Risk level: {label}</h3>
      <p className="mt-2 text-sm">{summary || 'Risk summary unavailable.'}</p>
      {riskLevel === 'possible_medical_emergency' && (
        <p className="mt-2 text-sm font-semibold">Emergency warning: seek immediate medical help.</p>
      )}
    </section>
  )
}
