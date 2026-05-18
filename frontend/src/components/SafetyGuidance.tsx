interface SafetyGuidanceProps {
  drivingGuidance: string
  continueGuidance: string
  hydrationGuidance: string
  foodGuidance: string
  medicalWarning: string
}

export default function SafetyGuidance({
  drivingGuidance,
  continueGuidance,
  hydrationGuidance,
  foodGuidance,
  medicalWarning,
}: SafetyGuidanceProps) {
  const item = (title: string, value: string, tone?: 'warning' | 'normal') => (
    <div className={`rounded-xl border p-4 ${tone === 'warning' ? 'border-rose-200 bg-rose-50' : 'border-slate-200 bg-white'}`}>
      <h4 className="text-sm font-semibold text-slate-900">{title}</h4>
      <p className="mt-1 text-sm text-slate-700">{value || 'Not available.'}</p>
    </div>
  )

  return (
    <div className="grid gap-3 md:grid-cols-2">
      {item('Driving guidance', drivingGuidance, 'warning')}
      {item('Continue drinking guidance', continueGuidance, 'warning')}
      {item('Hydration', hydrationGuidance)}
      {item('Food', foodGuidance)}
      <div className="md:col-span-2">{item('Medical warning', medicalWarning, 'warning')}</div>
    </div>
  )
}
