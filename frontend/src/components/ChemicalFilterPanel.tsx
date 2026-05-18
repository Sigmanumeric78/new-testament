interface ChemicalFilterPanelProps {
  classes: string[]
  selectedClass: string
  has3dOnly: boolean
  onClassChange: (value: string) => void
  onHas3dOnlyChange: (value: boolean) => void
}

export default function ChemicalFilterPanel({
  classes,
  selectedClass,
  has3dOnly,
  onClassChange,
  onHas3dOnlyChange,
}: ChemicalFilterPanelProps) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <h3 className="text-sm font-semibold text-slate-900">Filters</h3>

      <div className="mt-3 space-y-3">
        <div>
          <label htmlFor="chemical-class" className="block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Chemical class
          </label>
          <select
            id="chemical-class"
            value={selectedClass}
            onChange={(event) => onClassChange(event.target.value)}
            className="focus-ring mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"
          >
            <option value="">All classes</option>
            {classes.map((entry) => (
              <option key={entry} value={entry}>
                {entry}
              </option>
            ))}
          </select>
        </div>

        <label className="inline-flex items-center gap-2 text-sm text-slate-700">
          <input
            type="checkbox"
            checked={has3dOnly}
            onChange={(event) => onHas3dOnlyChange(event.target.checked)}
            className="h-4 w-4 rounded border-slate-300"
          />
          Has 3D conformer only
        </label>
      </div>
    </section>
  )
}
