interface ChemicalSearchBarProps {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  loading: boolean
}

export default function ChemicalSearchBar({ value, onChange, onSubmit, loading }: ChemicalSearchBarProps) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <label htmlFor="chemical-search" className="text-sm font-semibold text-slate-800">
        Search compounds
      </label>
      <div className="mt-2 flex flex-col gap-2 sm:flex-row">
        <input
          id="chemical-search"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder="Try ethanol, sulfites, histamine, acetaldehyde..."
          className="focus-ring w-full rounded-xl border border-slate-300 px-3 py-2 text-sm"
        />
        <button
          type="button"
          onClick={onSubmit}
          disabled={loading}
          className="focus-ring rounded-xl bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>
    </div>
  )
}
