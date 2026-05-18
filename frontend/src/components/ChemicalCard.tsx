import type { ChemicalSummary } from '../lib/chemicalTypes'

interface ChemicalCardProps {
  item: ChemicalSummary
  selected: boolean
  onSelect: (compoundId: string) => void
}

export default function ChemicalCard({ item, selected, onSelect }: ChemicalCardProps) {
  return (
    <button
      type="button"
      onClick={() => onSelect(item.compound_id)}
      className={`focus-ring w-full rounded-xl border p-3 text-left transition ${
        selected ? 'border-brand-500 bg-brand-50' : 'border-slate-200 bg-white hover:border-slate-300'
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-sm font-semibold text-slate-900">{item.compound_name}</p>
          <p className="text-xs text-slate-600">{item.chemical_class || 'unknown'}</p>
        </div>
        <span
          className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
            item.has_3d_conformer
              ? 'border-emerald-300 bg-emerald-50 text-emerald-700'
              : 'border-slate-300 bg-slate-100 text-slate-600'
          }`}
        >
          {item.has_3d_conformer ? '3D' : 'No 3D'}
        </span>
      </div>

      <div className="mt-2 text-xs text-slate-600">
        <p>PubChem CID: {item.pubchem_cid ?? 'N/A'}</p>
        <p>Beverages: {item.beverage_count}</p>
      </div>
    </button>
  )
}
