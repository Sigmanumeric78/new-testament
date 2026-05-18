import type { ChemicalSummary } from '../lib/chemicalTypes'
import ChemicalCard from './ChemicalCard'

interface ChemicalListProps {
  items: ChemicalSummary[]
  selectedCompoundId: string
  onSelect: (compoundId: string) => void
  loading: boolean
}

export default function ChemicalList({ items, selectedCompoundId, onSelect, loading }: ChemicalListProps) {
  if (loading) {
    return (
      <div className="rounded-2xl border border-slate-200 bg-white p-4 text-sm text-slate-600" role="status" aria-live="polite">
        Loading compounds...
      </div>
    )
  }

  if (!items.length) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-6 text-sm text-slate-600">
        No compounds match the current filters.
      </div>
    )
  }

  return (
    <div className="space-y-2" aria-label="Chemical results list">
      {items.map((item) => (
        <ChemicalCard
          key={item.compound_id}
          item={item}
          selected={item.compound_id === selectedCompoundId}
          onSelect={onSelect}
        />
      ))}
    </div>
  )
}
