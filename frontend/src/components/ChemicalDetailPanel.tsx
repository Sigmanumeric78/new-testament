import type { ChemicalConformerResponse, ChemicalDetail } from '../lib/chemicalTypes'
import Chemical3DViewer from './Chemical3DViewer'

interface ChemicalDetailPanelProps {
  detail: ChemicalDetail | null
  conformer: ChemicalConformerResponse | null
  conformerLoading: boolean
  conformerError: string
}

function badge(label: string, tone: 'ok' | 'warn' | 'neutral' = 'neutral') {
  const cls =
    tone === 'ok'
      ? 'border-emerald-300 bg-emerald-50 text-emerald-700'
      : tone === 'warn'
        ? 'border-amber-300 bg-amber-50 text-amber-700'
        : 'border-slate-300 bg-slate-100 text-slate-700'
  return <span className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${cls}`}>{label}</span>
}

export default function ChemicalDetailPanel({ detail, conformer, conformerLoading, conformerError }: ChemicalDetailPanelProps) {
  if (!detail) {
    return (
      <section className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-6 text-sm text-slate-600">
        Select a compound to view details.
      </section>
    )
  }

  return (
    <section className="space-y-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-xl font-semibold tracking-tight text-slate-900">{detail.compound_name}</h3>
          <p className="text-sm text-slate-600">PubChem CID: {detail.pubchem_cid ?? 'N/A'}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {badge(detail.chemical_class || 'unknown')}
          {detail.has_3d_conformer ? badge('3D available', 'ok') : badge('No 3D conformer', 'warn')}
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Canonical SMILES</p>
          <p className="mt-1 break-all font-mono text-xs text-slate-800">{detail.canonical_smiles || 'Unavailable'}</p>
        </div>
        <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Beverage coverage</p>
          <p className="mt-1 text-sm text-slate-700">{detail.beverage_count} beverages</p>
          <p className="mt-1 text-xs text-slate-600">{detail.beverage_examples.join(', ') || 'No beverage examples available.'}</p>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-xl border border-slate-200 p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Metabolism relevance</p>
          <p className="mt-1 text-sm text-slate-700">{detail.metabolism_relevance}</p>
        </div>
        <div className="rounded-xl border border-slate-200 p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Toxicity relevance</p>
          <p className="mt-1 text-sm text-slate-700">{detail.toxicity_relevance}</p>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 p-3">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Related beverages</p>
        <p className="mt-1 text-sm text-slate-700">{detail.related_beverages.join(', ') || 'No related beverages listed.'}</p>
      </div>

      <div className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">3D conformer viewer</p>
        <Chemical3DViewer conformer={conformer} loading={conformerLoading} error={conformerError} />
      </div>
    </section>
  )
}
