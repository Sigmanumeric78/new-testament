import { useCallback, useEffect, useMemo, useState } from 'react'
import ChemicalDetailPanel from '../components/ChemicalDetailPanel'
import ChemicalFilterPanel from '../components/ChemicalFilterPanel'
import ChemicalList from '../components/ChemicalList'
import ChemicalSearchBar from '../components/ChemicalSearchBar'
import HealthBadge from '../components/HealthBadge'
import Layout from '../components/Layout'
import { getHealth } from '../lib/api'
import { getChemicalConformer, getChemicalDetail, listChemicals } from '../lib/chemicalApi'
import type { ChemicalConformerResponse, ChemicalDetail, ChemicalSummary } from '../lib/chemicalTypes'
import type { HealthResponse } from '../lib/types'

const PAGE_SIZE = 24

export default function ChemicalExplorerPage() {
  const [searchInput, setSearchInput] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedClass, setSelectedClass] = useState('')
  const [has3dOnly, setHas3dOnly] = useState(false)
  const [offset, setOffset] = useState(0)

  const [items, setItems] = useState<ChemicalSummary[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [selectedCompoundId, setSelectedCompoundId] = useState('')
  const [detail, setDetail] = useState<ChemicalDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState('')

  const [conformer, setConformer] = useState<ChemicalConformerResponse | null>(null)
  const [conformerLoading, setConformerLoading] = useState(false)
  const [conformerError, setConformerError] = useState('')

  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthLoading, setHealthLoading] = useState(true)

  const classOptions = useMemo(() => {
    const uniq = new Set<string>()
    for (const item of items) {
      if (item.chemical_class) {
        uniq.add(item.chemical_class)
      }
    }
    return Array.from(uniq).sort((a, b) => a.localeCompare(b))
  }, [items])

  const pageLabel = useMemo(() => {
    if (!total) return '0 compounds'
    const start = offset + 1
    const end = Math.min(offset + PAGE_SIZE, total)
    return `${start}-${end} of ${total}`
  }, [offset, total])

  const loadList = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const payload = await listChemicals({
        q: searchQuery,
        chemical_class: selectedClass,
        has_3d: has3dOnly ? true : undefined,
        limit: PAGE_SIZE,
        offset,
      })
      setItems(payload.items)
      setTotal(payload.total)

      if (!payload.items.length) {
        setSelectedCompoundId('')
        setDetail(null)
        setConformer(null)
      } else if (!payload.items.some((item) => item.compound_id === selectedCompoundId)) {
        setSelectedCompoundId(payload.items[0].compound_id)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load compounds.')
      setItems([])
      setTotal(0)
    } finally {
      setLoading(false)
    }
  }, [offset, searchQuery, selectedClass, has3dOnly, selectedCompoundId])

  useEffect(() => {
    void loadList()
  }, [loadList])

  useEffect(() => {
    let mounted = true
    const run = async () => {
      setHealthLoading(true)
      try {
        const payload = await getHealth()
        if (mounted) {
          setHealth(payload)
        }
      } catch {
        if (mounted) {
          setHealth({ status: 'error', components: {} })
        }
      } finally {
        if (mounted) {
          setHealthLoading(false)
        }
      }
    }

    void run()
    return () => {
      mounted = false
    }
  }, [])

  useEffect(() => {
    if (!selectedCompoundId) {
      setDetail(null)
      setConformer(null)
      return
    }

    let mounted = true

    const run = async () => {
      setDetailLoading(true)
      setDetailError('')
      setConformerLoading(true)
      setConformerError('')
      try {
        const [detailPayload, conformerPayload] = await Promise.all([
          getChemicalDetail(selectedCompoundId),
          getChemicalConformer(selectedCompoundId),
        ])
        if (!mounted) return
        setDetail(detailPayload)
        setConformer(conformerPayload)
      } catch (err) {
        if (!mounted) return
        const message = err instanceof Error ? err.message : 'Failed to load compound detail.'
        setDetailError(message)
        setConformerError(message)
        setDetail(null)
        setConformer(null)
      } finally {
        if (mounted) {
          setDetailLoading(false)
          setConformerLoading(false)
        }
      }
    }

    void run()

    return () => {
      mounted = false
    }
  }, [selectedCompoundId])

  const sidebar = (
    <aside className="w-full space-y-4 border-r border-slate-200 bg-slate-100/70 p-4 lg:w-96">
      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-900">Chemical Explorer</h2>
        <p className="mt-1 text-xs text-slate-600">Explore compounds, beverage coverage, and 3D conformers.</p>
      </div>

      <ChemicalFilterPanel
        classes={classOptions}
        selectedClass={selectedClass}
        has3dOnly={has3dOnly}
        onClassChange={(value) => {
          setSelectedClass(value)
          setOffset(0)
        }}
        onHas3dOnlyChange={(value) => {
          setHas3dOnly(value)
          setOffset(0)
        }}
      />

      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">System health</p>
        <HealthBadge health={health} isLoading={healthLoading} />
      </div>
    </aside>
  )

  return (
    <Layout sidebar={sidebar}>
      <div className="mx-auto max-w-7xl space-y-4">
        <section className="rounded-2xl bg-gradient-to-r from-brand-50 to-white p-6">
          <h1 className="text-2xl font-semibold tracking-tight text-slate-900">Chemical Explorer</h1>
          <p className="mt-1 text-sm text-slate-700">
            Browse beverage-related compounds and view 3D conformers when available.
          </p>
        </section>

        <ChemicalSearchBar
          value={searchInput}
          loading={loading}
          onChange={setSearchInput}
          onSubmit={() => {
            setSearchQuery(searchInput)
            setOffset(0)
          }}
        />

        <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-slate-600">
          <p>{pageLabel}</p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setOffset((current) => Math.max(0, current - PAGE_SIZE))}
              disabled={offset <= 0 || loading}
              className="focus-ring rounded-lg border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Previous
            </button>
            <button
              type="button"
              onClick={() => setOffset((current) => current + PAGE_SIZE)}
              disabled={offset + PAGE_SIZE >= total || loading}
              className="focus-ring rounded-lg border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Next
            </button>
          </div>
        </div>

        {error && <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">{error}</div>}

        <div className="grid gap-4 xl:grid-cols-[380px_1fr]">
          <ChemicalList
            items={items}
            selectedCompoundId={selectedCompoundId}
            onSelect={setSelectedCompoundId}
            loading={loading}
          />

          <div className="space-y-3">
            {detailLoading && (
              <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-600" role="status" aria-live="polite">
                Loading compound details...
              </div>
            )}
            {detailError && <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">{detailError}</div>}
            {!detailLoading && !detailError && (
              <ChemicalDetailPanel
                detail={detail}
                conformer={conformer}
                conformerLoading={conformerLoading}
                conformerError={conformerError}
              />
            )}
          </div>
        </div>
      </div>
    </Layout>
  )
}
