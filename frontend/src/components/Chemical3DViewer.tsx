import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChemicalConformerResponse } from '../lib/chemicalTypes'

declare global {
  interface Window {
    $3Dmol?: {
      createViewer: (element: HTMLElement, options?: Record<string, unknown>) => {
        clear: () => void
        addModel: (data: string, format: string) => void
        setStyle: (selection: Record<string, unknown>, style: Record<string, unknown>) => void
        zoomTo: () => void
        render: () => void
      }
    }
  }
}

type ViewerStyle = 'stick' | 'line'

interface Chemical3DViewerProps {
  conformer: ChemicalConformerResponse | null
  loading: boolean
  error: string
}

let loadingPromise: Promise<void> | null = null

function load3DmolScript(): Promise<void> {
  if (typeof window === 'undefined') {
    return Promise.reject(new Error('window unavailable'))
  }

  if (window.$3Dmol) {
    return Promise.resolve()
  }

  if (loadingPromise) {
    return loadingPromise
  }

  loadingPromise = new Promise((resolve, reject) => {
    const existing = document.getElementById('threedmol-script') as HTMLScriptElement | null
    if (existing) {
      existing.addEventListener('load', () => resolve(), { once: true })
      existing.addEventListener('error', () => reject(new Error('3Dmol failed to load')), { once: true })
      return
    }

    const script = document.createElement('script')
    script.id = 'threedmol-script'
    script.src = 'https://3Dmol.csb.pitt.edu/build/3Dmol-min.js'
    script.async = true
    script.onload = () => resolve()
    script.onerror = () => reject(new Error('3Dmol failed to load'))
    document.head.appendChild(script)
  })

  return loadingPromise
}

export default function Chemical3DViewer({ conformer, loading, error }: Chemical3DViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const viewerRef = useRef<any>(null)
  const [style, setStyle] = useState<ViewerStyle>('stick')
  const [engineError, setEngineError] = useState('')

  const canRender3d = useMemo(() => {
    return !!(conformer?.has_3d_conformer && conformer?.format === 'sdf' && conformer?.sdf)
  }, [conformer])

  useEffect(() => {
    if (!canRender3d || !containerRef.current || !conformer?.sdf) {
      viewerRef.current = null
      return
    }

    let cancelled = false

    const renderModel = async () => {
      try {
        setEngineError('')
        await load3DmolScript()
        if (cancelled || !containerRef.current || !window.$3Dmol) {
          return
        }

        const viewer = window.$3Dmol.createViewer(containerRef.current, {
          backgroundColor: 'white',
        })
        viewer.clear()
        viewer.addModel(conformer.sdf, 'sdf')
        viewer.setStyle({}, style === 'stick' ? { stick: {} } : { line: {} })
        viewer.zoomTo()
        viewer.render()
        viewerRef.current = viewer
      } catch {
        if (!cancelled) {
          setEngineError('3D viewer engine is unavailable in this environment.')
          viewerRef.current = null
        }
      }
    }

    void renderModel()

    return () => {
      cancelled = true
    }
  }, [canRender3d, conformer?.sdf, style])

  if (loading) {
    return <p className="text-sm text-slate-600">Loading conformer...</p>
  }

  if (error) {
    return <p className="text-sm text-rose-700">{error}</p>
  }

  if (!conformer || !canRender3d) {
    return (
      <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-600">
        {conformer?.message || '3D conformer is not available for this compound.'}
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <label htmlFor="viewer-style" className="text-xs font-semibold uppercase tracking-wide text-slate-500">
          Style
        </label>
        <select
          id="viewer-style"
          value={style}
          onChange={(event) => setStyle(event.target.value as ViewerStyle)}
          className="focus-ring rounded-lg border border-slate-300 bg-white px-2 py-1 text-sm"
        >
          <option value="stick">Stick</option>
          <option value="line">Line</option>
        </select>
        <button
          type="button"
          onClick={() => {
            viewerRef.current?.zoomTo()
            viewerRef.current?.render()
          }}
          className="focus-ring rounded-lg border border-slate-300 bg-white px-2 py-1 text-xs font-semibold text-slate-700 hover:bg-slate-100"
        >
          Reset view
        </button>
      </div>

      <div
        ref={containerRef}
        className="h-72 w-full rounded-xl border border-slate-200 bg-white"
        aria-label="3D conformer viewer"
      />

      {engineError && <p className="text-xs text-amber-700">{engineError}</p>}
    </div>
  )
}
