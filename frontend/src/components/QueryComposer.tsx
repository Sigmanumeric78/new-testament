import type { FormEvent } from 'react'
import type { ResponseStyle } from '../lib/types'

interface QueryComposerProps {
  query: string
  responseStyle: ResponseStyle
  debug: boolean
  loading: boolean
  onQueryChange: (value: string) => void
  onResponseStyleChange: (value: ResponseStyle) => void
  onDebugChange: (value: boolean) => void
  onSubmit: () => void
}

export default function QueryComposer({
  query,
  responseStyle,
  debug,
  loading,
  onQueryChange,
  onResponseStyleChange,
  onDebugChange,
  onSubmit,
}: QueryComposerProps) {
  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    onSubmit()
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <label htmlFor="query" className="text-sm font-semibold text-slate-800">
        Ask about current alcohol risk
      </label>
      <textarea
        id="query"
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
        rows={5}
        className="focus-ring w-full rounded-xl border border-slate-300 p-3 text-sm leading-relaxed"
        placeholder="Describe what you drank, when, and what you need to decide."
        required
      />

      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-sm font-medium text-slate-700" htmlFor="style-select">
            Response style
          </label>
          <select
            id="style-select"
            value={responseStyle}
            onChange={(event) => onResponseStyleChange(event.target.value as ResponseStyle)}
            className="focus-ring rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm"
            disabled={loading}
          >
            <option value="layman">Simple</option>
            <option value="technical">Detailed</option>
            <option value="scientific">Scientific</option>
          </select>

          <label className="inline-flex items-center gap-2 text-sm text-slate-700">
            <input
              type="checkbox"
              checked={debug}
              onChange={(event) => onDebugChange(event.target.checked)}
              className="h-4 w-4 rounded border-slate-300"
              disabled={loading}
            />
            Debug
          </label>
        </div>

        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="focus-ring rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? 'Checking...' : 'Run check'}
        </button>
      </div>
    </form>
  )
}
