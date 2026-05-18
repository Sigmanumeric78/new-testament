import { useCallback, useEffect, useMemo, useState } from 'react'
import Layout from '../components/Layout'
import Sidebar from '../components/Sidebar'
import QueryComposer from '../components/QueryComposer'
import IntakeForm from '../components/IntakeForm'
import ResultPanel from '../components/ResultPanel'
import LoadingState from '../components/LoadingState'
import ErrorState from '../components/ErrorState'
import { askQuestion, getHealth, submitIntake } from '../lib/api'
import { sanitizeResponseForDisplay } from '../lib/format'
import type { AskResponse, HealthResponse, IntakeRequest, ResponseStyle } from '../lib/types'

export default function AskPage() {
  const [mode, setMode] = useState<'ask' | 'intake'>('ask')
  const [query, setQuery] = useState('')
  const [responseStyle, setResponseStyle] = useState<ResponseStyle>('layman')
  const [debug, setDebug] = useState(false)

  const [result, setResult] = useState<AskResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthLoading, setHealthLoading] = useState(true)

  const safetyNotice =
    'Estimates only. Not medical or legal advice. Never use this to decide whether it is safe to drive.'

  const handleAsk = useCallback(async () => {
    if (!query.trim()) return
    setLoading(true)
    setError('')
    try {
      const response = await askQuestion({
        query: query.trim(),
        response_style: responseStyle,
        debug,
      })
      setResult(sanitizeResponseForDisplay(response))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed.')
    } finally {
      setLoading(false)
    }
  }, [query, responseStyle, debug])

  const handleIntake = useCallback(async (payload: IntakeRequest) => {
    setLoading(true)
    setError('')
    try {
      const response = await submitIntake(payload)
      setResult(sanitizeResponseForDisplay(response))
      setMode('ask')
      setQuery(response.query)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Intake request failed.')
    } finally {
      setLoading(false)
    }
  }, [])

  const refreshHealth = useCallback(async () => {
    setHealthLoading(true)
    try {
      const response = await getHealth()
      setHealth(response)
    } catch {
      setHealth({ status: 'error', components: {} })
    } finally {
      setHealthLoading(false)
    }
  }, [])

  useEffect(() => {
    void refreshHealth()
  }, [refreshHealth])

  const title = useMemo(
    () => (mode === 'ask' ? 'Ask a question' : 'Guided intake'),
    [mode],
  )

  function resetFlow() {
    setResult(null)
    setError('')
    setQuery('')
    setMode('ask')
  }

  const sidebar = (
    <Sidebar
      mode={mode}
      onModeChange={setMode}
      onNewCheck={resetFlow}
      onSelectPrompt={(prompt) => {
        setQuery(prompt)
        setMode('ask')
      }}
      health={health}
      healthLoading={healthLoading}
    />
  )

  return (
    <Layout sidebar={sidebar}>
      <div className="mx-auto max-w-5xl space-y-5">
        <section className="rounded-2xl bg-gradient-to-r from-brand-50 to-white p-6 shadow-sm">
          <h2 className="text-2xl font-semibold tracking-tight text-slate-900">{title}</h2>
          <p className="mt-2 text-sm text-slate-700">{safetyNotice}</p>
        </section>

        <div className="inline-flex rounded-xl border border-slate-200 bg-white p-1 shadow-sm">
          <button
            type="button"
            onClick={() => setMode('ask')}
            className={`focus-ring rounded-lg px-3 py-2 text-sm font-medium ${
              mode === 'ask' ? 'bg-brand-600 text-white' : 'text-slate-700'
            }`}
          >
            Ask a question
          </button>
          <button
            type="button"
            onClick={() => setMode('intake')}
            className={`focus-ring rounded-lg px-3 py-2 text-sm font-medium ${
              mode === 'intake' ? 'bg-brand-600 text-white' : 'text-slate-700'
            }`}
          >
            Guided intake
          </button>
        </div>

        {mode === 'ask' ? (
          <QueryComposer
            query={query}
            responseStyle={responseStyle}
            debug={debug}
            loading={loading}
            onQueryChange={setQuery}
            onResponseStyleChange={setResponseStyle}
            onDebugChange={setDebug}
            onSubmit={() => void handleAsk()}
          />
        ) : (
          <IntakeForm loading={loading} onSubmit={(payload) => void handleIntake(payload)} />
        )}

        {loading && <LoadingState />}
        {!loading && error && <ErrorState message={error} />}
        {!loading && !error && result && <ResultPanel result={result} debugEnabled={debug} />}
      </div>
    </Layout>
  )
}
