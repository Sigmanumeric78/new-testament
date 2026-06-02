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
    'Educational insights only. Not medical advice. For personal understanding, not clinical decision-making.'

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
    () => (mode === 'ask' ? 'Ask a health or intake question' : 'Guided Health Intake'),
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

  const focusHealthCheck = () => {
    setMode('ask')
    window.setTimeout(() => document.getElementById('health-check-panel')?.scrollIntoView({ behavior: 'smooth' }), 0)
  }

  const navigateToExplorer = () => {
    window.history.pushState({}, '', '/explorer')
    window.dispatchEvent(new Event('app:navigate'))
  }

  return (
    <Layout sidebar={sidebar}>
      <div className="mx-auto max-w-6xl space-y-6">
        <section className="overflow-hidden rounded-2xl border border-emerald-100 bg-gradient-to-br from-white via-sky-50 to-emerald-50 p-6 shadow-sm md:p-8">
          <div className="grid gap-6 lg:grid-cols-[1.25fr_0.75fr] lg:items-center">
            <div>
              <p className="text-sm font-semibold uppercase text-brand-700">ZER0 GONDOGOL</p>
              <h2 className="mt-3 max-w-3xl text-3xl font-semibold text-slate-950 md:text-4xl">
                Understand food, ingredients, and intake risks with evidence-aware AI.
              </h2>
              <p className="mt-4 max-w-3xl text-base leading-7 text-slate-700">
                ZER0 GONDOGOL helps explain nutrition labels, ingredient signals, dietary tradeoffs, and intake-related
                risk estimates in simple language.
              </p>
              <div className="mt-5 flex flex-col gap-3 sm:flex-row">
                <button
                  type="button"
                  onClick={focusHealthCheck}
                  className="focus-ring rounded-lg bg-brand-600 px-4 py-3 text-sm font-semibold text-white hover:bg-brand-700"
                >
                  Start Health Check
                </button>
                <button
                  type="button"
                  onClick={navigateToExplorer}
                  className="focus-ring rounded-lg border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-800 hover:bg-sky-50"
                >
                  Explore Ingredients
                </button>
              </div>
              <p className="mt-4 text-xs font-medium text-slate-600">{safetyNotice}</p>
            </div>
            <div className="rounded-2xl border border-white/80 bg-white/75 p-5 shadow-sm">
              <p className="text-sm font-semibold text-slate-900">Food Health Intelligence</p>
              <div className="mt-4 grid gap-3">
                {[
                  ['Nutrition insight', 'Serving sizes, sodium, sugar, daily value signals.'],
                  ['Ingredient context', 'Compounds, additives, allergens, and possible relevance.'],
                  ['Risk signal', 'Current intake-safety estimates from the existing pipeline.'],
                ].map(([label, copy]) => (
                  <div key={label} className="rounded-xl border border-slate-200 bg-white p-3">
                    <p className="text-sm font-semibold text-slate-900">{label}</p>
                    <p className="mt-1 text-xs leading-5 text-slate-600">{copy}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-3 md:grid-cols-3">
          <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <h3 className="text-sm font-semibold text-slate-950">Nutrition & Label Intelligence</h3>
            <p className="mt-2 text-sm leading-6 text-slate-700">
              Decode nutrients, serving sizes, daily value signals, and product tradeoffs.
            </p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <h3 className="text-sm font-semibold text-slate-950">Ingredient & Additive Awareness</h3>
            <p className="mt-2 text-sm leading-6 text-slate-700">
              Understand ingredients, compounds, allergens, and possible risk signals.
            </p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <h3 className="text-sm font-semibold text-slate-950">Intake Safety Module</h3>
            <p className="mt-2 text-sm leading-6 text-slate-700">
              Estimate alcohol-related intake risk using the existing simulation and safety pipeline.
            </p>
          </div>
        </section>

        <section id="health-check-panel" className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-2xl font-semibold text-slate-950">{title}</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-700">
            Ask about food products, ingredients, nutrition tradeoffs, or current intake risk. The current live module
            supports alcohol/intake safety estimates while broader food-health intelligence is being added.
          </p>
        </section>

        <div className="inline-flex rounded-xl border border-slate-200 bg-white p-1 shadow-sm">
          <button
            type="button"
            onClick={() => setMode('ask')}
            className={`focus-ring rounded-lg px-3 py-2 text-sm font-medium ${
              mode === 'ask' ? 'bg-brand-600 text-white' : 'text-slate-700'
            }`}
          >
            Health Check
          </button>
          <button
            type="button"
            onClick={() => setMode('intake')}
            className={`focus-ring rounded-lg px-3 py-2 text-sm font-medium ${
              mode === 'intake' ? 'bg-brand-600 text-white' : 'text-slate-700'
            }`}
          >
            Guided Health Intake
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

        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-xl font-semibold text-slate-950">About ZER0 GONDOGOL</h2>
          <p className="mt-3 text-sm leading-6 text-slate-700">
            ZER0 GONDOGOL is an evidence-aware food and health intelligence platform. It combines structured nutrition
            data, ingredient knowledge, graph-based reasoning, semantic evidence retrieval, and safety-aware response
            generation to help users understand what they consume. The current live module focuses on alcohol and
            intake-safety estimation, while the platform is being expanded toward packaged food analysis, nutrition
            label decoding, ingredient explanations, meal analysis, and diet-goal compatibility.
          </p>
        </section>

        <section className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-xl font-semibold text-slate-950">How it works</h2>
            <ol className="mt-4 space-y-3 text-sm text-slate-700">
              {[
                'Ask or enter structured input',
                'System routes the question',
                'It calculates, retrieves, or explains using available evidence',
                'A safety layer checks the response',
                'You get a plain-language health insight',
              ].map((step, index) => (
                <li key={step} className="flex gap-3">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-xs font-semibold text-brand-700">
                    {index + 1}
                  </span>
                  <span>{step}</span>
                </li>
              ))}
            </ol>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-xl font-semibold text-slate-950">Built on evidence-aware reasoning</h2>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              {[
                ['Simulation', 'Current intake estimates use deterministic calculation modules rather than guessing.'],
                [
                  'Knowledge Graph',
                  'Relationships between compounds, physiology, nutrients, and risk signals can be represented as explainable paths.',
                ],
                ['Semantic Retrieval', 'Evidence and ingredient knowledge can be retrieved by meaning, not only keywords.'],
                ['Safety Layer', 'Responses are checked to avoid unsafe or unsupported health claims.'],
              ].map(([label, copy]) => (
                <div key={label} className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                  <h3 className="text-sm font-semibold text-slate-950">{label}</h3>
                  <p className="mt-2 text-sm leading-6 text-slate-700">{copy}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>
    </Layout>
  )
}
