import { useMemo, useState, type FormEvent } from 'react'
import type { IntakeRequest } from '../lib/types'

interface IntakeFormProps {
  loading: boolean
  onSubmit: (payload: IntakeRequest) => void
}

type IntakeState = {
  sex: IntakeRequest['sex']
  weight_kg: string
  age: string
  fed_state: IntakeRequest['fed_state']
  drink_type: string
  amount_ml: string
  duration_h: string
  goal: IntakeRequest['goal']
}

const INITIAL_STATE: IntakeState = {
  sex: 'unknown',
  weight_kg: '',
  age: '',
  fed_state: 'unknown',
  drink_type: '',
  amount_ml: '',
  duration_h: '',
  goal: 'time_to_sober',
}

export default function IntakeForm({ loading, onSubmit }: IntakeFormProps) {
  const [state, setState] = useState<IntakeState>(INITIAL_STATE)
  const [error, setError] = useState('')

  const isValid = useMemo(() => {
    const weight = Number(state.weight_kg)
    const amount = Number(state.amount_ml)
    const duration = state.duration_h ? Number(state.duration_h) : null
    return (
      state.drink_type.trim().length > 0 &&
      Number.isFinite(weight) &&
      weight > 20 &&
      weight < 250 &&
      Number.isFinite(amount) &&
      amount > 0 &&
      amount < 3000 &&
      (duration == null || (Number.isFinite(duration) && duration > 0 && duration < 24))
    )
  }, [state])

  function update<K extends keyof IntakeState>(key: K, value: IntakeState[K]) {
    setState((current) => ({ ...current, [key]: value }))
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    setError('')

    const weight = Number(state.weight_kg)
    const amount = Number(state.amount_ml)
    const duration = state.duration_h ? Number(state.duration_h) : null
    const age = state.age ? Number(state.age) : null

    if (!isValid) {
      setError('Please provide valid intake values before submitting.')
      return
    }

    onSubmit({
      sex: state.sex,
      weight_kg: weight,
      age: age && age >= 18 ? age : null,
      fed_state: state.fed_state,
      drink_type: state.drink_type.trim(),
      amount_ml: amount,
      duration_h: duration,
      goal: state.goal,
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div>
        <h3 className="text-base font-semibold text-slate-950">Guided Health Intake</h3>
        <p className="mt-2 text-sm leading-6 text-slate-700">
          Use structured inputs for the current intake-safety module. More guided flows for meals, food labels, and
          product analysis are coming soon.
        </p>
      </div>

      <fieldset className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
        <legend className="px-1 text-sm font-semibold text-slate-900">Your profile</legend>
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          <label className="text-sm text-slate-700">
            Sex <span className="text-rose-600" aria-hidden="true">*</span>
            <select className="focus-ring mt-1 w-full rounded-lg border border-slate-300 bg-white p-2" value={state.sex} onChange={(e) => update('sex', e.target.value as IntakeRequest['sex'])}>
              <option value="unknown">unknown</option>
              <option value="male">male</option>
              <option value="female">female</option>
            </select>
          </label>

          <label className="text-sm text-slate-700">
            Weight (kg) <span className="text-rose-600" aria-hidden="true">*</span>
            <input className="focus-ring mt-1 w-full rounded-lg border border-slate-300 bg-white p-2" type="number" value={state.weight_kg} onChange={(e) => update('weight_kg', e.target.value)} required />
          </label>

          <label className="text-sm text-slate-700">
            Age (optional)
            <input className="focus-ring mt-1 w-full rounded-lg border border-slate-300 bg-white p-2" type="number" value={state.age} onChange={(e) => update('age', e.target.value)} />
          </label>

          <label className="text-sm text-slate-700">
            Fed state <span className="text-rose-600" aria-hidden="true">*</span>
            <select className="focus-ring mt-1 w-full rounded-lg border border-slate-300 bg-white p-2" value={state.fed_state} onChange={(e) => update('fed_state', e.target.value as IntakeRequest['fed_state'])}>
              <option value="unknown">unknown</option>
              <option value="fed">fed</option>
              <option value="fasted">fasted</option>
            </select>
          </label>
        </div>
      </fieldset>

      <fieldset className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
        <legend className="px-1 text-sm font-semibold text-slate-900">Current intake</legend>
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          <label className="text-sm text-slate-700">
            Drink type <span className="text-rose-600" aria-hidden="true">*</span>
            <input className="focus-ring mt-1 w-full rounded-lg border border-slate-300 bg-white p-2" value={state.drink_type} onChange={(e) => update('drink_type', e.target.value)} required />
          </label>

          <label className="text-sm text-slate-700">
            Amount (ml) <span className="text-rose-600" aria-hidden="true">*</span>
            <input className="focus-ring mt-1 w-full rounded-lg border border-slate-300 bg-white p-2" type="number" value={state.amount_ml} onChange={(e) => update('amount_ml', e.target.value)} required />
          </label>

          <label className="text-sm text-slate-700">
            Duration (hours, optional)
            <input className="focus-ring mt-1 w-full rounded-lg border border-slate-300 bg-white p-2" type="number" step="0.1" value={state.duration_h} onChange={(e) => update('duration_h', e.target.value)} />
          </label>
        </div>
      </fieldset>

      <fieldset className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
        <legend className="px-1 text-sm font-semibold text-slate-900">What do you want to understand?</legend>
        <label className="mt-3 block text-sm text-slate-700">
          Goal
          <select className="focus-ring mt-1 w-full rounded-lg border border-slate-300 bg-white p-2" value={state.goal} onChange={(e) => update('goal', e.target.value as IntakeRequest['goal'])}>
            <option value="drive_check">driving risk signal</option>
            <option value="time_to_sober">estimated time to sober</option>
            <option value="hangover_risk">next-day discomfort risk</option>
            <option value="should_i_keep_drinking">continue intake guidance</option>
          </select>
        </label>
      </fieldset>

      {error && <p className="text-sm text-rose-700">{error}</p>}
      {!isValid && (
        <p className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          Some estimates are unavailable because required fields are missing. Provide sex, weight, fed state, drink type,
          amount, and duration for a more complete estimate.
        </p>
      )}

      <button
        type="submit"
        disabled={loading || !isValid}
        className="focus-ring rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {loading ? 'Checking...' : 'Submit intake'}
      </button>
    </form>
  )
}
