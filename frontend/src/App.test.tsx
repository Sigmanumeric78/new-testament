import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

vi.mock('./lib/api', () => ({
  getHealth: vi.fn(async () => ({ status: 'ok', components: { api: { ok: true, detail: 'ok' } } })),
  askQuestion: vi.fn(async () => ({
    query: 'test',
    answer: 'For your situation, you should not drink more right now.',
    risk_level: 'moderate',
    risk_summary: 'moderate risk',
    estimated_peak_bac: 0.05,
    estimated_time_to_sober_h: 4,
    estimated_time_to_peak_h: 1,
    ethanol_dose_g: 20,
    drink_abv_percent: 40,
    drink_volume_ml: 60,
    beverage_type: 'vodka',
    likely_compounds: [],
    body_processes: [],
    detail_level: 'layman',
    driving_guidance: 'Do not drive.',
    continue_drinking_guidance: 'Do not drink more.',
    hydration_guidance: 'Sip water.',
    food_guidance: 'Food may help comfort.',
    medical_warning: 'Seek help for severe symptoms.',
    assumptions: [],
    missing_info: [],
    safe_for_display: true,
    advisor_fallback_used: false,
    synthesis_blocked: false,
    blocked_synthesis_reasons: [],
    blocked_request_type: null,
  })),
  submitIntake: vi.fn(async () => ({
    query: 'intake',
    answer: 'intake answer',
    risk_level: 'low',
    risk_summary: 'low risk',
    estimated_peak_bac: 0.01,
    estimated_time_to_sober_h: 1,
    estimated_time_to_peak_h: 1,
    ethanol_dose_g: 5,
    drink_abv_percent: 5,
    drink_volume_ml: 100,
    beverage_type: 'beer',
    likely_compounds: [],
    body_processes: [],
    detail_level: 'layman',
    driving_guidance: 'Do not drive.',
    continue_drinking_guidance: 'Do not drink more.',
    hydration_guidance: 'Sip water.',
    food_guidance: 'Food may help comfort.',
    medical_warning: 'Seek help for severe symptoms.',
    assumptions: [],
    missing_info: [],
    safe_for_display: true,
    advisor_fallback_used: false,
    synthesis_blocked: false,
    blocked_synthesis_reasons: [],
    blocked_request_type: null,
  })),
}))

const listChemicalsMock = vi.fn(async () => ({
  items: [],
  total: 0,
  limit: 24,
  offset: 0,
}))

vi.mock('./lib/chemicalApi', () => ({
  listChemicals: (...args: unknown[]) => listChemicalsMock(...args),
  getChemicalDetail: vi.fn(async () => {
    throw new Error('no detail')
  }),
  getChemicalConformer: vi.fn(async () => {
    throw new Error('no conformer')
  }),
}))

beforeEach(() => {
  window.history.pushState({}, '', '/')
})

describe('App navigation', () => {
  it('navigates between Ask and Chemical Explorer', async () => {
    render(<App />)

    expect(screen.getByRole('heading', { name: /ask a question/i, level: 2 })).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: /chemical explorer/i }))

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /chemical explorer/i, level: 1 })).toBeInTheDocument()
    })

    await userEvent.click(screen.getByRole('button', { name: /^ask$/i }))

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /ask a question/i, level: 2 })).toBeInTheDocument()
    })
  })
})
