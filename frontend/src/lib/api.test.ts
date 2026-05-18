import { afterEach, describe, expect, it, vi } from 'vitest'
import type { AskRequest, IntakeRequest } from './types'

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

async function loadApi(baseUrl = 'http://localhost:8000') {
  vi.resetModules()
  vi.doMock('./env', () => ({
    getApiBaseUrl: () => baseUrl,
  }))
  return import('./api')
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('api client', () => {
  it('getHealth parses health response', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ status: 'ok', components: { api: { ok: true, detail: 'ok' } } }))
    vi.stubGlobal('fetch', fetchMock)

    const { getHealth } = await loadApi()
    const data = await getHealth()

    expect(data.status).toBe('ok')
    expect(data.components.api.ok).toBe(true)
    expect(fetchMock).toHaveBeenCalledWith('http://localhost:8000/health', expect.any(Object))
  })

  it('askQuestion sends POST /ask correctly', async () => {
    const payload = {
      query: 'Can I drive after drinking 180ml whisky?',
      answer: 'Do not drive.',
      risk_level: 'high',
      risk_summary: 'High impairment risk.',
      estimated_peak_bac: 0.08,
      estimated_time_to_sober_h: 10,
      estimated_time_to_peak_h: 1,
      driving_guidance: 'Do not drive.',
      continue_drinking_guidance: 'Do not drink more right now.',
      hydration_guidance: 'Sip water.',
      food_guidance: 'Food may help comfort.',
      medical_warning: 'Seek help for severe symptoms.',
      assumptions: [],
      missing_info: [],
      safe_for_display: true,
      blocked_request_type: null,
      advisor_fallback_used: false,
      synthesis_blocked: false,
      blocked_synthesis_reasons: [],
    }

    const fetchMock = vi.fn(async () => jsonResponse(payload))
    vi.stubGlobal('fetch', fetchMock)

    const { askQuestion } = await loadApi()
    const req: AskRequest = {
      query: payload.query,
      response_style: 'layman',
      debug: false,
    }

    const data = await askQuestion(req)

    expect(data.answer).toBe('Do not drive.')
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/ask',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify(req),
      }),
    )
  })

  it('submitIntake sends POST /intake correctly', async () => {
    const payload = {
      query: 'I am male 75 kg, I drank 200 ml vodka. How long until I sober up?',
      answer: 'Estimated time to sober is about 10 hours.',
      risk_level: 'moderate',
      risk_summary: 'Moderate risk.',
      estimated_peak_bac: 0.07,
      estimated_time_to_sober_h: 10,
      estimated_time_to_peak_h: 1,
      driving_guidance: 'Do not drive.',
      continue_drinking_guidance: 'Do not drink more right now.',
      hydration_guidance: 'Sip water.',
      food_guidance: 'Food may help comfort.',
      medical_warning: 'Seek help for severe symptoms.',
      assumptions: [],
      missing_info: [],
      safe_for_display: true,
      blocked_request_type: null,
      advisor_fallback_used: false,
      synthesis_blocked: false,
      blocked_synthesis_reasons: [],
    }

    const fetchMock = vi.fn(async () => jsonResponse(payload))
    vi.stubGlobal('fetch', fetchMock)

    const { submitIntake } = await loadApi()
    const req: IntakeRequest = {
      sex: 'male',
      weight_kg: 75,
      age: 30,
      fed_state: 'fed',
      drink_type: 'vodka',
      amount_ml: 200,
      duration_h: 1,
      goal: 'time_to_sober',
    }

    const data = await submitIntake(req)

    expect(data.risk_level).toBe('moderate')
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8000/intake',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify(req),
      }),
    )
  })

  it('non-2xx backend errors become user-friendly errors', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse(
        {
          detail: {
            error: true,
            message: 'ask pipeline failed',
            stage: 'ask',
          },
        },
        500,
      ),
    )
    vi.stubGlobal('fetch', fetchMock)

    const { askQuestion } = await loadApi()

    await expect(askQuestion({ query: 'test', response_style: 'layman', debug: false })).rejects.toThrow(
      'ask pipeline failed (stage: ask)',
    )
  })

  it('network failure becomes user-friendly error', async () => {
    const fetchMock = vi.fn(async () => {
      throw new Error('socket failure')
    })
    vi.stubGlobal('fetch', fetchMock)

    const { getHealth } = await loadApi()

    await expect(getHealth()).rejects.toThrow('Network error: unable to reach backend API.')
  })

  it('VITE_API_BASE_URL is respected', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ status: 'ok', components: { api: { ok: true, detail: 'ok' } } }))
    vi.stubGlobal('fetch', fetchMock)

    const { getHealth } = await loadApi('http://127.0.0.1:9000')
    await getHealth()

    expect(fetchMock).toHaveBeenCalledWith('http://127.0.0.1:9000/health', expect.any(Object))
  })
})
