import type { AskRequest, AskResponse, BackendErrorShape, HealthResponse, IntakeRequest } from './types'
import { getApiBaseUrl } from './env'

const API_BASE_URL = getApiBaseUrl()

function buildUrl(path: string): string {
  return `${API_BASE_URL}${path}`
}

async function parseError(response: Response): Promise<never> {
  let message = `Request failed with status ${response.status}`
  try {
    const body = (await response.json()) as { detail?: BackendErrorShape; message?: string }
    if (body?.detail?.message) {
      message = `${body.detail.message}${body.detail.stage ? ` (stage: ${body.detail.stage})` : ''}`
    } else if (body?.message) {
      message = body.message
    }
  } catch {
    // keep default
  }
  throw new Error(message)
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(buildUrl(path), {
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers ?? {}),
      },
      ...init,
    })
  } catch {
    throw new Error('Network error: unable to reach backend API.')
  }

  if (!response.ok) {
    await parseError(response)
  }

  return (await response.json()) as T
}

export async function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>('/health')
}

export async function askQuestion(payload: AskRequest): Promise<AskResponse> {
  return requestJson<AskResponse>('/ask', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function submitIntake(payload: IntakeRequest): Promise<AskResponse> {
  return requestJson<AskResponse>('/intake', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}
