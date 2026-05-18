import { getApiBaseUrl } from './env'
import type {
  ChemicalConformerResponse,
  ChemicalDetail,
  ChemicalListResponse,
  ChemicalSearchParams,
} from './chemicalTypes'

const API_BASE_URL = getApiBaseUrl()

function buildUrl(path: string): string {
  return `${API_BASE_URL}${path}`
}

async function parseError(response: Response): Promise<never> {
  let message = `Request failed with status ${response.status}`
  try {
    const body = (await response.json()) as { detail?: { message?: string; stage?: string }; message?: string }
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

async function requestJson<T>(path: string): Promise<T> {
  let response: Response
  try {
    response = await fetch(buildUrl(path), {
      headers: { 'Content-Type': 'application/json' },
    })
  } catch {
    throw new Error('Network error: unable to reach backend API.')
  }

  if (!response.ok) {
    await parseError(response)
  }

  return (await response.json()) as T
}

function toQueryString(params: ChemicalSearchParams): string {
  const query = new URLSearchParams()
  if (params.q?.trim()) query.set('q', params.q.trim())
  if (params.chemical_class?.trim()) query.set('chemical_class', params.chemical_class.trim())
  if (typeof params.has_3d === 'boolean') query.set('has_3d', String(params.has_3d))
  if (typeof params.limit === 'number') query.set('limit', String(params.limit))
  if (typeof params.offset === 'number') query.set('offset', String(params.offset))
  const encoded = query.toString()
  return encoded ? `?${encoded}` : ''
}

export async function listChemicals(params: ChemicalSearchParams = {}): Promise<ChemicalListResponse> {
  return requestJson<ChemicalListResponse>(`/chemicals${toQueryString(params)}`)
}

export async function getChemicalDetail(compoundId: string): Promise<ChemicalDetail> {
  const safeId = encodeURIComponent(compoundId)
  return requestJson<ChemicalDetail>(`/chemicals/${safeId}`)
}

export async function getChemicalConformer(compoundId: string): Promise<ChemicalConformerResponse> {
  const safeId = encodeURIComponent(compoundId)
  return requestJson<ChemicalConformerResponse>(`/chemicals/${safeId}/conformer`)
}
