import type { AskResponse } from './types'

const BANNED_TERMS = [
  'pbpk',
  'neo4j',
  'weaviate',
  'causal path',
  'graph',
  'vector',
  'embedding',
  'simulator fallback',
  'confidence score',
]

export function toTitleCase(text: string): string {
  return text
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((p) => p[0].toUpperCase() + p.slice(1).toLowerCase())
    .join(' ')
}

export function formatBac(value: number | null): string {
  if (value == null || Number.isNaN(value)) return 'Unavailable'
  return `about ${value.toFixed(2)}%`
}

export function formatHours(value: number | null): string {
  if (value == null || Number.isNaN(value)) return 'Unavailable'
  const rounded = Math.round(value)
  return rounded === 1 ? 'about 1 hour' : `about ${rounded} hours`
}

export function formatMl(value: number | null): string {
  if (value == null || Number.isNaN(value)) return 'Unavailable'
  const rounded = Math.round(value / 10) * 10
  return `about ${rounded} ml`
}

export function healthLabel(status: string): 'Healthy' | 'Degraded' | 'Offline' {
  if (status === 'ok') return 'Healthy'
  if (status === 'degraded') return 'Degraded'
  return 'Offline'
}

export function sanitizeUserFacingText(input: string): string {
  let out = input
  for (const term of BANNED_TERMS) {
    const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    out = out.replace(new RegExp(`\\b${escaped}\\b`, 'gi'), '')
  }
  return out.replace(/\s+/g, ' ').trim()
}

export function sanitizeResponseForDisplay(response: AskResponse): AskResponse {
  return {
    ...response,
    answer: sanitizeUserFacingText(response.answer),
    risk_summary: sanitizeUserFacingText(response.risk_summary),
    driving_guidance: sanitizeUserFacingText(response.driving_guidance),
    continue_drinking_guidance: sanitizeUserFacingText(response.continue_drinking_guidance),
    hydration_guidance: sanitizeUserFacingText(response.hydration_guidance),
    food_guidance: sanitizeUserFacingText(response.food_guidance),
    medical_warning: sanitizeUserFacingText(response.medical_warning),
    assumptions: response.assumptions.map(sanitizeUserFacingText),
    missing_info: response.missing_info.map(sanitizeUserFacingText),
    blocked_synthesis_reasons: response.blocked_synthesis_reasons.map(sanitizeUserFacingText),
    threshold_explanation: response.threshold_explanation
      ? sanitizeUserFacingText(response.threshold_explanation)
      : null,
    likely_compounds: response.likely_compounds.map(sanitizeUserFacingText).filter(Boolean),
    body_processes: response.body_processes.map((process) => ({
      stage: sanitizeUserFacingText(process.stage),
      plain_explanation: sanitizeUserFacingText(process.plain_explanation),
      technical_explanation: process.technical_explanation
        ? sanitizeUserFacingText(process.technical_explanation)
        : null,
    })),
  }
}
