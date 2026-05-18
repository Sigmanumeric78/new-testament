import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ResultPanel from './ResultPanel'
import type { AskResponse } from '../lib/types'

function makeResult(overrides: Partial<AskResponse> = {}): AskResponse {
  return {
    query: 'Can I drive?',
    answer: 'PBPK and Neo4j and Weaviate with causal path graph vector embedding simulator fallback confidence score',
    risk_level: 'high',
    risk_summary: 'High impairment risk.',
    estimated_peak_bac: 0.08,
    estimated_time_to_sober_h: 11,
    estimated_time_to_peak_h: 1,
    ethanol_dose_g: 56.8,
    drink_abv_percent: 40,
    drink_volume_ml: 180,
    legal_limit_reference_bac: 0.08,
    is_estimated_below_0_08: false,
    estimated_total_volume_for_0_08_ml: 160,
    estimated_additional_volume_to_0_08_ml: 0,
    threshold_explanation:
      'Under the same assumptions, a total intake around 160 ml of 40% whisky could push the estimate near or above 0.08%. This is a risk threshold estimate, not a recommendation to drink that amount.',
    beverage_type: 'whisky',
    likely_compounds: ['ethanol', 'water', 'fusel alcohols'],
    body_processes: [
      {
        stage: 'absorption',
        plain_explanation: 'Alcohol moves into your blood.',
        technical_explanation: 'Absorption is faster on an empty stomach.',
      },
    ],
    detail_level: 'layman',
    driving_guidance: 'Do not drive right now.',
    continue_drinking_guidance: 'Do not drink more right now.',
    hydration_guidance: 'Sip water. Water does not speed alcohol clearance.',
    food_guidance: 'Food may slow absorption but will not sober you instantly.',
    medical_warning: 'Seek help for severe symptoms.',
    assumptions: ['Assumed adult age because age was not provided.'],
    missing_info: ['Age'],
    safe_for_display: true,
    blocked_request_type: null,
    advisor_fallback_used: false,
    synthesis_blocked: false,
    blocked_synthesis_reasons: [],
    debug: {
      route: { intent: 'drive_check' },
      orchestration: { modules: ['pbpk', 'neo4j'] },
    },
    ...overrides,
  }
}

describe('ResultPanel', () => {
  it('hides internal terms in default output', () => {
    render(<ResultPanel result={makeResult()} debugEnabled={false} />)

    const banned = [
      'PBPK',
      'Neo4j',
      'Weaviate',
      'causal path',
      'graph',
      'vector',
      'embedding',
      'simulator fallback',
      'confidence score',
    ]

    for (const term of banned) {
      expect(screen.queryByText(new RegExp(term, 'i'))).not.toBeInTheDocument()
    }
  })

  it('shows high risk prominently', () => {
    render(<ResultPanel result={makeResult({ risk_level: 'high' })} debugEnabled={false} />)
    expect(screen.getByText(/risk level: high/i)).toBeInTheDocument()
  })

  it('shows emergency warning prominently for possible_medical_emergency', () => {
    render(
      <ResultPanel
        result={makeResult({
          risk_level: 'possible_medical_emergency',
          medical_warning: 'Seek emergency help immediately.',
        })}
        debugEnabled={false}
      />,
    )

    expect(screen.getByText(/risk level: possible medical emergency/i)).toBeInTheDocument()
    expect(screen.getByText(/emergency warning: seek immediate medical help/i)).toBeInTheDocument()
  })

  it('renders assumptions and missing_info', () => {
    render(<ResultPanel result={makeResult()} debugEnabled={false} />)
    expect(screen.getByText('Assumptions')).toBeInTheDocument()
    expect(screen.getByText('Missing info')).toBeInTheDocument()
    expect(screen.getByText(/assumed adult age/i)).toBeInTheDocument()
    expect(screen.getByText('Age')).toBeInTheDocument()
  })

  it('renders fallback metadata with user-safe wording only', () => {
    render(
      <ResultPanel
        result={makeResult({
          advisor_fallback_used: true,
          synthesis_blocked: true,
          blocked_synthesis_reasons: ['Grounding score below threshold (0.70)'],
        })}
        debugEnabled={false}
      />,
    )

    expect(screen.getByText(/conservative safety fallback response was used/i)).toBeInTheDocument()
    expect(screen.queryByText(/grounding score below threshold/i)).not.toBeInTheDocument()
  })

  it('keeps debug payload hidden by default', () => {
    render(<ResultPanel result={makeResult()} debugEnabled={false} />)
    expect(screen.queryByText(/debug details/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/"route"/i)).not.toBeInTheDocument()
  })

  it('shows debug payload only when debug is enabled', () => {
    render(<ResultPanel result={makeResult()} debugEnabled={true} />)
    expect(screen.getByText(/debug details/i)).toBeInTheDocument()
    expect(screen.getByText(/"route"/i)).toBeInTheDocument()
  })

  it('renders scientific sections when detail mode is scientific', () => {
    render(
      <ResultPanel
        result={makeResult({
          detail_level: 'scientific',
        })}
        debugEnabled={false}
      />,
    )
    expect(screen.getByText(/mode: scientific/i)).toBeInTheDocument()
    expect(screen.getByText(/estimated alcohol dose/i)).toBeInTheDocument()
    expect(screen.getByText(/drink chemistry/i)).toBeInTheDocument()
    expect(screen.getByText(/what is happening in your body/i)).toBeInTheDocument()
    expect(screen.getByText(/0.08% threshold context/i)).toBeInTheDocument()
  })

  it('does not render chemistry/process sections in layman mode', () => {
    render(
      <ResultPanel
        result={makeResult({
          detail_level: 'layman',
          likely_compounds: [],
          body_processes: [],
          ethanol_dose_g: null,
        })}
        debugEnabled={false}
      />,
    )
    expect(screen.getByText(/mode: simple/i)).toBeInTheDocument()
    expect(screen.queryByText(/estimated alcohol dose/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/drink chemistry/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/what is happening in your body/i)).not.toBeInTheDocument()
  })

  it('renders threshold context section when threshold fields exist', () => {
    render(<ResultPanel result={makeResult()} debugEnabled={false} />)
    expect(screen.getByText(/0.08% threshold context/i)).toBeInTheDocument()
    expect(screen.getByText(/this is not a recommendation or a safe drinking limit/i)).toBeInTheDocument()
    expect(screen.getByText(/reference threshold: 0.08%/i)).toBeInTheDocument()
  })

  it('does not render threshold section when threshold fields are null', () => {
    render(
      <ResultPanel
        result={makeResult({
          estimated_peak_bac: null,
          legal_limit_reference_bac: null,
          estimated_total_volume_for_0_08_ml: null,
          threshold_explanation: null,
        })}
        debugEnabled={false}
      />,
    )
    expect(screen.queryByText(/0.08% threshold context/i)).not.toBeInTheDocument()
  })
})
