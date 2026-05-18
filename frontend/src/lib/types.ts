export type ResponseStyle = 'layman' | 'technical' | 'scientific'

export interface AskRequest {
  query: string
  response_style?: ResponseStyle | null
  debug?: boolean
}

export interface AskResponse {
  query: string
  answer: string
  risk_level: string
  risk_summary: string
  estimated_peak_bac: number | null
  estimated_time_to_sober_h: number | null
  estimated_time_to_peak_h: number | null
  ethanol_dose_g: number | null
  drink_abv_percent: number | null
  drink_volume_ml: number | null
  legal_limit_reference_bac: number | null
  is_estimated_below_0_08: boolean | null
  estimated_total_volume_for_0_08_ml: number | null
  estimated_additional_volume_to_0_08_ml: number | null
  threshold_explanation: string | null
  beverage_type: string | null
  likely_compounds: string[]
  body_processes: Array<{
    stage: string
    plain_explanation: string
    technical_explanation: string | null
  }>
  detail_level: ResponseStyle
  driving_guidance: string
  continue_drinking_guidance: string
  hydration_guidance: string
  food_guidance: string
  medical_warning: string
  assumptions: string[]
  missing_info: string[]
  safe_for_display: boolean
  blocked_request_type: string | null
  advisor_fallback_used: boolean
  synthesis_blocked: boolean
  blocked_synthesis_reasons: string[]
  debug?: Record<string, unknown>
}

export interface IntakeRequest {
  sex: 'male' | 'female' | 'unknown'
  weight_kg: number
  age: number | null
  fed_state: 'fed' | 'fasted' | 'unknown'
  drink_type: string
  amount_ml: number
  duration_h: number | null
  goal: 'drive_check' | 'time_to_sober' | 'hangover_risk' | 'should_i_keep_drinking'
}

export interface HealthComponent {
  ok: boolean
  detail: string
  missing_required_count?: number | null
  missing_required?: string[] | null
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'error'
  components: Record<string, HealthComponent>
}

export interface BackendErrorShape {
  error?: boolean
  message?: string
  stage?: string
}
