import type { AskResponse } from '../lib/types'
import { formatBac, formatMl, sanitizeResponseForDisplay } from '../lib/format'
import DebugDrawer from './DebugDrawer'
import EstimateCards from './EstimateCards'
import RiskCard from './RiskCard'
import SafetyGuidance from './SafetyGuidance'

interface ResultPanelProps {
  result: AskResponse
  debugEnabled: boolean
}

export default function ResultPanel({ result, debugEnabled }: ResultPanelProps) {
  const safeResult = sanitizeResponseForDisplay(result)
  const showDetailedSections =
    safeResult.detail_level !== 'layman' &&
    (safeResult.ethanol_dose_g != null ||
      safeResult.likely_compounds.length > 0 ||
      safeResult.body_processes.length > 0)
  const modeLabel =
    safeResult.detail_level === 'scientific'
      ? 'Scientific'
      : safeResult.detail_level === 'technical'
        ? 'Detailed'
        : 'Simple'
  const showThresholdSection =
    safeResult.estimated_peak_bac != null &&
    safeResult.legal_limit_reference_bac != null &&
    safeResult.estimated_total_volume_for_0_08_ml != null

  return (
    <section className="space-y-4" aria-live="polite">
      <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-lg font-semibold text-slate-950">Summary Answer</h3>
          <p className="rounded-full bg-sky-50 px-3 py-1 text-xs font-semibold text-sky-800">Mode: {modeLabel}</p>
        </div>
        <p className="mt-2 text-sm leading-relaxed text-slate-900">{safeResult.answer}</p>
      </div>

      {safeResult.advisor_fallback_used && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
          Conservative safety response shown because some inputs or evidence were incomplete.
        </div>
      )}

      <div>
        <h3 className="mb-2 text-sm font-semibold uppercase text-slate-500">Risk Level</h3>
        <RiskCard riskLevel={safeResult.risk_level} summary={safeResult.risk_summary} />
      </div>

      <div>
        <h3 className="mb-2 text-sm font-semibold uppercase text-slate-500">Key Estimates</h3>
        <EstimateCards
          peakBac={safeResult.estimated_peak_bac}
          timeToSober={safeResult.estimated_time_to_sober_h}
          timeToPeak={safeResult.estimated_time_to_peak_h}
        />
      </div>

      {showThresholdSection && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
          <h4 className="text-sm font-semibold text-amber-900">Risk threshold context</h4>
          <ul className="mt-2 space-y-1 text-sm text-amber-900">
            <li>Current estimate: {formatBac(safeResult.estimated_peak_bac)}</li>
            <li>Reference threshold: {safeResult.legal_limit_reference_bac?.toFixed(2)}%</li>
            <li>
              Approx total same-drink volume near threshold:{' '}
              {formatMl(safeResult.estimated_total_volume_for_0_08_ml)}
            </li>
          </ul>
          {safeResult.threshold_explanation ? (
            <p className="mt-2 text-sm text-amber-900">{safeResult.threshold_explanation}</p>
          ) : null}
          <p className="mt-2 text-xs font-semibold text-amber-950">
            This is not a recommendation or a safe drinking limit.
          </p>
        </div>
      )}

      {showDetailedSections && (
        <div className="grid gap-3 md:grid-cols-2">
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <h4 className="text-sm font-semibold text-slate-900">Estimated alcohol dose</h4>
            <p className="mt-2 text-sm text-slate-700">
              {safeResult.ethanol_dose_g != null ? `about ${safeResult.ethanol_dose_g.toFixed(1)} g ethanol` : 'Unavailable'}
            </p>
          </div>
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <h4 className="text-sm font-semibold text-slate-900">Drink chemistry</h4>
            {safeResult.likely_compounds.length > 0 ? (
              <ul className="mt-2 list-disc pl-4 text-sm text-slate-700">
                {safeResult.likely_compounds.map((compound) => (
                  <li key={compound}>{compound}</li>
                ))}
              </ul>
            ) : (
              <p className="mt-2 text-sm text-slate-600">Unavailable.</p>
            )}
          </div>
        </div>
      )}

      {showDetailedSections && safeResult.body_processes.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <h4 className="text-sm font-semibold text-slate-900">Why this matters</h4>
          <ul className="mt-2 space-y-2 text-sm text-slate-700">
            {safeResult.body_processes.map((process) => (
              <li key={process.stage}>
                <span className="font-semibold capitalize">{process.stage}:</span> {process.plain_explanation}
                {safeResult.detail_level !== 'layman' && process.technical_explanation ? ` ${process.technical_explanation}` : ''}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <h3 className="mb-2 text-sm font-semibold uppercase text-slate-500">Safety Guidance</h3>
        <SafetyGuidance
          drivingGuidance={safeResult.driving_guidance}
          continueGuidance={safeResult.continue_drinking_guidance}
          hydrationGuidance={safeResult.hydration_guidance}
          foodGuidance={safeResult.food_guidance}
          medicalWarning={safeResult.medical_warning}
        />
      </div>

      <div>
        <h3 className="mb-2 text-sm font-semibold uppercase text-slate-500">Assumptions & Missing Info</h3>
        <div className="grid gap-3 md:grid-cols-2">
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <h4 className="text-sm font-semibold text-slate-900">Assumptions</h4>
            {safeResult.assumptions.length > 0 ? (
              <ul className="mt-2 list-disc pl-4 text-sm text-slate-700">
                {safeResult.assumptions.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            ) : (
              <p className="mt-2 text-sm text-slate-600">None.</p>
            )}
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <h4 className="text-sm font-semibold text-slate-900">Missing info</h4>
            {safeResult.missing_info.length > 0 ? (
              <>
                <p className="mt-2 text-sm text-slate-700">
                  Some estimates are unavailable because required fields are missing. Provide sex, weight, fed state,
                  drink type, amount, and duration for a more complete estimate.
                </p>
                <ul className="mt-2 list-disc pl-4 text-sm text-slate-700">
                  {safeResult.missing_info.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="mt-2 text-sm text-slate-600">None.</p>
            )}
          </div>
        </div>
      </div>

      <DebugDrawer debugEnabled={debugEnabled} debugPayload={safeResult.debug} />
    </section>
  )
}
