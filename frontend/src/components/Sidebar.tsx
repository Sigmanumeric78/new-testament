import type { HealthResponse } from '../lib/types'
import ExamplePrompts from './ExamplePrompts'
import HealthBadge from './HealthBadge'

interface SidebarProps {
  mode: 'ask' | 'intake'
  onModeChange: (mode: 'ask' | 'intake') => void
  onNewCheck: () => void
  onSelectPrompt: (prompt: string) => void
  health: HealthResponse | null
  healthLoading: boolean
}

export default function Sidebar({ mode, onModeChange, onNewCheck, onSelectPrompt, health, healthLoading }: SidebarProps) {
  const modeBtn = (value: 'ask' | 'intake', label: string) => (
    <button
      type="button"
      onClick={() => onModeChange(value)}
      className={`focus-ring w-full rounded-lg px-3 py-2 text-left text-sm font-medium transition ${
        mode === value ? 'bg-brand-600 text-white' : 'bg-white text-slate-700 hover:bg-emerald-50'
      }`}
    >
      {label}
    </button>
  )

  const comingSoon = [
    'Food Product Analyzer',
    'Nutrition Label Decoder',
    'Ingredient Explainer',
    'Diet Goal Match',
    'Food Comparison',
    'Meal Analyzer',
  ]

  return (
    <aside className="w-full border-r border-slate-200 bg-white/65 p-4 lg:w-80">
      <button
        type="button"
        onClick={onNewCheck}
        className="focus-ring mb-4 w-full rounded-lg bg-slate-950 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-800"
      >
        New Health Check
      </button>

      <div className="mb-4 space-y-2">
        {modeBtn('ask', 'Health Check')}
        {modeBtn('intake', 'Guided Health Intake')}
      </div>

      <div className="mb-4">
        <p className="mb-2 text-xs font-semibold uppercase text-slate-500">Platform Status</p>
        <HealthBadge health={health} isLoading={healthLoading} />
      </div>

      <div className="mb-4">
        <p className="mb-2 text-xs font-semibold uppercase text-slate-500">Coming soon</p>
        <div className="space-y-2">
          {comingSoon.map((item) => (
            <button
              key={item}
              type="button"
              disabled
              className="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-left text-xs font-medium text-slate-500"
            >
              {item}
              <span className="ml-2 rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-800">
                Coming soon
              </span>
            </button>
          ))}
        </div>
      </div>

      <div>
        <p className="mb-2 text-xs font-semibold uppercase text-slate-500">Example prompts</p>
        <ExamplePrompts onSelect={onSelectPrompt} />
      </div>
    </aside>
  )
}
