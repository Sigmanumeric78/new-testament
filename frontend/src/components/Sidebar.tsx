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
        mode === value ? 'bg-brand-600 text-white' : 'bg-white text-slate-700 hover:bg-slate-100'
      }`}
    >
      {label}
    </button>
  )

  return (
    <aside className="w-full border-r border-slate-200 bg-slate-100/70 p-4 lg:w-80">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-slate-900">SoberScope</h2>
        <p className="text-xs text-slate-600">Guided alcohol risk checks</p>
        <p className="mt-1 text-[11px] text-slate-500">Use the top navigation to open Chemical Explorer.</p>
      </div>

      <button
        type="button"
        onClick={onNewCheck}
        className="focus-ring mb-4 w-full rounded-lg bg-slate-900 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-700"
      >
        New check
      </button>

      <div className="mb-4 space-y-2">
        {modeBtn('ask', 'Ask mode')}
        {modeBtn('intake', 'Intake mode')}
      </div>

      <div className="mb-4">
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">System health</p>
        <HealthBadge health={health} isLoading={healthLoading} />
      </div>

      <div>
        <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Example prompts</p>
        <ExamplePrompts onSelect={onSelectPrompt} />
      </div>
    </aside>
  )
}
