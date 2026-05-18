interface DebugDrawerProps {
  debugEnabled: boolean
  debugPayload?: Record<string, unknown>
}

export default function DebugDrawer({ debugEnabled, debugPayload }: DebugDrawerProps) {
  if (!debugEnabled || !debugPayload) return null

  return (
    <details className="rounded-xl border border-slate-300 bg-slate-950/95 p-4 text-slate-100">
      <summary className="cursor-pointer text-sm font-semibold">Debug details</summary>
      <pre className="mt-3 overflow-x-auto text-xs leading-relaxed">{JSON.stringify(debugPayload, null, 2)}</pre>
    </details>
  )
}
