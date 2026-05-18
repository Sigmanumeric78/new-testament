import { useEffect, useState } from 'react'

export default function Header() {
  const [path, setPath] = useState<string>(() => window.location.pathname || '/')

  useEffect(() => {
    const update = () => setPath(window.location.pathname || '/')
    window.addEventListener('popstate', update)
    window.addEventListener('app:navigate', update as EventListener)
    return () => {
      window.removeEventListener('popstate', update)
      window.removeEventListener('app:navigate', update as EventListener)
    }
  }, [])

  function navigate(target: string): void {
    if (window.location.pathname === target) return
    window.history.pushState({}, '', target)
    window.dispatchEvent(new Event('app:navigate'))
  }

  function navClass(isActive: boolean): string {
    return `focus-ring rounded-lg px-3 py-2 text-sm font-semibold transition ${
      isActive ? 'bg-brand-600 text-white' : 'text-slate-700 hover:bg-slate-100'
    }`
  }

  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 px-6 py-4 backdrop-blur">
      <div className="mx-auto flex w-full max-w-[1600px] items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-slate-900">SoberScope</h1>
          <p className="text-sm text-slate-600">Alcohol risk estimation, explained clearly.</p>
        </div>

        <nav className="flex items-center gap-1 rounded-xl border border-slate-200 bg-white p-1">
          <button type="button" onClick={() => navigate('/')} className={navClass(path === '/')}>
            Ask
          </button>
          <button
            type="button"
            onClick={() => navigate('/explorer')}
            className={navClass(path === '/explorer')}
          >
            Chemical Explorer
          </button>
        </nav>
      </div>
    </header>
  )
}
