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
      isActive ? 'bg-brand-600 text-white' : 'text-slate-700 hover:bg-emerald-50'
    }`
  }

  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 px-4 py-4 backdrop-blur md:px-6">
      <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-3">
          <img src="/logo.png" alt="ZER0 GONDOGOL logo" className="h-[34px] w-[34px] shrink-0 rounded-lg object-contain" />
          <div>
            <h1 className="text-xl font-semibold text-slate-950">ZER0 GONDOGOL</h1>
            <p className="text-sm text-slate-600">Food, nutrition, ingredient, and intake intelligence.</p>
          </div>
        </div>

        <nav className="flex w-full items-center gap-1 overflow-x-auto rounded-xl border border-slate-200 bg-white p-1 shadow-sm md:w-auto">
          <button type="button" onClick={() => navigate('/')} className={navClass(path === '/')}>
            Health Check
          </button>
          <button
            type="button"
            onClick={() => navigate('/explorer')}
            className={navClass(path === '/explorer')}
          >
            Ingredient Explorer
          </button>
        </nav>
      </div>
    </header>
  )
}
