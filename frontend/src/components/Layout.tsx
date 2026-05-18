import type { ReactNode } from 'react'
import Header from './Header'

interface LayoutProps {
  sidebar?: ReactNode
  children: ReactNode
}

export default function Layout({ sidebar, children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 text-slate-900">
      <Header />
      <div className="mx-auto flex max-w-[1600px] flex-col lg:min-h-[calc(100vh-73px)] lg:flex-row">
        {sidebar ? sidebar : null}
        <main className="flex-1 p-4 md:p-6 lg:p-8">{children}</main>
      </div>
    </div>
  )
}
