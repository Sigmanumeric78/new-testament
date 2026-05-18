import { useEffect, useState } from 'react'
import AskPage from './pages/AskPage'
import ChemicalExplorerPage from './pages/ChemicalExplorerPage'

export default function App() {
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

  if (path === '/explorer') {
    return <ChemicalExplorerPage />
  }

  return (
    <AskPage />
  )
}
