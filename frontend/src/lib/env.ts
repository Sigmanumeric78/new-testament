export function resolveApiBaseUrl(raw: unknown): string {
  const base = typeof raw === 'string' && raw.trim() ? raw.trim() : 'http://localhost:8000'
  return base.replace(/\/+$/, '')
}

export function getApiBaseUrl(): string {
  return resolveApiBaseUrl(import.meta.env.VITE_API_BASE_URL)
}
