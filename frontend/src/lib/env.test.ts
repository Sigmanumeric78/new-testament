import { describe, expect, it } from 'vitest'
import { resolveApiBaseUrl } from './env'

describe('env', () => {
  it('defaults API base URL to localhost:8000 when missing', () => {
    expect(resolveApiBaseUrl(undefined)).toBe('http://localhost:8000')
    expect(resolveApiBaseUrl('')).toBe('http://localhost:8000')
  })

  it('normalizes trailing slashes', () => {
    expect(resolveApiBaseUrl('http://localhost:8000/')).toBe('http://localhost:8000')
  })
})
