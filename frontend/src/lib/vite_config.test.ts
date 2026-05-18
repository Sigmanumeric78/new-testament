import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('vite config', () => {
  it('uses deterministic local dev port 5173 with strictPort', () => {
    const file = readFileSync(resolve(process.cwd(), 'vite.config.ts'), 'utf-8')
    expect(file).toMatch(/port:\s*5173/)
    expect(file).toMatch(/strictPort:\s*true/)
    expect(file).toMatch(/host:\s*['"]localhost['"]/)
  })
})
