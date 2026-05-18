import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ComponentProps } from 'react'
import { describe, expect, it, vi } from 'vitest'
import QueryComposer from './QueryComposer'

function renderComposer(overrides: Partial<ComponentProps<typeof QueryComposer>> = {}) {
  const onQueryChange = vi.fn()
  const onResponseStyleChange = vi.fn()
  const onDebugChange = vi.fn()
  const onSubmit = vi.fn()

  render(
    <QueryComposer
      query="Can I drive after drinking 180ml whisky?"
      responseStyle="layman"
      debug={false}
      loading={false}
      onQueryChange={onQueryChange}
      onResponseStyleChange={onResponseStyleChange}
      onDebugChange={onDebugChange}
      onSubmit={onSubmit}
      {...overrides}
    />,
  )

  return {
    onQueryChange,
    onResponseStyleChange,
    onDebugChange,
    onSubmit,
  }
}

describe('QueryComposer', () => {
  it('shows user-friendly style labels', () => {
    renderComposer()
    const select = screen.getByLabelText(/response style/i)
    expect(select).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Simple' })).toHaveValue('layman')
    expect(screen.getByRole('option', { name: 'Detailed' })).toHaveValue('technical')
    expect(screen.getByRole('option', { name: 'Scientific' })).toHaveValue('scientific')
  })

  it('selecting Simple maps to response_style=layman', async () => {
    const { onResponseStyleChange } = renderComposer({ responseStyle: 'technical' })
    await userEvent.selectOptions(screen.getByLabelText(/response style/i), 'layman')
    expect(onResponseStyleChange).toHaveBeenCalledWith('layman')
  })

  it('selecting Detailed maps to response_style=technical', async () => {
    const { onResponseStyleChange } = renderComposer()
    await userEvent.selectOptions(screen.getByLabelText(/response style/i), 'technical')
    expect(onResponseStyleChange).toHaveBeenCalledWith('technical')
  })

  it('selecting Scientific maps to response_style=scientific', async () => {
    const { onResponseStyleChange } = renderComposer()
    await userEvent.selectOptions(screen.getByLabelText(/response style/i), 'scientific')
    expect(onResponseStyleChange).toHaveBeenCalledWith('scientific')
  })

  it('submits when form is valid', async () => {
    const { onSubmit } = renderComposer()
    await userEvent.click(screen.getByRole('button', { name: /run check/i }))
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })
})
