import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import IntakeForm from './IntakeForm'

describe('IntakeForm', () => {
  it('requires valid fields before submit', async () => {
    const onSubmit = vi.fn()
    render(<IntakeForm loading={false} onSubmit={onSubmit} />)

    const submit = screen.getByRole('button', { name: /submit intake/i })
    expect(submit).toBeDisabled()

    await userEvent.click(submit)
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('rejects impossible weight', async () => {
    const onSubmit = vi.fn()
    render(<IntakeForm loading={false} onSubmit={onSubmit} />)

    await userEvent.type(screen.getByLabelText(/drink type/i), 'vodka')
    await userEvent.type(screen.getByLabelText(/weight/i), '10')
    await userEvent.type(screen.getByLabelText(/amount/i), '200')

    const submit = screen.getByRole('button', { name: /submit intake/i })
    expect(submit).toBeDisabled()
  })

  it('rejects impossible amount', async () => {
    const onSubmit = vi.fn()
    render(<IntakeForm loading={false} onSubmit={onSubmit} />)

    await userEvent.type(screen.getByLabelText(/drink type/i), 'vodka')
    await userEvent.type(screen.getByLabelText(/weight/i), '75')
    await userEvent.type(screen.getByLabelText(/amount/i), '3500')

    const submit = screen.getByRole('button', { name: /submit intake/i })
    expect(submit).toBeDisabled()
  })

  it('submits valid payload correctly', async () => {
    const onSubmit = vi.fn()
    render(<IntakeForm loading={false} onSubmit={onSubmit} />)

    await userEvent.selectOptions(screen.getByLabelText(/^sex/i), 'female')
    await userEvent.type(screen.getByLabelText(/weight/i), '60')
    await userEvent.type(screen.getByLabelText(/age/i), '28')
    await userEvent.selectOptions(screen.getByLabelText(/fed state/i), 'fasted')
    await userEvent.type(screen.getByLabelText(/drink type/i), 'whisky')
    await userEvent.type(screen.getByLabelText(/amount/i), '180')
    await userEvent.type(screen.getByLabelText(/duration/i), '1.5')
    await userEvent.selectOptions(screen.getByLabelText(/^goal/i), 'should_i_keep_drinking')

    const submit = screen.getByRole('button', { name: /submit intake/i })
    expect(submit).toBeEnabled()

    await userEvent.click(submit)

    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith({
      sex: 'female',
      weight_kg: 60,
      age: 28,
      fed_state: 'fasted',
      drink_type: 'whisky',
      amount_ml: 180,
      duration_h: 1.5,
      goal: 'should_i_keep_drinking',
    })
  })

  it('goal selection maps correctly', async () => {
    const onSubmit = vi.fn()
    render(<IntakeForm loading={false} onSubmit={onSubmit} />)

    await userEvent.type(screen.getByLabelText(/drink type/i), 'beer')
    await userEvent.type(screen.getByLabelText(/weight/i), '75')
    await userEvent.type(screen.getByLabelText(/amount/i), '500')
    await userEvent.selectOptions(screen.getByLabelText(/^goal/i), 'drive_check')

    await userEvent.click(screen.getByRole('button', { name: /submit intake/i }))

    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        goal: 'drive_check',
      }),
    )
  })
})
