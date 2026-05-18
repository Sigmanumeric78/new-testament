import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ChemicalExplorerPage from '../pages/ChemicalExplorerPage'

vi.mock('../lib/api', () => ({
  getHealth: vi.fn(async () => ({ status: 'ok', components: { api: { ok: true, detail: 'ok' } } })),
}))

const listChemicalsMock = vi.fn()
const getChemicalDetailMock = vi.fn()
const getChemicalConformerMock = vi.fn()

vi.mock('../lib/chemicalApi', () => ({
  listChemicals: (...args: unknown[]) => listChemicalsMock(...args),
  getChemicalDetail: (...args: unknown[]) => getChemicalDetailMock(...args),
  getChemicalConformer: (...args: unknown[]) => getChemicalConformerMock(...args),
}))

const BASE_ITEMS = [
  {
    compound_id: 'cmp-cid-702',
    compound_name: 'ethanol',
    normalized_compound_name: 'ethanol',
    pubchem_cid: 702,
    chemical_class: 'active_alcohol',
    canonical_smiles: 'CCO',
    rdkit_valid: true,
    beverage_count: 12,
    beverage_examples: ['Whisky', 'Beer'],
    has_3d_conformer: true,
  },
  {
    compound_id: 'cmp-cid-1119',
    compound_name: 'sulfite',
    normalized_compound_name: 'sulfite',
    pubchem_cid: 1119,
    chemical_class: 'sulfites',
    canonical_smiles: null,
    rdkit_valid: false,
    beverage_count: 4,
    beverage_examples: ['Wine'],
    has_3d_conformer: false,
  },
]

beforeEach(() => {
  listChemicalsMock.mockReset()
  getChemicalDetailMock.mockReset()
  getChemicalConformerMock.mockReset()

  listChemicalsMock.mockResolvedValue({
    items: BASE_ITEMS,
    total: BASE_ITEMS.length,
    limit: 24,
    offset: 0,
  })

  getChemicalDetailMock.mockResolvedValue({
    ...BASE_ITEMS[0],
    related_beverages: ['Whisky', 'Beer'],
    source_compound_class: 'alcohols',
    metabolism_relevance: 'likely_relevant',
    toxicity_relevance: 'context_dependent',
    available_structure_formats: ['sdf_3d'],
    conformer_availability_summary: {
      has_3d_conformer: true,
      has_2d_structure: true,
      available_formats: ['sdf_3d'],
    },
  })

  getChemicalConformerMock.mockResolvedValue({
    compound_id: 'cmp-cid-702',
    compound_name: 'ethanol',
    pubchem_cid: 702,
    has_3d_conformer: false,
    format: null,
    sdf: null,
    message: '3D conformer not available for this compound.',
  })
})

describe('ChemicalExplorerPage', () => {
  it('renders explorer page and list', async () => {
    render(<ChemicalExplorerPage />)

    expect(screen.getByRole('heading', { name: /chemical explorer/i, level: 1 })).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByText('ethanol')).toBeInTheDocument()
      expect(screen.getByText('sulfite')).toBeInTheDocument()
    })
  })

  it('search and filters trigger list calls', async () => {
    render(<ChemicalExplorerPage />)

    await waitFor(() => expect(listChemicalsMock).toHaveBeenCalled())

    await userEvent.type(screen.getByLabelText(/search compounds/i), 'ethanol')
    await userEvent.click(screen.getByRole('button', { name: /search/i }))

    await waitFor(() => {
      expect(listChemicalsMock).toHaveBeenCalledWith(
        expect.objectContaining({
          q: expect.stringContaining('ethanol'),
        }),
      )
    })

    await userEvent.click(screen.getByLabelText(/has 3d conformer only/i))
    await waitFor(() => {
      expect(listChemicalsMock).toHaveBeenCalledWith(
        expect.objectContaining({
          has_3d: true,
        }),
      )
    })
  })

  it('renders detail panel for selected compound', async () => {
    render(<ChemicalExplorerPage />)

    await waitFor(() => expect(getChemicalDetailMock).toHaveBeenCalled())

    expect(screen.getByText(/canonical smiles/i)).toBeInTheDocument()
    expect(screen.getByText(/metabolism relevance/i)).toBeInTheDocument()
  })

  it('renders 3D fallback message when conformer is missing', async () => {
    render(<ChemicalExplorerPage />)

    await waitFor(() => {
      expect(screen.getByText(/3d conformer not available for this compound/i)).toBeInTheDocument()
    })
  })
})
