export interface ChemicalSummary {
  compound_id: string
  compound_name: string
  normalized_compound_name: string
  pubchem_cid: number | null
  chemical_class: string
  canonical_smiles: string | null
  rdkit_valid: boolean
  beverage_count: number
  beverage_examples: string[]
  has_3d_conformer: boolean
}

export interface ChemicalListResponse {
  items: ChemicalSummary[]
  total: number
  limit: number
  offset: number
}

export interface ChemicalDetail extends ChemicalSummary {
  related_beverages: string[]
  source_compound_class: string
  metabolism_relevance: string
  toxicity_relevance: string
  available_structure_formats: string[]
  conformer_availability_summary: {
    has_3d_conformer: boolean
    has_2d_structure: boolean
    available_formats: string[]
  }
}

export interface ChemicalConformerResponse {
  compound_id: string
  compound_name: string
  pubchem_cid: number | null
  has_3d_conformer: boolean
  format: 'sdf' | null
  sdf: string | null
  message: string
}

export interface ChemicalSearchParams {
  q?: string
  chemical_class?: string
  has_3d?: boolean
  limit?: number
  offset?: number
}
