# Weaviate Schema Design (Deterministic Freeze)

## Scope
This document freezes retrieval ontology design only. No ingestion, no live Weaviate calls, and no vector generation are performed in ETL_07A.

Input artifacts analyzed:
- `rag/neo4j/neo4j_graph_schema_design.md` (present)
- `data/processed/beverage/compound_profiles/beverage_compound_matrix_expanded.csv` (present)
- `data/processed/human/human_metabolism_parameters.csv` (present)
- `data/processed/pbpk/pbpk_parameter_library.csv` (present)
- `data/processed/pbpk/beverage_effect_modifiers.csv` (present)

## Retrieval Collections
### BeverageKnowledge
- Primary key: `beverage_id`
- Source artifacts: `beverage_compound_matrix, beverage_effect_modifiers`
- Retrieval intent: Resolve beverage composition, class expansions, and likely downstream exposure drivers.
- Searchable content fields: `beverage_name`, `category`, `compound_name`, `source_compound_class`, `modifier_reason`, `trigger_compounds`
- Metadata fields: `beverage_id`, `category`, `expansion_type`, `chemical_category`, `compound_role`, `estimated_concentration`, `concentration_unit`
- Provenance fields: `source_dataset`, `source_file`, `source_row`, `confidence_score`
- Expected query examples:
  - What compounds are in whisky and which are family-expanded?
  - Which beverage categories carry sulfur-related compounds?
  - Which compounds in bourbon are linked to modifier signals?
- Design readiness: `True` (missing searchable=0, missing provenance=0)

### CompoundKnowledge
- Primary key: `normalized_compound_name`
- Source artifacts: `beverage_compound_matrix`
- Retrieval intent: Answer molecule-level lookup and category membership questions with provenance.
- Searchable content fields: `compound_name`, `normalized_compound_name`, `pubchem_cid`, `chemical_category`, `compound_role`, `source_compound_class`
- Metadata fields: `chemical_category`, `compound_role`, `estimated_concentration`, `concentration_unit`, `digestion_effect`, `metabolic_burden`, `expansion_type`
- Provenance fields: `source_dataset`, `source_file`, `source_row`, `confidence_score`
- Expected query examples:
  - What is the PubChem CID for isoamyl alcohol?
  - Which compounds belong to organic acid class?
  - Which compounds are marked digestion-relevant?
- Design readiness: `True` (missing searchable=0, missing provenance=0)

### MetabolismKnowledge
- Primary key: `parameter_id`
- Source artifacts: `human_metabolism_parameters`
- Retrieval intent: Retrieve physiology and metabolism evidence by domain, population, and condition.
- Searchable content fields: `parameter_name`, `domain`, `population_group`, `condition`, `value`, `unit`, `evidence_text`
- Metadata fields: `modifier_type`, `effect_direction`, `extract_method`, `source_page`
- Provenance fields: `source_document`, `source_page`, `confidence_score`, `extract_method`
- Expected query examples:
  - What evidence supports slower gastric emptying in fed state?
  - How does sex affect body water distribution parameters?
  - Which records quantify ethanol elimination rate?
- Design readiness: `True` (missing searchable=0, missing provenance=0)

### PBPKKnowledge
- Primary key: `parameter_id`
- Source artifacts: `pbpk_parameter_library`
- Retrieval intent: Ground simulator parameter selection and compartment-specific interpretation.
- Searchable content fields: `parameter_name`, `compartment`, `base_value`, `unit`, `modifier_reason`, `source_parameter_id`
- Metadata fields: `population_group`, `modifier`, `confidence_score`
- Provenance fields: `source_document`, `source_parameter_id`, `confidence_score`
- Expected query examples:
  - What is the base ethanol elimination rate and evidence source?
  - Which parameters affect the liver compartment?
  - How is first-pass metabolism parameterized?
- Design readiness: `True` (missing searchable=0, missing provenance=0)

### ToxicityKnowledge
- Primary key: `modifier_id`
- Source artifacts: `beverage_effect_modifiers`
- Retrieval intent: Explain toxicity and hangover-linked modifier signals from beverage chemistry.
- Searchable content fields: `beverage_name`, `category`, `parameter_name`, `modifier_reason`, `trigger_compounds`, `source_compound_class`
- Metadata fields: `beverage_id`, `compartment`, `modifier`, `confidence_score`
- Provenance fields: `source_compound_class`, `confidence_score`
- Expected query examples:
  - Which compounds contribute to hangover amplification risk?
  - What modifier signals are associated with histamine in beer?
  - Which beverages have sulfite-linked sensitivity modifiers?
- Design readiness: `True` (missing searchable=0, missing provenance=0)

### PopulationKnowledge
- Primary key: `population_group`
- Source artifacts: `human_metabolism_parameters, pbpk_parameter_library`
- Retrieval intent: Resolve group-specific physiology effects and PBPK implications.
- Searchable content fields: `population_group`, `domain`, `condition`, `effect_direction`, `parameter_name`, `modifier_reason`
- Metadata fields: `value`, `unit`, `modifier_type`, `extract_method`, `compartment`
- Provenance fields: `source_document`, `source_page`, `confidence_score`, `source_parameter_id`
- Expected query examples:
  - How does fasted state affect absorption-relevant physiology?
  - Which parameters differ for female vs male groups?
  - What evidence exists for liver impairment modifiers?
- Design readiness: `True` (missing searchable=0, missing provenance=0)

### ScientificEvidence
- Primary key: `derived:evidence_id=sha1(source_document|source_page|source_parameter_id|source_row|parameter_name)`
- Source artifacts: `human_metabolism_parameters, pbpk_parameter_library, beverage_compound_matrix`
- Retrieval intent: Provide auditable evidence retrieval and citation surfaces for downstream QA/RAG.
- Searchable content fields: `evidence_text`, `source_document`, `parameter_name`, `domain`, `condition`, `modifier_reason`, `compound_name`
- Metadata fields: `source_page`, `unit`, `value`, `extract_method`, `chemical_category`, `compartment`
- Provenance fields: `source_document`, `source_page`, `source_file`, `source_row`, `confidence_score`
- Expected query examples:
  - Show evidence text for ethanol elimination rate assumptions.
  - Which source documents support body water distribution differences?
  - What are the source rows behind whisky-related chemistry claims?
- Design readiness: `True` (missing searchable=0, missing provenance=0)

## Deterministic Chunking Strategy
- No random splitting.
- No token-length randomization or stochastic overlap.
- Chunk boundaries follow semantic entity boundaries only.
- Chunk IDs are deterministic from collection + primary key.
- Provenance fields are copied to every chunk.
- Chunk ID format: `WVC::<CollectionName>::<normalized_primary_key_value>`
- Chunk ID normalization rules:
  - trim whitespace
  - lowercase
  - replace non-alphanumeric runs with underscore
  - strip leading/trailing underscores
- Per-collection semantic boundaries:
  - `BeverageKnowledge`: one chunk per beverage_id
  - `CompoundKnowledge`: one chunk per normalized_compound_name
  - `MetabolismKnowledge`: one chunk per parameter_id row
  - `PBPKKnowledge`: one chunk per parameter_id row
  - `ToxicityKnowledge`: one chunk per modifier_id row
  - `PopulationKnowledge`: one chunk per population_group, grouped deterministically by parameter_name then condition
  - `ScientificEvidence`: one chunk per deterministic evidence_id from source provenance tuple
- Ordering rules:
  - Within each chunk, fields are serialized in fixed schema order.
  - Rows are sorted by primary key ascending before chunk materialization.
  - Multi-value lists are sorted lexicographically before serialization.

## Embedding Contract
- Model: `nomic-ai/nomic-embed-text-v1`
- Vector dimensions: `768`
- Deterministic preprocessing:
  - Unicode normalization: NFC.
  - Normalize line endings to LF.
  - Trim leading/trailing whitespace per field.
  - Collapse internal whitespace runs to a single space.
  - Lowercase field keys; preserve original field values except whitespace normalization.
  - Serialize fields in fixed per-collection order using '\n' separators.
- Missing field handling:
  - Missing/empty values become literal token 'unknown'.
  - Do not drop missing fields; preserve placeholders for positional consistency.
  - If all searchable fields are missing, skip vectorization and flag chunk as 'insufficient_text'.
- Reproducibility guards:
  - No random seed usage required; transformations are pure string functions.
  - Identical input rows must produce byte-identical embedding payload text.
  - Schema changes require version bump before new vector generation.

## Final Gate
- `safe_for_weaviate_materialization`: `true`
