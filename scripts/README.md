# Dataset Processing Pipeline

This directory processes 9 datasets to a standardized JSONL format with UMLS mapping and `mapping_method` attribution.

## Datasets
CDR, NCBI, MedMentions, CADEC, ADR, Mantra-GSC, ShareCLEF, COMETA, MIMIC-IV EL.

## Raw Data Setup
Refer to each dataset's licensing and download instructions. Place the data in `./data/` (or supply `--data-root`) matching the `data_path` fields in `process_datasets.py` (paths are resolved relative to this file when `--data-root` is not given).

- Licensed: MIMIC-IV EL, ShareCLEF (obtain credentials; we cannot redistribute).
- Cleaning: ADR, ShareCLEF, MIMIC-IV EL — cleaning logic lives in `utils/cleaning.py` and dataset processors.

## Run Processing
```bash
python scripts/process_datasets.py --all --use-local-umls \
  --umls-path /path/to/UMLS/2025AA \
  --output-root ./data_processed --subdir-suffix _new
```
- Standardizes document `doc_id`
- Adds `mapping_method` to each mention: native_id_mapping, exact_match, semantic_containment, text_fallback, no_mapping
- Produces per-dataset enhanced mappings (including TTY) and overall combined mappings

## Outputs
- `data_processed/documents[_new]/*jsonl`
- `data_processed/mappings[_new]/*json`
- `data_processed/stats[_new]/*json`

## Combined Mappings
The pipeline automatically generates combined cross-dataset mapping summaries; no separate mapping script is required.

## Notes
- Use `--limit N` for quick tests.
- Use `--datasets name1 name2` to run a subset.

# UMLS Dataset Processing Results

This directory contains the processed and standardized versions of 9 biomedical datasets,
all mapped to UMLS CUIs and formatted consistently.

## Datasets Processed

### CDR (Chemical Disease Relations)

- **Description**: PubMed abstracts with chemical-disease relations annotated with MeSH IDs
- **Native Ontologies**: MSH
- **Entity Types**: Chemical, Disease
- **Splits: train, dev, test**
- **Data Path**: `../../data/CDR_Data/CDR.Corpus.v010516`

### NCBI Disease

- **Description**: PubMed abstracts with disease mentions annotated with MeSH and OMIM IDs
- **Native Ontologies**: MSH, OMIM
- **Entity Types**: Disease
- **Splits: train, dev, test**
- **Data Path**: `../../data/NCBI`

### MedMentions

- **Description**: PubMed abstracts with biomedical entity mentions mapped to UMLS CUIs
- **Native Ontologies**: UMLS
- **Entity Types**: Biomedical Entity
- **Splits: train, dev, test**
- **Data Path**: `../../data/medmentions`

### CADEC

- **Description**: Patient forum posts with adverse drug events annotated with SNOMED CT and MedDRA
- **Native Ontologies**: SNOMEDCT_US, MDR
- **Entity Types**: Adverse Drug Event
- **No predefined splits**
- **Data Path**: `../../data/CADEC`

### COMETA

- **Description**: Biomedical entity mentions linked to SNOMED CT concepts
- **Native Ontologies**: SNOMEDCT_US
- **Entity Types**: Biomedical Entity
- **No predefined splits**
- **Data Path**: `../../data/cometa`

### ADR

- **Description**: Drug-related adverse events annotated with MedDRA concept IDs
- **Native Ontologies**: MDR
- **Entity Types**: Adverse Drug Reaction
- **No predefined splits**
- **Data Path**: `../../data/ADR_cleaned`

### Mantra-GSC

- **Description**: Multilingual biomedical entity mentions (Gene, Species, Chemical)
- **Native Ontologies**: NCBI_GENE, NCBI_TAXONOMY, CHEBI
- **Entity Types**: Gene, Species, Chemical
- **No predefined splits**
- **Data Path**: `../../data/Mantra-GSC`

### MIMIC-IV EL (Licensed)

- **Description**: Clinical notes with entity linking to SNOMED CT concepts
- **Native Ontologies**: SNOMEDCT_US
- **Entity Types**: Clinical Entity
- **No predefined splits**
- **Data Path**: `../../data/MIMICIV_EL_cleaned_no_placeholders`

### ShareCLEF (Licensed)

- **Description**: Clinical cases annotated with UMLS CUIs
- **Native Ontologies**: UMLS
- **Entity Types**: Clinical Entity
- **Splits: train, test**
- **Data Path**: `../../data/shareclef_cleaned`


## Directory Structure

```
umls_datasets/
├── data_processed/
│   ├── documents/          # Standardized JSONL files
│   │   ├── cdr.jsonl
│   │   ├── ncbi_train.jsonl
│   │   ├── ncbi_dev.jsonl
│   │   ├── ncbi_test.jsonl
│   │   └── ...
│   ├── mappings/           # Vocabulary mappings
│   │   ├── cui_to_vocabs.json
│   │   └── cui_to_tui.json
│   └── stats/              # Statistics and visualizations
│       ├── dataset_comparison.html
│       └── individual_stats/
├── scripts/                # Processing scripts
└── cleaning_scripts/      # Data cleaning scripts for licensed datasets
```

## Output Format

All datasets are converted to a standardized JSONL format where each line represents a document:

```json
{
  "id": "document_id",
  "text": "Full document text...",
  "mentions": [
    {
      "start": 0,
      "end": 10,
      "text": "mention text",
      "native_id": "original_concept_id",
      "native_ontology_name": "MSH",
      "cui": "C0001234",
      "umls_name": "Preferred UMLS term",
      "semantic_type": "T047"
    }
  ]
}
```

## Usage

To process all datasets:
```bash
python umls_datasets/scripts/process_datasets.py --all
```

To process specific datasets:
```bash
python umls_datasets/scripts/process_datasets.py --datasets cdr ncbi medmentions
```

To generate statistics only:
```bash
python umls_datasets/scripts/process_datasets.py --stats-only
```

## Licensed Datasets

The following datasets require licensed access:
- **MIMIC-IV EL** (Entity Linking) - requires PhysioNet credentialed access
- **ShareCLEF** - requires CLEF eHealth evaluation lab access

These datasets cannot be redistributed. You must:
1. Obtain proper licensing from the respective organizations
2. Download and place data files in the correct directory structure
3. Run cleaning scripts (if needed) before processing

## Datasets Requiring Cleaning

The following datasets need text cleaning to remove noise and artifacts:
- **ADR** - removes XML artifacts and normalizes spacing
- **ShareCLEF** - removes metadata lines and normalizes annotations
- **MIMIC-IV EL** - removes anonymization placeholders and adjusts offsets

Cleaning scripts are provided in `cleaning_scripts/` directory.

## Statistics and Visualizations

Comprehensive statistics are generated for each dataset and cross-dataset comparisons,
including:

- Entity mention counts and distributions
- UMLS mapping coverage and success rates
- Vocabulary distribution across ontologies
- Semantic type analysis
- Interactive visualizations and plots

View the results in `data_processed/stats/dataset_comparison.html`.

## Citation

If you use these processed datasets, please cite the original dataset papers
and mention this processing pipeline.

Generated on: {Path().cwd()}

