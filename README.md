# MedPath

A unified pipeline for processing biomedical datasets into UMLS-linked JSONL and extracting hierarchical ontology paths (SNOMED CT, MeSH, NCI, LOINC, MedDRA, HPO, GO, LCH_NW, ICD).

Accepted at IJCNLP–AACL 2025. Preprint forthcoming on arXiv.

## Table of Contents

- Overview
- Datasets
- Cleaning Scripts (offset-preserving)
- Data Directory Layout (preserve original structure)
- Processing to UMLS-linked JSONL (CLI and examples)
- Outputs and Visualizations
- Ontology Path Extraction (CLI and examples)
- Configuration & Credentials
- Reproducibility & Logging
- Citation

## Datasets

The following datasets are supported. We are not sharing the datasets, since some of them have Data Usage Agreements, but they can be easily downloaded from the links shared below.

- CDR (Chemical–Disease Relation)
  - Download: https://github.com/JHnlp/BioCreative-V-CDR-Corpus
- NCBI Disease
  - Download: https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/
- MedMentions (CC0 1.0 Universal License)
  - Download: https://github.com/chanzuckerberg/MedMentions
- ADR (TAC Adverse Drug Reaction 2017)
  - Download: https://bionlp.nlm.nih.gov/tac2017adversereactions/
- COMETA
  - Download: https://github.com/cambridgeltl/cometa
- Mantra-GSC (English Subset,GNU General Public License v3.0)
  - Download: https://github.com/mi-erasmusmc/Mantra-Gold-Standard-Corpus
- ShAReCLEF eHealth 2013: Natural Language Processing and Information Retrieval for Clinical Care (PhysioNet Credentialed Health Data License 1.5.0)
  - Download: https://physionet.org/content/shareclefehealth2013/1.0/
- MIMIC-IV-EL (SNOMED CT Entity Linking Challenge, PhysioNet Credentialed Health Data License 1.5.0)
  - Download: https://physionet.org/content/snomed-ct-entity-challenge/1.0.0/
- CADEC
  - Download: https://data.csiro.au/collection/csiro:10948 (CSIRO Adverse Drug Event Corpus, CSIRO Data Licence)

Cleaning scripts (pre-processing cleanup while preserving mention offsets):
- MIMIC-IV-EL: `scripts/utils/clean_mimiciv_el.py`
- SHARE/CLEF: `scripts/utils/clean_shareclef.py`
- ADR: `scripts/utils/clean_adr.py`

These remove noisy characters and spacing while retaining exact character offsets used for mentions.

Example usage (adjust input/output paths):
```bash
# MIMIC-IV-EL
python scripts/utils/clean_mimiciv_el.py \
  --input /path/to/raw/mimic_iv_el \
  --output /path/to/clean/MIMICIV_EL_cleaned_no_placeholders

# SHARE/CLEF
python scripts/utils/clean_shareclef.py \
  --input /path/to/raw/shareclef \
  --output /path/to/clean/shareclef_cleaned

# ADR
python scripts/utils/clean_adr.py \
  --input /path/to/raw/adr \
  --output /path/to/clean/ADR_cleaned
```

## Data directory layout (expected before processing)

Preserve the original download structure exactly (as in `EL_gen/data`). Do not rearrange files; only change the top-level root with `--data_dir` if needed. Examples below mirror the expected layout used by the processors:

```text
<data_root>/
  CDR_Data/
    CDR.Corpus.v010516/
      CDR_TrainingSet.PubTator.txt
      CDR_DevelopmentSet.PubTator.txt
      CDR_TestSet.PubTator.txt

  NCBI/
    train/
    dev/
    test/

  medmentions/
    corpus/
    st21pv/

  ADR_cleaned/
    train/
    dev/
    test/

  cometa/
    splits/
      random/
        train/
        dev/
        test/

  Mantra-GSC/
    en/
      train/
      dev/
      test/

  shareclef_cleaned/
    train/
    test/

  MIMICIV_EL_cleaned_no_placeholders/
    notes/
    annotations/
    ...
  CADEC/
    AMT-SCT/
    cadec/
      meddra/
      original/
      sct/
      text/
    MedDRA/
    Original
```

Notes:
- The processors remap relative `DATASET_CONFIGS` paths under `--data_dir` but expect the inner trees to match the original downloads.
- COMETA must be under `cometa/splits/random/{train,dev,test}`.
- If you are not using the cleaned versions of MIMIC-IV-EL, shareclef, and ADR, then change the path/names accordingly in the corresponding `scripts/dataset_processors/{dataset}_processor.py` 

## Processing datasets to UMLS-linked JSONL

Entry point: `scripts/process_datasets.py`

Key arguments (underscore-style):
- `--all`: process all supported datasets found under the resolved data root
- `--datasets`: comma- or space-separated subset to process (e.g., `cdr,ncbi`)
- `--data_dir`: optional datasets root. If set, each dataset’s configured relative path is mapped under this root while preserving its original substructure.
- `--output`: root for outputs. Defaults to `MedPath/data_processed` regardless of CWD.
- `--subdir_suffix`: optional suffix to append to every output subdirectory (e.g., `_sample`)
- `--use_local_umls`: use local UMLS files for mapping
- `--umls_path`: path to local UMLS release (e.g., `/path/to/UMLS/2025AA`) when `--use_local_umls` is set
- `--stats_only`: only compute statistics from existing processed outputs
- `--skip_mappings`, `--skip_docs`: skip parts of the pipeline as needed

UMLS via API (when not using local files): ensure credentials are configured in `scripts/configs/credentials.yaml` (copy from the template) and omit `--use_local_umls`/`--umls_path`.

Outputs are saved under `--output` (default `MedPath/data_processed`) honoring `--subdir_suffix`:

```text
MedPath/data_processed/
  documents{_suffix}/         # Per-dataset processed JSONL
  mappings{_suffix}/          # Per-dataset and combined UMLS mappings (incl. *_with_tty.json)
  stats{_suffix}/             # Per-dataset and cross-dataset statistics + plots
```

Important combined mapping artifact (preserves TTY):
- `mappings{_suffix}/combined_cui_to_vocab_codes_with_tty.json`

Sample command (using included sample data):
```bash
cd /data/storage_hpc_nishant/MedPath
python -u scripts/process_datasets.py \
  --all \
  --data_dir ./data_sample \
  --output ./data_processed \
  --subdir_suffix _sample \
  --use_local_umls \
  --umls_path /path/to/UMLS/2025AA
```

The script ensures outputs land under `MedPath/data_processed` even when executed from different working directories.

Logging:
- Main processing logs: `scripts/logs/dataset_processing.log`
- Per-dataset processor logs (e.g., `adr_processing.log`) are also centralized under `scripts/logs/`.

## Outputs and Visualizations

Per-dataset statistics and plots are generated under `stats{_suffix}/`:

- Dataset JSON summaries under `stats{_suffix}/datasets/`:
  - `<dataset>_statistics.json` (includes totals, rates, distributions; large arrays summarized)
- Plots under `stats{_suffix}/plots/` for each dataset:
  - `<dataset>_document_lengths.png`
  - `<dataset>_mention_lengths.png`
  - `<dataset>_mentions_per_document.png`
  - `<dataset>_top_cuis.png` (if CUIs present)
  - `<dataset>_semantic_types.png` (if available)
  - `<dataset>_native_ontologies.png` (if available)
- Cross-dataset comparison (if not suppressed by `--skip_comparison`):
  - `stats{_suffix}/dataset_comparison_summary.json`
  - `stats{_suffix}/plots/comparison_<metric>.png` for key metrics (`total_documents`, `total_mentions`, `unique_cuis`, `mapping_success_rate`, `avg_mentions_per_document`)
- Cross-dataset mapping summary and combined artifacts in `mappings{_suffix}/`:
  - `combined_cui_to_vocab_codes_with_tty.json`
  - `combined_cui_to_vocab_codes.json`
  - `combined_cui_to_tuis.json`
  - `cross_dataset_mapping_summary.json` containing:
    - `total_unique_cuis`, `cuis_with_comprehensive_vocab_mappings`, `cuis_with_comprehensive_semantic_types`
    - `vocabulary_sources_found`, `semantic_types_found`, `dataset_cui_counts`
    - Coverage metrics across datasets

## Ontology path extraction

Path extraction converts vocabulary codes per CUI into hierarchical root→leaf paths for each vocabulary.

Script: `scripts/path_extraction/extract_hierarchical_paths.py`

Key arguments:
- `--vocabs` or `--vocab`: one or more vocabularies; aliases supported (e.g., `SNOMED_CT_V2`, `SNOMEDCT_US`, `MSH/MESH`, `NCI`, `LNC/LOINC`, `MDR`, `LCH_NW`, `HPO`, `GO`, `ICD9CM`, `ICD10CM`)
- `--mappings_path`: path to `combined_cui_to_vocab_codes_with_tty.json` (typically from `mappings{_suffix}`)
- `--output`: output root for hierarchical paths (default: `MedPath/data_processed/hierarchical_paths`)
- `--subdir_suffix`: optional suffix for the hierarchical paths directory (e.g., `_sample`)
- `--parallel`: number of workers (default 2)
- `--version`: optional API/release tag (read by some API-based extractors)
- `--resume`, `--verbose`

Which vocabularies are API- vs file-based?
- API first (requires credentials in `scripts/configs/credentials.yaml`):
  - SNOMED CT (Snowstorm), LOINC (FHIR), MedDRA (MAPIs), NCBI Taxonomy (E-utilities), optionally MeSH tree numbers (disabled by default)
- Local files first:
  - MeSH paths (built from `desc{year}.xml` + cached pickle), NCI (OWL), HPO (OBO), GO (OBO), LCH_NW (JSON-LD), ICD-9/10 (local text files)

Credentials: copy and fill `scripts/configs/credentials.yaml` from `scripts/configs/credentials.template.yaml`.
- Includes MeSH offline config (default), MedDRA token instructions, and NCBI E-utilities fields (`api_key`).

Output directory structure:
```text
MedPath/data_processed/hierarchical_paths{_suffix}/
  logs/                         # Orchestrator logs
  <vocab_name>/
    results/
      <VOCAB>_paths.json        # CUI→code→paths
      <VOCAB>_statistics.json   # extraction statistics
    progress/
      <VOCAB>_checkpoint.json
    logs/                       # Per-vocab extractor logs (API ops, errors, stats)
```

Example command (sample mapping file):
```bash
cd /data/storage_hpc_nishant/MedPath
python -u scripts/path_extraction/extract_hierarchical_paths.py \
  --vocabs MSH NCI LCH_NW HPO GO SNOMEDCT_US ICD9CM ICD10CM LNC MDR \
  --mappings_path ./data_processed/mappings_sample/combined_cui_to_vocab_codes_with_tty.json \
  --output ./data_processed/hierarchical_paths \
  --subdir_suffix _sample \
  --parallel 2
```

More details and vocabulary-specific notes are in `scripts/path_extraction/README.md`.

## Configuration & Credentials

- Copy `scripts/configs/credentials.template.yaml` to `scripts/configs/credentials.yaml` and fill values.
- `early_stopping_vocabs` should use names without the old “_V2” suffix (e.g., `SNOMED_CT`).
- Versioning:
  - SNOMED branch: `apis.snomed.branch` → `versions.SNOMED_branch` → `versions.GLOBAL_VERSION_OVERRIDE` → `MAIN` (fallback)
  - LOINC `systemVersion`: `versions.LOINC_version` or `--version` overrides when supported
  - MeSH is offline by default; build the tree cache locally (see path extraction README)

## Citation

If you use MedPath (scripts or processed datasets) in your research, please cite:

Mishra, N. MedPath: A Unified Pipeline for Biomedical Entity Linking and Hierarchical Path Extraction. In Proceedings of IJCNLP–AACL 2025. arXiv:XXXXX.XXXX.
