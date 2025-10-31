# MedPath

A unified pipeline for processing biomedical datasets into UMLS-linked JSONL and extracting hierarchical ontology paths (SNOMED CT, MeSH, NCI, LOINC, MedDRA, HPO, GO, LCH_NW, ICD).

## Datasets

The following datasets are supported. Add download links and licenses where noted.

- CDR (Chemical–Disease Relation)
  - Download: 
  - License: 
- NCBI Disease
  - Download: 
  - License: 
- MedMentions
  - Download: 
  - License: 
- ADR (Adverse Drug Reaction)
  - Download: 
  - License: 
- COMETA
  - Download: 
  - License: 
- Mantra-GSC (English)
  - Download: 
  - License: 
- SHARE/CLEF eHealth
  - Download: 
  - License: 
- MIMIC-IV-EL
  - Download: 
  - License: 

Cleaning scripts for restricted datasets (run before processing):
- MIMIC-IV-EL: see `scripts/cleaning/mimic_iv_el_clean.py` (usage below)
- SHARE/CLEF: see `scripts/cleaning/shareclef_clean.py`
- ADR: see `scripts/cleaning/adr_clean.py`

Example usage (adjust input/output paths):
```bash
# MIMIC-IV-EL
python scripts/cleaning/mimic_iv_el_clean.py \
  --input /path/to/raw/mimic_iv_el \
  --output /path/to/clean/mimic_iv_el_clean

# SHARE/CLEF
auth python scripts/cleaning/shareclef_clean.py \
  --input /path/to/raw/shareclef \
  --output /path/to/clean/shareclef_clean

# ADR
python scripts/cleaning/adr_clean.py \
  --input /path/to/raw/adr \
  --output /path/to/clean/adr_clean
```

## Data directory layout (expected before processing)

Place datasets under a single data root similar to the `EL_gen/data` layout. For example:

```text
<data_root>/
  cdr/
    train/  dev/  test/
  ncbi/
    train/  dev/  test/
  medmentions/
    corpus/  st21pv/
  adr/
    train/  dev/  test/
  cometa/
    splits/
      random/
        train/  dev/  test/
  mantra-gsc/
    en/
      train/  dev/  test/
  shareclef/
    ...
  mimic-iv-el/
    ...
```

If unsure about a dataset’s exact tree, leave it under `<data_root>/<dataset_name>/...` preserving its original structure.

## Processing datasets to UMLS-linked JSONL

Entry point: `scripts/process_datasets.py`

Key arguments (underscore-style):
- `--all`: process all supported datasets found under the resolved data root
- `--datasets`: comma- or space-separated subset to process (e.g., `cdr,ncbi`)
- `--data_dir`: root directory for datasets. If omitted, defaults to `../../data/` relative to the script (i.e., `MedPath/data`).
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
  - SNOMED CT (Snowstorm), LOINC (FHIR), MedDRA (MAPIs), optionally MeSH tree numbers (disabled by default)
- Local files first:
  - MeSH paths (built from `desc{year}.xml` + cached pickle), NCI (OWL), HPO (OBO), GO (OBO), LCH_NW (JSON-LD), ICD-9/10 (local text files)

Credentials: copy and fill `scripts/configs/credentials.yaml` from `scripts/configs/credentials.template.yaml`.
- Includes MeSH offline config (default) and MedDRA token instructions.

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
