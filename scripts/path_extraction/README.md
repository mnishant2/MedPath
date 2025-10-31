# Ontology Path Extraction

Extract hierarchical root→leaf paths for vocabularies using codes from the combined mapping file produced by dataset processing.

## Overview

- Entry point: `scripts/path_extraction/extract_hierarchical_paths.py`
- Reads: `combined_cui_to_vocab_codes_with_tty.json` (from `data_processed/mappings{_suffix}`)
- Writes: `data_processed/hierarchical_paths{_suffix}/<vocab>/{results,progress,logs}`
- Robust extractors with retries, caching, and offline fallbacks where possible.

## Supported vocabularies and backends

- SNOMED CT (`SNOMEDCT_US`, `SNOMED_CT`) — API (Snowstorm). Requires base URL in credentials; supports versioned branches.
- MeSH (`MSH`, `MESH`) — Offline-first. Builds paths from MeSH tree numbers using `desc{year}.xml` and a cached pickle. MeSH API for tree numbers is optional and disabled by default.
- NCI Thesaurus (`NCI`) — Local OWL file (`Thesaurus.owl`).
- LOINC (`LNC`, `LOINC`) — API (FHIR). Requires username/password in credentials; supports `systemVersion`.
- MedDRA (`MDR`, `MEDDRA`) — API (MAPIs). Requires token in credentials; uses configured version.
- HPO (`HPO`) — Local OBO file (`hp.obo`).
- GO (`GO`) — Local OBO file (`go-basic.obo`).
- LCH_NW (`LCH_NW`) — Local JSON-LD file (`subjects.skosrdf.jsonld`).
- ICD (`ICD9CM`, `ICD10CM`) — Local text files.

## Configuration

Copy `scripts/configs/credentials.template.yaml` to `scripts/configs/credentials.yaml` and fill as needed:

- MeSH (offline default):
  - `local_files.mesh.xml_file` (e.g., `desc2025.xml`)
  - `local_files.mesh.pickle_cache` (e.g., `mesh_tree_map_2025.pkl`)
  - Optional MeSH API: `apis.mesh.enable_api: true` (default false)
- SNOMED:
  - `apis.snomed.base_url`
  - Optional branch precedence: `apis.snomed.branch` → `versions.SNOMED_branch` → `versions.GLOBAL_VERSION_OVERRIDE` → `MAIN`
    - e.g., `MAIN/SNOMEDCT-US/2024-09-01`
- LOINC:
  - `apis.loinc.username`, `apis.loinc.password`
  - Optional: `versions.LOINC_version` (sent as FHIR `systemVersion`)
- MedDRA:
  - `apis.meddra.token` (see template for curl token instructions)
  - `apis.meddra.base_url`, `apis.meddra.version`
- NCI/HPO/GO/LCH_NW/ICD local file paths under `local_files` (resolved relative to `MedPath/path_data` by default)

CLI `--version` sets `versions.GLOBAL_VERSION_OVERRIDE` at runtime as a fallback for supported extractors.

## File-based vocabularies: required files and layout

Place files under `MedPath/path_data/` (use the repository’s `path_data` as a template). URLs are intentionally left blank for you to fill.

```text
MedPath/path_data/
  Thesaurus.owl                           # NCI Thesaurus OWL (NCI) — URL: 
  desc2025.xml                            # MeSH XML (MeSH) — URL: 
  mesh_tree_map_2025.pkl                  # Cached MeSH tree map (MeSH) — built locally (see below)
  go-basic.obo                            # Gene Ontology OBO (GO) — URL: 
  hp.obo                                  # HPO OBO (HPO) — URL: 
  subjects.skosrdf.jsonld                 # LCH/NW JSON-LD (LCH_NW) — URL: 
  ICD-9-CM-v32-master-descriptions/
    CMS32_DESC_LONG_DX.txt                # ICD-9 names — URL: 
    CMS32_DESC_SHORT_DX.txt               # ICD-9 names — URL: 
  icd10cm-Code Descriptions-2026/
    icd10cm-codes-2026.txt                # ICD-10 codes+names — URL: 
```

### Building the MeSH tree map (offline)

If you update the MeSH `desc{year}.xml`, rebuild the pickle cache:
```bash
python scripts/path_extraction/utils/build_mesh_tree_map.py \
  --xml-file ./path_data/desc2025.xml \
  --output-pkl ./path_data/mesh_tree_map_2025.pkl
```

MeSH extraction uses the local tree cache by default and yields the same paths as API-derived trees.

## CLI usage

```bash
python -u scripts/path_extraction/extract_hierarchical_paths.py \
  --vocabs MSH NCI LCH_NW HPO GO SNOMEDCT_US ICD9CM ICD10CM LNC MDR \
  --mappings_path ./data_processed/mappings_sample/combined_cui_to_vocab_codes_with_tty.json \
  --output ./data_processed/hierarchical_paths \
  --subdir_suffix _sample \
  --parallel 2
```

Arguments:
- `--vocabs` or `--vocab`: one or more vocabularies (space or comma-separated). Aliases supported.
- `--mappings_path`: path to `combined_cui_to_vocab_codes_with_tty.json` (use suffixed folder if applicable).
- `--output`: output root. Default: `MedPath/data_processed/hierarchical_paths`.
- `--subdir_suffix`: optional suffix appended to the output dir name (e.g., `_sample`).
- `--parallel`: number of workers. Default: 2.
- `--version`: optional API/release identifier; affects SNOMED branch and LOINC `systemVersion` when not explicitly set in credentials.
- `--resume`: resume from checkpoints.
- `--verbose`: more logging.

## Outputs

Directory structure:
```text
MedPath/data_processed/hierarchical_paths{_suffix}/
  mesh/
    results/                   # MSH_paths.json, MSH_statistics.json
    progress/                  # MSH_checkpoint.json
    logs/                      # mesh_api.log, mesh_errors.log, mesh_stats.log
  nci/
    results/                   # NCI_paths.json, NCI_statistics.json
    progress/                  # NCI_checkpoint.json
    logs/
  snomed/
    results/                   # SNOMEDCT_US_paths.json, SNOMEDCT_US_statistics.json
    progress/                  # SNOMEDCT_US_checkpoint.json
    logs/                      # snomed_api.log, snomed_errors.log, snomed_stats.log
  loinc/
    results/                   # LNC_paths.json, LNC_statistics.json
    progress/                  # LNC_checkpoint.json
    logs/                      # loinc_api_ops.log, loinc_errors.log, loinc_stats.log
  meddra/
    results/                   # MDR_paths.json, MDR_statistics.json
    progress/                  # MDR_checkpoint.json
    logs/
  ...
```

Notes:
- Logs are written per vocabulary under `<vocab>/logs`. No top-level `results/` or `progress/` directories are created.
- The orchestrator tolerates different mapping shapes (codes list vs dict-of-codes with TTY).
- MeSH runs in offline mode by default; enable the API only if needed.

## Tips

- Use `--subdir_suffix` to keep runs separate (e.g., `_sample`, `_v2`).
- Ensure credentials are valid before running API-backed vocabularies (LOINC, MedDRA, SNOMED).
- For quick tests, target fewer vocabs or a subset of CUIs via a filtered mapping file.
