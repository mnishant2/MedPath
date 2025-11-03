# Ontology Path Extraction

Extract hierarchical root→leaf paths for vocabularies using codes from the combined mapping file produced by dataset processing.

Accepted at IJCNLP–AACL 2025 (MedPath). Preprint forthcoming on arXiv.

## Table of Contents

- Overview
- Supported Vocabularies and Backends
- Configuration (credentials, versions/branches)
- File-based Vocabularies: Required Files and Layout
- Building the MeSH Tree Map
- CLI Usage and Examples
- Outputs (structure and logs)
- Tips and Troubleshooting

## Overview

- Entry point: `scripts/path_extraction/extract_hierarchical_paths.py`
- Reads: `combined_cui_to_vocab_codes_with_tty.json` (from `data_processed/mappings{_suffix}`)
- Writes: `data_processed/hierarchical_paths{_suffix}/<vocab>/{results,progress,logs}`
- Robust extractors with retries, caching, and offline fallbacks where possible.

## Supported vocabularies and backends

- SNOMED CT (`SNOMEDCT_US`, `SNOMED_CT`) — API (Snowstorm). Requires base URL in credentials; supports versioned branches.
- MeSH (`MSH`, `MESH`) — Offline-first. Builds paths from MeSH tree numbers using `desc{year}.xml` and a cached pickle. MeSH API for tree numbers is optional and disabled by default.
  Download Link: https://www.nlm.nih.gov/databases/download/mesh.html
- NCI Thesaurus (`NCI`) — Local OWL file (`Thesaurus.owl`).
  Download Link: https://evs.nci.nih.gov/evs-download/thesaurus-downloads
- LOINC (`LNC`, `LOINC`) — API (FHIR). Requires username/password in credentials; supports `systemVersion`.
- MedDRA (`MDR`, `MEDDRA`) — API (MAPIs). Requires token in credentials; uses configured version.
- NCBI Taxonomy (`NCBI`) — API (NCBI E-utilities). Optional `api_key` recommended. Snapshot is metadata-only (no branchable versions).
- HPO (`HPO`) — Local OBO file (`hp.obo`), 
  Download Link: https://hpo.jax.org/data/ontology
- GO (`GO`) — Local OBO file (`go-basic.obo`).
  Download Link: https://geneontology.org/docs/download-ontology/
- LCH_NW (`LCH_NW`) — Local JSON-LD file (`subjects.skosrdf.jsonld`).
  Download Link: https://id.loc.gov/authorities/subjects.html (Bulk Exports section(SKOS/RDF JSONLD))
- ICD (`ICD9CM`, `ICD10CM`) — Local text files.
  Download Link: https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-9-cm-diagnosis-procedure-codes-abbreviated-and-full-code-titles(ICD9CM), https://www.cdc.gov/nchs/icd/icd-10-cm/files.html(ICD10CM)

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
- NCBI Taxonomy:
  - `apis.ncbi.base_url` (default: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`)
  - Optional: `apis.ncbi.api_key` (recommended for higher rate limits)
  - Optional metadata: `versions.NCBI_taxonomy_snapshot` (e.g., `2025-10-01`) used for logging/reporting
- NCI/HPO/GO/LCH_NW/ICD local file paths under `local_files` (resolved relative to `MedPath/path_data` by default)

CLI `--version` sets `versions.GLOBAL_VERSION_OVERRIDE` at runtime as a fallback for supported extractors.

Notes on naming and aliases:
- Use canonical names without legacy “_V2” suffixes (e.g., `SNOMED_CT`). Aliases remain supported (e.g., `SNOMEDCT_US`, `MESH`/`MSH`).

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
  --vocabs MSH NCI LCH_NW HPO GO SNOMEDCT_US ICD9CM ICD10CM LNC MDR NCBI \
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
  ncbi/
    results/                   # NCBI_paths.json, NCBI_statistics.json
    progress/                  # NCBI_checkpoint.json
    logs/
  ...
```

Notes:
- Logs are written both by the orchestrator (`hierarchical_paths{_suffix}/logs/`) and per vocabulary (`<vocab>/logs/`).
- The orchestrator tolerates different mapping shapes (codes list vs dict-of-codes with TTY) and common aliases (e.g., `MSH`/`MESH`, `SNOMED_CT`/`SNOMEDCT_US`).
- MeSH runs in offline mode by default; enable the API only if needed.

## Tips and Troubleshooting

- Ensure `combined_cui_to_vocab_codes_with_tty.json` comes from the same `{_suffix}` run you intend to extract from.
- For API-backed vocabularies, verify credentials and optional version/branch settings before running.
- If CUIs processed appear as 0 for a vocabulary, verify the mapping file contains codes for that vocabulary (aliases are supported but must match).

## Tips

- Use `--subdir_suffix` to keep runs separate (e.g., `_sample`, `_v2`).
- Ensure credentials are valid before running API-backed vocabularies (LOINC, MedDRA, SNOMED).
- For quick tests, target fewer vocabs or a subset of CUIs via a filtered mapping file.
