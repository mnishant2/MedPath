# MedPath Data Samples

This directory contains small, redistributable samples (≈10 documents each) of several datasets so you can quickly test the processing pipeline without downloading full corpora.

Included datasets (redistributable):
- CDR: `CDR_Data/CDR.Corpus.v010516/*.PubTator.txt` (first 10 documents per split)
- NCBI Disease: `NCBI/*corpus.txt` (first 10 documents per split)
- MedMentions: `medmentions/corpus_pubtator.txt` (first 10 documents)
- ADR (cleaned subset expected by pipeline): `ADR_cleaned/train/*.xml` (10 XML drug labels)
- COMETA: `cometa/chv.csv` (first 200 lines) and `COMETA_id_sf_dictionary.txt` (first 500 lines)
- Mantra-GSC (Dutch, EMEA subset): `Mantra-GSC/Dutch/EMEA_ec22-cui-best_man/*.txt` + matching `.ann` (10 pairs)

Excluded (licensed/restricted):
- MIMIC-IV EL, ShARe/CLEF, CADEC (obtain licenses; we cannot redistribute)

Run the processing pipeline against these samples:
```bash
# From MedPath root
python scripts/process_datasets.py \
  --datasets cdr ncbi medmentions adr cometa mantra-gsc \
  --use-local-umls \
  --umls-path ./umls/2025AA \
  --output-root ./data_processed --subdir-suffix _sample \
  --data-root ./data_sample \
  --limit 10
```

Run path extraction against combined mappings (if you have them), or use a mini mapping:
```bash
python scripts/path_extraction/extract_hierarchical_paths.py \
  --vocab MESH \
  --output-dir ./data_processed/hierarchical_paths_sample \
  --mappings-path ./data_processed/mappings/combined_cui_to_vocab_codes_with_tty.json
```

Notes:
- Samples preserve original formats expected by the processors (PubTator, XML, CSV/TXT, txt+ann).
- Document limits are small; final statistics may be trivial.
- Replace `--umls-path` with your local UMLS installation path.





