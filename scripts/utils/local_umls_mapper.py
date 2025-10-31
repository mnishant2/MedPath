#!/usr/bin/env python3
"""
Local UMLS mapper that uses downloaded UMLS files instead of API calls.
Much faster than API-based mapping for large datasets.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from tqdm import tqdm
import pickle
from collections import defaultdict

class LocalUMLSMapper:
    """Class for mapping biomedical entity mentions to UMLS CUIs using local UMLS files."""
    
    def __init__(self, umls_path: str, dataset_name: str = "default", cache_dir: Optional[str] = None):
        """
        Initialize the local UMLS mapper.
        
        Args:
            umls_path: Path to UMLS installation (e.g., "../umls/2025AA")
            dataset_name: Name of the dataset being processed
            cache_dir: Directory for cache files
        """
        self.umls_path = Path(umls_path)
        self.dataset_name = dataset_name
        self.meta_path = self.umls_path / "META"
        
        # Set up cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "cache" / dataset_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file paths
        self.mrconso_cache = self.cache_dir / "mrconso_mappings.pkl"
        self.cui_names_cache = self.cache_dir / "cui_names_mappings.pkl"
        self.mrsty_cache = self.cache_dir / "mrsty_mappings.pkl"
        self.cui_vocab_tty_cache = self.cache_dir / "cui_vocab_tty_mappings.pkl"
        self.name_to_cui_cache = self.cache_dir / "name_to_cui.pkl"
        self.all_eng_names_cache = self.cache_dir / "all_eng_names.pkl"
        self.name_to_cui_cache = self.cache_dir / "name_to_cui.pkl"
        self.nn_buckets_cache = self.cache_dir / "nn_buckets.pkl"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"LocalUMLSMapper_{dataset_name}")
        
        # Load or build mappings
        self.source_to_cui, self.cui_to_names = self._load_or_build_mrconso_mappings()
        self.cui_to_semantic_types = self._load_or_build_mrsty_mappings()
        self.cui_to_vocab_codes_with_tty = self._load_or_build_cui_vocab_tty_mappings()
        # Fast indices for exact/substring fallback over all ENG names
        self.name_to_cui = self._load_or_build_name_to_cui()
        self.all_eng_names = self._load_or_build_all_eng_names()
        # Bucketized index by first 1-2 normalized chars for faster substring search (persisted)
        self.nn_buckets = self._load_or_build_nn_buckets(self.all_eng_names)
        # Per-text cache to avoid repeated scans
        self.text_to_cui_cache: Dict[str, Optional[str]] = {}
        # Parallel cache for mapping method used for text (exact_match / semantic_containment)
        self.text_to_cui_method_cache: Dict[str, str] = {}
        # Last text mapping method used (set by get_cui_from_text)
        self.last_text_mapping_method: Optional[str] = None
        
        self.logger.info(f"Local UMLS mapper initialized with {len(self.source_to_cui)} source mappings")
        self.logger.info(f"Semantic types available for {len(self.cui_to_semantic_types)} CUIs")
    
    def _load_or_build_mrconso_mappings(self) -> tuple[Dict[str, str], Dict[str, str]]:
        """Load or build MRCONSO mappings from source codes to CUIs and CUI names."""
        if self.mrconso_cache.exists() and self.cui_names_cache.exists():
            self.logger.info("Loading MRCONSO mappings from cache...")
            with open(self.mrconso_cache, 'rb') as f:
                source_to_cui = pickle.load(f)
            with open(self.cui_names_cache, 'rb') as f:
                cui_to_names = pickle.load(f)
            return source_to_cui, cui_to_names
        
        self.logger.info("Building MRCONSO mappings from UMLS files...")
        return self._build_mrconso_mappings()
    
    def _build_mrconso_mappings(self) -> tuple[Dict[str, str], Dict[str, str]]:
        """Build mappings from MRCONSO.RRF file using optimized direct file reading."""
        mrconso_file = self.meta_path / "MRCONSO.RRF"
        
        if not mrconso_file.exists():
            raise FileNotFoundError(f"MRCONSO.RRF not found at {mrconso_file}")
        
        source_to_cui = {}
        cui_to_names = {}
        
        # MRCONSO.RRF format: CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF
        # We need: CUI (0), SAB (11), CODE (13), STR (14)
        
        self.logger.info("Reading MRCONSO.RRF file (optimized)...")
        
        # Get file size for progress tracking
        file_size = mrconso_file.stat().st_size
        processed_bytes = 0
        
        with open(mrconso_file, 'r', encoding='utf-8', buffering=8192*16) as f:
            # Process in larger chunks for better performance
            chunk_lines = []
            chunk_size = 50000  # Process 50k lines at a time
            
            with tqdm(total=file_size, desc="Processing MRCONSO file", unit='B', unit_scale=True) as pbar:
                for line_num, line in enumerate(f):
                    processed_bytes += len(line.encode('utf-8'))
                    
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    chunk_lines.append(line.strip())
                    
                    # Process chunk when it reaches the size limit
                    if len(chunk_lines) >= chunk_size:
                        self._process_mrconso_chunk(chunk_lines, source_to_cui, cui_to_names)
                        chunk_lines = []
                        pbar.update(processed_bytes - pbar.n)
                
                # Process remaining lines
                if chunk_lines:
                    self._process_mrconso_chunk(chunk_lines, source_to_cui, cui_to_names)
                    pbar.update(processed_bytes - pbar.n)
        
        # Save to cache
        self.logger.info(f"Built {len(source_to_cui)} source-to-CUI mappings")
        self.logger.info(f"Built {len(cui_to_names)} CUI-to-name mappings")
        
        with open(self.mrconso_cache, 'wb') as f:
            pickle.dump(source_to_cui, f)
        
        with open(self.cui_names_cache, 'wb') as f:
            pickle.dump(cui_to_names, f)
        
        return source_to_cui, cui_to_names

    def _load_or_build_name_to_cui(self) -> Dict[str, str]:
        """Load or build exact English name -> CUI index (lowercased)."""
        if self.name_to_cui_cache.exists():
            try:
                with open(self.name_to_cui_cache, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        mapping = self._build_name_to_cui()
        try:
            with open(self.name_to_cui_cache, 'wb') as f:
                pickle.dump(mapping, f)
        except Exception:
            pass
        return mapping

    def _build_name_to_cui(self) -> Dict[str, str]:
        """Scan MRCONSO to build name_lower -> preferred CUI mapping (ENG only)."""
        mrconso_file = self.meta_path / "MRCONSO.RRF"
        if not mrconso_file.exists():
            return {}
        preferred_ttys = {"PT", "PN", "PF"}
        name_index: Dict[str, tuple[str, str, str]] = {}
        with open(mrconso_file, 'r', encoding='utf-8', buffering=8192*16) as f:
            for line in f:
                if not line.strip():
                    continue
                fields = line.split('|', 15)
                if len(fields) < 15:
                    continue
                cui = fields[0]
                lat = fields[1]
                ispref = fields[6]
                tty = fields[12]
                name = fields[14]
                if lat != 'ENG' or not name:
                    continue
                key = name.lower().strip()
                if key not in name_index:
                    name_index[key] = (cui, ispref, tty)
                else:
                    cur_cui, cur_pref, cur_tty = name_index[key]
                    if cur_pref != 'Y' and ispref == 'Y':
                        name_index[key] = (cui, ispref, tty)
                    elif (cur_tty not in preferred_ttys) and (tty in preferred_ttys):
                        name_index[key] = (cui, ispref, tty)
        return {k: v[0] for k, v in name_index.items()}

    def _normalize_text(self, s: str) -> str:
        return ''.join(ch for ch in s.lower().strip() if ch.isalnum())

    def _load_or_build_all_eng_names(self) -> List[tuple]:
        """Load or build list of (cui, name_lower, normalized_name) for all ENG atoms."""
        if self.all_eng_names_cache.exists():
            try:
                with open(self.all_eng_names_cache, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        entries = self._build_all_eng_names()
        try:
            with open(self.all_eng_names_cache, 'wb') as f:
                pickle.dump(entries, f)
        except Exception:
            pass
        return entries

    def _build_all_eng_names(self) -> List[tuple]:
        mrconso_file = self.meta_path / "MRCONSO.RRF"
        if not mrconso_file.exists():
            return []
        entries: List[tuple] = []
        with open(mrconso_file, 'r', encoding='utf-8', buffering=8192*16) as f:
            for line in f:
                if not line.strip():
                    continue
                fields = line.split('|', 15)
                if len(fields) < 15:
                    continue
                cui = fields[0]
                lat = fields[1]
                name = fields[14]
                if lat != 'ENG' or not name:
                    continue
                nlow = name.lower().strip()
                nnorm = self._normalize_text(nlow)
                if nnorm:
                    entries.append((cui, nlow, nnorm))
        return entries

    def _load_or_build_nn_buckets(self, entries: List[tuple]) -> Dict[str, List[tuple]]:
        """Load or build normalized-name buckets used for fast substring pruning."""
        if self.nn_buckets_cache.exists():
            try:
                with open(self.nn_buckets_cache, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        buckets = self._build_nn_buckets(entries)
        try:
            with open(self.nn_buckets_cache, 'wb') as f:
                pickle.dump(buckets, f)
        except Exception:
            pass
        return buckets

    def _build_nn_buckets(self, entries: List[tuple]) -> Dict[str, List[tuple]]:
        buckets: Dict[str, List[tuple]] = {}
        for cui, nlow, nn in entries:
            if not nn:
                continue
            key2 = nn[:2] if len(nn) >= 2 else nn
            key1 = nn[:1]
            for key in {key1, key2}:
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append((cui, nlow, nn))
        return buckets
    
    def _process_mrconso_chunk(self, lines: List[str], source_to_cui: Dict[str, str], cui_to_names: Dict[str, str]):
        """Process a chunk of MRCONSO lines efficiently."""
        # Pre-compile common values for faster comparison
        empty_values = {'', 'nan', 'NULL', 'null'}
        
        # English-only vocabulary sources (remove non-English versions)
        english_only_vocabs = {
            'MSH', 'SNOMEDCT_US', 'RXNORM', 'VANDF', 'NDDF', 'MTHSPL', 'DRUGBANK',
            'NDFRT', 'MMSL', 'VANDF', 'USPMG', 'GS', 'MTHMST', 'NCBI', 'OMIM',
            'HPO', 'HGNC', 'GO', 'CHEBI', 'MESH', 'ICD10CM', 'ICD9CM', 'CPT',
            'HCPCS', 'LOINC', 'RADLEX', 'FMA', 'NCI', 'MEDLINEPLUS', 'UWDA',
            'AIR', 'BI', 'CCC', 'CCPSS', 'CCS', 'CDT', 'CHV', 'COSTAR', 'CPM',
            'CSP', 'CST', 'CVX', 'DXP', 'HCDT', 'HCPT', 'HCPCS', 'HL7V2.5',
            'HL7V3.0', 'ICPC', 'LCH', 'LCH_NW', 'LNC', 'MCM', 'MDDB', 'MDREX',
            'MEDX', 'MTHMST', 'MTHSPL', 'NCBI', 'NDFRT', 'NEWT', 'NIC', 'NOC',
            'NUCCPT', 'OMS', 'PCDS', 'PDQ', 'PSY', 'QMR', 'RAM', 'RCD', 'SNMI',
            'SNM', 'SNOMED', 'SOP', 'SPN', 'TKMT', 'UMD', 'USPMG', 'UWDA', 'WHO',
            'MDR'  # ADD MedDRA for ADR dataset support
        }
        
        for line in lines:
            # Split by pipe character - limit splits for performance
            fields = line.split('|', 15)  # Need 15 to get STR at index 14
            
            # Ensure we have enough fields (need at least 15 for STR at index 14)
            if len(fields) < 15:
                continue
            
            cui = fields[0]      # CUI (position 0)
            lat = fields[1]      # LAT (position 1) - Language
            sab = fields[11]     # SAB (position 11) 
            code = fields[13]    # CODE (position 13)
            concept_name = fields[14]  # STR (position 14)
            
            # Skip if code is empty or invalid (using set lookup for speed)
            if not code or code in empty_values:
                continue
            
            # Filter to English-only vocabularies
            if sab not in english_only_vocabs:
                continue
            
            # Create mapping key - use direct string concatenation for speed
            key = sab + ':' + code
            
            # Only store first occurrence (faster than checking existence)
            if key not in source_to_cui:
                source_to_cui[key] = cui
            
            # Store concept name for CUI - prioritize English names
            if cui not in cui_to_names:
                cui_to_names[cui] = concept_name
            else:
                # Prioritize English names
                if lat == 'ENG' and concept_name:
                    cui_to_names[cui] = concept_name
                elif lat == 'ENG' and not cui_to_names[cui]:
                    cui_to_names[cui] = concept_name
                elif lat != 'ENG' and not cui_to_names[cui]:
                    cui_to_names[cui] = concept_name
    
    def _load_or_build_mrsty_mappings(self) -> Dict[str, List[str]]:
        """Load or build MRSTY mappings from CUIs to semantic types."""
        if self.mrsty_cache.exists():
            self.logger.info("Loading MRSTY mappings from cache...")
            with open(self.mrsty_cache, 'rb') as f:
                return pickle.load(f)
        
        self.logger.info("Building MRSTY mappings from UMLS files...")
        return self._build_mrsty_mappings()
    
    def _build_mrsty_mappings(self) -> Dict[str, List[str]]:
        """Build semantic type mappings from MRSTY.RRF file using optimized reading."""
        mrsty_file = self.meta_path / "MRSTY.RRF"
        
        if not mrsty_file.exists():
            raise FileNotFoundError(f"MRSTY.RRF not found at {mrsty_file}")
        
        cui_to_semantic_types = defaultdict(list)
        
        # MRSTY.RRF format: CUI|TUI|STN|STY|ATUI|CVF
        # We need: CUI (0), TUI (1), STY (3)
        
        self.logger.info("Reading MRSTY.RRF file (optimized)...")
        
        # Get file size for progress tracking
        file_size = mrsty_file.stat().st_size
        processed_bytes = 0
        
        with open(mrsty_file, 'r', encoding='utf-8', buffering=8192*16) as f:
            with tqdm(total=file_size, desc="Processing MRSTY file", unit='B', unit_scale=True) as pbar:
                for line in f:
                    processed_bytes += len(line.encode('utf-8'))
                    
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    # Split by pipe character
                    fields = line.strip().split('|')
                    
                    # Ensure we have enough fields
                    if len(fields) >= 4:
                        cui = fields[0]
                        tui = fields[1] 
                        # sty = fields[3]  # Not needed for our use case
                        
                        cui_to_semantic_types[cui].append(tui)
                    
                    # Update progress every 1000 lines for better performance
                    if processed_bytes - pbar.n > 1024*1024:  # Update every MB
                        pbar.update(processed_bytes - pbar.n)
                
                # Final update
                pbar.update(processed_bytes - pbar.n)
        
        # Convert to regular dict
        cui_to_semantic_types = dict(cui_to_semantic_types)
        
        # Save to cache
        self.logger.info(f"Built semantic type mappings for {len(cui_to_semantic_types)} CUIs")
        with open(self.mrsty_cache, 'wb') as f:
            pickle.dump(cui_to_semantic_types, f)
        
        return cui_to_semantic_types
    
    def _load_or_build_cui_vocab_tty_mappings(self) -> Dict[str, Dict[str, Dict[str, Dict]]]:
        """Load or build CUI to vocabulary codes with TTY mappings."""
        if self.cui_vocab_tty_cache.exists():
            self.logger.info("Loading CUI-vocab-TTY mappings from cache...")
            with open(self.cui_vocab_tty_cache, 'rb') as f:
                return pickle.load(f)
        
        self.logger.info("Building CUI-vocab-TTY mappings from UMLS files...")
        return self._build_cui_vocab_tty_mappings()
    
    def _build_cui_vocab_tty_mappings(self) -> Dict[str, Dict[str, Dict[str, Dict]]]:
        """
        Build comprehensive CUI to vocabulary codes with TTY mappings from MRCONSO.
        
        Returns:
            dict: {cui: {vocab: {code: {"tty": tty, "term": term}}}}
        """
        mrconso_file = self.meta_path / "MRCONSO.RRF"
        
        if not mrconso_file.exists():
            raise FileNotFoundError(f"MRCONSO.RRF not found at {mrconso_file}")
        
        cui_vocab_tty = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        
        # English-only vocabulary sources
        english_only_vocabs = {
            'MSH', 'SNOMEDCT_US', 'RXNORM', 'VANDF', 'NDDF', 'MTHSPL', 'DRUGBANK',
            'NDFRT', 'MMSL', 'VANDF', 'USPMG', 'GS', 'MTHMST', 'NCBI', 'OMIM',
            'HPO', 'HGNC', 'GO', 'CHEBI', 'MESH', 'ICD10CM', 'ICD9CM', 'CPT',
            'HCPCS', 'LOINC', 'RADLEX', 'FMA', 'NCI', 'MEDLINEPLUS', 'UWDA',
            'AIR', 'BI', 'CCC', 'CCPSS', 'CCS', 'CDT', 'CHV', 'COSTAR', 'CPM',
            'CSP', 'CST', 'CVX', 'DXP', 'HCDT', 'HCPT', 'HCPCS', 'HL7V2.5',
            'HL7V3.0', 'ICPC', 'LCH', 'LCH_NW', 'LNC', 'MCM', 'MDDB', 'MDREX',
            'MEDX', 'MTHMST', 'MTHSPL', 'NCBI', 'NDFRT', 'NEWT', 'NIC', 'NOC',
            'NUCCPT', 'OMS', 'PCDS', 'PDQ', 'PSY', 'QMR', 'RAM', 'RCD', 'SNMI',
            'SNM', 'SNOMED', 'SOP', 'SPN', 'TKMT', 'UMD', 'USPMG', 'UWDA', 'WHO',
            'MDR'  # ADD MedDRA for ADR dataset support
        }
        
        # MRCONSO.RRF format: CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF
        # We need: CUI (0), SAB (11), TTY (12), CODE (13), STR (14)
        
        self.logger.info("Reading MRCONSO.RRF file for TTY mappings...")
        
        # Get file size for progress tracking
        file_size = mrconso_file.stat().st_size
        processed_bytes = 0
        empty_values = {'', 'nan', 'NULL', 'null'}
        
        with open(mrconso_file, 'r', encoding='utf-8', buffering=8192*16) as f:
            with tqdm(total=file_size, desc="Building CUI-vocab-TTY mappings", unit='B', unit_scale=True) as pbar:
                for line_num, line in enumerate(f):
                    processed_bytes += len(line.encode('utf-8'))
                    
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    # Split by pipe character
                    fields = line.split('|', 15)  # Need 15 to get STR at index 14
                    
                    # Ensure we have enough fields
                    if len(fields) < 15:
                        continue
                    
                    cui = fields[0]      # CUI (position 0)
                    lat = fields[1]      # LAT (position 1) - Language
                    sab = fields[11]     # SAB (position 11) - Source vocabulary
                    tty = fields[12]     # TTY (position 12) - Term type
                    code = fields[13]    # CODE (position 13)
                    concept_name = fields[14]  # STR (position 14) - Term
                    
                    # Skip if code is empty or invalid
                    if not code or code in empty_values:
                        continue
                    
                    # Filter to English-only vocabularies
                    if sab not in english_only_vocabs:
                        continue
                    
                    # Skip non-English terms for cleaner data
                    if lat != 'ENG':
                        continue
                    
                    # Store code with TTY and term info
                    if code not in cui_vocab_tty[cui][sab]:
                        cui_vocab_tty[cui][sab][code] = {
                            'tty': tty,
                            'term': concept_name
                        }
                    else:
                        # If multiple TTYs for same code, collect them
                        existing_tty = cui_vocab_tty[cui][sab][code]['tty']
                        if tty and tty not in existing_tty.split('|'):
                            cui_vocab_tty[cui][sab][code]['tty'] = f"{existing_tty}|{tty}" if existing_tty else tty
                        
                        # Keep the preferred term (first one or override with better term)
                        if not cui_vocab_tty[cui][sab][code]['term'] or concept_name:
                            cui_vocab_tty[cui][sab][code]['term'] = concept_name
                    
                    # Update progress every MB for better performance
                    if processed_bytes - pbar.n > 1024*1024:
                        pbar.update(processed_bytes - pbar.n)
                
                # Final update
                pbar.update(processed_bytes - pbar.n)
        
        # Convert to regular dicts
        result = {}
        for cui, vocabs in cui_vocab_tty.items():
            result[cui] = {}
            for vocab, codes in vocabs.items():
                result[cui][vocab] = dict(codes)
        
        # Save to cache
        self.logger.info(f"Built CUI-vocab-TTY mappings for {len(result)} CUIs")
        with open(self.cui_vocab_tty_cache, 'wb') as f:
            pickle.dump(result, f)
        
        return result
    
    def get_enhanced_vocabularies_for_cui(self, cui: str) -> Dict[str, Dict[str, Dict]]:
        """
        Get vocabulary codes with TTY information for a CUI from local files.
        
        Args:
            cui (str): The UMLS CUI to query
            
        Returns:
            dict: Mapping of {vocab: {code: {"tty": tty_code, "term": term_text}}}
        """
        if not cui or cui == '' or cui == 'nan':
            return {}
            
        # Clean the CUI format
        cui = str(cui).strip()
        if not cui.startswith('C'):
            cui = f"C{cui}"
        
        return self.cui_to_vocab_codes_with_tty.get(cui, {})
    
    def get_cui_from_ontology_id(self, code: str, source: str, code_type: str = "") -> Optional[str]:
        """
        Get UMLS CUI from ontology ID using local files.
        
        Args:
            code: The ontology code
            source: The source vocabulary (e.g., 'MDR', 'SNOMEDCT_US', 'MSH')
            code_type: Type of code (optional, for compatibility)
        
        Returns:
            UMLS CUI if found, None otherwise
        """
        if not code or code == '' or code == 'nan':
            return None
        
        # Clean the code - handle different formats
        code = str(code).strip()
        
        # Remove decimal parts
        if '.' in code:
            code = code.split('.')[0]
        
        # Skip very short codes that are likely invalid
        if len(code) < 3:
            return None
        
        # Special case: if source is 'UMLS' and code looks like a CUI, directly validate it
        if source == 'UMLS' and code.startswith('C') and len(code) == 8:
            # Check if this CUI exists in our UMLS data
            if code in self.cui_to_names:
                return code
            else:
                return None
        
        # Try exact match first
        key = f"{source}:{code}"
        if key in self.source_to_cui:
            return self.source_to_cui[key]
        
        # For MeSH, support D-prefixed descriptors and C-prefixed SCRs.
        if source == 'MSH':
            # Try code directly (covers Dxxxx and Cxxxx SCRs when present)
            direct_key = f"{source}:{code}"
            if direct_key in self.source_to_cui:
                return self.source_to_cui[direct_key]
            # Try with and without D prefix for descriptor keys only
            if code.startswith('D'):
                alt_key = f"{source}:{code[1:]}"
                if alt_key in self.source_to_cui:
                    return self.source_to_cui[alt_key]
            else:
                alt_key = f"{source}:D{code}"
                if alt_key in self.source_to_cui:
                    return self.source_to_cui[alt_key]
        
        # For SNOMED, try alternative formats
        if source == 'SNOMEDCT_US':
            # Try without any prefixes
            clean_code = code.lstrip('0')
            if clean_code != code:
                alt_key = f"{source}:{clean_code}"
                if alt_key in self.source_to_cui:
                    return self.source_to_cui[alt_key]
            
            # Try with leading zeros stripped
            if code.startswith('0'):
                alt_key = f"{source}:{code.lstrip('0')}"
                if alt_key in self.source_to_cui:
                    return self.source_to_cui[alt_key]
        
        # For MedDRA, try different formats
        if source == 'MDR':
            # Try as integer (remove leading zeros)
            try:
                int_code = str(int(code))
                if int_code != code:
                    alt_key = f"{source}:{int_code}"
                    if alt_key in self.source_to_cui:
                        return self.source_to_cui[alt_key]
            except (ValueError, TypeError):
                pass
        
        return None
    
    def get_cui_from_text(self, text: str) -> Optional[str]:
        """
        Get UMLS CUI from mention text using local files.
        Implements text-to-CUI mapping by searching concept names.
        
        Args:
            text: The mention text
        
        Returns:
            UMLS CUI if found, None otherwise
        """
        if not text or text == '' or text == 'nan':
            return None
        
        # Clean and normalize text for matching
        text_lower = text.lower().strip()
        if text_lower in self.text_to_cui_cache:
            # Restore last method from cache for downstream consumers
            self.last_text_mapping_method = self.text_to_cui_method_cache.get(text_lower)
            return self.text_to_cui_cache[text_lower]
        
        # Direct exact match (case-insensitive) via fast index
        fast_cui = self.name_to_cui.get(text_lower)
        if fast_cui:
            self.text_to_cui_cache[text_lower] = fast_cui
            self.text_to_cui_method_cache[text_lower] = 'exact_match'
            self.last_text_mapping_method = 'exact_match'
            return fast_cui
        # Fallback to preferred-name dict
        for cui, name in self.cui_to_names.items():
            if name and name.lower() == text_lower:
                self.text_to_cui_cache[text_lower] = cui
                self.text_to_cui_method_cache[text_lower] = 'exact_match'
                self.last_text_mapping_method = 'exact_match'
                return cui
        
        # Substring matching across all ENG names using normalized forms (bidirectional containment)
        norm_text = self._normalize_text(text_lower)
        best_cui = None
        # scoring: prioritize higher token overlap; break ties by smaller length difference
        best_tuple = None  # (overlap_count, -length_diff)
        # prepare tokens from original lowercased text (length>=3)
        import re as _re
        text_tokens = [tok for tok in _re.split(r"[^a-z0-9]+", text_lower) if len(tok) >= 3]
        if norm_text:
            # Use buckets to prune search
            candidates = []
            k2 = norm_text[:2] if len(norm_text) >= 2 else norm_text
            k1 = norm_text[:1]
            if k2 in self.nn_buckets:
                candidates.extend(self.nn_buckets[k2])
            if k1 in self.nn_buckets:
                candidates.extend(self.nn_buckets[k1])
            seen = set()
            for cui, nlow, nn in candidates if candidates else self.all_eng_names:
                if nn in seen:
                    continue
                seen.add(nn)
                if norm_text in nn or nn in norm_text:
                    # compute token overlap on non-normalized lowercased strings
                    cand_tokens = [tok for tok in _re.split(r"[^a-z0-9]+", nlow) if len(tok) >= 3]
                    overlap = len(set(text_tokens) & set(cand_tokens)) if text_tokens else 0
                    length_diff = abs(len(nn) - len(norm_text))
                    score_tuple = (overlap, -length_diff)
                    if (best_tuple is None) or (score_tuple > best_tuple):
                        best_cui = cui
                        best_tuple = score_tuple
        if best_cui:
            self.text_to_cui_cache[text_lower] = best_cui
            self.text_to_cui_method_cache[text_lower] = 'semantic_containment'
            self.last_text_mapping_method = 'semantic_containment'
            return best_cui
        
        # Do not use reverse substring matching (disabled)
        
        self.text_to_cui_cache[text_lower] = None
        self.last_text_mapping_method = None
        return None
    
    def get_semantic_types(self, cui: str) -> List[str]:
        """
        Get semantic types for a CUI using local files.
        
        Args:
            cui: The UMLS CUI
        
        Returns:
            List of semantic type TUIs
        """
        if not cui or cui == '':
            return []
        
        return self.cui_to_semantic_types.get(cui, [])
    
    def get_umls_name(self, cui: str) -> str:
        """
        Get the preferred name for a CUI.
        
        Args:
            cui: The UMLS CUI
        
        Returns:
            UMLS concept name if found, empty string otherwise
        """
        if not cui or cui == '':
            return ''
        
        return self.cui_to_names.get(cui, '')

    def get_closest_umls_name_for_text(self, cui: str, text: str) -> str:
        """
        Choose the closest UMLS synonym for the given CUI relative to the mention text.
        Strategy:
          - If the mention text (case-insensitive) is a synonym of this CUI, return the
            mention text as-is (preserve casing as seen in the document).
          - Else, scan all English synonyms for the CUI (from cached MRCONSO terms in
            cui_to_vocab_tty) and return the one with the highest token overlap; break
            ties by minimal normalized-length difference. Fall back to preferred name.
        """
        if not cui:
            return ''
        mention = (text or '').strip()
        if not mention:
            return self.get_umls_name(cui)

        mention_lower = mention.lower().strip()
        norm_mention = self._normalize_text(mention_lower)

        # Collect all ENG synonyms with original casing from cui_to_vocab_tty
        synonyms: List[str] = []
        try:
            vocabs = self.cui_to_vocab_tty.get(cui, {})
            for src, codes in vocabs.items():
                for code, info in codes.items():
                    term = (info or {}).get('term') or ''
                    if term:
                        synonyms.append(term)
        except Exception:
            pass

        # Include preferred name as a candidate if not already
        pref = self.cui_to_names.get(cui, '')
        if pref:
            synonyms.append(pref)

        # Deduplicate by lower-case
        seen = set()
        uniq_syns: List[str] = []
        for s in synonyms:
            sl = s.lower().strip()
            if sl and sl not in seen:
                seen.add(sl)
                uniq_syns.append(s)

        if not uniq_syns:
            return self.get_umls_name(cui)

        # If mention text exactly equals any synonym (case-insensitive), prefer mention text
        if mention_lower in seen:
            return mention

        # Score by token overlap then by normalized length difference
        import re as _re
        mention_tokens = {tok for tok in _re.split(r"[^a-z0-9]+", mention_lower) if len(tok) >= 3}

        best_name = None
        best_score = None  # tuple(overlap, -len_diff)

        for syn in uniq_syns:
            syn_lower = syn.lower().strip()
            syn_norm = self._normalize_text(syn_lower)
            syn_tokens = {tok for tok in _re.split(r"[^a-z0-9]+", syn_lower) if len(tok) >= 3}
            overlap = len(mention_tokens & syn_tokens) if mention_tokens else 0
            len_diff = abs(len(syn_norm) - len(norm_mention)) if syn_norm and norm_mention else abs(len(syn_lower) - len(mention_lower))
            score = (overlap, -len_diff)
            if (best_score is None) or (score > best_score):
                best_score = score
                best_name = syn

        return best_name or self.get_umls_name(cui)
    
    def validate_cui(self, cui: str) -> bool:
        """
        Validate if a CUI exists in UMLS.
        
        Args:
            cui: The CUI to validate
        
        Returns:
            True if CUI exists, False otherwise
        """
        return cui in self.cui_to_semantic_types
    
    def get_comprehensive_vocabularies_for_cui(self, cui: str) -> Dict[str, List[str]]:
        """
        Get comprehensive vocabulary mappings for a CUI.
        This would require processing MRCONSO for reverse mappings.
        
        Args:
            cui: The UMLS CUI
        
        Returns:
            Dictionary mapping vocabulary sources to codes
        """
        # This would require building a reverse index from CUI to source codes
        # For now, return empty dict
        return {}
    
    def get_comprehensive_semantic_types_for_cui(self, cui: str) -> List[str]:
        """
        Get comprehensive semantic types for a CUI.
        Same as get_semantic_types for local implementation.
        
        Args:
            cui: The UMLS CUI
        
        Returns:
            List of semantic type TUIs
        """
        return self.get_semantic_types(cui)
    
    def create_comprehensive_mapping_files(self, cuis: Set[str], output_dir: str, dataset_name: str) -> Dict[str, str]:
        """
        Create comprehensive mapping files for a set of CUIs using local UMLS data.
        
        Args:
            cuis: Set of CUIs to create mappings for
            output_dir: Directory to save mapping files
            dataset_name: Name of the dataset
        
        Returns:
            Dictionary mapping file types to file paths
        """
        output_path = Path(output_dir)
        mappings_path = output_path / "mappings"
        mappings_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating comprehensive mappings for {len(cuis)} CUIs...")
        
        # Create CUI to semantic types mapping
        cui_to_tuis = {}
        for cui in tqdm(cuis, desc="Creating semantic type mappings"):
            tuis = self.get_semantic_types(cui)
            if tuis:
                cui_to_tuis[cui] = tuis
        
        # Create CUI to vocabulary codes mapping by reverse lookup
        cui_to_vocab_codes = defaultdict(lambda: defaultdict(list))
        
        # Reverse lookup from source_to_cui to build cui_to_vocab_codes
        for source_code_key, cui in tqdm(self.source_to_cui.items(), desc="Building vocabulary mappings"):
            if cui in cuis:
                # Parse the key: "SAB:CODE"
                if ':' in source_code_key:
                    source, code = source_code_key.split(':', 1)
                    cui_to_vocab_codes[cui][source].append(code)
        
        # Convert to regular dict and remove duplicates
        cui_to_vocab_final = {}
        for cui, sources in cui_to_vocab_codes.items():
            cui_to_vocab_final[cui] = {}
            for source, codes in sources.items():
                cui_to_vocab_final[cui][source] = list(set(codes))  # Remove duplicates
        
        # Save semantic types mapping
        tui_file = mappings_path / f"{dataset_name}_cui_to_tuis.json"
        with open(tui_file, 'w') as f:
            json.dump(cui_to_tuis, f, indent=2)
        
        # Save vocabulary mappings
        vocab_file = mappings_path / f"{dataset_name}_cui_to_vocab_codes.json"
        with open(vocab_file, 'w') as f:
            json.dump(cui_to_vocab_final, f, indent=2)
        
        # Create mapping summary
        summary = {
            'dataset_name': dataset_name,
            'total_cuis': len(cuis),
            'cuis_with_semantic_types': len(cui_to_tuis),
            'cuis_with_vocab_mappings': len(cui_to_vocab_final),
            'vocabulary_sources': list(set(source for cui_vocabs in cui_to_vocab_final.values() 
                                         for source in cui_vocabs.keys())),
            'semantic_types_found': list(set(tui for tuis in cui_to_tuis.values() for tui in tuis))
        }
        
        summary_file = mappings_path / f"{dataset_name}_mapping_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Created comprehensive mapping files:")
        self.logger.info(f"  - {len(cui_to_tuis)} CUIs with semantic types")
        self.logger.info(f"  - {len(cui_to_vocab_final)} CUIs with vocabulary mappings")
        self.logger.info(f"  - {len(summary['vocabulary_sources'])} vocabulary sources")
        
        return {
            'cui_to_tuis': str(tui_file),
            'cui_to_vocab_codes': str(vocab_file),
            'mapping_summary': str(summary_file)
        }
    
    def create_enhanced_mapping_files(self, cuis: Set[str], output_dir: str, dataset_name: str) -> Dict[str, str]:
        """
        Create comprehensive CUI-to-vocabulary mapping files with TTY information.
        
        Args:
            cuis: Set of CUIs to analyze
            output_dir: Output directory for mapping files  
            dataset_name: Name of the dataset
            
        Returns:
            dict: Dictionary with paths to created mapping files
        """
        self.logger.info(f"Creating enhanced mapping files with TTY for {len(cuis)} CUIs")
        
        # Create mappings directory
        mappings_path = Path(output_dir) / "mappings" 
        mappings_path.mkdir(parents=True, exist_ok=True)
        
        # Filter valid CUIs
        valid_cuis = [cui for cui in cuis if cui and cui != '' and cui != 'CUI-less']
        
        # Generate enhanced vocabularies with TTY
        cui_to_vocab_codes_with_tty = {}
        cui_to_tuis = {}
        
        self.logger.info(f"Processing {len(valid_cuis)} CUIs for enhanced mappings...")
        
        for cui in tqdm(valid_cuis, desc="Processing CUIs for TTY"):
            # Get enhanced vocabulary codes with TTY
            enhanced_vocab_codes = self.get_enhanced_vocabularies_for_cui(cui)
            if enhanced_vocab_codes:
                cui_to_vocab_codes_with_tty[cui] = enhanced_vocab_codes
            
            # Get semantic types
            if cui in self.cui_to_semantic_types:
                cui_to_tuis[cui] = self.cui_to_semantic_types[cui]
        
        # Save enhanced vocabulary mappings with TTY
        enhanced_vocab_file = mappings_path / f"{dataset_name}_cui_to_vocab_codes_with_tty.json"
        with open(enhanced_vocab_file, 'w') as f:
            json.dump(cui_to_vocab_codes_with_tty, f, indent=2)
        
        # Save semantic types mapping
        tui_file = mappings_path / f"{dataset_name}_cui_to_tuis.json"
        with open(tui_file, 'w') as f:
            json.dump(cui_to_tuis, f, indent=2)
        
        # Create TTY statistics
        tty_stats = defaultdict(lambda: defaultdict(int))
        total_codes = 0
        
        for cui, vocabs in cui_to_vocab_codes_with_tty.items():
            for vocab, codes in vocabs.items():
                for code, info in codes.items():
                    tty = info.get('tty', '')
                    if tty:
                        # Handle multiple TTYs
                        for t in tty.split('|'):
                            if t.strip():
                                tty_stats[vocab][t.strip()] += 1
                    total_codes += 1
        
        # Save TTY statistics
        tty_stats_file = mappings_path / f"{dataset_name}_tty_statistics.json"
        stats = {
            'total_codes': total_codes,
            'cuis_processed': len(valid_cuis),
            'cuis_with_vocab_codes': len(cui_to_vocab_codes_with_tty),
            'cuis_with_semantic_types': len(cui_to_tuis),
            'tty_distribution_by_vocab': dict(tty_stats),
            'summary': {}
        }
        
        for vocab, ttys in tty_stats.items():
            stats['summary'][vocab] = {
                'total_codes': sum(ttys.values()),
                'unique_ttys': len(ttys),
                'top_ttys': sorted(ttys.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        with open(tty_stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create enhanced mapping summary
        summary = {
            'dataset_name': dataset_name,
            'total_cuis': len(cuis),
            'cuis_with_semantic_types': len(cui_to_tuis),
            'cuis_with_enhanced_vocab_mappings': len(cui_to_vocab_codes_with_tty),
            'vocabulary_sources': list(set(source for cui_vocabs in cui_to_vocab_codes_with_tty.values() 
                                         for source in cui_vocabs.keys())),
            'semantic_types_found': list(set(tui for tuis in cui_to_tuis.values() for tui in tuis)),
            'total_vocabulary_codes': total_codes,
            'vocabularies_with_tty': len(tty_stats)
        }
        
        summary_file = mappings_path / f"{dataset_name}_enhanced_mapping_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Created enhanced mapping files with TTY:")
        self.logger.info(f"  - {len(cui_to_vocab_codes_with_tty)} CUIs with enhanced vocabulary mappings")
        self.logger.info(f"  - {len(cui_to_tuis)} CUIs with semantic types")
        self.logger.info(f"  - {total_codes} total vocabulary codes")
        self.logger.info(f"  - {len(summary['vocabulary_sources'])} vocabulary sources")
        
        # Print TTY summary
        for vocab, summary_info in stats['summary'].items():
            if summary_info['total_codes'] > 0:
                self.logger.info(f"  - {vocab}: {summary_info['total_codes']} codes, {summary_info['unique_ttys']} unique TTYs")
        
        return {
            'cui_to_vocab_codes_with_tty': str(enhanced_vocab_file),
            'cui_to_tuis': str(tui_file),
            'tty_statistics': str(tty_stats_file),
            'enhanced_mapping_summary': str(summary_file)
        }
    
    def close(self):
        """Close the mapper (compatibility method)."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass 