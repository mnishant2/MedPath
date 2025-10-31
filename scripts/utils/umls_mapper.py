#!/usr/bin/env python3
"""
A reusable module for mapping biomedical entity mentions to UMLS CUIs.
Supports different standards including MedDRA, SNOMED CT, MeSH, and direct text mapping.

Adapted from the original UMLSMapper for the streamlined processing pipeline.
"""

import os
import json
import time
import requests
from collections import defaultdict
from tqdm import tqdm
import logging
from typing import Dict, List, Set, Any

class UMLSMapper:
    """Class for mapping biomedical entity mentions to UMLS CUIs."""
    
    def __init__(self, api_key, dataset_name=None, cache_dir=None):
        """
        Initialize the UMLS mapper.
        
        Args:
            api_key (str): UMLS API key
            dataset_name (str, optional): Name of the dataset being processed
            cache_dir (str, optional): Directory for cache files
        """
        self.api_key = api_key
        self.dataset_name = dataset_name or "default"
        self.umls_api_base = "https://uts-ws.nlm.nih.gov/rest"
        
        # Set up cache directory
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "cache", self.dataset_name)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up file paths
        self.cache_file = os.path.join(self.cache_dir, "umls_mapping_cache.json")
        self.debug_log = os.path.join(self.cache_dir, "umls_mapping_debug.log")
        
        # Initialize logging
        self._setup_logging()
        
        # Load or initialize cache
        self._load_cache()
        
        # Create API session
        self.session = self._get_api_session()

        # Track text mapping method for last lookup and cache methods per text
        self.last_text_mapping_method = None  # 'exact_match' | 'semantic_containment' | None
        # Extend cache to store methods for text searches
        if 'text_method' not in self.cache:
            self.cache['text_method'] = {}
        
    def _setup_logging(self):
        """Set up logging for debug information."""
        logging.basicConfig(
            filename=self.debug_log,
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='a'
        )
        self.logger = logging.getLogger(f"UMLSMapper_{self.dataset_name}")
        
    def _load_cache(self):
        """Load the mapping cache from file or initialize a new cache."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        else:
            self.cache = {
                'text': {},
                'meddra_pt': {},
                'meddra_llt': {},
                'snomed_ct': {},
                'mesh': {},
                'icd10': {},
                'icd9': {},
                'omim': {},
                'cui_validation': {},
                'cui_vocabularies': {},  # Cache for comprehensive vocabulary mappings
                'cui_semantic_types': {},  # Cache for semantic types
                'cui_atoms': {},  # Cache for atom information
                'cui_atoms_enhanced': {}  # Cache for enhanced atom information with TTY
            }
    
    def _save_cache(self):
        """Save the mapping cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _get_api_session(self):
        """Create and return an API session with authentication."""
        session = requests.Session()
        session.params = {"apiKey": self.api_key}
        return session
    
    def log_debug(self, message):
        """Log a debug message."""
        self.logger.debug(message)
    
    def get_cui_from_ontology_id(self, code, source, code_type=""):
        """
        Get UMLS CUI from ontology ID (e.g., MedDRA, SNOMED CT, MeSH).
        
        Args:
            code (str): The ontology code
            source (str): The source vocabulary (e.g., 'MDR', 'SNOMEDCT_US', 'MSH')
            code_type (str, optional): Type of code (e.g., 'PT', 'LLT')
        
        Returns:
            str or None: UMLS CUI if found, None otherwise
        """
        # Handle empty or invalid codes
        if code is None or False or code == '' or code == 'nan':
            return None
        
        # Convert to string and clean
        code = str(code).split('.')[0]
        
        # Determine cache key based on source
        cache_key = f"{code}_{source}"
        cache_category = self._get_cache_category(source, code_type)
        
        # Check cache first
        if cache_key in self.cache[cache_category]:
            return self.cache[cache_category][cache_key]
        
        try:
            # Query the UMLS API
            search_endpoint = f"{self.umls_api_base}/search/current"
            params = {
                "string": code,
                "searchType": "exact",
                "inputType": "sourceUi",
                "sabs": source,
                "returnIdType": "concept"
            }
            
            response = self.session.get(search_endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            self.log_debug(f"API response for {source} {code_type} ID {code}: Status {response.status_code}")
            
            # Process results
            if 'result' in data and 'results' in data['result'] and len(data['result']['results']) > 0:
                for result in data['result']['results']:
                    if 'ui' in result:
                        cui = result['ui']
                        self.log_debug(f"Found CUI {cui} for {source} {code_type} ID {code}")
                        
                        # Save to cache
                        self.cache[cache_category][cache_key] = cui
                        self._periodic_cache_save()
                        
                        return cui
            else:
                self.log_debug(f"No results found for {source} {code_type} ID {code}")
            
            # Save empty result to cache
            self.cache[cache_category][cache_key] = None
            return None
        
        except Exception as e:
            self.log_debug(f"Error querying UMLS API for {source} {code_type} ID {code}: {str(e)}")
            time.sleep(1)  # Rate limiting
            return None
    
    def get_cui_from_text(self, text):
        """
        Get UMLS CUI from mention text.
        
        Args:
            text (str): The mention text
        
        Returns:
            str or None: UMLS CUI if found, None otherwise
        """
        # Handle empty text
        if text is None or False or text == '' or text == 'nan':
            return None
        
        # Check cache first
        if text in self.cache['text']:
            # Restore method for downstream consumers
            self.last_text_mapping_method = self.cache.get('text_method', {}).get(text)
            return self.cache['text'][text]
        
        try:
            # Query the UMLS API
            search_endpoint = f"{self.umls_api_base}/search/current"
            params = {
                "string": text,
                "searchType": "exact",
                "returnIdType": "concept"
            }
            
            response = self.session.get(search_endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            preview_text = text[:50] + '...' if len(text) > 50 else text
            self.log_debug(f"API response for text '{preview_text}': Status {response.status_code}")
            
            # Process results
            if 'result' in data and 'results' in data['result'] and len(data['result']['results']) > 0:
                for result in data['result']['results']:
                    if 'ui' in result:
                        cui = result['ui']
                        self.log_debug(f"Found CUI {cui} for text '{preview_text}'")
                        
                        # Save to cache
                        self.cache['text'][text] = cui
                        self.cache['text_method'][text] = 'exact_match'
                        self.last_text_mapping_method = 'exact_match'
                        self._periodic_cache_save()
                        
                        return cui
            else:
                self.log_debug(f"No results found for text '{preview_text}'")
            
            # If exact match failed, try a relaxed search and pick the closest candidate
            try:
                params_words = {
                    "string": text,
                    "searchType": "words",
                    "returnIdType": "concept"
                }
                response2 = self.session.get(search_endpoint, params=params_words)
                response2.raise_for_status()
                data2 = response2.json()
                best_cui = None
                best_score = None  # (overlap, -len_diff)
                best_name = None
                import re as _re
                text_lower = text.lower().strip()
                norm_text = ''.join(ch for ch in text_lower if ch.isalnum())
                text_tokens = {tok for tok in _re.split(r"[^a-z0-9]+", text_lower) if len(tok) >= 3}
                results = data2.get('result', {}).get('results', []) if isinstance(data2, dict) else []
                for res in results:
                    cui = res.get('ui')
                    name = res.get('name') or ''
                    name_lower = (name or '').lower().strip()
                    if not cui or not name_lower:
                        continue
                    name_norm = ''.join(ch for ch in name_lower if ch.isalnum())
                    cand_tokens = {tok for tok in _re.split(r"[^a-z0-9]+", name_lower) if len(tok) >= 3}
                    overlap = len(text_tokens & cand_tokens) if text_tokens else 0
                    len_diff = abs(len(name_norm) - len(norm_text)) if name_norm and norm_text else abs(len(name_lower) - len(text_lower))
                    score = (overlap, -len_diff)
                    if (best_score is None) or (score > best_score):
                        best_score = score
                        best_cui = cui
                        best_name = name
                if best_cui:
                    self.cache['text'][text] = best_cui
                    self.cache['text_method'][text] = 'semantic_containment'
                    self.last_text_mapping_method = 'semantic_containment'
                    self._periodic_cache_save()
                    return best_cui
            except Exception as e2:
                self.log_debug(f"Error in relaxed search for text '{preview_text}': {str(e2)}")

            # Save empty result to cache
            self.cache['text'][text] = None
            self.cache['text_method'][text] = None
            self.last_text_mapping_method = None
            return None
        
        except Exception as e:
            preview_text = text[:50] + '...' if len(text) > 50 else text
            self.log_debug(f"Error querying UMLS API for text '{preview_text}': {str(e)}")
            time.sleep(1)  # Rate limiting
            return None
    
    def validate_cui(self, cui):
        """
        Validate if a UMLS CUI is still valid in the current UMLS version.
        
        Args:
            cui (str): The UMLS CUI to validate
            
        Returns:
            str or None: The same CUI if still valid, None otherwise
        """
        # Handle empty CUI
        if cui is None or False or cui == '' or cui == 'nan':
            return None
            
        # Clean the CUI format
        cui = str(cui).strip()
        if not cui.startswith('C'):
            cui = f"C{cui}"
            
        # Check cache first
        if cui in self.cache['cui_validation']:
            return self.cache['cui_validation'][cui]
            
        try:
            # Query the UMLS API to verify the CUI exists
            content_endpoint = f"{self.umls_api_base}/content/current/CUI/{cui}"
            
            response = self.session.get(content_endpoint)
            self.log_debug(f"API response for validating CUI {cui}: Status {response.status_code}")
            
            if response.status_code == 200:
                self.log_debug(f"CUI {cui} is valid in current UMLS version")
                self.cache['cui_validation'][cui] = cui
                self._periodic_cache_save()
                return cui
            else:
                self.log_debug(f"CUI {cui} is not valid in current UMLS version")
                self.cache['cui_validation'][cui] = None
                return None
                
        except Exception as e:
            self.log_debug(f"Error validating CUI {cui}: {str(e)}")
            time.sleep(1)  # Rate limiting
            return None
    
    def get_semantic_types(self, cui):
        """
        Get semantic types for a UMLS CUI.
        
        Args:
            cui (str): The UMLS CUI
            
        Returns:
            list: List of semantic type codes (TUIs)
        """
        if cui is None or cui == '' or cui == 'nan':
            return []
            
        try:
            content_endpoint = f"{self.umls_api_base}/content/current/CUI/{cui}"
            
            response = self.session.get(content_endpoint)
            self.log_debug(f"API response for semantic types of CUI {cui}: Status {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'result' in data and 'semanticTypes' in data['result']:
                    return [st['TUI'] for st in data['result']['semanticTypes']]
            
            return []
        
        except Exception as e:
            self.log_debug(f"Error getting semantic types for CUI {cui}: {str(e)}")
            time.sleep(1)  # Rate limiting
            return []
    
    def get_umls_name(self, cui: str) -> str:
        """
        Get the preferred name for a CUI using API.
        
        Args:
            cui: The UMLS CUI
        
        Returns:
            UMLS concept name if found, empty string otherwise
        """
        if not cui or cui == '':
            return ''
        
        try:
            # Query the UMLS API for concept details
            concept_endpoint = f"{self.umls_api_base}/content/current/CUI/{cui}"
            
            response = self.session.get(concept_endpoint)
            response.raise_for_status()
            data = response.json()
            
            if 'result' in data:
                name = data['result'].get('name', '')
                return name
            
            return ''
        
        except Exception as e:
            self.log_debug(f"Error getting name for CUI {cui}: {str(e)}")
            return ''

    def get_closest_umls_name_for_text(self, cui: str, text: str) -> str:
        """
        Choose the closest UMLS synonym for the given CUI relative to the mention text
        using API-derived atoms. Prefers the mention text itself when it is a synonym,
        otherwise selects the synonym with highest token overlap, breaking ties by
        minimal normalized length difference. Falls back to preferred name.
        """
        if not cui:
            return ''
        mention = (text or '').strip()
        if not mention:
            return self.get_umls_name(cui)

        mention_lower = mention.lower().strip()
        norm_mention = ''.join(ch for ch in mention_lower if ch.isalnum())

        # Gather synonyms via enhanced vocabularies (includes term text) and include preferred name
        synonyms: List[str] = []
        try:
            vocabs = self.get_enhanced_vocabularies_for_cui(cui)
            for src, codes in vocabs.items():
                for code, info in codes.items():
                    term = (info or {}).get('term') or ''
                    if term:
                        synonyms.append(term)
        except Exception:
            pass
        pref = self.get_umls_name(cui)
        if pref:
            synonyms.append(pref)

        # Deduplicate by lowercase
        seen = set()
        uniq_syns: List[str] = []
        for s in synonyms:
            sl = s.lower().strip()
            if sl and sl not in seen:
                seen.add(sl)
                uniq_syns.append(s)

        if not uniq_syns:
            return self.get_umls_name(cui)

        # If mention text is one of the synonyms (case-insensitive), prefer the mention casing
        if mention_lower in seen:
            return mention

        import re as _re
        mention_tokens = {tok for tok in _re.split(r"[^a-z0-9]+", mention_lower) if len(tok) >= 3}
        best_name = None
        best_score = None

        for syn in uniq_syns:
            syn_lower = syn.lower().strip()
            syn_norm = ''.join(ch for ch in syn_lower if ch.isalnum())
            syn_tokens = {tok for tok in _re.split(r"[^a-z0-9]+", syn_lower) if len(tok) >= 3}
            overlap = len(mention_tokens & syn_tokens) if mention_tokens else 0
            len_diff = abs(len(syn_norm) - len(norm_mention)) if syn_norm and norm_mention else abs(len(syn_lower) - len(mention_lower))
            score = (overlap, -len_diff)
            if (best_score is None) or (score > best_score):
                best_score = score
                best_name = syn

        return best_name or self.get_umls_name(cui)
    
    def _get_cache_category(self, source, code_type):
        """Get the appropriate cache category based on source and code type."""
        if source == "MDR":
            if code_type == "PT":
                return "meddra_pt"
            elif code_type == "LLT":
                return "meddra_llt"
            else:
                return "meddra_pt"  # Default to PT
        elif source in ["SNOMEDCT_US", "SCTSPA", "SNOMEDCT", "SNOMEDCT_CORE"]:
            return "snomed_ct"
        elif source == "MSH":
            return "mesh"
        elif source == "ICD10CM":
            return "icd10"
        elif source == "ICD9CM":
            return "icd9"
        elif source == "OMIM":
            return "omim"
        else:
            return "text"  # Fallback
    
    def _periodic_cache_save(self):
        """Save cache periodically to avoid losing data."""
        total_entries = sum(len(v) for v in self.cache.values())
        if total_entries % 50 == 0:
            self._save_cache()
    
    def close(self):
        """Clean up and save cache."""
        self._save_cache()
        self.log_debug("UMLSMapper session closed and cache saved")

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 

    def get_comprehensive_vocabularies_for_cui(self, cui: str) -> Dict[str, List[str]]:
        """
        Get comprehensive vocabulary information for a CUI using UMLS API.
        
        Args:
            cui (str): The UMLS CUI to query
            
        Returns:
            dict: Dictionary mapping vocabulary sources to lists of codes
        """
        # Handle empty CUI
        if cui is None or False or cui == '' or cui == 'nan':
            return {}
            
        # Clean the CUI format
        cui = str(cui).strip()
        if not cui.startswith('C'):
            cui = f"C{cui}"
            
        # Check cache first
        if cui in self.cache['cui_vocabularies']:
            return self.cache['cui_vocabularies'][cui]
        
        try:
            # Query UMLS API for atoms (which contain source vocabulary info)
            atoms_url = f"{self.umls_api_base}/content/current/CUI/{cui}/atoms"
            
            response = self.session.get(atoms_url)
            self.log_debug(f"API response for vocabularies of CUI {cui}: Status {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract vocabulary codes grouped by source
                vocabulary_codes = defaultdict(list)
                
                if 'result' in data:
                    for atom in data['result']:
                        root_source = atom.get('rootSource', '')
                        code = atom.get('code', '')
                        
                        if root_source and code:
                            vocabulary_codes[root_source].append(code)
                
                # Convert to regular dict and remove duplicates
                result = {}
                for source, codes in vocabulary_codes.items():
                    result[source] = list(set(codes))  # Remove duplicates
                
                # Cache the result
                self.cache['cui_vocabularies'][cui] = result
                self._periodic_cache_save()
                
                self.log_debug(f"Found {len(result)} vocabularies for CUI {cui}: {list(result.keys())}")
                return result
            else:
                self.log_debug(f"Failed to get vocabularies for CUI {cui}: Status {response.status_code}")
                self.cache['cui_vocabularies'][cui] = {}
                return {}
                
        except Exception as e:
            self.log_debug(f"Error getting vocabularies for CUI {cui}: {str(e)}")
            time.sleep(1)  # Rate limiting
            self.cache['cui_vocabularies'][cui] = {}
            return {}

    def get_comprehensive_semantic_types_for_cui(self, cui: str) -> List[str]:
        """
        Get comprehensive semantic type information for a CUI.
        
        Args:
            cui (str): The UMLS CUI to query
            
        Returns:
            list: List of semantic type codes (TUIs)
        """
        # Handle empty CUI
        if cui is None or False or cui == '' or cui == 'nan':
            return []
            
        # Clean the CUI format
        cui = str(cui).strip()
        if not cui.startswith('C'):
            cui = f"C{cui}"
            
        # Check cache first
        if cui in self.cache['cui_semantic_types']:
            return self.cache['cui_semantic_types'][cui]
            
        try:
            # Query the UMLS API for CUI content
            content_endpoint = f"{self.umls_api_base}/content/current/CUI/{cui}"
            
            response = self.session.get(content_endpoint)
            self.log_debug(f"API response for semantic types of CUI {cui}: Status {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                semantic_types = []
                
                if 'result' in data and 'semanticTypes' in data['result']:
                    for st in data['result']['semanticTypes']:
                        if 'TUI' in st:
                            semantic_types.append(st['TUI'])
                
                # Cache the result
                self.cache['cui_semantic_types'][cui] = semantic_types
                self._periodic_cache_save()
                
                self.log_debug(f"Found {len(semantic_types)} semantic types for CUI {cui}")
                return semantic_types
            else:
                self.log_debug(f"Failed to get semantic types for CUI {cui}: Status {response.status_code}")
                self.cache['cui_semantic_types'][cui] = []
                return []
        
        except Exception as e:
            self.log_debug(f"Error getting semantic types for CUI {cui}: {str(e)}")
            time.sleep(1)  # Rate limiting
            self.cache['cui_semantic_types'][cui] = []
            return []

    def get_enhanced_vocabularies_for_cui(self, cui: str) -> Dict[str, Dict[str, Dict]]:
        """
        Get vocabulary codes with TTY information for a CUI.
        
        Args:
            cui (str): The UMLS CUI to query
            
        Returns:
            dict: Mapping of {vocab: {code: {"tty": tty_code, "term": term_text}}}
        """
        # Handle empty CUI
        if cui is None or cui == '' or cui == 'nan':
            return {}
            
        # Clean the CUI format
        cui = str(cui).strip()
        if not cui.startswith('C'):
            cui = f"C{cui}"
        
        # Check cache first
        if cui in self.cache['cui_atoms_enhanced']:
            return self.cache['cui_atoms_enhanced'][cui]
        
        try:
            # Query UMLS API for atoms (which contain TTY info)
            atoms_url = f"{self.umls_api_base}/content/current/CUI/{cui}/atoms"
            
            response = self.session.get(atoms_url)
            self.log_debug(f"API response for enhanced vocabularies of CUI {cui}: Status {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract vocabulary codes with TTY information
                vocabulary_data = defaultdict(lambda: defaultdict(dict))
                
                if 'result' in data:
                    for atom in data['result']:
                        root_source = atom.get('rootSource', '')
                        code = atom.get('code', '')
                        tty = atom.get('termType', '')  # TTY information
                        term = atom.get('name', '')  # The actual term text
                        
                        if root_source and code:
                            # Store code with TTY and term info
                            if code not in vocabulary_data[root_source]:
                                vocabulary_data[root_source][code] = {
                                    'tty': tty,
                                    'term': term
                                }
                            else:
                                # If multiple TTYs for same code, collect them
                                existing_tty = vocabulary_data[root_source][code]['tty']
                                if tty and tty not in existing_tty.split('|'):
                                    vocabulary_data[root_source][code]['tty'] = f"{existing_tty}|{tty}" if existing_tty else tty
                
                # Convert to regular dict
                result = dict(vocabulary_data)
                for vocab in result:
                    result[vocab] = dict(result[vocab])
                
                # Cache the result
                self.cache['cui_atoms_enhanced'][cui] = result
                self._periodic_cache_save()
                
                self.log_debug(f"Found enhanced vocabularies for CUI {cui}: {list(result.keys())}")
                return result
            else:
                self.log_debug(f"Failed to get enhanced vocabularies for CUI {cui}: Status {response.status_code}")
                self.cache['cui_atoms_enhanced'][cui] = {}
                return {}
                
        except Exception as e:
            self.log_debug(f"Error getting enhanced vocabularies for CUI {cui}: {str(e)}")
            time.sleep(1)  # Rate limiting
            self.cache['cui_atoms_enhanced'][cui] = {}
            return {}

    def create_comprehensive_mapping_files(self, cuis: Set[str], output_dir: str, dataset_name: str) -> Dict[str, str]:
        """
        Create comprehensive CUI-to-vocabulary and CUI-to-TUI mapping files using UMLS API.
        
        Args:
            cuis (set): Set of CUIs to analyze
            output_dir (str): Output directory for mapping files
            dataset_name (str): Name of the dataset
            
        Returns:
            dict: Dictionary with paths to created mapping files
        """
        self.log_debug(f"Creating comprehensive mappings for {len(cuis)} CUIs")
        
        # Initialize mapping dictionaries
        cui_to_vocab_codes = {}
        cui_to_tuis = {}
        
        # Process CUIs with progress bar
        valid_cuis = [cui for cui in cuis if cui and cui != '' and cui != 'CUI-less']
        
        print(f"Generating comprehensive UMLS mappings for {len(valid_cuis)} CUIs...")
        
        for cui in tqdm(valid_cuis, desc="Processing CUIs"):
            # Get vocabulary codes
            vocab_codes = self.get_comprehensive_vocabularies_for_cui(cui)
            if vocab_codes:
                cui_to_vocab_codes[cui] = vocab_codes
            
            # Get semantic types
            semantic_types = self.get_comprehensive_semantic_types_for_cui(cui)
            if semantic_types:
                cui_to_tuis[cui] = semantic_types
            
            # Add small delay to avoid overwhelming the API
            time.sleep(0.1)
        
        # Create output directory
        mappings_dir = os.path.join(output_dir, 'mappings')
        os.makedirs(mappings_dir, exist_ok=True)
        
        # Save mapping files
        vocab_codes_file = os.path.join(mappings_dir, f'{dataset_name}_cui_to_vocab_codes.json')
        tuis_file = os.path.join(mappings_dir, f'{dataset_name}_cui_to_tuis.json')
        
        with open(vocab_codes_file, 'w') as f:
            json.dump(cui_to_vocab_codes, f, indent=2, sort_keys=True)
        
        with open(tuis_file, 'w') as f:
            json.dump(cui_to_tuis, f, indent=2, sort_keys=True)
        
        # Also create a summary file
        summary_file = os.path.join(mappings_dir, f'{dataset_name}_mapping_summary.json')
        summary = {
            'total_cuis_processed': len(valid_cuis),
            'cuis_with_vocab_codes': len(cui_to_vocab_codes),
            'cuis_with_semantic_types': len(cui_to_tuis),
            'vocabulary_sources_found': list(set(
                source for codes in cui_to_vocab_codes.values() for source in codes.keys()
            )),
            'unique_semantic_types_found': list(set(
                tui for tuis in cui_to_tuis.values() for tui in tuis
            ))
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Comprehensive mappings created:")
        print(f"  - Vocabulary codes: {len(cui_to_vocab_codes)} CUIs mapped")
        print(f"  - Semantic types: {len(cui_to_tuis)} CUIs mapped")
        print(f"  - Vocabulary sources found: {len(summary['vocabulary_sources_found'])}")
        print(f"  - Unique semantic types found: {len(summary['unique_semantic_types_found'])}")
        
        self.log_debug(f"Created comprehensive mappings: {summary}")
        
        return {
            'cui_to_vocab_codes': vocab_codes_file,
            'cui_to_tuis': tuis_file,
            'mapping_summary': summary_file
        }

    def create_enhanced_mapping_files(self, cuis: Set[str], output_dir: str, dataset_name: str) -> Dict[str, str]:
        """
        Create comprehensive CUI-to-vocabulary mapping files with TTY information using UMLS API.
        
        Args:
            cuis (set): Set of CUIs to analyze
            output_dir (str): Output directory for mapping files
            dataset_name (str): Name of the dataset
            
        Returns:
            dict: Dictionary with paths to created mapping files
        """
        self.log_debug(f"Creating enhanced mappings with TTY for {len(cuis)} CUIs")
        
        # Initialize mapping dictionaries
        cui_to_vocab_codes_with_tty = {}
        cui_to_tuis = {}
        
        # Process CUIs with progress bar
        valid_cuis = [cui for cui in cuis if cui and cui != '' and cui != 'CUI-less']
        
        print(f"Generating enhanced UMLS mappings with TTY for {len(valid_cuis)} CUIs...")
        
        for cui in tqdm(valid_cuis, desc="Processing CUIs with TTY"):
            # Get enhanced vocabulary codes with TTY information
            enhanced_vocab_codes = self.get_enhanced_vocabularies_for_cui(cui)
            if enhanced_vocab_codes:
                cui_to_vocab_codes_with_tty[cui] = enhanced_vocab_codes
            
            # Get semantic types
            semantic_types = self.get_comprehensive_semantic_types_for_cui(cui)
            if semantic_types:
                cui_to_tuis[cui] = semantic_types
            
            # Add small delay to avoid overwhelming the API
            time.sleep(0.1)
        
        # Create output directory
        mappings_dir = os.path.join(output_dir, 'mappings')
        os.makedirs(mappings_dir, exist_ok=True)
        
        # Save enhanced mapping files
        enhanced_vocab_codes_file = os.path.join(mappings_dir, f'{dataset_name}_cui_to_vocab_codes_with_tty.json')
        tuis_file = os.path.join(mappings_dir, f'{dataset_name}_cui_to_tuis.json')
        
        with open(enhanced_vocab_codes_file, 'w') as f:
            json.dump(cui_to_vocab_codes_with_tty, f, indent=2, sort_keys=True)
        
        with open(tuis_file, 'w') as f:
            json.dump(cui_to_tuis, f, indent=2, sort_keys=True)
        
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
        tty_stats_file = os.path.join(mappings_dir, f'{dataset_name}_tty_statistics.json')
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
        
        print(f"Enhanced mappings with TTY created:")
        print(f"  - Vocabulary codes with TTY: {len(cui_to_vocab_codes_with_tty)} CUIs mapped")
        print(f"  - Semantic types: {len(cui_to_tuis)} CUIs mapped")
        print(f"  - Total vocabulary codes: {total_codes}")
        print(f"  - Vocabularies with TTY data: {len(tty_stats)}")
        
        # Print TTY summary
        for vocab, summary in stats['summary'].items():
            if summary['total_codes'] > 0:
                print(f"  - {vocab}: {summary['total_codes']} codes, {summary['unique_ttys']} unique TTYs")
        
        self.log_debug(f"Created enhanced mappings with TTY: {stats}")
        
        return {
            'cui_to_vocab_codes_with_tty': enhanced_vocab_codes_file,
            'cui_to_tuis': tuis_file,
            'tty_statistics': tty_stats_file
        } 