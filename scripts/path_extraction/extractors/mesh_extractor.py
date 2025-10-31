"""
MeSH (Medical Subject Headings) hierarchical path extractor.
Enhanced version with comprehensive error handling, caching, statistics, and logging.
"""

import pickle
import requests
import re
import time
import statistics
from pathlib import Path
from typing import List, Dict, Optional, Any
from lxml import etree
from tqdm import tqdm
import logging

from .base_extractor import BaseExtractor, ExtractorRegistry

# Root category mapping for MeSH tree letters
ROOT_CATEGORY = {
    "A": "Anatomy",
    "B": "Organisms", 
    "C": "Diseases",
    "D": "Chemicals and Drugs",
    "E": "Analytical, Diagnostic and Therapeutic Techniques, and Equipment",
    "F": "Psychiatry and Psychology",
    "G": "Phenomena and Processes",
    "H": "Disciplines and Occupations", 
    "I": "Anthropology, Education, Sociology and Social Phenomena",
    "J": "Technology, Industry, Agriculture",
    "K": "Humanities",
    "L": "Information Science",
    "M": "Named Groups",
    "N": "Health Care",
    "O": "Publication Characteristics",
    "P": "Publication Components",
    "Q": "Geographic Locations",
    "V": "Publication Types",
    "Z": "Geographical Locations"
}


class MeshExtractor(BaseExtractor):
    """Enhanced MeSH extractor with comprehensive error handling and optimization."""
    
    def __init__(self, config: Dict[str, Any], vocab_name: str):
        super().__init__(config, vocab_name)
        
        # Get MeSH-specific configuration
        mesh_config = config.get('local_files', {}).get('mesh', {})
        self.xml_file = mesh_config.get('xml_file', 'desc2025.xml')
        self.pickle_cache = mesh_config.get('pickle_cache', 'mesh_tree_map_2025.pkl')
        
        # Enhanced error handling settings - OPTIMIZED FOR PERFORMANCE
        self.max_retries = 3  # Reduced retries for faster failure recovery
        self.base_retry_delay = 0.5  # Faster initial retry
        self.max_retry_delay = 8.0  # Lower backoff cap
        self.timeout = 15  # Reduced timeout for faster failure detection
        self.rate_limit_delay = 0.1  # Optimized rate limiting for NLM API
        
        # Multi-level caching for performance
        self.tree_numbers_cache = {}  # Cache API responses for tree numbers
        self.path_cache = {}  # Cache built paths
        self.failed_codes_cache = set()  # Cache known failed codes
        
        # Enhanced statistics tracking
        self.stats = {
            'total_codes_processed': 0,
            'successful_extractions': 0,
            'api_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_paths_extracted': 0,
            'timeouts': 0,
            'rate_limit_hits': 0,
            'path_counts': [],  # Track paths per successful code
            'path_lengths': []  # Track length of each path
        }
        
        # CUI-level tracking (synced with base class)
        self.cui_results = {}  # Track results per CUI
        
        # Batch processing settings - OPTIMIZED FOR SPEED
        self.batch_size = 200  # Larger batches for better throughput
        self.save_frequency = 100  # Less frequent saves for better performance
        
        # Enhanced logging setup
        self._setup_enhanced_logging()
        
        # API configuration (optional; default to offline)
        api_cfg = config.get('apis', {}).get('mesh', {})
        self.api_base = api_cfg.get('base_url', "https://id.nlm.nih.gov/mesh")
        self.api_enabled = bool(api_cfg.get('enable_api', False))
        
        # Load or build tree mapping
        self.tree_map = self._load_tree_map()
        # Build UI -> tree numbers index from local tree map to avoid API dependency
        self.ui_to_tree_numbers = {}
        for tree_num, info in self.tree_map.items():
            ui = info.get('ui')
            if ui:
                self.ui_to_tree_numbers.setdefault(ui, []).append(tree_num)
        
        self.logger.info(f"Initialized MeSH extractor with {len(self.tree_map):,} tree mappings")
        self.logger.info(f"Configuration: max_retries={self.max_retries}, timeout={self.timeout}s, rate_limit={self.rate_limit_delay}s")
        
    def _setup_enhanced_logging(self):
        """Setup comprehensive logging with separate files for different types."""
        log_dir = Path(self.config.get('runtime', {}).get('logs_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        # API calls log
        api_logger = logging.getLogger(f"{self.vocab_name}_api")
        api_handler = logging.FileHandler(log_dir / f"mesh_api.log")
        api_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        api_logger.addHandler(api_handler)
        api_logger.setLevel(logging.INFO)
        
        # Statistics log
        stats_logger = logging.getLogger(f"{self.vocab_name}_stats")
        stats_handler = logging.FileHandler(log_dir / f"mesh_stats.log")
        stats_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        stats_logger.addHandler(stats_handler)
        stats_logger.setLevel(logging.INFO)
        
        # Errors log
        error_logger = logging.getLogger(f"{self.vocab_name}_errors")
        error_handler = logging.FileHandler(log_dir / f"mesh_errors.log")
        error_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        error_logger.addHandler(error_handler)
        error_logger.setLevel(logging.ERROR)
        
    def validate_code(self, code: str) -> bool:
        """
        Validate MeSH Descriptor UI format.
        MeSH UIs are typically in format D######.
        """
        if not code or not isinstance(code, str):
            return False
        
        # Remove any whitespace
        code = code.strip()
        
        # Should match pattern like D123456, C123456, etc.
        return bool(re.match(r'^[A-Z]\d{6,9}$', code))
        
    def extract_paths(self, code: str, tty: Optional[str] = None, cui: Optional[str] = None) -> List[List[Dict[str, str]]]:
        """
        Extract hierarchical paths for a MeSH descriptor with enhanced error handling.
        
        Args:
            code: MeSH Descriptor UI (e.g., D012307)
            tty: Not used for MeSH
            cui: CUI for tracking statistics (optional)
            
        Returns:
            List of paths from root to descriptor, each path as list of {'code', 'name'} dicts
        """
        # Update processing statistics
        self.stats['total_codes_processed'] += 1
        
        # Initialize CUI tracking if needed
        if cui and cui not in self.cui_results:
            self.cui_results[cui] = {
                'codes_processed': 0,
                'codes_with_paths': 0,
                'total_paths': 0,
                'failed_codes': []
            }
        
        if not self.validate_code(code):
            self.logger.warning(f"Invalid MeSH code format: {code}")
            if cui:
                self.track_cui_code_result(cui, code, 'error')
                self.cui_results[cui]['failed_codes'].append(code)
                self.cui_results[cui]['codes_processed'] += 1
            return []
        
        # Check if this code has failed before
        if code in self.failed_codes_cache:
            self.logger.debug(f"Skipping known failed code: {code}")
            if cui:
                self.track_cui_code_result(cui, code, 'cached_failure')
                self.cui_results[cui]['codes_processed'] += 1
            return []
        
        # Check path cache first
        if code in self.path_cache:
            self.stats['cache_hits'] += 1
            paths = self.path_cache[code]
            if cui:
                self.track_cui_code_result(cui, code, 'success', len(paths))
                self.cui_results[cui]['codes_processed'] += 1
                if paths:
                    self.cui_results[cui]['codes_with_paths'] += 1
                    self.cui_results[cui]['total_paths'] += len(paths)
            return paths
        
        self.stats['cache_misses'] += 1
            
        try:
            # Get tree numbers for this descriptor with retry logic
            tree_numbers = self._get_tree_numbers_with_retry(code)
            
            if not tree_numbers:
                self.logger.warning(f"No tree numbers found for MeSH descriptor: {code}")
                self.failed_codes_cache.add(code)
                if cui:
                    self.track_cui_code_result(cui, code, 'not_found')
                    self.cui_results[cui]['failed_codes'].append(code)
                    self.cui_results[cui]['codes_processed'] += 1
                return []
                
            # Build paths for each tree number
            all_paths = []
            for tree_num in tree_numbers:
                path = self._build_path_from_tree_number(tree_num)
                if path:
                    all_paths.append(path)
                    # Track path statistics
                    self.stats['path_lengths'].append(len(path))
            
            # Cache the result
            self.path_cache[code] = all_paths
            
            # Update statistics
            if all_paths:
                self.stats['successful_extractions'] += 1
                self.stats['total_paths_extracted'] += len(all_paths)
                self.stats['path_counts'].append(len(all_paths))
                
                if cui:
                    self.track_cui_code_result(cui, code, 'success', len(all_paths))
                    self.cui_results[cui]['codes_processed'] += 1
                    self.cui_results[cui]['codes_with_paths'] += 1
                    self.cui_results[cui]['total_paths'] += len(all_paths)
            else:
                self.failed_codes_cache.add(code)
                if cui:
                    self.track_cui_code_result(cui, code, 'not_found')
                    self.cui_results[cui]['failed_codes'].append(code)
                    self.cui_results[cui]['codes_processed'] += 1
                
            # Log progress periodically - reduced frequency for speed
            if self.stats['total_codes_processed'] % 200 == 0:
                self._log_progress_stats()
                
            return all_paths
            
        except Exception as e:
            self.stats['api_errors'] += 1
            self.failed_codes_cache.add(code)
            self.logger.error(f"Error extracting MeSH paths for {code}: {str(e)}")
            
            error_logger = logging.getLogger(f"{self.vocab_name}_errors")
            error_logger.error(f"Failed to extract paths for {code}: {str(e)}")
            
            if cui:
                self.track_cui_code_result(cui, code, 'error')
                self.cui_results[cui]['failed_codes'].append(code)
                self.cui_results[cui]['codes_processed'] += 1
            return []
            
    def _get_tree_numbers_with_retry(self, ui: str) -> List[str]:
        """Get tree numbers for a MeSH descriptor with retry logic."""
        # Check cache first
        if ui in self.tree_numbers_cache:
            return self.tree_numbers_cache[ui]
        
        # Offline-first: prefer local tree numbers (avoids brittle API .json redirects)
        if ui in self.ui_to_tree_numbers or not self.api_enabled:
            self.tree_numbers_cache[ui] = self.ui_to_tree_numbers[ui]
            return self.tree_numbers_cache[ui]

        url = f"{self.api_base}/{ui}.json"
        api_logger = logging.getLogger(f"{self.vocab_name}_api")
        
        accept_order = ["application/json", "application/ld+json"]
        for attempt in range(self.max_retries):
            try:
                api_logger.info(f"API call attempt {attempt + 1}/{self.max_retries}: {url}")
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                # Use explicit JSON Accept header and follow redirects; try alt content type on later attempts
                headers = {"Accept": accept_order[min(attempt, len(accept_order)-1)]}
                response = requests.get(url, timeout=self.timeout, headers=headers, allow_redirects=True)
                response.raise_for_status()
                
                # Some servers may redirect to HTML; guard against non-JSON content
                ctype = response.headers.get("Content-Type", "")
                if "application/json" not in ctype:
                    raise requests.exceptions.RequestException(f"Unexpected content-type: {ctype}")
                data = response.json()
                tn = data.get("treeNumber", [])
                
                # Service returns *either* a string or a list
                if isinstance(tn, str):
                    tree_numbers = [tn.split("/")[-1]]  # e.g. 'C06.552'
                else:
                    tree_numbers = [x.split("/")[-1] for x in tn]
                
                # Cache the result
                self.tree_numbers_cache[ui] = tree_numbers
                return tree_numbers
                
            except requests.exceptions.Timeout:
                self.stats['timeouts'] += 1
                api_logger.warning(f"Timeout on attempt {attempt + 1} for {ui}")
                if attempt < self.max_retries - 1:
                    delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                    time.sleep(delay)
                    continue
                else:
                    raise
                    
            except requests.exceptions.RequestException as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    self.stats['rate_limit_hits'] += 1
                    api_logger.warning(f"Rate limit hit on attempt {attempt + 1} for {ui}")
                    if attempt < self.max_retries - 1:
                        delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                        time.sleep(delay)
                        continue
                    else:
                        raise
                else:
                    api_logger.error(f"API error on attempt {attempt + 1} for {ui}: {str(e)}; falling back to local tree map if possible")
                    # If we have local tree numbers, return them; otherwise retry/backoff
                    if ui in self.ui_to_tree_numbers:
                        self.tree_numbers_cache[ui] = self.ui_to_tree_numbers[ui]
                        return self.tree_numbers_cache[ui]
                    if attempt < self.max_retries - 1:
                        delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                        time.sleep(delay)
                        continue
                    else:
                        raise
                        
        # If we get here, all retries failed
        raise Exception(f"All {self.max_retries} attempts failed for {ui}")
        
    def _log_progress_stats(self):
        """Log current progress statistics."""
        stats_logger = logging.getLogger(f"{self.vocab_name}_stats")
        
        total_processed = self.stats['total_codes_processed']
        successful = self.stats['successful_extractions']
        success_rate = (successful / total_processed * 100) if total_processed > 0 else 0
        cache_hit_rate = (self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) * 100) if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
        
        stats_logger.info(f"Progress: {total_processed} codes processed, {successful} successful ({success_rate:.1f}%), "
                         f"cache hit rate: {cache_hit_rate:.1f}%, API errors: {self.stats['api_errors']}")
        
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including median, min, max values."""
        base_stats = {
            'vocab_name': self.vocab_name,
            'rate_limit': self.rate_limit_delay,
            'max_retries': self.max_retries,
            'timeout': self.timeout
        }
        
        # Calculate path statistics
        path_stats = {}
        if self.stats['path_counts']:
            path_stats['total_paths'] = sum(self.stats['path_counts'])
            path_stats['mean_paths_per_code'] = statistics.mean(self.stats['path_counts'])
            path_stats['median_paths_per_code'] = statistics.median(self.stats['path_counts'])
            path_stats['min_paths_per_code'] = min(self.stats['path_counts'])
            path_stats['max_paths_per_code'] = max(self.stats['path_counts'])
        
        if self.stats['path_lengths']:
            path_stats['mean_path_length'] = statistics.mean(self.stats['path_lengths'])
            path_stats['median_path_length'] = statistics.median(self.stats['path_lengths'])
            path_stats['min_path_length'] = min(self.stats['path_lengths'])
            path_stats['max_path_length'] = max(self.stats['path_lengths'])
        
        # CUI-level statistics
        cui_stats = {
            'total_cuis_processed': len(self.cui_results),
            'cuis_with_paths': len([cui for cui, data in self.cui_results.items() if data['codes_with_paths'] > 0]),
            'cuis_without_paths': len([cui for cui, data in self.cui_results.items() if data['codes_with_paths'] == 0])
        }
        cui_stats['cui_success_rate'] = (cui_stats['cuis_with_paths'] / cui_stats['total_cuis_processed']) if cui_stats['total_cuis_processed'] > 0 else 0
        
        # Enhanced statistics
        enhanced_stats = {
            **base_stats,
            **self.stats,
            'success_rate': (self.stats['successful_extractions'] / self.stats['total_codes_processed']) if self.stats['total_codes_processed'] > 0 else 0,
            'cache_hit_rate': (self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])) if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0,
            'avg_paths_per_code': (self.stats['total_paths_extracted'] / self.stats['successful_extractions']) if self.stats['successful_extractions'] > 0 else 0,
            'api_calls_saved_by_cache': self.stats['cache_hits'],
            'path_statistics': path_stats,
            'cui_statistics': cui_stats
        }
        
        return enhanced_stats
        
    def _load_tree_map(self) -> Dict[str, Dict[str, str]]:
        """Load or build the tree number to descriptor mapping."""
        pickle_path = Path(self.pickle_cache)
        if not pickle_path.exists() and not pickle_path.is_absolute():
            alt = Path(__file__).parents[3] / "path_data" / self.pickle_cache
            if alt.exists():
                pickle_path = alt
        
        # Try to load existing pickle file
        if pickle_path.exists():
            try:
                with open(pickle_path, 'rb') as f:
                    tree_map = pickle.load(f)
                self.logger.info(f"Loaded cached tree map with {len(tree_map):,} entries")
                return tree_map
            except Exception as e:
                self.logger.warning(f"Failed to load pickle cache: {e}, rebuilding...")
        
        # Build new mapping
        return self._build_tree_mapping()
        
    def _build_tree_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Build tree number to descriptor mapping from XML file.
        """
        xml_path = Path(self.xml_file)
        if not xml_path.exists() and not xml_path.is_absolute():
            alt = Path(__file__).parents[3] / "path_data" / self.xml_file
            if alt.exists():
                xml_path = alt
        
        if not xml_path.exists():
            raise FileNotFoundError(f"MeSH XML file not found: {xml_path}")
            
        self.logger.info(f"Building MeSH tree mapping from {xml_path}")
        
        tree_map = {}
        
        try:
            context = etree.iterparse(str(xml_path), events=("end",), tag="DescriptorRecord")
            
            for _, elem in tqdm(context, desc="Processing MeSH descriptors"):
                ui = elem.findtext(".//DescriptorUI")
                name = elem.findtext(".//DescriptorName/String")
                
                if ui and name:
                    # Get all tree numbers for this descriptor
                    for tn_elem in elem.findall(".//TreeNumber"):
                        tree_num = tn_elem.text
                        if tree_num:
                            tree_map[tree_num] = {"ui": ui, "name": name}
                
                # Clear element to save memory
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
                    
        except Exception as e:
            self.logger.error(f"Error parsing MeSH XML: {e}")
            raise
            
        # Cache the mapping
        pickle_path = Path(self.pickle_cache)
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(tree_map, f)
            self.logger.info(f"Cached tree mapping to {pickle_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cache tree mapping: {e}")
            
        self.logger.info(f"Built tree mapping with {len(tree_map):,} entries")
        return tree_map
        
    def _build_path_from_tree_number(self, tree_number: str) -> List[Dict[str, str]]:
        """
        Build hierarchical path from tree number.
        
        Args:
            tree_number: MeSH tree number (e.g., 'C06.552')
            
        Returns:
            Path from root to leaf as list of {'code', 'name'} dicts
        """
        # Get all ancestor tree numbers (leaf to root)
        ancestor_tree_nums = self._tree_to_root(tree_number)
        
        path = []
        for tree_num in ancestor_tree_nums:
            # Look up descriptor info for this tree number
            descriptor_info = self.tree_map.get(tree_num)
            
            if descriptor_info:
                path.append({
                    'code': descriptor_info['ui'],
                    'name': descriptor_info['name']
                })
            else:
                # Handle single-letter root categories
                letter = tree_num.split('.')[0]
                if letter in ROOT_CATEGORY:
                    path.append({
                        'code': letter,
                        'name': ROOT_CATEGORY[letter]
                    })
                else:
                    # Handle unknown tree numbers
                    path.append({
                        'code': tree_num,
                        'name': f'Unknown: {tree_num}'
                    })
                    self.logger.warning(f"Unknown tree number in path: {tree_num}")
                
        return path
        
    def _tree_to_root(self, tree_number: str) -> List[str]:
        """
        Walk upward by stripping the *last* segment each time.
        'C06.552' → ['C06.552', 'C06', 'C']
        
        Args:
            tree_number: Tree number like 'C06.552'
            
        Returns:
            List like ['C06.552', 'C06', 'C'] (leaf to root)
        """
        ancestors = [tree_number]
        tn = tree_number
        while '.' in tn:
            tn = tn.rsplit('.', 1)[0]  # remove everything after last dot
            ancestors.append(tn)
        if len(tn) > 1:  # add single-letter root if missing
            ancestors.append(tn[0])
        return ancestors


# Register the enhanced extractor
ExtractorRegistry.register('MSH', MeshExtractor)
ExtractorRegistry.register('MESH', MeshExtractor)  # Fallback name