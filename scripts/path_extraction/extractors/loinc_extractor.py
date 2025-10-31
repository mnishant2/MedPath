"""
LOINC hierarchical path extractor.
Enhanced version with comprehensive error handling, caching, statistics, and logging.
Uses FHIR API to extract parent-child relationships with optimized rate limiting.
"""

import base64
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import time
import statistics
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
import logging

from .base_extractor import BaseExtractor, ExtractorRegistry


class LOINCExtractor(BaseExtractor):
    """Enhanced LOINC extractor with comprehensive error handling and optimization."""
    
    def __init__(self, config: Dict[str, Any], vocab_name: str):
        super().__init__(config, vocab_name)
        
        # Enhanced error handling settings - ULTRA-OPTIMIZED FOR SPEED
        self.max_retries = 2  # Reduced retries for faster failure recovery
        self.base_retry_delay = 0.3  # Faster retry for API speed
        self.max_retry_delay = 3.0  # Lower backoff cap for speed
        self.timeout = 15  # Reduced timeout for faster failure detection
        self.rate_limit_delay = 0.1  # Ultra-aggressive rate limit for LOINC API
        
        # Get LOINC-specific configuration
        loinc_config = config.get('apis', {}).get('loinc', {})
        self.base_url = loinc_config.get('base_url', 'https://fhir.loinc.org')
        self.username = loinc_config.get('username', '')
        self.password = loinc_config.get('password', '')
        self.system = "http://loinc.org"
        # Optional version (FHIR CodeSystem systemVersion)
        self.loinc_version = (config.get('versions', {}) or {}).get('LOINC_version')
        
        if not self.username or not self.password:
            raise ValueError("LOINC username and password are required in config")
            
        # Set up authentication and optimized session
        auth_string = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {auth_string}",
            "Accept": "application/json",
            "User-Agent": "UMLS-PathExtractor/1.0",
            "Connection": "keep-alive"
        }
        
        # Create optimized session with connection pooling
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=0.3
        )
        
        # Configure HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=retry_strategy
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update(self.headers)
        
        # API endpoint
        self.lookup_url = f"{self.base_url}/CodeSystem/$lookup"
        
        # Multi-level caching for performance
        self.path_cache = {}  # Cache computed paths
        self.lookup_cache = {}  # Cache API lookup responses
        self.parent_cache = {}  # Cache parent relationships
        self.failed_codes_cache = set()  # Cache known failed codes
        
        # Enhanced statistics tracking
        self.stats = {
            'total_codes_processed': 0,
            'successful_extractions': 0,
            'api_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_paths_extracted': 0,
            'authentication_errors': 0,
            'not_found_errors': 0,
            'path_counts': [],  # Track paths per successful code
            'path_lengths': []  # Track length of each path
        }
        
        # CUI-level tracking (synced with base class)
        self.cui_results = {}  # Track results per CUI
        
        # Batch processing settings - ULTRA-OPTIMIZED FOR SPEED
        self.batch_size = 100  # Larger batches for better throughput
        self.save_frequency = 50  # Less frequent saves for speed
        
        # Enhanced logging setup
        self._setup_enhanced_logging()
        
        # Performance settings
        self.max_path_length = 20  # Reasonable depth for LOINC
        self.max_paths_per_code = 100  # Limit for LOINC hierarchies
        
        # Test API connection
        self._test_api_connection()
        
        self.logger.info(f"Initialized LOINC extractor with base URL: {self.base_url}")
        self.logger.info(f"Configuration: max_retries={self.max_retries}, rate_limit={self.rate_limit_delay}s")
        
    def _setup_enhanced_logging(self):
        """Setup comprehensive logging with separate files for different types."""
        log_dir = Path(self.config.get('runtime', {}).get('logs_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        # API operations log
        api_logger = logging.getLogger(f"{self.vocab_name}_api")
        api_handler = logging.FileHandler(log_dir / f"loinc_api_ops.log")
        api_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        api_logger.addHandler(api_handler)
        api_logger.setLevel(logging.INFO)
        
        # Statistics log
        stats_logger = logging.getLogger(f"{self.vocab_name}_stats")
        stats_handler = logging.FileHandler(log_dir / f"loinc_stats.log")
        stats_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        stats_logger.addHandler(stats_handler)
        stats_logger.setLevel(logging.INFO)
        
        # Errors log
        error_logger = logging.getLogger(f"{self.vocab_name}_errors")
        error_handler = logging.FileHandler(log_dir / f"loinc_errors.log")
        error_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        error_logger.addHandler(error_handler)
        error_logger.setLevel(logging.ERROR)
        
    def _test_api_connection(self):
        """Test LOINC API connection with a simple lookup."""
        try:
            # Try a common LOINC code for testing
            test_response = self._lookup_code_with_retry("2339-0")  # Glucose measurement
            self.logger.info("✅ LOINC API connection test successful")
        except Exception as e:
            self.logger.warning(f"⚠️ LOINC API connection test failed: {e}")
            # Don't raise exception, just log the warning
        
    def validate_code(self, code: str) -> bool:
        """
        Validate LOINC code format.
        LOINC codes from UMLS can be in various formats:
        - Standard format: ####-# (4+ digits, dash, 1 digit) e.g., 2339-0
        - MTHU codes: MTHU###### (MTHU prefix + 6 digits) e.g., MTHU013518
        - LP codes: LP#####-# (LP prefix + digits + dash + digit) e.g., LP74849-8
        - LA codes: LA#####-# (LA prefix + digits + dash + digit) e.g., LA15157-3
        """
        if not code or not isinstance(code, str):
            return False
        
        # Remove any whitespace
        code = code.strip()
        
        # Check various LOINC formats from UMLS
        patterns = [
            r'^\d{4,}-\d$',           # Standard LOINC: 2339-0
            r'^MTHU\d{6}$',           # MTHU codes: MTHU013518
            r'^LP\d{5}-\d$',          # LP codes: LP74849-8
            r'^LA\d{5}-\d$',          # LA codes: LA15157-3
            r'^LG\d{5}-\d$',          # LG codes (if any)
            r'^LL\d{4,}-\d$',         # LL codes (if any)
            r'^LP\d{4,}-\d$',         # LP codes with variable length
            r'^LA\d{4,}-\d$',         # LA codes with variable length
        ]
        
        for pattern in patterns:
            if re.match(pattern, code):
                return True
                
        return False
        
    def extract_paths(self, code: str, tty: Optional[str] = None, cui: Optional[str] = None) -> List[List[Dict[str, str]]]:
        """
        Extract hierarchical paths for a LOINC code with enhanced error handling.
        
        Args:
            code: LOINC code (e.g., "2339-0")
            tty: Not used for LOINC
            cui: CUI for tracking statistics (optional)
            
        Returns:
            List of paths from root to concept, each path as list of {'code', 'name'} dicts
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
            self.logger.warning(f"Invalid LOINC code format: {code}")
            if cui:
                self.track_cui_code_result(cui, code, 'error')
                self.cui_results[cui]['failed_codes'].append(code)
                self.cui_results[cui]['codes_processed'] += 1
            return []
        
        code = code.strip()
        
        # SPEED OPTIMIZATION: Skip known problematic code patterns
        if code.startswith('MTHU'):
            self.logger.debug(f"Skipping known MTHU code (not in LOINC FHIR): {code}")
            if cui:
                self.track_cui_code_result(cui, code, 'skipped_mthu')
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
            # Extract paths with retry logic
            paths = self._get_ancestor_paths_with_retry(code)
            
            # Cache the result
            self.path_cache[code] = paths
            
            # Update statistics
            if paths:
                self.stats['successful_extractions'] += 1
                self.stats['total_paths_extracted'] += len(paths)
                self.stats['path_counts'].append(len(paths))
                
                # Track path lengths
                for path in paths:
                    self.stats['path_lengths'].append(len(path))
                
                if cui:
                    self.track_cui_code_result(cui, code, 'success', len(paths))
                    self.cui_results[cui]['codes_processed'] += 1
                    self.cui_results[cui]['codes_with_paths'] += 1
                    self.cui_results[cui]['total_paths'] += len(paths)
            else:
                self.failed_codes_cache.add(code)
                if cui:
                    self.track_cui_code_result(cui, code, 'not_found')
                    self.cui_results[cui]['failed_codes'].append(code)
                    self.cui_results[cui]['codes_processed'] += 1
            
            # Log progress periodically - reduced frequency for speed
            if self.stats['total_codes_processed'] % 100 == 0:
                self._log_progress_stats()
                
            return paths
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [401, 403]:
                self.stats['authentication_errors'] += 1
                self.logger.error(f"LOINC API authentication error for {code}: {str(e)}")
            elif e.response.status_code == 404:
                self.stats['not_found_errors'] += 1
                self.logger.warning(f"LOINC code {code} not found")
            else:
                self.stats['api_errors'] += 1
                self.logger.error(f"LOINC API error for {code}: {str(e)}")
            
            self.failed_codes_cache.add(code)
            if cui:
                self.track_cui_code_result(cui, code, 'error')
                self.cui_results[cui]['failed_codes'].append(code)
                self.cui_results[cui]['codes_processed'] += 1
            return []
            
        except Exception as e:
            self.stats['api_errors'] += 1
            self.failed_codes_cache.add(code)
            self.logger.error(f"Error extracting LOINC paths for {code}: {str(e)}")
            
            error_logger = logging.getLogger(f"{self.vocab_name}_errors")
            error_logger.error(f"Failed to extract paths for {code}: {str(e)}")
            
            if cui:
                self.track_cui_code_result(cui, code, 'error')
                self.cui_results[cui]['failed_codes'].append(code)
                self.cui_results[cui]['codes_processed'] += 1
            return []
            
    def _get_ancestor_paths_with_retry(self, code: str) -> List[List[Dict[str, str]]]:
        """Get ancestor paths with retry logic for API operations."""
        api_logger = logging.getLogger(f"{self.vocab_name}_api")
        
        for attempt in range(self.max_retries):
            try:
                api_logger.info(f"Path extraction attempt {attempt + 1}/{self.max_retries} for code: {code}")
                
                paths = self._get_ancestor_paths(code)
                return paths
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                    time.sleep(delay)
                    api_logger.warning(f"API retry {attempt + 1} failed for {code}, retrying in {delay}s: {str(e)}")
                    continue
                else:
                    raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                    time.sleep(delay)
                    api_logger.warning(f"Retry {attempt + 1} failed for {code}, retrying in {delay}s: {str(e)}")
                    continue
                else:
                    raise
                    
        # If we get here, all retries failed
        raise Exception(f"All {self.max_retries} attempts failed for {code}")
        
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
            'timeout': self.timeout,
            'max_path_length': self.max_path_length,
            'max_paths_per_code': self.max_paths_per_code
        }
        
        # Calculate path statistics
        path_stats = {}
        if self.stats['path_counts']:
            path_stats['paths_per_code'] = {
                'mean': statistics.mean(self.stats['path_counts']),
                'median': statistics.median(self.stats['path_counts']),
                'min': min(self.stats['path_counts']),
                'max': max(self.stats['path_counts'])
            }
        
        if self.stats['path_lengths']:
            path_stats['path_lengths'] = {
                'mean': statistics.mean(self.stats['path_lengths']),
                'median': statistics.median(self.stats['path_lengths']),
                'min': min(self.stats['path_lengths']),
                'max': max(self.stats['path_lengths'])
            }
        
        # CUI-level statistics
        cui_stats = {
            'total_cuis_processed': len(self.cui_results),
            'cuis_with_paths': len([cui for cui, data in self.cui_results.items() if data['codes_with_paths'] > 0]),
            'cuis_without_paths': len([cui for cui, data in self.cui_results.items() if data['codes_with_paths'] == 0])
        }
        cui_stats['cui_success_rate'] = (cui_stats['cuis_with_paths'] / cui_stats['total_cuis_processed']) if cui_stats['total_cuis_processed'] > 0 else 0
        
        # Enhanced statistics like SNOMED/MeSH/NCI
        enhanced_stats = {
            **base_stats,
            **self.stats,
            'success_rate': (self.stats['successful_extractions'] / self.stats['total_codes_processed']) if self.stats['total_codes_processed'] > 0 else 0,
            'cache_hit_rate': (self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])) if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0,
            'avg_paths_per_code': (self.stats['total_paths_extracted'] / self.stats['successful_extractions']) if self.stats['successful_extractions'] > 0 else 0,
            'api_calls_saved_by_cache': self.stats['cache_hits'],
            'path_statistics': path_stats,
            'cui_statistics': cui_stats,
            
            # LOINC-specific stats
            'total_codes_with_paths': self.stats['successful_extractions'],
            'total_inactive_codes': 0,  # LOINC doesn't have inactive codes like SNOMED
            'code_success_rate': (self.stats['successful_extractions'] / self.stats['total_codes_processed']) if self.stats['total_codes_processed'] > 0 else 0,
            'total_cuis_processed': len(self.cui_results),
            'cuis_with_paths': cui_stats['cuis_with_paths'],
            'cuis_without_paths': cui_stats['cuis_without_paths'],
            'cui_success_rate': cui_stats['cui_success_rate'],
            
            # API-specific stats
            'authentication_errors': self.stats['authentication_errors'],
            'not_found_errors': self.stats['not_found_errors']
        }
        
        return enhanced_stats
        
    def _lookup_code_with_retry(self, code: str) -> Dict[str, Any]:
        """Perform FHIR $lookup operation with retry logic."""
        # Check lookup cache first
        if code in self.lookup_cache:
            return self.lookup_cache[code]
        
        params = {
            "system": self.system,
            "code": code,
            "property": "parent",
            "_format": "json"
        }
        if self.loinc_version:
            params["systemVersion"] = self.loinc_version
        
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                response = self.session.get(
                    self.lookup_url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                # Cache the result
                self.lookup_cache[code] = result
                return result
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                    time.sleep(delay)
                    self.logger.warning(f"LOINC API retry {attempt + 1} for {code}: {str(e)}")
                    continue
                else:
                    self.logger.error(f"LOINC FHIR lookup failed for {code} after {self.max_retries} attempts: {str(e)}")
                    raise
                    
    def _parse_lookup_response(self, response: Dict[str, Any]) -> tuple[Optional[str], List[str]]:
        """Parse FHIR lookup response to extract display name and parent codes."""
        display_name = None
        parent_codes = []
        
        parameters = response.get("parameter", [])
        
        for param in parameters:
            param_name = param.get("name")
            
            if param_name == "display":
                display_name = param.get("valueString")
            elif param_name == "property":
                # Parse property parameters for parent relationships
                parts = param.get("part", [])
                is_parent_property = False
                parent_code = None
                
                for part in parts:
                    part_name = part.get("name")
                    
                    if part_name == "code" and part.get("valueCode") == "parent":
                        is_parent_property = True
                    elif part_name == "value" and is_parent_property:
                        # Extract parent code from valueCoding
                        value_coding = part.get("valueCoding", {})
                        parent_code = value_coding.get("code")
                        
                if is_parent_property and parent_code:
                    parent_codes.append(parent_code)
                    
        return display_name, parent_codes
        
    def _get_ancestor_paths(self, code: str) -> List[List[Dict[str, str]]]:
        """
        Get all ancestor paths for a LOINC code using depth-first search.
        PATHS ARE ROOT->LEAF (consistent with SNOMED/MeSH/NCI)
        
        Args:
            code: Starting LOINC code
            
        Returns:
            List of paths from root ancestors to the given code (ROOT->LEAF)
        """
        all_paths = []
        visited = set()  # To prevent infinite loops
        
        def dfs(current_code: str, current_path: List[Dict[str, str]]):
            """Depth-first search to find all paths - builds ROOT->LEAF"""
            if current_code in visited or len(current_path) > self.max_path_length:
                return
            visited.add(current_code)
            
            try:
                # Lookup current code
                response = self._lookup_code_with_retry(current_code)
                display_name, parent_codes = self._parse_lookup_response(response)
                
                # Create node for current code
                node = {
                    "code": current_code,
                    "name": display_name or current_code
                }
                
                # Build path properly: append current node to path (ROOT->LEAF)
                current_path_with_node = current_path + [node]
                
                if not parent_codes:
                    # This is a root node - add the complete path (already ROOT->LEAF)
                    all_paths.append(current_path_with_node)
                else:
                    # Continue searching up the hierarchy
                    for parent_code in parent_codes:
                        if parent_code not in visited:  # Avoid cycles
                            # For parents, we need to build the path from parent down to current
                            # So we recursively call with empty path and then append our current path
                            parent_paths = []
                            self._find_parent_to_root_paths(parent_code, [], parent_paths, visited.copy())
                            
                            # Combine parent paths with current path
                            for parent_path in parent_paths:
                                complete_path = parent_path + current_path_with_node
                                all_paths.append(complete_path)
                            
            except Exception as e:
                self.logger.error(f"Error in DFS for LOINC code {current_code}: {str(e)}")
                # Add incomplete path if we can't continue
                if current_path:
                    node = {
                        "code": current_code,
                        "name": f"Error: {current_code}"
                    }
                    incomplete_path = current_path + [node]
                    all_paths.append(incomplete_path)
        
        # Start with the target code and work backwards to roots
        self._find_parent_to_root_paths(code, [], all_paths, set())
        
        # Remove duplicates while preserving order
        unique_paths = []
        seen_paths = set()
        
        for path in all_paths:
            # Create hashable representation
            path_key = tuple((node['code'], node['name']) for node in path)
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                unique_paths.append(path)
                
        # Limit number of paths
        if len(unique_paths) > self.max_paths_per_code:
            self.logger.warning(f"LOINC code {code} has {len(unique_paths)} paths, truncating to {self.max_paths_per_code}")
            unique_paths = unique_paths[:self.max_paths_per_code]
                
        return unique_paths
        
    def _find_parent_to_root_paths(self, code: str, current_path: List[Dict[str, str]], 
                                  all_paths: List[List[Dict[str, str]]], visited: Set[str]):
        """Helper method to find paths from root to target code (ROOT->LEAF)."""
        if code in visited or len(current_path) > self.max_path_length:
            return
        visited.add(code)
        
        try:
            # Lookup current code
            response = self._lookup_code_with_retry(code)
            display_name, parent_codes = self._parse_lookup_response(response)
            
            # Create node for current code
            node = {
                "code": code,
                "name": display_name or code
            }
            
            # Build path: append current node (ROOT->LEAF direction)
            new_path = current_path + [node]
            
            if not parent_codes:
                # This is a root - add complete path
                all_paths.append(new_path)
            else:
                # Continue recursively with parents
                for parent_code in parent_codes:
                    if parent_code not in visited:
                        # Recursively find paths from parent to root, then append our path
                        parent_paths = []
                        self._find_parent_to_root_paths(parent_code, [], parent_paths, visited.copy())
                        
                        # Combine parent paths with current node
                        for parent_path in parent_paths:
                            complete_path = parent_path + [node]
                            all_paths.append(complete_path)
                            
        except Exception as e:
            self.logger.error(f"Error finding parent paths for LOINC code {code}: {str(e)}")
            # Add incomplete path
            node = {"code": code, "name": f"Error: {code}"}
            incomplete_path = current_path + [node]
            all_paths.append(incomplete_path)


# Register the enhanced extractor
ExtractorRegistry.register('LNC', LOINCExtractor)
ExtractorRegistry.register('LNC_V2', LOINCExtractor)  # Alternative name