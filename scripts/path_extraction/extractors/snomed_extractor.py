"""
SNOMED CT hierarchical path extractor
Robust implementation with proper error handling, inactive code detection, and caching.
Based on the working test code approach with enhanced orchestration.
"""

import time
import json
import requests
import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict
import random
from .base_extractor import BaseExtractor, ExtractorRegistry


class SnomedExtractor(BaseExtractor):
    """
    Robust SNOMED CT extractor with comprehensive error handling and performance optimizations.
    """
    
    def __init__(self, config: Dict[str, Any], vocab_name: str):
        super().__init__(config, vocab_name)
        
        # SNOMED-specific configuration
        snomed_config = config.get('apis', {}).get('snomed', {})
        self.base_url = snomed_config.get('base_url', 'https://snowstorm.mi-x.nl/')
        # Version/branch selection
        versions_cfg = config.get('versions', {})
        self.branch = (
            snomed_config.get('branch')
            or versions_cfg.get('SNOMED_branch')
            or versions_cfg.get('GLOBAL_VERSION_OVERRIDE')
            or 'MAIN'
        )
        
        # Ensure URL ends with slash
        if not self.base_url.endswith('/'):
            self.base_url += '/'
            
        # Enhanced error handling settings - OPTIMIZED FOR SPEED
        self.max_retries = 3  # Reduced retries for faster failure recovery
        self.base_retry_delay = 0.5  # Faster initial retry
        self.max_retry_delay = 8.0  # Lower backoff cap
        self.timeout = 15  # Reduced timeout for faster failure detection
        self.rate_limit_delay = 0.05  # Ultra aggressive rate limit - proven faster
        
        # Performance caching
        self.parents_cache = {}  # Cache parent API calls
        self.concept_cache = {}  # Cache concept details
        self.inactive_codes_cache = set()  # Cache known inactive codes
        
        # Enhanced statistics tracking
        self.stats = {
            'total_codes_processed': 0,
            'successful_extractions': 0,
            'inactive_codes_skipped': 0,
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
        
        self.logger.info(f"Initialized SNOMED CT extractor with base URL: {self.base_url}")
        self.logger.info(f"Configuration: branch={self.branch}, max_retries={self.max_retries}, timeout={self.timeout}s, rate_limit={self.rate_limit_delay}s")
        
    def _setup_enhanced_logging(self):
        """Set up comprehensive logging for debugging and monitoring."""
        # Use orchestrator-provided per-vocab logs dir if available
        logs_dir = Path(self.config.get('runtime', {}).get('logs_dir', 'logs'))
        logs_dir.mkdir(exist_ok=True)
        
        # Error-specific logger
        self.error_logger = logging.getLogger(f'{self.vocab_name}_errors')
        if not self.error_logger.handlers:
            error_handler = logging.FileHandler(logs_dir / 'snomed_errors.log')
            error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            error_handler.setFormatter(error_formatter)
            self.error_logger.addHandler(error_handler)
            self.error_logger.setLevel(logging.WARNING)
        
        # API call logger
        self.api_logger = logging.getLogger(f'{self.vocab_name}_api')
        if not self.api_logger.handlers:
            api_handler = logging.FileHandler(logs_dir / 'snomed_api.log')
            api_formatter = logging.Formatter('%(asctime)s - %(message)s')
            api_handler.setFormatter(api_formatter)
            self.api_logger.addHandler(api_handler)
            self.api_logger.setLevel(logging.INFO)
            
        # Statistics logger
        self.stats_logger = logging.getLogger(f'{self.vocab_name}_stats')
        if not self.stats_logger.handlers:
            stats_handler = logging.FileHandler(logs_dir / 'snomed_stats.log')
            stats_formatter = logging.Formatter('%(asctime)s - %(message)s')
            stats_handler.setFormatter(stats_formatter)
            self.stats_logger.addHandler(stats_handler)
            self.stats_logger.setLevel(logging.INFO)
            
    def validate_code(self, code: str) -> bool:
        """
        Validate SNOMED CT code format.
        SNOMED codes are numeric strings, typically 6-18 digits.
        """
        if not code or not isinstance(code, str):
            return False
        
        code = code.strip()
        return code.isdigit() and 6 <= len(code) <= 18
        
    def _make_request_with_retry(self, url: str, params: Optional[Dict] = None, **kwargs) -> Optional[Any]:
        """
        Make HTTP request with exponential backoff retry logic and comprehensive error handling.
        """
        if params is None:
            params = {}
            
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                # Log API call
                self.api_logger.info(f"API call attempt {attempt + 1}/{self.max_retries}: {url}")
                
                kwargs.setdefault('timeout', self.timeout)
                headers = kwargs.pop('headers', {})
                headers.setdefault('Accept', 'application/json')
                response = requests.get(url, params=params, headers=headers, **kwargs)
                
                # Handle different response codes
                if response.status_code == 200:
                    if "application/json" in response.headers.get("Content-Type", ""):
                        return response.json()
                    return response.text
                elif response.status_code == 404:
                    # Code not found - not an error for our purposes
                    self.api_logger.info(f"Code not found (404): {url}")
                    return None
                elif response.status_code == 429:
                    # Rate limited
                    self.stats['rate_limit_hits'] += 1
                    retry_after = int(response.headers.get('Retry-After', self.base_retry_delay * (2 ** attempt)))
                    self.logger.warning(f"Rate limited. Waiting {retry_after}s before retry {attempt + 1}/{self.max_retries}")
                    time.sleep(retry_after)
                    continue
                else:
                    # Other HTTP error
                    response.raise_for_status()
                    
            except requests.exceptions.Timeout:
                self.stats['timeouts'] += 1
                self.error_logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries} for {url}")
                
            except requests.exceptions.RequestException as e:
                self.stats['api_errors'] += 1
                self.error_logger.warning(f"Request error on attempt {attempt + 1}/{self.max_retries} for {url}: {str(e)}")
                
            except Exception as e:
                self.error_logger.error(f"Unexpected error on attempt {attempt + 1}/{self.max_retries} for {url}: {str(e)}")
            
            # Exponential backoff for retries
            if attempt < self.max_retries - 1:
                delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                # Add jitter to prevent thundering herd
                delay += random.uniform(0, 0.5)
                self.logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
        
        # All retries failed
        self.error_logger.error(f"All {self.max_retries} attempts failed for {url}")
        return None
        
    def _get_concept_details(self, sctid: str) -> Optional[Dict[str, Any]]:
        """
        Get concept details including active status and preferred term.
        Uses caching for performance.
        """
        # Check cache first
        if sctid in self.concept_cache:
            self.stats['cache_hits'] += 1
            return self.concept_cache[sctid]
            
        # Check if we know it's inactive
        if sctid in self.inactive_codes_cache:
            self.stats['cache_hits'] += 1
            return None
            
        self.stats['cache_misses'] += 1
        
        url = f"{self.base_url}browser/{self.branch}/concepts/{sctid}"
        response = self._make_request_with_retry(url)
        
        if response:
            # Cache the response
            self.concept_cache[sctid] = response
            
            # If inactive, add to inactive cache
            if not response.get('active', False):
                self.inactive_codes_cache.add(sctid)
                
        return response
        
    def _get_snomed_parents(self, sctid: str) -> List[Dict[str, str]]:
        """
        Get SNOMED parents using the exact working approach with enhanced caching.
        """
        # Check cache first
        if sctid in self.parents_cache:
            self.stats['cache_hits'] += 1
            return self.parents_cache[sctid]
            
        self.stats['cache_misses'] += 1
        
        url = f"{self.base_url}browser/{self.branch}/concepts/{sctid}/parents"
        data = self._make_request_with_retry(url, params={"form": "inferred"})
        
        parents = []
        if isinstance(data, list):
            parents = [
                {"code": p["conceptId"], "name": p.get("pt", {}).get("term", "N/A")} 
                for p in data
            ]
        elif isinstance(data, dict) and 'items' in data:
            parents = [
                {"code": p["conceptId"], "name": p.get("pt", {}).get("term", "N/A")} 
                for p in data['items']
            ]
        
        # Cache the result
        self.parents_cache[sctid] = parents
        return parents
        
    def _get_ancestry_paths(self, start_id: str, start_name: str) -> Dict[str, Any]:
        """
        Get ancestry paths using the exact working approach with enhanced error handling.
        """
        all_paths = []
        to_process = [(start_id, [{"code": start_id, "name": start_name}])]
        processed_paths = set()
        
        iterations = 0
        max_iterations = 3000  # Production setting for comprehensive extraction
        
        while to_process and iterations < max_iterations:
            iterations += 1
            current_id, current_path = to_process.pop(0)
            
            # Get parents with error handling
            try:
                parents = self._get_snomed_parents(current_id)
            except Exception as e:
                self.error_logger.error(f"Error getting parents for {current_id}: {str(e)}")
                continue
                
            if not parents:
                # Reached root - add reversed path
                all_paths.append(current_path[::-1])
            else:
                for parent in parents:
                    p_id, p_name = parent.get("code"), parent.get("name")
                    if p_id:
                        # Check for cycles using path key
                        path_key = tuple(p['code'] for p in current_path) + (p_id,)
                        if path_key not in processed_paths:
                            new_path = current_path + [{"code": p_id, "name": p_name}]
                            processed_paths.add(path_key)
                            to_process.append((p_id, new_path))
                            
            # Safety check for runaway processing
            if iterations % 100 == 0:
                self.logger.debug(f"Processing iteration {iterations} for {start_id}, {len(to_process)} remaining")
        
        if iterations >= max_iterations:
            self.error_logger.warning(f"Max iterations ({max_iterations}) reached for {start_id}")
            
        return {"native_code": start_id, "paths": all_paths}
        
    def extract_paths(self, code: str, tty: Optional[str] = None, cui: Optional[str] = None) -> List[List[Dict[str, str]]]:
        """
        Extract hierarchical paths for a SNOMED CT code with comprehensive error handling.
        
        Args:
            code: SNOMED CT concept ID (SCTID)
            tty: Not used for SNOMED CT
            cui: CUI for tracking statistics (optional)
            
        Returns:
            List of paths from root to concept, each path as list of {'code', 'name'} dicts
        """
        self.stats['total_codes_processed'] += 1
        
        # Validate code format
        if not self.validate_code(code):
            self.error_logger.warning(f"Invalid SNOMED code format: {code}")
            return []
            
        try:
            # Get concept details
            concept_details = self._get_concept_details(code)
            
            if not concept_details:
                self.logger.debug(f"Concept not found: {code}")
                return []
            
            # Check if concept is active
            if not concept_details.get('active', False):
                self.stats['inactive_codes_skipped'] += 1
                self.total_inactive_codes += 1  # Update base class statistic
                
                # Track CUI-level inactive codes
                if cui:
                    if cui not in self.cui_results:
                        self.cui_results[cui] = {'codes_processed': 0, 'paths_found': 0, 'inactive_codes': 0}
                    self.cui_results[cui]['codes_processed'] += 1
                    self.cui_results[cui]['inactive_codes'] += 1
                
                if self.stats['inactive_codes_skipped'] % 100 == 0:
                    self.logger.info(f"SNOMED inactive codes skipped: {self.stats['inactive_codes_skipped']}")
                self.logger.debug(f"Skipping inactive code: {code}")
                return []
            
            # Get concept name
            concept_name = concept_details.get('pt', {}).get('term', 'Unknown concept')
            
            # Extract ancestry paths
            result = self._get_ancestry_paths(code, concept_name)
            paths = result.get('paths', [])
            
            # Update comprehensive statistics
            if paths:
                self.stats['successful_extractions'] += 1
                self.stats['total_paths_extracted'] += len(paths)
                self.stats['path_counts'].append(len(paths))  # Track paths per code
                
                # Track path lengths
                for path in paths:
                    self.stats['path_lengths'].append(len(path))
                
                # Update base class statistics
                self.total_codes_with_paths += 1
                
                # Track CUI-level results
                if cui:
                    if cui not in self.cui_results:
                        self.cui_results[cui] = {'codes_processed': 0, 'paths_found': 0, 'inactive_codes': 0}
                    self.cui_results[cui]['codes_processed'] += 1
                    self.cui_results[cui]['paths_found'] += len(paths)
                
                self.logger.debug(f"Extracted {len(paths)} paths for {code}")
            else:
                # Track CUI-level results for unsuccessful codes
                if cui:
                    if cui not in self.cui_results:
                        self.cui_results[cui] = {'codes_processed': 0, 'paths_found': 0, 'inactive_codes': 0}
                    self.cui_results[cui]['codes_processed'] += 1
                
                self.logger.debug(f"No paths found for {code}")
                
            # Log progress periodically - reduced frequency for speed
            if self.stats['total_codes_processed'] % 200 == 0:
                self._log_progress_stats()
                
            return paths
            
        except Exception as e:
            self.stats['api_errors'] += 1
            self.error_logger.error(f"Error extracting paths for {code}: {str(e)}")
            return []
            
    def _log_progress_stats(self):
        """Log current processing statistics."""
        stats = self.stats
        total = stats['total_codes_processed']
        success_rate = (stats['successful_extractions'] / total * 100) if total > 0 else 0
        cache_hit_rate = (stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
        
        self.logger.info(f"Progress: {total} codes processed, {success_rate:.1f}% success rate, "
                        f"{stats['inactive_codes_skipped']} inactive, {stats['api_errors']} errors, "
                        f"{cache_hit_rate:.1f}% cache hit rate")
        
        # Log to stats file
        self.stats_logger.info(json.dumps({
            'timestamp': time.time(),
            'codes_processed': total,
            'success_rate': success_rate,
            'inactive_codes': stats['inactive_codes_skipped'],
            'api_errors': stats['api_errors'],
            'cache_hit_rate': cache_hit_rate,
            'total_paths': stats['total_paths_extracted']
        }))
        
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        base_stats = super().get_processing_stats()
        
        # Calculate derived statistics
        total_processed = self.stats['total_codes_processed']
        success_rate = (self.stats['successful_extractions'] / total_processed) if total_processed > 0 else 0
        
        total_cache_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = (self.stats['cache_hits'] / total_cache_requests) if total_cache_requests > 0 else 0
        
        # Calculate path statistics
        path_stats = {}
        if self.stats['path_counts']:
            import statistics
            path_stats.update({
                'paths_per_code': {
                    'mean': statistics.mean(self.stats['path_counts']),
                    'median': statistics.median(self.stats['path_counts']),
                    'min': min(self.stats['path_counts']),
                    'max': max(self.stats['path_counts'])
                }
            })
        
        if self.stats['path_lengths']:
            import statistics
            path_stats.update({
                'path_lengths': {
                    'mean': statistics.mean(self.stats['path_lengths']),
                    'median': statistics.median(self.stats['path_lengths']),
                    'min': min(self.stats['path_lengths']),
                    'max': max(self.stats['path_lengths'])
                }
            })
        
        # Calculate CUI-level statistics
        cui_stats = {
            'total_cuis_processed': len(self.cui_results),
            'cuis_with_paths': len([cui for cui, data in self.cui_results.items() if data['paths_found'] > 0]),
            'cuis_without_paths': len([cui for cui, data in self.cui_results.items() if data['paths_found'] == 0]),
        }
        cui_stats['cui_success_rate'] = (cui_stats['cuis_with_paths'] / cui_stats['total_cuis_processed']) if cui_stats['total_cuis_processed'] > 0 else 0
        
        enhanced_stats = {
            **base_stats,
            **self.stats,
            'success_rate': success_rate,
            'cache_hit_rate': cache_hit_rate,
            'avg_paths_per_code': (self.stats['total_paths_extracted'] / self.stats['successful_extractions']) if self.stats['successful_extractions'] > 0 else 0,
            'api_calls_saved_by_cache': self.stats['cache_hits'],
            'path_statistics': path_stats,
            'cui_statistics': cui_stats,
            # Override base stats with correct values
            'total_codes_processed': self.stats['total_codes_processed'],
            'total_codes_with_paths': self.stats['successful_extractions'],
            'total_inactive_codes': self.stats['inactive_codes_skipped'],
            'total_api_errors': self.stats['api_errors'],
            'code_success_rate': success_rate
        }
        
        return enhanced_stats
        
    def save_progress(self, output_dir: Path):
        """Save current progress and statistics."""
        progress_file = output_dir / 'snomed_progress.json'
        
        progress_data = {
            'timestamp': time.time(),
            'statistics': self.get_processing_stats(),
            'cache_sizes': {
                'parents_cache': len(self.parents_cache),
                'concept_cache': len(self.concept_cache),
                'inactive_codes_cache': len(self.inactive_codes_cache)
            }
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
            
        self.logger.info(f"Progress saved to {progress_file}")
        
    def clear_caches(self):
        """Clear all caches to free memory."""
        cache_sizes = {
            'parents': len(self.parents_cache),
            'concepts': len(self.concept_cache),
            'inactive': len(self.inactive_codes_cache)
        }
        
        self.parents_cache.clear()
        self.concept_cache.clear()
        # Keep inactive codes cache for efficiency
        
        self.logger.info(f"Cleared caches: {cache_sizes}")


# Register the extractor for multiple SNOMED vocabulary names
ExtractorRegistry.register('SNOMED_CT_V2', SnomedExtractor)
ExtractorRegistry.register('SNOMEDCT_US', SnomedExtractor)
ExtractorRegistry.register('SNOMED_CT', SnomedExtractor)