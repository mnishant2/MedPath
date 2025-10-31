"""
HPO hierarchical path extractor.
Enhanced version with comprehensive error handling, caching, statistics, and logging.
Uses HPO OBO file for hierarchical path extraction with goatools.
"""

import pandas as pd
import re
import time
import statistics
from typing import List, Dict, Optional, Any, Set
from collections import defaultdict
import logging
from pathlib import Path

from .base_extractor import BaseExtractor, ExtractorRegistry


class HPOExtractor(BaseExtractor):
    """Enhanced HPO extractor with comprehensive error handling and optimization."""
    
    def __init__(self, config: Dict[str, Any], vocab_name: str):
        super().__init__(config, vocab_name)
        
        # Enhanced error handling settings - OPTIMIZED FOR FILE PROCESSING
        self.max_retries = 2  # Fewer retries for file operations
        self.base_retry_delay = 0.1  # Fast retry for file ops
        self.max_retry_delay = 1.0  # Lower backoff cap for file processing
        self.timeout = 30  # Timeout for file operations
        self.rate_limit_delay = 0.001  # Minimal delay for file-based processing
        
        # HPO data storage
        self.hpo_dag = None
        
        # Multi-level caching for performance
        self.path_cache = {}  # Cache computed paths
        self.failed_codes_cache = set()  # Cache known failed codes
        
        # Enhanced statistics tracking
        self.stats = {
            'total_codes_processed': 0,
            'successful_extractions': 0,
            'file_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_paths_extracted': 0,
            'data_load_errors': 0,
            'cycle_detections': 0,
            'path_counts': [],  # Track paths per successful code
            'path_lengths': [],  # Track length of each path
            'max_paths_encountered': 0,  # Track maximum paths found for any code (before truncation)
            'codes_truncated': 0  # Track how many codes were truncated
        }
        
        # CUI-level tracking (synced with base class)
        self.cui_results = {}  # Track results per CUI
        
        # Batch processing settings - ULTRA-OPTIMIZED FOR SPEED
        self.batch_size = 500  # Larger batches for file-based processing
        self.save_frequency = 200  # Less frequent saves for better performance
        
        # Enhanced logging setup
        self._setup_enhanced_logging()
        
        # Performance settings
        self.max_path_length = 200  # Reasonable depth for HPO hierarchies
        self.max_paths_per_code = 2000  # Same as SNOMED limit
        
        # Load HPO data with retry logic
        self._load_hpo_data_with_retry()
        
        self.logger.info(f"Initialized HPO extractor for {vocab_name}")
        self.logger.info(f"Configuration: max_retries={self.max_retries}, max_path_length={self.max_path_length}")
        
    def _setup_enhanced_logging(self):
        """Setup comprehensive logging compatible with SNOMED structure."""
        # Use the logger from base extractor - don't create separate logs here
        # This will use the main orchestrator logging setup
        pass
        
    def validate_code(self, code: str) -> bool:
        """
        Validate HPO code format.
        HPO codes follow the pattern HP:0000000 (HP: followed by 7 digits)
        """
        if not code or not isinstance(code, str):
            return False
        
        # Remove any whitespace
        code = code.strip()
        
        # HPO pattern: HP:0000000 (HP: followed by exactly 7 digits)
        hpo_pattern = r'^HP:\d{7}$'
        
        return bool(re.match(hpo_pattern, code))
        
    def extract_paths(self, code: str, tty: Optional[str] = None, cui: Optional[str] = None) -> List[List[Dict[str, str]]]:
        """
        Extract hierarchical paths for an HPO code with enhanced error handling.
        
        Args:
            code: HPO code (e.g., "HP:0004322")
            tty: Not used for HPO
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
            self.logger.warning(f"Invalid HPO code format: {code}")
            if cui:
                self.track_cui_code_result(cui, code, 'error')
                self.cui_results[cui]['failed_codes'].append(code)
                self.cui_results[cui]['codes_processed'] += 1
            return []
        
        code = code.strip()
        
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
            paths = self._find_paths_with_retry(code)
            
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
            if self.stats['total_codes_processed'] % 500 == 0:
                self._log_progress_stats()
                
            return paths
            
        except Exception as e:
            self.stats['file_errors'] += 1
            self.failed_codes_cache.add(code)
            self.logger.error(f"Error extracting HPO paths for {code}: {str(e)}")
            
            if cui:
                self.track_cui_code_result(cui, code, 'error')
                self.cui_results[cui]['failed_codes'].append(code)
                self.cui_results[cui]['codes_processed'] += 1
            return []
            
    def _find_paths_with_retry(self, code: str) -> List[List[Dict[str, str]]]:
        """Find HPO paths with retry logic for file operations."""
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Path extraction attempt {attempt + 1}/{self.max_retries} for code: {code}")
                
                paths = self._find_paths(code)
                return paths
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                    time.sleep(delay)
                    self.logger.warning(f"Retry {attempt + 1} failed for {code}, retrying in {delay}s: {str(e)}")
                    continue
                else:
                    raise
                    
        # If we get here, all retries failed
        raise Exception(f"All {self.max_retries} attempts failed for {code}")
        
    def _find_paths(self, code: str) -> List[List[Dict[str, str]]]:
        """
        Find hierarchical paths for an HPO code.
        Returns paths in ROOT->LEAF format (consistent with other vocabularies).
        """
        if not self.hpo_dag:
            raise Exception("HPO DAG not loaded")
        
        # Use the HPO.py logic adapted for our format
        if code not in self.hpo_dag:
            return []
        
        term = self.hpo_dag[code]
        
        def _traverse(node, path):
            # Convert to our standard format: {'code': id, 'name': name}
            current_node = {'code': node.id, 'name': node.name}
            
            if not node.parents:
                # Root node - return path starting from root (this node) + accumulated path
                return [[current_node] + path]
            
            all_paths = []
            for parent in node.parents:
                # Build path from parent down to current node
                parent_paths = _traverse(parent, [current_node] + path)
                all_paths.extend(parent_paths)
            return all_paths
        
        # Get all paths - this builds ROOT->LEAF paths correctly
        paths = _traverse(term, [])
        
        # NO REVERSAL NEEDED - paths are already ROOT->LEAF from traversal
        # The original HPO.py was wrong, we want ROOT->LEAF like SNOMED
        
        # Apply path length and count limits (same as SNOMED)
        filtered_paths = []
        for path in paths:
            if len(path) <= self.max_path_length:
                filtered_paths.append(path)
            else:
                self.stats['codes_truncated'] += 1
        
        # Track maximum paths encountered
        self.stats['max_paths_encountered'] = max(self.stats['max_paths_encountered'], len(paths))
        
        # Limit number of paths per code
        if len(filtered_paths) > self.max_paths_per_code:
            filtered_paths = filtered_paths[:self.max_paths_per_code]
            self.stats['codes_truncated'] += 1
        
        return filtered_paths
        
    def _log_progress_stats(self):
        """Log current progress statistics."""
        total_processed = self.stats['total_codes_processed']
        successful = self.stats['successful_extractions']
        success_rate = (successful / total_processed * 100) if total_processed > 0 else 0
        cache_hit_rate = (self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) * 100) if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
        
        self.logger.info(f"Progress: {total_processed} codes processed, {successful} successful ({success_rate:.1f}%), "
                         f"cache hit rate: {cache_hit_rate:.1f}%, file errors: {self.stats['file_errors']}")
        
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
        
        # Enhanced statistics like other vocabularies
        enhanced_stats = {
            **base_stats,
            **self.stats,
            'success_rate': (self.stats['successful_extractions'] / self.stats['total_codes_processed']) if self.stats['total_codes_processed'] > 0 else 0,
            'cache_hit_rate': (self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])) if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0,
            'avg_paths_per_code': (self.stats['total_paths_extracted'] / self.stats['successful_extractions']) if self.stats['successful_extractions'] > 0 else 0,
            'path_statistics': path_stats,
            'cui_statistics': cui_stats,
            
            # HPO-specific stats
            'total_codes_with_paths': self.stats['successful_extractions'],
            'total_inactive_codes': 0,  # HPO doesn't have inactive codes like SNOMED
            'code_success_rate': (self.stats['successful_extractions'] / self.stats['total_codes_processed']) if self.stats['total_codes_processed'] > 0 else 0,
            'total_cuis_processed': len(self.cui_results),
            'cuis_with_paths': cui_stats['cuis_with_paths'],
            'cuis_without_paths': cui_stats['cuis_without_paths'],
            'cui_success_rate': cui_stats['cui_success_rate'],
            
            # File operation stats
            'data_load_errors': self.stats['data_load_errors'],
            'cycle_detections': self.stats['cycle_detections'],
            
            # Path truncation stats
            'max_paths_encountered': self.stats['max_paths_encountered'],
            'codes_truncated': self.stats['codes_truncated']
        }
        
        return enhanced_stats
        
    def _load_hpo_data_with_retry(self):
        """Load HPO data with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"HPO data load attempt {attempt + 1}/{self.max_retries}")
                self._load_hpo_data()
                return
                
            except Exception as e:
                self.stats['data_load_errors'] += 1
                if attempt < self.max_retries - 1:
                    delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                    time.sleep(delay)
                    self.logger.warning(f"HPO data load retry {attempt + 1} failed, retrying in {delay}s: {str(e)}")
                    continue
                else:
                    raise
                    
        # If we get here, all retries failed
        raise Exception(f"All {self.max_retries} HPO data load attempts failed")
        
    def _load_hpo_data(self):
        """Load HPO data from OBO file."""
        try:
            # Import goatools dynamically to avoid issues in multiprocess workers
            import importlib.util
            
            # Check if goatools is available
            try:
                from goatools.obo_parser import GODag
            except ImportError:
                raise Exception("goatools library not found. Please install with: pip install goatools")
            
            # HPO OBO file path
            from pathlib import Path
            hpo_obo_path = str((Path(__file__).parents[3] / "path_data" / "hp.obo").resolve())
            
            # Load HPO DAG
            self.logger.info(f"Loading HPO OBO from {hpo_obo_path}")
            self.hpo_dag = GODag(hpo_obo_path)
            
            self.logger.info(f"Loaded {len(self.hpo_dag)} HPO terms")
            
        except Exception as e:
            self.logger.error(f"Error loading HPO data: {e}")
            raise


# Register the enhanced extractor
ExtractorRegistry.register('HPO', HPOExtractor)