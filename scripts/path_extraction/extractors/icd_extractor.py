"""
ICD-9 and ICD-10 hierarchical path extractor.
Enhanced version with comprehensive error handling, caching, statistics, and logging.
Uses local ICD files for hierarchical path extraction - supports both ICD-9 and ICD-10.
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


class ICDExtractor(BaseExtractor):
    """Enhanced ICD extractor with comprehensive error handling and optimization."""
    
    def __init__(self, config: Dict[str, Any], vocab_name: str):
        super().__init__(config, vocab_name)
        
        # Enhanced error handling settings - OPTIMIZED FOR FILE PROCESSING
        self.max_retries = 2  # Fewer retries for file operations
        self.base_retry_delay = 0.1  # Fast retry for file ops
        self.max_retry_delay = 1.0  # Lower backoff cap for file processing
        self.timeout = 30  # Timeout for file operations
        self.rate_limit_delay = 0.001  # Minimal delay for file-based processing
        
        # Get ICD-specific configuration
        icd_config = config.get('local_files', {}).get('icd', {})
        
        # ICD data storage
        self.icd9_data = {}
        self.icd9_hierarchy = {}
        self.icd10_data = {}
        self.icd10_hierarchy = {}
        
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
        self.max_path_length = 200  # Reasonable depth for ICD hierarchies
        self.max_paths_per_code = 2000  # Same as SNOMED limit
        
        # Load ICD data with retry logic
        self._load_icd_data_with_retry()
        
        self.logger.info(f"Initialized ICD extractor for {vocab_name}")
        self.logger.info(f"Configuration: max_retries={self.max_retries}, max_path_length={self.max_path_length}")
        
    def _setup_enhanced_logging(self):
        """Setup comprehensive logging compatible with SNOMED structure."""
        # Use the logger from base extractor - don't create separate logs here
        # This will use the main orchestrator logging setup
        pass
        
    def validate_code(self, code: str) -> bool:
        """
        Validate ICD-9 or ICD-10 code format with enhanced pattern matching.
        """
        if not code or not isinstance(code, str):
            return False
        
        # Remove any whitespace
        code = code.strip()
        
        # Skip obviously invalid patterns upfront
        if self._is_invalid_pattern(code):
            return False
        
        # Check ICD-10 patterns (enhanced)
        icd10_patterns = [
            r'^[A-Z]\d{2}(\.\d{1,4})?$',        # A00 or A00.1234
            r'^[A-Z]\d{2}[A-Z](\.\d{1,3})?$',   # A00A or A00A.123 (extended)
            r'^[A-Z]\d{2}-[A-Z]\d{2}$',         # Chapter ranges like A00-B99
            r'^[A-Z]\d[A-Z](\.\d{1,3})?$',      # C4A, Z3A patterns
            r'^[A-Z]\d{2}[XYZ](\d|\.[XYZ]\d?[A-Z]?)?$',  # Placeholder patterns S06.2X, S06.2XA
        ]
        
        # Check ICD-9 patterns (enhanced)
        icd9_patterns = [
            r'^\d{3}(\.\d{1,2})?$',              # 123 or 123.45
            r'^V\d{2}(\.\d{1,2})?$',             # V12 or V12.34
            r'^E\d{3,4}(\.\d)?$',                # E123 or E1234.1
            r'^\d{3}-\d{3}(\.99)?$',             # Chapter ranges like 001-139, 001-139.99
            r'^[EV]\d{2,4}-[EV]?\d{2,4}(\.9?9?)?$', # E000-E999, V01-V91, with .99 suffix
            r'^\d{2}(\.\d{1,2})?$',              # Two-digit codes like 30, 50 (some are valid categories)
        ]
        
        all_patterns = icd10_patterns + icd9_patterns
        
        for pattern in all_patterns:
            if re.match(pattern, code):
                return True
                
        return False
    
    def _is_invalid_pattern(self, code: str) -> bool:
        """Check for patterns that are definitely not ICD diagnosis codes."""
        
        # Procedure codes (ICD-9-PCS) - typically 2-4 digits with 1-2 decimal places
        # and start with certain ranges that are procedures, not diagnoses
        if re.match(r'^\d{2}\.\d{1,2}$', code):
            # Two-digit procedure codes like 49.46, 87.64, 89.34, 95.21
            prefix = int(code.split('.')[0])
            # Common procedure code ranges (not exhaustive, but covers major ones)
            procedure_ranges = [
                (0, 16),    # Procedures on nervous system (00-16)
                (17, 20),   # Procedures on endocrine system (17-20)
                (21, 29),   # Procedures on eye (21-29)
                (30, 34),   # Procedures on ear (30-34)
                (35, 39),   # Procedures on cardiovascular system (35-39)
                (40, 41),   # Procedures on hemic and lymphatic system (40-41)
                (42, 54),   # Procedures on respiratory system (42-54)
                (55, 59),   # Procedures on urinary system (55-59)
                (60, 71),   # Procedures on male genital organs (60-71)
                (72, 75),   # Procedures on female genital organs (72-75)
                (76, 84),   # Procedures on musculoskeletal system (76-84)
                (85, 86),   # Procedures on integumentary system (85-86)
                (87, 99),   # Miscellaneous diagnostic and therapeutic procedures (87-99)
            ]
            
            for start, end in procedure_ranges:
                if start <= prefix <= end:
                    return True
        
        # Range codes with .99 suffix are often invalid
        if re.match(r'^\d+-\d+\.99$', code):
            return True
            
        # Single or two digit codes without decimal are often invalid categories
        if re.match(r'^\d{1,2}$', code) and code not in ['30', '50', '52', '53']:  # Allow some valid categories
            return True
            
        # Very broad ranges like 00-99.99 are administrative, not specific codes
        if re.match(r'^0+-9+\.99$', code):
            return True
            
        return False
        
    def extract_paths(self, code: str, tty: Optional[str] = None, cui: Optional[str] = None) -> List[List[Dict[str, str]]]:
        """
        Extract hierarchical paths for an ICD code with enhanced error handling.
        
        Args:
            code: ICD code (e.g., "A01.00" for ICD-10, "789.1" for ICD-9)
            tty: Not used for ICD
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
            self.logger.warning(f"Invalid ICD code format: {code}")
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
            self.logger.error(f"Error extracting ICD paths for {code}: {str(e)}")
            
            if cui:
                self.track_cui_code_result(cui, code, 'error')
                self.cui_results[cui]['failed_codes'].append(code)
                self.cui_results[cui]['codes_processed'] += 1
            return []
            
    def _find_paths_with_retry(self, code: str) -> List[List[Dict[str, str]]]:
        """Find ICD paths with retry logic for file operations."""
        
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
        Find hierarchical paths for an ICD code.
        Returns paths in ROOT->LEAF format (consistent with other vocabularies).
        """
        # Determine if ICD-9 or ICD-10 based on code pattern
        if self._is_icd10_code(code):
            return self._get_icd10_paths(code)
        else:
            return self._get_icd9_paths(code)
    
    def _is_icd10_code(self, code: str) -> bool:
        """Determine if code is ICD-10 format."""
        return bool(re.match(r'^[A-Z]', code))
    
    def _get_icd10_paths(self, code: str) -> List[List[Dict[str, str]]]:
        """Get ICD-10 ancestral paths from loaded data."""
        if code not in self.icd10_data:
            return []
        
        path = []
        current_code = code
        visited = set()
        
        while current_code and current_code not in visited:
            visited.add(current_code)
            
            if current_code in self.icd10_data:
                path.insert(0, {
                    'code': current_code,
                    'name': self.icd10_data[current_code]['name']
                })
                
                parents = self.icd10_hierarchy.get(current_code, [])
                if parents:
                    current_code = parents[0]
                else:
                    break
            else:
                break
        
        return [path] if path else []
    
    def _get_icd9_paths(self, code: str) -> List[List[Dict[str, str]]]:
        """Get ICD-9 ancestral paths from loaded data."""
        if code not in self.icd9_data:
            return []
        
        path = []
        current_code = code
        visited = set()
        
        while current_code and current_code not in visited:
            visited.add(current_code)
            
            if current_code in self.icd9_data:
                path.insert(0, {
                    'code': current_code,
                    'name': self.icd9_data[current_code]['name']
                })
                
                parents = self.icd9_hierarchy.get(current_code, [])
                if parents:
                    current_code = parents[0]
                else:
                    break
            else:
                break
        
        return [path] if path else []
        
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
            
            # ICD-specific stats
            'total_codes_with_paths': self.stats['successful_extractions'],
            'total_inactive_codes': 0,  # ICD doesn't have inactive codes like SNOMED
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
        
    def _load_icd_data_with_retry(self):
        """Load ICD data with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"ICD data load attempt {attempt + 1}/{self.max_retries}")
                self._load_icd_data()
                return
                
            except Exception as e:
                self.stats['data_load_errors'] += 1
                if attempt < self.max_retries - 1:
                    delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                    time.sleep(delay)
                    self.logger.warning(f"ICD data load retry {attempt + 1} failed, retrying in {delay}s: {str(e)}")
                    continue
                else:
                    raise
                    
        # If we get here, all retries failed
        raise Exception(f"All {self.max_retries} ICD data load attempts failed")
        
    def _load_icd_data(self):
        """Load ICD-9 and ICD-10 data based on vocabulary name."""
        try:
            if self.vocab_name == "ICD9CM":
                self._load_icd9_only()
            elif self.vocab_name == "ICD10CM":
                self._load_icd10_only()
            else:
                # Load both for general ICD extractor
                self._load_icd9_only()
                self._load_icd10_only()
        except Exception as e:
            self.logger.error(f"Error loading ICD data: {e}")
            raise
            
    def _load_icd9_only(self):
        """Load ICD-9 data from files."""
        # File paths for ICD-9
        icd9_long_file = "/home/nmishra/data/storage_hpc_nishant/EL_gen/umls_datasets/path_data/ICD-9-CM-v32-master-descriptions/CMS32_DESC_LONG_DX.txt"
        icd9_short_file = "/home/nmishra/data/storage_hpc_nishant/EL_gen/umls_datasets/path_data/ICD-9-CM-v32-master-descriptions/CMS32_DESC_SHORT_DX.txt"
        
        # Import using importlib to avoid relative import issues
        import sys
        import os
        import importlib.util
        
        # Get the ICD_9_10.py module path
        icd_module_path = "/home/nmishra/data/storage_hpc_nishant/EL_gen/umls_datasets/scripts/path_extraction/ICD_9_10.py"
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("icd_9_10", icd_module_path)
        icd_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(icd_module)
        
        finder = icd_module.ICD910PathFinder(icd9_long_file, icd9_short_file, None)
        self.icd9_data = finder.icd9_data
        self.icd9_hierarchy = finder.icd9_hierarchy
        
        self.logger.info(f"Loaded {len(self.icd9_data)} ICD-9 codes")
        
    def _load_icd10_only(self):
        """Load ICD-10 data from files."""
        # File path for ICD-10
        icd10_file = "/home/nmishra/data/storage_hpc_nishant/EL_gen/umls_datasets/path_data/icd10cm-Code Descriptions-2026/icd10cm-codes-2026.txt"
        
        # Import using importlib to avoid relative import issues
        import sys
        import os
        import importlib.util
        
        # Get the ICD_9_10.py module path
        icd_module_path = "/home/nmishra/data/storage_hpc_nishant/EL_gen/umls_datasets/scripts/path_extraction/ICD_9_10.py"
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("icd_9_10", icd_module_path)
        icd_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(icd_module)
        
        finder = icd_module.ICD910PathFinder(None, None, icd10_file)
        self.icd10_data = finder.icd10_data
        self.icd10_hierarchy = finder.icd10_hierarchy
        
        self.logger.info(f"Loaded {len(self.icd10_data)} ICD-10 codes")


# Register the enhanced extractors
ExtractorRegistry.register('ICD9CM', ICDExtractor)
ExtractorRegistry.register('ICD10CM', ICDExtractor)
ExtractorRegistry.register('ICD', ICDExtractor)  # General ICD extractor