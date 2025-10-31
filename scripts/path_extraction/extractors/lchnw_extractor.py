"""
LCH_NW (Library of Congress Subject Headings) hierarchical path extractor.
Enhanced version with comprehensive error handling, caching, statistics, and logging.
Uses local JSONLD SKOS file with optimized file processing and intelligent caching.
"""

import json
import re
import time
import statistics
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
from collections import defaultdict
import logging

from .base_extractor import BaseExtractor, ExtractorRegistry


class LCHNWExtractor(BaseExtractor):
    """Enhanced LCH_NW extractor with comprehensive error handling and optimization."""
    
    def __init__(self, config: Dict[str, Any], vocab_name: str):
        super().__init__(config, vocab_name)
        
        # Enhanced error handling settings - OPTIMIZED FOR FILE PROCESSING
        self.max_retries = 2  # Fewer retries for file operations
        self.base_retry_delay = 0.1  # Fast retry for file ops
        self.max_retry_delay = 1.0  # Lower backoff cap for file processing
        self.timeout = 30  # Timeout for file operations
        self.rate_limit_delay = 0.001  # Minimal delay for file-based processing
        
        # Get LCH_NW-specific configuration
        lcsh_config = config.get('local_files', {}).get('lcsh', {})
        self.jsonld_file = lcsh_config.get('jsonld_file', 'subjects.skosrdf.jsonld')
        
        # Multi-level caching for performance
        self.path_cache = {}  # Cache computed paths
        self.subject_cache = {}  # Cache subject details
        self.broader_cache = {}  # Cache broader relationships
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
        self.max_path_length = 200  # Reasonable depth for LCSH hierarchies
        self.max_paths_per_code = 2000  # Same as SNOMED limit
        
        # Initialize data structures
        self.subjects = {}
        self.broader_map = defaultdict(set)  # child -> {parents}
        self.narrower_map = defaultdict(set)  # parent -> {children}
        self.uri_to_code = {}  # URI -> code mapping
        
        # Load LCSH data with retry logic
        self._load_lcsh_data_with_retry()
        
        self.logger.info(f"Initialized LCH_NW extractor with {len(self.subjects):,} subjects")
        self.logger.info(f"Configuration: max_retries={self.max_retries}, max_path_length={self.max_path_length}")
        
    def _setup_enhanced_logging(self):
        """Setup comprehensive logging compatible with SNOMED structure."""
        # Use the logger from base extractor - don't create separate logs here
        # This will use the main orchestrator logging setup
        pass
        
    def validate_code(self, code: str) -> bool:
        """
        Validate LCH_NW code format.
        LCH_NW codes are typically LCSH identifiers like "sh85060226".
        """
        if not code or not isinstance(code, str):
            return False
        
        # Remove any whitespace
        code = code.strip()
        
        # Check common LCSH patterns
        patterns = [
            r'^sh\d+$',           # Standard LCSH: sh85060226
            r'^n\d+$',            # Name authority: n79021164
            r'^nb\d+$',           # Name authority: nb2007024370
            r'^no\d+$',           # Name authority: no2008123456
            r'^nr\d+$',           # Name authority: nr95123456
            r'^gf\d+$',           # Genre/form: gf2014026123
            r'^fst\d+$',          # Faceted application: fst00123456
        ]
        
        for pattern in patterns:
            if re.match(pattern, code):
                return True
                
        # Also allow generic alphanumeric codes
        return bool(re.match(r'^[a-zA-Z0-9]+$', code) and len(code) > 3)
        
    def extract_paths(self, code: str, tty: Optional[str] = None, cui: Optional[str] = None) -> List[List[Dict[str, str]]]:
        """
        Extract hierarchical paths for an LCH_NW code with enhanced error handling.
        
        Args:
            code: LCH_NW code (e.g., "sh85060226")
            tty: Not used for LCH_NW
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
            self.logger.warning(f"Invalid LCH_NW code format: {code}")
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
            paths = self._find_all_parent_paths_with_retry(code)
            
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
            self.logger.error(f"Error extracting LCH_NW paths for {code}: {str(e)}")
            
            self.logger.error(f"Failed to extract paths for {code}: {str(e)}")
            
            if cui:
                self.track_cui_code_result(cui, code, 'error')
                self.cui_results[cui]['failed_codes'].append(code)
                self.cui_results[cui]['codes_processed'] += 1
            return []
            
    def _find_all_parent_paths_with_retry(self, code: str) -> List[List[Dict[str, str]]]:
        """Find parent paths with retry logic for file operations."""
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Path extraction attempt {attempt + 1}/{self.max_retries} for code: {code}")
                
                paths = self._find_all_parent_paths(code)
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
            'api_calls_saved_by_cache': self.stats['cache_hits'],  # File ops saved by cache
            'path_statistics': path_stats,
            'cui_statistics': cui_stats,
            
            # LCH_NW-specific stats
            'total_codes_with_paths': self.stats['successful_extractions'],
            'total_inactive_codes': 0,  # LCH_NW doesn't have inactive codes like SNOMED
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
        
    def _load_lcsh_data_with_retry(self):
        """Load LCSH JSONLD data with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"LCSH data load attempt {attempt + 1}/{self.max_retries}")
                self._load_lcsh_data()
                return
                
            except Exception as e:
                self.stats['data_load_errors'] += 1
                if attempt < self.max_retries - 1:
                    delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                    time.sleep(delay)
                    self.logger.warning(f"LCSH data load retry {attempt + 1} failed, retrying in {delay}s: {str(e)}")
                    continue
                else:
                    raise
                    
        # If we get here, all retries failed
        raise Exception(f"All {self.max_retries} LCSH data load attempts failed")
        
    def _load_lcsh_data(self):
        """Load LCSH JSONLD data and build relationship maps."""
        jsonld_path = Path(self.jsonld_file)
        
        if not jsonld_path.exists():
            # Try absolute path
            if not jsonld_path.is_absolute():
                # Look in common locations
                possible_paths = [
                    Path.cwd() / self.jsonld_file,
                    Path(__file__).parents[3] / "path_data" / self.jsonld_file
                ]
                
                for path in possible_paths:
                    if path.exists():
                        jsonld_path = path
                        break
                else:
                    raise FileNotFoundError(f"LCSH JSONLD file not found: {self.jsonld_file}")
        
        self.logger.info(f"Loading LCSH data from {jsonld_path}")
        
        try:
            # Load JSONLD data
            with open(jsonld_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if line.strip():
                            data = json.loads(line.strip())
                            # Handle @graph structure
                            if isinstance(data, dict) and '@graph' in data:
                                for concept in data['@graph']:
                                    self._process_subject(concept)
                            elif isinstance(data, list):
                                for concept in data:
                                    self._process_subject(concept)
                            else:
                                self._process_subject(data)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"JSON decode error at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error processing line {line_num}: {e}")
                        continue
            
            # Build reverse mappings for faster lookups
            self._build_reverse_mappings()
            
        except Exception as e:
            self.logger.error(f"Error loading LCSH data: {e}")
            raise
            
    def _process_subject(self, subject_data: Dict):
        """Process a single subject from JSONLD data."""
        try:
            # Extract subject ID and URI
            subject_id = subject_data.get('@id')
            if not subject_id:
                return
            
            # Only process actual SKOS concepts, not change sets
            if subject_data.get('@type') != 'skos:Concept':
                return
            
            # Extract code from URI (e.g., http://id.loc.gov/authorities/subjects/sh85060226 -> sh85060226)
            code = self._extract_code_from_uri(subject_id)
            if not code:
                return
            
            # Get preferred label
            pref_label = None
            if 'skos:prefLabel' in subject_data:
                labels = subject_data['skos:prefLabel']
                if isinstance(labels, list) and labels:
                    pref_label = labels[0].get('@value', str(labels[0]))
                elif isinstance(labels, dict):
                    pref_label = labels.get('@value', str(labels))
                elif isinstance(labels, str):
                    pref_label = labels
            
            if not pref_label:
                pref_label = code  # Fallback to code
            
            # Store subject
            self.subjects[code] = {
                'uri': subject_id,
                'label': pref_label,
                'code': code
            }
            
            self.uri_to_code[subject_id] = code
            
            # Process broader relationships
            if 'skos:broader' in subject_data:
                broader_refs = subject_data['skos:broader']
                if not isinstance(broader_refs, list):
                    broader_refs = [broader_refs]
                
                for broader_ref in broader_refs:
                    broader_uri = broader_ref.get('@id') if isinstance(broader_ref, dict) else str(broader_ref)
                    if broader_uri:
                        broader_code = self._extract_code_from_uri(broader_uri)
                        if broader_code:
                            self.broader_map[code].add(broader_code)
                            
        except Exception as e:
            self.logger.warning(f"Error processing subject {subject_data.get('@id', 'unknown')}: {e}")
            
    def _extract_code_from_uri(self, uri: str) -> Optional[str]:
        """Extract LCSH code from URI."""
        if not uri:
            return None
        
        # Handle different URI patterns
        patterns = [
            r'/authorities/subjects/([a-zA-Z0-9]+)$',  # http://id.loc.gov/authorities/subjects/sh85060226
            r'/authorities/names/([a-zA-Z0-9]+)$',     # http://id.loc.gov/authorities/names/n79021164
            r'/authorities/genreForms/([a-zA-Z0-9]+)$', # http://id.loc.gov/authorities/genreForms/gf2014026123
            r'/vocabulary/relators/([a-zA-Z0-9]+)$',    # http://id.loc.gov/vocabulary/relators/aut
        ]
        
        for pattern in patterns:
            match = re.search(pattern, uri)
            if match:
                return match.group(1)
        
        # Fallback: try to extract any alphanumeric code at end of URI
        match = re.search(r'/([a-zA-Z0-9]+)$', uri)
        if match:
            return match.group(1)
        
        return None
        
    def _build_reverse_mappings(self):
        """Build reverse mappings for faster lookups."""
        # Build narrower relationships (parent -> children)
        for child, parents in self.broader_map.items():
            for parent in parents:
                self.narrower_map[parent].add(child)
        
        self.logger.info(f"Built relationship maps: {len(self.broader_map)} broader, {len(self.narrower_map)} narrower")
        
    def _find_all_parent_paths(self, code: str, current_path: Optional[List[Dict]] = None, 
                              visited: Optional[Set[str]] = None) -> List[List[Dict[str, str]]]:
        """
        Find all parent paths for an LCH_NW code using DFS.
        PATHS ARE ROOT->LEAF (consistent with other vocabularies)
        
        Args:
            code: LCH_NW code to find paths for
            current_path: Current path being built (root->leaf during construction)
            visited: Set of visited codes to prevent cycles
            
        Returns:
            List of complete paths from root to concept (ROOT->LEAF)
        """
        if current_path is None:
            current_path = []
        if visited is None:
            visited = set()
            
        # Prevent infinite loops and excessive depth
        if code in visited or len(current_path) > self.max_path_length:
            if code in visited:
                self.stats['cycle_detections'] += 1
            return []
            
        subject = self.subjects.get(code)
        if not subject:
            # Unknown subject - create minimal entry
            unknown_node = {'code': code, 'name': f'Unknown LCSH {code}'}
            return [current_path + [unknown_node]] if current_path else [[unknown_node]]
            
        visited.add(code)
        
        # Get subject information
        subject_name = subject.get('label', code)
        current_node = {'code': code, 'name': subject_name}
        
        # Get broader (parent) concepts
        broader_codes = self.broader_map.get(code, set())
        
        if not broader_codes:
            # This is a root concept - return complete path (ROOT->LEAF)
            complete_path = current_path + [current_node]
            return [complete_path]
            
        # Recursively find paths through all broader concepts
        all_paths = []
        for broader_code in broader_codes:
            if broader_code not in visited and broader_code in self.subjects:
                # Build path properly: start from broader concept, then append current
                broader_paths = self._find_all_parent_paths(
                    broader_code, 
                    current_path,  # Pass current path to build from root
                    visited.copy()
                )
                # Append current node to each broader path
                for broader_path in broader_paths:
                    complete_path = broader_path + [current_node]
                    all_paths.append(complete_path)
                
        # If no valid broader paths found, treat as root
        if not all_paths:
            complete_path = current_path + [current_node]
            all_paths.append(complete_path)
                
        # Track maximum paths encountered
        original_path_count = len(all_paths)
        if original_path_count > self.stats['max_paths_encountered']:
            self.stats['max_paths_encountered'] = original_path_count
        
        # Limit number of paths
        if original_path_count > self.max_paths_per_code:
            self.logger.warning(f"LCH_NW code {code} has {original_path_count} paths, truncating to {self.max_paths_per_code}")
            all_paths = all_paths[:self.max_paths_per_code]
            self.stats['codes_truncated'] += 1
            
        return all_paths


# Register the enhanced extractor
ExtractorRegistry.register('LCH_NW', LCHNWExtractor)
ExtractorRegistry.register('LCH_NW_V2', LCHNWExtractor)  # Alternative name (kept for compatibility)