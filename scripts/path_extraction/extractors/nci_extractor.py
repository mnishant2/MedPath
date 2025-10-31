"""
NCI Thesaurus hierarchical path extractor.
Enhanced version with comprehensive error handling, caching, statistics, and logging.
"""

import os
import re
import time
import statistics
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
import logging

try:
    from owlready2 import get_ontology
    OWLREADY2_AVAILABLE = True
except ImportError:
    OWLREADY2_AVAILABLE = False

from .base_extractor import BaseExtractor, ExtractorRegistry


class NCIExtractor(BaseExtractor):
    """Enhanced NCI extractor with comprehensive error handling and optimization."""
    
    def __init__(self, config: Dict[str, Any], vocab_name: str):
        super().__init__(config, vocab_name)
        
        if not OWLREADY2_AVAILABLE:
            raise ImportError("owlready2 is required for NCI extraction. Install with: pip install owlready2")
        
        # Enhanced error handling settings - OPTIMIZED FOR FILE-BASED PROCESSING
        self.max_retries = 2  # Fewer retries for file operations
        self.base_retry_delay = 0.1  # Fast retry for file ops
        self.max_retry_delay = 1.0  # Lower backoff cap
        self.timeout = 30  # Timeout for file operations
        self.rate_limit_delay = 0.001  # Minimal delay for file-based processing
        
        # Multi-level caching for performance
        self.path_cache = {}  # Cache computed paths
        self.parent_cache = {}  # Cache parent relationships
        self.label_cache = {}  # Cache concept labels
        self.failed_codes_cache = set()  # Cache known failed codes
        
        # Enhanced statistics tracking
        self.stats = {
            'total_codes_processed': 0,
            'successful_extractions': 0,
            'file_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_paths_extracted': 0,
            'ontology_errors': 0,
            'cycle_detections': 0,
            'path_counts': [],  # Track paths per successful code
            'path_lengths': []  # Track length of each path
        }
        
        # CUI-level tracking (synced with base class)
        self.cui_results = {}  # Track results per CUI
        
        # Batch processing settings - OPTIMIZED FOR SPEED
        self.batch_size = 500  # Larger batches for file-based processing
        self.save_frequency = 200  # Less frequent saves for better performance
        
        # Enhanced logging setup
        self._setup_enhanced_logging()
        
        # Get NCI-specific configuration
        nci_config = config.get('local_files', {}).get('nci', {})
        self.owl_file = nci_config.get('owl_file', 'Thesaurus.owl')
        
        # Performance settings
        self.max_path_length = 50  # Reasonable depth for NCI
        self.max_paths_per_code = 1000  # High limit for NCI's complex hierarchies
        
        # Load ontology
        self.ontology = None
        self.code_to_concept = {}
        self._load_ontology_with_retry()
        
        self.logger.info(f"Initialized NCI extractor with {len(self.code_to_concept):,} concepts")
        self.logger.info(f"Configuration: max_retries={self.max_retries}, max_path_length={self.max_path_length}")
        
    def _setup_enhanced_logging(self):
        """Setup comprehensive logging with separate files for different types."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # File operations log
        file_logger = logging.getLogger(f"{self.vocab_name}_file")
        file_handler = logging.FileHandler(log_dir / f"nci_file_ops.log")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        file_logger.addHandler(file_handler)
        file_logger.setLevel(logging.INFO)
        
        # Statistics log
        stats_logger = logging.getLogger(f"{self.vocab_name}_stats")
        stats_handler = logging.FileHandler(log_dir / f"nci_stats.log")
        stats_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        stats_logger.addHandler(stats_handler)
        stats_logger.setLevel(logging.INFO)
        
        # Errors log
        error_logger = logging.getLogger(f"{self.vocab_name}_errors")
        error_handler = logging.FileHandler(log_dir / f"nci_errors.log")
        error_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        error_logger.addHandler(error_handler)
        error_logger.setLevel(logging.ERROR)
        
    def validate_code(self, code: str) -> bool:
        """
        Validate NCI code format.
        NCI codes start with 'C' followed by digits.
        """
        if not code or not isinstance(code, str):
            return False
        
        code = code.strip().upper()
        return bool(re.match(r'^C\d+$', code))
        
    def extract_paths(self, code: str, tty: Optional[str] = None, cui: Optional[str] = None) -> List[List[Dict[str, str]]]:
        """
        Extract hierarchical paths for an NCI code with enhanced error handling.
        
        Args:
            code: NCI code (e.g., "C79698")
            tty: Not used for NCI
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
            self.logger.warning(f"Invalid NCI code format: {code}")
            if cui:
                self.track_cui_code_result(cui, code, 'error')
                self.cui_results[cui]['failed_codes'].append(code)
                self.cui_results[cui]['codes_processed'] += 1
            return []
        
        code = code.upper()
        
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
            self.logger.error(f"Error extracting NCI paths for {code}: {str(e)}")
            
            error_logger = logging.getLogger(f"{self.vocab_name}_errors")
            error_logger.error(f"Failed to extract paths for {code}: {str(e)}")
            
            if cui:
                self.track_cui_code_result(cui, code, 'error')
                self.cui_results[cui]['failed_codes'].append(code)
                self.cui_results[cui]['codes_processed'] += 1
            return []
            
    def _find_all_parent_paths_with_retry(self, code: str) -> List[List[Dict[str, str]]]:
        """Find parent paths with retry logic for file operations."""
        file_logger = logging.getLogger(f"{self.vocab_name}_file")
        
        for attempt in range(self.max_retries):
            try:
                file_logger.info(f"Path extraction attempt {attempt + 1}/{self.max_retries} for code: {code}")
                
                paths = self._find_all_parent_paths(code)
                return paths
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                    time.sleep(delay)
                    file_logger.warning(f"Retry {attempt + 1} failed for {code}, retrying in {delay}s: {str(e)}")
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
        
        # Enhanced statistics like SNOMED/MeSH
        enhanced_stats = {
            **base_stats,
            **self.stats,
            'success_rate': (self.stats['successful_extractions'] / self.stats['total_codes_processed']) if self.stats['total_codes_processed'] > 0 else 0,
            'cache_hit_rate': (self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])) if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0,
            'avg_paths_per_code': (self.stats['total_paths_extracted'] / self.stats['successful_extractions']) if self.stats['successful_extractions'] > 0 else 0,
            'api_calls_saved_by_cache': self.stats['cache_hits'],  # File ops saved by cache
            'path_statistics': path_stats,
            'cui_statistics': cui_stats,
            
            # NCI-specific stats
            'total_codes_with_paths': self.stats['successful_extractions'],
            'total_inactive_codes': 0,  # NCI doesn't have inactive codes like SNOMED
            'code_success_rate': (self.stats['successful_extractions'] / self.stats['total_codes_processed']) if self.stats['total_codes_processed'] > 0 else 0,
            'total_cuis_processed': len(self.cui_results),
            'cuis_with_paths': cui_stats['cuis_with_paths'],
            'cuis_without_paths': cui_stats['cuis_without_paths'],
            'cui_success_rate': cui_stats['cui_success_rate'],
            
            # File operation stats
            'ontology_load_errors': self.stats['ontology_errors'],
            'cycle_detections': self.stats['cycle_detections']
        }
        
        return enhanced_stats
        
    def _load_ontology_with_retry(self):
        """Load NCI Thesaurus OWL ontology with retry logic."""
        file_logger = logging.getLogger(f"{self.vocab_name}_file")
        
        for attempt in range(self.max_retries):
            try:
                file_logger.info(f"Ontology load attempt {attempt + 1}/{self.max_retries}")
                self._load_ontology()
                return
                
            except Exception as e:
                self.stats['ontology_errors'] += 1
                if attempt < self.max_retries - 1:
                    delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                    time.sleep(delay)
                    file_logger.warning(f"Ontology load retry {attempt + 1} failed, retrying in {delay}s: {str(e)}")
                    continue
                else:
                    raise
                    
        # If we get here, all retries failed
        raise Exception(f"All {self.max_retries} ontology load attempts failed")
        
    def _load_ontology(self):
        """Load NCI Thesaurus OWL ontology."""
        owl_path = Path(self.owl_file)
        
        if not owl_path.exists():
            # Try absolute path
            if not owl_path.is_absolute():
                # Look in common locations
                possible_paths = [
                    Path.cwd() / self.owl_file,
                    Path(__file__).parents[3] / "path_data" / self.owl_file
                ]
                
                for path in possible_paths:
                    if path.exists():
                        owl_path = path
                        break
                else:
                    raise FileNotFoundError(f"NCI OWL file not found: {self.owl_file}")
        
        self.logger.info(f"Loading NCI Thesaurus from {owl_path}")
        
        try:
            # Load ontology
            ontology_uri = f"file://{os.path.abspath(owl_path)}"
            self.ontology = get_ontology(ontology_uri).load()
            
            # Build code to concept mapping
            self._build_code_mapping()
            
        except Exception as e:
            self.logger.error(f"Error loading NCI ontology: {e}")
            raise
            
    def _build_code_mapping(self):
        """Build mapping from NCI codes to concepts."""
        self.logger.info("Building NCI code to concept mapping...")
        
        self.code_to_concept = {}
        
        for concept in self.ontology.classes():
            if concept.name and concept.name.startswith('C') and concept.name[1:].isdigit():
                self.code_to_concept[concept.name.upper()] = concept
                
        self.logger.info(f"Built mapping for {len(self.code_to_concept):,} NCI concepts")
        
    def _get_concept_label(self, concept) -> str:
        """Get preferred label for a concept with caching."""
        concept_name = concept.name
        
        # Check cache first
        if concept_name in self.label_cache:
            return self.label_cache[concept_name]
        
        try:
            if hasattr(concept, 'label') and concept.label:
                labels = list(concept.label)
                label = str(labels[0]) if labels else concept_name
            else:
                label = concept_name
        except:
            label = concept_name or 'Unknown'
        
        # Cache the result
        self.label_cache[concept_name] = label
        return label
            
    def _get_parent_concepts(self, concept) -> List:
        """Get parent concepts from is_a relationships with caching."""
        concept_name = concept.name
        
        # Check cache first
        if concept_name in self.parent_cache:
            return self.parent_cache[concept_name]
        
        parents = []
        
        try:
            for parent in concept.is_a:
                # Skip owl:Thing and focus on actual NCI concepts
                if hasattr(parent, 'name') and parent.name and parent.name != 'Thing':
                    # Handle direct class references
                    if parent.name.startswith('C') and parent.name[1:].isdigit():
                        parents.append(parent)
                    # Handle property restrictions
                    elif hasattr(parent, 'value') and hasattr(parent.value, 'name'):
                        if parent.value.name.startswith('C') and parent.value.name[1:].isdigit():
                            parents.append(parent.value)
        except Exception as e:
            self.logger.warning(f"Error getting parents for {concept_name}: {e}")
            
        # Cache the result
        self.parent_cache[concept_name] = parents
        return parents
        
    def _find_all_parent_paths(self, code: str, current_path: Optional[List[Dict]] = None, 
                              visited: Optional[Set[str]] = None) -> List[List[Dict[str, str]]]:
        """
        Find all parent paths for an NCI concept using DFS.
        PATHS ARE ROOT->LEAF (consistent with SNOMED/MeSH)
        
        Args:
            code: NCI code to find paths for
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
            
        concept = self.code_to_concept.get(code)
        if not concept:
            # Unknown concept - create minimal path
            unknown_node = {'code': code, 'name': f'Unknown NCI Concept {code}'}
            return [current_path + [unknown_node]] if current_path else [[unknown_node]]
            
        visited.add(code)
        
        # Get concept information
        concept_name = self._get_concept_label(concept)
        current_node = {'code': code, 'name': concept_name}
        
        # Get parent concepts
        parents = self._get_parent_concepts(concept)
        
        if not parents:
            # This is a root concept - return complete path (ROOT->LEAF)
            complete_path = current_path + [current_node]
            return [complete_path]
            
        # Recursively find paths through all parents
        all_paths = []
        for parent in parents:
            parent_code = parent.name
            if parent_code and parent_code not in visited:
                # Build path properly: start from parent, then append current node
                parent_paths = self._find_all_parent_paths(
                    parent_code, 
                    current_path,  # Don't include current node yet
                    visited.copy()
                )
                # Append current node to each parent path
                for parent_path in parent_paths:
                    complete_path = parent_path + [current_node]
                    all_paths.append(complete_path)
                
        # Limit number of paths
        if len(all_paths) > self.max_paths_per_code:
            self.logger.warning(f"NCI code {code} has {len(all_paths)} paths, truncating to {self.max_paths_per_code}")
            all_paths = all_paths[:self.max_paths_per_code]
            
        return all_paths


# Register the enhanced extractor
ExtractorRegistry.register('NCI', NCIExtractor)
ExtractorRegistry.register('NCI_V2', NCIExtractor)  # Alternative name