"""
NCBI Taxonomy hierarchical path extractor.
Enhanced version with comprehensive error handling, caching, statistics, and logging.
Uses NCBI E-utilities API for taxonomic path extraction.
"""

import requests
import xml.etree.ElementTree as ET
import re
import time
import statistics
from typing import List, Dict, Optional, Any, Set
from collections import defaultdict
import logging

from .base_extractor import BaseExtractor, ExtractorRegistry


class NCBIExtractor(BaseExtractor):
    """Enhanced NCBI extractor with comprehensive error handling and optimization."""
    
    def __init__(self, config: Dict[str, Any], vocab_name: str):
        super().__init__(config, vocab_name)
        
        # Enhanced error handling settings - OPTIMIZED FOR API
        self.max_retries = 3  # More retries for API operations
        self.base_retry_delay = 0.2  # Slower retry for API respect
        self.max_retry_delay = 5.0  # Higher backoff cap for API
        self.timeout = 30  # Timeout for API operations
        self.rate_limit_delay = 0.2  # Respectful delay for NCBI API
        
        # Get NCBI-specific configuration (no hardcoded secrets)
        ncbi_config = config.get('apis', {}).get('ncbi', {})
        self.api_key = ncbi_config.get('api_key')
        self.base_url = ncbi_config.get('base_url', "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")
        # Optional snapshot/version metadata for logging/reporting only
        self.ncbi_snapshot = config.get('versions', {}).get('NCBI_taxonomy_snapshot', '')
        
        # Multi-level caching for performance
        self.path_cache = {}  # Cache computed paths
        self.taxon_cache = {}  # Cache taxon details
        self.failed_codes_cache = set()  # Cache known failed codes
        
        # Enhanced statistics tracking
        self.stats = {
            'total_codes_processed': 0,
            'successful_extractions': 0,
            'api_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_paths_extracted': 0,
            'rate_limit_hits': 0,
            'timeouts': 0,
            'path_counts': [],  # Track paths per successful code
            'path_lengths': [],  # Track length of each path
            'max_paths_encountered': 0,  # Track maximum paths found for any code (before truncation)
            'codes_truncated': 0  # Track how many codes were truncated
        }
        
        # CUI-level tracking (synced with base class)
        self.cui_results = {}  # Track results per CUI
        
        # Batch processing settings - OPTIMIZED FOR API RESPECT
        self.batch_size = 50  # Smaller batches for API processing
        self.save_frequency = 100  # Less frequent saves for better performance
        
        # Enhanced logging setup
        self._setup_enhanced_logging()
        
        # Performance settings
        self.max_path_length = 200  # Reasonable depth for NCBI taxonomies
        self.max_paths_per_code = 2000  # Same as SNOMED limit
        
        self.logger.info("Initialized NCBI extractor (E-utilities)")
        self.logger.info(f"Configuration: max_retries={self.max_retries}, rate_limit={self.rate_limit_delay}s")
        
    def _setup_enhanced_logging(self):
        """Setup comprehensive logging compatible with SNOMED structure."""
        # Use the logger from base extractor - don't create separate logs here
        # This will use the main orchestrator logging setup
        pass
        
    def validate_code(self, code: str) -> bool:
        """
        Validate NCBI taxonomy ID format.
        NCBI taxonomy IDs are typically numeric like "9606", "562".
        """
        if not code or not isinstance(code, str):
            return False
        
        # Remove any whitespace
        code = code.strip()
        
        # Check if it's a valid numeric taxonomy ID
        if re.match(r'^\d+$', code):
            return True
            
        return False
        
    def extract_paths(self, code: str, tty: Optional[str] = None, cui: Optional[str] = None) -> List[List[Dict[str, str]]]:
        """
        Extract hierarchical paths for an NCBI taxonomy ID with enhanced error handling.
        
        Args:
            code: NCBI taxonomy ID (e.g., "9606")
            tty: Not used for NCBI
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
            self.logger.warning(f"Invalid NCBI taxonomy ID format: {code}")
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
            paths = self._get_ancestral_paths_with_retry(code)
            
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
            
        except Exception as e:
            self.stats['api_errors'] += 1
            self.failed_codes_cache.add(code)
            self.logger.error(f"Error extracting NCBI paths for {code}: {str(e)}")
            
            if cui:
                self.track_cui_code_result(cui, code, 'error')
                self.cui_results[cui]['failed_codes'].append(code)
                self.cui_results[cui]['codes_processed'] += 1
            return []
            
    def _get_ancestral_paths_with_retry(self, taxid: str) -> List[List[Dict[str, str]]]:
        """Get NCBI taxonomic paths with retry logic for API operations."""
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"NCBI API attempt {attempt + 1}/{self.max_retries} for taxid: {taxid}")
                
                paths = self._get_ancestral_paths(taxid)
                return paths
                
            except requests.exceptions.Timeout:
                self.stats['timeouts'] += 1
                if attempt < self.max_retries - 1:
                    delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                    time.sleep(delay)
                    self.logger.warning(f"Timeout retry {attempt + 1} for {taxid}, retrying in {delay}s")
                    continue
                else:
                    raise
            except requests.exceptions.RequestException as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    self.stats['rate_limit_hits'] += 1
                    if attempt < self.max_retries - 1:
                        delay = min(self.base_retry_delay * (2 ** attempt) * 2, self.max_retry_delay)  # Longer delay for rate limits
                        time.sleep(delay)
                        self.logger.warning(f"Rate limit retry {attempt + 1} for {taxid}, retrying in {delay}s")
                        continue
                    else:
                        raise
                else:
                    if attempt < self.max_retries - 1:
                        delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                        time.sleep(delay)
                        self.logger.warning(f"API retry {attempt + 1} failed for {taxid}, retrying in {delay}s: {str(e)}")
                        continue
                    else:
                        raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                    time.sleep(delay)
                    self.logger.warning(f"Retry {attempt + 1} failed for {taxid}, retrying in {delay}s: {str(e)}")
                    continue
                else:
                    raise
                    
        # If we get here, all retries failed
        raise Exception(f"All {self.max_retries} attempts failed for {taxid}")
        
    def _get_ancestral_paths(self, taxid: str) -> List[List[Dict[str, str]]]:
        """
        Get NCBI taxonomic paths - works for most organisms.
        Returns paths in ROOT->LEAF format (consistent with other vocabularies).
        """
        url = f"{self.base_url.rstrip('/')}" + "/efetch.fcgi"
        params = {
            'db': 'taxonomy',
            'id': taxid,
            'retmode': 'xml',
        }
        # API key is optional but recommended for higher rate limits
        if self.api_key:
            params['api_key'] = self.api_key
        
        # Apply rate limiting
        time.sleep(self.rate_limit_delay)
        
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        
        root = ET.fromstring(response.text)
        taxon = root.find('.//Taxon')
        
        if taxon is None:
            self.logger.debug(f"No taxonomic data found for {taxid}")
            return []
        
        # Build simple path: code + name only (ROOT->LEAF format)
        path = []
        
        # Get lineage (ancestors from root to parent)
        lineage_ex = taxon.find('LineageEx')
        if lineage_ex is not None:
            for taxon_elem in lineage_ex.findall('Taxon'):
                path.append({
                    'code': taxon_elem.find('TaxId').text,
                    'name': taxon_elem.find('ScientificName').text
                })
        
        # Add current taxon (leaf)
        current_id = taxon.find('TaxId').text
        current_name = taxon.find('ScientificName').text
        
        path.append({
            'code': current_id,
            'name': current_name
        })
        
        # Track maximum paths encountered
        original_path_count = 1  # NCBI typically returns single linear path
        if original_path_count > self.stats['max_paths_encountered']:
            self.stats['max_paths_encountered'] = original_path_count
        
        # NCBI paths are typically linear, so no truncation needed
        # But we'll track this for consistency
        if original_path_count > self.max_paths_per_code:
            self.logger.warning(f"NCBI taxid {taxid} has {original_path_count} paths, truncating to {self.max_paths_per_code}")
            self.stats['codes_truncated'] += 1
            return [path] if path else []
        
        self.logger.debug(f"NCBI taxid {taxid} - Found path with {len(path)} levels")
        return [path] if path else []
        
    def _log_progress_stats(self):
        """Log current progress statistics."""
        total_processed = self.stats['total_codes_processed']
        successful = self.stats['successful_extractions']
        success_rate = (successful / total_processed * 100) if total_processed > 0 else 0
        cache_hit_rate = (self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) * 100) if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
        
        self.logger.info(f"Progress: {total_processed} codes processed, {successful} successful ({success_rate:.1f}%), "
                         f"cache hit rate: {cache_hit_rate:.1f}%, API errors: {self.stats['api_errors']}, "
                         f"rate limit hits: {self.stats['rate_limit_hits']}")
        
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
            'api_calls_saved_by_cache': self.stats['cache_hits'],  # API calls saved by cache
            'path_statistics': path_stats,
            'cui_statistics': cui_stats,
            
            # NCBI-specific stats
            'total_codes_with_paths': self.stats['successful_extractions'],
            'total_inactive_codes': 0,  # NCBI doesn't have inactive codes like SNOMED
            'code_success_rate': (self.stats['successful_extractions'] / self.stats['total_codes_processed']) if self.stats['total_codes_processed'] > 0 else 0,
            'total_cuis_processed': len(self.cui_results),
            'cuis_with_paths': cui_stats['cuis_with_paths'],
            'cuis_without_paths': cui_stats['cuis_without_paths'],
            'cui_success_rate': cui_stats['cui_success_rate'],
            
            # API-specific stats
            'rate_limit_hits': self.stats['rate_limit_hits'],
            'timeouts': self.stats['timeouts'],
            
            # Path truncation stats
            'max_paths_encountered': self.stats['max_paths_encountered'],
            'codes_truncated': self.stats['codes_truncated']
        }
        
        return enhanced_stats


# Register the extractor
ExtractorRegistry.register('NCBI', NCBIExtractor)