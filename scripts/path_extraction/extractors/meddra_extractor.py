#!/usr/bin/env python3
"""
MedDRA extractor - optimized for speed
Optimized based on SNOMED performance improvements with aggressive caching and error handling.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base_extractor import BaseExtractor, ExtractorRegistry


class MedDRAExtractor(BaseExtractor):
    """
    Ultra-optimized MedDRA extractor with SNOMED-level performance optimizations.
    """
    
    def __init__(self, config: Dict[str, Any], vocab_name: str = 'MDR'):
        super().__init__(config, vocab_name)
        
        # MedDRA API configuration
        meddra_config = config.get('apis', {}).get('meddra', {})
        self.token = meddra_config.get('token', '')
        self.base_url = meddra_config.get('base_url', 'https://mapisbx.meddra.org/api')
        self.version = meddra_config.get('version', '27.0')
        
        if not self.token:
            raise ValueError("MedDRA API token is required but not found in config")
        
        # OPTIMIZED PERFORMANCE SETTINGS - Balanced speed and stability
        self.rate_limit_delay = 0.05  # Slightly faster - 0.05s for better speed
        self.timeout = 12  # Optimized timeout
        self.max_retries = 2  # As requested
        
        # Session with connection pooling
        self.session = requests.Session()
        
        # Optimized retry strategy - DON'T retry 400 errors (invalid codes)
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],  # Removed 400!
            allowed_methods=["HEAD", "GET", "POST", "OPTIONS"],
            backoff_factor=0.1  # Faster backoff
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,  # More connections
            pool_maxsize=50
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # API endpoint
        self.detail_url = f"{self.base_url}/detail"
        
        # ULTRA-AGGRESSIVE CACHING
        self.detail_cache = {}  # API response cache
        self.path_cache = {}  # Final path cache
        self.failed_codes_cache = set()  # Cache 400 errors IMMEDIATELY
        self.invalid_codes_cache = set()  # Cache invalid codes
        
        # Enhanced statistics tracking
        self.stats = {
            'total_codes_processed': 0,
            'successful_extractions': 0,
            'api_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_paths_extracted': 0,
            'invalid_codes': 0,
            'failed_400_codes': 0,  # Track 400 errors separately
            'path_counts': [],
            'path_lengths': []
        }
        
        # CUI-level tracking
        self.cui_results = {}
        
        # Performance settings
        self.max_path_length = 6  # MedDRA max depth: SOC->HLGT->HLT->PT->LLT
        self.max_paths_per_code = 20  # Reasonable limit
        
        # Test API connection quickly
        self._test_api_connection_fast()
        
        self.logger.info(f"Initialized MedDRA extractor (optimized) with base URL: {self.base_url}")
        self.logger.info(f"Performance config: retries={self.max_retries}, rate_limit={self.rate_limit_delay}s")
        
    def _test_api_connection_fast(self):
        """Fast API connection test - don't retry on failure."""
        try:
            test_response = self._get_detail_fast("10019211", "PT")
            if test_response.get('mds'):
                self.logger.info("✅ MedDRA API connection test successful")
            else:
                self.logger.warning("⚠️ MedDRA API connection test returned empty response")
        except Exception as e:
            self.logger.warning(f"⚠️ MedDRA API connection test failed: {e}")
        
    def validate_code(self, code: str) -> bool:
        """Validate MedDRA code format - 8-digit integers."""
        if not code or not isinstance(code, str):
            return False
        code = code.strip()
        return bool(re.match(r'^\d{8}$', code))
        
    def extract_paths(self, code: str, tty: Optional[str] = None, cui: Optional[str] = None) -> List[List[Dict[str, str]]]:
        """
        Ultra-optimized path extraction with aggressive caching and fast failure.
        """
        self.stats['total_codes_processed'] += 1
        
        # Initialize CUI tracking
        if cui and cui not in self.cui_results:
            self.cui_results[cui] = {
                'codes_processed': 0,
                'codes_with_paths': 0,
                'total_paths': 0,
                'failed_codes': []
            }
        
        if not self.validate_code(code):
            self.logger.debug(f"Invalid MedDRA code format: {code}")
            self.stats['invalid_codes'] += 1
            if cui:
                self.track_cui_code_result(cui, code, 'invalid')
                self.cui_results[cui]['failed_codes'].append(code)
                self.cui_results[cui]['codes_processed'] += 1
            return []
        
        code = code.strip()
        
        # Only skip truly invalid codes (400 errors), not rate limit failures
        # This was preventing valid LLT codes from being processed
        # if code in self.failed_codes_cache:
        #     self.logger.debug(f"Skipping known failed code: {code}")
        #     if cui:
        #         self.track_cui_code_result(cui, code, 'cached_failure')
        #         self.cui_results[cui]['codes_processed'] += 1
        #     return []
        
        # Check path cache
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
        
        # Default to PT if no tty provided
        if not tty:
            tty = "PT"
        
        # Get details with fast failure
        try:
            details = self._get_detail_fast(code, tty)
        except Exception as e:
            # Only cache 400 errors (truly invalid codes), not rate limit failures
            self.logger.debug(f"Failed to get details for MedDRA code {code}: {e}")
            self.stats['api_errors'] += 1
            if "400" in str(e):
                self.stats['failed_400_codes'] += 1
                self.failed_codes_cache.add(code)  # Only cache 400 errors
            if cui:
                self.track_cui_code_result(cui, code, 'error')
                self.cui_results[cui]['failed_codes'].append(code)
                self.cui_results[cui]['codes_processed'] += 1
            return []
        
        if not details or not details.get('mds'):
            self.failed_codes_cache.add(code)
            if cui:
                self.track_cui_code_result(cui, code, 'no_data')
                self.cui_results[cui]['codes_processed'] += 1
            return []
        
        # Extract paths
        paths = self._extract_paths_from_response(details, code)
        
        # Cache the result
        self.path_cache[code] = paths
        
        # Update statistics
        if paths:
            self.stats['successful_extractions'] += 1
            self.stats['total_paths_extracted'] += len(paths)
            self.stats['path_counts'].append(len(paths))
            for path in paths:
                self.stats['path_lengths'].append(len(path))
        
        # Update CUI tracking
        if cui:
            self.track_cui_code_result(cui, code, 'success', len(paths))
            self.cui_results[cui]['codes_processed'] += 1
            if paths:
                self.cui_results[cui]['codes_with_paths'] += 1
                self.cui_results[cui]['total_paths'] += len(paths)
        
        return paths
        
    def _get_detail_fast(self, code: str, term_type: str = "PT") -> Dict[str, Any]:
        """Ultra-fast API call with minimal retries and aggressive caching."""
        cache_key = f"{code}_{term_type}"
        
        # Check cache first
        if cache_key in self.detail_cache:
            return self.detail_cache[cache_key]
        
        # Payload structure from working MedDRA.py
        payload = {
            "bview": "SOC",
            "rsview": "release",
            "code": int(code),
            "pcode": 0,
            "syncode": 0,
            "lltcode": 0,
            "ptcode": 0,
            "hltcode": 0,
            "hlgtcode": 0,
            "soccode": 0,
            "smqcode": 0,
            "type": term_type,
            "addlangs": [],
            "rtype": "M",
            "lang": "English",
            "ver": float(self.version),
            "kana": False,
            "separator": 2
        }
        
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }
        
        # SINGLE ATTEMPT WITH FAST FAILURE
        try:
            self._rate_limit()
            response = requests.post(
                self.detail_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            # Handle 400 errors immediately - don't retry
            if response.status_code == 400:
                self.logger.debug(f"MedDRA code {code} returned 400 - invalid/discontinued code")
                self.failed_codes_cache.add(code)
                self.stats['failed_400_codes'] += 1
                raise requests.exceptions.RequestException(f"400 Client Error: Bad Request for code {code}")
            
            response.raise_for_status()
            result = response.json()
            
            # Cache successful response
            self.detail_cache[cache_key] = result
            return result
            
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"MedDRA API call failed for {code}: {e}")
            self.stats['api_errors'] += 1
            # Only cache 400 errors, not rate limit failures
            if "400" in str(e):
                self.failed_codes_cache.add(code)
            raise
        
    def _extract_paths_from_response(self, details: Dict[str, Any], code: str) -> List[List[Dict[str, str]]]:
        """Extract hierarchical paths from MedDRA API response."""
        paths = []
        
        # Extract paths from mds array
        for md in details.get('mds', []):
            path = []
            
            # Build hierarchy: SOC -> HLGT -> HLT -> PT -> LLT (ROOT → LEAF)
            if md.get('soccode') and md.get('socname'):
                path.append({
                    'code': str(md['soccode']),
                    'name': md['socname']
                })
            
            if md.get('hlgtcode') and md.get('hlgtname'):
                path.append({
                    'code': str(md['hlgtcode']),
                    'name': md['hlgtname']
                })
            
            if md.get('hltcode') and md.get('hltname'):
                path.append({
                    'code': str(md['hltcode']),
                    'name': md['hltname']
                })
            
            if md.get('ptcode') and md.get('ptname'):
                path.append({
                    'code': str(md['ptcode']),
                    'name': md['ptname']
                })
            
            # Add LLT if present and different from PT
            if md.get('lltcode') and md.get('lltname') and md['lltcode'] != md.get('ptcode'):
                path.append({
                    'code': str(md['lltcode']),
                    'name': md['lltname']
                })
            
            if path:
                paths.append(path)
        
        return paths
        
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        base_stats = self.get_processing_stats()
        
        # Calculate rates and performance metrics
        total_processed = self.stats['total_codes_processed']
        total_api_calls = self.stats['cache_misses']
        
        enhanced_stats = {
            **base_stats,
            'detailed_metrics': {
                'cache_hit_rate': self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0,
                'success_rate': self.stats['successful_extractions'] / total_processed if total_processed > 0 else 0,
                'api_calls_saved_by_cache': self.stats['cache_hits'],
                'avg_paths_per_successful_code': sum(self.stats['path_counts']) / len(self.stats['path_counts']) if self.stats['path_counts'] else 0,
                'path_length_distribution': {
                    'mean': sum(self.stats['path_lengths']) / len(self.stats['path_lengths']) if self.stats['path_lengths'] else 0,
                    'min': min(self.stats['path_lengths']) if self.stats['path_lengths'] else 0,
                    'max': max(self.stats['path_lengths']) if self.stats['path_lengths'] else 0
                },
                'error_breakdown': {
                    'total_api_errors': self.stats['api_errors'],
                    'failed_400_codes': self.stats['failed_400_codes'],
                    'invalid_format_codes': self.stats['invalid_codes'],
                    'error_rate': self.stats['api_errors'] / total_processed if total_processed > 0 else 0
                }
            },
            'performance_metrics': {
                'total_api_calls': total_api_calls,
                'failed_api_calls': self.stats['api_errors'],
                'cached_responses': self.stats['cache_hits'],
                'failed_codes_cached': len(self.failed_codes_cache)
            }
        }
        
        return enhanced_stats


# Register the optimized extractor
ExtractorRegistry.register('MDR_V2', MedDRAExtractor)
ExtractorRegistry.register('MDR', MedDRAExtractor)  # Override default