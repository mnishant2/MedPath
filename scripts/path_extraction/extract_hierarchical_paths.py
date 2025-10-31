#!/usr/bin/env python3
"""
Enhanced hierarchical path extraction orchestrator
Improved error recovery, progress tracking, and robust SNOMED CT processing.

Usage:
    python extract_hierarchical_paths.py --vocab SNOMED_CT --parallel 2
    python extract_hierarchical_paths.py --stress-test --cui-count 40
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import random

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from extractors.base_extractor import BaseExtractor, ExtractorRegistry
from extractors.snomed_extractor import SnomedExtractor
from extractors.mesh_extractor import MeshExtractor
from extractors.nci_extractor import NCIExtractor
from extractors.loinc_extractor import LOINCExtractor
from extractors.meddra_extractor import MedDRAExtractor
from extractors.lchnw_extractor import LCHNWExtractor
from utils.storage import PathStorage, ProgressTracker
from utils.statistics import PathStatistics


class EnhancedHierarchicalPathOrchestrator:
    """
    Enhanced orchestrator with improved error recovery and comprehensive monitoring.
    """
    
    def __init__(self, config_path: Optional[str] = None, output_dir: Optional[str] = None,
                 parallel_workers: int = 2, verbose: bool = False,
                 version: Optional[str] = None, mappings_path: Optional[str] = None,
                 subdir_suffix: Optional[str] = None):
        """
        Initialize the enhanced orchestrator.
        
        Args:
            config_path: Path to credentials.yaml file
            output_dir: Output directory for results
            parallel_workers: Number of parallel workers (reduced default for stability)
            verbose: Enable verbose logging
        """
        # Load configuration
        self.config = BaseExtractor.load_config(config_path)
        self.verbose = verbose
        
        # Set up output directory: default to MedPath's data_processed/hierarchical_paths
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent.parent / "data_processed" / "hierarchical_paths")

        # Apply optional suffix by appending to the last path component
        self.subdir_suffix = subdir_suffix or ""
        base_path = Path(output_dir)
        if self.subdir_suffix and not base_path.name.endswith(self.subdir_suffix):
            base_path = base_path.parent / f"{base_path.name}{self.subdir_suffix}"

        self.output_dir = base_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create orchestrator logs directory only (no top-level results/progress)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # Initialize components (avoid top-level results/progress stores)
        self.statistics = PathStatistics()
        
        # Set up enhanced logging
        self.logger = self._setup_enhanced_logging()
        
        # Optional version override (used by API-based extractors if supported)
        if version:
            # Store under a standard key; extractors can read it
            self.config.setdefault('versions', {})
            self.config['versions']['GLOBAL_VERSION_OVERRIDE'] = version

        # Load CUI mappings
        self.cui_mappings = self._load_cui_mappings(mappings_path)
        
        # Processing settings
        self.parallel_workers = parallel_workers
        self.batch_size = 100  # Larger batches for better throughput
        self.checkpoint_frequency = 25  # Save progress every N CUIs
        
        self.logger.info(f"Enhanced orchestrator initialized with {len(self.cui_mappings):,} CUIs")
        self.logger.info(f"Configuration: workers={parallel_workers}, batch_size={self.batch_size}")
        
    def _setup_enhanced_logging(self) -> logging.Logger:
        """Set up comprehensive logging with multiple levels."""
        # Main log file
        log_file = self.output_dir / 'logs' / 'orchestrator.log'
        
        # Configure root logger
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Create specialized loggers
        self.error_logger = logging.getLogger('orchestrator_errors')
        error_handler = logging.FileHandler(self.output_dir / 'logs' / 'orchestrator_errors.log')
        error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.error_logger.addHandler(error_handler)
        self.error_logger.setLevel(logging.WARNING)
        
        self.progress_logger = logging.getLogger('orchestrator_progress')
        progress_handler = logging.FileHandler(self.output_dir / 'logs' / 'orchestrator_progress.log')
        progress_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.progress_logger.addHandler(progress_handler)
        self.progress_logger.setLevel(logging.INFO)
        
        return logging.getLogger(__name__)
        
    def _vocab_aliases(self, vocab_name: str) -> List[str]:
        """Return acceptable mapping keys for a given vocab name."""
        v = vocab_name.upper()
        aliases = {v}
        if v in {"SNOMED_CT_V2", "SNOMEDCT_US", "SNOMED_CT"}:
            aliases.update({"SNOMEDCT_US", "SNOMED_CT", "SNOMED_CT_V2"})
        elif v in {"MSH", "MESH"}:
            aliases.update({"MSH", "MESH"})
        elif v in {"LNC", "LOINC", "LNC_V2"}:
            aliases.update({"LNC", "LOINC", "LNC_V2"})
        elif v in {"MDR", "MEDDRA"}:
            aliases.update({"MDR", "MEDDRA"})
        elif v in {"ICD", "ICD9CM", "ICD10CM"}:
            aliases.update({"ICD", "ICD9CM", "ICD10CM"})
        elif v in {"LCH_NW", "LCHNW"}:
            aliases.update({"LCH_NW", "LCHNW"})
        else:
            aliases.add(v)
        return list(aliases)

    def _get_vocab_data_for_cui(self, cui: str, vocab_name: str) -> Tuple[List[str], Dict[str, Optional[str]]]:
        """Return (codes, code_to_tty) for given CUI and vocabulary, handling multiple mapping formats and aliases."""
        mappings = self.cui_mappings.get(cui, {})
        if not isinstance(mappings, dict):
            return [], {}
        codes: List[str] = []
        code_to_tty: Dict[str, Optional[str]] = {}
        # Try aliases
        for key in self._vocab_aliases(vocab_name):
            if key in mappings:
                vocab_data = mappings[key]
                # Format A: { 'codes': [...], 'code_meta' or 'meta': {code: {'tty': ...}} }
                if isinstance(vocab_data, dict) and 'codes' in vocab_data and isinstance(vocab_data['codes'], list):
                    codes = list(vocab_data['codes'])
                    meta_map = vocab_data.get('code_meta') or vocab_data.get('meta') or {}
                    if isinstance(meta_map, dict):
                        for c, meta in meta_map.items():
                            if isinstance(meta, dict) and 'tty' in meta:
                                code_to_tty[c] = meta.get('tty')
                    break
                # Format B: dict-of-codes -> meta (including tty)
                if isinstance(vocab_data, dict):
                    # If the dict looks like {code: {...}} use keys as codes
                    # Avoid treating known keys like 'codes', 'meta' as codes
                    candidate_codes = [k for k in vocab_data.keys() if k not in {'codes', 'meta', 'code_meta'}]
                    if candidate_codes:
                        codes = candidate_codes
                        for c in candidate_codes:
                            meta = vocab_data.get(c)
                            if isinstance(meta, dict) and 'tty' in meta:
                                code_to_tty[c] = meta.get('tty')
                        break
                # Format C: list of codes directly
                if isinstance(vocab_data, list):
                    codes = list(vocab_data)
                    break
        return codes, code_to_tty

    def _load_cui_mappings(self, mappings_path: Optional[str] = None) -> Dict[str, Any]:
        """Load CUI to vocabulary code mappings with error recovery."""
        possible_paths = []
        if mappings_path:
            possible_paths.append(Path(mappings_path))
        # Relative to repo structure
        root = Path(__file__).parent.parent.parent / "data_processed"
        # Try suffixed mappings directory first if suffix provided
        if getattr(self, 'subdir_suffix', None):
            possible_paths.append(root / f"mappings{self.subdir_suffix}" / "combined_cui_to_vocab_codes_with_tty.json")
        possible_paths.append(root / "mappings" / "combined_cui_to_vocab_codes_with_tty.json")
        # CWD fallback
        if getattr(self, 'subdir_suffix', None):
            possible_paths.append(Path(f"data_processed/mappings{self.subdir_suffix}/combined_cui_to_vocab_codes_with_tty.json"))
        possible_paths.append(Path("data_processed/mappings/combined_cui_to_vocab_codes_with_tty.json"))
        
        for mappings_file in possible_paths:
            try:
                if mappings_file.exists():
                    self.logger.info(f"Loading CUI mappings from {mappings_file}")
                    with open(mappings_file, 'r') as f:
                        return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load from {mappings_file}: {e}")
                continue
                
        raise FileNotFoundError("Could not find CUI mappings file in any expected location")
        
    def extract_vocabulary_paths_robust(self, vocab_name: str, cui_list: Optional[List[str]] = None,
                                      resume: bool = False) -> Dict[str, Any]:
        """
        Extract paths for a vocabulary with robust error handling and progress tracking.
        
        Args:
            vocab_name: Name of the vocabulary to process
            cui_list: Specific list of CUIs to process (for testing)
            resume: Whether to resume from previous progress
            
        Returns:
            Dictionary with extraction results and statistics
        """
        self.logger.info(f"Starting robust path extraction for {vocab_name}")
        
        # Create vocab-specific directory structure like other vocabs
        vocab_dir_name = vocab_name.lower().replace('_', '').replace('v2', '')
        if vocab_dir_name == 'mdr':
            vocab_dir_name = 'meddra'
        elif vocab_dir_name == 'snomedctus':
            vocab_dir_name = 'snomed'
        elif vocab_dir_name == 'lnc':
            vocab_dir_name = 'loinc'
        elif vocab_dir_name == 'msh':
            vocab_dir_name = 'mesh'
        elif vocab_dir_name == 'nci':
            vocab_dir_name = 'nci'
        
        vocab_output_dir = self.output_dir / vocab_dir_name
        vocab_output_dir.mkdir(exist_ok=True)
        (vocab_output_dir / 'results').mkdir(exist_ok=True)
        (vocab_output_dir / 'progress').mkdir(exist_ok=True)
        (vocab_output_dir / 'logs').mkdir(exist_ok=True)
        
        # Initialize extractor
        try:
            # Provide per-vocabulary logs dir to extractors
            import copy
            extractor_config = copy.deepcopy(self.config)
            extractor_config.setdefault('runtime', {})['logs_dir'] = str(vocab_output_dir / 'logs')
            extractor = ExtractorRegistry.get_extractor(vocab_name, extractor_config)
        except ValueError as e:
            error_msg = f"No extractor available for {vocab_name}: {e}"
            self.error_logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        # Determine CUIs to process
        if cui_list:
            target_cuis = cui_list
            self.logger.info(f"Processing specific CUI list: {len(target_cuis)} CUIs")
        else:
            target_cuis = self._filter_cuis_for_vocabulary(vocab_name)
            self.logger.info(f"Processing all CUIs for {vocab_name}: {len(target_cuis)} CUIs")
        
        # Handle resume functionality
        processed_cuis = set()
        if resume:
            processed_cuis = self._load_processed_cuis(vocab_name, vocab_output_dir)
            target_cuis = [cui for cui in target_cuis if cui not in processed_cuis]
            self.logger.info(f"Resuming: {len(processed_cuis)} already processed, {len(target_cuis)} remaining")
        
        # Initialize tracking
        results = {}
        start_time = time.time()
        last_checkpoint = time.time()
        error_count = 0
        max_consecutive_errors = 20  # Stop if too many consecutive errors
        consecutive_errors = 0
        
        # Process CUIs in batches
        total_cuis = len(target_cuis)
        batches = [target_cuis[i:i + self.batch_size] for i in range(0, len(target_cuis), self.batch_size)]
        
        self.logger.info(f"Processing {total_cuis} CUIs in {len(batches)} batches of {self.batch_size}")
        
        for batch_idx, batch_cuis in enumerate(batches):
            batch_start_time = time.time()
            batch_results = {}
            batch_errors = 0
            
            self.logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_cuis)} CUIs)")
            
            for cui_idx, cui in enumerate(batch_cuis):
                try:
                    # Get vocabulary codes for this CUI
                    codes, code_to_tty = self._get_vocab_data_for_cui(cui, vocab_name)
                    if not codes:
                        continue
                    
                    # Extract paths for each code - PROPER FORMAT WITH CODES
                    cui_codes = {}
                    total_cui_paths = 0
                    
                    for code in codes:
                        try:
                            tty = code_to_tty.get(code)
                            paths = extractor.extract_paths(code, tty=tty, cui=cui)
                            cui_codes[code] = {
                                'paths': paths,
                                'path_count': len(paths)
                            }
                            total_cui_paths += len(paths)
                        except Exception as e:
                            self.error_logger.warning(f"Error extracting paths for CUI {cui}, code {code}: {str(e)}")
                            cui_codes[code] = {
                                'paths': [],
                                'path_count': 0,
                                'error': str(e)
                            }
                            error_count += 1
                            batch_errors += 1
                            consecutive_errors += 1
                            
                            if consecutive_errors >= max_consecutive_errors:
                                error_msg = f"Too many consecutive errors ({consecutive_errors}). Stopping extraction."
                                self.error_logger.error(error_msg)
                                return {'success': False, 'error': error_msg}
                    
                    # Store results with proper code structure
                    batch_results[cui] = {
                        'vocabulary': vocab_name,
                        'codes': cui_codes,
                        'total_paths': total_cui_paths
                    }
                    
                    if total_cui_paths > 0:
                        consecutive_errors = 0  # Reset counter on success
                    
                    # Progress logging
                    if (cui_idx + 1) % 10 == 0 or cui_idx == len(batch_cuis) - 1:
                        progress = ((batch_idx * self.batch_size + cui_idx + 1) / total_cuis) * 100
                        elapsed = time.time() - start_time
                        rate = (batch_idx * self.batch_size + cui_idx + 1) / elapsed if elapsed > 0 else 0
                        eta = (total_cuis - (batch_idx * self.batch_size + cui_idx + 1)) / rate if rate > 0 else 0
                        
                        self.progress_logger.info(f"Progress: {progress:.1f}% ({cui_idx + 1}/{len(batch_cuis)} in batch), "
                                                f"Rate: {rate:.1f} CUIs/sec, ETA: {eta/60:.1f}min, Errors: {error_count}")
                        
                except KeyboardInterrupt:
                    self.logger.warning("Extraction interrupted by user")
                    break
                except Exception as e:
                    self.error_logger.error(f"Unexpected error processing CUI {cui}: {str(e)}")
                    self.error_logger.error(f"Traceback: {traceback.format_exc()}")
                    error_count += 1
                    consecutive_errors += 1
            
            # Add batch results to main results
            results.update(batch_results)
            
            # Checkpoint: Save progress periodically
            current_time = time.time()
            if current_time - last_checkpoint > 300:  # Every 5 minutes
                self._save_checkpoint(vocab_name, results, batch_idx, len(batches), vocab_output_dir)
                last_checkpoint = current_time
            
            # Batch summary
            batch_time = time.time() - batch_start_time
            self.logger.info(f"Batch {batch_idx + 1} completed: {len(batch_results)} CUIs with paths, "
                           f"{batch_errors} errors, {batch_time:.1f}s")
        
        # Final processing
        processing_time = time.time() - start_time
        
        # Save final results
        if results:
            self._save_results(vocab_name, results, vocab_output_dir)
            
        # Generate statistics
        stats = self._calculate_comprehensive_stats(vocab_name, results, processing_time, error_count, extractor, vocab_output_dir)
        
        # Save final checkpoint
        self._save_checkpoint(vocab_name, results, len(batches), len(batches), vocab_output_dir, final=True)
        
        self.logger.info(f"Extraction completed: {len(results)} CUIs processed, "
                        f"{sum(r['total_paths'] for r in results.values())} total paths, "
                        f"{error_count} errors, {processing_time:.1f}s")
        
        return {
            'success': True,
            'vocabulary': vocab_name,
            'cuis_processed': len(results),
            'total_paths': sum(r['total_paths'] for r in results.values()),
            'processing_time': processing_time,
            'error_count': error_count,
            'statistics': stats
        }
        
    def _filter_cuis_for_vocabulary(self, vocab_name: str) -> List[str]:
        """Filter CUIs that have at least one code for the specified vocabulary, across formats and aliases."""
        vocab_cuis: List[str] = []
        aliases = set(self._vocab_aliases(vocab_name))
        for cui, mappings in self.cui_mappings.items():
            if not isinstance(mappings, dict):
                continue
            # Find any alias present
            present_key = None
            for k in mappings.keys():
                if k.upper() in aliases:
                    present_key = k
                    break
            if not present_key:
                continue
            data = mappings[present_key]
            has_codes = False
            if isinstance(data, dict):
                if 'codes' in data and isinstance(data['codes'], list) and len(data['codes']) > 0:
                    has_codes = True
                else:
                    # dict-of-codes format
                    candidate_codes = [k for k in data.keys() if k not in {'codes', 'meta', 'code_meta'}]
                    if len(candidate_codes) > 0:
                        has_codes = True
            elif isinstance(data, list) and len(data) > 0:
                has_codes = True
            if has_codes:
                vocab_cuis.append(cui)
        return vocab_cuis
        
    def _load_processed_cuis(self, vocab_name: str, vocab_output_dir: Path) -> set:
        """Load previously processed CUIs from checkpoint."""
        checkpoint_file = vocab_output_dir / 'progress' / f'{vocab_name}_checkpoint.json'
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                return set(checkpoint.get('processed_cuis', []))
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
        return set()
        
    def _save_checkpoint(self, vocab_name: str, results: Dict, batch_idx: int, total_batches: int, vocab_output_dir: Path, final: bool = False):
        """Save processing checkpoint."""
        checkpoint_file = vocab_output_dir / 'progress' / f'{vocab_name}_checkpoint.json'
        
        checkpoint_data = {
            'timestamp': time.time(),
            'vocabulary': vocab_name,
            'batch_progress': f"{batch_idx}/{total_batches}",
            'processed_cuis': list(results.keys()),
            'total_cuis_processed': len(results),
            'total_paths_extracted': sum(r['total_paths'] for r in results.values()),
            'is_final': final
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
        self.logger.debug(f"Checkpoint saved: {len(results)} CUIs processed")
        
    def _save_results(self, vocab_name: str, results: Dict, vocab_output_dir: Path):
        """Save extraction results."""
        results_file = vocab_output_dir / 'results' / f'{vocab_name}_paths.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Results saved to {results_file}")
        
    def _calculate_comprehensive_stats(self, vocab_name: str, results: Dict, processing_time: float, 
                                     error_count: int, extractor, vocab_output_dir: Path) -> Dict[str, Any]:
        """Calculate comprehensive processing statistics."""
        total_cuis = len(results)
        total_paths = sum(r['total_paths'] for r in results.values())
        avg_paths_per_cui = total_paths / total_cuis if total_cuis > 0 else 0
        
        # Get comprehensive extractor-specific stats (like SNOMED)
        extractor_stats = {}
        if hasattr(extractor, 'get_comprehensive_statistics'):
            extractor_stats = extractor.get_comprehensive_statistics()
        elif hasattr(extractor, 'get_processing_stats'):
            extractor_stats = extractor.get_processing_stats()
        
        stats = {
            'vocabulary': vocab_name,
            'processing_summary': {
                'total_cuis_processed': total_cuis,
                'total_paths_extracted': total_paths,
                'avg_paths_per_cui': avg_paths_per_cui,
                'processing_time_seconds': processing_time,
                'processing_time_minutes': processing_time / 60,
                'cuis_per_second': total_cuis / processing_time if processing_time > 0 else 0,
                'total_errors': error_count
            },
            'extractor_stats': extractor_stats,
            'timestamp': time.time()
        }
        
        # Save stats to file
        stats_file = vocab_output_dir / 'results' / f'{vocab_name}_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        return stats
        
    def stress_test_snomed(self, cui_count: int = 40) -> Dict[str, Any]:
        """
        Run stress test with random CUIs to validate the SNOMED extractor.
        
        Args:
            cui_count: Number of random CUIs to test
            
        Returns:
            Test results and validation report
        """
        self.logger.info(f"Starting SNOMED stress test with {cui_count} random CUIs")
        
        # Get SNOMED CUIs
        snomed_cuis = self._filter_cuis_for_vocabulary('SNOMEDCT_US')  # Primary SNOMED vocabulary
        
        if len(snomed_cuis) < cui_count:
            cui_count = len(snomed_cuis)
            self.logger.warning(f"Only {len(snomed_cuis)} SNOMED CUIs available, using all")
        
        # Randomly select CUIs for testing
        test_cuis = random.sample(snomed_cuis, cui_count)
        
        self.logger.info(f"Selected {len(test_cuis)} CUIs for stress testing: {test_cuis[:10]}{'...' if len(test_cuis) > 10 else ''}")
        
        # Run extraction
        results = self.extract_vocabulary_paths_robust('SNOMED_CT', cui_list=test_cuis)
        
        # Validation checks
        validation_report = self._validate_stress_test_results(results, test_cuis)
        
        # Save stress test report
        stress_test_file = self.output_dir / 'snomed' / 'results' / 'snomed_stress_test_report.json'
        stress_test_file.parent.mkdir(parents=True, exist_ok=True)
        
        stress_test_report = {
            'test_configuration': {
                'cui_count_requested': cui_count,
                'cui_count_tested': len(test_cuis),
                'test_cuis': test_cuis,
                'vocabulary': 'SNOMED_CT'
            },
            'extraction_results': results,
            'validation_report': validation_report
        }
        
        with open(stress_test_file, 'w') as f:
            json.dump(stress_test_report, f, indent=2)
        
        self.logger.info(f"Stress test completed. Report saved to {stress_test_file}")
        
        return stress_test_report
        
    def _validate_stress_test_results(self, results: Dict[str, Any], test_cuis: List[str]) -> Dict[str, Any]:
        """Validate stress test results and generate report."""
        validation = {
            'test_success': results.get('success', False),
            'cuis_with_results': 0,
            'total_paths_found': 0,
            'output_format_valid': True,
            'inactive_codes_handled': True,
            'error_handling_effective': True,
            'performance_metrics': {},
            'sample_outputs': [],
            'issues_found': []
        }
        
        if not results.get('success'):
            validation['issues_found'].append(f"Extraction failed: {results.get('error', 'Unknown error')}")
            return validation
        
        # Check results structure
        extraction_results = results.get('statistics', {}).get('extractor_stats', {})
        
        # Count CUIs with results
        if 'cuis_processed' in results:
            validation['cuis_with_results'] = results['cuis_processed']
        
        # Check total paths
        if 'total_paths' in results:
            validation['total_paths_found'] = results['total_paths']
        
        # Performance metrics
        if 'processing_time' in results:
            processing_time = results['processing_time']
            validation['performance_metrics'] = {
                'processing_time_seconds': processing_time,
                'processing_time_minutes': processing_time / 60,
                'cuis_per_second': len(test_cuis) / processing_time if processing_time > 0 else 0
            }
        
        # Check extractor-specific stats
        if extraction_results:
            # Check inactive code handling
            inactive_codes = extraction_results.get('inactive_codes_skipped', 0)
            validation['inactive_codes_handled'] = inactive_codes >= 0
            
            # Check error handling
            api_errors = extraction_results.get('api_errors', 0)
            total_processed = extraction_results.get('total_codes_processed', 0)
            if total_processed > 0:
                error_rate = api_errors / total_processed
                validation['error_handling_effective'] = error_rate < 0.1  # Less than 10% error rate
            
            # Add detailed stats to report
            validation['detailed_stats'] = {
                'inactive_codes_skipped': inactive_codes,
                'api_errors': api_errors,
                'cache_hit_rate': extraction_results.get('cache_hit_rate', 0),
                'success_rate': extraction_results.get('success_rate', 0)
            }
        
        # Validate output format by checking for proper structure
        # This would require loading the actual results file, which we'll do if it exists
        results_file = self.output_dir / 'snomed' / 'results' / 'SNOMED_CT_paths.json'
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    paths_data = json.load(f)
                    
                # Check format of first few results
                sample_count = min(3, len(paths_data))
                for cui in list(paths_data.keys())[:sample_count]:
                    cui_data = paths_data[cui]
                    if 'paths' in cui_data and isinstance(cui_data['paths'], list):
                        # Check path structure
                        if cui_data['paths']:
                            first_path = cui_data['paths'][0]
                            if isinstance(first_path, list) and first_path:
                                first_step = first_path[0]
                                if isinstance(first_step, dict) and 'code' in first_step and 'name' in first_step:
                                    validation['sample_outputs'].append({
                                        'cui': cui,
                                        'path_count': len(cui_data['paths']),
                                        'sample_path_length': len(first_path),
                                        'sample_step': first_step
                                    })
                                else:
                                    validation['output_format_valid'] = False
                                    validation['issues_found'].append(f"Invalid path step format for CUI {cui}")
                            else:
                                validation['issues_found'].append(f"Empty or invalid path format for CUI {cui}")
                    else:
                        validation['output_format_valid'] = False
                        validation['issues_found'].append(f"Missing or invalid paths structure for CUI {cui}")
                        
            except Exception as e:
                validation['output_format_valid'] = False
                validation['issues_found'].append(f"Error reading results file: {str(e)}")
        
        # Overall validation
        validation['overall_success'] = (
            validation['test_success'] and
            validation['output_format_valid'] and
            validation['error_handling_effective'] and
            validation['cuis_with_results'] > 0
        )
        
        return validation


def main():
    """Main entry point for the enhanced orchestrator."""
    parser = argparse.ArgumentParser(
        description="Enhanced hierarchical path extraction with robust error handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_hierarchical_paths.py --vocab SNOMED_CT --parallel 2
  python extract_hierarchical_paths.py --stress-test --cui-count 40
  python extract_hierarchical_paths.py --vocab SNOMED_CT --resume --verbose
        """
    )
    
    # Single vocab and multi-vocab (underscore style + backwards-compatible)
    parser.add_argument('--vocab', type=str, help='Vocabulary to process (e.g., SNOMED_CT)')
    parser.add_argument('--vocabs', nargs='+', help='Space/comma-separated list of vocabularies to process')
    parser.add_argument('--parallel', type=int, default=2, help='Number of parallel workers (default: 2)')
    # Output directory and suffix (underscore style + legacy)
    parser.add_argument('--output', dest='output_dir', type=str, help='Output root directory (default: MedPath data_processed/hierarchical_paths)')
    parser.add_argument('--output-dir', dest='output_dir', type=str, help='[Deprecated] Same as --output')
    parser.add_argument('--subdir_suffix', type=str, default=None, help='Optional suffix to append to output dir name (e.g., _sample)')
    parser.add_argument('--config', type=str, help='Path to credentials.yaml file')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--stress-test', action='store_true', help='Run SNOMED stress test')
    parser.add_argument('--cui-count', type=int, default=40, help='Number of CUIs for stress test (default: 40)')
    parser.add_argument('--version', type=str, help='Vocabulary API/version identifier (e.g., SNOMED, MeSH release)')
    parser.add_argument('--mappings_path', dest='mappings_path', type=str, help='Path to combined_cui_to_vocab_codes_with_tty.json')
    parser.add_argument('--mappings-path', dest='mappings_path', type=str, help='[Deprecated] Same as --mappings_path')
    
    args = parser.parse_args()
    
    try:
        # Initialize orchestrator
        orchestrator = EnhancedHierarchicalPathOrchestrator(
            config_path=args.config,
            output_dir=args.output_dir,
            parallel_workers=args.parallel,
            verbose=args.verbose,
            version=args.version,
            mappings_path=args.mappings_path,
            subdir_suffix=args.subdir_suffix
        )
        
        # Handle stress test
        if args.stress_test:
            print(f"🧪 Running SNOMED stress test with {args.cui_count} CUIs...")
            result = orchestrator.stress_test_snomed(args.cui_count)
            
            if result['validation_report']['overall_success']:
                print(f"✅ Stress test PASSED!")
                print(f"   - {result['validation_report']['cuis_with_results']} CUIs processed")
                print(f"   - {result['validation_report']['total_paths_found']} paths extracted")
                print(f"   - Processing time: {result['validation_report']['performance_metrics'].get('processing_time_minutes', 0):.1f} minutes")
            else:
                print(f"❌ Stress test FAILED!")
                for issue in result['validation_report']['issues_found']:
                    print(f"   - {issue}")
                sys.exit(1)
            return
        
        # Handle regular vocabulary processing
        target_vocabs: List[str] = []
        if args.vocabs:
            # Support comma-separated values inside a single token too
            expanded = []
            for item in args.vocabs:
                expanded.extend([v for v in (s.strip() for s in item.split(',')) if v])
            target_vocabs = expanded
        elif args.vocab:
            target_vocabs = [args.vocab]
        else:
            print("Error: --vocabs or --vocab is required for regular processing (or use --stress-test)")
            parser.print_help()
            sys.exit(1)

        last_result = None
        for vocab in target_vocabs:
            print(f"🚀 Starting path extraction for {vocab}...")
            result = orchestrator.extract_vocabulary_paths_robust(vocab, resume=args.resume)
            last_result = result
        
        # Print final status
        if last_result and last_result.get('success', False):
            print(f"✅ Path extraction completed successfully for {len(target_vocabs)} vocab(s)")
            print(f"   - Last vocabulary: {last_result['vocabulary']}")
            print(f"   - CUIs processed: {last_result['cuis_processed']:,}")
            print(f"   - Total paths: {last_result['total_paths']:,}")
            print(f"   - Processing time: {last_result['processing_time']/60:.1f} minutes")
            if last_result['error_count'] > 0:
                print(f"   - Errors encountered: {last_result['error_count']}")
        else:
            print(f"❌ Path extraction failed for one or more vocabularies")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        if args.verbose:
            print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()