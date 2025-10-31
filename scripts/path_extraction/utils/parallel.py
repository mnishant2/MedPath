"""
Parallel processing utilities for hierarchical path extraction.
"""

import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Callable, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from tqdm import tqdm
from .checkpoint import CheckpointManager, ExtractorCheckpoint


@dataclass
class ProcessingTask:
    """Represents a single processing task."""
    cui: str
    vocab_name: str
    codes: List[Tuple[str, str]]  # List of (code, tty) tuples
    
    
@dataclass
class ProcessingResult:
    """Represents the result of processing a task."""
    cui: str
    vocab_name: str
    success: bool
    paths_data: Dict[str, Any]
    errors: List[str]
    processing_time: float


class ProgressReporter:
    """Thread-safe progress reporter."""
    
    def __init__(self, total_tasks: int, vocab_name: str):
        self.total_tasks = total_tasks
        self.vocab_name = vocab_name
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # For ETA calculation
        self.last_update_time = self.start_time
        self.last_completed = 0
        
    def update(self, success: bool = True):
        """Update progress counters."""
        with self.lock:
            self.completed_tasks += 1
            if not success:
                self.failed_tasks += 1
                
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information."""
        with self.lock:
            elapsed = time.time() - self.start_time
            completed = self.completed_tasks
            
            # Calculate rate and ETA
            if elapsed > 0:
                rate = completed / elapsed
                eta_seconds = (self.total_tasks - completed) / rate if rate > 0 else 0
            else:
                rate = 0
                eta_seconds = 0
                
            return {
                'total': self.total_tasks,
                'completed': completed,
                'failed': self.failed_tasks,
                'remaining': self.total_tasks - completed,
                'percentage': (completed / self.total_tasks) * 100 if self.total_tasks > 0 else 0,
                'elapsed_time': elapsed,
                'rate': rate,
                'eta_seconds': eta_seconds,
                'eta_formatted': str(timedelta(seconds=int(eta_seconds))),
                'vocab_name': self.vocab_name
            }
            
    def log_progress(self, force: bool = False):
        """Log current progress if enough time has passed."""
        current_time = time.time()
        
        # Log every 30 seconds or when forced
        if force or (current_time - self.last_update_time) >= 30:
            with self.lock:
                progress = self.get_progress_info()
                
                self.logger.info(
                    f"{self.vocab_name}: {progress['completed']}/{progress['total']} "
                    f"({progress['percentage']:.1f}%) - "
                    f"Rate: {progress['rate']:.1f} CUIs/sec - "
                    f"ETA: {progress['eta_formatted']} - "
                    f"Failed: {progress['failed']}"
                )
                
                self.last_update_time = current_time


class ParallelExtractor:
    """Parallel processor for hierarchical path extraction."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
    def process_vocabulary(self, 
                          vocab_name: str,
                          cui_mappings: Dict[str, Any],
                          extractor,
                          resume_cuis: Optional[set] = None,
                          use_checkpointing: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process all CUIs for a vocabulary in parallel with checkpointing support.
        
        Args:
            vocab_name: Name of the vocabulary
            cui_mappings: CUI mappings from combined file
            extractor: Extractor instance for this vocabulary
            resume_cuis: Set of CUIs to skip (for resume functionality)
            use_checkpointing: Enable checkpoint/resume functionality
            
        Returns:
            Tuple of (results_dict, statistics_dict)
        """
        # Initialize checkpoint managers
        checkpoint_mgr = None
        extractor_checkpoint = None
        
        if use_checkpointing:
            checkpoint_mgr = CheckpointManager(vocab_name)
            extractor_checkpoint = ExtractorCheckpoint(extractor, vocab_name)
            
            # Get already completed CUIs from checkpoint
            completed_cuis = checkpoint_mgr.get_completed_cuis()
            if completed_cuis:
                self.logger.info(f"📁 Resume mode: Found {len(completed_cuis)} already processed CUIs")
                if resume_cuis:
                    resume_cuis.update(completed_cuis)
                else:
                    resume_cuis = completed_cuis
        
        vocab_key = self._get_vocab_key(vocab_name)
        
        # Create tasks
        tasks = self._create_tasks(vocab_name, cui_mappings, vocab_key, resume_cuis or set())
        
        if not tasks:
            self.logger.warning(f"No tasks to process for {vocab_name}")
            if checkpoint_mgr:
                return checkpoint_mgr.get_results(), checkpoint_mgr.get_stats()
            return {}, {}
            
        self.logger.info(f"Starting parallel processing of {len(tasks)} CUIs for {vocab_name} with {self.max_workers} workers")
        if checkpoint_mgr and completed_cuis:
            self.logger.info(f"📁 Checkpoint info: {checkpoint_mgr.get_resume_info()['message']}")
        
        # Initialize progress reporter
        progress = ProgressReporter(len(tasks), vocab_name)
        
        # Process tasks
        results = {}
        api_calls = 0
        api_errors = 0
        start_time = time.time()
        
        # Add overall progress bar with detailed stats
        pbar = tqdm(total=len(tasks), desc=f"Processing {vocab_name} CUIs", unit="CUI")
        
        def process_task(task: ProcessingTask) -> ProcessingResult:
            """Process a single task with optional early stopping optimization."""
            nonlocal api_calls, api_errors
            
            try:
                task_start = time.time()
                
                # Check optimization settings
                optimization_config = extractor.config.get('optimization', {})
                enable_early_stopping = optimization_config.get('enable_early_stopping', False)
                early_stopping_vocabs = optimization_config.get('early_stopping_vocabs', [])
                collect_all_paths = optimization_config.get('collect_all_paths', False)
                
                # Determine if early stopping applies to this vocabulary
                use_early_stopping = (enable_early_stopping and 
                                    vocab_name in early_stopping_vocabs and 
                                    not collect_all_paths)
                
                # Extract paths for codes in this CUI
                all_paths = []
                source_codes = []
                errors = []
                codes_processed = 0
                codes_skipped = 0
                
                for code, tty in task.codes:
                    codes_processed += 1
                    api_calls += 1
                    try:
                        # Pass CUI to extractor for statistics tracking
                        if hasattr(extractor, 'extract_paths'):
                            paths = extractor.extract_paths(code, tty, cui=task.cui)
                        else:
                            paths, path_errors = extractor.extract_with_retry(code, tty)
                            if path_errors:
                                errors.extend(path_errors)
                                api_errors += len(path_errors)
                        
                        if paths:
                            all_paths.extend(paths)
                            source_codes.append(code)
                            
                            # Early stopping: if we got paths and early stopping is enabled, stop here
                            if use_early_stopping:
                                codes_skipped = len(task.codes) - codes_processed
                                if codes_skipped > 0:
                                    self.logger.debug(f"Early stopping for CUI {task.cui}: got paths from {code}, skipping {codes_skipped} remaining codes")
                                break
                                
                    except Exception as e:
                        error_msg = f"Error processing code {code}: {str(e)}"
                        errors.append(error_msg)
                        api_errors += 1
                        self.logger.error(error_msg)
                
                # Create result data with optimization info
                paths_data = {
                    'paths': all_paths,
                    'source_codes': source_codes,
                    'extraction_date': datetime.now().isoformat(),
                    'errors': errors,
                    'optimization': {
                        'early_stopping_used': use_early_stopping,
                        'codes_processed': codes_processed,
                        'codes_skipped': codes_skipped,
                        'total_codes': len(task.codes)
                    }
                }
                
                processing_time = time.time() - task_start
                success = len(all_paths) > 0
                
                return ProcessingResult(
                    cui=task.cui,
                    vocab_name=task.vocab_name,
                    success=success,
                    paths_data=paths_data,
                    errors=errors,
                    processing_time=processing_time
                )
                
            except Exception as e:
                processing_time = time.time() - task_start
                error_msg = f"Fatal error processing CUI {task.cui}: {str(e)}"
                self.logger.error(error_msg)
                
                return ProcessingResult(
                    cui=task.cui,
                    vocab_name=task.vocab_name,
                    success=False,
                    paths_data={'paths': [], 'source_codes': [], 'errors': [error_msg]},
                    errors=[error_msg],
                    processing_time=processing_time
                )
        
        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(process_task, task): task for task in tasks}
            
            # Process results as they complete
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results[result.cui] = result.paths_data
                    progress.update(result.success)
                    
                    # Save to checkpoint if enabled
                    if checkpoint_mgr:
                        checkpoint_mgr.add_result(result.cui, result.paths_data)
                    
                    # Update progress bar
                    success_count = len([r for r in results.values() if r.get('paths')])
                    failed_count = len([r for r in results.values() if not r.get('paths')])
                    pbar.update(1)
                    pbar.set_postfix(Success=success_count, Failed=failed_count)
                    
                    # Log progress periodically
                    progress.log_progress()
                    
                except Exception as e:
                    task = future_to_task[future]
                    self.logger.error(f"Exception processing task for CUI {task.cui}: {str(e)}")
                    progress.update(False)
                    pbar.update(1)
        
        # Close progress bar
        pbar.close()
        
        # Final progress log
        progress.log_progress(force=True)
        
        total_time = time.time() - start_time
        
        # Get extractor-specific statistics
        extractor_stats = {}
        if hasattr(extractor, 'get_processing_stats'):
            extractor_stats = extractor.get_processing_stats()
        
        # Calculate optimization statistics
        optimization_stats = {
            'early_stopping_enabled': False,
            'total_codes_processed': 0,
            'total_codes_skipped': 0,
            'total_possible_codes': 0,
            'efficiency_gain_percent': 0,
            'time_saved_estimate_minutes': 0
        }
        
        for result_data in results.values():
            opt_info = result_data.get('optimization', {})
            if opt_info.get('early_stopping_used', False):
                optimization_stats['early_stopping_enabled'] = True
            optimization_stats['total_codes_processed'] += opt_info.get('codes_processed', 0)
            optimization_stats['total_codes_skipped'] += opt_info.get('codes_skipped', 0)
            optimization_stats['total_possible_codes'] += opt_info.get('total_codes', 0)
        
        # Calculate efficiency gain
        if optimization_stats['total_possible_codes'] > 0:
            optimization_stats['efficiency_gain_percent'] = (
                optimization_stats['total_codes_skipped'] / 
                optimization_stats['total_possible_codes']
            ) * 100
            # Estimate time saved (assuming 0.5s per skipped code)
            optimization_stats['time_saved_estimate_minutes'] = (
                optimization_stats['total_codes_skipped'] * 0.5
            ) / 60

        # Create statistics
        statistics = {
            'vocabulary': vocab_name,
            'processing_stats': {
                'total_processing_time': total_time,
                'total_tasks': len(tasks),
                'api_calls_made': api_calls,
                'api_errors': api_errors,
                'success_rate': len([r for r in results.values() if r.get('paths')]) / len(results) if results else 0
            },
            'optimization_stats': optimization_stats,
            'extractor_stats': extractor_stats
        }
        
        # Final checkpoint saves
        if checkpoint_mgr:
            checkpoint_mgr.update_stats(statistics)
            checkpoint_mgr.save_checkpoint(force=True)
            self.logger.info(f"💾 Final checkpoint saved with {len(results)} results")
        
        if extractor_checkpoint:
            extractor_checkpoint.save_extractor_state()
            self.logger.info(f"💾 Extractor state saved to checkpoint")
        
        self.logger.info(f"Completed processing {vocab_name}: {len(results)} CUIs in {total_time:.1f}s")
        
        return results, statistics
        
    def _create_tasks(self, vocab_name: str, cui_mappings: Dict, vocab_key: str, resume_cuis: set) -> List[ProcessingTask]:
        """Create processing tasks from CUI mappings."""
        tasks = []
        
        for cui, cui_data in cui_mappings.items():
            # Skip if resuming and already processed
            if cui in resume_cuis:
                continue
                
            # Check if this CUI has mappings for this vocabulary
            if vocab_key not in cui_data:
                continue
                
            # Get vocab data for this CUI
            vocab_data = cui_data[vocab_key]
            
            # Extract codes and ttys from the structure
            vocab_codes = vocab_data.get('codes', [])
            vocab_ttys = vocab_data.get('ttys', [])
            
            # Create list of (code, tty) tuples
            codes = []
            for i, code in enumerate(vocab_codes):
                # Get corresponding tty if available
                tty = vocab_ttys[i] if i < len(vocab_ttys) else ''
                codes.append((code, tty))
                
            if codes:
                tasks.append(ProcessingTask(
                    cui=cui,
                    vocab_name=vocab_name,
                    codes=codes
                ))
                
        return tasks
        
    def _get_vocab_key(self, vocab_name: str) -> str:
        """Map vocabulary names to keys used in CUI mappings."""
        vocab_mapping = {
            'SNOMED_CT': 'SNOMEDCT_US',
            'MESH': 'MSH',
            'MDR': 'MDR',
            'LNC': 'LNC',
            'ICD9CM': 'ICD9CM',
            'ICD10CM': 'ICD10CM',
            'NCI': 'NCI',
            'GO': 'GO',
            'HPO': 'HPO',
            'LCH_NW': 'LCH_NW',
            'NCBI': 'NCBI'
        }
        return vocab_mapping.get(vocab_name, vocab_name)


class RateLimitedThreadPool:
    """Thread pool with global rate limiting for API calls."""
    
    def __init__(self, max_workers: int, rate_limit: float):
        self.max_workers = max_workers
        self.rate_limit = rate_limit  # seconds between requests
        self.last_request_time = 0
        self.lock = threading.Lock()
        
    def wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits."""
        with self.lock:
            if self.rate_limit > 0:
                elapsed = time.time() - self.last_request_time
                if elapsed < self.rate_limit:
                    sleep_time = self.rate_limit - elapsed
                    time.sleep(sleep_time)
                self.last_request_time = time.time()