"""
Statistics calculation utilities for hierarchical path extraction.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import logging


class PathStatistics:
    """Calculate comprehensive statistics for extracted hierarchical paths."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_vocabulary_statistics(self, 
                                      cui_mappings: Dict[str, Dict[str, Any]], 
                                      extracted_paths: Dict[str, Dict[str, Any]], 
                                      vocab_name: str,
                                      processing_time: float,
                                      api_calls: int = 0,
                                      api_errors: int = 0) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for a vocabulary.
        
        Args:
            cui_mappings: Original CUI mappings from combined file
            extracted_paths: Extracted path data
            vocab_name: Name of the vocabulary
            processing_time: Total processing time in seconds
            api_calls: Number of API calls made
            api_errors: Number of API errors encountered
            
        Returns:
            Dictionary with comprehensive statistics
        """
        vocab_key = self._get_vocab_key(vocab_name)
        
        # Filter CUI mappings that have this vocabulary
        relevant_cuis = {cui: data for cui, data in cui_mappings.items() 
                        if vocab_key in data}
        
        # Count total codes for this vocabulary
        total_codes = 0
        for cui_data in relevant_cuis.values():
            if vocab_key in cui_data:
                total_codes += len(cui_data[vocab_key])
        
        # Calculate CUI-level statistics
        cui_stats = self._calculate_cui_stats(relevant_cuis, extracted_paths, vocab_key)
        
        # Calculate path statistics
        path_stats = self._calculate_path_stats(extracted_paths)
        
        # Calculate processing statistics
        processing_stats = self._calculate_processing_stats(
            processing_time, len(relevant_cuis), api_calls, api_errors
        )
        
        # Calculate error statistics
        error_stats = self._calculate_error_stats(extracted_paths)
        
        return {
            'vocabulary': vocab_name,
            'cui_stats': cui_stats,
            'path_stats': path_stats,
            'processing_stats': processing_stats,
            'error_stats': error_stats,
            'summary': {
                'total_cuis_with_mapping': len(relevant_cuis),
                'total_codes': total_codes,
                'successfully_processed_cuis': cui_stats['successfully_processed'],
                'total_paths_extracted': path_stats['total_paths_extracted'],
                'success_rate': cui_stats['successfully_processed'] / len(relevant_cuis) if relevant_cuis else 0,
                'average_paths_per_cui': path_stats['total_paths_extracted'] / len(extracted_paths) if extracted_paths else 0
            }
        }
        
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
        
    def _calculate_cui_stats(self, relevant_cuis: Dict, extracted_paths: Dict, vocab_key: str) -> Dict[str, Any]:
        """Calculate CUI-level statistics."""
        total_cuis_with_mapping = len(relevant_cuis)
        successfully_processed = len([cui for cui in extracted_paths.keys() 
                                    if extracted_paths[cui].get('paths')])
        failed_cuis = total_cuis_with_mapping - successfully_processed
        
        # Count total codes
        total_codes = 0
        codes_not_found = 0
        successful_codes = 0
        
        for cui, cui_data in relevant_cuis.items():
            if vocab_key in cui_data:
                cui_codes = len(cui_data[vocab_key])
                total_codes += cui_codes
                
                if cui in extracted_paths:
                    extracted_data = extracted_paths[cui]
                    if extracted_data.get('paths'):
                        successful_codes += len(extracted_data.get('source_codes', []))
                    else:
                        codes_not_found += cui_codes
                else:
                    codes_not_found += cui_codes
        
        failed_codes = total_codes - successful_codes
        
        return {
            'total_cuis_with_mapping': total_cuis_with_mapping,
            'successfully_processed': successfully_processed,
            'failed_cuis': failed_cuis,
            'total_codes': total_codes,
            'successful_codes': successful_codes,
            'failed_codes': failed_codes,
            'codes_not_found': codes_not_found
        }
        
    def _calculate_path_stats(self, extracted_paths: Dict) -> Dict[str, Any]:
        """Calculate path-level statistics."""
        all_paths = []
        paths_per_code = []
        path_lengths = []
        
        for cui_data in extracted_paths.values():
            if cui_data.get('paths'):
                cui_paths = cui_data['paths']
                all_paths.extend(cui_paths)
                paths_per_code.append(len(cui_paths))
                
                for path in cui_paths:
                    path_lengths.append(len(path))
        
        return {
            'total_paths_extracted': len(all_paths),
            'paths_per_code': self._calculate_distribution_stats(paths_per_code),
            'path_lengths': self._calculate_distribution_stats(path_lengths),
            'unique_path_lengths': len(set(path_lengths)) if path_lengths else 0
        }
        
    def _calculate_processing_stats(self, processing_time: float, total_cuis: int, 
                                  api_calls: int, api_errors: int) -> Dict[str, Any]:
        """Calculate processing performance statistics."""
        return {
            'total_processing_time': processing_time,
            'average_time_per_cui': processing_time / total_cuis if total_cuis > 0 else 0,
            'api_calls_made': api_calls,
            'api_errors': api_errors,
            'api_success_rate': (api_calls - api_errors) / api_calls if api_calls > 0 else 0,
            'cuis_per_second': total_cuis / processing_time if processing_time > 0 else 0
        }
        
    def _calculate_error_stats(self, extracted_paths: Dict) -> Dict[str, Any]:
        """Calculate error statistics."""
        total_errors = 0
        error_types = Counter()
        cuis_with_errors = 0
        
        for cui_data in extracted_paths.values():
            if cui_data.get('errors'):
                cuis_with_errors += 1
                errors = cui_data['errors']
                total_errors += len(errors)
                
                # Categorize errors
                for error in errors:
                    if 'timeout' in error.lower():
                        error_types['timeout'] += 1
                    elif 'not found' in error.lower():
                        error_types['not_found'] += 1
                    elif 'api' in error.lower():
                        error_types['api_error'] += 1
                    else:
                        error_types['other'] += 1
        
        return {
            'total_errors': total_errors,
            'cuis_with_errors': cuis_with_errors,
            'error_types': dict(error_types),
            'error_rate': cuis_with_errors / len(extracted_paths) if extracted_paths else 0
        }
        
    def _calculate_distribution_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate distribution statistics for a list of values."""
        if not values:
            return {
                'min': 0,
                'max': 0,
                'mean': 0,
                'median': 0,
                'std': 0,
                'count': 0
            }
            
        return {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'count': len(values)
        }
        
    def generate_summary_report(self, all_vocab_stats: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a human-readable summary report.
        
        Args:
            all_vocab_stats: Dictionary with statistics for all vocabularies
            
        Returns:
            Formatted summary report string
        """
        report = ["=" * 80]
        report.append("HIERARCHICAL PATH EXTRACTION SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        total_cuis = 0
        total_paths = 0
        total_time = 0
        
        for vocab_name, stats in all_vocab_stats.items():
            summary = stats['summary']
            processing = stats['processing_stats']
            
            total_cuis += summary['total_cuis_with_mapping']
            total_paths += summary['total_paths_extracted']
            total_time += processing['total_processing_time']
            
            report.append(f"VOCABULARY: {vocab_name}")
            report.append("-" * 40)
            report.append(f"  CUIs with mappings: {summary['total_cuis_with_mapping']:,}")
            report.append(f"  Successfully processed: {summary['successfully_processed_cuis']:,} ({summary['success_rate']:.1%})")
            report.append(f"  Total paths extracted: {summary['total_paths_extracted']:,}")
            report.append(f"  Avg paths per CUI: {summary['average_paths_per_cui']:.2f}")
            report.append(f"  Processing time: {processing['total_processing_time']:.1f}s ({processing['cuis_per_second']:.1f} CUIs/sec)")
            
            if stats['path_stats']['path_lengths']['count'] > 0:
                path_len = stats['path_stats']['path_lengths']
                report.append(f"  Path length - Min: {path_len['min']:.0f}, Max: {path_len['max']:.0f}, Mean: {path_len['mean']:.1f}")
            
            error_stats = stats['error_stats']
            if error_stats['total_errors'] > 0:
                report.append(f"  Errors: {error_stats['total_errors']} ({error_stats['error_rate']:.1%} of CUIs)")
            
            report.append("")
            
        report.append("=" * 80)
        report.append("OVERALL SUMMARY")
        report.append("=" * 80)
        report.append(f"Total vocabularies processed: {len(all_vocab_stats)}")
        report.append(f"Total CUIs with mappings: {total_cuis:,}")
        report.append(f"Total paths extracted: {total_paths:,}")
        report.append(f"Total processing time: {total_time:.1f}s ({total_time/3600:.1f} hours)")
        report.append(f"Average processing rate: {total_cuis/total_time:.1f} CUIs/second")
        report.append("")
        
        return "\n".join(report)