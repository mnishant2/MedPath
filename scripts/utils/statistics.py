#!/usr/bin/env python3
"""
Statistics and visualization utilities for dataset analysis.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional
import os
from pathlib import Path

def convert_for_json(obj):
    """Convert numpy and other non-serializable objects to JSON-serializable format."""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (Counter, defaultdict)):
        return dict(obj)
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    else:
        return obj

class DatasetStatistics:
    """Utility class for generating dataset statistics and visualizations."""
    
    def __init__(self, output_dir: str = "stats"):
        """Initialize with output directory for plots."""
        self.output_dir = output_dir
        self.dataset_stats_dir = os.path.join(output_dir, "datasets")
        self.plots_dir = os.path.join(output_dir, "plots")
        
        # Create organized directory structure
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.dataset_stats_dir, exist_ok=True) 
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def calculate_basic_stats(self, documents: List[Dict]) -> Dict[str, Any]:
        """Calculate basic statistics for a dataset."""
        stats = {
            'total_documents': len(documents),
            'total_mentions': 0,
            'unique_mentions_by_text': 0,
            'unique_mentions_by_span': 0,
            'unique_cuis': 0,
            'mapped_mentions': 0,
            'unmapped_mentions': 0,
            'cui_less_mentions': 0,  # NEW: Track CUI-less mentions
            'no_concept_id_mentions': 0,  # NEW: Track mentions with no concept ID
            'mentions_per_document': {},
            'mention_lengths': [],
            'document_lengths': [],
            'cui_distribution': Counter(),
            'semantic_type_distribution': Counter(),
            'native_ontology_distribution': Counter(),
            # NEW: Enhanced mapping source tracking
            'mapping_source_stats': {
                'specific_snomed': 0,  # COMETA specific SNOMED
                'general_snomed': 0,   # COMETA general SNOMED fallback
                'meddra_pt': 0,        # ADR MedDRA PT
                'meddra_llt': 0,       # ADR MedDRA LLT fallback
                'cui_validation': 0,   # CUI validation success
                'text_mapping': 0,     # Text-based fallback
                'ontology_id': 0,      # Standard ontology ID mapping
                'unmapped': 0          # Failed to map
            },
            # NEW: Track comprehensive mapping improvements
            'mapping_improvements': {
                'originally_cui_less': 0,       # Originally CUI-less but got mapped
                'originally_no_concept': 0,     # Originally no concept but got mapped
                'cui_validation_success': 0,    # CUI validation succeeded
                'fallback_success': 0           # Fallback mapping succeeded
            }
        }
        
        all_mentions = []
        mention_texts = set()
        mention_spans = set()
        cuis = set()
        
        for doc in documents:
            doc_mentions = doc.get('mentions', [])
            stats['total_mentions'] += len(doc_mentions)
            stats['document_lengths'].append(len(doc['text']))
            
            # Document-level mention count
            doc_id = doc.get('doc_id', doc.get('id', f'doc_{len(stats["mentions_per_document"])}'))
            stats['mentions_per_document'][doc_id] = len(doc_mentions)
            
            for mention in doc_mentions:
                all_mentions.append(mention)
                
                # Text and span uniqueness
                mention_texts.add(mention['text'])
                mention_spans.add((doc_id, mention['start'], mention['end']))
                
                # Mention length
                stats['mention_lengths'].append(len(mention['text']))
                
                # Analyze original mention state and mapping results
                cui = mention.get('cui', '')
                native_id = mention.get('native_id', '')
                specific_snomed_id = mention.get('specific_snomed_id', '')
                general_snomed_id = mention.get('general_snomed_id', '')
                
                # Track CUI-less and no concept ID mentions
                if not native_id or native_id in ['', 'nan', 'NULL', 'null']:
                    if not specific_snomed_id and not general_snomed_id:
                        stats['no_concept_id_mentions'] += 1
                
                if native_id and native_id.upper() in ['CUI-LESS', 'CUILESS', 'CUI LESS']:
                    stats['cui_less_mentions'] += 1
                
                # CUI mapping results
                if cui and cui != '' and cui != 'CUI-less':
                    stats['mapped_mentions'] += 1
                    cuis.add(cui)
                    stats['cui_distribution'][cui] += 1
                    
                    # Track mapping source for different datasets
                    self._track_mapping_source(mention, stats)
                    
                    # Track mapping improvements
                    self._track_mapping_improvements(mention, stats)
                    
                else:
                    stats['unmapped_mentions'] += 1
                    stats['mapping_source_stats']['unmapped'] += 1
                
                # Semantic types
                semantic_type = mention.get('semantic_type')
                if semantic_type:
                    stats['semantic_type_distribution'][semantic_type] += 1
                
                # Native ontology
                native_ontology = mention.get('native_ontology_name')
                if native_ontology:
                    stats['native_ontology_distribution'][native_ontology] += 1
        
        stats['unique_mentions_by_text'] = len(mention_texts)
        stats['unique_mentions_by_span'] = len(mention_spans)
        stats['unique_cuis'] = len(cuis)
        
        # Calculate derived statistics
        if stats['total_documents'] > 0:
            stats['avg_mentions_per_document'] = stats['total_mentions'] / stats['total_documents']
            stats['avg_document_length'] = np.mean(stats['document_lengths'])
            stats['median_document_length'] = np.median(stats['document_lengths'])
        
        if stats['total_mentions'] > 0:
            stats['mapping_success_rate'] = stats['mapped_mentions'] / stats['total_mentions']
            stats['cui_less_rate'] = stats['cui_less_mentions'] / stats['total_mentions']
            stats['no_concept_rate'] = stats['no_concept_id_mentions'] / stats['total_mentions']
            stats['avg_mention_length'] = np.mean(stats['mention_lengths'])
            stats['median_mention_length'] = np.median(stats['mention_lengths'])
        
        return stats
    
    def _track_mapping_source(self, mention: Dict, stats: Dict):
        """Track the source of successful mapping for different datasets."""
        # COMETA specific: check if specific or general SNOMED was used
        specific_snomed_id = mention.get('specific_snomed_id', '')
        general_snomed_id = mention.get('general_snomed_id', '')
        native_id = mention.get('native_id', '')
        
        if specific_snomed_id and specific_snomed_id == native_id:
            stats['mapping_source_stats']['specific_snomed'] += 1
        elif general_snomed_id and general_snomed_id == native_id:
            stats['mapping_source_stats']['general_snomed'] += 1
        elif native_id and native_id.startswith('C'):
            # Likely CUI validation
            stats['mapping_source_stats']['cui_validation'] += 1
        elif native_id:
            # Standard ontology ID mapping
            stats['mapping_source_stats']['ontology_id'] += 1
        else:
            # Likely text-based mapping
            stats['mapping_source_stats']['text_mapping'] += 1
    
    def _track_mapping_improvements(self, mention: Dict, stats: Dict):
        """Track mapping improvements from comprehensive mapping process."""
        native_id = mention.get('native_id', '')
        cui = mention.get('cui', '')
        
        # Check if originally had no concept ID but now has CUI
        if (not native_id or native_id in ['', 'nan', 'NULL', 'null']) and cui:
            stats['mapping_improvements']['originally_no_concept'] += 1
        
        # Check if originally was CUI-less but now has CUI  
        if native_id and native_id.upper() in ['CUI-LESS', 'CUILESS', 'CUI LESS'] and cui:
            stats['mapping_improvements']['originally_cui_less'] += 1
        
        # Check if CUI validation was successful (old CUI mapped to new CUI)
        if native_id and native_id.startswith('C') and cui and cui != native_id:
            stats['mapping_improvements']['cui_validation_success'] += 1
        
        # Check if fallback mapping was successful (no native ID but got CUI from text)
        if not native_id and cui:
            stats['mapping_improvements']['fallback_success'] += 1
    
    def generate_plots(self, stats: Dict[str, Any], dataset_name: str):
        """Generate visualization plots for dataset statistics."""
        
        # 1. Document length distribution
        plt.figure(figsize=(10, 6))
        plt.hist(stats['document_lengths'], bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{dataset_name}: Document Length Distribution')
        plt.xlabel('Document Length (characters)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, f'{dataset_name}_document_lengths.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Mention length distribution
        plt.figure(figsize=(10, 6))
        plt.hist(stats['mention_lengths'], bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{dataset_name}: Mention Length Distribution')
        plt.xlabel('Mention Length (characters)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, f'{dataset_name}_mention_lengths.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Mentions per document distribution
        mentions_per_doc_counts = list(stats['mentions_per_document'].values())
        plt.figure(figsize=(10, 6))
        plt.hist(mentions_per_doc_counts, bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'{dataset_name}: Mentions per Document Distribution')
        plt.xlabel('Number of Mentions')
        plt.ylabel('Number of Documents')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.plots_dir, f'{dataset_name}_mentions_per_document.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Top CUIs
        if stats['cui_distribution']:
            top_cuis = stats['cui_distribution'].most_common(20)
            cuis, counts = zip(*top_cuis)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(cuis)), counts)
            plt.yticks(range(len(cuis)), cuis)
            plt.title(f'{dataset_name}: Top 20 Most Frequent CUIs')
            plt.xlabel('Frequency')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'{dataset_name}_top_cuis.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Semantic type distribution
        if stats['semantic_type_distribution']:
            top_types = stats['semantic_type_distribution'].most_common(15)
            types, counts = zip(*top_types)
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(types)), counts)
            plt.xticks(range(len(types)), types, rotation=45, ha='right')
            plt.title(f'{dataset_name}: Semantic Type Distribution')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'{dataset_name}_semantic_types.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Native ontology distribution
        if stats['native_ontology_distribution']:
            ontologies = list(stats['native_ontology_distribution'].keys())
            counts = list(stats['native_ontology_distribution'].values())
            
            plt.figure(figsize=(10, 6))
            plt.bar(ontologies, counts)
            plt.title(f'{dataset_name}: Native Ontology Distribution')
            plt.ylabel('Number of Mentions')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'{dataset_name}_native_ontologies.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_stats_to_file(self, stats: Dict[str, Any], dataset_name: str):
        """Save statistics to JSON file."""
        # Create a copy to avoid modifying the original
        serializable_stats = dict(stats)
        
        # Remove large lists to keep file size manageable and convert to summary stats
        for key in ['mention_lengths', 'document_lengths', 'mentions_per_document']:
            if key in serializable_stats:
                # Keep only summary statistics
                if key == 'mentions_per_document':
                    # Convert to summary stats
                    values = list(serializable_stats[key].values())
                    serializable_stats[f'{key}_summary'] = {
                        'mean': float(np.mean(values)),
                        'median': float(np.median(values)),
                        'min': int(np.min(values)),
                        'max': int(np.max(values)),
                        'std': float(np.std(values))
                    }
                del serializable_stats[key]
        
        # Convert non-serializable objects using the robust conversion function
        serializable_stats = convert_for_json(serializable_stats)
        
        output_file = os.path.join(self.dataset_stats_dir, f'{dataset_name}_statistics.json')
        with open(output_file, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
    
    def generate_comparison_plot(self, dataset_stats: Dict[str, Dict], metric: str, title: str):
        """Generate comparison plot across datasets."""
        datasets = list(dataset_stats.keys())
        values = [dataset_stats[dataset].get(metric, 0) for dataset in datasets]
        
        plt.figure(figsize=(12, 6))
        plt.bar(datasets, values)
        plt.title(title)
        plt.ylabel(metric.replace('_', ' ').title())
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'comparison_{metric}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_mapping_files(self, documents: List[Dict], dataset_name: str) -> Dict[str, str]:
        """Create basic CUI to vocabulary and TUI mapping files from processed data."""
        # Skip basic mapping creation since we now have comprehensive mappings
        # This avoids duplicate files like cdr_cui_to_tui.json vs cdr_cui_to_tuis.json
        return {}

    def generate_cross_dataset_mapping_summary(self, processed_dirs: List[Path]) -> Dict[str, str]:
        """Generate cross-dataset mapping summary and combined mapping files."""
        
        # Collect all CUIs across datasets
        all_cuis = set()
        dataset_cui_counts = {}
        
        # Collect comprehensive mapping files
        comprehensive_vocab_files = []
        comprehensive_tui_files = []
        
        for dataset_dir in processed_dirs:
            dataset_name = dataset_dir.name
            mappings_dir = dataset_dir.parent / 'mappings'
            
            # Look for comprehensive mapping files
            vocab_codes_file = mappings_dir / f'{dataset_name}_cui_to_vocab_codes.json'
            tuis_file = mappings_dir / f'{dataset_name}_cui_to_tuis.json'
            
            if vocab_codes_file.exists():
                comprehensive_vocab_files.append(vocab_codes_file)
                
            if tuis_file.exists():
                comprehensive_tui_files.append(tuis_file)
            
            # Count CUIs per dataset
            documents_file = dataset_dir / f'{dataset_name}.jsonl'
            if documents_file.exists():
                dataset_cuis = set()
                try:
                    with open(documents_file, 'r') as f:
                        for line in f:
                            doc = json.loads(line.strip())
                            for mention in doc.get('mentions', []):
                                cui = mention.get('cui')
                                if cui and cui != '' and cui != 'CUI-less':
                                    dataset_cuis.add(cui)
                                    all_cuis.add(cui)
                    dataset_cui_counts[dataset_name] = len(dataset_cuis)
                except Exception as e:
                    print(f"Error processing {documents_file}: {e}")
        
        # Create combined comprehensive mappings
        combined_vocab_codes = {}
        combined_vocab_codes_with_tty = {}
        combined_tuis = {}
        vocabulary_sources = set()
        all_semantic_types = set()
        
        # Merge vocabulary code mappings
        for vocab_file in comprehensive_vocab_files:
            try:
                with open(vocab_file, 'r') as f:
                    data = json.load(f)
                    for cui, vocab_mapping in data.items():
                        if cui not in combined_vocab_codes:
                            combined_vocab_codes[cui] = {}
                        if cui not in combined_vocab_codes_with_tty:
                            combined_vocab_codes_with_tty[cui] = {}

                        # Merge vocabulary mappings (support both codes-only and with-tty structures)
                        for source, value in vocab_mapping.items():
                            vocabulary_sources.add(source)

                            # Ensure containers exist
                            if source not in combined_vocab_codes[cui]:
                                combined_vocab_codes[cui][source] = []
                            if source not in combined_vocab_codes_with_tty[cui]:
                                combined_vocab_codes_with_tty[cui][source] = {}

                            # with-tty structure: value is dict of code -> metadata (including TTY)
                            if isinstance(value, dict):
                                for code, meta in value.items():
                                    if code not in combined_vocab_codes[cui][source]:
                                        combined_vocab_codes[cui][source].append(code)
                                    if code not in combined_vocab_codes_with_tty[cui][source]:
                                        combined_vocab_codes_with_tty[cui][source][code] = meta if isinstance(meta, dict) else {"meta": meta}
                                    else:
                                        if isinstance(meta, dict):
                                            combined_vocab_codes_with_tty[cui][source][code].update(meta)
                            # codes-only list
                            elif isinstance(value, list):
                                for code in value:
                                    if code not in combined_vocab_codes[cui][source]:
                                        combined_vocab_codes[cui][source].append(code)
                                    combined_vocab_codes_with_tty[cui][source].setdefault(code, {})
                            else:
                                # Single code string fallback
                                code = str(value)
                                if code not in combined_vocab_codes[cui][source]:
                                    combined_vocab_codes[cui][source].append(code)
                                combined_vocab_codes_with_tty[cui][source].setdefault(code, {})
            except Exception as e:
                print(f"Error processing {vocab_file}: {e}")
        
        # Merge semantic type mappings
        for tui_file in comprehensive_tui_files:
            try:
                with open(tui_file, 'r') as f:
                    data = json.load(f)
                    for cui, tuis in data.items():
                        if cui not in combined_tuis:
                            combined_tuis[cui] = []
                        
                        # Add TUIs if not already present
                        for tui in tuis:
                            all_semantic_types.add(tui)
                            if tui not in combined_tuis[cui]:
                                combined_tuis[cui].append(tui)
            except Exception as e:
                print(f"Error processing {tui_file}: {e}")
        
        # Save combined mapping files under sibling of stats dir, preserving suffix
        stats_dir = Path(self.output_dir)
        if stats_dir.name.startswith('stats'):
            mappings_dir = stats_dir.parent / stats_dir.name.replace('stats', 'mappings')
        else:
            mappings_dir = stats_dir.parent / 'mappings'
        os.makedirs(mappings_dir, exist_ok=True)
        
        combined_vocab_file = mappings_dir / 'combined_cui_to_vocab_codes.json'
        combined_vocab_file_with_tty = mappings_dir / 'combined_cui_to_vocab_codes_with_tty.json'
        combined_tui_file = mappings_dir / 'combined_cui_to_tuis.json'
        cross_dataset_summary_file = mappings_dir / 'cross_dataset_mapping_summary.json'
        
        # Save combined files
        with open(combined_vocab_file, 'w') as f:
            json.dump(combined_vocab_codes, f, indent=2, sort_keys=True)
        with open(combined_vocab_file_with_tty, 'w') as f:
            json.dump(combined_vocab_codes_with_tty, f, indent=2, sort_keys=True)
        
        with open(combined_tui_file, 'w') as f:
            json.dump(combined_tuis, f, indent=2, sort_keys=True)
        
        # Create cross-dataset summary
        cross_dataset_summary = {
            'total_unique_cuis': len(all_cuis),
            'cuis_with_comprehensive_vocab_mappings': len(combined_vocab_codes),
            'cuis_with_comprehensive_semantic_types': len(combined_tuis),
            'vocabulary_sources_found': sorted(list(vocabulary_sources)),
            'total_vocabulary_sources': len(vocabulary_sources),
            'semantic_types_found': sorted(list(all_semantic_types)),
            'total_semantic_types': len(all_semantic_types),
            'dataset_cui_counts': dataset_cui_counts,
            'comprehensive_mapping_coverage': {
                'vocab_codes_coverage': len(combined_vocab_codes) / len(all_cuis) if all_cuis else 0,
                'semantic_types_coverage': len(combined_tuis) / len(all_cuis) if all_cuis else 0
            }
        }
        
        with open(cross_dataset_summary_file, 'w') as f:
            json.dump(cross_dataset_summary, f, indent=2)
        
        print(f"\nCross-dataset mapping summary:")
        print(f"  - Total unique CUIs across all datasets: {len(all_cuis):,}")
        print(f"  - CUIs with comprehensive vocabulary mappings: {len(combined_vocab_codes):,}")
        print(f"  - CUIs with comprehensive semantic types: {len(combined_tuis):,}")
        print(f"  - Vocabulary sources found: {len(vocabulary_sources)}")
        print(f"  - Semantic types found: {len(all_semantic_types)}")
        
        return {
            'combined_vocab_codes': str(combined_vocab_file),
            'combined_tuis': str(combined_tui_file),
            'cross_dataset_summary': str(cross_dataset_summary_file)
        }

    def generate_cross_dataset_mapping_summary_fixed(self, documents_dir: Path, dataset_names: set) -> Dict[str, str]:
        """Generate cross-dataset mapping summary for new file structure."""
        
        # Collect all CUIs across datasets
        all_cuis = set()
        dataset_cui_counts = {}
        
        # Collect comprehensive mapping files
        comprehensive_vocab_files = []
        comprehensive_tui_files = []
        # Respect the same suffix as the documents directory (e.g., documents_sample -> mappings_sample)
        if documents_dir.name.startswith('documents'):
            mappings_subdir = documents_dir.name.replace('documents', 'mappings')
        else:
            mappings_subdir = 'mappings'
        mappings_dir = documents_dir.parent / mappings_subdir
        
        for dataset_name in dataset_names:
            # Look for comprehensive mapping files (prefer _with_tty if present)
            vocab_codes_file_tty = mappings_dir / f'{dataset_name}_cui_to_vocab_codes_with_tty.json'
            vocab_codes_file = mappings_dir / f'{dataset_name}_cui_to_vocab_codes.json'
            tuis_file = mappings_dir / f'{dataset_name}_cui_to_tuis.json'
            
            if vocab_codes_file_tty.exists():
                comprehensive_vocab_files.append(vocab_codes_file_tty)
            elif vocab_codes_file.exists():
                comprehensive_vocab_files.append(vocab_codes_file)
                
            if tuis_file.exists():
                comprehensive_tui_files.append(tuis_file)
            
            # Count CUIs per dataset from documents directory
            dataset_files = [
                documents_dir / f'{dataset_name}.jsonl',
                documents_dir / f'{dataset_name}_train.jsonl',
                documents_dir / f'{dataset_name}_dev.jsonl',
                documents_dir / f'{dataset_name}_test.jsonl'
            ]
            
            dataset_cuis = set()
            for documents_file in dataset_files:
                if documents_file.exists():
                    try:
                        with open(documents_file, 'r') as f:
                            for line in f:
                                doc = json.loads(line.strip())
                                for mention in doc.get('mentions', []):
                                    cui = mention.get('cui')
                                    if cui and cui != '' and cui != 'CUI-less':
                                        dataset_cuis.add(cui)
                                        all_cuis.add(cui)
                    except Exception as e:
                        print(f"Error processing {documents_file}: {e}")
            
            if dataset_cuis:
                dataset_cui_counts[dataset_name] = len(dataset_cuis)
        
        # Create combined comprehensive mappings
        combined_vocab_codes = {}
        combined_vocab_codes_with_tty = {}
        combined_tuis = {}
        vocabulary_sources = set()
        all_semantic_types = set()
        
        # Merge vocabulary code mappings
        for vocab_file in comprehensive_vocab_files:
            try:
                with open(vocab_file, 'r') as f:
                    data = json.load(f)
                    for cui, vocab_mapping in data.items():
                        if cui not in combined_vocab_codes:
                            combined_vocab_codes[cui] = {}
                        if cui not in combined_vocab_codes_with_tty:
                            combined_vocab_codes_with_tty[cui] = {}

                        # Merge vocabulary mappings (support both codes-only and with-tty structures)
                        for source, value in vocab_mapping.items():
                            vocabulary_sources.add(source)
                            if source not in combined_vocab_codes[cui]:
                                combined_vocab_codes[cui][source] = []
                            if source not in combined_vocab_codes_with_tty[cui]:
                                combined_vocab_codes_with_tty[cui][source] = {}

                            if isinstance(value, dict):
                                for code, meta in value.items():
                                    if code not in combined_vocab_codes[cui][source]:
                                        combined_vocab_codes[cui][source].append(code)
                                    if code not in combined_vocab_codes_with_tty[cui][source]:
                                        combined_vocab_codes_with_tty[cui][source][code] = meta if isinstance(meta, dict) else {"meta": meta}
                                    else:
                                        if isinstance(meta, dict):
                                            combined_vocab_codes_with_tty[cui][source][code].update(meta)
                            elif isinstance(value, list):
                                for code in value:
                                    if code not in combined_vocab_codes[cui][source]:
                                        combined_vocab_codes[cui][source].append(code)
                                    combined_vocab_codes_with_tty[cui][source].setdefault(code, {})
                            else:
                                code = str(value)
                                if code not in combined_vocab_codes[cui][source]:
                                    combined_vocab_codes[cui][source].append(code)
                                combined_vocab_codes_with_tty[cui][source].setdefault(code, {})
            except Exception as e:
                print(f"Error processing {vocab_file}: {e}")
        
        # Merge semantic type mappings
        for tui_file in comprehensive_tui_files:
            try:
                with open(tui_file, 'r') as f:
                    data = json.load(f)
                    for cui, tuis in data.items():
                        if cui not in combined_tuis:
                            combined_tuis[cui] = []
                        
                        # Add TUIs if not already present
                        for tui in tuis:
                            all_semantic_types.add(tui)
                            if tui not in combined_tuis[cui]:
                                combined_tuis[cui].append(tui)
            except Exception as e:
                print(f"Error processing {tui_file}: {e}")
        
        # Save combined mapping files
        os.makedirs(mappings_dir, exist_ok=True)
        
        combined_vocab_file = mappings_dir / 'combined_cui_to_vocab_codes.json'
        combined_vocab_file_with_tty = mappings_dir / 'combined_cui_to_vocab_codes_with_tty.json'
        combined_tui_file = mappings_dir / 'combined_cui_to_tuis.json'
        cross_dataset_summary_file = mappings_dir / 'cross_dataset_mapping_summary.json'
        
        # Save combined files
        with open(combined_vocab_file, 'w') as f:
            json.dump(combined_vocab_codes, f, indent=2, sort_keys=True)
        with open(combined_vocab_file_with_tty, 'w') as f:
            json.dump(combined_vocab_codes_with_tty, f, indent=2, sort_keys=True)
        
        with open(combined_tui_file, 'w') as f:
            json.dump(combined_tuis, f, indent=2, sort_keys=True)
        
        # Create cross-dataset summary
        cross_dataset_summary = {
            'total_unique_cuis': len(all_cuis),
            'cuis_with_comprehensive_vocab_mappings': len(combined_vocab_codes),
            'cuis_with_comprehensive_semantic_types': len(combined_tuis),
            'vocabulary_sources_found': sorted(list(vocabulary_sources)),
            'total_vocabulary_sources': len(vocabulary_sources),
            'semantic_types_found': sorted(list(all_semantic_types)),
            'total_semantic_types': len(all_semantic_types),
            'dataset_cui_counts': dataset_cui_counts,
            'comprehensive_mapping_coverage': {
                'vocab_codes_coverage': len(combined_vocab_codes) / len(all_cuis) if all_cuis else 0,
                'semantic_types_coverage': len(combined_tuis) / len(all_cuis) if all_cuis else 0
            }
        }
        
        with open(cross_dataset_summary_file, 'w') as f:
            json.dump(cross_dataset_summary, f, indent=2)
        
        print(f"\nCross-dataset mapping summary:")
        print(f"  - Total unique CUIs across all datasets: {len(all_cuis):,}")
        print(f"  - CUIs with comprehensive vocabulary mappings: {len(combined_vocab_codes):,}")
        print(f"  - CUIs with comprehensive semantic types: {len(combined_tuis):,}")
        print(f"  - Vocabulary sources found: {len(vocabulary_sources)}")
        print(f"  - Semantic types found: {len(all_semantic_types)}")
        
        return {
            'combined_vocab_codes': str(combined_vocab_file),
            'combined_tuis': str(combined_tui_file),
            'cross_dataset_summary': str(cross_dataset_summary_file)
        } 

    def generate_dataset_statistics(self, dataset_name: str, dataset_dir: Path):
        """Generate comprehensive statistics for a single dataset."""
        
        # Look for the main dataset file
        dataset_files = [
            dataset_dir / f'{dataset_name}.jsonl',
            dataset_dir / f'{dataset_name}_train.jsonl',
            dataset_dir / f'{dataset_name}_test.jsonl',
            dataset_dir / f'{dataset_name}_dev.jsonl'
        ]
        
        # Find existing files
        existing_files = [f for f in dataset_files if f.exists()]
        
        if not existing_files:
            print(f"Warning: No JSONL files found for dataset {dataset_name}")
            return
        
        # Process the main file or combined files
        all_documents = []
        
        for file_path in existing_files:
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        doc = json.loads(line.strip())
                        all_documents.append(doc)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        if not all_documents:
            print(f"Warning: No documents found for dataset {dataset_name}")
            return
        
        # Generate statistics
        stats = self.calculate_basic_stats(all_documents)
        self.generate_plots(stats, dataset_name)
        self.save_stats_to_file(stats, dataset_name)
        
        print(f"Generated statistics for {dataset_name}: {len(all_documents)} documents, {stats['total_mentions']} mentions")

    def generate_comparison_statistics(self, dataset_dirs: List[Path]):
        """Generate comparison statistics across multiple datasets."""
        
        comparison_stats = {}
        
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            
            # Look for statistics file under the dataset_stats_dir
            stats_file = Path(self.dataset_stats_dir) / f'{dataset_name}_statistics.json'
            
            if stats_file.exists():
                try:
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                        comparison_stats[dataset_name] = stats
                except Exception as e:
                    print(f"Error reading statistics for {dataset_name}: {e}")
        
        if not comparison_stats:
            print("Warning: No statistics files found for comparison")
            return
        
        # Generate comparison plots
        metrics_to_compare = [
            'total_documents', 'total_mentions', 'unique_cuis', 
            'mapping_success_rate', 'avg_mentions_per_document'
        ]
        
        for metric in metrics_to_compare:
            if all(metric in stats for stats in comparison_stats.values()):
                self.generate_comparison_plot(
                    comparison_stats, 
                    metric, 
                    f'Cross-Dataset Comparison: {metric.replace("_", " ").title()}'
                )
        
        # Save comparison summary
        comparison_summary_file = os.path.join(self.output_dir, 'dataset_comparison_summary.json')
        with open(comparison_summary_file, 'w') as f:
            json.dump(comparison_stats, f, indent=2)
        
        print(f"Generated comparison statistics for {len(comparison_stats)} datasets") 