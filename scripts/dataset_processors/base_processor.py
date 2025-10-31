#!/usr/bin/env python3
"""
Base processor class for standardized dataset processing.

All dataset-specific processors inherit from this class to ensure consistent
output format and common functionality.
"""

import json
import os
import logging
import shutil
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.umls_mapper import UMLSMapper
from utils.statistics import DatasetStatistics

class BaseProcessor(ABC):
    """Base class for dataset processors."""
    
    def __init__(self, 
                 dataset_name: str,
                 data_dir: str,
                 output_dir: str,
                 umls_api_key: Optional[str] = None,
                 use_local_umls: bool = False,
                 umls_path: Optional[str] = None,
                 limit: Optional[int] = None,
                 subdir_suffix: str = ""):
        """
        Initialize the base processor.
        
        Args:
            dataset_name: Name of the dataset
            data_dir: Directory containing the raw data
            output_dir: Directory to save processed data
            umls_api_key: UMLS API key (if not using local UMLS)
            use_local_umls: Whether to use local UMLS files
            umls_path: Path to local UMLS installation
            limit: Maximum number of documents to process (for testing)
        """
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.umls_api_key = umls_api_key
        self.use_local_umls = use_local_umls
        self.limit = limit
        self.subdir_suffix = subdir_suffix or ""
        
        # Create output directories (allow writing to documents_new/mappings_new/stats_new)
        self.documents_dir = self.output_dir / f"documents{self.subdir_suffix}"
        self.mappings_dir = self.output_dir / f"mappings{self.subdir_suffix}"
        self.stats_dir = self.output_dir / f"stats{self.subdir_suffix}"
        self.cache_dir = self.output_dir / f"cache{self.subdir_suffix}"
        
        for dir_path in [self.documents_dir, self.mappings_dir, self.stats_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize UMLS mapper
        if use_local_umls:
            # Add parent directories to sys.path for import
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from utils.local_umls_mapper import LocalUMLSMapper
            if not umls_path:
                umls_path = "../../../umls/2025AA"  # Default path
            
            self.umls_mapper = LocalUMLSMapper(
                umls_path=umls_path,
                dataset_name=dataset_name,
                cache_dir=str(self.cache_dir)
            )
        else:
            # Use API-based mapper
            api_key = umls_api_key or os.environ.get('UMLS_API_KEY')
            if not api_key:
                raise ValueError("UMLS API key is required. Provide it as parameter or set UMLS_API_KEY environment variable.")
            
            self.umls_mapper = UMLSMapper(
                api_key=api_key,
                dataset_name=dataset_name,
                cache_dir=str(self.cache_dir)
            )
        
        self.statistics = DatasetStatistics(output_dir=str(self.stats_dir))
        
        # Setup logging
        self._setup_logging()
        
        # Dataset-specific configuration
        self.native_ontologies = self._get_native_ontologies()
        
    def _setup_logging(self):
        """Setup logging for the processor."""
        logs_dir = Path(__file__).parent.parent / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"{self.dataset_name}_processing.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"{self.dataset_name}_processor")
    
    @abstractmethod
    def _get_native_ontologies(self) -> List[str]:
        """Return list of native ontologies used by this dataset."""
        pass
    
    @abstractmethod
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load the raw dataset and return as list of documents with mentions."""
        pass
    
    @abstractmethod
    def _extract_mentions(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract mentions from raw data and return in internal format."""
        pass
    
    def _map_to_umls(self, mentions: List[Dict]) -> List[Dict]:
        """Map mentions to UMLS CUIs using the UMLSMapper."""
        self.logger.info(f"Starting UMLS mapping for {len(mentions)} mentions...")
        
        # Group mentions by unique combinations to minimize API calls
        unique_combinations = {}
        mention_to_combo = {}
        
        for i, mention in enumerate(mentions):
            # Create a combo key based on available identifiers
            combo_key = self._create_mapping_key(mention)
            mention_to_combo[i] = combo_key
            
            if combo_key not in unique_combinations:
                unique_combinations[combo_key] = mention
        
        self.logger.info(f"Found {len(unique_combinations)} unique combinations to map")
        
        # Map unique combinations with concurrent processing
        combo_to_result = self._batch_map_combinations(unique_combinations)
        
        # Collect all unique CUIs for batch semantic type lookup
        unique_cuis = set()
        for entry in combo_to_result.values():
            cui = entry or ''
            if isinstance(entry, dict):
                cui = entry.get('cui', '')
            if cui and cui != '':
                unique_cuis.add(cui)
        
        self.logger.info(f"Fetching semantic types for {len(unique_cuis)} unique CUIs...")
        
        # Batch fetch semantic types for all unique CUIs using concurrent processing
        cui_to_semantic_type = self._batch_fetch_semantic_types(unique_cuis)
        
        # Apply mappings back to all mentions with progress bar (now much faster)
        mapped_mentions = []
        for i, mention in tqdm(enumerate(mentions), desc="Updating mentions with CUIs", total=len(mentions)):
            combo_key = mention_to_combo[i]
            result_entry = combo_to_result.get(combo_key, '')
            if isinstance(result_entry, dict):
                cui = result_entry.get('cui', '')
                mapping_method = result_entry.get('method', '')
            else:
                # Backward compatibility if only CUI is returned
                cui = result_entry or ''
                mapping_method = mention.get('mapping_method', '')
            
            # Update mention with CUI, mapping method, and semantic type
            updated_mention = mention.copy()
            updated_mention['cui'] = cui or ''
            if mapping_method:
                updated_mention['mapping_method'] = mapping_method
            
            # Get semantic type and UMLS name from pre-computed lookup
            if cui and cui != '':
                updated_mention['semantic_type'] = cui_to_semantic_type.get(cui, '')
                if hasattr(self.umls_mapper, 'get_closest_umls_name_for_text'):
                    # Prefer the closest synonym to the mention text
                    updated_mention['umls_name'] = self.umls_mapper.get_closest_umls_name_for_text(cui, mention.get('text', ''))
                else:
                    updated_mention['umls_name'] = self.umls_mapper.get_umls_name(cui) if hasattr(self.umls_mapper, 'get_umls_name') else ''
            else:
                updated_mention['semantic_type'] = ''
                updated_mention['umls_name'] = ''
            
            mapped_mentions.append(updated_mention)
        
        self.logger.info(f"Completed UMLS mapping")
        return mapped_mentions
    
    def _create_mapping_key(self, mention: Dict) -> str:
        """Create a unique key for mapping based on available identifiers."""
        # Default implementation, override in subclasses for dataset-specific logic
        key_parts = []
        
        # Add native IDs
        native_id = mention.get('native_id', '')
        if native_id:
            key_parts.append(f"id:{native_id}")
        
        # Add ontology
        ontology = mention.get('native_ontology_name', '')
        if ontology:
            key_parts.append(f"ont:{ontology}")
        
        # Add text as fallback
        text = mention.get('text', '')
        if text:
            key_parts.append(f"text:{text}")
        
        return "|".join(key_parts)
    
    def _map_single_mention(self, mention: Dict) -> Optional[str]:
        """Map a single mention to UMLS CUI and record mapping method on the mention."""
        # Try native ID first
        native_id = mention.get('native_id')
        native_ontology = mention.get('native_ontology_name')

        if native_id and native_ontology:
            cui = self.umls_mapper.get_cui_from_ontology_id(
                code=native_id,
                source=native_ontology,
                code_type=mention.get('code_type', '')
            )
            if cui:
                mention['mapping_method'] = 'native_id_mapping'
                return cui

        # Try text as fallback
        text = mention.get('text')
        if text:
            cui = self.umls_mapper.get_cui_from_text(text)
            if cui:
                # If local mapper exposes last_text_mapping_method, use it
                method = getattr(self.umls_mapper, 'last_text_mapping_method', None)
                if method in {'exact_match', 'semantic_containment'}:
                    mention['mapping_method'] = method
                else:
                    mention['mapping_method'] = 'text_fallback'
                return cui

        mention['mapping_method'] = 'no_mapping'
        return ''
    
    def _batch_map_combinations(self, unique_combinations: Dict[str, Dict]) -> Dict[str, Dict[str, str]]:
        """
        Batch map unique combinations to CUIs.
        
        Args:
            unique_combinations: Dictionary mapping combo keys to mention dictionaries
            
        Returns:
            Dictionary mapping combo keys to CUIs
        """
        combo_to_result: Dict[str, Dict[str, str]] = {}
        
        # For local UMLS files, use simple sequential processing (no threading needed)
        if self.use_local_umls:
            for combo_key, mention in tqdm(unique_combinations.items(), desc="Mapping to UMLS CUIs"):
                try:
                    cui = self._map_single_mention(mention)
                    combo_to_result[combo_key] = {
                        'cui': cui or '',
                        'method': mention.get('mapping_method', '')
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to map combination {combo_key}: {e}")
                    combo_to_result[combo_key] = {
                        'cui': '',
                        'method': 'no_mapping'
                    }
        else:
            # For API-based mapping, use concurrent processing
            def map_single_combination(combo_key_mention):
                """Map a single combination to CUI."""
                combo_key, mention = combo_key_mention
                try:
                    cui = self._map_single_mention(mention)
                    return combo_key, cui, mention.get('mapping_method', '')
                except Exception as e:
                    self.logger.warning(f"Failed to map combination {combo_key}: {e}")
                    return combo_key, None, 'no_mapping'
            
            # Use ThreadPoolExecutor for concurrent API calls
            max_workers = min(10, len(unique_combinations))  # Limit to 10 concurrent requests
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all requests
                future_to_combo = {executor.submit(map_single_combination, item): item[0] for item in unique_combinations.items()}
                
                # Collect results with progress bar
                for future in tqdm(as_completed(future_to_combo), total=len(unique_combinations), desc="Mapping to UMLS CUIs"):
                    combo_key, cui, method = future.result()
                    combo_to_result[combo_key] = {
                        'cui': cui or '',
                        'method': method or ''
                    }
                    
                    # Small delay to avoid overwhelming the API
                    time.sleep(0.1)
        
        return combo_to_result
    
    def _batch_fetch_semantic_types(self, cuis: set) -> Dict[str, str]:
        """
        Batch fetch semantic types for multiple CUIs.
        
        Args:
            cuis: Set of CUIs to fetch semantic types for
            
        Returns:
            Dictionary mapping CUI to semantic type
        """
        cui_to_semantic_type = {}
        
        # Handle edge case where there are no CUIs to process
        if len(cuis) == 0:
            return cui_to_semantic_type
        
        # For local UMLS files, use simple sequential processing (no threading needed)
        if self.use_local_umls:
            for cui in tqdm(cuis, desc="Fetching semantic types"):
                try:
                    semantic_types = self.umls_mapper.get_semantic_types(cui)
                    cui_to_semantic_type[cui] = semantic_types[0] if semantic_types else ''
                except Exception as e:
                    self.logger.warning(f"Failed to fetch semantic type for {cui}: {e}")
                    cui_to_semantic_type[cui] = ''
        else:
            # For API-based mapping, use concurrent processing
            def fetch_semantic_type(cui):
                """Fetch semantic type for a single CUI."""
                try:
                    semantic_types = self.umls_mapper.get_semantic_types(cui)
                    return cui, semantic_types[0] if semantic_types else ''
                except Exception as e:
                    self.logger.warning(f"Failed to fetch semantic type for {cui}: {e}")
                    return cui, ''
            
            # Use ThreadPoolExecutor for concurrent API calls
            max_workers = min(10, len(cuis))  # Limit to 10 concurrent requests
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all requests
                future_to_cui = {executor.submit(fetch_semantic_type, cui): cui for cui in cuis}
                
                # Collect results with progress bar
                for future in tqdm(as_completed(future_to_cui), total=len(cuis), desc="Fetching semantic types"):
                    cui, semantic_type = future.result()
                    cui_to_semantic_type[cui] = semantic_type
                    
                    # Small delay to avoid overwhelming the API
                    time.sleep(0.1)
        
        return cui_to_semantic_type
    
    def _convert_to_standard_format(self, mentions: List[Dict]) -> List[Dict]:
        """Convert mentions to standardized document format."""
        self.logger.info("Converting to standardized format...")
        
        # Group mentions by document
        documents_dict = {}
        
        for mention in tqdm(mentions, desc="Converting to standard format"):
            doc_id = mention['doc_id']
            
            if doc_id not in documents_dict:
                documents_dict[doc_id] = {
                    'doc_id': doc_id,
                    'text': mention['doc_text'],
                    'mentions': []
                }
            
            # Create standardized mention
            standardized_mention = {
                'start': mention['start'],
                'end': mention['end'],
                'text': mention['text'],
                'native_id': mention.get('native_id', ''),
                'native_ontology_name': mention.get('native_ontology_name', ''),
                'cui': mention.get('cui', ''),
                'umls_name': mention.get('umls_name', ''),
                'semantic_type': mention.get('semantic_type', ''),
                'mapping_method': mention.get('mapping_method', '')
            }
            
            documents_dict[doc_id]['mentions'].append(standardized_mention)
        
        documents = list(documents_dict.values())
        self.logger.info(f"Created {len(documents)} documents with mentions")
        
        return documents
    
    def _save_documents(self, documents: List[Dict]):
        """Save documents to JSONL file."""
        output_file = self.documents_dir / f"{self.dataset_name}.jsonl"
        
        self.logger.info(f"Saving {len(documents)} documents to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Documents saved successfully")
    
    def _generate_statistics(self, documents: List[Dict]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Generate comprehensive statistics and mappings for the processed documents."""
        
        # Calculate basic statistics
        stats = self.statistics.calculate_basic_stats(documents)
        
        # Generate plots
        self.statistics.generate_plots(stats, self.dataset_name)
        
        # Save statistics to file
        self.statistics.save_stats_to_file(stats, self.dataset_name)
        
        # Create basic mapping files (from processed data)
        basic_mapping_files = self.statistics.create_mapping_files(documents, self.dataset_name)
        
        # Extract all unique CUIs from the documents
        unique_cuis = set()
        for doc in documents:
            for mention in doc.get('mentions', []):
                cui = mention.get('cui')
                if cui and cui != '' and cui != 'CUI-less':
                    unique_cuis.add(cui)
        
        self.logger.info(f"Found {len(unique_cuis)} unique CUIs for comprehensive mapping")
        
        # Create comprehensive mapping files using UMLS API
        comprehensive_mapping_files = {}
        if unique_cuis:
            try:
                self.logger.info("Generating enhanced UMLS mappings with TTY information...")
                comprehensive_mapping_files = self.umls_mapper.create_enhanced_mapping_files(
                    cuis=unique_cuis,
                    output_dir=str(self.output_dir),
                    dataset_name=self.dataset_name
                )
                self.logger.info("Enhanced mappings with TTY information generated successfully")
            except Exception as e:
                self.logger.warning(f"Failed to generate comprehensive mappings: {e}")
                self.logger.info("Continuing with basic mappings only")
        
        # If we are using a subdir suffix, mirror enhanced mapping files into mappings_new
        if comprehensive_mapping_files and self.subdir_suffix:
            try:
                for key, src_path in list(comprehensive_mapping_files.items()):
                    if not src_path:
                        continue
                    src = Path(src_path)
                    if src.exists():
                        dst = self.mappings_dir / src.name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.copy2(src, dst)
                            comprehensive_mapping_files[key] = str(dst)
                        except Exception as copy_err:
                            self.logger.warning(f"Failed to copy mapping file {src} to {dst}: {copy_err}")
            except Exception as e:
                self.logger.warning(f"Error mirroring mapping files to {self.mappings_dir}: {e}")
        
        # Combine all mapping files
        all_mapping_files = {**basic_mapping_files, **comprehensive_mapping_files}
        
        return stats, all_mapping_files
    
    def process(self) -> Dict[str, Any]:
        """
        Main processing method that orchestrates the entire pipeline.
        
        Returns:
            Dictionary with processing results and statistics
        """
        self.logger.info(f"Starting processing of {self.dataset_name} dataset")
        
        try:
            # Step 1: Load raw data
            self.logger.info("Step 1: Loading raw data...")
            raw_data = self._load_raw_data()
            
            # Apply limit if specified (for testing) - BEFORE expensive operations
            if self.limit and len(raw_data) > self.limit:
                self.logger.info(f"TEST MODE: Limiting to {self.limit} documents (was {len(raw_data)})")
                raw_data = raw_data[:self.limit]
            
            # Step 2: Extract mentions
            self.logger.info("Step 2: Extracting mentions...")
            mentions = self._extract_mentions(raw_data)
            
            # Step 3: Map to UMLS
            self.logger.info("Step 3: Mapping to UMLS...")
            mapped_mentions = self._map_to_umls(mentions)
            
            # Step 4: Convert to standard format
            self.logger.info("Step 4: Converting to standard format...")
            documents = self._convert_to_standard_format(mapped_mentions)
            
            # Step 5: Save documents
            self.logger.info("Step 5: Saving documents...")
            self._save_documents(documents)
            
            # Step 6: Generate statistics
            self.logger.info("Step 6: Generating statistics...")
            stats, mapping_files = self._generate_statistics(documents)
            
            # Clean up
            self.umls_mapper.close()
            
            self.logger.info(f"Processing completed successfully for {self.dataset_name}")
            
            return {
                'dataset_name': self.dataset_name,
                'status': 'success',
                'documents_count': len(documents),
                'total_mentions': stats['total_mentions'],
                'mapped_mentions': stats['mapped_mentions'],
                'mapping_success_rate': stats.get('mapping_success_rate', 0),
                'output_files': {
                    'documents': str(self.documents_dir / f"{self.dataset_name}.jsonl"),
                    'statistics': str(self.stats_dir / f"{self.dataset_name}_statistics.json"),
                    'mappings': mapping_files
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {self.dataset_name}: {str(e)}")
            self.umls_mapper.close()
            return {
                'dataset_name': self.dataset_name,
                'status': 'error',
                'error': str(e)
            }
    
 