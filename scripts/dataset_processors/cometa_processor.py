#!/usr/bin/env python3
"""
COMETA dataset processor.

Processes the COMETA corpus which contains medical terms with SNOMED CT codes
in CSV format with train/dev/test splits.
"""

import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import re
from tqdm import tqdm

from .base_processor import BaseProcessor

class CometaProcessor(BaseProcessor):
    """Processor for the COMETA dataset."""

    def _get_native_ontologies(self) -> List[str]:
        """COMETA uses SNOMED CT codes."""
        return ['SNOMEDCT_US']
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load COMETA corpus CSV files."""
        self.logger.info("Loading COMETA corpus from CSV files...")
        
        splits_path = self.data_dir / 'splits' / 'random'
        
        if not splits_path.exists():
            self.logger.error(f"COMETA splits directory not found: {splits_path}")
            return []
        
        documents = []
        
        # Process each split
        for split in ['train', 'dev', 'test']:
            csv_file = splits_path / f'{split}.csv'
            if not csv_file.exists():
                self.logger.warning(f"COMETA {split} file not found: {csv_file}")
                continue
            
            self.logger.info(f"Processing {split} split...")
            
            try:
                df = pd.read_csv(csv_file, sep='\t')
                self.logger.info(f"Loaded {len(df)} rows from {split} split")
                
                for idx, row in df.iterrows():
                    # Extract mention information
                    mention_id = str(row['ID'])
                    term = str(row['Term'])
                    sentence = str(row['Example'])
                    
                    # Extract BOTH specific and general SNOMED IDs (like utils/data_scripts/cometa)
                    specific_id = str(row.get('Specific SNOMED ID', ''))
                    general_id = str(row.get('General SNOMED ID', ''))
                    
                    # Clean the IDs
                    specific_id = specific_id if specific_id and specific_id != 'nan' else ''
                    general_id = general_id if general_id and general_id != 'nan' else ''
                    
                    # Find span of mention in sentence
                    start, end = self._find_mention_span(sentence, term)
                    
                    # Create document with single mention - store BOTH IDs for fallback mapping
                    doc = {
                        'doc_id': f"cometa_{split}_{mention_id}",
                        'text': sentence,
                        'annotations': [{
                            'mention_id': mention_id,
                            'mention_text': term,
                            'start': start,
                            'end': end,
                            'specific_snomed_id': specific_id,
                            'general_snomed_id': general_id
                        }],
                        'split': split  # Keep the actual split name
                    }
                    
                    documents.append(doc)
                    
            except Exception as e:
                self.logger.error(f"Error processing {split} split: {e}")
                continue
        
        self.logger.info(f"Loaded {len(documents)} documents from COMETA corpus")
        return documents
    
    def _extract_mentions(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract mentions from COMETA documents."""
        self.logger.info("Extracting mentions from COMETA documents...")
        
        mentions = []
        
        for doc in raw_data:
            doc_id = doc['doc_id']
            doc_text = doc['text']
            split = doc.get('split', 'unknown')
            
            for annotation in doc.get('annotations', []):
                mention = {
                    'doc_id': doc_id,
                    'doc_text': doc_text,
                    'start': annotation['start'],
                    'end': annotation['end'],
                    'text': annotation['mention_text'],
                    'specific_snomed_id': annotation.get('specific_snomed_id', ''),
                    'general_snomed_id': annotation.get('general_snomed_id', ''),
                    'native_ontology_name': 'SNOMEDCT_US',
                    'entity_type': 'Medical Term',
                    'split': split
                }
                mentions.append(mention)
        
        self.logger.info(f"Extracted {len(mentions)} mentions from COMETA documents")
        return mentions
    
    def _find_mention_span(self, text: str, mention: str) -> tuple:
        """Find the span of mention in text."""
        # Ensure both are strings
        text = str(text)
        mention = str(mention)
        
        # Find all occurrences of the mention in the text
        matches = list(re.finditer(re.escape(mention), text, re.IGNORECASE))
        
        if not matches:
            # If no exact match, try to find the closest match
            mention_lower = mention.lower()
            text_lower = text.lower()
            if mention_lower in text_lower:
                start = text_lower.find(mention_lower)
                end = start + len(mention)
                return start, end
            return 0, len(mention)
        
        # Use the first occurrence
        match = matches[0]
        return match.start(), match.end()
    
    def _map_single_mention(self, mention: Dict[str, Any]) -> str:
        """Map a single mention to UMLS CUI with specific → general → text fallback strategy."""
        mention_text = mention['text']
        specific_snomed_id = mention.get('specific_snomed_id', '')
        general_snomed_id = mention.get('general_snomed_id', '')
        
        # First try to map specific SNOMED ID to UMLS CUI
        if specific_snomed_id and specific_snomed_id.strip():
            snomed_id = specific_snomed_id.strip()
            
            # Try multiple SNOMED sources for specific ID
            snomed_sources = ['SNOMEDCT_US', 'SCTSPA', 'SNOMEDCT', 'SNOMEDCT_CORE']
            
            for source in snomed_sources:
                cui = self.umls_mapper.get_cui_from_ontology_id(snomed_id, source)
                if cui:
                    mention['mapping_method'] = 'native_id_mapping'
                    return cui
            
            # Check if specific ID might be a CUI
            if snomed_id.startswith('C') and len(snomed_id) >= 7:
                if hasattr(self.umls_mapper, 'validate_cui'):
                    if hasattr(self.umls_mapper, 'api_key'):  # API-based mapper
                        validated_cui = self.umls_mapper.validate_cui(snomed_id)
                        if validated_cui and isinstance(validated_cui, str):
                            mention['mapping_method'] = 'native_id_mapping'
                            return validated_cui
                    else:  # Local mapper
                        is_valid = self.umls_mapper.validate_cui(snomed_id)
                        if is_valid:
                            mention['mapping_method'] = 'native_id_mapping'
                            return snomed_id
        
        # If specific SNOMED mapping fails, try general SNOMED ID as fallback
        if general_snomed_id and general_snomed_id.strip():
            snomed_id = general_snomed_id.strip()
            
            # Try multiple SNOMED sources for general ID
            snomed_sources = ['SNOMEDCT_US', 'SCTSPA', 'SNOMEDCT', 'SNOMEDCT_CORE']
            
            for source in snomed_sources:
                cui = self.umls_mapper.get_cui_from_ontology_id(snomed_id, source)
                if cui:
                    mention['mapping_method'] = 'native_id_mapping'
                    return cui
            
            # Check if general ID might be a CUI
            if snomed_id.startswith('C') and len(snomed_id) >= 7:
                if hasattr(self.umls_mapper, 'validate_cui'):
                    if hasattr(self.umls_mapper, 'api_key'):  # API-based mapper
                        validated_cui = self.umls_mapper.validate_cui(snomed_id)
                        if validated_cui and isinstance(validated_cui, str):
                            mention['mapping_method'] = 'native_id_mapping'
                            return validated_cui
                    else:  # Local mapper
                        is_valid = self.umls_mapper.validate_cui(snomed_id)
                        if is_valid:
                            mention['mapping_method'] = 'native_id_mapping'
                            return snomed_id
        
        # If both SNOMED mappings fail, try text-based mapping as final fallback
        if mention_text and mention_text.strip():
            cui = self.umls_mapper.get_cui_from_text(mention_text.strip())
            if cui:
                method = getattr(self.umls_mapper, 'last_text_mapping_method', None)
                mention['mapping_method'] = method if method in {'exact_match', 'semantic_containment'} else 'text_fallback'
                return cui
        
        # If no mapping found, return empty string
        mention['mapping_method'] = 'no_mapping'
        return ''
    
    def _create_mapping_key(self, mention: Dict) -> str:
        """Create mapping key specific to COMETA dataset with both SNOMED IDs."""
        # COMETA has both specific and general SNOMED IDs
        specific_id = mention.get('specific_snomed_id', '')
        general_id = mention.get('general_snomed_id', '')
        text = mention.get('text', '')
        
        key_parts = []
        if specific_id and specific_id != '':
            key_parts.append(f"specific:{specific_id}")
        if general_id and general_id != '':
            key_parts.append(f"general:{general_id}")
        if text:
            key_parts.append(f"text:{text}")
        
        return "|".join(key_parts)

    def _convert_to_standard_format(self, mentions: List[Dict]) -> List[Dict]:
        """Convert mentions to standardized document format with split preservation."""
        self.logger.info("Converting to standardized format...")
        
        # Group mentions by document
        documents_dict = {}
        
        for mention in mentions:
            doc_id = mention['doc_id']
            
            if doc_id not in documents_dict:
                documents_dict[doc_id] = {
                    'doc_id': doc_id,
                    'text': mention['doc_text'],
                    'mentions': [],
                    'split': mention.get('split', 'unknown')  # Preserve split information
                }
            
            # Create standardized mention
            standardized_mention = {
                'start': mention['start'],
                'end': mention['end'],
                'text': mention['text'],
                'native_id': mention.get('specific_snomed_id', '') or mention.get('general_snomed_id', ''),
                'native_ontology_name': mention.get('native_ontology_name', ''),
                'cui': mention.get('cui', ''),
                'umls_name': mention.get('umls_name', ''),
                'semantic_type': mention.get('semantic_type', ''),
                # Keep COMETA-specific fields for tracking
                'specific_snomed_id': mention.get('specific_snomed_id', ''),
                'general_snomed_id': mention.get('general_snomed_id', ''),
                'mapping_method': mention.get('mapping_method', '')
            }
            
            documents_dict[doc_id]['mentions'].append(standardized_mention)
        
        documents = list(documents_dict.values())
        self.logger.info(f"Created {len(documents)} documents with mentions")
        
        return documents

    def _save_documents(self, documents: List[Dict]):
        """Save documents with split information.""" 
        # Group documents by split
        split_documents = {}
        for doc in documents:
            split = doc.get('split', 'unknown')
            if split not in split_documents:
                split_documents[split] = []
            split_documents[split].append(doc)
        
        # Save each split separately to documents directory
        for split, split_docs in split_documents.items():
            if split_docs:
                split_file = self.documents_dir / f"{self.dataset_name}_{split}.jsonl"
                with open(split_file, 'w', encoding='utf-8') as f:
                    for doc in split_docs:
                        # Remove split info from the saved document
                        doc_to_save = doc.copy()
                        if 'split' in doc_to_save:
                            del doc_to_save['split']
                        f.write(json.dumps(doc_to_save, ensure_ascii=False) + '\n')
                self.logger.info(f"Saved {len(split_docs)} documents to {split_file}")

    def _perform_umls_mapping(self, unique_combinations):
        """
        Perform UMLS mapping for unique combinations.
        Strategy: text mapping first for specific term → then fall back to general → if both code and text don't work
        """
        mapping_results = {}
        
        for combination in tqdm(unique_combinations, desc="Mapping to UMLS"):
            mention_text, specific_snomed_id, general_snomed_id = combination
            
            # Initialize mapping result
            mapping_result = {
                'cui': None,
                'mapping_source': 'unmapped',
                'originally_cui_less': specific_snomed_id == '' and general_snomed_id == '',
                'originally_no_concept': specific_snomed_id == '' and general_snomed_id == '',
                'cui_validation_success': False,
                'fallback_success': False
            }
            
            # Strategy: text mapping first for specific term → then fall back to general
            try:
                # Step 1: Try text mapping for specific term (most specific first)
                if specific_snomed_id:
                    cui = self.umls_mapper.get_cui_from_text(mention_text)
                    if cui:
                        mapping_result['cui'] = cui
                        mapping_result['mapping_source'] = 'text_mapping'
                        mapping_result['fallback_success'] = True
                        mapping_results[combination] = mapping_result
                        continue
                
                # Step 2: Try ontology mapping for specific SNOMED ID
                if specific_snomed_id:
                    cui = self.umls_mapper.get_cui_from_ontology_id(specific_snomed_id, "SNOMEDCT_US")
                    if cui:
                        mapping_result['cui'] = cui
                        mapping_result['mapping_source'] = 'specific_snomed'
                        mapping_results[combination] = mapping_result
                        continue
                
                # Step 3: Fall back to general - try text mapping first
                if general_snomed_id:
                    cui = self.umls_mapper.get_cui_from_text(mention_text)
                    if cui:
                        mapping_result['cui'] = cui
                        mapping_result['mapping_source'] = 'text_mapping'
                        mapping_result['fallback_success'] = True
                        mapping_results[combination] = mapping_result
                        continue
                
                # Step 4: Try ontology mapping for general SNOMED ID
                if general_snomed_id:
                    cui = self.umls_mapper.get_cui_from_ontology_id(general_snomed_id, "SNOMEDCT_US")
                    if cui:
                        mapping_result['cui'] = cui
                        mapping_result['mapping_source'] = 'general_snomed'
                        mapping_result['fallback_success'] = True
                        mapping_results[combination] = mapping_result
                        continue
                
                # Step 5: Final fallback - try text mapping without any SNOMED context
                if mention_text:
                    cui = self.umls_mapper.get_cui_from_text(mention_text)
                    if cui:
                        mapping_result['cui'] = cui
                        mapping_result['mapping_source'] = 'text_mapping'
                        mapping_result['fallback_success'] = True
                        mapping_results[combination] = mapping_result
                        continue
                
                # No mapping found
                mapping_results[combination] = mapping_result
                
            except Exception as e:
                print(f"Error mapping combination {combination}: {e}")
                mapping_results[combination] = mapping_result
        
        return mapping_results
