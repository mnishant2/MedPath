#!/usr/bin/env python3
"""
Mantra-GSC dataset processor.

Processes the Mantra-GSC corpus which contains biomedical entity mentions
mapped to UMLS CUIs directly in BRAT format with .txt and .ann files.
"""

import os
import json
from typing import List, Dict, Any
from pathlib import Path
import re

from .base_processor import BaseProcessor

class MantraGSCProcessor(BaseProcessor):
    """Processor for the Mantra-GSC dataset."""
    
    def _get_native_ontologies(self) -> List[str]:
        """Mantra-GSC uses UMLS CUIs directly."""
        return ['UMLS']
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load Mantra-GSC corpus files."""
        self.logger.info("Loading Mantra-GSC corpus from BRAT files...")
        
        english_dir = self.data_dir / 'English'
        
        if not english_dir.exists():
            self.logger.error(f"Mantra-GSC English directory not found: {english_dir}")
            return []
        
        documents = []
        
        # Get all subdirectories in the English folder
        subdirs = [d for d in english_dir.iterdir() if d.is_dir()]
        self.logger.info(f"Found {len(subdirs)} subdirectories in English folder")
        
        # Process each subdirectory
        for subdir in subdirs:
            self.logger.info(f"Processing subdirectory: {subdir.name}")
            
            # Get all txt files
            txt_files = list(subdir.glob('*.txt'))
            self.logger.info(f"Found {len(txt_files)} text files in {subdir.name}")
            
            # Process each txt file and its corresponding ann file
            for txt_file in txt_files:
                ann_file = txt_file.with_suffix('.ann')
                if not ann_file.exists():
                    self.logger.warning(f"No annotation file found for {txt_file.name}")
                    continue
                
                try:
                    # Read text content
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    
                    # Parse annotation file
                    annotations = self._parse_brat_annotation_file(ann_file, text_content, txt_file)
                    
                    # Create document with annotations
                    doc = {
                        'doc_id': f"{subdir.name}_{txt_file.stem}",
                        'text': text_content,
                        'annotations': annotations,
                        'dataset': subdir.name,
                        'source_file': txt_file.name
                    }
                    documents.append(doc)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {txt_file.name}: {e}")
                    continue
        
        self.logger.info(f"Loaded {len(documents)} documents from Mantra-GSC corpus")
        return documents
    
    def _extract_mentions(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract mentions from Mantra-GSC documents."""
        self.logger.info("Extracting mentions from Mantra-GSC documents...")
        
        mentions = []
        
        for doc in raw_data:
            doc_id = doc['doc_id']
            doc_text = doc['text']
            dataset = doc.get('dataset', 'unknown')
            source_file = doc.get('source_file', '')
            
            for annotation in doc.get('annotations', []):
                mention = {
                    'doc_id': doc_id,
                    'doc_text': doc_text,
                    'start': annotation['start'],
                    'end': annotation['end'],
                    'text': annotation['mention_text'],
                    'native_id': annotation['cui'],
                    'native_ontology_name': 'UMLS',
                    'entity_type': 'Biomedical Entity',
                    'dataset': dataset,
                    'source_file': source_file
                }
                mentions.append(mention)
        
        self.logger.info(f"Extracted {len(mentions)} mentions from Mantra-GSC documents")
        return mentions
    
    def _parse_brat_annotation_file(self, ann_file: Path, text_content: str, txt_file: Path) -> List[Dict[str, Any]]:
        """Parse BRAT annotation file and extract annotations."""
        annotations = []
        
        with open(ann_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or not line.startswith('T'):
                    continue
                
                try:
                    parts = line.split('\t')
                    if len(parts) < 3:
                        continue
                    
                    mention_id = parts[0]
                    annotation_info = parts[1]
                    mention_text = parts[2]
                    
                    # Parse annotation info: "CUI start end"
                    info_parts = annotation_info.split()
                    if len(info_parts) < 3:
                        continue
                    
                    cui = info_parts[0]
                    start = int(info_parts[1])
                    end = int(info_parts[2])
                    
                    # Validate spans
                    if start < 0 or end > len(text_content) or start >= end:
                        continue
                    
                    annotation = {
                        'mention_id': mention_id,
                        'mention_text': mention_text,
                        'start': start,
                        'end': end,
                        'cui': cui
                    }
                    
                    annotations.append(annotation)
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing line {line_num} in {ann_file}: {e}")
                    continue
        
        return annotations
    
    def _extract_sentence(self, text: str, start: int, end: int) -> str:
        """Extract the sentence containing the mention span."""
        try:
            # Find the start of the sentence
            sentence_start = start
            while sentence_start > 0 and text[sentence_start-1] not in '.!?':
                sentence_start -= 1
            
            # Find the end of the sentence
            sentence_end = end
            while sentence_end < len(text) and text[sentence_end] not in '.!?':
                sentence_end += 1
                
            return text[sentence_start:sentence_end].strip()
        except Exception as e:
            self.logger.warning(f"Error extracting sentence: {e}")
            return text[start:end]  # Return just the mention if sentence extraction fails
    
    def _map_single_mention(self, mention: Dict[str, Any]) -> str:
        """Map a single mention to UMLS CUI with proper validation."""
        # Mantra-GSC has CUIs that may be from older versions and need validation
        cui = mention.get('native_id', '')
        mention_text = mention.get('text', '')
        
        # First try to validate the existing CUI
        if cui and cui.strip():
            cui = cui.strip()
            
            # Basic CUI format check
            if cui.startswith('C') and len(cui) >= 7:
                # Try to validate the CUI
                if hasattr(self.umls_mapper, 'validate_cui'):
                    if hasattr(self.umls_mapper, 'api_key'):  # API-based mapper
                        validated_cui = self.umls_mapper.validate_cui(cui)
                        if validated_cui and isinstance(validated_cui, str):
                            mention['mapping_method'] = 'native_id_mapping'
                            return validated_cui
                    else:  # Local mapper
                        is_valid = self.umls_mapper.validate_cui(cui)
                        if is_valid:
                            mention['mapping_method'] = 'native_id_mapping'
                            return cui
                else:
                    # Fallback - try to get semantic types to validate existence
                    try:
                        semantic_types = self.umls_mapper.get_semantic_types(cui)
                        if semantic_types:
                            mention['mapping_method'] = 'native_id_mapping'
                            return cui
                    except:
                        pass
        
        # If CUI validation fails, try text-based mapping as fallback
        if mention_text and mention_text.strip():
            text_cui = self.umls_mapper.get_cui_from_text(mention_text.strip())
            if text_cui:
                method = getattr(self.umls_mapper, 'last_text_mapping_method', None)
                mention['mapping_method'] = method if method in {'exact_match', 'semantic_containment'} else 'text_fallback'
                return text_cui
        
        # If no mapping found, return empty string
        mention['mapping_method'] = 'no_mapping'
        return ''
    
    def _save_documents(self, documents: List[Dict]):
        """Save documents - treat all as one dataset, no sub-grouping."""
        # Save all documents to single combined file in documents directory
        combined_file = self.documents_dir / f"{self.dataset_name}.jsonl"
        with open(combined_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        self.logger.info(f"Saved {len(documents)} documents to {combined_file}") 