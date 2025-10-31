#!/usr/bin/env python3
"""
ShareCLEF dataset processor.

Processes the ShareCLEF corpus which contains clinical cases annotated with
UMLS CUIs. This is a licensed dataset with train/test splits.
"""

import os
import json
from typing import List, Dict, Any
from pathlib import Path

from .base_processor import BaseProcessor

class ShareCLEFProcessor(BaseProcessor):
    """Processor for the ShareCLEF dataset (licensed)."""
    
    def _get_native_ontologies(self) -> List[str]:
        """ShareCLEF already has UMLS CUIs."""
        return ['UMLS']
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load ShareCLEF corpus files."""
        self.logger.info("Loading ShareCLEF corpus data...")
        
        # This is a licensed dataset - user must provide the data
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"ShareCLEF data directory not found: {self.data_dir}\n"
                "This is a licensed dataset. Please obtain access and place the files in the data directory."
            )
        
        raw_data = []
        
        # Look for training_corpus and test_corpus directories
        for corpus_dir in ['training_corpus', 'test_corpus']:
            corpus_path = self.data_dir / corpus_dir
            if corpus_path.exists():
                split_name = 'train' if corpus_dir == 'training_corpus' else 'test'
                
                # Look for text and annotation directories
                text_dir = corpus_path / 'train' if split_name == 'train' else corpus_path / 'test'
                ann_dir = corpus_path / 'train_ann' if split_name == 'train' else corpus_path / 'test_ann'
                
                if text_dir.exists() and ann_dir.exists():
                    self.logger.info(f"Processing {split_name} split from {corpus_path}")
                    split_data = self._process_split_directories(text_dir, ann_dir, split_name)
                    raw_data.extend(split_data)
        
        self.logger.info(f"Loaded {len(raw_data)} documents from ShareCLEF corpus")
        return raw_data
    
    def _process_split_directories(self, text_dir: Path, ann_dir: Path, split_name: str) -> List[Dict[str, Any]]:
        """Process text and annotation directories for a split."""
        documents = []
        
        # Load all text files
        text_files = list(text_dir.glob("*.txt"))
        
        for text_file in text_files:
            # Read text content
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            except Exception as e:
                self.logger.warning(f"Error reading text file {text_file}: {e}")
                continue
            
            # Look for corresponding annotation file
            ann_file = ann_dir / f"{text_file.stem}.pipe.txt"
            annotations = []
            
            if ann_file.exists():
                try:
                    annotations = self._parse_pipe_annotations(ann_file, text_content)
                except Exception as e:
                    self.logger.warning(f"Error parsing annotation file {ann_file}: {e}")
            
            documents.append({
                'doc_id': text_file.stem,
                'text': text_content,
                'annotations': annotations,
                'split': split_name
            })
        
        return documents
    
    def _parse_pipe_annotations(self, ann_file: Path, text_content: str) -> List[Dict[str, Any]]:
        """Parse pipe-separated annotation file."""
        annotations = []
        
        with open(ann_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # The format repeats: filename|spans|CUI|...fields...|filename|spans|CUI|...
        # Let's find all filename occurrences and extract annotations
        filename_base = ann_file.stem.replace('.pipe', '')
        
        # Split by the filename pattern
        parts = content.split(filename_base + '.txt|')
        
        for part in parts:
            if not part.strip():
                continue
            
            # Each part starts with spans|CUI|...
            annotation_parts = part.split('|')
            if len(annotation_parts) < 2:
                continue
            
            spans = annotation_parts[0]
            cui = annotation_parts[1]
            
            # Skip CUI-less entries
            if cui == 'CUI-less' or cui == 'NULL':
                continue
            
            # Parse spans (can be comma-separated or range like 98-101)
            span_parts = spans.split(',')
            for span_part in span_parts:
                span_part = span_part.strip()
                if '-' in span_part:
                    try:
                        start, end = map(int, span_part.split('-'))
                        
                        # Extract mention text
                        mention_text = ''
                        if 0 <= start < end <= len(text_content):
                            mention_text = text_content[start:end]
                        
                        if mention_text:
                            annotations.append({
                                'start': start,
                                'end': end,
                                'text': mention_text,
                                'cui': cui
                            })
                    except ValueError:
                        continue
        
        return annotations
    
    def _extract_mentions(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract mentions from ShareCLEF documents."""
        self.logger.info("Extracting mentions from ShareCLEF documents...")
        
        mentions = []
        
        for doc in raw_data:
            doc_id = doc.get('doc_id', '')
            doc_text = doc.get('text', '')
            annotations = doc.get('annotations', [])
            split = doc.get('split', '')
            
            for annotation in annotations:
                start = annotation['start']
                end = annotation['end']
                mention_text = annotation['text']
                cui = annotation['cui']
                
                mention = {
                    'doc_id': doc_id,
                    'doc_text': doc_text,
                    'start': start,
                    'end': end,
                    'text': mention_text,
                    'native_id': cui,
                    'native_ontology_name': 'UMLS',
                    'entity_type': 'Clinical Entity',
                    'split': split,
                    'annotation_id': f"{doc_id}_{start}_{end}"
                }
                
                mentions.append(mention)
        
        self.logger.info(f"Extracted {len(mentions)} mentions from ShareCLEF documents")
        return mentions
    
    def _create_mapping_key(self, mention: Dict) -> str:
        """Create a unique key for UMLS mapping cache."""
        return f"{mention['native_id']}|{mention['text']}|{mention['entity_type']}"
    
    def _map_single_mention(self, mention: Dict) -> str:
        """Map a single mention to UMLS CUI with proper validation."""
        # ShareCLEF has UMLS CUIs that may be from older versions and need validation
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
    
    def _convert_to_standard_format(self, mentions: List[Dict]) -> List[Dict]:
        """Convert mentions to standard format."""
        documents = {}
        
        for mention in mentions:
            doc_id = mention['doc_id']
            
            if doc_id not in documents:
                documents[doc_id] = {
                    'doc_id': doc_id,
                    'text': mention['doc_text'],
                    'mentions': [],
                    'split': mention.get('split', '')
                }
            
            mention_entry = {
                'start': mention['start'],
                'end': mention['end'],
                'text': mention['text'],
                'cui': mention.get('cui', ''),
                'umls_name': mention.get('umls_name', ''),
                'native_id': mention.get('native_id', ''),
                'native_ontology_name': mention.get('native_ontology_name', ''),
                'entity_type': mention.get('entity_type', ''),
                'annotation_id': mention.get('annotation_id', ''),
                'mapping_method': mention.get('mapping_method', '')
            }
            
            documents[doc_id]['mentions'].append(mention_entry)
        
        return list(documents.values())
    
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
                self.logger.info(f"Saving {len(split_docs)} {split} documents to {split_file}")
                
                with open(split_file, 'w', encoding='utf-8') as f:
                    for doc in split_docs:
                        # Remove split info from the saved document
                        doc_to_save = doc.copy()
                        if 'split' in doc_to_save:
                            del doc_to_save['split']
                        f.write(json.dumps(doc_to_save, ensure_ascii=False) + '\n')
