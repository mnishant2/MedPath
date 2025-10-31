#!/usr/bin/env python3
"""
CDR (Chemical-Disease Relations) dataset processor.

Processes the CDR corpus which contains PubMed abstracts annotated with 
chemicals and diseases along with their MeSH IDs.
"""

import os
import re
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
import json

from .base_processor import BaseProcessor

class CDRProcessor(BaseProcessor):
    """Processor for the CDR dataset."""
    
    def _get_native_ontologies(self) -> List[str]:
        """CDR uses MeSH (Medical Subject Headings) IDs."""
        return ['MSH']
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load CDR corpus files in PubTator format."""
        self.logger.info("Loading CDR corpus data...")
        
        # CDR has train, dev, and test files
        datasets = ['train', 'dev', 'test']
        file_mapping = {
            'train': 'CDR_TrainingSet.PubTator.txt',
            'dev': 'CDR_DevelopmentSet.PubTator.txt', 
            'test': 'CDR_TestSet.PubTator.txt'
        }
        
        raw_data = []
        
        for dataset_name in datasets:
            file_path = self.data_dir / file_mapping[dataset_name]
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                continue
                
            self.logger.info(f"Loading {dataset_name} data from {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by empty lines to get documents
            documents = content.strip().split('\n\n')
            
            for doc_text in documents:
                if not doc_text.strip():
                    continue
                
                doc_data = self._parse_pubtator_document(doc_text, dataset_name)
                if doc_data:
                    raw_data.append(doc_data)
        
        self.logger.info(f"Loaded {len(raw_data)} documents from CDR corpus")
        return raw_data
    
    def _parse_pubtator_document(self, doc_text: str, split: str) -> Dict[str, Any]:
        """Parse a single PubTator format document."""
        lines = doc_text.strip().split('\n')
        
        doc_id = None
        title = ""
        abstract = ""
        annotations = []
        
        for line in lines:
            if '|t|' in line:  # Title line
                parts = line.split('|t|', 1)
                doc_id = parts[0].strip()
                title = parts[1].strip() if len(parts) > 1 else ""
            elif '|a|' in line:  # Abstract line
                parts = line.split('|a|', 1)
                abstract = parts[1].strip() if len(parts) > 1 else ""
            elif '\t' in line and doc_id:  # Annotation line
                annotation = self._parse_annotation_line(line, doc_id)
                if annotation:
                    annotations.append(annotation)
        
        if not doc_id:
            return None
        
        # Combine title and abstract
        full_text = title
        if abstract:
            full_text += " " + abstract
        
        return {
            'doc_id': doc_id,
            'title': title,
            'abstract': abstract,
            'text': full_text,
            'annotations': annotations,
            'split': split
        }
    
    def _parse_annotation_line(self, line: str, doc_id: str) -> Dict[str, Any]:
        """Parse an annotation line from PubTator format."""
        parts = line.strip().split('\t')
        
        if len(parts) < 6:
            return None
        
        try:
            pmid = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            mention_text = parts[3]
            entity_type = parts[4]  # Chemical or Disease
            mesh_id = parts[5]
            # Handle composite IDs separated by '|'
            mesh_ids = [mid.strip() for mid in str(mesh_id).split('|') if mid and mid.strip() and mid.strip() != '-']
            
            return {
                'doc_id': doc_id,
                'start': start,
                'end': end,
                'text': mention_text,
                'entity_type': entity_type,
                'mesh_id': mesh_id,
                'mesh_ids': mesh_ids if mesh_ids else ([mesh_id] if mesh_id else [])
            }
        except (ValueError, IndexError):
            self.logger.warning(f"Could not parse annotation line: {line}")
            return None
    
    def _extract_mentions(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract mentions from CDR documents."""
        self.logger.info("Extracting mentions from CDR documents...")
        
        mentions = []
        
        for doc in tqdm(raw_data, desc="Extracting mentions from documents"):
            doc_id = doc['doc_id']
            doc_text = doc['text']
            
            for annotation in doc['annotations']:
                mesh_ids = annotation.get('mesh_ids', []) or ([annotation.get('mesh_id')] if annotation.get('mesh_id') else [])
                # Create one mention per mesh id for composite annotations
                for m_id in mesh_ids:
                    mention = {
                        'doc_id': doc_id,
                        'doc_text': doc_text,
                        'start': annotation['start'],
                        'end': annotation['end'],
                        'text': annotation['text'],
                        'native_id': m_id,
                        'native_ontology_name': 'MSH',
                        'entity_type': annotation['entity_type'],
                        'split': doc.get('split', 'unknown')
                    }
                    mentions.append(mention)
        
        self.logger.info(f"Extracted {len(mentions)} mentions")
        return mentions
    
    def _create_mapping_key(self, mention: Dict) -> str:
        """Create mapping key specific to CDR dataset."""
        # CDR has MeSH IDs, so prioritize those
        mesh_id = mention.get('native_id', '')
        text = mention.get('text', '')
        entity_type = mention.get('entity_type', '')
        
        key_parts = []
        if mesh_id and mesh_id != '' and mesh_id != '-':
            key_parts.append(f"mesh:{mesh_id}")
        if entity_type:
            key_parts.append(f"type:{entity_type}")
        if text:
            key_parts.append(f"text:{text}")
        
        return "|".join(key_parts)
    
    def _convert_to_standard_format(self, mentions: List[Dict]) -> List[Dict]:
        """Convert CDR mentions to standardized document format."""
        self.logger.info("Converting to standardized format...")
        
        # Group mentions by document
        documents_dict = {}
        
        for mention in mentions:
            doc_id = mention['doc_id']
            
            if doc_id not in documents_dict:
                documents_dict[doc_id] = {
                    'doc_id': doc_id,
                    'text': mention['doc_text'],
                    'split': mention.get('split', 'unknown'),  # Add split information
                    'mentions': []
                }
            
            # Create standardized mention
            standardized_mention = {
                'start': mention['start'],
                'end': mention['end'],
                'text': mention['text'],
                'native_id': mention.get('native_id', ''),
                'native_ontology_name': 'MSH',
                'cui': mention.get('cui', ''),
                'umls_name': mention.get('umls_name', ''),
                'semantic_type': mention.get('semantic_type', ''),
                'entity_type': mention.get('entity_type', ''),  # Keep CDR-specific field
                'mapping_method': mention.get('mapping_method', '')
            }
            
            documents_dict[doc_id]['mentions'].append(standardized_mention)
        
        documents = list(documents_dict.values())
        self.logger.info(f"Created {len(documents)} documents with mentions")
        
        return documents

    def _save_documents(self, documents: List[Dict]):
        """Save documents by split (train/dev/test)."""
        self.logger.info("Saving documents by split...")
        
        # Group documents by split
        split_documents = {}
        for doc in documents:
            split = doc.get('split', 'unknown')
            if split not in split_documents:
                split_documents[split] = []
            split_documents[split].append(doc)
        
        # Save each split separately
        for split, split_docs in split_documents.items():
            output_file = self.documents_dir / f'cdr_{split}.jsonl'
            
            self.logger.info(f"Saving {len(split_docs)} {split} documents to {output_file}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for doc in split_docs:
                    f.write(json.dumps(doc) + '\n')
        
        self.logger.info("Documents saved successfully") 