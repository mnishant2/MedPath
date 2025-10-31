#!/usr/bin/env python3
"""
MedMentions dataset processor.

Processes the MedMentions corpus which contains PubMed abstracts annotated with 
biomedical entity mentions and their UMLS CUIs (pre-mapped).
"""

import os
import re
import pandas as pd
import json
from typing import List, Dict, Any
from pathlib import Path

from .base_processor import BaseProcessor

class MedMentionsProcessor(BaseProcessor):
    """Processor for the MedMentions dataset."""

    def _get_native_ontologies(self) -> List[str]:
        """MedMentions already has UMLS CUIs (no native ontology mapping needed)."""
        return ['UMLS']
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load MedMentions corpus file and split files."""
        self.logger.info("Loading MedMentions corpus data...")
        
        # Load PMID split files
        train_pmids = self._load_pmids('corpus_pubtator_pmids_trng.txt')
        dev_pmids = self._load_pmids('corpus_pubtator_pmids_dev.txt')
        test_pmids = self._load_pmids('corpus_pubtator_pmids_test.txt')
        
        self.logger.info(f"Loaded PMIDs: {len(train_pmids)} train, {len(dev_pmids)} dev, {len(test_pmids)} test")
        
        # Parse main corpus file
        corpus_file = self.data_dir / 'corpus_pubtator.txt'
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        documents = self._parse_corpus(corpus_file, train_pmids, dev_pmids, test_pmids)
        
        self.logger.info(f"Loaded {len(documents)} documents from MedMentions corpus")
        return documents
    
    def _load_pmids(self, filename: str) -> set:
        """Load PMIDs from a file into a set."""
        pmid_file = self.data_dir / filename
        if not pmid_file.exists():
            self.logger.warning(f"PMID file not found: {pmid_file}")
            return set()
        
        with open(pmid_file, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}
    
    def _parse_corpus(self, corpus_file: Path, train_pmids: set, dev_pmids: set, test_pmids: set) -> List[Dict[str, Any]]:
        """Parse the MedMentions corpus file."""
        self.logger.info("Parsing MedMentions corpus file...")
        
        documents = {}
        current_pmid = None
        
        # First pass: collect titles and abstracts
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    current_pmid = None
                    continue
                    
                if '|t|' in line:  # Title line
                    parts = line.split('|t|', 1)
                    current_pmid = parts[0].strip()
                    title = parts[1].strip()
                    
                    # Determine split
                    if current_pmid in train_pmids:
                        split = 'train'
                    elif current_pmid in dev_pmids:
                        split = 'dev' 
                    elif current_pmid in test_pmids:
                        split = 'test'
                    else:
                        split = 'unknown'
                    
                    documents[current_pmid] = {
                        'pmid': current_pmid,
                        'title': title,
                        'abstract': "",
                        'text': title,
                        'mentions': [],
                        'split': split
                    }
                elif '|a|' in line and current_pmid:  # Abstract line
                    abstract = line.split('|a|', 1)[1].strip()
                    documents[current_pmid]['abstract'] = abstract
                    # Combine title and abstract
                    documents[current_pmid]['text'] = documents[current_pmid]['title'] + " " + abstract
        
        # Second pass: collect entity mentions
        self.logger.info("Collecting entity mentions...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '|t|' in line or '|a|' in line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 6:  # Entity mention line
                    pmid = parts[0].strip()
                    if pmid in documents:
                        start_offset = int(parts[1])
                        end_offset = int(parts[2])
                        mention_text = parts[3]
                        semantic_types = parts[4]  # Contains T codes
                        cui = parts[5]
                        
                        # Extract the first T code (if multiple are present)
                        t_code_match = re.search(r'(T\d+)', semantic_types)
                        t_code = t_code_match.group(1) if t_code_match else None
                        
                        mention = {
                            'pmid': pmid,
                            'start': start_offset,
                            'end': end_offset,
                            'text': mention_text,
                            'semantic_type': semantic_types,
                            't_code': t_code,
                            'cui': cui
                        }
                        documents[pmid]['mentions'].append(mention)
        
        return list(documents.values())
    
    def _extract_mentions(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract mentions from MedMentions documents."""
        self.logger.info("Extracting mentions from MedMentions documents...")
        
        mentions = []
        
        for doc in raw_data:
            pmid = doc['pmid']
            doc_text = doc['text']
            
            for mention in doc['mentions']:
                mention_data = {
                    'doc_id': pmid,
                    'doc_text': doc_text,
                    'start': mention['start'],
                    'end': mention['end'],
                    'text': mention['text'],
                    'native_id': mention['cui'],  # UMLS CUI is the "native" ID for MedMentions
                    'native_ontology_name': 'UMLS',
                    'semantic_type': mention['t_code'],  # Use T-code as semantic type
                    'semantic_types_full': mention['semantic_type'],  # Keep full semantic types
                    'cui': mention['cui'],  # Already has CUI
                    'split': doc.get('split', 'unknown')
                }
                mentions.append(mention_data)
        
        self.logger.info(f"Extracted {len(mentions)} mentions")
        return mentions
    
    def _map_single_mention(self, mention: Dict) -> str:
        """Validate existing UMLS CUI for MedMentions and map to current version."""
        # MedMentions has older CUIs that need validation/mapping to current UMLS version
        existing_cui = mention.get('cui', '') or mention.get('native_id', '')
        text = mention.get('text', '')
        
        # First try to validate/map the existing CUI to current UMLS version
        if existing_cui and existing_cui.strip():
            existing_cui = existing_cui.strip()
            
            # Check if we have a validate_cui method and use it
            if hasattr(self.umls_mapper, 'validate_cui'):
                # For API-based mapper, validate_cui returns the CUI if valid, None otherwise
                if hasattr(self.umls_mapper, 'api_key'):  # API-based mapper
                    validated_cui = self.umls_mapper.validate_cui(existing_cui)
                    if validated_cui and isinstance(validated_cui, str):
                        mention['mapping_method'] = 'native_id_mapping'
                        return validated_cui
                else:  # Local mapper
                    # For local mapper, validate_cui returns boolean, so use it differently
                    is_valid = self.umls_mapper.validate_cui(existing_cui)
                    if is_valid:
                        mention['mapping_method'] = 'native_id_mapping'
                        return existing_cui
            else:
                # Fallback - try to get CUI info to validate existence
                try:
                    semantic_types = self.umls_mapper.get_semantic_types(existing_cui)
                    if semantic_types:  # If we get semantic types, CUI is valid
                        mention['mapping_method'] = 'native_id_mapping'
                        return existing_cui
                except:
                    pass
        
        # If CUI validation fails, try mapping by text as fallback
        if text and text.strip():
            cui = self.umls_mapper.get_cui_from_text(text.strip())
            if cui:
                method = getattr(self.umls_mapper, 'last_text_mapping_method', None)
                mention['mapping_method'] = method if method in {'exact_match', 'semantic_containment'} else 'text_fallback'
                return cui
        
        # If no mapping found, return empty string
        mention['mapping_method'] = 'no_mapping'
        return ''
    
    def _create_mapping_key(self, mention: Dict) -> str:
        """Create mapping key specific to MedMentions dataset."""
        # MedMentions has existing CUI + text, prioritize CUI validation
        cui = mention.get('cui', '') or mention.get('native_id', '')
        text = mention.get('text', '')
        semantic_type = mention.get('semantic_type', '')
        
        key_parts = []
        if cui and cui != '':
            key_parts.append(f"cui:{cui}")
        if semantic_type:
            key_parts.append(f"type:{semantic_type}")
        if text:
            key_parts.append(f"text:{text}")
        
        return "|".join(key_parts)
    
    def _convert_to_standard_format(self, mentions: List[Dict]) -> List[Dict]:
        """Convert MedMentions mentions to standardized document format with split information."""
        self.logger.info("Converting to standardized format...")
        
        # Group mentions by document and split
        documents_dict = {}
        
        for mention in mentions:
            doc_id = mention['doc_id']
            split = mention.get('split', 'unknown')
            
            # Create a unique key that includes split information
            doc_key = f"{doc_id}_{split}"
            
            if doc_key not in documents_dict:
                documents_dict[doc_key] = {
                    'doc_id': doc_id,
                    'text': mention['doc_text'],
                    'split': split,
                    'mentions': []
                }
            
            # Create standardized mention
            standardized_mention = {
                'start': mention['start'],
                'end': mention['end'],
                'text': mention['text'],
                'native_id': mention.get('native_id', ''),
                'native_ontology_name': 'UMLS',
                'cui': mention.get('cui', ''),
                'umls_name': mention.get('umls_name', ''),
                'semantic_type': mention.get('semantic_type', ''),
                'semantic_types_full': mention.get('semantic_types_full', ''),  # Keep MedMentions-specific field
                'mapping_method': mention.get('mapping_method', '')
            }
            
            documents_dict[doc_key]['mentions'].append(standardized_mention)
        
        documents = list(documents_dict.values())
        self.logger.info(f"Created {len(documents)} documents with mentions")
        
        return documents
    
    def _save_documents(self, documents: List[Dict]):
        """Save documents to JSONL files, separated by split."""
        # Group documents by split
        documents_by_split = {}
        for doc in documents:
            split = doc.get('split', 'unknown')
            if split not in documents_by_split:
                documents_by_split[split] = []
            documents_by_split[split].append(doc)
        
        # Save each split separately
        for split, split_docs in documents_by_split.items():
            if split == 'unknown':
                output_file = self.documents_dir / f"{self.dataset_name}.jsonl"
            else:
                output_file = self.documents_dir / f"{self.dataset_name}_{split}.jsonl"
            
            self.logger.info(f"Saving {len(split_docs)} {split} documents to {output_file}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for doc in split_docs:
                    # Remove split info from the saved document
                    doc_to_save = doc.copy()
                    if 'split' in doc_to_save:
                        del doc_to_save['split']
                    f.write(json.dumps(doc_to_save, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Documents saved successfully across {len(documents_by_split)} splits")
