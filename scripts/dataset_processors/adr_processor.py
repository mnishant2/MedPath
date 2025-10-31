#!/usr/bin/env python3
"""
ADR dataset processor.

Processes the ADR (Adverse Drug Reaction) corpus which contains drug-related adverse events
annotated with MedDRA concept IDs.
"""

import os
import re
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

from .base_processor import BaseProcessor

class ADRProcessor(BaseProcessor):
    """Processor for the ADR dataset."""
    
    def _get_native_ontologies(self) -> List[str]:
        """ADR uses MedDRA codes."""
        return ['MDR']
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load ADR corpus files."""
        self.logger.info("Loading ADR corpus data...")
        
        raw_data = []
        
        # Look for XML files in train/test directories
        xml_files_by_split = {}
        for split_dir in ['train', 'test']:
            split_path = self.data_dir / split_dir
            if split_path.exists():
                xml_files_by_split[split_dir] = list(split_path.glob("*.xml"))
        
        # If no split directories, look in main directory
        if not xml_files_by_split:
            xml_files_by_split['unknown'] = list(self.data_dir.glob("*.xml"))
        
        total_files = sum(len(files) for files in xml_files_by_split.values())
        if total_files == 0:
            raise FileNotFoundError(f"No XML files found in {self.data_dir}")
        
        self.logger.info(f"Found {total_files} XML files across splits: {list(xml_files_by_split.keys())}")
        
        for split, xml_files in xml_files_by_split.items():
            self.logger.info(f"Processing {len(xml_files)} files from {split} split")
            for xml_file in xml_files:
                doc_data = self._process_xml_document(xml_file, split)
                if doc_data:
                    raw_data.append(doc_data)
        
        self.logger.info(f"Loaded {len(raw_data)} documents from ADR corpus")
        return raw_data
    
    def _process_xml_document(self, xml_file: Path, split: str = 'unknown') -> Dict[str, Any]:
        """Process an ADR XML document."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract drug name and track info
            drug_name = root.get('drug', '')
            track = root.get('track', '')
            
            # Extract text sections
            text_sections = {}
            text_elem = root.find('Text')
            if text_elem is not None:
                for section in text_elem.findall('Section'):
                    section_id = section.get('id')
                    section_name = section.get('name')
                    section_text = section.text or ''
                    text_sections[section_id] = {
                        'name': section_name,
                        'text': section_text
                    }
            
            # Extract mentions
            mentions = []
            mentions_elem = root.find('Mentions')
            if mentions_elem is not None:
                for mention in mentions_elem.findall('Mention'):
                    mention_id = mention.get('id')
                    section_id = mention.get('section')
                    mention_type = mention.get('type')
                    start = int(mention.get('start', 0))
                    length = int(mention.get('len', 0))
                    mention_text = mention.get('str', '')
                    
                    mentions.append({
                        'mention_id': mention_id,
                        'section_id': section_id,
                        'type': mention_type,
                        'start': start,
                        'length': length,
                        'end': start + length,
                        'text': mention_text
                    })
            
            # Extract reactions and their MedDRA normalizations
            reactions = {}
            reactions_case_insensitive = {}  # For case-insensitive lookup
            reactions_elem = root.find('Reactions')
            if reactions_elem is not None:
                for reaction in reactions_elem.findall('Reaction'):
                    reaction_id = reaction.get('id')
                    reaction_str = reaction.get('str')
                    
                    # Extract MedDRA normalization
                    normalization = reaction.find('Normalization')
                    meddra_pt_id = ''
                    meddra_pt = ''
                    
                    if normalization is not None:
                        meddra_pt_id = normalization.get('meddra_pt_id', '')
                        meddra_pt = normalization.get('meddra_pt', '')
                    
                    reaction_data = {
                        'reaction_id': reaction_id,
                        'meddra_pt_id': meddra_pt_id,
                        'meddra_pt': meddra_pt
                    }
                    
                    reactions[reaction_str] = reaction_data
                    # Also store with lowercase key for case-insensitive lookup
                    reactions_case_insensitive[reaction_str.lower()] = reaction_data
            
            # Combine full text and track section positions for offset adjustment
            full_text = ''
            section_positions = {}  # Maps section_id to its start position in full_text
            
            for section_id, section_data in text_sections.items():
                section_positions[section_id] = len(full_text)  # Record where this section starts
                full_text += section_data['text'] + '\n\n'
            
            # Remove only trailing whitespace to avoid offset issues
            # Leading spaces in sections are part of the XML offset calculations
            doc_text = full_text.rstrip()
            
            return {
                'doc_id': xml_file.stem,
                'drug_name': drug_name,
                'track': track,
                'text': doc_text,
                'text_sections': text_sections,
                'section_positions': section_positions,
                'mentions': mentions,
                'reactions': reactions,
                'reactions_case_insensitive': reactions_case_insensitive,
                'split': split
            }
            
        except Exception as e:
            self.logger.warning(f"Error processing XML file {xml_file}: {e}")
            return {}
    
    def _extract_mentions(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract mentions from ADR documents."""
        self.logger.info("Extracting mentions from ADR documents...")
        
        mentions = []
        
        for doc in raw_data:
            doc_id = doc['doc_id']
            doc_text = doc['text']
            text_sections = doc['text_sections']
            section_positions = doc.get('section_positions', {})
            doc_mentions = doc['mentions']
            reactions = doc['reactions']
            reactions_case_insensitive = doc.get('reactions_case_insensitive', {})
            split = doc.get('split', 'unknown')
            
            for mention in doc_mentions:
                section_id = mention['section_id']
                mention_text = mention['text']
                # Get section-relative offsets from XML
                section_start = mention['start']
                section_end = mention['end']
                
                # Adjust offsets to be relative to combined document text
                section_position = section_positions.get(section_id, 0)
                start = section_position + section_start
                end = section_position + section_end
                
                # Validate that the adjusted offsets point to the correct text
                if end <= len(doc_text):
                    actual_text = doc_text[start:end]
                    if actual_text != mention_text:
                        self.logger.warning(f"Offset mismatch in {doc_id}: expected '{mention_text}', got '{actual_text}' at {start}-{end}")
                else:
                    self.logger.warning(f"Offset out of bounds in {doc_id}: {start}-{end} > {len(doc_text)}")
                
                # Get section text for context
                section_text = ''
                if section_id in text_sections:
                    section_text = text_sections[section_id]['text']
                
                # Find MedDRA ID for this mention
                meddra_id = ''
                meddra_pt = ''
                
                # Look up the reaction in the reactions dictionary
                # First try exact match, then case-insensitive match
                if mention_text in reactions:
                    reaction_info = reactions[mention_text]
                    meddra_id = reaction_info['meddra_pt_id']
                    meddra_pt = reaction_info['meddra_pt']
                elif mention_text.lower() in reactions_case_insensitive:
                    reaction_info = reactions_case_insensitive[mention_text.lower()]
                    meddra_id = reaction_info['meddra_pt_id']
                    meddra_pt = reaction_info['meddra_pt']
                
                # Create mention entry
                mention_entry = {
                    'doc_id': doc_id,
                    'doc_text': doc_text,
                    'section_id': section_id,
                    'section_text': section_text,
                    'start': start,
                    'end': end,
                    'text': mention_text,
                    'native_id': meddra_id,
                    'native_ontology_name': 'MDR',
                    'entity_type': mention.get('type', 'AdverseReaction'),
                    'mention_id': mention['mention_id'],
                    'meddra_pt': meddra_pt,
                    'split': split
                }
                
                mentions.append(mention_entry)
        
        self.logger.info(f"Extracted {len(mentions)} mentions from ADR documents")
        return mentions
    
    def _create_mapping_key(self, mention: Dict) -> str:
        """Create a unique key for UMLS mapping cache."""
        return f"{mention['native_id']}|{mention['text']}|{mention['entity_type']}"
    
    def _perform_umls_mapping(self, unique_combinations):
        """
        Perform UMLS mapping for unique combinations.
        Strategy: Prioritize LLT → then text → then PT
        """
        mapping_results = {}
        
        for combination in tqdm(unique_combinations, desc="Mapping to UMLS"):
            mention_text, meddra_pt_id, meddra_pt = combination
            
            # Initialize mapping result
            mapping_result = {
                'cui': None,
                'mapping_source': 'unmapped',
                'originally_cui_less': meddra_pt_id == '' and meddra_pt == '',
                'originally_no_concept': meddra_pt_id == '' and meddra_pt == '',
                'cui_validation_success': False,
                'fallback_success': False
            }
            
            # Strategy: Prioritize LLT → then text → then PT
            try:
                # Step 1: Try ontology mapping for LLT (Lower Level Term - most specific)
                if meddra_pt_id:
                    cui = self.umls_mapper.get_cui_from_ontology_id(meddra_pt_id, 'MDR', 'LLT')
                    if cui:
                        mapping_result['cui'] = cui
                        mapping_result['mapping_source'] = 'meddra_llt'
                        mapping_results[combination] = mapping_result
                        continue
                
                # Step 2: Try text mapping
                if mention_text:
                    cui = self.umls_mapper.get_cui_from_text(mention_text)
                    if cui:
                        mapping_result['cui'] = cui
                        mapping_result['mapping_source'] = 'text_mapping'
                        mapping_result['fallback_success'] = True
                        mapping_results[combination] = mapping_result
                        continue
                
                # Step 3: Fall back to PT (Preferred Term)
                if meddra_pt_id:
                    cui = self.umls_mapper.get_cui_from_ontology_id(meddra_pt_id, 'MDR', 'PT')
                    if cui:
                        mapping_result['cui'] = cui
                        mapping_result['mapping_source'] = 'meddra_pt'
                        mapping_result['fallback_success'] = True
                        mapping_results[combination] = mapping_result
                        continue
                
                # No mapping found
                mapping_results[combination] = mapping_result
                
            except Exception as e:
                print(f"Error mapping combination {combination}: {e}")
                mapping_results[combination] = mapping_result
        
        return mapping_results
    
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
                    'split': mention.get('split', 'unknown')
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
                'mention_id': mention.get('mention_id', ''),
                'meddra_pt': mention.get('meddra_pt', ''),
                'mapping_method': mention.get('mapping_method', '')
            }
            
            documents[doc_id]['mentions'].append(mention_entry)
        
        return list(documents.values())
    
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