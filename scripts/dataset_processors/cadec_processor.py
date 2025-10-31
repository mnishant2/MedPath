#!/usr/bin/env python3
"""
CADEC dataset processor.

Processes the CADEC (CSIRO Adverse Drug Event Corpus) which contains patient reports
from health forums annotated with both SNOMED CT and MedDRA concept IDs.
"""

import re
import pandas as pd
import json
from typing import List, Dict, Any
from pathlib import Path

from .base_processor import BaseProcessor

class CADECProcessor(BaseProcessor):
    """Processor for the CADEC dataset."""

    def _get_native_ontologies(self) -> List[str]:
        """CADEC uses both SNOMED CT and MedDRA codes."""
        return ['SNOMEDCT_US', 'MDR']
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load CADEC corpus files."""
        self.logger.info("Loading CADEC corpus data...")
        
        # CADEC has a specific directory structure
        text_dir = self.data_dir / 'text'
        sct_dir = self.data_dir / 'sct'
        meddra_dir = self.data_dir / 'meddra'
        
        if not text_dir.exists():
            # Try alternative path structure
            cadec_subdir = self.data_dir / 'cadec'
            if cadec_subdir.exists():
                text_dir = cadec_subdir / 'text'
                sct_dir = cadec_subdir / 'sct'
                meddra_dir = cadec_subdir / 'meddra'
        
        if not text_dir.exists():
            raise FileNotFoundError(f"CADEC text directory not found at {text_dir}")
        
        self.logger.info(f"Using CADEC directories: {text_dir}, {sct_dir}, {meddra_dir}")
        
        # Get all document IDs from text files
        doc_ids = [file.stem for file in text_dir.glob("*.txt")]
        self.logger.info(f"Found {len(doc_ids)} documents")
        
        raw_data = []
        for doc_id in sorted(doc_ids):
            doc_data = self._process_document(doc_id, text_dir, sct_dir, meddra_dir)
            if doc_data:
                raw_data.append(doc_data)
        
        self.logger.info(f"Loaded {len(raw_data)} documents from CADEC corpus")
        return raw_data
    
    def _process_document(self, doc_id: str, text_dir: Path, sct_dir: Path, meddra_dir: Path) -> Dict[str, Any]:
        """Process a single CADEC document."""
        # Read text file
        text_file = text_dir / f"{doc_id}.txt"
        try:
            with open(text_file, 'r', encoding='latin-1') as f:
                text_content = f.read()
        except Exception as e:
            self.logger.warning(f"Error reading text file {text_file}: {e}")
            return None
        
        if not text_content.strip():
            self.logger.warning(f"Empty text file: {text_file}")
            return None
        
        # Parse annotation files
        sct_file = sct_dir / f"{doc_id}.ann"
        meddra_file = meddra_dir / f"{doc_id}.ann"
        
        sct_annotations = self._parse_annotation_file(sct_file, is_snomed=True)
        meddra_annotations = self._parse_annotation_file(meddra_file, is_snomed=False)
        
        return {
            'doc_id': doc_id,
            'text': text_content,
            'sct_annotations': sct_annotations,
            'meddra_annotations': meddra_annotations
        }
    
    def _parse_annotation_file(self, file_path: Path, is_snomed: bool = False) -> List[Dict[str, Any]]:
        """Parse CADEC annotation file."""
        annotations = []
        
        if not file_path.exists() or file_path.stat().st_size == 0:
            return annotations
        
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) < 3:
                        continue
                    
                    annotation_id = parts[0]
                    annotation_info = parts[1]
                    text = parts[2]
                    
                    # Extract span information using regex
                    span_match = re.search(r'(\w+)\s+(\d+)\s+(\d+)', annotation_info)
                    if span_match:
                        entity_type = span_match.group(1)
                        start = int(span_match.group(2))
                        end = int(span_match.group(3))
                    else:
                        # Fallback parsing
                        info_parts = annotation_info.split(' ')
                        if len(info_parts) < 3:
                            continue
                        
                        entity_type = info_parts[0]
                        start = None
                        end = None
                        
                        for part in info_parts[1:]:
                            if part.isdigit():
                                if start is None:
                                    start = int(part)
                                else:
                                    end = int(part)
                                    break
                        
                        if start is None or end is None:
                            continue
                    
                    # Extract concept information
                    concept_id = None
                    concept_name = None
                    
                    if is_snomed:
                        # SNOMED CT format handling
                        if '||' in annotation_info:
                            concept_part = annotation_info.split('||')[1].strip()
                            if '|' in concept_part:
                                id_and_name = concept_part.split('|', 1)
                                concept_id = id_and_name[0].strip()
                                if len(id_and_name) > 1:
                                    concept_name = id_and_name[1].strip()
                                
                                # Clean concept ID (remove "or" etc.)
                                if concept_id and 'or' in concept_id:
                                    concept_id = concept_id.split('or')[0].strip()
                        
                        elif '|' in annotation_info:
                            # Extract concept name and try to find ID
                            concept_name_part = annotation_info.split('|', 1)[1].strip()
                            concept_name = self._clean_concept_name(concept_name_part)
                            
                            # Try to extract ID from before the |
                            id_part = annotation_info.split('|', 1)[0]
                            id_match = re.search(r'(CONCEPT[_-]LESS|\d+)', id_part, flags=re.IGNORECASE)
                            if id_match:
                                concept_id = id_match.group(1)
                        
                        else:
                            # Try to extract numeric ID or explicit CONCEPT-LESS token
                            id_match = re.search(r'(CONCEPT[_-]LESS|\d+)', annotation_info, flags=re.IGNORECASE)
                            if id_match:
                                concept_id = id_match.group(1)
                    
                    else:
                        # MedDRA format - extract numeric ID or explicit CONCEPT-LESS token
                        id_match = re.search(r'(CONCEPT[_-]LESS|\d+)', annotation_info, flags=re.IGNORECASE)
                        if id_match:
                            concept_id = id_match.group(1)

                    # Normalize concept-less marker
                    if concept_id and isinstance(concept_id, str) and concept_id.upper() in ['CONCEPT-LESS', 'CONCEPT_LESS', 'CONCEPTLESS']:
                        concept_id = 'CONCEPT-LESS'
                    
                    annotation = {
                        'annotation_id': annotation_id,
                        'start': start,
                        'end': end,
                        'text': text,
                        'entity_type': entity_type,
                        'concept_id': concept_id
                    }
                    
                    if is_snomed and concept_name:
                        annotation['concept_name'] = concept_name
                    
                    annotations.append(annotation)
        
        except Exception as e:
            self.logger.warning(f"Error parsing annotation file {file_path}: {e}")
        
        return annotations
    
    def _clean_concept_name(self, concept_name: str) -> str:
        """Clean SNOMED CT concept name by removing span and formatting information."""
        if not concept_name:
            return concept_name
        
        # Remove span information and complex formatting
        if '|' in concept_name:
            parts = concept_name.split('|', 1)
            concept_name = parts[0].strip()
            
            if not concept_name or concept_name == '+':
                second_part = parts[1].strip()
                second_part = re.sub(r'\+\s*\d+\|', '', second_part)
                second_part = re.sub(r'\d+\s+\d+(;\d+\s+\d+)*', '', second_part)
                second_part = re.sub(r'^or\s+', '', second_part)
                
                if second_part and not second_part.isspace():
                    concept_name = second_part.strip()
                else:
                    concept_name = "Unknown concept"
        
        return concept_name.strip()
    
    def _extract_mentions(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract mentions from CADEC documents."""
        self.logger.info("Extracting mentions from CADEC documents...")
        
        mentions = []
        
        for doc in raw_data:
            doc_id = doc['doc_id']
            doc_text = doc['text']
            
            # Create span-to-annotation mappings
            span_to_sct = {}
            span_to_meddra = {}
            
            # Map SNOMED CT annotations by span
            for ann in doc['sct_annotations']:
                span = (ann['start'], ann['end'])
                span_to_sct[span] = ann
            
            # Map MedDRA annotations by span
            for ann in doc['meddra_annotations']:
                span = (ann['start'], ann['end'])
                span_to_meddra[span] = ann
            
            # Get all unique spans
            all_spans = set(list(span_to_sct.keys()) + list(span_to_meddra.keys()))
            
            # Process each span
            for span in all_spans:
                start, end = span
                span_text = doc_text[start:end]
                
                # Create mention with both SNOMED CT and MedDRA information
                mention = {
                    'doc_id': doc_id,
                    'doc_text': doc_text,
                    'start': start,
                    'end': end,
                    'text': span_text,
                    'snomed_id': '',
                    'snomed_name': '',
                    'meddra_id': '',
                    'annotation_id': ''
                }
                
                # Add SNOMED CT information if available
                if span in span_to_sct:
                    sct_ann = span_to_sct[span]
                    mention['snomed_id'] = sct_ann.get('concept_id', '') or ''
                    if isinstance(mention['snomed_id'], str) and mention['snomed_id'].upper() in ['CONCEPT-LESS', 'CONCEPT_LESS', 'CONCEPTLESS']:
                        mention['snomed_id'] = ''
                        mention['concept_less'] = True
                    mention['snomed_name'] = sct_ann.get('concept_name', '') or ''
                    mention['annotation_id'] = sct_ann.get('annotation_id', '') or ''
                
                # Add MedDRA information if available
                if span in span_to_meddra:
                    meddra_ann = span_to_meddra[span]
                    mention['meddra_id'] = meddra_ann.get('concept_id', '') or ''
                    # Validate MedDRA ID: ensure numeric and reasonable length; else mark concept-less
                    try:
                        if mention['meddra_id']:
                            med_id_str = str(mention['meddra_id']).strip()
                            if med_id_str.upper() in ['CONCEPT-LESS', 'CONCEPT_LESS', 'CONCEPTLESS']:
                                mention['meddra_id'] = ''
                                mention['concept_less'] = True
                            elif (not med_id_str.isdigit()) or (len(med_id_str) < 5):
                                mention['meddra_id'] = ''
                    except Exception:
                        mention['meddra_id'] = ''
                    
                    # Use MedDRA annotation ID if no SNOMED ID available
                    if not mention['annotation_id']:
                        mention['annotation_id'] = meddra_ann.get('annotation_id', '') or ''
                
                # Determine primary native ID and ontology
                if mention['snomed_id']:
                    mention['native_id'] = mention['snomed_id']
                    mention['native_ontology_name'] = 'SNOMEDCT_US'
                elif mention['meddra_id']:
                    mention['native_id'] = mention['meddra_id']
                    mention['native_ontology_name'] = 'MDR'
                else:
                    mention['native_id'] = ''
                    mention['native_ontology_name'] = ''
                
                mentions.append(mention)
        
        self.logger.info(f"Extracted {len(mentions)} mentions")
        return mentions
    
    def _create_mapping_key(self, mention: Dict) -> str:
        """Create mapping key specific to CADEC dataset."""
        # CADEC has both SNOMED CT and MedDRA IDs, prioritize SNOMED CT
        snomed_id = mention.get('snomed_id', '')
        meddra_id = mention.get('meddra_id', '')
        text = mention.get('text', '')
        
        key_parts = []
        if snomed_id and snomed_id != '' and snomed_id.upper() not in ['CONCEPT-LESS', 'CONCEPTLESS']:
            key_parts.append(f"snomed:{snomed_id}")
        if meddra_id and meddra_id != '':
            key_parts.append(f"meddra:{meddra_id}")
        if text:
            key_parts.append(f"text:{text}")
        
        return "|".join(key_parts)
    
    def _map_single_mention(self, mention: Dict) -> str:
        """Map a single CADEC mention to UMLS CUI."""
        # Try SNOMED CT ID first
        snomed_id = mention.get('snomed_id', '')
        if snomed_id and snomed_id.upper() not in ['CONCEPT-LESS', 'CONCEPTLESS']:
            # Try multiple SNOMED sources
            for source in ['SNOMEDCT_US', 'SCTSPA', 'SNOMEDCT', 'SNOMEDCT_CORE']:
                cui = self.umls_mapper.get_cui_from_ontology_id(snomed_id, source)
                if cui:
                    mention['mapping_method'] = 'native_id_mapping'
                    return cui
        
        # Try MedDRA ID if SNOMED CT fails
        meddra_id = mention.get('meddra_id', '')
        if meddra_id:
            cui = self.umls_mapper.get_cui_from_ontology_id(meddra_id, 'MDR')
            if cui:
                mention['mapping_method'] = 'native_id_mapping'
                return cui
        
        # Try text as fallback
        text = mention.get('text', '')
        if text:
            cui = self.umls_mapper.get_cui_from_text(text)
            if cui:
                method = getattr(self.umls_mapper, 'last_text_mapping_method', None)
                mention['mapping_method'] = method if method in {'exact_match', 'semantic_containment'} else 'text_fallback'
                return cui
        
        mention['mapping_method'] = 'no_mapping'
        return ''
    
    def _convert_to_standard_format(self, mentions: List[Dict]) -> List[Dict]:
        """Convert CADEC mentions to standardized document format."""
        self.logger.info("Converting to standardized format...")
        
        # Group mentions by document
        documents_dict = {}
        
        for mention in mentions:
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
                'mapping_method': mention.get('mapping_method', ''),
                # Keep CADEC-specific fields
                'snomed_id': mention.get('snomed_id', ''),
                'snomed_name': mention.get('snomed_name', ''),
                'meddra_id': mention.get('meddra_id', ''),
                'annotation_id': mention.get('annotation_id', '')
            }
            
            documents_dict[doc_id]['mentions'].append(standardized_mention)
        
        documents = list(documents_dict.values())
        self.logger.info(f"Created {len(documents)} documents with mentions")
        
        return documents
