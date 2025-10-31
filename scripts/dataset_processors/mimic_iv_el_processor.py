#!/usr/bin/env python3
"""
MIMIC-IV EL dataset processor.

Processes the MIMIC-IV Entity Linking dataset which contains clinical notes
annotated with SNOMED CT concept IDs. This is a licensed dataset.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

from .base_processor import BaseProcessor

class MIMICIVELProcessor(BaseProcessor):
    """Processor for the MIMIC-IV EL dataset (licensed)."""
    
    def _get_native_ontologies(self) -> List[str]:
        """MIMIC-IV EL uses SNOMED CT codes."""
        return ['SNOMEDCT_US']
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load MIMIC-IV EL corpus files."""
        self.logger.info("Loading MIMIC-IV EL corpus data...")
        
        # This is a licensed dataset - user must provide the data
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"MIMIC-IV EL data directory not found: {self.data_dir}\n"
                "This is a licensed dataset. Please obtain access and place the files in the data directory."
            )
        
        # Look for CSV files - cleaned format
        notes_file = self.data_dir / "mimic-iv_notes_training_set_cleaned.csv"
        annotations_file = self.data_dir / "train_annotations_cleaned.csv"
        
        if not notes_file.exists() or not annotations_file.exists():
            raise FileNotFoundError(
                f"Required CSV files not found in {self.data_dir}.\n"
                f"Expected: {notes_file} and {annotations_file}"
            )
        
        # Load notes data
        self.logger.info(f"Loading notes from {notes_file}")
        notes_df = pd.read_csv(notes_file)
        
        # Load annotations data
        self.logger.info(f"Loading annotations from {annotations_file}")
        annotations_df = pd.read_csv(annotations_file)
        
        # Create a dictionary to store notes by note_id for faster access
        notes_dict = dict(zip(notes_df['note_id'], notes_df['text']))
        
        # Process annotations and create documents
        raw_data = []
        processed_notes = set()
        
        for _, row in annotations_df.iterrows():
            note_id = row['note_id']
            start = row['start']
            end = row['end']
            concept_id = row['concept_id']
            
            # Get the full text of the note
            if note_id not in notes_dict:
                self.logger.warning(f"Note ID {note_id} not found in notes data")
                continue
            
            note_text = notes_dict[note_id]
            
            # Extract the mention text from the note
            try:
                mention_text = note_text[start:end]
            except Exception as e:
                self.logger.warning(f"Error extracting mention from note {note_id}: {e}")
                continue
            
            # Create or update document
            doc_id = str(note_id)
            
            # Find existing document or create new one
            doc_data = None
            for existing_doc in raw_data:
                if existing_doc['doc_id'] == doc_id:
                    doc_data = existing_doc
                    break
            
            if doc_data is None:
                doc_data = {
                    'doc_id': doc_id,
                    'text': note_text,
                    'annotations': []
                }
                raw_data.append(doc_data)
            
            # Add annotation
            doc_data['annotations'].append({
                'start': start,
                'end': end,
                'text': mention_text,
                'concept_id': str(concept_id)
            })
        
        self.logger.info(f"Loaded {len(raw_data)} documents from MIMIC-IV EL corpus")
        return raw_data
    
    def _extract_mentions(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract mentions from MIMIC-IV EL documents."""
        self.logger.info("Extracting mentions from MIMIC-IV EL documents...")
        
        mentions = []
        
        for doc in raw_data:
            doc_id = doc.get('doc_id', '')
            doc_text = doc.get('text', '')
            annotations = doc.get('annotations', [])
            
            for annotation in annotations:
                start = annotation['start']
                end = annotation['end']
                mention_text = annotation['text']
                snomed_id = annotation['concept_id']
                
                mention = {
                    'doc_id': doc_id,
                    'doc_text': doc_text,
                    'start': start,
                    'end': end,
                    'text': mention_text,
                    'native_id': snomed_id,
                    'native_ontology_name': 'SNOMEDCT_US',
                    'entity_type': 'Clinical Entity',
                    'annotation_id': f"{doc_id}_{start}_{end}"
                }
                
                mentions.append(mention)
        
        self.logger.info(f"Extracted {len(mentions)} mentions from MIMIC-IV EL documents")
        return mentions
    
    def _create_mapping_key(self, mention: Dict) -> str:
        """Create a unique key for UMLS mapping cache."""
        return f"{mention['native_id']}|{mention['text']}|{mention['entity_type']}"
    
    def _map_single_mention(self, mention: Dict) -> str:
        """Map a single mention to UMLS CUI."""
        # Since this is SNOMED CT -> UMLS mapping, we use the SNOMED CT ID
        snomed_id = mention.get('native_id', '')
        mention_text = mention.get('text', '')
        
        if not snomed_id:
            # If no SNOMED ID, try mapping by text
            cui = self.umls_mapper.get_cui_from_text(mention_text) or ''
            if cui:
                method = getattr(self.umls_mapper, 'last_text_mapping_method', None)
                mention['mapping_method'] = method if method in {'exact_match', 'semantic_containment'} else 'text_fallback'
            else:
                mention['mapping_method'] = 'no_mapping'
            return cui
        
        # Try to map SNOMED CT ID to UMLS CUI
        cui = self.umls_mapper.get_cui_from_ontology_id(snomed_id, 'SNOMEDCT_US')
        
        if cui:
            mention['mapping_method'] = 'native_id_mapping'
        if not cui:
            # Fallback to text-based mapping
            cui = self.umls_mapper.get_cui_from_text(mention_text)
            if cui:
                method = getattr(self.umls_mapper, 'last_text_mapping_method', None)
                mention['mapping_method'] = method if method in {'exact_match', 'semantic_containment'} else 'text_fallback'
            else:
                mention['mapping_method'] = 'no_mapping'
        
        return cui or ''
    
    def _convert_to_standard_format(self, mentions: List[Dict]) -> List[Dict]:
        """Convert mentions to standard format."""
        documents = {}
        
        for mention in mentions:
            doc_id = mention['doc_id']
            
            if doc_id not in documents:
                documents[doc_id] = {
                    'doc_id': doc_id,
                    'text': mention['doc_text'],
                    'mentions': []
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