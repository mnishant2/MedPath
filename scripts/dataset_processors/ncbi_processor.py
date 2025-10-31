#!/usr/bin/env python3
"""
NCBI Disease dataset processor.

Processes the NCBI Disease corpus which contains PubMed abstracts annotated with 
disease mentions along with their MeSH and OMIM IDs.
"""

import os
import re
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import json

from .base_processor import BaseProcessor

class NCBIProcessor(BaseProcessor):
    """Processor for the NCBI Disease dataset."""
    
    def _get_native_ontologies(self) -> List[str]:
        """NCBI uses MeSH and OMIM IDs."""
        return ['MSH', 'OMIM']
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load NCBI Disease corpus files."""
        self.logger.info("Loading NCBI Disease corpus data...")
        
        # NCBI has train, dev, and test files
        datasets = ['train', 'dev', 'test']
        file_mapping = {
            'train': 'NCBItrainset_corpus.txt',
            'dev': 'NCBIdevelopset_corpus.txt', 
            'test': 'NCBItestset_corpus.txt'
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
        
        self.logger.info(f"Loaded {len(raw_data)} documents from NCBI corpus")
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
            entity_type = parts[4]  # Disease
            concept_id = parts[5]
            
            # Determine if MeSH or OMIM ID
            mesh_id = None
            omim_id = None
            native_ontology = None
            
            if "OMIM:" in concept_id:
                omim_id = concept_id.replace("OMIM:", "")
                native_ontology = "OMIM"
            else:
                mesh_id = concept_id
                native_ontology = "MSH"
            
            return {
                'doc_id': doc_id,
                'start': start,
                'end': end,
                'text': mention_text,
                'entity_type': entity_type,
                'concept_id': concept_id,
                'mesh_id': mesh_id,
                'omim_id': omim_id,
                'native_ontology': native_ontology
            }
        except (ValueError, IndexError):
            self.logger.warning(f"Could not parse annotation line: {line}")
            return None
    
    def _extract_mentions(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """Extract mentions from NCBI documents."""
        self.logger.info("Extracting mentions from NCBI documents...")
        
        mentions = []
        
        for doc in raw_data:
            doc_id = doc['doc_id']
            doc_text = doc['text']
            
            for annotation in doc['annotations']:
                # Concept ID may include multiple codes separated by '|' (same format) or '+' (combined)
                concept_id_str = str(annotation.get('concept_id', '') or '').strip()

                # Helper to parse a token into (source, code)
                def parse_token(token: str) -> Any:
                    t = token.strip()
                    if not t or t == '-':
                        return None
                    # Do not treat bare 'C' codes as UMLS CUIs in NCBI; these are often MeSH SCR IDs
                    if t.upper().startswith('OMIM:'):
                        return ('OMIM', t.split(':', 1)[1].strip())
                    # Default to MeSH when looks like Dxxxxxx or lacks explicit source
                    # Keep MeSH code as-is; do not force D-prefix normalization here
                    if re.match(r'^[Dd]\d+$', t):
                        return ('MSH', t.upper())
                    if t.isdigit():
                        # Numeric with no prefix is likely OMIM; include as OMIM candidate
                        return ('OMIM', t)
                    return ('MSH', t)

                pipe_segments = [seg.strip() for seg in concept_id_str.split('|') if seg.strip()]
                has_plus = any('+' in seg for seg in pipe_segments)

                # Parse candidates per segment
                parsed_per_segment = []  # List[List[(source, code)]]
                for seg in pipe_segments if pipe_segments else [concept_id_str]:
                    plus_tokens = [tok.strip() for tok in re.split(r"\s*\+\s*", seg) if tok.strip()]
                    parsed_tokens = [parse_token(tok) for tok in plus_tokens]
                    parsed_tokens = [pt for pt in parsed_tokens if pt is not None]
                    if parsed_tokens:
                        parsed_per_segment.append(parsed_tokens)

                # Decide splitting strategy
                split_into_multiple = False
                if parsed_per_segment and not has_plus:
                    # If each segment has exactly one token and all sources are the same, split
                    if all(len(seg_list) == 1 for seg_list in parsed_per_segment):
                        sources = {seg_list[0][0] for seg_list in parsed_per_segment}
                        if len(sources) == 1:
                            split_into_multiple = True

                if split_into_multiple:
                    # Create one mention per same-format code separated by '|'
                    for seg_list in parsed_per_segment:
                        src, code = seg_list[0]
                        mention = {
                            'doc_id': doc_id,
                            'doc_text': doc_text,
                            'start': annotation['start'],
                            'end': annotation['end'],
                            'text': annotation['text'],
                            'native_id': code,
                            'native_ontology_name': src,
                            'entity_type': annotation['entity_type'],
                            'concept_id': annotation['concept_id'],
                            'candidate_native_ids': [(src, code)],
                            'split': doc.get('split', 'unknown')
                        }
                        mentions.append(mention)
                else:
                    # Single mention with potentially multiple candidate IDs (from '+' and/or mixed sources)
                    candidates = []
                    for seg_list in parsed_per_segment:
                        for src, code in seg_list:
                            candidates.append((src, code))

                    # Fallback to single native id fields when no candidates parsed
                    if not candidates:
                        native_id = annotation['mesh_id'] if annotation['mesh_id'] else annotation['omim_id']
                        native_ontology = annotation['native_ontology']
                        candidates = [(native_ontology, native_id)] if native_id and native_ontology else []

                    # Use first candidate as primary native_id for compatibility
                    primary_src, primary_code = candidates[0] if candidates else ('', '')
                    mention = {
                        'doc_id': doc_id,
                        'doc_text': doc_text,
                        'start': annotation['start'],
                        'end': annotation['end'],
                        'text': annotation['text'],
                        'native_id': primary_code,
                        'native_ontology_name': primary_src,
                        'entity_type': annotation['entity_type'],
                        'concept_id': annotation['concept_id'],
                        'candidate_native_ids': candidates,
                        'split': doc.get('split', 'unknown')
                    }
                    mentions.append(mention)
        
        self.logger.info(f"Extracted {len(mentions)} mentions")
        return mentions
    
    def _create_mapping_key(self, mention: Dict) -> str:
        """Create mapping key specific to NCBI dataset."""
        # NCBI has MeSH and OMIM IDs, so prioritize those
        native_id = mention.get('native_id', '')
        native_ontology = mention.get('native_ontology_name', '')
        text = mention.get('text', '')
        entity_type = mention.get('entity_type', '')
        candidates = mention.get('candidate_native_ids', [])
        
        key_parts = []
        if candidates:
            try:
                cand_str = ';'.join([f"{src}:{code}" for src, code in candidates])
                key_parts.append(f"cands:{cand_str}")
            except Exception:
                pass
        if native_id and native_id != '' and native_id != '-':
            key_parts.append(f"{native_ontology}:{native_id}")
        if entity_type:
            key_parts.append(f"type:{entity_type}")
        if text:
            key_parts.append(f"text:{text}")
        
        return "|".join(key_parts)

    def _map_single_mention(self, mention: Dict) -> str:
        """Map a single NCBI mention to UMLS CUI with proper validation."""
        # NCBI has concept_ids that might be CUIs, MeSH, or OMIM IDs that need proper mapping to current UMLS
        native_id = mention.get('native_id', '')
        native_ontology = mention.get('native_ontology_name', '')
        text = mention.get('text', '')
        concept_id = mention.get('concept_id', '')
        candidate_ids = mention.get('candidate_native_ids', [])  # List[(source, code)]
        
        # First check if concept_id is already a CUI that needs validation
        if concept_id and concept_id.strip():
            concept_id = concept_id.strip()
            
            # Check if it looks like a CUI (starts with C and has right length)
            if concept_id.startswith('C') and len(concept_id) >= 7:
                # Try to validate the existing CUI
                if hasattr(self.umls_mapper, 'validate_cui'):
                    if hasattr(self.umls_mapper, 'api_key'):  # API-based mapper
                        validated_cui = self.umls_mapper.validate_cui(concept_id)
                        if validated_cui and isinstance(validated_cui, str):
                            mention['mapping_method'] = 'native_id_mapping'
                            return validated_cui
                    else:  # Local mapper
                        is_valid = self.umls_mapper.validate_cui(concept_id)
                        if is_valid:
                            mention['mapping_method'] = 'native_id_mapping'
                            return concept_id
                else:
                    # Fallback - try to get semantic types to validate existence
                    try:
                        semantic_types = self.umls_mapper.get_semantic_types(concept_id)
                        if semantic_types:
                            mention['mapping_method'] = 'native_id_mapping'
                            return concept_id
                    except:
                        pass
        
        # If concept_id CUI validation fails, try mapping using any available candidate native IDs
        if candidate_ids:
            for src, code in candidate_ids:
                if code and code != '' and code != '-' and src:
                    cui = self.umls_mapper.get_cui_from_ontology_id(code, src)
                    if cui:
                        mention['mapping_method'] = 'native_id_mapping'
                        return cui
        else:
            # Fallback to single native_id if candidates not provided
            if native_id and native_id != '' and native_id != '-' and native_ontology:
                cui = self.umls_mapper.get_cui_from_ontology_id(native_id, native_ontology)
                if cui:
                    mention['mapping_method'] = 'native_id_mapping'
                    return cui
        
        # If ontology mapping fails, try text-based mapping as fallback
        if text and text.strip():
            cui = self.umls_mapper.get_cui_from_text(text.strip())
            if cui:
                method = getattr(self.umls_mapper, 'last_text_mapping_method', None)
                mention['mapping_method'] = method if method in {'exact_match', 'semantic_containment'} else 'text_fallback'
                return cui
        
        # If no mapping found, return empty string
        mention['mapping_method'] = 'no_mapping'
        return ''
    
    def _convert_to_standard_format(self, mentions: List[Dict]) -> List[Dict]:
        """Convert NCBI mentions to standardized document format with split information."""
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
                'native_ontology_name': mention.get('native_ontology_name', ''),
                'cui': mention.get('cui', ''),
                'umls_name': mention.get('umls_name', ''),
                'semantic_type': mention.get('semantic_type', ''),
                'entity_type': mention.get('entity_type', ''),  # Keep NCBI-specific field
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