#!/usr/bin/env python3
"""
Text cleaning utilities for biomedical datasets.

Adapted from existing cleaning scripts for ShareCLEF and MIMIC-IV EL datasets.
"""

import re
import pandas as pd
from typing import Tuple, Dict, List, Optional

class TextCleaner:
    """Utility class for cleaning biomedical text data."""
    
    @staticmethod
    def clean_shareclef_text(text: str) -> str:
        """Clean ShareCLEF text by removing metadata and normalizing whitespace."""
        if not isinstance(text, str):
            return text
        
        # Remove the metadata line with |||| separators (first line)
        lines = text.split('\n')
        if lines and '||||' in lines[0]:
            lines = lines[1:]
            text = '\n'.join(lines)
        
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +([.,;:!?)])', r'\1', text)
        text = re.sub(r'(\() +', r'\1', text)
        text = re.sub(r'_{3,}', '___', text)
        
        return text.strip()
    
    @staticmethod
    def clean_mimiciv_text(text: str) -> str:
        """Clean MIMIC-IV text by removing anonymization placeholders."""
        if not isinstance(text, str):
            return text
        
        # Remove anonymization placeholders (any sequence of 3 or more underscores)
        text = re.sub(r'_{3,}', '', text)
        
        # Remove standalone underscore placeholders
        text = re.sub(r'\b___\b', '', text)
        text = re.sub(r'\b__\b', '', text)
        text = re.sub(r'\b_\b', '', text)
        
        # Clean up extra spaces that result from removing placeholders
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Clean up spacing around punctuation
        text = re.sub(r' +([.,;:!?)])', r'\1', text)
        text = re.sub(r'(\() +', r'\1', text)
        text = re.sub(r' +(\))', r'\1', text)
        
        # Clean up spacing around colons and dashes
        text = re.sub(r'(\w) +:', r'\1:', text)
        text = re.sub(r' +- +', ' - ', text)
        text = re.sub(r' +/ +', '/', text)
        
        # Remove any remaining excessive whitespace
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\n +', '\n', text)
        text = re.sub(r' +\n', '\n', text)
        
        return text.strip()
    
    @staticmethod
    def clean_general_text(text: str) -> str:
        """General text cleaning for biomedical datasets."""
        if not isinstance(text, str):
            return text
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Clean up spacing around punctuation
        text = re.sub(r' +([.,;:!?)])', r'\1', text)
        text = re.sub(r'(\() +', r'\1', text)
        
        return text.strip()
    
    @staticmethod
    def normalize_cui_less(cui: str) -> str:
        """Normalize CUI-Less mentions to a standardized format."""
        if cui is None:
            return "CUI-less"
        if str(cui).upper() in ["CUI-LESS", "CUILESS", "CUI LESS", "NULL", "NONE"]:
            return "CUI-less"
        return cui
    
    @staticmethod
    def calculate_offset_mapping(original_text: str, cleaned_text: str) -> List[int]:
        """Calculate mapping from original positions to cleaned positions."""
        mapping = []
        
        orig_pos = 0
        clean_pos = 0
        
        while orig_pos < len(original_text):
            if clean_pos < len(cleaned_text) and orig_pos < len(original_text):
                # Check if characters match
                if original_text[orig_pos] == cleaned_text[clean_pos]:
                    mapping.append(clean_pos)
                    orig_pos += 1
                    clean_pos += 1
                else:
                    # Character was removed, advance original position
                    mapping.append(clean_pos)
                    orig_pos += 1
                    
                    # Look ahead to see if we need to advance clean_pos
                    found_match = False
                    for look_ahead in range(min(10, len(original_text) - orig_pos)):
                        if (orig_pos + look_ahead < len(original_text) and 
                            clean_pos < len(cleaned_text) and
                            original_text[orig_pos + look_ahead] == cleaned_text[clean_pos]):
                            found_match = True
                            break
                    
                    if not found_match and clean_pos < len(cleaned_text):
                        clean_pos += 1
            else:
                # Past the end of one of the texts
                mapping.append(min(clean_pos, len(cleaned_text)))
                orig_pos += 1
        
        return mapping
    
    @staticmethod
    def adjust_annotation_offsets(
        annotations: List[Dict], 
        text_mappings: Dict[str, List[int]]
    ) -> List[Dict]:
        """Adjust annotation offsets based on text cleaning."""
        updated_annotations = []
        
        for annotation in annotations:
            doc_id = annotation.get('doc_id') or annotation.get('note_id')
            start = int(annotation['start'])
            end = int(annotation['end'])
            
            if doc_id in text_mappings:
                mapping = text_mappings[doc_id]
                
                # Adjust start position
                if start < len(mapping):
                    new_start = mapping[start]
                else:
                    new_start = len(mapping) - 1 if mapping else 0
                
                # Adjust end position
                if end <= len(mapping):
                    new_end = mapping[end - 1] + 1 if end > 0 else 0
                else:
                    new_end = len(mapping) if mapping else 0
                
                # Ensure valid range
                new_start = max(0, new_start)
                new_end = max(new_start, new_end)
                
                # Update annotation
                updated_annotation = annotation.copy()
                updated_annotation['start'] = new_start
                updated_annotation['end'] = new_end
                updated_annotation['original_start'] = start
                updated_annotation['original_end'] = end
                
                updated_annotations.append(updated_annotation)
            else:
                # No mapping available, keep original
                updated_annotations.append(annotation.copy())
        
        return updated_annotations
    
    @staticmethod
    def parse_discontinuous_offsets(offset_str: str) -> Tuple[List[int], List[int]]:
        """Parse discontinuous offsets into lists of start positions and lengths."""
        ranges = offset_str.split(',')
        starts = []
        lengths = []
        
        for range_str in ranges:
            range_str = range_str.strip()
            if '-' in range_str:
                start, end = range_str.split('-')
                starts.append(int(start))
                lengths.append(int(end) - int(start))
        
        return starts, lengths
    
    @staticmethod
    def extract_mention_text(text: str, starts: List[int], lengths: List[int]) -> str:
        """Extract mention text from the document based on starts and lengths."""
        parts = []
        for start, length in zip(starts, lengths):
            if start + length <= len(text):
                parts.append(text[start:start+length])
        return "".join(parts) 