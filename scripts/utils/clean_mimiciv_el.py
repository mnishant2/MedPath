#!/usr/bin/env python3
"""
MIMICIV_EL Dataset Cleaning Script

Cleans the MIMICIV_EL dataset by removing stray characters and normalizing whitespace
while preserving annotation offset information.
"""

import os
import pandas as pd
import re

def clean_text(text):
    """Clean text by removing anonymization placeholders and normalizing whitespace."""
    if not isinstance(text, str):
        return text
    
    # Remove anonymization placeholders (any sequence of 3 or more underscores)
    text = re.sub(r'_{3,}', '', text)
    
    # Remove standalone underscore placeholders that might be names/dates
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
    text = re.sub(r'  +', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n +', '\n', text)  # Space at beginning of lines
    text = re.sub(r' +\n', '\n', text)  # Space at end of lines
    
    return text.strip()

def calculate_offset_mapping(original_text, cleaned_text):
    """Calculate mapping from original positions to cleaned positions."""
    mapping = []
    
    # Create a more sophisticated mapping by tracking character-by-character changes
    orig_pos = 0
    clean_pos = 0
    
    while orig_pos < len(original_text):
        # Find the current position in cleaned text
        if clean_pos < len(cleaned_text) and orig_pos < len(original_text):
            # Check if characters match
            if original_text[orig_pos] == cleaned_text[clean_pos]:
                mapping.append(clean_pos)
                orig_pos += 1
                clean_pos += 1
            else:
                # Character was removed or changed, try to find next match
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

def adjust_annotation_offsets(annotations_df, text_mappings):
    """Adjust annotation offsets based on text cleaning."""
    updated_annotations = []
    
    for _, row in annotations_df.iterrows():
        note_id = row['note_id']
        start = int(row['start'])
        end = int(row['end'])
        concept_id = row['concept_id']
        
        if note_id in text_mappings:
            mapping = text_mappings[note_id]
            
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
            
            updated_annotations.append({
                'note_id': note_id,
                'start': new_start,
                'end': new_end,
                'concept_id': concept_id,
                'original_start': start,
                'original_end': end
            })
        else:
            updated_annotations.append({
                'note_id': note_id,
                'start': start,
                'end': end,
                'concept_id': concept_id,
                'original_start': start,
                'original_end': end
            })
    
    return pd.DataFrame(updated_annotations)

def process_dataset(data_dir="data/MIMICIV_EL", output_dir="data/MIMICIV_EL_cleaned_no_placeholders"):
    """Process the entire MIMICIV_EL dataset."""
    print("Loading MIMICIV_EL dataset...")
    
    notes_file = os.path.join(data_dir, "mimic-iv_notes_training_set.csv")
    annotations_file = os.path.join(data_dir, "train_annotations.csv")
    
    if not os.path.exists(notes_file):
        raise FileNotFoundError(f"Notes file not found: {notes_file}")
    
    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    notes_df = pd.read_csv(notes_file)
    annotations_df = pd.read_csv(annotations_file)
    
    print(f"Loaded {len(notes_df)} notes and {len(annotations_df)} annotations")
    
    # Clean notes and calculate mappings
    print("Cleaning notes and calculating offset mappings...")
    cleaned_notes = []
    text_mappings = {}
    total_chars_removed = 0
    
    for idx, row in notes_df.iterrows():
        note_id = row['note_id']
        original_text = row['text']
        
        if pd.isna(original_text):
            cleaned_text = ""
            mapping = []
        else:
            cleaned_text = clean_text(str(original_text))
            mapping = calculate_offset_mapping(str(original_text), cleaned_text)
            total_chars_removed += len(str(original_text)) - len(cleaned_text)
        
        text_mappings[note_id] = mapping
        cleaned_notes.append({'note_id': note_id, 'text': cleaned_text})
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(notes_df)} notes...")
    
    cleaned_notes_df = pd.DataFrame(cleaned_notes)
    
    # Adjust annotation offsets
    print("Adjusting annotation offsets...")
    adjusted_annotations_df = adjust_annotation_offsets(annotations_df, text_mappings)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save cleaned data
    print("Saving cleaned data...")
    output_notes_file = os.path.join(output_dir, "mimic-iv_notes_training_set_cleaned.csv")
    output_annotations_file = os.path.join(output_dir, "train_annotations_cleaned.csv")
    
    cleaned_notes_df.to_csv(output_notes_file, index=False)
    adjusted_annotations_df[['note_id', 'start', 'end', 'concept_id']].to_csv(
        output_annotations_file, index=False
    )
    
    # Save debug version with original offsets
    debug_annotations_file = os.path.join(output_dir, "train_annotations_with_original.csv")
    adjusted_annotations_df.to_csv(debug_annotations_file, index=False)
    
    # Calculate statistics
    annotation_changes = sum(1 for _, row in adjusted_annotations_df.iterrows() 
                            if row['start'] != row['original_start'] or row['end'] != row['original_end'])
    
    return {
        'total_notes': len(notes_df),
        'total_annotations': len(annotations_df),
        'total_chars_removed': total_chars_removed,
        'annotations_changed': annotation_changes,
        'annotation_change_rate': annotation_changes / len(annotations_df) if len(annotations_df) > 0 else 0,
        'output_directory': output_dir
    }

def main():
    """Main execution function."""
    data_dir = "data/MIMICIV_EL"
    output_dir = "data/MIMICIV_EL_cleaned_no_placeholders"
    
    try:
        print("Starting MIMICIV_EL dataset cleaning...")
        results = process_dataset(data_dir, output_dir)
        
        print("\n" + "="*60)
        print("CLEANING SUMMARY")
        print("="*60)
        print(f"Total notes processed: {results['total_notes']:,}")
        print(f"Total annotations processed: {results['total_annotations']:,}")
        print(f"Total characters removed: {results['total_chars_removed']:,}")
        print(f"Annotations with changed offsets: {results['annotations_changed']:,}")
        print(f"Annotation change rate: {results['annotation_change_rate']:.2%}")
        print(f"Cleaned dataset saved to: {results['output_directory']}")
        print("\nRun verify_mimiciv_el_cleaning.py to validate the results.")
        
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        raise

if __name__ == "__main__":
    main() 