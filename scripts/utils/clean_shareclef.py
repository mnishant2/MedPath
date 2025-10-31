#!/usr/bin/env python3
"""ShareCLEF Dataset Cleaning Script"""

import os
import re
import glob
import shutil

def clean_text(text):
    """Clean text by removing unnecessary characters and normalizing whitespace."""
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

def parse_discontinuous_offsets(offset_str):
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

def extract_mention_text(text, starts, lengths):
    """Extract mention text from the document based on starts and lengths."""
    parts = []
    for start, length in zip(starts, lengths):
        if start + length <= len(text):
            parts.append(text[start:start+length])
    return "".join(parts)

def adjust_annotation_offsets(annotation_line, original_text, cleaned_text):
    """Adjust annotation offsets based on the cleaning performed."""
    try:
        parts = annotation_line.strip().split('|')
        if len(parts) < 2:
            return annotation_line, False, "Invalid annotation format"
        
        offset_str = parts[1]
        starts, lengths = parse_discontinuous_offsets(offset_str)
        
        # Find new positions for each segment
        new_starts = []
        new_lengths = []
        
        for start, length in zip(starts, lengths):
            # Count characters removed before this segment
            text_before = original_text[:start]
            cleaned_before = clean_text(text_before)
            chars_removed_before = len(text_before) - len(cleaned_before)
            
            # Calculate new start position
            new_start = start - chars_removed_before
            
            # Get the segment text
            segment_text = original_text[start:start+length]
            
            # Try to find the segment in the cleaned text
            search_start = max(0, new_start - 50)
            search_end = min(len(cleaned_text), new_start + length + 50)
            search_area = cleaned_text[search_start:search_end]
            
            pos = search_area.find(segment_text)
            if pos != -1:
                new_start = search_start + pos
                new_length = len(segment_text)
            else:
                # Fallback: search in the entire cleaned text
                pos = cleaned_text.find(segment_text)
                if pos != -1:
                    new_start = pos
                    new_length = len(segment_text)
                else:
                    return annotation_line, False, f"Could not find segment '{segment_text}'"
            
            new_starts.append(new_start)
            new_lengths.append(new_length)
        
        # Create new offset string
        if len(new_starts) == 1:
            new_offset_str = f"{new_starts[0]}-{new_starts[0] + new_lengths[0]}"
        else:
            ranges = [f"{start}-{start + length}" for start, length in zip(new_starts, new_lengths)]
            new_offset_str = ",".join(ranges)
        
        # Update the annotation line
        parts[1] = new_offset_str
        updated_line = "|".join(parts)
        
        return updated_line, True, ""
        
    except Exception as e:
        return annotation_line, False, f"Error: {str(e)}"

def process_document_pair(txt_file, ann_file, output_txt_dir, output_ann_dir):
    """Process a single document and its annotation file."""
    filename = os.path.basename(txt_file)
    
    try:
        # Read and clean text
        with open(txt_file, 'r', encoding='utf-8') as f:
            original_text = f.read()
        
        cleaned_text = clean_text(original_text)
        
        # Process annotations if they exist
        annotation_results = []
        if os.path.exists(ann_file):
            with open(ann_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        updated_line, success, error = adjust_annotation_offsets(
                            line, original_text, cleaned_text
                        )
                        annotation_results.append({
                            'updated': updated_line,
                            'success': success,
                            'error': error
                        })
        
        # Save cleaned text
        output_txt_path = os.path.join(output_txt_dir, filename)
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        # Save updated annotations
        if annotation_results:
            ann_filename = os.path.basename(ann_file)
            output_ann_path = os.path.join(output_ann_dir, ann_filename)
            with open(output_ann_path, 'w', encoding='utf-8') as f:
                for result in annotation_results:
                    f.write(result['updated'])
        
        # Return statistics
        successful_annotations = sum(1 for r in annotation_results if r['success'])
        
        return {
            'filename': filename,
            'success': True,
            'chars_removed': len(original_text) - len(cleaned_text),
            'total_annotations': len(annotation_results),
            'successful_annotations': successful_annotations,
        }
        
    except Exception as e:
        return {
            'filename': filename,
            'success': False,
            'error': str(e)
        }

def process_corpus(corpus_dir, output_dir):
    """Process an entire corpus (training or test)."""
    corpus_name = os.path.basename(corpus_dir)
    print(f"\nProcessing {corpus_name} corpus...")
    
    # Determine subdirectories
    if 'training' in corpus_name.lower():
        txt_subdir = 'train'
        ann_subdir = 'train_ann'
    else:
        txt_subdir = 'test'
        ann_subdir = 'test_ann'
    
    txt_dir = os.path.join(corpus_dir, txt_subdir)
    ann_dir = os.path.join(corpus_dir, ann_subdir)
    
    # Create output directories
    output_txt_dir = os.path.join(output_dir, txt_subdir)
    output_ann_dir = os.path.join(output_dir, ann_subdir)
    os.makedirs(output_txt_dir, exist_ok=True)
    os.makedirs(output_ann_dir, exist_ok=True)
    
    # Process all text files
    txt_files = glob.glob(os.path.join(txt_dir, '*.txt'))
    results = []
    
    for txt_file in sorted(txt_files):
        filename = os.path.basename(txt_file)
        ann_filename = filename.replace('.txt', '.pipe.txt')
        ann_file = os.path.join(ann_dir, ann_filename)
        
        print(f"  Processing {filename}...")
        result = process_document_pair(txt_file, ann_file, output_txt_dir, output_ann_dir)
        results.append(result)
    
    # Generate summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    return {
        'corpus_name': corpus_name,
        'total_files': len(txt_files),
        'successful_files': len(successful),
        'failed_files': len(failed),
        'total_chars_removed': sum(r.get('chars_removed', 0) for r in successful),
        'total_annotations': sum(r.get('total_annotations', 0) for r in successful),
        'successful_annotations': sum(r.get('successful_annotations', 0) for r in successful),
    }

def main():
    """Main execution function."""
    # Configuration
    data_dir = "data/shareclef"
    output_dir = "data/shareclef_cleaned"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process both training and test corpora
    training_dir = os.path.join(data_dir, "training_corpus")
    test_dir = os.path.join(data_dir, "test_corpus")
    
    results = {}
    
    if os.path.exists(training_dir):
        output_training_dir = os.path.join(output_dir, "training_corpus")
        results['training'] = process_corpus(training_dir, output_training_dir)
    
    if os.path.exists(test_dir):
        output_test_dir = os.path.join(output_dir, "test_corpus")
        results['test'] = process_corpus(test_dir, output_test_dir)
    
    # Copy other files (like CSV files)
    other_files = glob.glob(os.path.join(data_dir, "*.csv"))
    for file_path in other_files:
        filename = os.path.basename(file_path)
        shutil.copy2(file_path, os.path.join(output_dir, filename))
        print(f"Copied {filename}")
    
    # Print summary
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    
    for corpus_name, summary in results.items():
        print(f"\n{corpus_name.upper()} CORPUS:")
        print(f"  Files processed: {summary['successful_files']}/{summary['total_files']}")
        print(f"  Characters removed: {summary['total_chars_removed']:,}")
        print(f"  Annotations processed: {summary['successful_annotations']}/{summary['total_annotations']}")
        if summary['total_annotations'] > 0:
            success_rate = summary['successful_annotations'] / summary['total_annotations']
            print(f"  Annotation success rate: {success_rate:.2%}")
    
    print(f"\nCleaned dataset saved to: {output_dir}")
    print("Run verify_shareclef_cleaning.py to validate the results.")

if __name__ == "__main__":
    main() 