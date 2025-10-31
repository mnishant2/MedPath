#!/usr/bin/env python3
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
import glob

def clean_text(text):
    """Clean text by removing extra whitespace while preserving the overall structure."""
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    # Remove spaces before punctuation
    text = re.sub(r' ([.,;:!?)])', r'\1', text)
    # Remove spaces after opening parentheses
    text = re.sub(r'(\() ', r'\1', text)
    # Remove trailing whitespace at the end of lines
    text = re.sub(r' +\n', '\n', text)
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    return text

def parse_discontinuous_offsets(start_str, len_str):
    """Parse discontinuous offsets into lists of start positions and lengths."""
    starts = []
    lengths = []
    
    if ',' in start_str:
        starts = [int(s.strip()) for s in start_str.split(',')]
    else:
        starts = [int(start_str)]
        
    if ',' in len_str:
        lengths = [int(l.strip()) for l in len_str.split(',')]
    else:
        lengths = [int(len_str)]
        
    # Ensure both lists have the same length
    if len(starts) != len(lengths):
        raise ValueError(f"Mismatch between number of start positions ({len(starts)}) and lengths ({len(lengths)})")
        
    return starts, lengths

def extract_mention_text(text, starts, lengths):
    """Extract mention text from the document based on starts and lengths."""
    parts = []
    for start, length in zip(starts, lengths):
        if start + length <= len(text):
            parts.append(text[start:start+length])
    return "".join(parts)

def split_discontinuous_mention(mention, text):
    """
    Split a discontinuous mention into multiple continuous mentions.
    
    Args:
        mention: The original mention element
        text: The section text where the mention appears
    
    Returns:
        A list of new mention dictionaries with continuous offsets
    """
    start_str = mention.get('start', '0')
    len_str = mention.get('len', '0')
    mention_text = mention.get('str', '')
    
    # If it's not discontinuous, return the original mention
    if ',' not in start_str:
        return [mention]
    
    try:
        starts, lengths = parse_discontinuous_offsets(start_str, len_str)
        
        # Extract each segment text
        segments = []
        for start, length in zip(starts, lengths):
            if start + length <= len(text):
                segment_text = text[start:start+length]
                segments.append((start, length, segment_text))
        
        # Create a new mention for each segment
        continuous_mentions = []
        for i, (start, length, segment_text) in enumerate(segments):
            new_mention = ET.Element('Mention')
            for key, value in mention.attrib.items():
                new_mention.set(key, value)
            
            # Update the start, length, and text for this segment
            new_mention.set('start', str(start))
            new_mention.set('len', str(length))
            new_mention.set('str', segment_text)
            new_mention.set('part', str(i+1))  # Mark which part of the original mention
            new_mention.set('original_id', mention.get('id', ''))  # Save original ID
            new_mention.set('id', f"{mention.get('id', '')}_{i+1}")  # Create a new ID
            
            continuous_mentions.append(new_mention)
        
        return continuous_mentions
    
    except Exception as e:
        print(f"Error splitting mention: {str(e)}")
        return [mention]  # Return original in case of error

def adjust_mention_offsets(mention, old_text, new_text):
    """
    Adjust mention offsets based on the cleaning performed.
    Returns adjusted mention with updated start position and a verification flag.
    """
    start_str = mention.get('start', '0')
    len_str = mention.get('len', '0')
    mention_text = mention.get('str', '')
    
    try:
        # Handle both continuous and discontinuous mentions
        starts, lengths = parse_discontinuous_offsets(start_str, len_str)
        
        # Extract the original text from the document
        original_text = extract_mention_text(old_text, starts, lengths)
        
        # If the original text doesn't match the mention text, there's an issue
        if original_text != mention_text:
            return None, False, f"Original text '{original_text}' doesn't match mention text '{mention_text}'"
        
        # For each segment, find its new position in the cleaned text
        new_starts = []
        new_lengths = []
        extracted_parts = []
        
        for start, length in zip(starts, lengths):
            # Count characters removed before segment start
            text_before = old_text[:start]
            cleaned_before = clean_text(text_before)
            removed_before = len(text_before) - len(cleaned_before)
            
            # Calculate new start position
            new_start = start - removed_before
            
            # Calculate how many characters might be removed within the segment
            segment_text = old_text[start:start+length]
            cleaned_segment = clean_text(segment_text)
            new_length = len(cleaned_segment)
            
            # Extract text at new position to verify
            if new_start + new_length <= len(new_text):
                extracted_text = new_text[new_start:new_start+new_length]
                
                # If this doesn't match, try a fuzzy search nearby
                if extracted_text != cleaned_segment:
                    # Search a window around the expected position
                    window_size = 50
                    search_start = max(0, new_start - window_size)
                    search_end = min(len(new_text), new_start + new_length + window_size)
                    search_area = new_text[search_start:search_end]
                    
                    pos = search_area.find(cleaned_segment)
                    if pos != -1:
                        new_start = search_start + pos
                        extracted_text = cleaned_segment
                    else:
                        pos = search_area.find(segment_text)
                        if pos != -1:
                            new_start = search_start + pos
                            extracted_text = segment_text
                        else:
                            return None, False, f"Couldn't find segment in cleaned document"
                
                new_starts.append(new_start)
                new_lengths.append(len(extracted_text))
                extracted_parts.append(extracted_text)
            else:
                return None, False, f"New position out of bounds: {new_start}+{new_length} > {len(new_text)}"
        
        # Verify the entire extracted text matches the mention text
        full_extracted = "".join(extracted_parts)
        if full_extracted != mention_text:
            # Try a direct search in the full text if the pieced-together approach fails
            pos = new_text.find(mention_text)
            if pos != -1:
                # If found directly, use that instead
                new_starts = [pos]
                new_lengths = [len(mention_text)]
                full_extracted = mention_text
            else:
                return None, False, f"Extracted text '{full_extracted}' doesn't match mention text '{mention_text}'"
        
        # Create new mention with adjusted offsets
        new_mention = mention.attrib.copy()
        new_mention['original_start'] = start_str
        new_mention['original_len'] = len_str
        
        # Format new starts and lengths
        if len(new_starts) == 1:
            new_mention['start'] = str(new_starts[0])
            new_mention['len'] = str(new_lengths[0])
        else:
            new_mention['start'] = ",".join(str(s) for s in new_starts)
            new_mention['len'] = ",".join(str(l) for l in new_lengths)
        
        return new_mention, True, ""
        
    except Exception as e:
        return None, False, f"Error adjusting offsets: {str(e)}"

def process_xml_file(file_path):
    """
    Process a single XML file:
    1. Load and parse XML
    2. Extract text sections
    3. Clean text
    4. Split discontinuous mentions into continuous ones
    5. Adjust mention offsets
    6. Verify mentions can be extracted correctly
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Get all text sections
        text_elem = root.find('.//Text')
        if text_elem is None:
            print(f"No Text element found in {file_path}")
            return None
        
        # Process sections and their text
        sections = {}
        section_elems = text_elem.findall('.//Section')
        
        # If no sections are found, use the whole text element
        if not section_elems:
            full_text = "".join(text_elem.itertext())
            sections["main"] = full_text
        else:
            for section in section_elems:
                section_id = section.get('id', '')
                section_text = "".join(section.itertext())
                sections[section_id] = section_text
        
        # Clean each section's text
        cleaned_sections = {section_id: clean_text(text) for section_id, text in sections.items()}
        
        # Get all mentions
        mentions_elem = root.find('.//Mentions')
        if mentions_elem is None:
            # Some files might not have mentions
            mentions = []
        else:
            # Process mentions and split discontinuous ones
            original_mentions = []
            split_mentions = []
            discontinuous_count = 0
            
            for mention in mentions_elem.findall('.//Mention'):
                original_mentions.append(mention)
                section_id = mention.get('section')
                
                # Skip mentions without a valid section reference
                if section_id not in sections:
                    split_mentions.append(mention)
                    continue
                
                # Check if this is a discontinuous mention
                start_str = mention.get('start', '0')
                if ',' in start_str:
                    discontinuous_count += 1
                    # Split the discontinuous mention
                    splits = split_discontinuous_mention(mention, sections[section_id])
                    split_mentions.extend(splits)
                else:
                    # Keep continuous mentions as is
                    split_mentions.append(mention)
            
            mentions = split_mentions
        
        # Process each mention within its section context
        adjusted_mentions = []
        verification_results = []
        
        for mention in mentions:
            section_id = mention.get('section')
            mention_type = mention.get('type', '')
            mention_id = mention.get('id', '')
            is_part = 'part' in mention.attrib
            
            # Skip mentions without a valid section reference
            if section_id not in sections:
                verification_results.append({
                    'mention_id': mention_id,
                    'mention_type': mention_type,
                    'is_part': is_part,
                    'success': False,
                    'error': f"Section {section_id} not found",
                    'section': section_id,
                    'text': mention.get('str', ''),
                })
                continue
            
            # Get the original and cleaned section text
            original_section_text = sections[section_id]
            cleaned_section_text = cleaned_sections[section_id]
            
            # Adjust the offsets for this mention
            adjusted_mention, success, error_msg = adjust_mention_offsets(
                mention, original_section_text, cleaned_section_text
            )
            
            # Store results
            if success:
                adjusted_mentions.append(adjusted_mention)
                
                # Verify the extraction
                start_str = adjusted_mention.get('start')
                len_str = adjusted_mention.get('len')
                mention_str = adjusted_mention.get('str')
                
                try:
                    starts, lengths = parse_discontinuous_offsets(start_str, len_str)
                    extracted = extract_mention_text(cleaned_section_text, starts, lengths)
                    match = extracted == mention_str
                    
                    verification_results.append({
                        'mention_id': mention_id,
                        'mention_type': mention_type,
                        'is_part': is_part,
                        'success': match,
                        'error': "" if match else f"Verification failed: '{extracted}' != '{mention_str}'",
                        'section': section_id,
                        'text': mention_str,
                        'original_offset': mention.get('start'),
                        'new_offset': start_str,
                        'extracted_text': extracted
                    })
                except Exception as e:
                    verification_results.append({
                        'mention_id': mention_id,
                        'mention_type': mention_type,
                        'is_part': is_part,
                        'success': False,
                        'error': f"Verification error: {str(e)}",
                        'section': section_id,
                        'text': mention_str,
                        'original_offset': mention.get('start'),
                        'new_offset': start_str
                    })
            else:
                verification_results.append({
                    'mention_id': mention_id,
                    'mention_type': mention_type,
                    'is_part': is_part,
                    'success': False,
                    'error': error_msg,
                    'section': section_id,
                    'text': mention.get('str', ''),
                    'original_offset': mention.get('start')
                })
        
        return {
            'file_name': os.path.basename(file_path),
            'sections': sections,
            'cleaned_sections': cleaned_sections,
            'adjusted_mentions': adjusted_mentions,
            'verification_results': verification_results,
            'discontinuous_count': discontinuous_count
        }
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_dataset(data_dir="data/ADR"):
    """Process all XML files in the dataset directory."""
    results = {
        'train': [],
        'test': []
    }
    
    # Process training files
    train_dir = os.path.join(data_dir, "train_xml")
    if os.path.exists(train_dir):
        for xml_file in glob.glob(os.path.join(train_dir, "*.xml")):
            result = process_xml_file(xml_file)
            if result:
                results['train'].append(result)
    
    # Process test files
    test_dir = os.path.join(data_dir, "gold_xml")
    if os.path.exists(test_dir):
        for xml_file in glob.glob(os.path.join(test_dir, "*.xml")):
            result = process_xml_file(xml_file)
            if result:
                results['test'].append(result)
    
    return results

def generate_summary(results):
    """Generate a summary of the processing results."""
    summary = {
        'total_files': len(results['train']) + len(results['test']),
        'train_files': len(results['train']),
        'test_files': len(results['test']),
        'total_mentions': 0,
        'successfully_adjusted': 0,
        'failed_adjustments': 0,
        'successful_verifications': 0,
        'failed_verifications': 0,
        'discontinuous_mentions': 0,
        'split_parts': 0,
        'files_with_errors': [],
        'mentions_by_type': defaultdict(lambda: {'total': 0, 'success': 0, 'fail': 0})
    }
    
    for split in ['train', 'test']:
        for file_result in results[split]:
            success_count = 0
            total_mentions = len(file_result['verification_results'])
            part_count = sum(1 for v in file_result['verification_results'] if v.get('is_part', False))
            
            # Count discontinuous mentions
            summary['discontinuous_mentions'] += file_result.get('discontinuous_count', 0)
            summary['split_parts'] += part_count
            
            for v in file_result['verification_results']:
                mention_type = v.get('mention_type', 'unknown')
                summary['mentions_by_type'][mention_type]['total'] += 1
                
                if v['success']:
                    success_count += 1
                    summary['successful_verifications'] += 1
                    summary['mentions_by_type'][mention_type]['success'] += 1
                else:
                    summary['failed_verifications'] += 1
                    summary['mentions_by_type'][mention_type]['fail'] += 1
            
            # Track files with any failures
            if success_count < total_mentions:
                summary['files_with_errors'].append(file_result['file_name'])
            
            summary['total_mentions'] += total_mentions
            summary['successfully_adjusted'] += success_count
            summary['failed_adjustments'] += (total_mentions - success_count)
    
    # Calculate success rates
    if summary['total_mentions'] > 0:
        summary['adjustment_success_rate'] = summary['successfully_adjusted'] / summary['total_mentions'] * 100
        summary['verification_success_rate'] = summary['successful_verifications'] / summary['total_mentions'] * 100
    else:
        summary['adjustment_success_rate'] = 0
        summary['verification_success_rate'] = 0
    
    return summary

def fix_text_content(element, cleaned_text):
    """Properly update XML element text content with cleaned text."""
    # Clear all existing text and tail content
    element.text = None
    for child in element:
        child.tail = None
    
    # Set the cleaned text as the element's text
    element.text = cleaned_text

def save_cleaned_files(results, output_dir):
    """Save the cleaned XML files with adjusted mention offsets."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process both train and test results
    for split in ['train', 'test']:
        # Create split-specific output directory
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        for file_result in results[split]:
            try:
                # Load original XML
                original_file = os.path.join(
                    "data/ADR",
                    "train_xml" if split == "train" else "gold_xml", 
                    file_result['file_name']
                )
                
                tree = ET.parse(original_file)
                root = tree.getroot()
                
                # Update the text sections with cleaned text
                text_elem = root.find('.//Text')
                if text_elem is not None:
                    for section in text_elem.findall('.//Section'):
                        section_id = section.get('id', '')
                        if section_id in file_result['cleaned_sections']:
                            # Use proper method to update text content
                            fix_text_content(section, file_result['cleaned_sections'][section_id])
                
                # Get mentions element to update
                mentions_elem = root.find('.//Mentions')
                if mentions_elem is None:
                    continue
                
                # Clear all existing mentions
                for child in list(mentions_elem):
                    mentions_elem.remove(child)
                
                # Group adjusted mentions by original ID to handle split parts
                mentions_by_id = defaultdict(list)
                for mention in file_result['adjusted_mentions']:
                    # Check if this is a split part
                    original_id = mention.get('original_id', mention.get('id'))
                    mentions_by_id[original_id].append(mention)
                
                # Add adjusted mentions back to the XML
                for original_id, mentions in mentions_by_id.items():
                    if len(mentions) == 1 and 'part' not in mentions[0]:
                        # This is a regular (non-split) mention
                        mention_data = mentions[0]
                        mention_elem = ET.SubElement(mentions_elem, 'Mention')
                        
                        # Copy all non-temporary attributes
                        for key, value in mention_data.items():
                            if key not in ['original_start', 'original_len', 'original_id', 'part']:
                                mention_elem.set(key, value)
                    else:
                        # This is a split discontinuous mention, add each part as a separate mention
                        for i, mention_data in enumerate(mentions):
                            mention_elem = ET.SubElement(mentions_elem, 'Mention')
                            
                            # Copy all non-temporary attributes
                            for key, value in mention_data.items():
                                if key not in ['original_start', 'original_len', 'original_id']:
                                    mention_elem.set(key, value)
                
                # Save the updated XML
                output_file = os.path.join(split_dir, file_result['file_name'])
                tree.write(output_file, encoding='utf-8', xml_declaration=True)
                
            except Exception as e:
                print(f"Error saving cleaned file {file_result['file_name']}: {str(e)}")

def verify_output_files(results, output_dir):
    """Verify that the output files have correctly adjusted mentions."""
    verification_results = {
        'total_files': 0,
        'successful_files': 0,
        'failed_files': 0,
        'mention_verification': {
            'total': 0,
            'success': 0,
            'fail': 0
        },
        'files_with_errors': []
    }
    
    for split in ['train', 'test']:
        split_dir = os.path.join(output_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for file_result in results[split]:
            file_path = os.path.join(split_dir, file_result['file_name'])
            if not os.path.exists(file_path):
                continue
                
            verification_results['total_files'] += 1
            file_success = True
            
            try:
                # Parse the cleaned XML file
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # Get sections and mentions
                sections = {}
                for section in root.findall('.//Text//Section'):
                    section_id = section.get('id', '')
                    sections[section_id] = section.text or ""
                
                # Check each mention
                for mention in root.findall('.//Mentions//Mention'):
                    mention_id = mention.get('id', '')
                    section_id = mention.get('section', '')
                    mention_text = mention.get('str', '')
                    start_str = mention.get('start', '')
                    len_str = mention.get('len', '')
                    
                    verification_results['mention_verification']['total'] += 1
                    
                    if section_id not in sections:
                        file_success = False
                        verification_results['mention_verification']['fail'] += 1
                        continue
                    
                    section_text = sections[section_id]
                    
                    # Extract mention text using offsets
                    try:
                        starts, lengths = parse_discontinuous_offsets(start_str, len_str)
                        extracted_text = ""
                        
                        for start, length in zip(starts, lengths):
                            if start + length <= len(section_text):
                                extracted_text += section_text[start:start+length]
                        
                        # Check if extraction matches
                        if extracted_text == mention_text:
                            verification_results['mention_verification']['success'] += 1
                        else:
                            file_success = False
                            verification_results['mention_verification']['fail'] += 1
                    except Exception:
                        file_success = False
                        verification_results['mention_verification']['fail'] += 1
            
            except Exception as e:
                print(f"Error verifying file {file_path}: {str(e)}")
                file_success = False
            
            if file_success:
                verification_results['successful_files'] += 1
            else:
                verification_results['failed_files'] += 1
                verification_results['files_with_errors'].append(file_result['file_name'])
    
    return verification_results

def main():
    """Main function to process the ADR dataset."""
    print("Processing ADR dataset...")
    results = process_dataset()
    summary = generate_summary(results)
    
    print("\nProcessing Summary:")
    print(f"Total files processed: {summary['total_files']} (Train: {summary['train_files']}, Test: {summary['test_files']})")
    print(f"Total mentions: {summary['total_mentions']}")
    print(f"Discontinuous mentions found: {summary['discontinuous_mentions']} (split into {summary['split_parts']} parts)")
    print(f"Successfully adjusted mentions: {summary['successfully_adjusted']} ({summary['adjustment_success_rate']:.2f}%)")
    print(f"Failed adjustments: {summary['failed_adjustments']}")
    print(f"Successful verifications: {summary['successful_verifications']} ({summary['verification_success_rate']:.2f}%)")
    print(f"Failed verifications: {summary['failed_verifications']}")
    
    # Display mentions by type, focusing on AdverseReaction and DrugType
    print("\nMention Statistics by Type:")
    print(f"{'Type':<20} {'Total':<10} {'Success':<10} {'Fail':<10} {'Success %':<10}")
    print("-" * 60)
    
    for mention_type, counts in sorted(summary['mentions_by_type'].items()):
        total = counts['total']
        success = counts['success']
        fail = counts['fail']
        success_rate = (success / total * 100) if total > 0 else 0
        print(f"{mention_type:<20} {total:<10} {success:<10} {fail:<10} {success_rate:.2f}%")
    
    # Display files with errors
    if summary['files_with_errors']:
        print(f"\nFiles with errors ({len(summary['files_with_errors'])}):")
        for filename in summary['files_with_errors'][:10]:  # Show first 10
            print(f"  - {filename}")
        if len(summary['files_with_errors']) > 10:
            print(f"  ... and {len(summary['files_with_errors']) - 10} more")
    
    # Show a detailed example
    if results['train'] or results['test']:
        # Find a file with at least one successful mention to demonstrate
        example_file = None
        for split in ['train', 'test']:
            for file_result in results[split]:
                successful_mention = next((v for v in file_result['verification_results'] if v['success']), None)
                if successful_mention:
                    example_file = file_result
                    break
            if example_file:
                break
        
        if example_file:
            print(f"\nDetailed verification for {example_file['file_name']}:")
            
            # Show a sample of verifications (first 5)
            for i, verification in enumerate(example_file['verification_results'][:5]):
                mention_id = verification['mention_id']
                mention_text = verification['text']
                mention_type = verification.get('mention_type', 'unknown')
                is_part = verification.get('is_part', False)
                part_text = " (Split part)" if is_part else ""
                
                if verification['success']:
                    extracted_text = verification.get('extracted_text', '')
                    print(f"  Mention {mention_id} (Type: {mention_type}){part_text}: '{mention_text}'")
                    print(f"    Original offset: {verification['original_offset']}")
                    print(f"    New offset: {verification['new_offset']}")
                    print(f"    Extracted text: '{extracted_text}'")
                    print(f"    Match: {extracted_text == mention_text}")
                else:
                    print(f"  Mention {mention_id} (Type: {mention_type}){part_text}: '{mention_text}' - Failed: {verification['error']}")
    
    # Save cleaned XML files
    output_dir = "data/ADR_cleaned"
    print(f"\nSaving cleaned files to {output_dir}...")
    save_cleaned_files(results, output_dir)
    
    # Verify output files
    print("\nVerifying output files...")
    verification = verify_output_files(results, output_dir)
    
    print(f"Files verified: {verification['total_files']}")
    print(f"Files successfully verified: {verification['successful_files']}")
    print(f"Files with verification errors: {verification['failed_files']}")
    
    total_mentions = verification['mention_verification']['total']
    success_mentions = verification['mention_verification']['success']
    fail_mentions = verification['mention_verification']['fail']
    
    if total_mentions > 0:
        success_rate = success_mentions / total_mentions * 100
        print(f"Mentions verified: {total_mentions} (Success: {success_mentions}, Fail: {fail_mentions}, Success rate: {success_rate:.2f}%)")
    
    print("Done!")

if __name__ == "__main__":
    main()
