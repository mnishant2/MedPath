"""
LCSH (Library of Congress Subject Headings) hierarchical path extractor.
Uses local JSONLD file to extract SKOS broader relationships.
"""

import json
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
from collections import defaultdict

from .base_extractor import BaseExtractor, ExtractorRegistry


class LCSHExtractor(BaseExtractor):
    """Extractor for LCSH hierarchical paths using JSONLD SKOS data."""
    
    def __init__(self, config: Dict[str, Any], vocab_name: str):
        super().__init__(config, vocab_name)
        
        # Get LCSH-specific configuration
        lcsh_config = config.get('local_files', {}).get('lcsh', {})
        self.jsonld_file = lcsh_config.get('jsonld_file', 'subjects.skosrdf.jsonld')
        
        # Initialize data structures
        self.subjects = {}
        self.broader_map = defaultdict(list)  # child -> [parents]
        
        # Load LCSH data
        self._load_lcsh_data()
        
        self.logger.info(f"Initialized LCSH extractor with {len(self.subjects):,} subjects")
        
    def validate_code(self, code: str) -> bool:
        """
        Validate LCSH code format.
        LCSH codes are alphanumeric strings from Library of Congress.
        """
        if not code or not isinstance(code, str):
            return False
        
        # LCSH codes are typically alphanumeric strings
        code = code.strip()
        return bool(code and len(code) > 0)
        
    def extract_paths(self, code: str, tty: Optional[str] = None, cui: Optional[str] = None) -> List[List[Dict[str, str]]]:
        """
        Extract hierarchical paths for an LCSH code.
        
        Args:
            code: LCSH subject code
            tty: Not used for LCSH
            cui: CUI for tracking statistics (optional)
            
        Returns:
            List of paths from root to subject, each path as list of {'code', 'name'} dicts
        """
        if not self.validate_code(code):
            self.logger.warning(f"Invalid LCSH code format: {code}")
            if cui:
                self.track_cui_code_result(cui, code, 'error')
            return []
            
        try:
            paths = self._build_paths(code)
            
            # Update statistics
            if paths and cui:
                self.track_cui_code_result(cui, code, 'success', len(paths))
            elif cui:
                self.track_cui_code_result(cui, code, 'not_found')
                
            return paths
            
        except FileNotFoundError as e:
            self.logger.error(f"LCSH JSONLD file not found for {code}: {str(e)}")
            if cui:
                self.track_cui_code_result(cui, code, 'error')
            raise
        except Exception as e:
            self.logger.error(f"Error extracting LCSH paths for {code}: {str(e)}")
            if cui:
                self.track_cui_code_result(cui, code, 'error')
            raise
            
    def _load_lcsh_data(self):
        """Load LCSH data from JSONLD file."""
        jsonld_path = Path(self.jsonld_file)
        
        if not jsonld_path.exists():
            # Try different locations
                possible_paths = [
                    Path.cwd() / self.jsonld_file,
                    Path(__file__).parents[3] / "path_data" / self.jsonld_file
                ]
            
            for path in possible_paths:
                if path.exists():
                    jsonld_path = path
                    break
            else:
                raise FileNotFoundError(f"LCSH JSONLD file not found: {self.jsonld_file}")
        
        self.logger.info(f"Loading LCSH data from {jsonld_path}")
        
        try:
            self._load_jsonld_file(jsonld_path)
        except Exception as e:
            self.logger.error(f"Error loading LCSH JSONLD file: {e}")
            raise
            
    def _load_jsonld_file(self, file_path: Path):
        """Load and index the LCSH JSONLD file."""
        all_concepts = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Handle different JSONLD structures
                    if isinstance(data, dict) and '@graph' in data:
                        # Each line contains a document with @graph
                        for concept in data['@graph']:
                            all_concepts.append(concept)
                    elif isinstance(data, list):
                        # Line contains array of concepts
                        all_concepts.extend(data)
                    else:
                        # Single concept per line
                        all_concepts.append(data)
                        
                    line_count += 1
                    if line_count % 10000 == 0:
                        self.logger.info(f"Processed {line_count} lines, found {len(all_concepts)} concepts so far...")
                        
                except json.JSONDecodeError as json_err:
                    self.logger.warning(f"JSON decode error on line {line_count + 1}: {json_err}")
                    continue
        
        self.logger.info(f"Processing {len(all_concepts)} concepts...")
        
        for concept in all_concepts:
            self._process_concept(concept)
            
        self.logger.info(f"Loaded {len(self.subjects)} LCSH subjects with {len(self.broader_map)} broader relationships")
        
    def _process_concept(self, concept: Dict):
        """Process individual LCSH concept."""
        # Get subject ID from @id
        subject_id = concept.get('@id', '')
        if not subject_id or 'authorities/subjects/' not in subject_id:
            return
        
        code = subject_id.split('/')[-1]
        
        # Get English preferred label
        pref_label = self._extract_english_label(concept, 'skos:prefLabel')
        if not pref_label:
            pref_label = self._extract_english_label(concept, 'http://www.w3.org/2004/02/skos/core#prefLabel')
        if not pref_label:
            pref_label = self._extract_english_label(concept, 'http://www.loc.gov/mads/rdf/v1#authoritativeLabel')
        if not pref_label:
            pref_label = code
        
        # Store subject info
        self.subjects[code] = {
            'code': code,
            'name': pref_label,
            'uri': subject_id
        }
        
        # Extract broader relationships
        broader_terms = []
        for broader_key in ['skos:broader', 'http://www.w3.org/2004/02/skos/core#broader']:
            if broader_key in concept:
                broader_data = concept[broader_key]
                if isinstance(broader_data, list):
                    for broader in broader_data:
                        broader_uri = self._extract_uri(broader)
                        if broader_uri:
                            broader_terms.append(broader_uri)
                else:
                    broader_uri = self._extract_uri(broader_data)
                    if broader_uri:
                        broader_terms.append(broader_uri)
        
        # Build broader mapping
        if broader_terms:
            broader_codes = []
            for uri in broader_terms:
                if 'authorities/subjects/' in uri:
                    broader_code = uri.split('/')[-1]
                    broader_codes.append(broader_code)
            
            if broader_codes:
                self.broader_map[code] = broader_codes
                
    def _extract_english_label(self, concept: Dict, label_key: str) -> str:
        """Extract English label from concept."""
        if label_key not in concept:
            return ""
        
        labels = concept[label_key]
        if not isinstance(labels, list):
            labels = [labels]
        
        # Try to find English label
        for label in labels:
            if isinstance(label, dict):
                if label.get('@language') == 'en' and '@value' in label:
                    return label['@value']
                elif '@value' in label and '@language' not in label:
                    return label['@value']
            elif isinstance(label, str):
                return label
        
        return ""
        
    def _extract_uri(self, broader_data):
        """Extract URI from broader relationship."""
        if isinstance(broader_data, dict):
            return broader_data.get('@id', '')
        elif isinstance(broader_data, str):
            return broader_data
        return ""
        
    def _build_paths(self, code: str, current_path: Optional[List[Dict]] = None, 
                    visited: Optional[Set[str]] = None) -> List[List[Dict[str, str]]]:
        """
        Build hierarchical paths for LCSH code using DFS.
        
        Args:
            code: LCSH subject code
            current_path: Current path being built
            visited: Set of visited codes to prevent cycles
            
        Returns:
            List of paths from root to subject
        """
        if current_path is None:
            current_path = []
        if visited is None:
            visited = set()
            
        # Prevent cycles and excessive depth
        if code in visited or len(current_path) > self.max_path_length:
            return []
            
        # Check if subject exists
        if code not in self.subjects:
            self.logger.warning(f"LCSH subject not found: {code}")
            unknown_node = {'code': code, 'name': f'Unknown LCSH Subject {code}'}
            return [[unknown_node] + current_path] if not current_path else []
            
        visited.add(code)
        
        # Get subject information
        subject_info = self.subjects[code]
        current_node = {'code': code, 'name': subject_info['name']}
        
        # Get broader terms (parents)
        broader_codes = self.broader_map.get(code, [])
        
        if not broader_codes:
            # This is a root subject - return complete path
            complete_path = [current_node] + current_path
            return [list(reversed(complete_path))]  # Reverse to go from root to leaf
            
        # Recursively build paths through all broader terms
        all_paths = []
        for broader_code in broader_codes:
            if broader_code not in visited:
                broader_paths = self._build_paths(
                    broader_code,
                    [current_node] + current_path,
                    visited.copy()
                )
                all_paths.extend(broader_paths)
                
        # Limit number of paths
        if len(all_paths) > self.max_paths_per_code:
            self.logger.warning(f"LCSH subject {code} has {len(all_paths)} paths, truncating")
            all_paths = all_paths[:self.max_paths_per_code]
            
        return all_paths


# Register the extractor
ExtractorRegistry.register('LCH_NW', LCSHExtractor)