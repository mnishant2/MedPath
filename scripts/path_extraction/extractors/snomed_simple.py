"""
SNOMED CT extractor using your exact working code.
"""

import time
import requests
from typing import List, Dict, Optional, Any
from .base_extractor import BaseExtractor, ExtractorRegistry


class SnomedExtractor(BaseExtractor):
    """SNOMED extractor using your working snomed_ct.py approach."""
    
    def __init__(self, config: Dict[str, Any], vocab_name: str):
        super().__init__(config, vocab_name)
        self.base_url = "https://snowstorm.mi-x.nl/"
        
    def validate_code(self, code: str) -> bool:
        """Validate SNOMED code format."""
        if not code or not isinstance(code, str):
            return False
        return code.strip().isdigit() and 6 <= len(code.strip()) <= 18
        
    def extract_paths(self, code: str, tty: Optional[str] = None, cui: Optional[str] = None) -> List[List[Dict[str, str]]]:
        """Extract paths using your working approach."""
        if not self.validate_code(code):
            return []
            
        try:
            # Get concept name first
            concept_name = self._get_concept_name(code)
            if not concept_name:
                return []
                
            # Use your exact working method
            result = self._get_ancestry_paths(code, concept_name)
            return result.get('paths', [])
            
        except Exception as e:
            self.logger.error(f"Error extracting paths for {code}: {str(e)}")
            return []
    
    def _get_concept_name(self, sctid: str) -> Optional[str]:
        """Get concept name."""
        try:
            url = f"{self.base_url}browser/MAIN/concepts/{sctid}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 404:
                return None
                
            response.raise_for_status()
            data = response.json()
            
            if not data.get('active', False):
                return None
                
            return data.get('pt', {}).get('term', 'Unknown')
            
        except Exception:
            return None
    
    def _make_request(self, url: str, **kwargs) -> Optional[Any]:
        """Make request using your approach."""
        try:
            kwargs.setdefault('timeout', 30)
            response = requests.get(url, **kwargs)
            response.raise_for_status()
            if "application/json" in response.headers.get("Content-Type", ""):
                return response.json()
            return response.text
        except requests.exceptions.RequestException as e:
            # Don't log common errors
            pass
        return None

    def _get_snomed_parents(self, sctid: str) -> List[Dict[str, str]]:
        """Get parents using your exact approach."""
        data = self._make_request(f"{self.base_url}browser/MAIN/concepts/{sctid}/parents", params={"form": "inferred"})
        
        if isinstance(data, list):
            return [{"code": p["conceptId"], "name": p.get("pt", {}).get("term", "N/A")} for p in data]
        elif isinstance(data, dict) and 'items' in data:
            return [{"code": p["conceptId"], "name": p.get("pt", {}).get("term", "N/A")} for p in data['items']]
        
        return []

    def _get_ancestry_paths(self, start_id: str, start_name: str) -> Dict[str, Any]:
        """Get ancestry paths using your exact working approach."""
        all_paths = []
        to_process = [(start_id, [{"code": start_id, "name": start_name}])]
        processed_paths = set()
        
        iterations = 0
        while to_process and iterations < 500:
            iterations += 1
            current_id, current_path = to_process.pop(0)
            
            time.sleep(0.1)  # Rate limit like your code
            parents = self._get_snomed_parents(current_id)

            if not parents:
                all_paths.append(current_path[::-1])  # Reverse for root-to-leaf
            else:
                for parent in parents:
                    p_id, p_name = parent.get("code"), parent.get("name")
                    if p_id:
                        path_key = tuple(p['code'] for p in current_path) + (p_id,)
                        if path_key not in processed_paths:
                            new_path = current_path + [{"code": p_id, "name": p_name}]
                            processed_paths.add(path_key)
                            to_process.append((p_id, new_path))
        
        return {"native_code": start_id, "paths": all_paths}


# Register the extractor  
ExtractorRegistry.register('SNOMED_CT', SnomedExtractor)