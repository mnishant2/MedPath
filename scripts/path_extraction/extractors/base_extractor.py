"""
Base extractor class for hierarchical path extraction from medical vocabularies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import yaml
from pathlib import Path


class BaseExtractor(ABC):
    """
    Abstract base class for all vocabulary extractors.
    
    Each extractor should implement the extract_paths method to return
    hierarchical paths in the standardized format.
    """
    
    def __init__(self, config: Dict[str, Any], vocab_name: str):
        """
        Initialize the extractor with configuration.
        
        Args:
            config: Configuration dictionary loaded from credentials.yaml
            vocab_name: Name of the vocabulary (e.g., 'snomed', 'mesh')
        """
        self.config = config
        self.vocab_name = vocab_name
        self.logger = self._setup_logger()
        
        # Rate limiting
        self.rate_limit = config.get('rate_limits', {}).get(vocab_name, 
                                                           config.get('rate_limits', {}).get('default', 0.1))
        self.last_request_time = 0
        
        # Error handling - Set proper limits as requested
        self.max_retries = config.get('error_handling', {}).get('max_retries', 3)
        self.retry_delay = config.get('error_handling', {}).get('retry_delay', 1.0)
        self.max_path_length = config.get('error_handling', {}).get('max_path_length', 200)  # Max length 200
        self.max_paths_per_code = config.get('error_handling', {}).get('max_paths_per_code', 2000)  # Max 2000 paths
        
        # Timeout settings
        self.api_timeout = config.get('timeouts', {}).get('api_timeout', 30)
        
        # Statistics tracking
        self.cui_stats = {}  # Track stats per CUI
        self.total_codes_processed = 0
        self.total_codes_with_paths = 0
        self.total_inactive_codes = 0
        self.total_api_errors = 0
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for this extractor."""
        logger = logging.getLogger(f"extractor.{self.vocab_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Create console handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _rate_limit(self):
        """Implement rate limiting between requests."""
        if self.rate_limit > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit:
                sleep_time = self.rate_limit - elapsed
                time.sleep(sleep_time)
        self.last_request_time = time.time()
        
    @abstractmethod
    def extract_paths(self, code: str, tty: Optional[str] = None) -> List[List[Dict[str, str]]]:
        """
        Extract hierarchical paths for a given code.
        
        Args:
            code: The vocabulary-specific code (e.g., SNOMED ID, MeSH descriptor)
            tty: Term type information (important for MedDRA and some others)
            
        Returns:
            List of paths, where each path is a list of dictionaries with 'code' and 'name' keys.
            Returns empty list if code not found or error occurs.
            
        Example return format:
        [
            [
                {'code': '138875005', 'name': 'SNOMED CT Concept'},
                {'code': '404684003', 'name': 'Clinical finding'},
                {'code': '34608000', 'name': 'Alanine aminotransferase measurement'}
            ],
            [
                {'code': '138875005', 'name': 'SNOMED CT Concept'},
                {'code': '362981000', 'name': 'Qualifier value'},
                {'code': '34608000', 'name': 'Alanine aminotransferase measurement'}
            ]
        ]
        """
        pass
        
    @abstractmethod
    def validate_code(self, code: str) -> bool:
        """
        Validate if a code is in the correct format for this vocabulary.
        
        Args:
            code: The code to validate
            
        Returns:
            True if code format is valid, False otherwise
        """
        pass
        
    def validate_cui(self, cui: str) -> bool:
        """
        Validate CUI format.
        
        Args:
            cui: The CUI to validate (should be in format C#######)
            
        Returns:
            True if CUI format is valid, False otherwise
        """
        if not cui or not isinstance(cui, str):
            return False
        
        cui = cui.strip()
        # CUI should start with C followed by 7 digits
        return len(cui) == 8 and cui.startswith('C') and cui[1:].isdigit()
        
    def get_extraction_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this extractor.
        
        Returns:
            Dictionary with extractor metadata
        """
        return {
            'vocabulary': self.vocab_name,
            'extractor_class': self.__class__.__name__,
            'rate_limit': self.rate_limit,
            'max_retries': self.max_retries,
            'max_path_length': self.max_path_length,
            'max_paths_per_code': self.max_paths_per_code
        }
        
    def extract_with_retry(self, code: str, tty: Optional[str] = None) -> Tuple[List[List[Dict[str, str]]], List[str]]:
        """
        Extract paths with retry logic and error handling.
        
        Args:
            code: The vocabulary-specific code
            tty: Term type information
            
        Returns:
            Tuple of (paths, errors) where errors is list of error messages
        """
        errors = []
        
        for attempt in range(self.max_retries + 1):
            try:
                self._rate_limit()
                paths = self.extract_paths(code, tty)
                
                # Validate and limit paths
                if paths and len(paths) > self.max_paths_per_code:
                    self.logger.warning(f"Code {code} has {len(paths)} paths, truncating to {self.max_paths_per_code}")
                    paths = paths[:self.max_paths_per_code]
                
                # Validate path lengths
                validated_paths = []
                for path in paths:
                    if len(path) <= self.max_path_length:
                        validated_paths.append(path)
                    else:
                        self.logger.warning(f"Path for code {code} has length {len(path)}, truncating to {self.max_path_length}")
                        validated_paths.append(path[:self.max_path_length])
                
                return validated_paths, errors
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(f"Error extracting paths for {code}: {error_msg}")
                
                if attempt < self.max_retries:
                    sleep_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"Failed to extract paths for {code} after {self.max_retries + 1} attempts")
                    
        return [], errors
    
    def track_cui_code_result(self, cui: str, code: str, result_type: str, path_count: int = 0):
        """
        Track the result for a specific code within a CUI.
        
        Args:
            cui: The CUI being processed
            code: The vocabulary code being processed  
            result_type: 'success', 'inactive', 'error', 'not_found'
            path_count: Number of paths extracted (for successful codes)
        """
        # Validate CUI format
        if not self.validate_cui(cui):
            self.logger.warning(f"Invalid CUI format: {cui}")
            return
            
        if cui not in self.cui_stats:
            self.cui_stats[cui] = {
                'total_codes': 0,
                'successful_codes': 0,
                'inactive_codes': 0,
                'error_codes': 0,
                'not_found_codes': 0,
                'total_paths': 0
            }
        
        self.cui_stats[cui]['total_codes'] += 1
        self.total_codes_processed += 1
        
        if result_type == 'success':
            self.cui_stats[cui]['successful_codes'] += 1
            self.cui_stats[cui]['total_paths'] += path_count
            self.total_codes_with_paths += 1
        elif result_type == 'inactive':
            self.cui_stats[cui]['inactive_codes'] += 1
            self.total_inactive_codes += 1
        elif result_type == 'error':
            self.cui_stats[cui]['error_codes'] += 1
            self.total_api_errors += 1
        elif result_type == 'not_found':
            self.cui_stats[cui]['not_found_codes'] += 1
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics for this extractor."""
        
        # Calculate CUI-level statistics
        cuis_with_paths = len([cui for cui, stats in self.cui_stats.items() if stats['successful_codes'] > 0])
        cuis_without_paths = len([cui for cui, stats in self.cui_stats.items() if stats['successful_codes'] == 0])
        
        return {
            'vocab_name': self.vocab_name,
            'rate_limit': self.rate_limit,
            'max_retries': self.max_retries,
            'max_path_length': self.max_path_length,
            'max_paths_per_code': self.max_paths_per_code,
            # Code-level statistics
            'total_codes_processed': self.total_codes_processed,
            'total_codes_with_paths': self.total_codes_with_paths,
            'total_inactive_codes': self.total_inactive_codes,
            'total_api_errors': self.total_api_errors,
            'code_success_rate': self.total_codes_with_paths / self.total_codes_processed if self.total_codes_processed > 0 else 0,
            # CUI-level statistics
            'total_cuis_processed': len(self.cui_stats),
            'cuis_with_paths': cuis_with_paths,
            'cuis_without_paths': cuis_without_paths,
            'cui_success_rate': cuis_with_paths / len(self.cui_stats) if len(self.cui_stats) > 0 else 0
        }
        
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file, defaults to standard location
            
        Returns:
            Configuration dictionary
        """
        if config_path is None:
            # Default path
            current_dir = Path(__file__).parent
            config_path = current_dir.parent.parent / "configs" / "credentials.yaml"
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


class ExtractorRegistry:
    """Registry to manage all available extractors."""
    
    _extractors = {}
    
    @classmethod
    def register(cls, vocab_name: str, extractor_class):
        """Register an extractor class for a vocabulary."""
        cls._extractors[vocab_name] = extractor_class
        
    @classmethod
    def get_extractor(cls, vocab_name: str, config: Dict[str, Any]) -> BaseExtractor:
        """Get an instance of the extractor for the given vocabulary."""
        if vocab_name not in cls._extractors:
            raise ValueError(f"No extractor registered for vocabulary: {vocab_name}")
        
        extractor_class = cls._extractors[vocab_name]
        return extractor_class(config, vocab_name)
        
    @classmethod
    def list_vocabularies(cls) -> List[str]:
        """List all registered vocabularies."""
        return list(cls._extractors.keys())