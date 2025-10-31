"""
Storage utilities for hierarchical path data.
Supports JSON and LMDB storage formats.
"""

import json
import lmdb
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging


class PathStorage:
    """Handler for storing and retrieving hierarchical path data."""
    
    def __init__(self, output_dir: Union[str, Path], format: str = 'json'):
        """
        Initialize storage handler.
        
        Args:
            output_dir: Directory to store output files
            format: Storage format ('json' or 'lmdb')
        """
        self.output_dir = Path(output_dir)
        self.format = format.lower()
        self.logger = logging.getLogger(__name__)
        
        if self.format not in ['json', 'lmdb']:
            raise ValueError("Format must be 'json' or 'lmdb'")
            
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir = self.output_dir / 'statistics'
        self.stats_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        
    def save_paths(self, vocab_name: str, data: Dict[str, Any]) -> Path:
        """
        Save path data for a vocabulary.
        
        Args:
            vocab_name: Name of the vocabulary
            data: Dictionary with CUI -> path data mapping
            
        Returns:
            Path to the saved file
        """
        filename = f"paths_{vocab_name.lower()}"
        
        if self.format == 'json':
            filepath = self.output_dir / f"{filename}.json"
            self._save_json(filepath, data)
        else:  # lmdb
            filepath = self.output_dir / f"{filename}.lmdb"
            self._save_lmdb(filepath, data)
            
        self.logger.info(f"Saved {len(data)} CUI paths for {vocab_name} to {filepath}")
        return filepath
        
    def load_paths(self, vocab_name: str) -> Dict[str, Any]:
        """
        Load path data for a vocabulary.
        
        Args:
            vocab_name: Name of the vocabulary
            
        Returns:
            Dictionary with CUI -> path data mapping
        """
        filename = f"paths_{vocab_name.lower()}"
        
        if self.format == 'json':
            filepath = self.output_dir / f"{filename}.json"
            if filepath.exists():
                return self._load_json(filepath)
        else:  # lmdb
            filepath = self.output_dir / f"{filename}.lmdb"
            if filepath.exists():
                return self._load_lmdb(filepath)
                
        return {}
        
    def save_statistics(self, vocab_name: str, stats: Dict[str, Any]) -> Path:
        """
        Save statistics for a vocabulary.
        
        Args:
            vocab_name: Name of the vocabulary
            stats: Statistics dictionary
            
        Returns:
            Path to the saved statistics file
        """
        filepath = self.stats_dir / f"{vocab_name.lower()}_stats.json"
        
        # Add timestamp
        stats['extraction_date'] = datetime.now().isoformat()
        
        self._save_json(filepath, stats)
        self.logger.info(f"Saved statistics for {vocab_name} to {filepath}")
        return filepath
        
    def load_statistics(self, vocab_name: str) -> Dict[str, Any]:
        """
        Load statistics for a vocabulary.
        
        Args:
            vocab_name: Name of the vocabulary
            
        Returns:
            Statistics dictionary or empty dict if not found
        """
        filepath = self.stats_dir / f"{vocab_name.lower()}_stats.json"
        if filepath.exists():
            return self._load_json(filepath)
        return {}
        
    def get_processed_cuis(self, vocab_name: str) -> set:
        """
        Get set of CUIs already processed for a vocabulary.
        Useful for resume functionality.
        
        Args:
            vocab_name: Name of the vocabulary
            
        Returns:
            Set of processed CUI strings
        """
        data = self.load_paths(vocab_name)
        return set(data.keys())
        
    def append_cui_data(self, vocab_name: str, cui: str, cui_data: Dict[str, Any]):
        """
        Append data for a single CUI to existing storage.
        Useful for incremental processing.
        
        Args:
            vocab_name: Name of the vocabulary
            cui: CUI identifier
            cui_data: Data for this CUI
        """
        if self.format == 'json':
            # For JSON, we need to load all data, modify, and save
            all_data = self.load_paths(vocab_name)
            all_data[cui] = cui_data
            self.save_paths(vocab_name, all_data)
        else:  # lmdb
            # For LMDB, we can append efficiently
            filename = f"paths_{vocab_name.lower()}.lmdb"
            filepath = self.output_dir / filename
            
            env = lmdb.open(str(filepath), max_dbs=1, map_size=10**10)  # 10GB max
            with env.begin(write=True) as txn:
                txn.put(cui.encode(), pickle.dumps(cui_data))
            env.close()
            
    def _save_json(self, filepath: Path, data: Dict[str, Any]):
        """Save data as JSON with proper formatting."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _save_lmdb(self, filepath: Path, data: Dict[str, Any]):
        """Save data to LMDB database."""
        env = lmdb.open(str(filepath), max_dbs=1, map_size=10**10)  # 10GB max
        with env.begin(write=True) as txn:
            for key, value in data.items():
                txn.put(key.encode(), pickle.dumps(value))
        env.close()
        
    def _load_lmdb(self, filepath: Path) -> Dict[str, Any]:
        """Load data from LMDB database."""
        data = {}
        env = lmdb.open(str(filepath), readonly=True)
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                data[key.decode()] = pickle.loads(value)
        env.close()
        return data


class ProgressTracker:
    """Track and save processing progress."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize progress tracker.
        
        Args:
            output_dir: Directory to store progress files
        """
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def save_progress(self, vocab_name: str, progress_data: Dict[str, Any]):
        """
        Save current progress for a vocabulary.
        
        Args:
            vocab_name: Name of the vocabulary
            progress_data: Progress information
        """
        filepath = self.logs_dir / f"{vocab_name.lower()}_progress.json"
        
        progress_data.update({
            'last_updated': datetime.now().isoformat(),
            'vocabulary': vocab_name
        })
        
        with open(filepath, 'w') as f:
            json.dump(progress_data, f, indent=2)
            
    def load_progress(self, vocab_name: str) -> Dict[str, Any]:
        """
        Load progress for a vocabulary.
        
        Args:
            vocab_name: Name of the vocabulary
            
        Returns:
            Progress data or empty dict if not found
        """
        filepath = self.logs_dir / f"{vocab_name.lower()}_progress.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}
        
    def log_error(self, vocab_name: str, cui: str, code: str, error: str):
        """
        Log an error for debugging purposes.
        
        Args:
            vocab_name: Name of the vocabulary
            cui: CUI that had the error
            code: Code that caused the error
            error: Error message
        """
        error_file = self.logs_dir / f"{vocab_name.lower()}_errors.json"
        
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'cui': cui,
            'code': code,
            'error': error
        }
        
        # Append to error log
        errors = []
        if error_file.exists():
            with open(error_file, 'r') as f:
                errors = json.load(f)
                
        errors.append(error_entry)
        
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)