#!/usr/bin/env python3
"""mimiciv_el dataset processor (placeholder)."""
from typing import List, Dict, Any
try:
from .base_processor import BaseProcessor
except ImportError:
    BaseProcessor = object

class Mimiciv_elProcessor(BaseProcessor if BaseProcessor != object else object):
    def _get_native_ontologies(self) -> List[str]:
        return ['TODO']
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        raise NotImplementedError("mimiciv_el processor not yet implemented")
    def _extract_mentions(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        raise NotImplementedError("mimiciv_el processor not yet implemented")
