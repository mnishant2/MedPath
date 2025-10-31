"""Dataset processors package."""

from .base_processor import BaseProcessor
from .cdr_processor import CDRProcessor
from .ncbi_processor import NCBIProcessor
from .medmentions_processor import MedMentionsProcessor
from .cadec_processor import CADECProcessor
from .cometa_processor import CometaProcessor
from .adr_processor import ADRProcessor
from .mantra_gsc_processor import MantraGSCProcessor
from .mimic_iv_el_processor import MIMICIVELProcessor
from .shareclef_processor import ShareCLEFProcessor

# Registry mapping dataset names to processor classes
PROCESSOR_REGISTRY = {
    'cdr': CDRProcessor,
    'ncbi': NCBIProcessor,
    'medmentions': MedMentionsProcessor,
    'cadec': CADECProcessor,
    'cometa': CometaProcessor,
    'adr': ADRProcessor,
    'mantra-gsc': MantraGSCProcessor,
    'mantra_gsc': MantraGSCProcessor,  # Alternative naming
    'mimic-iv-el': MIMICIVELProcessor,
    'mimic_iv_el': MIMICIVELProcessor,  # Alternative naming
    'shareclef': ShareCLEFProcessor,
    'share-clef': ShareCLEFProcessor,  # Alternative naming
}

def get_processor_class(dataset_name: str):
    """Get processor class for a dataset."""
    dataset_name = dataset_name.lower().replace('-', '_')
    
    # Try exact match first
    if dataset_name in PROCESSOR_REGISTRY:
        return PROCESSOR_REGISTRY[dataset_name]
    
    # Try with alternative naming conventions
    alternative_names = {
        'ncbi_disease': 'ncbi',
        'ncbi-disease': 'ncbi',
        'med_mentions': 'medmentions',
        'med-mentions': 'medmentions',
        'mantra': 'mantra_gsc',
        'gsc': 'mantra_gsc',
        'mimic': 'mimic_iv_el',
        'mimic_iv': 'mimic_iv_el',
        'mimic-iv': 'mimic_iv_el',
        'clef': 'shareclef',
        'share_clef': 'shareclef',
    }
    
    if dataset_name in alternative_names:
        return PROCESSOR_REGISTRY[alternative_names[dataset_name]]
    
    return None

def list_available_processors():
    """List all available processor names."""
    return list(PROCESSOR_REGISTRY.keys())

__all__ = [
    'BaseProcessor',
    'CDRProcessor',
    'NCBIProcessor', 
    'MedMentionsProcessor',
    'CADECProcessor',
    'CometaProcessor',
    'ADRProcessor',
    'MantraGSCProcessor',
    'MIMICIVELProcessor',
    'ShareCLEFProcessor',
    'PROCESSOR_REGISTRY',
    'get_processor_class',
    'list_available_processors'
] 