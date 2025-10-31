#!/usr/bin/env python3
"""
Main script to process biomedical datasets for UMLS mapping.

This script provides a unified interface to process 9 biomedical datasets,
mapping their native concept IDs to UMLS CUIs and standardizing the output format.
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add the scripts directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from dataset_processors import get_processor_class, list_available_processors
from utils.statistics import DatasetStatistics


# Dataset configurations
DATASET_CONFIGS = {
    'cdr': {
        'name': 'CDR (Chemical Disease Relations)',
        'description': 'PubMed abstracts with chemical-disease relations annotated with MeSH IDs',
        'data_path': '../data/CDR_Data/CDR.Corpus.v010516',
        'licensed': False,
        'splits': ['train', 'dev', 'test'],
        'native_ontologies': ['MSH'],
        'entity_types': ['Chemical', 'Disease']
    },
    'ncbi': {
        'name': 'NCBI Disease',
        'description': 'PubMed abstracts with disease mentions annotated with MeSH and OMIM IDs',
        'data_path': '../data/NCBI',
        'licensed': False,
        'splits': ['train', 'dev', 'test'],
        'native_ontologies': ['MSH', 'OMIM'],
        'entity_types': ['Disease']
    },
    'medmentions': {
        'name': 'MedMentions',
        'description': 'PubMed abstracts with biomedical entity mentions mapped to UMLS CUIs',
        'data_path': '../data/medmentions',
        'licensed': False,
        'splits': ['train', 'dev', 'test'],
        'native_ontologies': ['UMLS'],
        'entity_types': ['Biomedical Entity']
    },
    'cadec': {
        'name': 'CADEC',
        'description': 'Patient forum posts with adverse drug events annotated with SNOMED CT and MedDRA',
        'data_path': '../data/CADEC',
        'licensed': False,
        'splits': [],
        'native_ontologies': ['SNOMEDCT_US', 'MDR'],
        'entity_types': ['Adverse Drug Event']
    },
    'cometa': {
        'name': 'COMETA',
        'description': 'Biomedical entity mentions linked to SNOMED CT concepts',
        'data_path': '../data/cometa',
        'licensed': False,
        'splits': [],
        'native_ontologies': ['SNOMEDCT_US'],
        'entity_types': ['Biomedical Entity']
    },
    'adr': {
        'name': 'ADR',
        'description': 'Drug-related adverse events annotated with MedDRA concept IDs',
        'data_path': '../data/ADR_cleaned',
        'licensed': False,
        'splits': [],
        'native_ontologies': ['MDR'],
        'entity_types': ['Adverse Drug Reaction']
    },
    'mantra-gsc': {
        'name': 'Mantra-GSC',
        'description': 'Multilingual biomedical entity mentions (Gene, Species, Chemical)',
        'data_path': '../data/Mantra-GSC',
        'licensed': False,
        'splits': [],
        'native_ontologies': ['NCBI_GENE', 'NCBI_TAXONOMY', 'CHEBI'],
        'entity_types': ['Gene', 'Species', 'Chemical']
    },
    'mimic-iv-el': {
        'name': 'MIMIC-IV EL',
        'description': 'Clinical notes with entity linking to SNOMED CT concepts',
        'data_path': '../data/MIMICIV_EL_cleaned_no_placeholders',
        'licensed': True,
        'splits': [],
        'native_ontologies': ['SNOMEDCT_US'],
        'entity_types': ['Clinical Entity']
    },
    'shareclef': {
        'name': 'ShareCLEF',
        'description': 'Clinical cases annotated with UMLS CUIs',
        'data_path': '../data/shareclef_cleaned',
        'licensed': True,
        'splits': ['train', 'test'],
        'native_ontologies': ['UMLS'],
        'entity_types': ['Clinical Entity']
    }
}


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Set up logging configuration."""
    logs_dir = (Path(__file__).parent / 'logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / 'dataset_processing.log'
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def process_dataset(dataset_name: str, config: Dict[str, Any], logger: logging.Logger, use_local_umls: bool = False, umls_path: Optional[str] = None, limit: Optional[int] = None, output_root: Optional[Path] = None, subdir_suffix: str = "", data_root: Optional[Path] = None) -> bool:
    """Process a single dataset."""
    logger.info(f"Processing {config['name']} dataset...")
    if limit:
        logger.info(f"Running in TEST MODE: limiting to {limit} documents")

    # Get processor class
    processor_class = get_processor_class(dataset_name)
    if not processor_class:
        logger.error(f"No processor found for dataset: {dataset_name}")
        return False

    # Resolve data path
    data_path_cfg = Path(config['data_path'])
    if data_root and not data_path_cfg.is_absolute():
        # Map relative path after 'data' onto provided data_root
        parts = list(data_path_cfg.parts)
        try:
            idx = parts.index('data')
            subparts = parts[idx+1:]
            data_path = Path(data_root).joinpath(*subparts)
        except ValueError:
            # Fallback: join entire relative path under data_root
            data_path = Path(data_root) / data_path_cfg
        data_path = data_path.resolve()
    else:
        # Resolve relative to scripts directory
        scripts_dir = Path(__file__).parent
        data_path = (data_path_cfg if data_path_cfg.is_absolute() else (scripts_dir / data_path_cfg).resolve())

    # Check if this is a licensed dataset
    if config.get('licensed', False):
        if not data_path.exists():
            logger.warning(f"Licensed dataset {dataset_name} not found at {data_path}")
            logger.info(f"Please obtain licensed access to {config['name']} and place files in {data_path}")
            logger.info(f"See README.md for setup instructions")
            return False

    try:
        # Initialize processor
        processor = processor_class(
            dataset_name=dataset_name,
            data_dir=data_path,
            output_dir=(output_root or Path("../data_processed")),
            use_local_umls=use_local_umls,
            umls_path=umls_path,
            limit=limit,  # Pass limit to processor
            subdir_suffix=subdir_suffix
        )

        # Process the dataset
        success = processor.process()

        if success:
            logger.info(f"Successfully processed {config['name']} dataset")
        else:
            logger.error(f"Failed to process {config['name']} dataset")

        return success

    except Exception as e:
        logger.error(f"Error processing {config['name']} dataset: {e}")
        logger.exception("Full traceback:")
        return False


def generate_cross_dataset_statistics(logger: logging.Logger, output_root: Path, subdir_suffix: str, skip_comparison: bool = False):
    """Generate cross-dataset statistics and comparisons."""
    logger.info("Generating cross-dataset statistics...")

    try:
        # Ensure stats are written under output_root respecting suffix
        stats_output_dir = output_root / f"stats{subdir_suffix}"
        stats_output_dir.mkdir(parents=True, exist_ok=True)
        stats_generator = DatasetStatistics(str(stats_output_dir))

        # Find all dataset files in the documents directory (respect output_root and suffix)
        documents_dir = output_root / f"documents{subdir_suffix}"

        if not documents_dir.exists():
            logger.warning("Documents directory not found")
            return

        # Find all unique dataset names from .jsonl files
        dataset_files = list(documents_dir.glob("*.jsonl"))
        dataset_names = set()

        for file_path in dataset_files:
            # Extract base dataset name (remove split suffixes like _train, _dev, _test)
            name = file_path.stem
            base_name = name.split('_')[0]  # Get the first part before any underscore
            dataset_names.add(base_name)

        if not dataset_names:
            logger.warning("No processed datasets found for statistics generation")
            return

        logger.info(f"Found datasets: {list(dataset_names)}")

        # Generate individual dataset statistics
        for dataset_name in dataset_names:
            logger.info(f"Generating statistics for {dataset_name}")

            try:
                stats_generator.generate_dataset_statistics(dataset_name, documents_dir)
            except Exception as e:
                logger.error(f"Error generating statistics for {dataset_name}: {e}")

        # Generate comparison statistics (optional)
        if not skip_comparison:
            logger.info("Generating cross-dataset comparison...")
            try:
                # Construct pseudo dataset dirs so names resolve correctly
                dataset_dirs = [documents_dir / dn for dn in sorted(dataset_names)]
                stats_generator.generate_comparison_statistics(dataset_dirs)
            except Exception as e:
                logger.warning(f"Failed to generate comparison statistics: {e}")
        else:
            logger.info("Skipping cross-dataset comparison (suppressed)")

        # Generate cross-dataset comprehensive mapping summary
        logger.info("Generating cross-dataset comprehensive mapping summary...")
        try:
            mapping_summary = stats_generator.generate_cross_dataset_mapping_summary_fixed(documents_dir, dataset_names)
            logger.info("Cross-dataset comprehensive mapping summary generated successfully")
            logger.info(f"Combined mapping files: {list(mapping_summary.keys())}")
        except Exception as e:
            logger.warning(f"Failed to generate cross-dataset mapping summary: {e}")
            logger.exception("Full traceback:")

        logger.info("Cross-dataset statistics generated successfully")

    except Exception as e:
        logger.error(f"Error generating cross-dataset statistics: {e}")
        logger.exception("Full traceback:")


# README generation removed for curated MedPath


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Process biomedical datasets for UMLS mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
{chr(10).join([f"  {name}: {config['name']}" for name, config in DATASET_CONFIGS.items()])}

Examples:
  %(prog)s --all                         # Process all datasets
  %(prog)s --datasets cdr ncbi           # Process specific datasets
  %(prog)s --stats_only                  # Generate statistics only
  %(prog)s --list                        # List available datasets
  %(prog)s --all --limit 10              # Quick test: 10 docs per dataset
  %(prog)s --datasets ncbi --use_local_umls --umls_path ../umls/2025AA \
           --data_dir ./data_sample --output ./data_processed --subdir_suffix _sample
        """
    )

    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_CONFIGS.keys()),
                       help='Specific datasets to process')
    parser.add_argument('--all', action='store_true',
                       help='Process all available datasets')
    parser.add_argument('--stats_only', action='store_true',
                       help='Generate statistics only (skip dataset processing)')
    parser.add_argument('--list', action='store_true',
                       help='List available datasets and exit')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set the logging level')
    parser.add_argument('--skip_stats', action='store_true',
                       help='Skip statistics generation')
    parser.add_argument('--skip_comparison', action='store_true',
                       help='Skip cross-dataset comparison (plots and summary)')
    parser.add_argument('--use_local_umls', action='store_true',
                       help='Use local UMLS files instead of API calls (much faster)')
    parser.add_argument('--umls_path', type=str, default='../umls/2025AA',
                       help='Path to local UMLS installation (default: ../umls/2025AA)')
    parser.add_argument('--limit', type=int, metavar='N',
                       help='Limit processing to N documents per dataset (for quick testing)')
    parser.add_argument('--output', type=str,
                       help='Override output root directory (default: ../data_processed)')
    parser.add_argument('--subdir_suffix', type=str, default='',
                       help='Suffix to append to subdirectories (e.g. _new -> documents_new)')
    parser.add_argument('--data_dir', type=str,
                       help='Override data root directory; relative paths under DATASET_CONFIGS will be mapped under this root')

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_level)

    # List datasets if requested
    if args.list:
        print("Available datasets:")
        for name, config in DATASET_CONFIGS.items():
            licensed = " (Licensed)" if config.get('licensed', False) else ""
            print(f"  {name}: {config['name']}{licensed}")
        return

    # Determine which datasets to process
    if args.all:
        datasets_to_process = list(DATASET_CONFIGS.keys())
    elif args.datasets:
        datasets_to_process = args.datasets
    elif args.stats_only:
        datasets_to_process = []
    else:
        parser.print_help()
        return

    logger.info("Starting UMLS dataset processing pipeline...")
    if args.limit:
        logger.info(f"TEST MODE: Processing limited to {args.limit} documents per dataset")

    # Create output directories
    # Default to MedPath/data_processed regardless of current working directory
    scripts_dir = Path(__file__).parent.resolve()
    medpath_root = scripts_dir.parent
    output_dir = Path(args.output).resolve() if args.output else (medpath_root / "data_processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    for subdir in ['documents', 'mappings', 'stats']:
        (output_dir / f"{subdir}{args.subdir_suffix}").mkdir(parents=True, exist_ok=True)

    # Process datasets
    if not args.stats_only:
        successful_datasets = []
        failed_datasets = []

        for dataset_name in datasets_to_process:
            config = DATASET_CONFIGS[dataset_name]

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {config['name']} ({dataset_name})")
            if args.use_local_umls:
                logger.info(f"Using local UMLS files from: {args.umls_path}")
            else:
                logger.info("Using UMLS API")
            if args.limit:
                logger.info(f"TEST MODE: Limited to {args.limit} documents")
            logger.info(f"{'='*60}")

            success = process_dataset(
                dataset_name,
                config,
                logger,
                args.use_local_umls,
                args.umls_path,
                args.limit,
                output_root=output_dir,
                subdir_suffix=args.subdir_suffix,
                data_root=Path(args.data_dir) if args.data_dir else None
            )

            if success:
                successful_datasets.append(dataset_name)
            else:
                failed_datasets.append(dataset_name)

        # Report results
        logger.info(f"\n{'='*60}")
        logger.info("PROCESSING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Successfully processed: {len(successful_datasets)} datasets")
        for dataset in successful_datasets:
            logger.info(f"  ✓ {dataset}")

        if failed_datasets:
            logger.info(f"Failed to process: {len(failed_datasets)} datasets")
            for dataset in failed_datasets:
                logger.info(f"  ✗ {dataset}")

    # Generate statistics
    if not args.skip_stats:
        logger.info(f"\n{'='*60}")
        logger.info("GENERATING STATISTICS")
        logger.info(f"{'='*60}")
        generate_cross_dataset_statistics(logger, output_dir, args.subdir_suffix, skip_comparison=args.skip_comparison)

    # README generation intentionally disabled in curated MedPath

    logger.info(f"\nPipeline completed successfully!")
    logger.info(f"Results available in: {output_dir}/")


if __name__ == "__main__":
    main() 