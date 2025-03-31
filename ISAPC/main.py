"""
ISAPC - IFU Spectrum Analysis Pipeline Cluster
Main program and command-line interface
Version 5.0.0 - Optimized for performance and consistency
"""
import os
import sys
import logging
import argparse
import traceback
from pathlib import Path
import numpy as np
from astropy.io import fits

from analysis.p2p import run_p2p_analysis
from analysis.voronoi import run_vnb_analysis
from analysis.radial import run_rdb_analysis
from utils.io import save_results_to_npz, load_results_from_npz, save_standardized_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('ISAPC')

def setup_parser():
    """Set up the command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="ISAPC - IFU Spectrum Analysis Pipeline Cluster"
    )
    
    # Required arguments
    parser.add_argument('filename', help='MUSE data cube filename')
    parser.add_argument('-z', '--redshift', type=float, required=True,
                        help='Galaxy redshift')
    parser.add_argument('-t', '--template', required=True,
                        help='Stellar template filename')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Output directory')
    
    # Analysis mode
    parser.add_argument('-m', '--mode', choices=['P2P', 'VNB', 'RDB', 'ALL'],
                        default='ALL', help='Analysis mode')
    
    # Fitting parameters
    parser.add_argument('--vel-init', type=float, default=0,
                        help='Initial velocity for fitting (km/s)')
    parser.add_argument('--sigma-init', type=float, default=150,
                        help='Initial velocity dispersion for fitting (km/s)')
    parser.add_argument('--poly-degree', type=int, default=3,
                        help='Degree of polynomial for continuum fitting')
    
    # Binning parameters
    parser.add_argument('--target-snr', type=float, default=30,
                        help='Target SNR for Voronoi binning')
    parser.add_argument('--min-snr', type=float, default=1,
                        help='Minimum SNR for valid data')
    parser.add_argument('--cvt', action='store_true',
                        help='Use CVT algorithm for Voronoi binning')
    parser.add_argument('--n-rings', type=int, default=10,
                        help='Number of rings for radial binning')
    parser.add_argument('--log-spacing', action='store_true',
                        help='Use logarithmic spacing for radial bins')
    parser.add_argument('--pa', type=float, default=None,
                        help='Position angle for radial bins (degrees)')
    parser.add_argument('--ellipticity', type=float, default=None,
                        help='Ellipticity for radial bins (0-1)')
    parser.add_argument('--center-x', type=float, default=None,
                        help='X coordinate of center (pixels)')
    parser.add_argument('--center-y', type=float, default=None,
                        help='Y coordinate of center (pixels)')
    
    # Plotting options
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    parser.add_argument('--equal-aspect', action='store_true',
                        help='Use equal aspect ratio for plots')
    
    # Performance options
    parser.add_argument('-j', '--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all cores)')
    
    # Analysis options
    parser.add_argument('--no-emission', action='store_true',
                        help='Skip emission line fitting')
    parser.add_argument('--no-indices', action='store_true',
                        help='Skip spectral indices calculation')
    
    # Data reuse options (modified)
    parser.add_argument('--auto-reuse', action='store_true', default=True,
                        help='Automatically load and reuse previous results if available')
    parser.add_argument('--no-auto-reuse', action='store_false', dest='auto_reuse',
                        help='Disable automatic reuse of previous results')
    
    # Find the argument parser setup section in main.py and add these lines:

    # # Spectral indices controls
    # parser.add_argument('--no-indices', action='store_true', default=False,
    #                 help='Skip calculation of spectral indices')

    # # Visualization controls
    # parser.add_argument('--no-plots', action='store_true', default=False,
    #                 help='Skip creation of visualization plots')
    # parser.add_argument('--no-lic', action='store_true', default=False,
    #                 help='Skip creation of LIC visualization')
    # parser.add_argument('--no-emission-plots', action='store_true', default=False,
    #                 help='Skip creation of emission line plots')
    # parser.add_argument('--no-kinematic-plots', action='store_true', default=False,
    #                 help='Skip creation of kinematic plots')
    # parser.add_argument('--no-bin-plots', action='store_true', default=False,
    #                 help='Skip creation of individual bin plots')
    # parser.add_argument('--max-bin-plots', type=int, default=5,
    #                 help='Maximum number of bin spectra to plot (default: 5)')
    
    return parser

def main():
    """Main entry point for ISAPC"""
    # Parse command-line arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # Extract galaxy name from filename
    galaxy_name = Path(args.filename).stem
    
    # Create output directories
    output_dir = Path(args.output_dir)
    galaxy_dir = output_dir / galaxy_name
    data_dir = galaxy_dir / 'Data'
    
    galaxy_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Starting ISAPC analysis for {galaxy_name} in mode {args.mode}")
    
    # Import the MUSECube class here to avoid circular imports
    try:
        from muse import MUSECube
    except ImportError:
        logger.error("Failed to import MUSECube class. Please ensure muse.py is in the Python path.")
        return 1
    
    # Load data cube
    try:
        logger.info(f"Loading data cube from {args.filename}")
        cube = MUSECube(
            filename=args.filename,
            redshift=args.redshift,
            use_good_wavelength=True
        )
        logger.info("Data cube loaded successfully")
    except Exception as e:
        logger.error(f"Error loading data cube: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    # Initialize shared results
    p2p_results = None
    
    # Execute analysis
    try:
        # Check for existing P2P results if we're running VNB or RDB
        if args.mode in ['VNB', 'RDB'] and args.auto_reuse:
            p2p_results_path = data_dir / f"{galaxy_name}_P2P_results.npz"
            std_results_path = data_dir / f"{galaxy_name}_P2P_standardized.npz"
            
            # Try both potential file paths
            if p2p_results_path.exists():
                logger.info(f"Found P2P results at {p2p_results_path}")
                try:
                    p2p_results = load_results_from_npz(p2p_results_path)
                    logger.info("Successfully loaded P2P results")
                except Exception as e:
                    logger.warning(f"Failed to load P2P results: {e}")
                    p2p_results = None
            elif std_results_path.exists():
                logger.info(f"Found standardized P2P results at {std_results_path}")
                try:
                    p2p_results = load_results_from_npz(std_results_path)
                    logger.info("Successfully loaded standardized P2P results")
                except Exception as e:
                    logger.warning(f"Failed to load standardized P2P results: {e}")
                    p2p_results = None
            
            if p2p_results is None:
                logger.warning("No P2P results found. VNB/RDB analyses may have limited functionality.")
        
        # P2P analysis
        if args.mode in ['P2P', 'ALL']:
            p2p_results = run_p2p_analysis(args, cube, Pmode=True)
            
            # Save results
            save_results_to_npz(
                output_file=data_dir / f"{galaxy_name}_P2P_results.npz",
                data_dict=p2p_results
            )
            logger.info(f"Saved P2P results to {data_dir / f'{galaxy_name}_P2P_results.npz'}")
        
        # VNB analysis
        if args.mode in ['VNB', 'ALL']:
            vnb_results = run_vnb_analysis(args, cube, p2p_results)
            
            # Save results
            save_results_to_npz(
                output_file=data_dir / f"{galaxy_name}_VNB_results.npz",
                data_dict=vnb_results
            )
            logger.info(f"Saved VNB results to {data_dir / f'{galaxy_name}_VNB_results.npz'}")
        
        # RDB analysis
        if args.mode in ['RDB', 'ALL']:
            rdb_results = run_rdb_analysis(args, cube, p2p_results)
            
            # Save results
            save_results_to_npz(
                output_file=data_dir / f"{galaxy_name}_RDB_results.npz",
                data_dict=rdb_results
            )
            logger.info(f"Saved RDB results to {data_dir / f'{galaxy_name}_RDB_results.npz'}")
        
        logger.info("ISAPC analysis completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == '__main__':
    sys.exit(main())