"""
ISAPC - IFU Spectrum Analysis Pipeline Cluster
Main program and command-line interface
"""
import os
import sys
import argparse
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

from muse import MUSECube
from analysis.p2p import run_p2p_analysis  
from analysis.voronoi import run_vnb_analysis
from analysis.radial import run_rdb_analysis
from utils.io import find_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='ISAPC - IFU Spectrum Analysis Pipeline Cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic parameters
    parser.add_argument('filename', help='Path to MUSE data cube file')
    parser.add_argument('--redshift', type=float, required=True, help='Galaxy redshift')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--template', default=None, help='Path to stellar template file')
    
    # Wavelength settings
    parser.add_argument('--wvl-range', type=float, nargs=2, default=None,
                      help='Wavelength range to analyze (Å), if not specified uses goodwavelengthrange from data')
    parser.add_argument('--no-good-wavelength', action='store_true', 
                      help='Do not use goodwavelengthrange from data')
    
    # Analysis mode
    parser.add_argument('--mode', choices=['P2P', 'VNB', 'RDB', 'ALL'], default='P2P',
                      help='Analysis mode: Pixel-to-pixel (P2P), Voronoi binning (VNB), Radial binning (RDB), or All (ALL)')
    
    # Run configuration
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs (-1 means using all CPUs)')
    parser.add_argument('--no-plots', action='store_true', help='Disable plotting')
    
    # Plotting options
    parser.add_argument('--equal-aspect', action='store_true', help='Keep aspect ratio equal for maps')
    
    # Fitting parameters
    parser.add_argument('--vel-init', type=float, default=0, help='Initial velocity guess (km/s)')
    parser.add_argument('--sigma-init', type=float, default=40, help='Initial dispersion guess (km/s)')
    parser.add_argument('--poly-degree', type=int, default=3, help='Degree of the additive polynomial for pPXF')
    parser.add_argument('--no-emission', action='store_true', help='Skip emission line fitting')
    parser.add_argument('--no-indices', action='store_true', help='Skip spectral indices calculation')
    
    # Voronoi binning parameters
    vnb_group = parser.add_argument_group('Voronoi binning options')
    vnb_group.add_argument('--target-snr', type=float, default=20, help='Target signal-to-noise ratio')
    
    # Radial binning parameters
    rdb_group = parser.add_argument_group('Radial binning options')
    rdb_group.add_argument('--n-rings', type=int, default=10, help='Number of radial rings')
    rdb_group.add_argument('--center-x', type=float, help='Center X coordinate')
    rdb_group.add_argument('--center-y', type=float, help='Center Y coordinate')
    rdb_group.add_argument('--pa', type=float, default=0.0, help='Position angle (degrees)')
    rdb_group.add_argument('--ellipticity', type=float, default=0.0, help='Ellipticity (0-1)')
    rdb_group.add_argument('--log-spacing', action='store_true', help='Use logarithmic spacing')
    
    return parser


def setup_logging(args):
    """Configure file logging"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    galaxy_name = Path(args.filename).stem
    
    log_file = log_dir / f"{galaxy_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)
    
    logger.info(f"ISAPC analysis started, target: {args.filename}")
    logger.info(f"Parameters: redshift={args.redshift}, wavelength range={args.wvl_range}, mode={args.mode}")


def main():
    """Main program entry point"""
    # Parse command-line arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set up file logging
    setup_logging(args)
    
    # Read data
    try:
        start_time = time.time()
        cube = MUSECube(
            filename=args.filename,
            redshift=args.redshift,
            wvl_air_angstrom_range=tuple(args.wvl_range) if args.wvl_range is not None else None,
            use_good_wavelength=not args.no_good_wavelength
        )
        logger.info(f"Data loaded in {time.time() - start_time:.1f} seconds")
        logger.info(f"Using wavelength range: {cube._wvl_air_angstrom_range}")
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    # Set template file
    if args.template is None:
        # Try to find template
        args.template = find_template()
        if args.template:
            logger.info(f"Automatically selected template: {args.template}")
        else:
            logger.error("No template file found, please specify using --template")
            return 1
    
    # Execute analysis
    p2p_results = None
    
    try:
        if args.mode in ['P2P', 'ALL']:
            p2p_results = run_p2p_analysis(args, cube)
        
        if args.mode in ['VNB', 'ALL']:
            run_vnb_analysis(args, cube, p2p_results)
        
        if args.mode in ['RDB', 'ALL']:
            run_rdb_analysis(args, cube, p2p_results)
            
        logger.info("ISAPC analysis completed")
        return 0
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())