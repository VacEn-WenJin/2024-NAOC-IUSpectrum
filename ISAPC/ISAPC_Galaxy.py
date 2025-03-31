#!/usr/bin/env python3
"""
ISAPC Galaxy Pipeline Processing Script
Processes a list of galaxies through the ISAPC code with appropriate parameters
"""

import os
import sys
import subprocess
import pandas as pd
import time
from datetime import datetime
import argparse
from tqdm import tqdm
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Galaxy data - from the observed galaxy list
GALAXIES = [
    {"name": "M60", "redshift": 0.0034, "type": "E2", "has_emission": False},
    {"name": "VCC1588", "redshift": 0.0042, "type": "Sd", "has_emission": True},
    {"name": "VCC1890", "redshift": 0.0040, "type": "dE", "has_emission": False},
    {"name": "VCC1368", "redshift": 0.0035, "type": "SBa", "has_emission": True},
    {"name": "VCC1902", "redshift": 0.0038, "type": "SBa", "has_emission": True},
    {"name": "VCC1523", "redshift": 0.0038, "type": "dE(N)", "has_emission": False},
    {"name": "VCC1949", "redshift": 0.0058, "type": "dS0(N)", "has_emission": True},
    {"name": "VCC990", "redshift": 0.0058, "type": "dS0(N)", "has_emission": False},
    {"name": "VCC1410", "redshift": 0.0054, "type": "Sd", "has_emission": True},
    {"name": "VCC1695", "redshift": 0.0058, "type": "dE", "has_emission": False},
    {"name": "VCC667", "redshift": 0.0048, "type": "Sd", "has_emission": True},
    {"name": "VCC1549", "redshift": 0.0046, "type": "dE(N)", "has_emission": False},
    {"name": "VCC308", "redshift": 0.0055, "type": "dE", "has_emission": False},
    {"name": "VCC1431", "redshift": 0.0050, "type": "dE", "has_emission": False},
    {"name": "VCC1811", "redshift": 0.0023, "type": "Sc", "has_emission": True},
    {"name": "VCC688", "redshift": 0.0038, "type": "Sc", "has_emission": True},
    {"name": "VCC1146", "redshift": 0.0023, "type": "E", "has_emission": False},
    {"name": "VCC1049", "redshift": 0.0021, "type": "dE(N)", "has_emission": False},
    {"name": "VCC1193", "redshift": 0.0025, "type": "Sd", "has_emission": True},
    {"name": "VCC1910", "redshift": 0.0007, "type": "dE(N)", "has_emission": False},
    {"name": "VCC1486", "redshift": 0.0004, "type": "Sc", "has_emission": True},
]

def setup_output_directory(base_dir):
    """Create output directory and logs subdirectory if they don't exist"""
    os.makedirs(base_dir, exist_ok=True)
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir

def build_command(galaxy, args):
    """Build the command to run ISAPC for a specific galaxy"""
    # Base command
    cmd = [
        sys.executable,
        "main.py",
        f"data/MUSE/{galaxy['name']}_stack.fits",
        "--redshift", f"{galaxy['redshift']}",
        "--n-job", f"{args.n_jobs}",
        "--mode", f"{args.mode}",
        "--target-snr", f"{args.target_snr}",
        "-t", f"{args.template}",
        "-o", f"{args.output_dir}/{galaxy['name']}"
    ]
    
    # Add emission line option if galaxy doesn't have emission lines
    if not galaxy['has_emission']:
        cmd.append("--no-emission")
        
    # Add additional arguments if provided
    if args.poly_degree:
        cmd.extend(["--poly-degree", f"{args.poly_degree}"])
    
    if args.sigma_init:
        cmd.extend(["--sigma-init", f"{args.sigma_init}"])
    
    if args.vel_init:
        cmd.extend(["--vel-init", f"{args.vel_init}"])
    
    if args.cvt:
        cmd.append("--cvt")
    
    if args.equal_aspect:
        cmd.append("--equal-aspect")
    
    return cmd

def run_galaxy_pipeline(galaxies, args):
    """Run ISAPC for each galaxy and save outputs"""
    logs_dir = setup_output_directory(os.path.join(args.output_dir, "logs"))
    
    # Summary information
    summary_path = os.path.join(args.output_dir, "processing_summary.csv")
    summary_data = []
    
    # Create progress bar
    pbar = tqdm(total=len(galaxies), desc="Processing galaxies")
    
    for i, galaxy in enumerate(galaxies):
        # Update progress bar
        pbar.set_description(f"Processing {galaxy['name']} ({i+1}/{len(galaxies)})")
        
        start_time = time.time()
        log_filename = os.path.join(logs_dir, f"{galaxy['name']}_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        try:
            # Build the command
            cmd = build_command(galaxy, args)
            
            # Log the command
            logger.info(f"Running: {' '.join(cmd)}")
            
            # Open log file and run command
            with open(log_filename, 'w') as log_file:
                # Write header to log file
                log_file.write(f"=== Processing {galaxy['name']} (Type: {galaxy['type']}, z={galaxy['redshift']}) ===\n")
                log_file.write(f"Command: {' '.join(cmd)}\n")
                log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                log_file.flush()  # Ensure header is written before process output
                
                # Run the process and capture output
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1  # Line buffered
                )
                
                # Stream output to log file and console
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()  # Ensure real-time logging
                    
                    # Extract and update progress information for tqdm bars
                    if "Fitting" in line and "%" in line:
                        try:
                            # Extract progress percentage
                            match = re.search(r'(\d+)%', line)
                            if match:
                                percent = int(match.group(1))
                                # Update tqdm bar description
                                pbar.set_postfix(task_progress=f"{percent}%")
                        except:
                            pass
                
                # Wait for process to complete
                process.wait()
                
                # Record completion in log
                end_time = time.time()
                duration = end_time - start_time
                log_file.write(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Duration: {duration:.2f} seconds\n")
                log_file.write(f"Exit code: {process.returncode}\n")
                
                # Add to summary data
                summary_data.append({
                    "galaxy": galaxy['name'],
                    "type": galaxy['type'],
                    "redshift": galaxy['redshift'],
                    "emission_lines": "Yes" if galaxy['has_emission'] else "No",
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": duration,
                    "status": "Success" if process.returncode == 0 else "Failed",
                    "exit_code": process.returncode,
                    "log_file": os.path.basename(log_filename)
                })
                
                # Update progress bar
                pbar.update(1)
                
        except Exception as e:
            logger.error(f"Error processing {galaxy['name']}: {str(e)}")
            
            # Record error in summary
            end_time = time.time()
            duration = end_time - start_time
            summary_data.append({
                "galaxy": galaxy['name'],
                "type": galaxy['type'],
                "redshift": galaxy['redshift'],
                "emission_lines": "Yes" if galaxy['has_emission'] else "No",
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": duration,
                "status": "Error",
                "exit_code": -1,
                "log_file": os.path.basename(log_filename),
                "error": str(e)
            })
            
            # Update progress bar
            pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    # Save summary to CSV
    pd.DataFrame(summary_data).to_csv(summary_path, index=False)
    logger.info(f"Processing summary saved to {summary_path}")
    
    # Print final summary
    success_count = sum(1 for item in summary_data if item['status'] == 'Success')
    failed_count = len(summary_data) - success_count
    logger.info(f"Processing complete. Success: {success_count}, Failed: {failed_count}")
    
    return summary_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISAPC Galaxy Processing Pipeline")
    
    # Required arguments
    parser.add_argument("--template", "-t", default="./templates/spectra_emiles_9.0.npz",
                      help="Template file for spectral fitting")
    parser.add_argument("--output-dir", "-o", default="./output",
                      help="Output directory for results")
    
    # Optional arguments
    parser.add_argument("--mode", default="ALL", choices=["P2P", "VNB", "RDB", "ALL"],
                      help="Analysis mode (default: ALL)")
    parser.add_argument("--target-snr", type=float, default=20,
                      help="Target SNR for binning (default: 20)")
    parser.add_argument("--n-jobs", type=int, default=12,
                      help="Number of parallel jobs (default: 12)")
    parser.add_argument("--poly-degree", type=int, default=None,
                      help="Polynomial degree for spectral fitting")
    parser.add_argument("--sigma-init", type=float, default=None,
                      help="Initial sigma guess for spectral fitting")
    parser.add_argument("--vel-init", type=float, default=None,
                      help="Initial velocity guess for spectral fitting")
    parser.add_argument("--cvt", action="store_true",
                      help="Use CVT for Voronoi binning")
    parser.add_argument("--equal-aspect", action="store_true",
                      help="Use equal aspect ratio for plots")
    parser.add_argument("--subset", type=int, nargs=2, default=None, 
                      help="Process only a subset of galaxies (start_index end_index)")
    
    args = parser.parse_args()
    
    # Process subset if requested
    if args.subset:
        start_idx, end_idx = args.subset
        galaxy_subset = GALAXIES[start_idx:end_idx+1]
        logger.info(f"Processing subset of galaxies: {start_idx} to {end_idx} ({len(galaxy_subset)} galaxies)")
        run_galaxy_pipeline(galaxy_subset, args)
    else:
        # Process all galaxies
        logger.info(f"Processing all {len(GALAXIES)} galaxies")
        run_galaxy_pipeline(GALAXIES, args)