"""
Radial binning analysis module for ISAPC
Version 5.0.0
"""
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.colors as mcolors
import spectral_indices
import galaxy_params
import visualization
import traceback
from utils.io import save_results_to_npz, save_standardized_results
from binning import (
    calculate_radial_bins, apply_velocity_shift, BinnedSpectra, RadialBinnedData,
    plot_binned_map, plot_radial_profile, calculate_wavelength_intersection,
    combine_spectra_efficiently, calculate_snr
)
from utils.calc import spectres

logger = logging.getLogger(__name__)

# Speed of light in km/s
C_KMS = 299792.458

def run_rdb_analysis(args, cube, p2p_results=None):
    """
    Run Radial binning analysis on MUSE data cube

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cube : MUSECube
        MUSE data cube object
    p2p_results : dict, optional
        Results from P2P analysis, used to get velocity field for correction
        
    Returns
    -------
    dict
        Analysis results with binned data and physical parameters
    """
    logger.info("Starting Radial binning analysis...")
    start_time = time.time()
    
    # Disable warnings for spectral indices
    spectral_indices.set_warnings(False)
    
    # Extract galaxy name from filename
    galaxy_name = Path(args.filename).stem
    
    # Create standardized output directories
    output_dir = Path(args.output_dir)
    galaxy_dir = output_dir / galaxy_name
    data_dir = galaxy_dir / 'Data'
    plots_dir = galaxy_dir / 'Plots' / 'RDB'
    
    galaxy_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Try to load P2P results if not provided but auto-reuse is enabled
    if p2p_results is None and hasattr(args, 'auto_reuse') and args.auto_reuse:
        from p2p_adapter import load_p2p_results_for_galaxy
        p2p_results = load_p2p_results_for_galaxy(galaxy_name, args.output_dir)
        
        if p2p_results is not None:
            logger.info("Successfully loaded P2P results for RDB analysis")
    
    # Adapt number of rings based on data size
    if hasattr(args, 'n_rings'):
        n_pixels = cube._n_y * cube._n_x
        if n_pixels < args.n_rings * 10:
            adjusted_n_rings = max(3, n_pixels // 10)
            logger.warning(f"Reducing n_rings from {args.n_rings} to {adjusted_n_rings} for small dataset")
            n_rings = adjusted_n_rings
        else:
            n_rings = args.n_rings
    else:
        n_rings = 10  # Default
    
    # Get center coordinates
    center_x = args.center_x if hasattr(args, 'center_x') and args.center_x is not None else cube._n_x // 2
    center_y = args.center_y if hasattr(args, 'center_y') and args.center_y is not None else cube._n_y // 2
    
    # Get position angle and ellipticity
    pa = args.pa if hasattr(args, 'pa') and args.pa is not None else 0
    ellipticity = args.ellipticity if hasattr(args, 'ellipticity') and args.ellipticity is not None else 0
    
    # Get log spacing flag
    log_spacing = args.log_spacing if hasattr(args, 'log_spacing') else False
    
    # Try to get PA and center from P2P results if available and not specified
    if p2p_results is not None:
        try:
            # Check for PA in global_kinematics
            if 'global_kinematics' in p2p_results and 'pa' in p2p_results['global_kinematics']:
                if pa == 0 or pa is None:  # Only use if not specified
                    pa = p2p_results['global_kinematics']['pa']
                    logger.info(f"Using PA={pa:.1f} from P2P results")
            
            # Check for center in global_kinematics
            if 'global_kinematics' in p2p_results and 'center' in p2p_results['global_kinematics']:
                center = p2p_results['global_kinematics']['center']
                if (isinstance(center, tuple) or isinstance(center, list)) and len(center) == 2:
                    if (args.center_x is None and args.center_y is None):
                        center_x, center_y = center
                        logger.info(f"Using center=({center_x:.1f}, {center_y:.1f}) from P2P results")
        except Exception as e:
            logger.warning(f"Error extracting parameters from P2P results: {e}")
    
    # Step 1: Extract coordinates for binning
    # ---------------------------------------------
    try:
        # Check for valid wavelength range
        wave_mask = None
        good_lambda = None

        # First check if cube has _goodwavelength attribute
        if hasattr(cube, '_goodwavelength') and cube._goodwavelength is not None:
            good_lambda = cube._goodwavelength
            wave_mask = (cube._lambda_gal >= good_lambda[0]) & (cube._lambda_gal <= good_lambda[1])
            logger.info(f"Using goodwavelength range from cube object: {good_lambda[0]:.1f} - {good_lambda[1]:.1f} Å")
        # If not, try to get from FITS header
        elif hasattr(cube, '_fits_hdu_header'):
            if 'WAVGOOD0' in cube._fits_hdu_header and 'WAVGOOD1' in cube._fits_hdu_header:
                good_lambda = (
                    float(cube._fits_hdu_header['WAVGOOD0']) / (1 + cube._redshift),
                    float(cube._fits_hdu_header['WAVGOOD1']) / (1 + cube._redshift)
                )
                wave_mask = (cube._lambda_gal >= good_lambda[0]) & (cube._lambda_gal <= good_lambda[1])
                logger.info(f"Found goodwavelength range in header with redshift correction: {good_lambda[0]:.1f} - {good_lambda[1]:.1f} Å")
            else:
                # If not found in header either
                wave_mask = np.ones_like(cube._lambda_gal, dtype=bool)
                logger.info("No goodwavelength range found in header, using full wavelength range")
        else:
            # If no goodwavelength is found, use full range
            wave_mask = np.ones_like(cube._lambda_gal, dtype=bool)
            logger.info("No goodwavelength range found, using full wavelength range")

        # Extract wavelength range
        wavelength = cube._lambda_gal[wave_mask]
        
        # Get coordinates for each pixel
        x = np.zeros(cube._n_y * cube._n_x)
        y = np.zeros(cube._n_y * cube._n_x)
        
        # Create grid of pixel indices
        y_indices, x_indices = np.indices((cube._n_y, cube._n_x))
        
        # Report center coordinates
        logger.info(f"Using center: ({center_x}, {center_y})")
        
        # Convert indices to physical coordinates (relative to center)
        x = (x_indices.ravel() - center_x) * cube._pxl_size_x
        y = (y_indices.ravel() - center_y) * cube._pxl_size_y
        
        logger.info(f"Extracted coordinates for {len(x)} spaxels")
    except Exception as e:
        logger.error(f"Error extracting coordinates: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # Step 2: Run Radial binning
    # --------------------------
    try:
        logger.info(f"Running Radial binning with {n_rings} rings")
        logger.info(f"Parameters: center=({center_x}, {center_y}), PA={pa}, ellipticity={ellipticity}")
        logger.info(f"Using {'logarithmic' if log_spacing else 'linear'} spacing")
        
        # Run radial binning
        bin_num, bin_edges, bin_radii = calculate_radial_bins(
            x, y, center_x=0, center_y=0,  # Already centered coordinates
            pa=pa, ellipticity=ellipticity,
            n_rings=n_rings, log_spacing=log_spacing
        )
        
        # Get velocity field from P2P results if available
        velocity_field = None
        if p2p_results is not None:
            try:
                # Try standardized format first
                if 'stellar_kinematics' in p2p_results and 'velocity_field' in p2p_results['stellar_kinematics']:
                    velocity_field = p2p_results['stellar_kinematics']['velocity_field']
                # Then try direct format
                elif 'velocity_field' in p2p_results:
                    velocity_field = p2p_results['velocity_field']
                
                if velocity_field is not None:
                    logger.info("Using velocity field from P2P results for velocity correction")
            except Exception as e:
                logger.warning(f"Error extracting velocity field from P2P results: {e}")
        
        # Get bin indices for each bin
        bin_indices = []
        for i in range(len(bin_radii)):
            indices = np.where(bin_num == i)[0]
            bin_indices.append(indices)
        
        logger.info(f"Created {len(bin_indices)} radial bins")
        
        # Calculate binned spectra with velocity correction if available
        if velocity_field is not None:
            # Calculate intersection of wavelength ranges accounting for velocity shifts
            wave_mask, min_wave, max_wave = calculate_wavelength_intersection(
                cube._lambda_gal, velocity_field, cube._n_x
            )
            logger.info(f"Applying velocity correction with range {min_wave:.1f} - {max_wave:.1f} Å")
        else:
            wave_mask = np.ones_like(cube._lambda_gal, dtype=bool)
            logger.info("No velocity correction applied")
        
        # Extract wavelength range
        wavelength = cube._lambda_gal[wave_mask]
        
        # Combine spectra in each bin
        binned_spectra = combine_spectra_efficiently(
            cube._spectra[wave_mask], wavelength, bin_indices, velocity_field, cube._n_x
        )
        
        logger.info(f"Combined spectra into {len(bin_indices)} bins")
        
        # Create metadata dictionary
        metadata = {
            'nx': cube._n_x,
            'ny': cube._n_y,
            'center_x': center_x,
            'center_y': center_y,
            'pa': pa,
            'ellipticity': ellipticity,
            'n_rings': len(bin_radii),
            'log_spacing': log_spacing,
            'bin_edges': bin_edges,
            'time': time.time(),
            'galaxy_name': galaxy_name,
            'analysis_type': 'RDB',
            'pixelsize_x': cube._pxl_size_x,
            'pixelsize_y': cube._pxl_size_y
        }
        
        # Create RadialBinnedData object
        binned_data = RadialBinnedData(
            bin_num=bin_num,
            bin_indices=bin_indices,
            spectra=binned_spectra,
            wavelength=wavelength,
            metadata=metadata,
            bin_radii=bin_radii
        )
        
        # Run analysis on binned spectra
        rdb_results = run_analysis_on_binned_data(args, binned_data, cube, p2p_results)
        
        # Create visualization plots
        if not args.no_plots:
            create_rdb_plots(args, binned_data, rdb_results, galaxy_name, plots_dir)
        
        # Prepare output dictionary with all results
        result_dict = {
            'analysis_type': 'RDB',
            'bin_num': bin_num,
            'bin_indices': bin_indices,
            'bin_radii': bin_radii,
            'bin_edges': bin_edges,
            'parameters': {
                'center_x': center_x,
                'center_y': center_y,
                'pa': pa,
                'ellipticity': ellipticity,
                'n_rings': len(bin_radii),
                'log_spacing': log_spacing
            }
        }
        
        # Add analysis results
        result_dict.update(rdb_results)
        
        # Save results
        save_standardized_results(galaxy_name, 'RDB', result_dict, output_dir)
        
        logger.info(f"RDB analysis completed in {time.time() - start_time:.1f} seconds")
        
        return result_dict
    
    except Exception as e:
        logger.error(f"Error in radial binning: {e}")
        logger.error(traceback.format_exc())
        # Return empty results dictionary
        return {
            'analysis_type': 'RDB',
            'status': 'error',
            'error': str(e)
        }

def run_analysis_on_binned_data(args, binned_data, cube, p2p_results=None):
    """
    Run additional analysis on binned data (stellar population, emission lines, etc.)
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    binned_data : RadialBinnedData
        Binned data object
    cube : MUSECube
        Original MUSE cube
    p2p_results : dict, optional
        P2P analysis results
        
    Returns
    -------
    dict
        Analysis results
    """
    try:
        logger.info("Running analysis on binned data...")
        
        # Import required modules
        from p2p_adapter import create_p2p_processor, BinnedDataAdapter, extract_bin_results
        from analysis.p2p import run_p2p_analysis
        
        # Create P2P processor
        p2p_processor = create_p2p_processor(run_p2p_analysis)
        
        # Run P2P analysis on binned data
        bin_p2p_results = p2p_processor(args, binned_data, p2p_results)
        
        # Extract bin results
        bin_adapter = BinnedDataAdapter(binned_data)
        results = extract_bin_results(bin_p2p_results, bin_adapter, result_type='rdb')
        
        # Format results for consistency with RDB output
        formatted_results = {
            'stellar_kinematics': {
                'velocity': bin_p2p_results.get('velocity_field', None),
                'dispersion': bin_p2p_results.get('dispersion_field', None)
            },
            'distance': {
                'bin_distances': binned_data.bin_radii,
                'pixelsize_x': cube._pxl_size_x,
                'pixelsize_y': cube._pxl_size_y
            }
        }
        
        # Add emission line results if available
        if 'emission' in bin_p2p_results:
            formatted_results['emission'] = {}
            
            # Process each emission line parameter
            for key, value in bin_p2p_results['emission'].items():
                if isinstance(value, np.ndarray) and value.shape == (cube._n_y, cube._n_x):
                    # Extract values for each radial bin
                    bin_values = []
                    for i, indices in enumerate(binned_data.bin_indices):
                        if len(indices) > 0:
                            # Get pixel coordinates for this bin
                            y_indices = [idx // cube._n_x for idx in indices]
                            x_indices = [idx % cube._n_x for idx in indices]
                            
                            # Extract values and compute median
                            bin_pixels = [value[y, x] for y, x in zip(y_indices, x_indices) 
                                         if 0 <= y < value.shape[0] and 0 <= x < value.shape[1]]
                            
                            if bin_pixels:
                                bin_values.append(np.nanmedian(bin_pixels))
                            else:
                                bin_values.append(np.nan)
                        else:
                            bin_values.append(np.nan)
                    
                    formatted_results['emission'][key] = np.array(bin_values)
                else:
                    formatted_results['emission'][key] = value
        
        # Add spectral indices if available
        if 'indices' in bin_p2p_results:
            formatted_results['indices'] = {}
            
            # Process each index
            for index_name, index_map in bin_p2p_results['indices'].items():
                if isinstance(index_map, np.ndarray) and index_map.shape == (cube._n_y, cube._n_x):
                    # Extract values for each radial bin
                    bin_values = []
                    for i, indices in enumerate(binned_data.bin_indices):
                        if len(indices) > 0:
                            # Get pixel coordinates for this bin
                            y_indices = [idx // cube._n_x for idx in indices]
                            x_indices = [idx % cube._n_x for idx in indices]
                            
                            # Extract values and compute median
                            bin_pixels = [index_map[y, x] for y, x in zip(y_indices, x_indices) 
                                         if 0 <= y < index_map.shape[0] and 0 <= x < index_map.shape[1]]
                            
                            if bin_pixels:
                                bin_values.append(np.nanmedian(bin_pixels))
                            else:
                                bin_values.append(np.nan)
                        else:
                            bin_values.append(np.nan)
                    
                    formatted_results['indices'][index_name] = np.array(bin_values)
                else:
                    formatted_results['indices'][index_name] = index_map
        
        # Add stellar population parameters if available
        if 'stellar_population' in bin_p2p_results:
            formatted_results['stellar_population'] = {}
            
            # Process each parameter
            for param_name, param_map in bin_p2p_results['stellar_population'].items():
                if isinstance(param_map, np.ndarray) and param_map.shape == (cube._n_y, cube._n_x):
                    # Extract values for each radial bin
                    bin_values = []
                    for i, indices in enumerate(binned_data.bin_indices):
                        if len(indices) > 0:
                            # Get pixel coordinates for this bin
                            y_indices = [idx // cube._n_x for idx in indices]
                            x_indices = [idx % cube._n_x for idx in indices]
                            
                            # Extract values and compute median
                            bin_pixels = [param_map[y, x] for y, x in zip(y_indices, x_indices) 
                                         if 0 <= y < param_map.shape[0] and 0 <= x < param_map.shape[1]]
                            
                            if bin_pixels:
                                bin_values.append(np.nanmedian(bin_pixels))
                            else:
                                bin_values.append(np.nan)
                        else:
                            bin_values.append(np.nan)
                    
                    formatted_results['stellar_population'][param_name] = np.array(bin_values)
                else:
                    formatted_results['stellar_population'][param_name] = param_map
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error in analysis on binned data: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

def create_rdb_plots(args, binned_data, rdb_results, galaxy_name, plots_dir):
    """
    Create visualization plots for RDB analysis
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    binned_data : RadialBinnedData
        Binned data object
    rdb_results : dict
        RDB analysis results
    galaxy_name : str
        Galaxy name
    plots_dir : Path
        Directory to save plots
    """
    try:
        # Create basic binning plots
        binned_data.create_visualization_plots(plots_dir, galaxy_name)
        
        # Create radial profile plots
        n_y, n_x = binned_data.bin_num.reshape(-1, 1).shape if hasattr(binned_data.bin_num, 'shape') else (1, len(binned_data.bin_num))
        
        # Create kinematics radial profile
        if 'stellar_kinematics' in rdb_results:
            velocity = rdb_results['stellar_kinematics'].get('velocity', None)
            dispersion = rdb_results['stellar_kinematics'].get('dispersion', None)
            
            if velocity is not None and dispersion is not None:
                try:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Plot velocity profile
                    axes[0].plot(binned_data.bin_radii, velocity, 'o-', label='Velocity')
                    axes[0].set_xlabel('Radius (arcsec)')
                    axes[0].set_ylabel('Velocity (km/s)')
                    axes[0].set_title('Stellar Velocity Profile')
                    axes[0].grid(True, alpha=0.3)
                    
                    # Plot dispersion profile
                    axes[1].plot(binned_data.bin_radii, dispersion, 'o-', label='Dispersion')
                    axes[1].set_xlabel('Radius (arcsec)')
                    axes[1].set_ylabel('Dispersion (km/s)')
                    axes[1].set_title('Stellar Dispersion Profile')
                    axes[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"{galaxy_name}_rdb_kinematics_profile.png", dpi=150)
                    plt.close(fig)
                    
                    # Create 2D bin map with kinematics
                    try:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Create 2D bin map
                        bin_map_2d = binned_data.bin_num.reshape(n_y, n_x)
                        
                        # Plot velocity map
                        velocity_map = np.zeros_like(bin_map_2d, dtype=float)
                        for i, vel in enumerate(velocity):
                            velocity_map[bin_map_2d == i] = vel
                        
                        im0 = axes[0].imshow(velocity_map, origin='lower', cmap='coolwarm')
                        plt.colorbar(im0, ax=axes[0], label='Velocity (km/s)')
                        axes[0].set_title('Stellar Velocity Map')
                        
                        # Plot dispersion map
                        dispersion_map = np.zeros_like(bin_map_2d, dtype=float)
                        for i, disp in enumerate(dispersion):
                            dispersion_map[bin_map_2d == i] = disp
                        
                        im1 = axes[1].imshow(dispersion_map, origin='lower', cmap='viridis')
                        plt.colorbar(im1, ax=axes[1], label='Dispersion (km/s)')
                        axes[1].set_title('Stellar Dispersion Map')
                        
                        plt.tight_layout()
                        plt.savefig(plots_dir / f"{galaxy_name}_rdb_kinematics_map.png", dpi=150)
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Error creating 2D kinematics maps: {str(e)}")
                except Exception as e:
                    logger.warning(f"Error creating kinematics profile plots: {str(e)}")
        
        # Create stellar population radial profiles
        if 'stellar_population' in rdb_results:
            try:
                params = rdb_results['stellar_population']
                param_names = list(params.keys())
                
                if param_names:
                    fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 4))
                    if len(param_names) == 1:
                        axes = [axes]
                    
                    for i, param_name in enumerate(param_names):
                        param_values = params[param_name]
                        
                        # Adjust display for age
                        if param_name == 'age':
                            param_values = param_values * 1e-9  # Convert to Gyr
                            label = 'Age (Gyr)'
                        elif param_name == 'log_age':
                            label = 'Log Age (yr)'
                        elif param_name == 'metallicity':
                            label = 'Metallicity [Z/H]'
                        else:
                            label = param_name
                        
                        axes[i].plot(binned_data.bin_radii, param_values, 'o-')
                        axes[i].set_xlabel('Radius (arcsec)')
                        axes[i].set_ylabel(label)
                        axes[i].set_title(f'Stellar {label} Profile')
                        axes[i].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"{galaxy_name}_rdb_stellar_pop_profile.png", dpi=150)
                    plt.close(fig)
                    
                    # Create 2D maps of stellar population parameters
                    try:
                        fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 4))
                        if len(param_names) == 1:
                            axes = [axes]
                        
                        # Create 2D bin map
                        bin_map_2d = binned_data.bin_num.reshape(n_y, n_x)
                        
                        for i, param_name in enumerate(param_names):
                            param_values = params[param_name]
                            
                            # Adjust display for age
                            if param_name == 'age':
                                param_values = param_values * 1e-9  # Convert to Gyr
                                label = 'Age (Gyr)'
                            elif param_name == 'log_age':
                                label = 'Log Age (yr)'
                            elif param_name == 'metallicity':
                                label = 'Metallicity [Z/H]'
                            else:
                                label = param_name
                            
                            param_map = np.zeros_like(bin_map_2d, dtype=float)
                            for j, value in enumerate(param_values):
                                param_map[bin_map_2d == j] = value
                            
                            im = axes[i].imshow(param_map, origin='lower', cmap='plasma')
                            plt.colorbar(im, ax=axes[i], label=label)
                            axes[i].set_title(f'Stellar {label} Map')
                        
                        plt.tight_layout()
                        plt.savefig(plots_dir / f"{galaxy_name}_rdb_stellar_pop_map.png", dpi=150)
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Error creating 2D stellar population maps: {str(e)}")
            except Exception as e:
                logger.warning(f"Error creating stellar population plots: {str(e)}")
        
        # Create emission line plots
        if 'emission' in rdb_results:
            try:
                emission = rdb_results['emission']
                
                # Find flux maps
                flux_maps = {}
                for key, value in emission.items():
                    if key.startswith('flux_') and isinstance(value, np.ndarray):
                        line_name = key[5:]  # Remove 'flux_' prefix
                        flux_maps[line_name] = value
                
                if flux_maps:
                    n_lines = min(len(flux_maps), 6)  # Show at most 6 lines
                    fig, axes = plt.subplots(1, n_lines, figsize=(4 * n_lines, 4))
                    if n_lines == 1:
                        axes = [axes]
                    
                    for i, (line_name, flux) in enumerate(list(flux_maps.items())[:n_lines]):
                        axes[i].plot(binned_data.bin_radii, flux, 'o-')
                        axes[i].set_xlabel('Radius (arcsec)')
                        axes[i].set_ylabel('Flux')
                        axes[i].set_title(f'{line_name} Flux Profile')
                        axes[i].grid(True, alpha=0.3)
                        
                        # Use log scale for y-axis if all values are positive
                        if np.all(flux > 0):
                            axes[i].set_yscale('log')
                    
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"{galaxy_name}_rdb_emission_profile.png", dpi=150)
                    plt.close(fig)
                    
                    # Create 2D maps of emission line fluxes
                    try:
                        fig, axes = plt.subplots(1, n_lines, figsize=(4 * n_lines, 4))
                        if n_lines == 1:
                            axes = [axes]
                        
                        # Create 2D bin map
                        bin_map_2d = binned_data.bin_num.reshape(n_y, n_x)
                        
                        for i, (line_name, flux) in enumerate(list(flux_maps.items())[:n_lines]):
                            flux_map = np.zeros_like(bin_map_2d, dtype=float)
                            for j, value in enumerate(flux):
                                flux_map[bin_map_2d == j] = value
                            
                            # Use log scale for better visualization
                            with np.errstate(divide='ignore', invalid='ignore'):
                                log_flux_map = np.log10(flux_map)
                                log_flux_map[~np.isfinite(log_flux_map)] = np.nan
                            
                            im = axes[i].imshow(log_flux_map, origin='lower', cmap='inferno')
                            plt.colorbar(im, ax=axes[i], label='Log Flux')
                            axes[i].set_title(f'{line_name} Flux Map')
                        
                        plt.tight_layout()
                        plt.savefig(plots_dir / f"{galaxy_name}_rdb_emission_map.png", dpi=150)
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Error creating 2D emission line maps: {str(e)}")
            except Exception as e:
                logger.warning(f"Error creating emission line plots: {str(e)}")
        
        # Create spectral indices plots
        if 'indices' in rdb_results:
            try:
                indices = rdb_results['indices']
                index_names = list(indices.keys())
                
                if index_names:
                    n_indices = min(len(index_names), 6)  # Show at most 6 indices
                    fig, axes = plt.subplots(1, n_indices, figsize=(4 * n_indices, 4))
                    if n_indices == 1:
                        axes = [axes]
                    
                    for i, index_name in enumerate(index_names[:n_indices]):
                        index_values = indices[index_name]
                        
                        axes[i].plot(binned_data.bin_radii, index_values, 'o-')
                        axes[i].set_xlabel('Radius (arcsec)')
                        axes[i].set_ylabel('Index Value')
                        axes[i].set_title(f'{index_name} Index Profile')
                        axes[i].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"{galaxy_name}_rdb_indices_profile.png", dpi=150)
                    plt.close(fig)
                    
                    # Create 2D maps of spectral indices
                    try:
                        fig, axes = plt.subplots(1, n_indices, figsize=(4 * n_indices, 4))
                        if n_indices == 1:
                            axes = [axes]
                        
                        # Create 2D bin map
                        bin_map_2d = binned_data.bin_num.reshape(n_y, n_x)
                        
                        for i, index_name in enumerate(index_names[:n_indices]):
                            index_values = indices[index_name]
                            
                            index_map = np.zeros_like(bin_map_2d, dtype=float)
                            for j, value in enumerate(index_values):
                                index_map[bin_map_2d == j] = value
                            
                            im = axes[i].imshow(index_map, origin='lower', cmap='viridis')
                            plt.colorbar(im, ax=axes[i], label='Index Value')
                            axes[i].set_title(f'{index_name} Index Map')
                        
                        plt.tight_layout()
                        plt.savefig(plots_dir / f"{galaxy_name}_rdb_indices_map.png", dpi=150)
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Error creating 2D spectral index maps: {str(e)}")
            except Exception as e:
                logger.warning(f"Error creating spectral indices plots: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error creating RDB plots: {str(e)}")
        logger.error(traceback.format_exc())