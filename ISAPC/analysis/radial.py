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

def combine_radial_spectra_with_velocity_correction(spectra, wavelength, bin_indices, velocity_field, n_x, n_y):
    """
    Combine spectra within radial bins with velocity correction.
    
    Parameters
    ----------
    spectra : numpy.ndarray
        Array of spectra [n_wave, n_spectra]
    wavelength : numpy.ndarray
        Wavelength array
    bin_indices : list
        List of arrays with indices for each bin
    velocity_field : numpy.ndarray
        Velocity field for correction
    n_x : int
        Number of pixels in x direction
    n_y : int
        Number of pixels in y direction
        
    Returns
    -------
    numpy.ndarray
        Combined bin spectra array [n_wave, n_bins]
    """
    n_wave = len(wavelength)
    n_bins = len(bin_indices)
    c = 299792.458  # Speed of light in km/s
    
    # Initialize output array
    bin_spectra = np.zeros((n_wave, n_bins))
    
    # Set velocity limits for outlier correction
    vel_limit = 300  # Maximum velocity difference from median (km/s)
    max_velocity = 300  # Maximum absolute velocity (km/s)
    
    # Process each bin
    for i, indices in enumerate(bin_indices):
        # Skip empty bins
        if len(indices) == 0:
            bin_spectra[:, i] = np.nan
            continue
        
        try:
            # Extract velocities for this bin
            bin_velocities = []
            for idx in indices:
                row = idx // n_x
                col = idx % n_x
                if row < n_y and col < n_x:
                    if velocity_field is not None and row < velocity_field.shape[0] and col < velocity_field.shape[1]:
                        vel = velocity_field[row, col]
                        if np.isfinite(vel):
                            bin_velocities.append(vel)
            
            # Calculate median velocity for this bin
            median_velocity = np.median(bin_velocities) if bin_velocities else 0
            
            # Collect velocity-corrected spectra
            corrected_spectra = []
            
            for idx in indices:
                spec = spectra[:, idx]
                if not np.all(~np.isfinite(spec)):
                    # Get velocity for this pixel
                    vel = 0
                    
                    if velocity_field is not None:
                        row = idx // n_x
                        col = idx % n_x
                        if row < velocity_field.shape[0] and col < velocity_field.shape[1]:
                            pixel_vel = velocity_field[row, col]
                            
                            # Apply velocity limits as mentioned in your code snippet
                            if np.isfinite(pixel_vel):
                                # Check for outliers compared to bin median
                                if abs(pixel_vel - median_velocity) > vel_limit:
                                    vel = median_velocity
                                    logger.debug(f"Velocity outlier in bin {i}: pixel_vel={pixel_vel:.1f}, median={median_velocity:.1f}")
                                # Check for extreme velocities
                                elif abs(pixel_vel) > max_velocity:
                                    vel = 0
                                    logger.debug(f"Extreme velocity in bin {i}: pixel_vel={pixel_vel:.1f}")
                                else:
                                    vel = pixel_vel
                    
                    # Apply velocity shift
                    if abs(vel) > 1.0:  # Only apply for non-negligible velocities
                        try:
                            # Shift wavelength in opposite direction of velocity
                            # For redshift (v > 0), we need a bluer template, so divide lambda
                            # For blueshift (v < 0), we need a redder template, so multiply lambda
                            lam_shifted = wavelength / (1 + vel/c)
                            
                            # Use spectres for resampling
                            corrected_spec = spectres(wavelength, lam_shifted, spec)
                            corrected_spectra.append(corrected_spec)
                        except Exception as e:
                            logger.debug(f"Error in velocity correction for bin {i}, pixel {idx}: {e}")
                            corrected_spectra.append(spec)  # Add original as fallback
                    else:
                        corrected_spectra.append(spec)
            
            # Combine spectra if any valid
            if corrected_spectra:
                # Convert to array for easier operations
                spectra_array = np.array(corrected_spectra)
                
                # Compute median spectrum - more robust than mean
                bin_spectra[:, i] = np.nanmedian(spectra_array, axis=0)
                
                # Set all-NaN wavelengths to NaN in result
                all_nan = np.all(~np.isfinite(spectra_array), axis=0)
                bin_spectra[all_nan, i] = np.nan
            else:
                # No valid spectra
                bin_spectra[:, i] = np.nan
                
        except Exception as e:
            logger.error(f"Error combining spectra for bin {i}: {e}")
            bin_spectra[:, i] = np.nan
    
    return bin_spectra

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
    
    # Define data arrays for analysis
    x = cube.x
    y = cube.y
    
    try:
        # Calculate radial bins
        indices = np.arange(len(x))
        bin_num, bin_edges, bin_radii = calculate_radial_bins(
            x, y, center_x=center_x, center_y=center_y,
            pa=pa, ellipticity=ellipticity,
            n_rings=n_rings, log_spacing=log_spacing
        )
        
        # Get valid mask (pixels that are assigned to bins)
        valid_mask = bin_num >= 0
        
        if np.sum(valid_mask) == 0:
            logger.error("No valid pixels assigned to bins")
            return {'status': 'error', 'message': 'No valid pixels assigned to bins'}
        
        # Create bin indices
        bin_indices = []
        max_bin = int(np.max(bin_num))
        
        for i in range(max_bin + 1):
            bin_indices.append(indices[bin_num == i])
        
        # Get velocity field from P2P results if available, for velocity correction
        velocity_field = None
        if p2p_results is not None:
            try:
                # Try standard format first
                if 'stellar_kinematics' in p2p_results and 'velocity_field' in p2p_results['stellar_kinematics']:
                    velocity_field = p2p_results['stellar_kinematics']['velocity_field']
                # Try direct format
                elif 'velocity_field' in p2p_results:
                    velocity_field = p2p_results['velocity_field']
                
                if velocity_field is not None:
                    logger.info("Using velocity field from P2P results for velocity correction")
            except Exception as e:
                logger.warning(f"Error extracting velocity field from P2P results: {e}")
        
        # Calculate intersection of wavelength ranges accounting for velocity shifts
        if velocity_field is not None:
            wave_mask, min_wave, max_wave = calculate_wavelength_intersection(
                cube._lambda_gal, velocity_field, cube._n_x
            )
            logger.info(f"Velocity correction wavelength range: {min_wave:.1f} - {max_wave:.1f} Å")
        else:
            wave_mask = np.ones_like(cube._lambda_gal, dtype=bool)
        
        # Apply wavelength mask
        wavelength = cube._lambda_gal[wave_mask]
        
        # Combine spectra with improved velocity correction
        bin_spectra = combine_radial_spectra_with_velocity_correction(
            cube._spectra[wave_mask], wavelength, bin_indices,
            velocity_field, cube._n_x, cube._n_y
        )
        
        # Create metadata
        metadata = {
            'nx': cube._n_x,
            'ny': cube._n_y,
            'center_x': center_x,
            'center_y': center_y,
            'pa': pa,
            'ellipticity': ellipticity,
            'n_rings': n_rings,
            'log_spacing': log_spacing,
            'bin_edges': bin_edges,
            'time': time.time(),
            'galaxy_name': galaxy_name,
            'analysis_type': 'RDB',
            'pixelsize_x': cube._pxl_size_x,
            'pixelsize_y': cube._pxl_size_y,
            'redshift': cube._redshift if hasattr(cube, '_redshift') else 0.0
        }
        
        # Create RadialBinnedData object
        binned_data = RadialBinnedData(
            bin_num=bin_num,
            bin_indices=bin_indices,
            spectra=bin_spectra,
            wavelength=wavelength,
            metadata=metadata,
            bin_radii=bin_radii
        )
        
        # Run analysis on binned spectra
        rdb_results = run_analysis_on_binned_data(args, binned_data, cube, p2p_results)
        
        # Create visualization plots
        if not args.no_plots:
            create_rdb_plots(args, binned_data, rdb_results, galaxy_name, plots_dir)
        
        # Prepare output dictionary
        result_dict = {
            'analysis_type': 'RDB',
            'bin_num': bin_num,
            'bin_indices': bin_indices,
            'bin_info': {
                'bin_radii': bin_radii,
                'bin_edges': bin_edges
            },
            'parameters': {
                'center_x': center_x,
                'center_y': center_y,
                'pa': pa,
                'ellipticity': ellipticity,
                'n_rings': n_rings,
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
        logger.error(f"Error in RDB analysis: {str(e)}")
        logger.error(traceback.format_exc())
        
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


# Add this function to help plotting stellar population parameters
def plot_stellar_parameter(ax, bin_map_2d, param_values, param_name, i):
    """Helper function to plot stellar population parameters correctly"""
    # Handle different dimension cases
    param_map = np.zeros_like(bin_map_2d, dtype=float)
    param_map.fill(np.nan)  # Fill with NaN initially
    
    try:
        # For different parameter dimension cases
        if not isinstance(param_values, np.ndarray):
            # Not an array - skip
            ax.text(0.5, 0.5, f"No valid {param_name} data", 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Handle 1D arrays
        if len(param_values.shape) == 1:
            # Fill bin map with corresponding values
            for j, value in enumerate(param_values):
                if j < len(param_values) and np.isfinite(value):
                    param_map[bin_map_2d == j] = value
        
        # Handle 2D arrays
        elif len(param_values.shape) == 2:
            # Direct copy if shapes match
            if param_values.shape == param_map.shape:
                param_map[:] = param_values
            else:
                # Try to reshape
                try:
                    reshaped = param_values.reshape(param_map.shape)
                    param_map[:] = reshaped
                except:
                    # Fall back to bin-by-bin assignment
                    for j in range(np.max(bin_map_2d) + 1):
                        mask = bin_map_2d == j
                        if np.any(mask) and j < param_values.size:
                            param_map[mask] = param_values.flat[j]
                    
        # For higher dimensions, flatten and use what we can
        else:
            flat_param = param_values.flatten()
            for j in range(min(np.max(bin_map_2d) + 1, len(flat_param))):
                mask = bin_map_2d == j
                if np.any(mask) and j < len(flat_param):
                    param_map[mask] = flat_param[j]
        
        # Adjust display for age
        if param_name == 'age':
            param_map = param_map * 1e-9  # Convert to Gyr
            label = 'Age (Gyr)'
        elif param_name == 'log_age':
            label = 'Log Age (yr)'
        elif param_name == 'metallicity':
            label = 'Metallicity [Z/H]'
        else:
            label = param_name
            
        # Check if we have valid data
        valid_param = param_map[np.isfinite(param_map)]
        if len(valid_param) > 0:
            vmin = np.percentile(valid_param, 5)
            vmax = np.percentile(valid_param, 95)
            im = ax.imshow(param_map, origin='lower', cmap='plasma',
                         vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, label=label)
            ax.set_title(f'Stellar {label} Map')
        else:
            ax.text(0.5, 0.5, f"No valid {param_name} data", 
                   ha='center', va='center', transform=ax.transAxes)
    except Exception as e:
        logger.warning(f"Error plotting {param_name}: {e}")
        ax.text(0.5, 0.5, f"Error plotting {param_name}", 
               ha='center', va='center', transform=ax.transAxes)


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
            
            if velocity is not None and dispersion is not None and np.any(np.isfinite(velocity)) and np.any(np.isfinite(dispersion)):
                try:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Ensure bin_radii and velocity/dispersion have the same length
                    radii = binned_data.bin_radii
                    if len(radii) != len(velocity):
                        # Keep only common length
                        common_length = min(len(radii), len(velocity))
                        radii = radii[:common_length]
                        velocity = velocity[:common_length]
                        dispersion = dispersion[:common_length] if len(dispersion) >= common_length else dispersion
                    
                    # Filter out NaN values
                    vel_mask = np.isfinite(velocity)
                    disp_mask = np.isfinite(dispersion)
                    
                    # Plot velocity profile
                    if np.any(vel_mask):
                        axes[0].plot(radii[vel_mask], velocity[vel_mask], 'o-', label='Velocity')
                        axes[0].set_xlabel('Radius (arcsec)')
                        axes[0].set_ylabel('Velocity (km/s)')
                        axes[0].set_title('Stellar Velocity Profile')
                        axes[0].grid(True, alpha=0.3)
                    else:
                        axes[0].text(0.5, 0.5, "No valid velocity data", 
                                  ha='center', va='center', transform=axes[0].transAxes)
                    
                    # Plot dispersion profile
                    if np.any(disp_mask):
                        axes[1].plot(radii[disp_mask], dispersion[disp_mask], 'o-', label='Dispersion')
                        axes[1].set_xlabel('Radius (arcsec)')
                        axes[1].set_ylabel('Dispersion (km/s)')
                        axes[1].set_title('Stellar Dispersion Profile')
                        axes[1].grid(True, alpha=0.3)
                    else:
                        axes[1].text(0.5, 0.5, "No valid dispersion data", 
                                  ha='center', va='center', transform=axes[1].transAxes)
                    
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
                        velocity_map.fill(np.nan)  # Fill with NaN initially
                        for i, vel in enumerate(velocity):
                            if i < len(velocity) and np.isfinite(vel):
                                velocity_map[bin_map_2d == i] = vel
                        
                        # Check if we have valid data
                        if np.any(np.isfinite(velocity_map)):
                            valid_vel = velocity_map[np.isfinite(velocity_map)]
                            vmin = np.percentile(valid_vel, 5)
                            vmax = np.percentile(valid_vel, 95)
                            im0 = axes[0].imshow(velocity_map, origin='lower', cmap='coolwarm',
                                              vmin=vmin, vmax=vmax)
                            plt.colorbar(im0, ax=axes[0], label='Velocity (km/s)')
                        else:
                            axes[0].text(0.5, 0.5, "No valid velocity data", 
                                      ha='center', va='center', transform=axes[0].transAxes)
                        axes[0].set_title('Stellar Velocity Map')
                        
                        # Plot dispersion map
                        dispersion_map = np.zeros_like(bin_map_2d, dtype=float)
                        dispersion_map.fill(np.nan)  # Fill with NaN initially
                        for i, disp in enumerate(dispersion):
                            if i < len(dispersion) and np.isfinite(disp):
                                dispersion_map[bin_map_2d == i] = disp
                        
                        # Check if we have valid data
                        if np.any(np.isfinite(dispersion_map)):
                            valid_disp = dispersion_map[np.isfinite(dispersion_map)]
                            vmin = np.percentile(valid_disp, 5)
                            vmax = np.percentile(valid_disp, 95)
                            im1 = axes[1].imshow(dispersion_map, origin='lower', cmap='viridis', 
                                              vmin=vmin, vmax=vmax)
                            plt.colorbar(im1, ax=axes[1], label='Dispersion (km/s)')
                        else:
                            axes[1].text(0.5, 0.5, "No valid dispersion data", 
                                      ha='center', va='center', transform=axes[1].transAxes)
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
                        
                        # Ensure bin_radii and param_values have the same length
                        radii = binned_data.bin_radii
                        if len(radii) != len(param_values):
                            # Keep only common length
                            common_length = min(len(radii), len(param_values))
                            radii = radii[:common_length]
                            param_values = param_values[:common_length]
                        
                        # Filter out NaN values
                        valid_mask = np.isfinite(param_values) & np.isfinite(radii)
                        
                        if np.any(valid_mask):
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
                            
                            axes[i].plot(radii[valid_mask], param_values[valid_mask], 'o-')
                            axes[i].set_xlabel('Radius (arcsec)')
                            axes[i].set_ylabel(label)
                            axes[i].set_title(f'Stellar {label} Profile')
                            axes[i].grid(True, alpha=0.3)
                        else:
                            axes[i].text(0.5, 0.5, f"No valid {param_name} data", 
                                      ha='center', va='center', transform=axes[i].transAxes)
                    
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"{galaxy_name}_rdb_stellar_pop_profile.png", dpi=150)
                    plt.close(fig)
                    
                    # Create 2D maps of stellar population parameters
                    # Replace the 2D stellar population maps section in create_rdb_plots with:
                    # In create_rdb_plots, replace the stellar population plotting code with:

                    # Create stellar population radial profiles
                    if 'stellar_population' in rdb_results:
                        try:
                            from visualization import safe_plot_array
                            
                            params = rdb_results['stellar_population']
                            param_names = list(params.keys())
                            
                            if param_names:
                                # 1. Create profile plots
                                fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 4))
                                if len(param_names) == 1:
                                    axes = [axes]
                                
                                for i, param_name in enumerate(param_names):
                                    param_values = params[param_name]
                                    
                                    # Ensure bin_radii and param_values have the same length
                                    radii = binned_data.bin_radii
                                    if len(radii) != len(param_values):
                                        # Keep only common length
                                        common_length = min(len(radii), len(param_values))
                                        radii = radii[:common_length]
                                        param_values = param_values[:common_length]
                                    
                                    # Filter out NaN values
                                    valid_mask = np.isfinite(param_values) & np.isfinite(radii)
                                    
                                    if np.any(valid_mask):
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
                                        
                                        axes[i].plot(radii[valid_mask], param_values[valid_mask], 'o-')
                                        axes[i].set_xlabel('Radius (arcsec)')
                                        axes[i].set_ylabel(label)
                                        axes[i].set_title(f'Stellar {label} Profile')
                                        axes[i].grid(True, alpha=0.3)
                                    else:
                                        axes[i].text(0.5, 0.5, f"No valid {param_name} data", 
                                                ha='center', va='center', transform=axes[i].transAxes)
                                
                                plt.tight_layout()
                                plt.savefig(plots_dir / f"{galaxy_name}_rdb_stellar_pop_profile.png", dpi=150)
                                plt.close(fig)
                                
                                # 2. Create 2D maps of stellar population parameters using safe function
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
                                            # Convert to Gyr before plotting
                                            if isinstance(param_values, np.ndarray):
                                                param_values = param_values * 1e-9
                                            label = 'Age (Gyr)'
                                        elif param_name == 'log_age':
                                            label = 'Log Age (yr)'
                                        elif param_name == 'metallicity':
                                            label = 'Metallicity [Z/H]'
                                        else:
                                            label = param_name
                                        
                                        # Use safe plotting function
                                        safe_plot_array(
                                            param_values, 
                                            bin_map_2d, 
                                            ax=axes[i], 
                                            title=f'Stellar {label}', 
                                            cmap='plasma', 
                                            label=label
                                        )
                                    
                                    plt.tight_layout()
                                    plt.savefig(plots_dir / f"{galaxy_name}_rdb_stellar_pop_map.png", dpi=150)
                                    plt.close(fig)
                                except Exception as e:
                                    logger.warning(f"Error creating 2D stellar population maps: {str(e)}")
                                    plt.close('all')
                                    
                        except Exception as e:
                            logger.warning(f"Error creating stellar population plots: {str(e)}")
                            plt.close('all')
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
                        # Ensure bin_radii and flux have the same length
                        radii = binned_data.bin_radii
                        if len(radii) != len(flux):
                            # Keep only common length
                            common_length = min(len(radii), len(flux))
                            radii = radii[:common_length]
                            flux = flux[:common_length]
                        
                        # Filter out NaN values
                        valid_mask = np.isfinite(flux) & np.isfinite(radii)
                        
                        if np.any(valid_mask):
                            axes[i].plot(radii[valid_mask], flux[valid_mask], 'o-')
                            axes[i].set_xlabel('Radius (arcsec)')
                            axes[i].set_ylabel('Flux')
                            axes[i].set_title(f'{line_name} Flux Profile')
                            axes[i].grid(True, alpha=0.3)
                            
                            # Use log scale for y-axis if all values are positive
                            if np.all(flux[valid_mask] > 0):
                                axes[i].set_yscale('log')
                        else:
                            axes[i].text(0.5, 0.5, f"No valid {line_name} data", 
                                      ha='center', va='center', transform=axes[i].transAxes)
                    
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
                            flux_map.fill(np.nan)  # Fill with NaN initially
                            for j, value in enumerate(flux):
                                if j < len(flux) and np.isfinite(value):
                                    flux_map[bin_map_2d == j] = value
                            
                            # Check if we have valid data
                            if np.any(np.isfinite(flux_map) & (flux_map > 0)):
                                # Use log scale for better visualization
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    log_flux_map = np.log10(flux_map)
                                    # Mark non-finite values as NaN
                                    log_flux_map[~np.isfinite(log_flux_map)] = np.nan
                                
                                valid_log_flux = log_flux_map[np.isfinite(log_flux_map)]
                                if len(valid_log_flux) > 0:
                                    vmin = np.percentile(valid_log_flux, 5)
                                    vmax = np.percentile(valid_log_flux, 95)
                                    im = axes[i].imshow(log_flux_map, origin='lower', cmap='inferno',
                                                     vmin=vmin, vmax=vmax)
                                    plt.colorbar(im, ax=axes[i], label='Log Flux')
                                else:
                                    axes[i].text(0.5, 0.5, f"No valid {line_name} data", 
                                              ha='center', va='center', transform=axes[i].transAxes)
                            else:
                                axes[i].text(0.5, 0.5, f"No valid {line_name} data", 
                                          ha='center', va='center', transform=axes[i].transAxes)
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
                        
                        # Ensure bin_radii and index_values have the same length
                        radii = binned_data.bin_radii
                        if len(radii) != len(index_values):
                            # Keep only common length
                            common_length = min(len(radii), len(index_values))
                            radii = radii[:common_length]
                            index_values = index_values[:common_length]
                        
                        # Filter out NaN values
                        valid_mask = np.isfinite(index_values) & np.isfinite(radii)
                        
                        if np.any(valid_mask):
                            axes[i].plot(radii[valid_mask], index_values[valid_mask], 'o-')
                            axes[i].set_xlabel('Radius (arcsec)')
                            axes[i].set_ylabel('Index Value')
                            axes[i].set_title(f'{index_name} Index Profile')
                            axes[i].grid(True, alpha=0.3)
                        else:
                            axes[i].text(0.5, 0.5, f"No valid {index_name} data", 
                                      ha='center', va='center', transform=axes[i].transAxes)
                    
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
                            index_map.fill(np.nan)  # Fill with NaN initially
                            for j, value in enumerate(index_values):
                                if j < len(index_values) and np.isfinite(value):
                                    index_map[bin_map_2d == j] = value
                            
                            # Check if we have valid data
                            if np.any(np.isfinite(index_map)):
                                valid_index = index_map[np.isfinite(index_map)]
                                vmin = np.percentile(valid_index, 5)
                                vmax = np.percentile(valid_index, 95)
                                im = axes[i].imshow(index_map, origin='lower', cmap='viridis',
                                                 vmin=vmin, vmax=vmax)
                                plt.colorbar(im, ax=axes[i], label='Index Value')
                            else:
                                axes[i].text(0.5, 0.5, f"No valid {index_name} data", 
                                          ha='center', va='center', transform=axes[i].transAxes)
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