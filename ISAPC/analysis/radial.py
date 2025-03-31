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
        try:
            # Try loading from standard paths
            p2p_results_path = data_dir / f"{galaxy_name}_P2P_results.npz"
            std_results_path = data_dir / f"{galaxy_name}_P2P_standardized.npz"
            
            if p2p_results_path.exists():
                p2p_results = np.load(p2p_results_path, allow_pickle=True)
                logger.info("Successfully loaded P2P results for RDB analysis")
            elif std_results_path.exists():
                p2p_results = np.load(std_results_path, allow_pickle=True)
                logger.info("Successfully loaded standardized P2P results for RDB analysis")
        except Exception as e:
            logger.warning(f"Error loading P2P results: {e}")
            p2p_results = None
    
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
        
        # Set up binning in the cube
        cube.setup_binning('RDB', binned_data)
        
        # Run analysis using the enhanced MUSECube methods
        velocity_field, dispersion_field, bestfit_field, optimal_tmpls, poly_coeffs = cube.fit_spectra(
            template_filename=args.template,
            ppxf_vel_init=args.vel_init,
            ppxf_vel_disp_init=args.sigma_init,
            ppxf_deg=args.poly_degree if hasattr(args, 'poly_degree') else 3,
            n_jobs=args.n_jobs
        )
        
        # Fit emission lines if requested
        emission_result = None
        if not args.no_emission:
            emission_result = cube.fit_emission_lines(
                template_filename=args.template,
                ppxf_vel_init=velocity_field,
                ppxf_sig_init=args.sigma_init,
                ppxf_deg=2,
                n_jobs=args.n_jobs
            )
        # Before returning the results, add:
        # Ensure emission lines are consistently processed
        if emission_result is not None and hasattr(cube, '_post_process_emission_results'):
            cube._post_process_emission_results()
        
        # Calculate spectral indices if requested
        indices_result = None
        if not args.no_indices:
            indices_result = cube.calculate_spectral_indices(
                n_jobs=args.n_jobs
            )
        
        # In run_vnb_analysis/run_rdb_analysis:

        
        
        # Prepare standardized output dictionary
        rdb_results = {
            'analysis_type': 'RDB',
            'stellar_kinematics': {
                'velocity': cube._bin_velocity,
                'dispersion': cube._bin_dispersion,
                'velocity_field': velocity_field,
                'dispersion_field': dispersion_field
            },
            'distance': {
                'bin_distances': bin_radii,
                'pixelsize_x': cube._pxl_size_x,
                'pixelsize_y': cube._pxl_size_y
            },
            'binning': {
                'bin_num': bin_num,
                'bin_indices': bin_indices,
                'bin_radii': bin_radii,
                'bin_edges': bin_edges,
                'center_x': center_x,
                'center_y': center_y,
                'pa': pa,
                'ellipticity': ellipticity,
                'n_rings': n_rings,
                'log_spacing': log_spacing
            }
        }
        
        # Add emission line results if available
        if emission_result is not None:
            rdb_results['emission'] = {}
            
            # Copy emission line fields from cube if available
            if hasattr(cube, '_bin_emission_flux'):
                rdb_results['emission']['flux'] = cube._bin_emission_flux
            if hasattr(cube, '_bin_emission_vel'):
                rdb_results['emission']['velocity'] = cube._bin_emission_vel
            if hasattr(cube, '_bin_emission_sig'):
                rdb_results['emission']['dispersion'] = cube._bin_emission_sig
                
            # Copy emission fields from emission_result
            for key in ['emission_flux', 'emission_vel', 'emission_sig']:
                if key in emission_result:
                    field_name = key.split('_')[1]  # extract 'flux', 'vel', 'sig'
                    rdb_results['emission'][field_name] = emission_result[key]
                    
            # Add emission line wavelengths if available
            if 'emission_wavelength' in emission_result:
                rdb_results['emission']['wavelengths'] = emission_result['emission_wavelength']
        

        # After calculating spectral indices
        if indices_result is not None:
            rdb_results['indices'] = indices_result  # This will be pixel-based
            
            # Add bin-level indices if available
            if hasattr(cube, '_bin_indices_result') and cube._bin_indices_result:
                rdb_results['bin_indices'] = cube._bin_indices_result

        # Add this after spectral indices calculation in both voronoi.py and radial.py:

        # Create spectral index visualizations if requested
        if indices_result is not None and not args.no_plots:
            try:
                # Create directory for spectral index visualization
                indices_plots_dir = plots_dir / "spectral_indices"
                indices_plots_dir.mkdir(exist_ok=True, parents=True)
                
                # Plot indices for a few sample bins
                n_bins_to_plot = min(5, cube._n_bins)
                bin_indices_to_plot = []
                
                # Try to select bins with valid velocities
                if hasattr(cube, '_bin_velocity'):
                    valid_bins = [i for i in range(len(cube._bin_velocity)) 
                                if np.isfinite(cube._bin_velocity[i])]
                    
                    if len(valid_bins) > 0:
                        # Get evenly spaced bins
                        step = max(1, len(valid_bins) // n_bins_to_plot)
                        bin_indices_to_plot = valid_bins[::step][:n_bins_to_plot]
                    
                # If no valid bins found, just use the first few
                if not bin_indices_to_plot:
                    bin_indices_to_plot = range(n_bins_to_plot)
                
                # Plot each bin
                for bin_idx in bin_indices_to_plot:
                    try:
                        # Use the dedicated method to ensure consistency with calculation
                        fig, axes = cube.plot_bin_index_calculation(bin_idx, save_dir=indices_plots_dir)
                        if fig is not None:
                            plt.close(fig)  # Close to avoid memory issues
                        
                    except Exception as e:
                        logger.warning(f"Error plotting spectral indices for bin {bin_idx}: {e}")
                        
            except Exception as e:
                logger.warning(f"Error creating spectral index visualizations: {e}")
        
        # Save binned data object
        binned_data_path = data_dir / f"{galaxy_name}_RDB_binned_data.npz"
        binned_data.save(binned_data_path)
        logger.info(f"Saved binned data to {binned_data_path}")
        
        # Save results
        save_standardized_results(galaxy_name, 'RDB', rdb_results, output_dir)
        
        # Create visualization plots if requested
        if not hasattr(args, 'no_plots') or not args.no_plots:
            binned_data.create_visualization_plots(plots_dir, galaxy_name)
            create_rdb_plots(cube, rdb_results, galaxy_name, plots_dir, args)

        # Add these lines near the end of the run_rdb_analysis function in radial.py
        # Just before returning the results

        # Create bin spectrum plots if requested
        if not args.no_plots:
            # Create directory for bin plots
            bin_plots_dir = plots_dir / "bin_spectra"
            bin_plots_dir.mkdir(exist_ok=True, parents=True)
            
            # Plot a sample of bin spectra
            try:
                cube.plot_bin_fits(n_bins=min(5, cube._n_bins), save_dir=bin_plots_dir)
            except Exception as e:
                logger.warning(f"Error creating bin fitting plots: {e}")
        
        logger.info(f"RDB analysis completed in {time.time() - start_time:.1f} seconds")

        # Create comprehensive visualization plots
        if not args.no_plots:
            try:
                # Create visualization plots directory
                viz_plots_dir = plots_dir / "visualizations"
                viz_plots_dir.mkdir(exist_ok=True, parents=True)
                
                # Generate all visualization plots
                cube.plot_bin_analysis_results(output_dir=viz_plots_dir)
                
                logger.info(f"Created visualization plots in {viz_plots_dir}")
            except Exception as e:
                logger.warning(f"Error creating visualization plots: {e}")
        
        return rdb_results
    
    except Exception as e:
        logger.error(f"Error in RDB analysis: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            'analysis_type': 'RDB',
            'status': 'error',
            'error': str(e)
        }

def create_rdb_plots(cube, rdb_results, galaxy_name, plots_dir, args):
    """
    Create visualization plots for RDB analysis
    
    Parameters
    ----------
    cube : MUSECube
        MUSE cube with binned data
    rdb_results : dict
        RDB analysis results
    galaxy_name : str
        Galaxy name
    plots_dir : Path
        Directory to save plots
    args : argparse.Namespace
        Command line arguments
    """
    try:
        import visualization
        
        # Create radial profiles of key parameters
        
        # Create kinematics radial profile
        if 'stellar_kinematics' in rdb_results and 'bin_distances' in rdb_results.get('distance', {}):
            try:
                bin_radii = rdb_results['distance']['bin_distances']
                velocity = rdb_results['stellar_kinematics']['velocity']
                dispersion = rdb_results['stellar_kinematics']['dispersion']
                
                # Create radial velocity profile
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(bin_radii, velocity, 'o-', label='Velocity')
                ax.set_xlabel('Radius (arcsec)')
                ax.set_ylabel('Velocity (km/s)')
                ax.set_title(f"{galaxy_name} - Radial Velocity Profile")
                ax.grid(True, alpha=0.3)
                visualization.standardize_figure_saving(fig, plots_dir / f"{galaxy_name}_RDB_velocity_profile.png")
                plt.close(fig)
                
                # Create radial dispersion profile
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(bin_radii, dispersion, 'o-', label='Dispersion')
                ax.set_xlabel('Radius (arcsec)')
                ax.set_ylabel('Dispersion (km/s)')
                ax.set_title(f"{galaxy_name} - Radial Dispersion Profile")
                ax.grid(True, alpha=0.3)
                visualization.standardize_figure_saving(fig, plots_dir / f"{galaxy_name}_RDB_dispersion_profile.png")
                plt.close(fig)
                
                # Create combined kinematics plot
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot velocity profile
                axes[0].plot(bin_radii, velocity, 'o-', label='Velocity')
                axes[0].set_xlabel('Radius (arcsec)')
                axes[0].set_ylabel('Velocity (km/s)')
                axes[0].set_title('Stellar Velocity Profile')
                axes[0].grid(True, alpha=0.3)
                
                # Plot dispersion profile
                axes[1].plot(bin_radii, dispersion, 'o-', label='Dispersion')
                axes[1].set_xlabel('Radius (arcsec)')
                axes[1].set_ylabel('Dispersion (km/s)')
                axes[1].set_title('Stellar Dispersion Profile')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                visualization.standardize_figure_saving(fig, plots_dir / f"{galaxy_name}_RDB_kinematics_profile.png")
                plt.close(fig)
                
                # Also create 2D maps of the radial bins
                bin_num = rdb_results['binning']['bin_num']
                
                # Ensure bin_num is properly formatted for plotting
                if isinstance(bin_num, np.ndarray) and bin_num.ndim > 1:
                    bin_num_2d = bin_num
                else:
                    # Reshape to 2D if possible
                    try:
                        bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
                    except:
                        # Keep as is if reshape fails
                        bin_num_2d = bin_num
                
                # Velocity map
                fig, ax = plt.subplots(figsize=(10, 8))
                visualization.plot_bin_map(
                    bin_num_2d, 
                    velocity, 
                    ax=ax, 
                    cmap='coolwarm',
                    title=f'{galaxy_name} - RDB Velocity',
                    vmin=np.percentile(velocity[np.isfinite(velocity)], 5),
                    vmax=np.percentile(velocity[np.isfinite(velocity)], 95),
                    colorbar_label='Velocity (km/s)'
                )
                visualization.standardize_figure_saving(fig, plots_dir / f"{galaxy_name}_RDB_velocity_map.png")
                plt.close(fig)
                
                # Dispersion map
                fig, ax = plt.subplots(figsize=(10, 8))
                visualization.plot_bin_map(
                    bin_num_2d, 
                    dispersion, 
                    ax=ax, 
                    cmap='viridis',
                    title=f'{galaxy_name} - RDB Dispersion',
                    vmin=np.percentile(dispersion[np.isfinite(dispersion)], 5),
                    vmax=np.percentile(dispersion[np.isfinite(dispersion)], 95),
                    colorbar_label='Dispersion (km/s)'
                )
                visualization.standardize_figure_saving(fig, plots_dir / f"{galaxy_name}_RDB_dispersion_map.png")
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Error creating kinematics plots: {e}")
                plt.close('all')
        
        # Create emission line plots if available
        if 'emission' in rdb_results and 'bin_distances' in rdb_results.get('distance', {}):
            try:
                bin_radii = rdb_results['distance']['bin_distances']
                emission = rdb_results['emission']
                bin_num = rdb_results['binning']['bin_num']
                
                # Ensure bin_num is properly formatted for plotting
                if isinstance(bin_num, np.ndarray) and bin_num.ndim > 1:
                    bin_num_2d = bin_num
                else:
                    # Reshape to 2D if possible
                    try:
                        bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
                    except:
                        # Keep as is if reshape fails
                        bin_num_2d = bin_num
                
                # Check for flux dictionaries
                flux_entries = {}
                
                # First try 'flux' dictionary
                if 'flux' in emission and isinstance(emission['flux'], dict):
                    flux_entries = emission['flux']
                
                # Also try other keys that might contain flux information
                for key in emission:
                    if key.startswith('flux_') and isinstance(emission[key], np.ndarray):
                        line_name = key[5:]  # Remove 'flux_' prefix
                        flux_entries[line_name] = emission[key]
                    elif isinstance(emission[key], dict) and 'flux' in key.lower():
                        # Might be a nested dictionary of fluxes
                        for subkey, subvalue in emission[key].items():
                            if isinstance(subvalue, np.ndarray):
                                flux_entries[subkey] = subvalue
                
                # Process each emission line flux
                for line_name, flux in flux_entries.items():
                    if isinstance(flux, np.ndarray):
                        # Check if flux has the right shape for radial profiles
                        if len(flux) == len(bin_radii):
                            # Good shape for radial profile - create plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Replace NaN with zeros for plotting
                            flux_plot = np.nan_to_num(flux, nan=0.0)
                            
                            ax.plot(bin_radii, flux_plot, 'o-', label=line_name)
                            ax.set_xlabel('Radius (arcsec)')
                            ax.set_ylabel('Flux')
                            ax.set_title(f"{galaxy_name} - {line_name} Radial Profile")
                            ax.grid(True, alpha=0.3)
                            
                            # Try using log scale for y-axis if all values are positive
                            if np.all(flux_plot[np.isfinite(flux_plot)] > 0):
                                ax.set_yscale('log')
                                
                            visualization.standardize_figure_saving(fig, plots_dir / f"{galaxy_name}_RDB_{line_name}_profile.png")
                            plt.close(fig)
                            
                            # Create 2D map if bin_num is properly formatted
                            fig, ax = plt.subplots(figsize=(10, 8))
                            visualization.plot_bin_map(
                                bin_num_2d, 
                                flux_plot, 
                                ax=ax, 
                                cmap='inferno',
                                title=f'{galaxy_name} - RDB {line_name} Flux',
                                log_scale=True,
                                colorbar_label='Log Flux'
                            )
                            visualization.standardize_figure_saving(fig, plots_dir / f"{galaxy_name}_RDB_{line_name}_map.png")
                            plt.close(fig)
                        else:
                            logger.warning(f"Skipping {line_name} plot - dimension mismatch: bin_radii shape {bin_radii.shape}, flux shape {flux.shape}")
            except Exception as e:
                logger.warning(f"Error creating emission line plots: {e}")
                plt.close('all')
        
        # Create spectral indices plots if available
        if 'indices' in rdb_results and 'bin_distances' in rdb_results.get('distance', {}):
            try:
                bin_radii = rdb_results['distance']['bin_distances']
                bin_num = rdb_results['binning']['bin_num']
                
                # Ensure bin_num is properly formatted for plotting
                if isinstance(bin_num, np.ndarray) and bin_num.ndim > 1:
                    bin_num_2d = bin_num
                else:
                    # Reshape to 2D if possible
                    try:
                        bin_num_2d = bin_num.reshape(cube._n_y, cube._n_x)
                    except:
                        # Keep as is if reshape fails
                        bin_num_2d = bin_num
                
                indices_found = False
                
                # Try bin indices first
                if 'bin_indices' in rdb_results:
                    for idx_name, idx_values in rdb_results['bin_indices'].items():
                        if isinstance(idx_values, np.ndarray):
                            # Check if dimensions match
                            if len(idx_values) == len(bin_radii):
                                indices_found = True
                                
                                # Create radial profile
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(bin_radii, idx_values, 'o-', label=idx_name)
                                ax.set_xlabel('Radius (arcsec)')
                                ax.set_ylabel('Index Value')
                                ax.set_title(f"{galaxy_name} - {idx_name} Radial Profile")
                                ax.grid(True, alpha=0.3)
                                visualization.standardize_figure_saving(fig, plots_dir / f"{galaxy_name}_RDB_{idx_name}_profile.png")
                                plt.close(fig)
                                
                                # Create 2D map
                                fig, ax = plt.subplots(figsize=(10, 8))
                                visualization.plot_bin_map(
                                    bin_num_2d, 
                                    idx_values, 
                                    ax=ax, 
                                    cmap='plasma',
                                    title=f'{galaxy_name} - RDB {idx_name}',
                                    colorbar_label='Index Value'
                                )
                                visualization.standardize_figure_saving(fig, plots_dir / f"{galaxy_name}_RDB_{idx_name}_map.png")
                                plt.close(fig)
                            else:
                                logger.warning(f"Skipping {idx_name} plot - dimension mismatch: bin_radii shape {bin_radii.shape}, index shape {idx_values.shape}")
                
                # If no bin indices, try indices
                if not indices_found and isinstance(rdb_results['indices'], dict):
                    for idx_name, idx_values in rdb_results['indices'].items():
                        # For map plots, we need to extract values for each bin
                        if hasattr(cube, '_bin_indices_result') and idx_name in cube._bin_indices_result:
                            bin_idx_values = cube._bin_indices_result[idx_name]
                            
                            # Check if dimensions match
                            if len(bin_idx_values) == len(bin_radii):
                                # Create radial profile
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(bin_radii, bin_idx_values, 'o-', label=idx_name)
                                ax.set_xlabel('Radius (arcsec)')
                                ax.set_ylabel('Index Value')
                                ax.set_title(f"{galaxy_name} - {idx_name} Radial Profile")
                                ax.grid(True, alpha=0.3)
                                visualization.standardize_figure_saving(fig, plots_dir / f"{galaxy_name}_RDB_{idx_name}_profile.png")
                                plt.close(fig)
                                
                                # Create 2D map
                                fig, ax = plt.subplots(figsize=(10, 8))
                                visualization.plot_bin_map(
                                    bin_num_2d, 
                                    bin_idx_values, 
                                    ax=ax, 
                                    cmap='plasma',
                                    title=f'{galaxy_name} - RDB {idx_name}',
                                    colorbar_label='Index Value'
                                )
                                visualization.standardize_figure_saving(fig, plots_dir / f"{galaxy_name}_RDB_{idx_name}_map.png")
                                plt.close(fig)
                            else:
                                logger.warning(f"Skipping {idx_name} plot - dimension mismatch: bin_radii shape {bin_radii.shape}, index shape {bin_idx_values.shape}")
            except Exception as e:
                logger.warning(f"Error creating spectral indices plots: {e}")
                plt.close('all')
            
            # Add spectral indices visualization
            if 'indices' in rdb_results and 'bin_indices' in rdb_results:
                try:
                    bin_indices = rdb_results['bin_indices']
                    if bin_indices and isinstance(bin_indices, dict):
                        for idx_name, idx_values in bin_indices.items():
                            if isinstance(idx_values, np.ndarray) and len(idx_values) > 0:
                                # Check if dimensions match
                                if len(idx_values) == len(bin_radii):
                                    # Create 2D index map
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    
                                    # Use safe_plot_array for robust plotting
                                    visualization.safe_plot_array(
                                        idx_values, 
                                        bin_num_2d, 
                                        ax=ax, 
                                        title=f'{galaxy_name} - {idx_name}',
                                        cmap='plasma',
                                        label='Index Value'
                                    )
                                    
                                    visualization.standardize_figure_saving(fig, plots_dir / f"{galaxy_name}_RDB_{idx_name}.png")
                                    plt.close(fig)
                                else:
                                    logger.warning(f"Skipping {idx_name} map - dimension mismatch: bin_radii shape {bin_radii.shape}, index shape {idx_values.shape}")
                except Exception as e:
                    logger.warning(f"Error creating spectral indices maps: {e}")
                    plt.close('all')
    
    except Exception as e:
        logger.error(f"Error in create_rdb_plots: {str(e)}")
        logger.error(traceback.format_exc())
        plt.close('all')