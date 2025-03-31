"""
Voronoi binning analysis module for ISAPC
Version 5.0.0 - Enhanced with improved SNR target selection
"""
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
import warnings
import traceback
import spectral_indices
import galaxy_params
import visualization
from utils.io import save_results_to_npz, save_standardized_results
from binning import (
    BinnedSpectra, VoronoiBinnedData, calculate_wavelength_intersection,
    combine_spectra_efficiently, calculate_snr
)
from utils.calc import spectres, apply_velocity_shift
from vorbin.voronoi_2d_binning import voronoi_2d_binning

logger = logging.getLogger(__name__)

# Speed of light in km/s
C_KMS = 299792.458

def run_vnb_analysis(args, cube, p2p_results=None):
    """
    Run Voronoi binning analysis on MUSE data cube with improved SNR targeting
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cube : MUSECube
        MUSE data cube object
    p2p_results : dict, optional
        P2P analysis results to use for binning
        
    Returns
    -------
    dict
        Analysis results
    """
    logger.info("Starting Voronoi binning analysis...")
    start_time = time.time()
    
    # Disable warnings for spectral indices
    spectral_indices.set_warnings(False)
    
    # Get galaxy name and create directories
    galaxy_name = Path(args.filename).stem
    output_dir = Path(args.output_dir)
    galaxy_dir = output_dir / galaxy_name
    data_dir = galaxy_dir / 'Data'
    plots_dir = galaxy_dir / 'Plots' / 'VNB'
    
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
                logger.info("Successfully loaded P2P results for VNB analysis")
            elif std_results_path.exists():
                p2p_results = np.load(std_results_path, allow_pickle=True)
                logger.info("Successfully loaded standardized P2P results for VNB analysis")
        except Exception as e:
            logger.warning(f"Error loading P2P results: {e}")
            p2p_results = None
    
    # Set up target SNR and other binning parameters
    # Base target SNR from arguments, but will be adjusted if binning fails
    target_snr = args.target_snr if hasattr(args, 'target_snr') else 30
    min_snr = args.min_snr if hasattr(args, 'min_snr') else 1
    use_cvt = args.cvt if hasattr(args, 'cvt') else True
    
    # Extract coordinates, signal, and noise for VNB
    # Use wavelength-integrated signal-to-noise
    x = cube.x
    y = cube.y
    
    # Following the notebook's approach, use a specific wavelength range for SNR calculation
    # This range (5075-5125 Å) is often used for continuum SNR assessment
    wave_mask = (cube._lambda_gal >= 5075) & (cube._lambda_gal <= 5125)
    if np.sum(wave_mask) > 0:
        # Calculate SNR using this specific range
        signal = np.nanmedian(cube._spectra[wave_mask], axis=0)
        noise = np.nanstd(cube._spectra[wave_mask], axis=0)
        logger.info("Using wavelength range 5075-5125 Å for SNR calculation")
    else:
        # Fallback to full spectrum if this range is not available
        signal = np.nanmedian(cube._spectra, axis=0)
        noise = np.nanmedian(np.sqrt(cube._log_variance), axis=0)
        logger.info("Using full spectrum for SNR calculation (preferred range not available)")
    
    # Preprocess signal and noise to avoid problems with very low values
    # Handle problematic pixels by setting minimum values
    min_threshold = 1.0
    
    # Find pixels with problematic values (very low or NaN)
    low_signal_mask = (signal < min_threshold) | ~np.isfinite(signal)
    low_noise_mask = (noise < min_threshold) | ~np.isfinite(noise)
    
    # Set minimum values for signal and noise consistently
    if np.any(low_signal_mask) or np.any(low_noise_mask):
        logger.warning(f"Found {np.sum(low_signal_mask)} pixels with signal < {min_threshold}")
        logger.warning(f"Found {np.sum(low_noise_mask)} pixels with noise < {min_threshold}")
        
        # Apply minimum value to signal
        signal[low_signal_mask] = min_threshold
        
        # Apply minimum value to noise
        noise[low_noise_mask] = min_threshold
        
        # For pixels where SNR < 1, set both signal and noise to the same value (=1)
        # This effectively sets SNR = 1 for these pixels
        low_snr_mask = (signal / noise) < 1.0
        if np.any(low_snr_mask):
            logger.warning(f"Setting both signal and noise to {min_threshold} for {np.sum(low_snr_mask)} pixels with SNR < 1")
            signal[low_snr_mask] = min_threshold
            noise[low_snr_mask] = min_threshold
    
    # Additional safety check - ensure no zeros or NaNs
    signal = np.nan_to_num(signal, nan=min_threshold)
    noise = np.nan_to_num(noise, nan=min_threshold)
    
    # Calculate per-pixel SNR values
    pixel_snr = signal / noise  # Element-wise division for each pixel
    valid_snr = pixel_snr  # All pixels should be valid now due to our preprocessing
    max_pixel_snr = np.nanmax(valid_snr)
    median_snr = np.nanmedian(valid_snr)
    
    logger.info(f"Maximum pixel SNR: {max_pixel_snr:.1f}, Median pixel SNR: {median_snr:.1f}")
    
    # Determine recommended SNR range with wider bounds as suggested
    min_recommended_snr = min(10, median_snr * 5)
    max_recommended_snr = max(max_pixel_snr * 0.5, median_snr * 100)
    
    # # Ensure min and max are in a reasonable range
    # min_recommended_snr = max(2, min_recommended_snr)  # At least 2
    # max_recommended_snr = max(15, max_recommended_snr)  # At least 15
    
    # If user-specified target_snr is outside the recommended range, adjust
    if target_snr < min_recommended_snr or target_snr > max_recommended_snr:
        logger.warning(f"Specified target SNR {target_snr} is outside recommended range " 
                     f"({min_recommended_snr:.1f} - {max_recommended_snr:.1f})")
        
        # Adjust target_snr to be within range
        safe_target_snr = max(min_recommended_snr, min(target_snr, max_recommended_snr))
        logger.info(f"Adjusting target SNR to {safe_target_snr:.1f}")
    else:
        safe_target_snr = target_snr
    
    logger.info(f"Running Voronoi binning with target SNR = {safe_target_snr:.1f}")
    
    # First attempt with the selected target SNR
    try:
        success = True
        
        # Use all valid pixels
        x_valid = x
        y_valid = y
        signal_valid = signal
        noise_valid = noise
        
        logger.info(f"Running Voronoi binning with {len(x_valid)} valid pixels")
        
        # Handle return values more robustly to accommodate different version of vorbin
        result = voronoi_2d_binning(
            x_valid, y_valid, 
            signal_valid, noise_valid, 
            safe_target_snr, plot=0, quiet=True, cvt=use_cvt)
        
        # Robust handling of return values
        if isinstance(result, tuple):
            # Check length of returned tuple
            if len(result) >= 6:
                # Extract the first 6 values we need
                bin_num = result[0]
                x_gen = result[1]
                y_gen = result[2]
                sn = result[3]
                n_pixels = result[4]
                scale = result[5]
                best_result = (bin_num, x_gen, y_gen, sn, n_pixels, scale)
                logger.info(f"Voronoi binning succeeded with target SNR = {safe_target_snr:.1f}")
            else:
                # Not enough values
                raise ValueError(f"Voronoi binning returned {len(result)} values, expected at least 6")
        else:
            # Single return value (unlikely but handle anyway)
            raise ValueError("Unexpected return format from Voronoi binning")
            
    except Exception as e:
        success = False
        best_result = None
        logger.warning(f"Initial Voronoi binning failed: {str(e)}")
        logger.info("Trying alternative binning approach")
        
        # Try a more comprehensive search through recommended SNR values
        search_range = np.linspace(min_recommended_snr, max_recommended_snr, 10)
        
        for snr_value in search_range:
            # Skip the original value which already failed
            if abs(snr_value - safe_target_snr) < 0.1:
                continue
            
            # Try with CVT first
            try:
                logger.info(f"Trying Voronoi binning with target SNR = {snr_value:.1f}, CVT=True")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = voronoi_2d_binning(
                        x_valid, y_valid, 
                        signal_valid, noise_valid, 
                        snr_value, plot=0, quiet=True, cvt=True)
                    
                    # Extract the first 6 values
                    if isinstance(result, tuple) and len(result) >= 6:
                        bin_num = result[0]
                        x_gen = result[1]
                        y_gen = result[2]
                        sn = result[3]
                        n_pixels = result[4]
                        scale = result[5]
                        best_result = (bin_num, x_gen, y_gen, sn, n_pixels, scale)
                        success = True
                        logger.info(f"Voronoi binning succeeded with target SNR = {snr_value:.1f}")
                        break
                    else:
                        raise ValueError("Unexpected return format from Voronoi binning")
            except Exception as e:
                # Now try without CVT
                try:
                    logger.info(f"Trying Voronoi binning with target SNR = {snr_value:.1f}, CVT=False")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = voronoi_2d_binning(
                            x_valid, y_valid, 
                            signal_valid, noise_valid, 
                            snr_value, plot=0, quiet=True, cvt=False)
                        
                        # Extract the first 6 values
                        if isinstance(result, tuple) and len(result) >= 6:
                            bin_num = result[0]
                            x_gen = result[1]
                            y_gen = result[2]
                            sn = result[3]
                            n_pixels = result[4]
                            scale = result[5]
                            best_result = (bin_num, x_gen, y_gen, sn, n_pixels, scale)
                            success = True
                            logger.info(f"Voronoi binning succeeded with target SNR = {snr_value:.1f}")
                            break
                        else:
                            raise ValueError("Unexpected return format from Voronoi binning")
                except Exception as e:
                    continue
    
    # Fallback to grid binning if all else fails
    if not success or best_result is None:
        logger.warning("All Voronoi binning attempts failed. Using fallback grid binning.")
        
        # Create a simple grid binning as fallback
        from binning import create_grid_binning
        
        grid_result = create_grid_binning(
            x, y, signal, noise, 
            nx=min(5, int(np.sqrt(len(x) / 20))),  # Adjust grid size based on data
            ny=min(5, int(np.sqrt(len(x) / 20)))
        )
        
        if grid_result is not None:
            best_result = grid_result
            success = True
            logger.info(f"Created grid binning with {len(best_result[1])} bins")
        else:
            logger.error("Failed to create bins. Cannot proceed with analysis.")
            return {'status': 'error', 'message': 'Failed to create bins'}
    
    # Extract binning result
    bin_num, x_gen, y_gen, sn, n_pixels, scale = best_result
    
    # Map bin numbers back to the original arrays
    full_bin_num = bin_num
    
    # Filter out low SNR bins if requested, but don't filter ALL bins
    if min_snr > 0:
        # Find bins with SNR below minimum
        bad_bins = np.where(sn < min_snr)[0]
        
        # Don't filter all bins
        if len(bad_bins) < len(sn):  # Only if some bins remain
            # Mark these bins as unbinned (-1)
            for bad_bin in bad_bins:
                full_bin_num[full_bin_num == bad_bin] = -1
            
            logger.info(f"Filtered out {len(bad_bins)} bins with SNR < {min_snr}")
        else:
            logger.warning(f"All bins have SNR < {min_snr}, but keeping them to avoid empty result")
    
    # Create bin indices
    bin_indices = []
    all_indices = np.arange(len(full_bin_num))
    
    # Maximum bin number
    max_bin = int(np.nanmax(full_bin_num))
    
    for i in range(max_bin + 1):
        bin_indices.append(all_indices[full_bin_num == i])
    
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
    spectra = cube._spectra[wave_mask]
    
    # Combine spectra into bins with velocity correction
    binned_spectra = combine_spectra_efficiently(
        spectra, wavelength, bin_indices, velocity_field, cube._n_x
    )
    
    # Create metadata
    metadata = {
        'nx': cube._n_x,
        'ny': cube._n_y,
        'target_snr': target_snr,
        'sn': sn,
        'n_pixels': n_pixels,
        'scale': scale,
        'time': time.time(),
        'galaxy_name': galaxy_name,
        'analysis_type': 'VNB',
        'pixelsize_x': cube._pxl_size_x,
        'pixelsize_y': cube._pxl_size_y,
        'redshift': cube._redshift if hasattr(cube, '_redshift') else 0.0
    }
    
    # Create VoronoiBinnedData object
    binned_data = VoronoiBinnedData(
        bin_num=full_bin_num,
        bin_indices=bin_indices,
        spectra=binned_spectra,
        wavelength=wavelength,
        metadata=metadata
    )
    
    # Set up binning in the cube
    cube.setup_binning('VNB', binned_data)
    
    # Run analysis using the enhanced MUSECube methods
    # These will automatically use the binned data
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
            ppxf_vel_init=velocity_field,  # Use stellar velocity field as initial guess
            ppxf_sig_init=args.sigma_init,
            ppxf_deg=2,
            n_jobs=args.n_jobs
        )
    
    # Calculate spectral indices if requested
    indices_result = None
    if not args.no_indices:
        indices_result = cube.calculate_spectral_indices(
            n_jobs=args.n_jobs
        )
    
    # In run_vnb_analysis/run_rdb_analysis:


    
    # Calculate bin distances
    bin_distances = np.zeros(len(bin_indices))
    for i in range(len(bin_indices)):
        # Get pixel indices for this bin
        pixel_indices = bin_indices[i]
        
        # Skip empty bins
        if len(pixel_indices) == 0:
            bin_distances[i] = np.nan
            continue
        
        # Calculate average distance for this bin
        distances = []
        for idx in pixel_indices:
            row = idx // cube._n_x
            col = idx % cube._n_x
            
            # Calculate distance from center
            center_y, center_x = cube._n_y // 2, cube._n_x // 2
            distance = np.sqrt(((row - center_y) * cube._pxl_size_y)**2 + 
                             ((col - center_x) * cube._pxl_size_x)**2)
            distances.append(distance)
        
        # Set average distance
        bin_distances[i] = np.mean(distances)
    
    # Prepare standardized output dictionary
    vnb_results = {
        'analysis_type': 'VNB',
        'stellar_kinematics': {
            'velocity': cube._bin_velocity,
            'dispersion': cube._bin_dispersion,
            'velocity_field': velocity_field,
            'dispersion_field': dispersion_field
        },
        'distance': {
            'bin_distances': bin_distances,
            'pixelsize_x': cube._pxl_size_x,
            'pixelsize_y': cube._pxl_size_y
        },
        'binning': {
            'bin_num': full_bin_num,
            'bin_indices': bin_indices,
            'bin_x': x_gen,
            'bin_y': y_gen,
            'n_pixels': n_pixels,
            'snr': sn,
            'target_snr': target_snr
        }
    }
    
    # Add emission line results if available
    if emission_result is not None:
        vnb_results['emission'] = {}
        
        # Copy emission line fields from cube if available
        if hasattr(cube, '_bin_emission_flux'):
            vnb_results['emission']['flux'] = cube._bin_emission_flux
        if hasattr(cube, '_bin_emission_vel'):
            vnb_results['emission']['velocity'] = cube._bin_emission_vel
        if hasattr(cube, '_bin_emission_sig'):
            vnb_results['emission']['dispersion'] = cube._bin_emission_sig
            
        # Copy emission fields from emission_result
        for key in ['emission_flux', 'emission_vel', 'emission_sig']:
            if key in emission_result:
                field_name = key.split('_')[1]  # extract 'flux', 'vel', 'sig'
                vnb_results['emission'][field_name] = emission_result[key]
                
        # Add emission line wavelengths if available
        if 'emission_wavelength' in emission_result:
            vnb_results['emission']['wavelengths'] = emission_result['emission_wavelength']
    
        # After calculating spectral indices
    if indices_result is not None:
        vnb_results['indices'] = indices_result  # This will be pixel-based
        
        # Add bin-level indices if available
        if hasattr(cube, '_bin_indices_result') and cube._bin_indices_result:
            vnb_results['bin_indices'] = cube._bin_indices_result

    # Save binned data object
    binned_data_path = data_dir / f"{galaxy_name}_VNB_binned_data.npz"
    binned_data.save(binned_data_path)
    logger.info(f"Saved binned data to {binned_data_path}")
    
    # Save results
    save_standardized_results(galaxy_name, 'VNB', vnb_results, output_dir)
    
    # Create visualization plots if requested
    if not hasattr(args, 'no_plots') or not args.no_plots:
        binned_data.create_visualization_plots(plots_dir, galaxy_name)
        create_vnb_plots(cube, vnb_results, galaxy_name, plots_dir, args)

    
    # Add these lines near the end of the run_vnb_analysis function in voronoi.py
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
    
    logger.info(f"VNB analysis completed in {time.time() - start_time:.1f} seconds")
    
    return vnb_results

def create_vnb_plots(cube, vnb_results, galaxy_name, plots_dir, args):
    """
    Create visualization plots for VNB analysis
    
    Parameters
    ----------
    cube : MUSECube
        MUSE cube with binned data
    vnb_results : dict
        VNB analysis results
    galaxy_name : str
        Galaxy name
    plots_dir : Path
        Directory to save plots
    args : argparse.Namespace
        Command line arguments
    """
    try:
        import visualization
        
        # Create kinematics plot
        if 'stellar_kinematics' in vnb_results:
            try:
                bin_num = vnb_results['binning']['bin_num']
                velocity = vnb_results['stellar_kinematics']['velocity']
                dispersion = vnb_results['stellar_kinematics']['dispersion']
                
                # Create velocity map
                fig, ax = plt.subplots(figsize=(10, 8))
                # Reshape bin_num to 2D if it's 1D
                bin_num_2d = bin_num
                if bin_num.ndim == 1:
                    # Reshape bin_num to 2D grid matching the original image dimensions
                    bin_num_2d = np.full((cube._n_y, cube._n_x), -1)
                    for i, bin_id in enumerate(bin_num):
                        if i < cube._n_y * cube._n_x:
                            row = i // cube._n_x
                            col = i % cube._n_x
                            bin_num_2d[row, col] = bin_id
                            
                visualization.plot_bin_map(
                    bin_num_2d, 
                    velocity, 
                    ax=ax, 
                    cmap='coolwarm',
                    title=f'{galaxy_name} - VNB Velocity',
                    vmin=np.percentile(velocity[np.isfinite(velocity)], 5),
                    vmax=np.percentile(velocity[np.isfinite(velocity)], 95),
                    colorbar_label='Velocity (km/s)'
                )
                plt.savefig(plots_dir / f"{galaxy_name}_VNB_velocity.png", dpi=150)
                plt.close(fig)
                
                # Create dispersion map
                fig, ax = plt.subplots(figsize=(10, 8))
                visualization.plot_bin_map(
                    bin_num_2d, 
                    dispersion, 
                    ax=ax, 
                    cmap='viridis',
                    title=f'{galaxy_name} - VNB Dispersion',
                    vmin=np.percentile(dispersion[np.isfinite(dispersion)], 5),
                    vmax=np.percentile(dispersion[np.isfinite(dispersion)], 95),
                    colorbar_label='Dispersion (km/s)'
                )
                plt.savefig(plots_dir / f"{galaxy_name}_VNB_dispersion.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Error creating kinematics plots: {e}")
                plt.close('all')
        
        # Create emission line plots if available
        if 'emission' in vnb_results:
            try:
                bin_num = vnb_results['binning']['bin_num']
                emission = vnb_results['emission']
                
                # Find emission flux maps
                for line_name, flux in emission.get('flux', {}).items():
                    if isinstance(flux, np.ndarray) and len(flux) > 0:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        visualization.plot_bin_map(
                            bin_num, 
                            flux, 
                            ax=ax, 
                            cmap='inferno',
                            title=f'{galaxy_name} - VNB {line_name} Flux',
                            log_scale=True,
                            colorbar_label='Log Flux'
                        )
                        plt.savefig(plots_dir / f"{galaxy_name}_VNB_{line_name}_flux.png", dpi=150)
                        plt.close(fig)
            except Exception as e:
                logger.warning(f"Error creating emission line plots: {e}")
                plt.close('all')
        
        # Create spectral indices plots if available
        if 'indices' in vnb_results:
            try:
                bin_num = vnb_results['binning']['bin_num']
                indices = vnb_results['indices']
                
                for idx_name, idx_values in indices.items():
                    if isinstance(idx_values, np.ndarray) and idx_values.shape == bin_num.shape:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        # Reshape bin_num to 2D if it's 1D
                        bin_num_2d = bin_num
                        if bin_num.ndim == 1:
                            # Reshape bin_num to 2D grid matching the original image dimensions
                            bin_num_2d = np.full((cube._n_y, cube._n_x), -1)
                            for i, bin_id in enumerate(bin_num):
                                if i < cube._n_y * cube._n_x:
                                    row = i // cube._n_x
                                    col = i % cube._n_x
                                    bin_num_2d[row, col] = bin_id
                                    
                        # Use safe_plot_array for robust plotting
                        visualization.safe_plot_array(
                            idx_values, 
                            bin_num_2d, 
                            ax=ax, 
                            title=f'{galaxy_name} - {idx_name}',
                            cmap='plasma',
                            label='Index Value'
                        )
                        plt.savefig(plots_dir / f"{galaxy_name}_VNB_{idx_name}.png", dpi=150)
                        plt.close(fig)
                    elif 'bin_indices' in vnb_results and idx_name in vnb_results['bin_indices']:
                        # Try bin-level indices
                        bin_values = vnb_results['bin_indices'][idx_name]
                        fig, ax = plt.subplots(figsize=(10, 8))
                        visualization.plot_bin_map(
                            bin_num, 
                            bin_values, 
                            ax=ax, 
                            cmap='plasma',
                            title=f'{galaxy_name} - VNB {idx_name}',
                            colorbar_label='Index Value'
                        )
                        plt.savefig(plots_dir / f"{galaxy_name}_VNB_{idx_name}.png", dpi=150)
                        plt.close(fig)
            except Exception as e:
                logger.warning(f"Error creating spectral indices plots: {e}")
                plt.close('all')
        

        # Add spectral indices visualization
        if 'indices' in vnb_results and 'bin_indices' in vnb_results:  # For voronoi.py
            try:
                bin_indices = vnb_results['bin_indices']
                if bin_indices and isinstance(bin_indices, dict):
                    for idx_name, idx_values in bin_indices.items():
                        if isinstance(idx_values, np.ndarray) and len(idx_values) > 0:
                            # Create 2D index map
                            fig, ax = plt.subplots(figsize=(10, 8))
                            
                            # Use safe_plot_array for robust plotting
                            visualization.safe_plot_array(
                                idx_values, 
                                bin_num, 
                                ax=ax, 
                                title=f'{galaxy_name} - {idx_name}',
                                cmap='plasma',
                                label='Index Value'
                            )
                            
                            plt.savefig(plots_dir / f"{galaxy_name}_VNB_{idx_name}.png", dpi=150)  # For voronoi.py
                            # Or for radial.py:
                            # plt.savefig(plots_dir / f"{galaxy_name}_RDB_{idx_name}.png", dpi=150)
                            plt.close(fig)
            except Exception as e:
                logger.warning(f"Error creating spectral indices maps: {e}")
                plt.close('all')
    
    except Exception as e:
        logger.error(f"Error in create_vnb_plots: {str(e)}")
        logger.error(traceback.format_exc())
        plt.close('all')