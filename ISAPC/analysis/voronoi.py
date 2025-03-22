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


from p2p_adapter import create_p2p_processor, BinnedDataAdapter, extract_bin_results
from analysis.p2p import run_p2p_analysis

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
        from p2p_adapter import load_p2p_results_for_galaxy
        p2p_results = load_p2p_results_for_galaxy(galaxy_name, args.output_dir)
        
        if p2p_results is not None:
            logger.info("Successfully loaded P2P results for VNB analysis")
    
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
        # Calculate SNR using this specific range (as done in the notebook)
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
    
    # Replace zeros with min_threshold
    signal[signal == 0] = min_threshold
    noise[noise == 0] = min_threshold
    
    # Apply minimum SNR threshold to avoid problems with very low SNR spaxels
    valid_mask = (signal / noise) >= min_snr
    if np.sum(valid_mask) < 10:
        # If too few valid spaxels, lower threshold
        logger.warning(f"Very few spaxels meet minimum SNR threshold ({np.sum(valid_mask)}). Lowering minimum SNR.")
        min_snr = max(0.5, min_snr / 2)
        valid_mask = (signal / noise) >= min_snr
        
        # Check if we still have too few pixels
        if np.sum(valid_mask) < 5:
            logger.warning("Still too few valid pixels after lowering threshold. Selecting pixels with highest SNR.")
            # Force selection of the top pixels with highest SNR
            snr_values = signal / noise
            # Get indices sorted by SNR (highest first)
            sorted_indices = np.argsort(snr_values)[::-1]
            # Create a new mask that selects at least N pixels with highest SNR
            min_pixels = min(100, max(10, int(0.01 * signal.size)))  # At least 1% of pixels or 10 pixels, whichever is more
            force_indices = sorted_indices[:min_pixels]
            new_mask = np.zeros_like(valid_mask)
            new_mask[force_indices] = True
            valid_mask = new_mask
            
            # If we still have no valid pixels (extremely rare case), use all pixels
            if np.sum(valid_mask) == 0:
                logger.warning("No valid pixels even after selection. Using all non-NaN pixels as fallback.")
                valid_mask = np.isfinite(signal) & np.isfinite(noise)
                # If still no valid pixels, this is truly bad data, but we'll try anyway
                if np.sum(valid_mask) == 0:
                    logger.warning("No valid pixels at all. Using all pixels regardless of quality.")
                    valid_mask = np.ones_like(signal, dtype=bool)
    
    logger.info(f"Found {np.sum(valid_mask)} spaxels above minimum SNR threshold {min_snr}")
    
    # If we have P2P results, try to use them to improve SNR
    if p2p_results is not None:
        try:
            # Use SNR from P2P results if available (like in the notebook)
            if 'signal_noise' in p2p_results:
                p2p_signal = p2p_results['signal_noise'].get('signal', None)
                p2p_noise = p2p_results['signal_noise'].get('noise', None)
                
                if p2p_signal is not None and p2p_noise is not None:
                    logger.info("Using signal and noise from P2P results")
                    
                    # Apply the same minimum threshold logic to P2P results
                    p2p_signal = np.nan_to_num(p2p_signal, nan=min_threshold)
                    p2p_noise = np.nan_to_num(p2p_noise, nan=min_threshold)
                    
                    p2p_signal[p2p_signal < min_threshold] = min_threshold
                    p2p_noise[p2p_noise < min_threshold] = min_threshold
                    
                    # Set SNR=1 for pixels with SNR < 1
                    low_snr_mask = (p2p_signal / p2p_noise) < 1.0
                    if np.any(low_snr_mask):
                        p2p_signal[low_snr_mask] = min_threshold
                        p2p_noise[low_snr_mask] = min_threshold
                    
                    signal = p2p_signal
                    noise = p2p_noise
                    
                    # Recalculate valid mask
                    valid_mask = (signal / noise) >= min_snr
            
            # Use only good spaxels based on P2P results if available
            if 'quality_mask' in p2p_results:
                good_mask = p2p_results['quality_mask']
                if good_mask is not None and good_mask.size == valid_mask.size:
                    valid_mask = valid_mask & good_mask
                    logger.info(f"Applied quality mask from P2P results, now have {np.sum(valid_mask)} valid spaxels")
        except Exception as e:
            logger.warning(f"Error applying P2P SNR data: {e}")
    
    # Get velocity field from P2P results if available
    velocity_field = None
    if p2p_results is not None:
        try:
            # First try the standardized format
            if 'stellar_kinematics' in p2p_results and 'velocity_field' in p2p_results['stellar_kinematics']:
                velocity_field = p2p_results['stellar_kinematics']['velocity_field']
                logger.info("Using velocity field from P2P results (standardized format)")
            # Then try the direct format
            elif 'velocity_field' in p2p_results:
                velocity_field = p2p_results['velocity_field']
                logger.info("Using velocity field from P2P results (direct format)")
            
            # Check if the velocity field is valid
            if velocity_field is not None and np.all(np.isnan(velocity_field)):
                logger.warning("Velocity field from P2P results contains only NaNs")
                velocity_field = None
        except Exception as e:
            logger.warning(f"Error extracting velocity field from P2P results: {e}")
            velocity_field = None
    
    try:
        # Determine per-pixel SNR values
        pixel_snr = signal / np.maximum(noise, 1e-10)  # Element-wise division for each pixel
        valid_snr = pixel_snr[valid_mask]
        max_pixel_snr = np.nanmax(valid_snr)
        median_snr = np.nanmedian(valid_snr)
        
        logger.info(f"Maximum pixel SNR: {max_pixel_snr:.1f}, Median pixel SNR: {median_snr:.1f}")
        
        # Determine recommended SNR range with wider bounds as suggested
        min_recommended_snr = min(10, median_snr * 5)
        max_recommended_snr = max(max_pixel_snr * 0.5, median_snr * 100)
        
        # Ensure min and max are in a reasonable range
        min_recommended_snr = max(2, min_recommended_snr)  # At least 2
        max_recommended_snr = max(15, max_recommended_snr)  # At least 15
        
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
            # Use the valid_mask to filter data
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            signal_valid = signal[valid_mask]
            noise_valid = noise[valid_mask]
            
            # Check if we have enough valid pixels
            if len(x_valid) == 0:
                raise ValueError("No valid pixels available for Voronoi binning")
            
            # Make sure arrays have consistent lengths
            min_len = min(len(x_valid), len(y_valid), len(signal_valid), len(noise_valid))
            if min_len == 0:
                raise ValueError("Empty arrays after filtering")
            
            x_valid = x_valid[:min_len]
            y_valid = y_valid[:min_len]
            signal_valid = signal_valid[:min_len]
            noise_valid = noise_valid[:min_len]
            
            # Double-check for any problematic values
            for i in range(len(signal_valid)):
                if signal_valid[i] < min_threshold or noise_valid[i] < min_threshold or (signal_valid[i]/noise_valid[i]) < 1.0:
                    signal_valid[i] = min_threshold
                    noise_valid[i] = min_threshold
            
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
        
        # If initial attempt failed, try a systematic search through recommended SNR values
        if not success:
            # Try a more comprehensive search through the recommended range
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
                                logger.info(f"Voronoi binning succeeded with target SNR = {snr_value:.1f} (CVT=False)")
                                break
                            else:
                                raise ValueError("Unexpected return format from Voronoi binning")
                    except Exception as e2:
                        continue
        
        # If still no success, try with a few fixed values known to often work
        if not success:
            for snr_fixed in [10, 15, 20, 25, 30, 40]:
                if abs(snr_fixed - safe_target_snr) < 0.1:
                    continue
                
                try:
                    logger.info(f"Trying with fixed target SNR = {snr_fixed}")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Try both with and without CVT
                        for use_cvt_val in [True, False]:
                            try:
                                result = voronoi_2d_binning(
                                    x_valid, y_valid, 
                                    signal_valid, noise_valid, 
                                    snr_fixed, plot=0, quiet=True, cvt=use_cvt_val)
                                
                                # Extract values
                                if isinstance(result, tuple) and len(result) >= 6:
                                    bin_num = result[0]
                                    x_gen = result[1]
                                    y_gen = result[2]
                                    sn = result[3]
                                    n_pixels = result[4]
                                    scale = result[5]
                                    best_result = (bin_num, x_gen, y_gen, sn, n_pixels, scale)
                                    success = True
                                    logger.info(f"Voronoi binning succeeded with fixed target SNR = {snr_fixed} (CVT={use_cvt_val})")
                                    break
                                else:
                                    continue
                            except:
                                continue
                        if success:
                            break
                except Exception as e:
                    continue
        
        # If all attempts failed, use a grid binning approach as fallback
        if not success or best_result is None:
            logger.warning("All Voronoi binning attempts failed, using grid binning as fallback")
            
            # Determine a reasonable number of bins based on data size and SNR
            valid_count = np.sum(valid_mask)
            
            # More sophisticated bin count calculation based on data size and SNR
            if max_pixel_snr > 0:
                # More bins for higher SNR data
                bin_factor = min(5, max(1, max_pixel_snr / 10))
            else:
                bin_factor = 1
            
            num_bins = min(100, max(4, int(np.sqrt(valid_count / bin_factor))))
            
            # Determine grid dimensions - try to match aspect ratio of data
            xrange = np.max(x_valid) - np.min(x_valid)
            yrange = np.max(y_valid) - np.min(y_valid)
            
            aspect = xrange / max(yrange, 1e-10)  # Avoid division by zero
            
            if aspect > 1.5:  # Wider than tall
                nx = int(np.sqrt(num_bins * aspect))
                ny = int(num_bins / nx)
            elif aspect < 0.67:  # Taller than wide
                ny = int(np.sqrt(num_bins / aspect))
                nx = int(num_bins / ny)
            else:  # Roughly square
                nx = ny = int(np.sqrt(num_bins))
            
            # Ensure at least 2x2 grid
            nx = max(2, nx)
            ny = max(2, ny)
            
            logger.info(f"Creating {nx}×{ny} grid binning with approximately {nx*ny} bins")
            
            # Get bounds of coordinates
            xmin, xmax = np.min(x_valid), np.max(x_valid)
            ymin, ymax = np.min(y_valid), np.max(y_valid)
            
            # Create grid binning
            bin_result = create_grid_binning(
                x_valid, y_valid,
                signal_valid, noise_valid,
                nx=nx, ny=ny, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
            )
            
            if bin_result is not None:
                # Map bin numbers back to the original arrays
                bin_num_valid, x_gen, y_gen, sn_values, n_pixels, scale = bin_result
                
                bin_num_full = np.full_like(x, -1, dtype=int)
                bin_num_full[valid_mask] = bin_num_valid
                
                best_result = (bin_num_valid, x_gen, y_gen, sn_values, n_pixels, scale)
                success = True
                logger.info(f"Created {len(x_gen)} grid bins as fallback")
            else:
                # Last resort: create a simple single bin
                logger.warning("Grid binning failed, creating simple radial bins as last resort")
                
                # Create simple radial bins - often more useful than a uniform grid
                center_x = np.median(x_valid)
                center_y = np.median(y_valid)
                
                # Calculate distance from center
                r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Create simple radial bins (4 is a reasonable minimum)
                n_radial_bins = min(8, max(4, int(np.sqrt(valid_count / 10))))
                
                # Get maximum radius with margin
                r_max = np.nanmax(r[valid_mask]) * 1.01
                
                # Create bin edges
                bin_edges = np.linspace(0, r_max, n_radial_bins + 1)
                
                # Assign bins
                bin_num_full = np.full_like(r, -1, dtype=int)
                
                for i in range(n_radial_bins):
                    r_min = bin_edges[i]
                    r_max_bin = bin_edges[i + 1]
                    mask = valid_mask & (r >= r_min) & (r < r_max_bin)
                    bin_num_full[mask] = i
                
                # Create generator coordinates, SNR values, and pixel counts
                x_gen = []
                y_gen = []
                sn_values = []
                n_pixels = []
                
                for i in range(n_radial_bins):
                    mask = bin_num_full == i
                    if np.any(mask):
                        x_gen.append(np.mean(x[mask]))
                        y_gen.append(np.mean(y[mask]))
                        n_pixels.append(np.sum(mask))
                        
                        # Estimate SNR
                        this_signal = np.mean(signal[mask])
                        this_noise = np.mean(noise[mask])
                        # Ensure minimum SNR of 1
                        snr = max(1.0, this_signal / this_noise if this_noise > 0 else 1.0)
                        sn_values.append(snr)
                
                # Convert to arrays
                x_gen = np.array(x_gen)
                y_gen = np.array(y_gen)
                sn_values = np.array(sn_values)
                n_pixels = np.array(n_pixels)
                
                # Check if we have any valid bins
                if len(x_gen) > 0:
                    best_result = (bin_num_full[valid_mask], x_gen, y_gen, sn_values, n_pixels, 1.0)
                    success = True
                    logger.info(f"Created {len(x_gen)} radial bins as fallback")
                else:
                    # If all else fails, use a simple single bin
                    bin_num_full = np.zeros_like(x, dtype=int)
                    bin_num_full[~valid_mask] = -1  # Mark invalid points as -1
                    x_gen = np.array([np.mean(x_valid)])
                    y_gen = np.array([np.mean(y_valid)])
                    sn_values = np.array([max(median_snr, 5.0)])
                    n_pixels = np.array([np.sum(valid_mask)])
                    
                    best_result = (bin_num_full[valid_mask], x_gen, y_gen, sn_values, n_pixels, 1.0)
                    success = True
                    logger.warning("Created a single bin as last resort fallback")
        
        # Extract the results from the successful binning operation
        bin_num, x_gen, y_gen, sn, n_pixels, scale = best_result
        
        # Convert bin_num back to full size
        full_bin_num = np.full(signal.shape, -1, dtype=int)
        full_bin_num[valid_mask] = bin_num
        
        # Get bin indices for each bin
        bin_indices = []
        valid_bins = np.unique(bin_num[bin_num >= 0])
        
        for i in valid_bins:
            indices = np.where(full_bin_num == i)[0]
            bin_indices.append(indices)
        
        logger.info(f"Created {len(bin_indices)} Voronoi bins")
        
        # Calculate intersection of wavelength ranges accounting for velocity shifts
        if velocity_field is not None:
            wave_mask, min_wave, max_wave = calculate_wavelength_intersection(
                cube._lambda_gal, velocity_field, cube._n_x
            )
            logger.info(f"Applying velocity correction with range {min_wave:.1f} - {max_wave:.1f} Å")
        else:
            wave_mask = np.ones_like(cube._lambda_gal, dtype=bool)
            logger.info("No velocity correction applied")
        
        # Apply wavelength mask
        wavelength = cube._lambda_gal[wave_mask]
        
        # Combine spectra in each bin with enhanced velocity correction
        # This approach similar to the notebook's Spectrum_ReSMP function
        binned_spectra = combine_spectra_with_velocity_correction(
            cube._spectra[wave_mask], wavelength, bin_indices, velocity_field, cube._n_x, cube._n_y
        )
        
        logger.info(f"Combined spectra into {len(bin_indices)} bins")
        
        # Create metadata dictionary
        metadata = {
            'nx': cube._n_x,
            'ny': cube._n_y,
            'target_snr': target_snr,
            'actual_target_snr': safe_target_snr,
            'min_snr': min_snr,
            'time': time.time(),
            'galaxy_name': galaxy_name,
            'sn': sn,
            'n_pixels': n_pixels,
            'scale': scale,
            'x_gen': x_gen,
            'y_gen': y_gen,
            'analysis_type': 'VNB',
            'pixelsize_x': cube._pxl_size_x,
            'pixelsize_y': cube._pxl_size_y
        }
        
        # Create VoronoiBinnedData object
        binned_data = VoronoiBinnedData(
            bin_num=full_bin_num,
            bin_indices=bin_indices,
            spectra=binned_spectra,
            wavelength=wavelength,
            metadata=metadata
        )
        
        # Run analysis on binned spectra (template fitting, etc.)
        vnb_results = run_analysis_on_binned_data(args, binned_data, cube, p2p_results)
        
        # Create visualization plots
        if not args.no_plots:
            create_vnb_plots(args, binned_data, vnb_results, galaxy_name, plots_dir)
        
        # Prepare output dictionary with all results
        result_dict = {
            'analysis_type': 'VNB',
            'bin_num': full_bin_num,
            'bin_indices': bin_indices,
            'bin_coordinates': {
                'x': x_gen,
                'y': y_gen
            },
            'bin_statistics': {
                'sn': sn,
                'n_pixels': n_pixels,
                'scale': scale
            },
            'parameters': {
                'target_snr': target_snr,
                'actual_target_snr': safe_target_snr,
                'min_snr': min_snr,
                'cvt': use_cvt
            }
        }
        
        # Add analysis results
        result_dict.update(vnb_results)
        
        # Save results
        save_standardized_results(galaxy_name, 'VNB', result_dict, output_dir)
        
        logger.info(f"VNB analysis completed in {time.time() - start_time:.1f} seconds")
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error in VNB analysis: {str(e)}")
        logger.error(traceback.format_exc())
        # Return empty results dictionary
        return {
            'analysis_type': 'VNB',
            'status': 'error',
            'error': str(e)
        }

def combine_spectra_with_velocity_correction(spectra, wavelength, bin_indices, velocity_field=None, n_x=None, n_y=None):
    """
    Combine spectra within bins with improved velocity correction.
    This implementation is inspired by the Spectrum_ReSMP function in the notebook.
    
    Parameters
    ----------
    spectra : numpy.ndarray
        Array of spectra [n_wave, n_spectra]
    wavelength : numpy.ndarray
        Wavelength array
    bin_indices : list
        List of arrays with indices for each bin
    velocity_field : numpy.ndarray, optional
        Velocity field for correction
    n_x : int, optional
        Number of pixels in x direction
    n_y : int, optional
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
    
    # Check if velocity correction is requested and available
    do_correction = velocity_field is not None and n_x is not None
    
    # Process each bin
    for i, indices in enumerate(bin_indices):
        # Skip empty bins
        if len(indices) == 0:
            bin_spectra[:, i] = np.nan
            continue
        
        try:
            # Extract velocities for this bin if available
            if do_correction:
                # Calculate median velocity for this bin
                bin_velocities = []
                for idx in indices:
                    row = idx // n_x
                    col = idx % n_x
                    if row < n_y and col < n_x:
                        if velocity_field is not None and row < velocity_field.shape[0] and col < velocity_field.shape[1]:
                            vel = velocity_field[row, col]
                            if np.isfinite(vel):
                                bin_velocities.append(vel)
                
                # Apply velocity correction if we have valid velocities
                if bin_velocities:
                    median_velocity = np.median(bin_velocities)
                    
                    # Only apply correction for non-negligible velocities
                    if abs(median_velocity) > 1.0:  # Minimum 1 km/s to apply correction
                        # Following the notebook approach:
                        # 1. Collect spectra for each spaxel
                        corrected_spectra = []
                        
                        for idx in indices:
                            spec = spectra[:, idx]
                            if not np.all(~np.isfinite(spec)):
                                # Get velocity for this spaxel
                                row = idx // n_x
                                col = idx % n_x
                                
                                if row < velocity_field.shape[0] and col < velocity_field.shape[1]:
                                    vel = velocity_field[row, col]
                                    
                                    # Check for outlier velocities within the bin
                                    vel_limit = 300  # km/s, from notebook
                                    if abs(vel - median_velocity) > vel_limit:
                                        vel = median_velocity
                                    
                                    if abs(vel) > 300:  # Skip extreme velocities
                                        vel = 0
                                    
                                    # Apply velocity shift to wavelength grid
                                    lam_shifted = wavelength / (1 + (vel/c))
                                    
                                    # Resample spectrum to original wavelength grid
                                    try:
                                        corrected_spec = spectres(wavelength, lam_shifted, spec)
                                        corrected_spectra.append(corrected_spec)
                                    except:
                                        # Fallback to original spectrum if resampling fails
                                        corrected_spectra.append(spec)
                                else:
                                    corrected_spectra.append(spec)
                        
                        # Combine corrected spectra
                        if corrected_spectra:
                            # Take median of all corrected spectra
                            bin_spectra[:, i] = np.nanmedian(np.array(corrected_spectra), axis=0)
                        else:
                            bin_spectra[:, i] = np.nan
                            
                        continue  # Skip remaining processing for this bin
            
            # If we get here, either no velocity correction was applied or it wasn't needed
            # Simply combine original spectra
            bin_data = []
            for idx in indices:
                spec = spectra[:, idx]
                if not np.all(~np.isfinite(spec)):
                    bin_data.append(spec)
            
            if bin_data:
                bin_spectra[:, i] = np.nanmedian(np.array(bin_data), axis=0)
            else:
                bin_spectra[:, i] = np.nan
                
        except Exception as e:
            logger.error(f"Error combining spectra for bin {i}: {e}")
            bin_spectra[:, i] = np.nan
    
    return bin_spectra

def create_grid_binning(x, y, signal, noise, nx=4, ny=4, xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Create a grid-based binning scheme as a fallback when Voronoi binning fails
    
    Parameters
    ----------
    x : numpy.ndarray
        X coordinates of pixels
    y : numpy.ndarray
        Y coordinates of pixels
    signal : numpy.ndarray
        Signal values for each pixel
    noise : numpy.ndarray
        Noise values for each pixel
    nx : int, default=4
        Number of bins in x direction
    ny : int, default=4
        Number of bins in y direction
    xmin, xmax, ymin, ymax : float, optional
        Bounds of the grid
        
    Returns
    -------
    tuple
        (bin_num, x_gen, y_gen, sn, n_pixels, scale)
    """
    try:
        # Set bounds if not provided
        if xmin is None:
            xmin = np.min(x)
        if xmax is None:
            xmax = np.max(x)
        if ymin is None:
            ymin = np.min(y)
        if ymax is None:
            ymax = np.max(y)
        
        # Create grid
        x_edges = np.linspace(xmin, xmax, nx + 1)
        y_edges = np.linspace(ymin, ymax, ny + 1)
        
        # Initialize bin numbers
        bin_num = np.full(x.shape, -1, dtype=int)
        
        # Assign bin numbers based on grid position
        bin_count = 0
        x_gen = []
        y_gen = []
        sn_values = []
        n_pixels_values = []
        
        for i in range(nx):
            for j in range(ny):
                # Define bin edges
                x_min, x_max = x_edges[i], x_edges[i + 1]
                y_min, y_max = y_edges[j], y_edges[j + 1]
                
                # Select pixels in this bin
                mask = ((x >= x_min) & (x < x_max) & 
                        (y >= y_min) & (y < y_max))
                
                if np.sum(mask) > 0:
                    bin_num[mask] = bin_count
                    x_gen.append(np.mean(x[mask]))
                    y_gen.append(np.mean(y[mask]))
                    
                    # Calculate SNR for this bin
                    this_signal = np.sum(signal[mask])
                    this_noise = np.sqrt(np.sum(noise[mask]**2))
                    # Ensure SNR is at least 1
                    this_snr = max(1.0, this_signal / this_noise if this_noise > 0 else 1.0)
                    
                    sn_values.append(this_snr)
                    n_pixels_values.append(np.sum(mask))
                    
                    bin_count += 1
        
        if bin_count == 0:
            logger.error("Grid binning failed: no bins created")
            return None
        
        # Convert to arrays
        x_gen = np.array(x_gen)
        y_gen = np.array(y_gen)
        sn_values = np.array(sn_values)
        n_pixels_values = np.array(n_pixels_values)
        
        # Create scale value (not really meaningful for grid binning)
        scale = 1.0
        
        return (bin_num, x_gen, y_gen, sn_values, n_pixels_values, scale)
    
    except Exception as e:
        logger.error(f"Error in grid binning: {e}")
        return None

def calculate_wavelength_intersection(wavelength, velocity_field, n_x):
    """
    Calculate common wavelength range accounting for velocity shifts.
    
    Parameters
    ----------
    wavelength : numpy.ndarray
        Original wavelength array
    velocity_field : numpy.ndarray
        Velocity field (2D array)
    n_x : int
        Number of pixels in x direction
        
    Returns
    -------
    tuple
        (mask, min_wave, max_wave)
    """
    c = 299792.458  # Speed of light in km/s
    
    # Find minimum and maximum velocities
    valid_velocities = velocity_field[np.isfinite(velocity_field)]
    if len(valid_velocities) == 0:
        # No valid velocities, return full range
        return np.ones_like(wavelength, dtype=bool), np.min(wavelength), np.max(wavelength)
    
    min_vel = np.min(valid_velocities)
    max_vel = np.max(valid_velocities)
    
    # Calculate wavelength limits
    # For redshifted spectra, the maximum wavelength becomes larger
    # For blueshifted spectra, the minimum wavelength becomes smaller
    min_factor = 1 + min_vel/c
    max_factor = 1 + max_vel/c
    
    # The observed range must be adjusted to account for all possible velocity shifts
    # This ensures that after shifting, all spectra cover the same rest-frame range
    rest_min = np.min(wavelength) / min(min_factor, max_factor)
    rest_max = np.max(wavelength) / max(min_factor, max_factor)
    
    # Get intersection range with some margin (1%)
    margin = 0.01 * (rest_max - rest_min)
    min_wave = rest_min + margin
    max_wave = rest_max - margin
    
    # Create mask for wavelength range
    mask = (wavelength >= min_wave) & (wavelength <= max_wave)
    
    # Ensure we have some valid wavelengths left
    if np.sum(mask) < 10:
        # If almost no wavelength points left, use most of the original range
        logger.warning("Velocity range too wide for wavelength intersection, using 80% of original range")
        wlen = len(wavelength)
        start_idx = int(wlen * 0.1)
        end_idx = int(wlen * 0.9)
        mask = np.zeros_like(wavelength, dtype=bool)
        mask[start_idx:end_idx] = True
        min_wave = wavelength[start_idx]
        max_wave = wavelength[end_idx-1]
    
    return mask, min_wave, max_wave

def run_analysis_on_binned_data(args, binned_data, cube, p2p_results=None):
    """
    Run additional analysis on binned data (stellar population, emission lines, etc.)
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    binned_data : VoronoiBinnedData
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
        
        # Create P2P processor
        p2p_processor = create_p2p_processor(run_p2p_analysis)
        
        # Run P2P analysis on binned data
        bin_p2p_results = p2p_processor(args, binned_data, p2p_results)
        
        # Extract bin results
        bin_adapter = BinnedDataAdapter(binned_data)
        results = extract_bin_results(bin_p2p_results, bin_adapter, result_type='vnb')
        
        # Format results for consistency with VNB output
        formatted_results = {
            'stellar_kinematics': {
                'velocity': bin_p2p_results.get('velocity_field', None),
                'dispersion': bin_p2p_results.get('dispersion_field', None)
            },
            'distance': {
                'bin_distances': np.sqrt(bin_adapter.x**2 + bin_adapter.y**2),
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
                    # Extract values at bin centers
                    bin_values = []
                    for i, (x, y) in enumerate(zip(bin_adapter.x, bin_adapter.y)):
                        # Find closest pixel
                        x_idx = min(max(int(x / cube._pxl_size_x + cube._n_x / 2), 0), cube._n_x - 1)
                        y_idx = min(max(int(y / cube._pxl_size_y + cube._n_y / 2), 0), cube._n_y - 1)
                        bin_values.append(value[y_idx, x_idx])
                    
                    formatted_results['emission'][key] = np.array(bin_values)
                else:
                    formatted_results['emission'][key] = value
        
        # Add spectral indices if available
        if 'indices' in bin_p2p_results:
            formatted_results['indices'] = {}
            
            # Process each index
            for index_name, index_map in bin_p2p_results['indices'].items():
                if isinstance(index_map, np.ndarray) and index_map.shape == (cube._n_y, cube._n_x):
                    # Extract values at bin centers
                    bin_values = []
                    for i, (x, y) in enumerate(zip(bin_adapter.x, bin_adapter.y)):
                        # Find closest pixel
                        x_idx = min(max(int(x / cube._pxl_size_x + cube._n_x / 2), 0), cube._n_x - 1)
                        y_idx = min(max(int(y / cube._pxl_size_y + cube._n_y / 2), 0), cube._n_y - 1)
                        bin_values.append(index_map[y_idx, x_idx])
                    
                    formatted_results['indices'][index_name] = np.array(bin_values)
                else:
                    formatted_results['indices'][index_name] = index_map
        
        # Add stellar population parameters if available
        if 'stellar_population' in bin_p2p_results:
            formatted_results['stellar_population'] = {}
            
            # Process each parameter
            for param_name, param_map in bin_p2p_results['stellar_population'].items():
                if isinstance(param_map, np.ndarray) and param_map.shape == (cube._n_y, cube._n_x):
                    # Extract values at bin centers
                    bin_values = []
                    for i, (x, y) in enumerate(zip(bin_adapter.x, bin_adapter.y)):
                        # Find closest pixel
                        x_idx = min(max(int(x / cube._pxl_size_x + cube._n_x / 2), 0), cube._n_x - 1)
                        y_idx = min(max(int(y / cube._pxl_size_y + cube._n_y / 2), 0), cube._n_y - 1)
                        bin_values.append(param_map[y_idx, x_idx])
                    
                    formatted_results['stellar_population'][param_name] = np.array(bin_values)
                else:
                    formatted_results['stellar_population'][param_name] = param_map
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error in analysis on binned data: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

def create_vnb_plots(args, binned_data, vnb_results, galaxy_name, plots_dir):
    """
    Create visualization plots for VNB analysis
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    binned_data : VoronoiBinnedData
        Binned data object
    vnb_results : dict
        VNB analysis results
    galaxy_name : str
        Galaxy name
    plots_dir : Path
        Directory to save plots
    """
    try:
        # Create basic binning plots
        create_binning_plots(binned_data, plots_dir, galaxy_name)
        
        # Create stellar kinematics plots
        if 'stellar_kinematics' in vnb_results:
            velocity = vnb_results['stellar_kinematics'].get('velocity', None)
            dispersion = vnb_results['stellar_kinematics'].get('dispersion', None)
            
            if velocity is not None and dispersion is not None:
                try:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Get bin centers
                    x_gen = binned_data.metadata.get('x_gen', None)
                    y_gen = binned_data.metadata.get('y_gen', None)
                    
                    if x_gen is None or y_gen is None:
                        # Reconstruct bin centers from bin map
                        x_centers = []
                        y_centers = []
                        bin_map = binned_data.bin_num.reshape(binned_data.metadata['ny'], binned_data.metadata['nx'])
                        for i in range(len(velocity)):
                            mask = bin_map == i
                            if np.any(mask):
                                y_indices, x_indices = np.where(mask)
                                x_centers.append(np.mean(x_indices))
                                y_centers.append(np.mean(y_indices))
                        x_gen = np.array(x_centers)
                        y_gen = np.array(y_centers)
                    
                    # Plot velocity field
                    sc0 = axes[0].scatter(x_gen, y_gen, c=velocity, cmap='coolwarm', 
                                        s=50, edgecolor='k')
                    plt.colorbar(sc0, ax=axes[0], label='Velocity (km/s)')
                    axes[0].set_title('Stellar Velocity')
                    
                    # Plot dispersion field
                    sc1 = axes[1].scatter(x_gen, y_gen, c=dispersion, cmap='viridis', 
                                        s=50, edgecolor='k')
                    plt.colorbar(sc1, ax=axes[1], label='Dispersion (km/s)')
                    axes[1].set_title('Stellar Velocity Dispersion')
                    
                    # Save figure
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"{galaxy_name}_vnb_kinematics.png", dpi=150)
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"Error creating kinematics plots: {str(e)}")
        
        # Create stellar population plots
        if 'stellar_population' in vnb_results:
            try:
                params = vnb_results['stellar_population']
                param_names = list(params.keys())
                
                if param_names:
                    fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 4))
                    if len(param_names) == 1:
                        axes = [axes]
                    
                    x_gen = binned_data.metadata.get('x_gen', None)
                    y_gen = binned_data.metadata.get('y_gen', None)
                    
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
                        
                        sc = axes[i].scatter(x_gen, y_gen, c=param_values, cmap='plasma', 
                                          s=50, edgecolor='k')
                        plt.colorbar(sc, ax=axes[i], label=label)
                        axes[i].set_title(f'Stellar {label}')
                    
                    # Save figure
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"{galaxy_name}_vnb_stellar_pop.png", dpi=150)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Error creating stellar population plots: {str(e)}")
        
        # Create emission line plots
        if 'emission' in vnb_results:
            try:
                emission = vnb_results['emission']
                
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
                    
                    x_gen = binned_data.metadata.get('x_gen', None)
                    y_gen = binned_data.metadata.get('y_gen', None)
                    
                    for i, (line_name, flux) in enumerate(list(flux_maps.items())[:n_lines]):
                        # Use log scale for better visualization
                        with np.errstate(divide='ignore', invalid='ignore'):
                            log_flux = np.log10(flux)
                        
                        valid_mask = np.isfinite(log_flux)
                        if np.any(valid_mask):
                            vmin = np.nanpercentile(log_flux[valid_mask], 5)
                            vmax = np.nanpercentile(log_flux[valid_mask], 95)
                            
                            sc = axes[i].scatter(x_gen, y_gen, c=log_flux, cmap='inferno', 
                                              s=50, edgecolor='k', vmin=vmin, vmax=vmax)
                            plt.colorbar(sc, ax=axes[i], label='Log Flux')
                        else:
                            axes[i].scatter(x_gen, y_gen, c='gray', s=50, edgecolor='k')
                        
                        axes[i].set_title(f'{line_name} Flux')
                    
                    # Save figure
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"{galaxy_name}_vnb_emission.png", dpi=150)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Error creating emission line plots: {str(e)}")
        
        # Create spectral indices plots
        if 'indices' in vnb_results:
            try:
                indices = vnb_results['indices']
                index_names = list(indices.keys())
                
                if index_names:
                    n_indices = min(len(index_names), 6)  # Show at most 6 indices
                    fig, axes = plt.subplots(1, n_indices, figsize=(4 * n_indices, 4))
                    if n_indices == 1:
                        axes = [axes]
                    
                    x_gen = binned_data.metadata.get('x_gen', None)
                    y_gen = binned_data.metadata.get('y_gen', None)
                    
                    for i, index_name in enumerate(index_names[:n_indices]):
                        index_values = indices[index_name]
                        
                        valid_mask = np.isfinite(index_values)
                        if np.any(valid_mask):
                            vmin = np.nanpercentile(index_values[valid_mask], 5)
                            vmax = np.nanpercentile(index_values[valid_mask], 95)
                            
                            sc = axes[i].scatter(x_gen, y_gen, c=index_values, cmap='viridis', 
                                              s=50, edgecolor='k', vmin=vmin, vmax=vmax)
                            plt.colorbar(sc, ax=axes[i], label='Index Value')
                        else:
                            axes[i].scatter(x_gen, y_gen, c='gray', s=50, edgecolor='k')
                        
                        axes[i].set_title(f'{index_name} Index')
                    
                    # Save figure
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"{galaxy_name}_vnb_indices.png", dpi=150)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Error creating spectral indices plots: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error creating VNB plots: {str(e)}")
        logger.error(traceback.format_exc())

def create_binning_plots(binned_data, plots_dir, galaxy_name):
    """
    Create basic binning visualization plots
    
    Parameters
    ----------
    binned_data : VoronoiBinnedData
        Binned data object
    plots_dir : Path
        Directory to save plots
    galaxy_name : str
        Galaxy name
    """
    try:
        # Create Voronoi bin map
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get bin centers and SNR values
        x_gen = binned_data.metadata.get('x_gen', None)
        y_gen = binned_data.metadata.get('y_gen', None)
        sn = binned_data.metadata.get('sn', None)
        
        # Create a bin map image
        n_y = binned_data.metadata['ny']
        n_x = binned_data.metadata['nx']
        bin_map = np.full((n_y, n_x), -1)
        
        # Get unique bin numbers
        unique_bins = np.unique(binned_data.bin_num)
        unique_bins = unique_bins[unique_bins >= 0]  # Remove negative values
        
        # Fill bin map with bin numbers
        for i in unique_bins:
            mask = binned_data.bin_num == i
            if np.any(mask):
                bin_map[mask.reshape(n_y, n_x)] = i
        
        # Create random colors for each bin
        n_bins = len(x_gen)
        cmap = plt.cm.get_cmap('tab20', n_bins)
        colors = [cmap(i) for i in range(n_bins)]
        
        # Create colored bin map
        rgba_colors = np.zeros((n_y, n_x, 4))
        for i, color in enumerate(colors):
            mask = bin_map == i
            if np.any(mask):
                rgba_colors[mask] = color
        
        # Show bin map
        ax.imshow(rgba_colors, origin='lower', aspect='equal')
        
        # Plot bin centers
        for i, (x, y) in enumerate(zip(x_gen, y_gen)):
            ax.text(x, y, str(i), color='black', fontsize=8, 
                   ha='center', va='center', backgroundcolor='white')
        
        # Add colorbar for SNR
        if sn is not None:
            # Create a scatter plot with SNR values
            sc = ax.scatter(x_gen, y_gen, c=sn, cmap='viridis', 
                           s=10, alpha=0.7)
            plt.colorbar(sc, ax=ax, label='Signal-to-Noise Ratio')
        
        ax.set_title(f'Voronoi Binning Map - {galaxy_name}')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(plots_dir / f"{galaxy_name}_vnb_binning_map.png", dpi=150)
        plt.close(fig)
        
        # Create bin SNR histogram
        if sn is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot histogram of SNR values
            ax.hist(sn, bins=20, color='skyblue', edgecolor='black')
            
            # Add target SNR line
            target_snr = binned_data.metadata.get('target_snr', None)
            if target_snr is not None:
                ax.axvline(x=target_snr, color='red', linestyle='--', 
                          label=f'Target SNR = {target_snr:.1f}')
            
            # Add median SNR line
            median_snr = np.median(sn)
            ax.axvline(x=median_snr, color='green', linestyle='-', 
                      label=f'Median SNR = {median_snr:.1f}')
            
            ax.set_title(f'Bin SNR Distribution - {galaxy_name}')
            ax.set_xlabel('Signal-to-Noise Ratio')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            # Save figure
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_vnb_snr_histogram.png", dpi=150)
            plt.close(fig)
        
        # Create bin size histogram
        n_pixels = binned_data.metadata.get('n_pixels', None)
        if n_pixels is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot histogram of bin sizes
            ax.hist(n_pixels, bins=20, color='lightgreen', edgecolor='black')
            
            # Add median bin size line
            median_pixels = np.median(n_pixels)
            ax.axvline(x=median_pixels, color='red', linestyle='-', 
                      label=f'Median Size = {median_pixels:.1f} pixels')
            
            ax.set_title(f'Bin Size Distribution - {galaxy_name}')
            ax.set_xlabel('Number of Pixels per Bin')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            # Save figure
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_vnb_binsize_histogram.png", dpi=150)
            plt.close(fig)
    
    except Exception as e:
        logger.error(f"Error creating binning plots: {str(e)}")
        logger.error(traceback.format_exc())