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

# In voronoi.py, modify run_vnb_analysis function

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

def combine_spectra_with_velocity_correction(spectra, wavelength, bin_indices, velocity_field, n_x, n_y):
    """
    Combine spectra within bins with velocity correction.
    
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
                            from utils.calc import spectres
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
    Create visualization plots for VNB analysis using physical coordinates
    
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
        create_robust_binning_plots(binned_data, plots_dir, galaxy_name)
        
        # Import the parameter map plotting function
        from visualization import plot_parameter_map
        
        # Get pixel scale for coordinate conversion
        pixel_size_x = binned_data.metadata.get('pixelsize_x', 1.0)
        pixel_size_y = binned_data.metadata.get('pixelsize_y', 1.0)
        
        # Get bin centers
        x_gen = binned_data.metadata.get('x_gen', None)
        y_gen = binned_data.metadata.get('y_gen', None)
        
        # Get dimensions
        n_y = binned_data.metadata['ny']
        n_x = binned_data.metadata['nx']
        
        # Create 2D bin map
        try:
            bin_map = binned_data.bin_num.reshape(n_y, n_x)
        except:
            # If reshaping fails, create a simple bin map
            bin_map = np.full((n_y, n_x), -1)
            for i, indices in enumerate(binned_data.bin_indices):
                for idx in indices:
                    y = idx // n_x
                    x = idx % n_x
                    if 0 <= y < n_y and 0 <= x < n_x:
                        bin_map[y, x] = i
        
        # Create stellar kinematics plots
        if 'stellar_kinematics' in vnb_results:
            velocity = vnb_results['stellar_kinematics'].get('velocity', None)
            dispersion = vnb_results['stellar_kinematics'].get('dispersion', None)
            
            if velocity is not None and dispersion is not None:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot velocity field
                plot_parameter_map(
                    velocity, bin_map, ax=axes[0], 
                    title='Stellar Velocity', cmap='coolwarm',
                    label='Velocity (km/s)'
                )
                
                # Plot dispersion field
                plot_parameter_map(
                    dispersion, bin_map, ax=axes[1], 
                    title='Stellar Dispersion', cmap='viridis',
                    label='Dispersion (km/s)'
                )
                
                plt.tight_layout()
                plt.savefig(plots_dir / f"{galaxy_name}_vnb_kinematics.png", dpi=150)
                plt.close(fig)
        
        # Create stellar population plots
        if 'stellar_population' in vnb_results:
            params = vnb_results['stellar_population']
            param_names = list(params.keys())
            
            if param_names:
                fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 4))
                if len(param_names) == 1:
                    axes = [axes]
                
                for i, param_name in enumerate(param_names):
                    param_values = params[param_name]
                    
                    # Determine label and prepare values
                    if param_name == 'age':
                        if isinstance(param_values, np.ndarray):
                            param_values = param_values * 1e-9  # Convert to Gyr
                        label = 'Age (Gyr)'
                    elif param_name == 'log_age':
                        label = 'Log Age (yr)'
                    elif param_name == 'metallicity':
                        label = 'Metallicity [Z/H]'
                    else:
                        label = param_name
                    
                    # Plot parameter map
                    plot_parameter_map(
                        param_values, bin_map, ax=axes[i], 
                        title=f'Stellar {label}', cmap='plasma',
                        label=label
                    )
                
                plt.tight_layout()
                plt.savefig(plots_dir / f"{galaxy_name}_vnb_stellar_pop.png", dpi=150)
                plt.close(fig)
        
        # Create emission line plots if available
        if 'emission' in vnb_results:
            # Find all flux maps
            flux_maps = {}
            for key, value in vnb_results['emission'].items():
                if key.startswith('flux_') and isinstance(value, np.ndarray):
                    line_name = key[5:]  # Remove 'flux_' prefix
                    flux_maps[line_name] = value
            
            if flux_maps:
                n_plots = min(len(flux_maps), 3)  # Show at most 3 lines
                fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
                if n_plots == 1:
                    axes = [axes]
                
                for i, (line_name, flux) in enumerate(list(flux_maps.items())[:n_plots]):
                    # Use log scale for flux maps with positive values
                    if isinstance(flux, np.ndarray) and np.any(np.isfinite(flux)) and np.nanmin(flux[np.isfinite(flux)]) > 0:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            log_flux = np.log10(flux)
                        plot_parameter_map(
                            log_flux, bin_map, ax=axes[i], 
                            title=f'{line_name} Flux', cmap='inferno',
                            label='Log Flux'
                        )
                    else:
                        plot_parameter_map(
                            flux, bin_map, ax=axes[i], 
                            title=f'{line_name} Flux', cmap='inferno',
                            label='Flux'
                        )
                
                plt.tight_layout()
                plt.savefig(plots_dir / f"{galaxy_name}_vnb_emission.png", dpi=150)
                plt.close(fig)
        
        # Create spectral indices plots if available
        if 'indices' in vnb_results:
            indices = vnb_results['indices']
            index_names = list(indices.keys())
            
            if index_names:
                n_plots = min(len(index_names), 3)  # Show at most 3 indices
                fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
                if n_plots == 1:
                    axes = [axes]
                
                for i, index_name in enumerate(index_names[:n_plots]):
                    index_values = indices[index_name]
                    plot_parameter_map(
                        index_values, bin_map, ax=axes[i], 
                        title=f'{index_name} Index', cmap='viridis',
                        label='Index Value'
                    )
                
                plt.tight_layout()
                plt.savefig(plots_dir / f"{galaxy_name}_vnb_indices.png", dpi=150)
                plt.close(fig)
                
    except Exception as e:
        logger.error(f"Error creating VNB plots: {str(e)}")
        logger.error(traceback.format_exc())
    """
    Create visualization plots for VNB analysis using physical coordinates
    
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
        create_robust_binning_plots(binned_data, plots_dir, galaxy_name)
        
        # Get pixel scale for coordinate conversion
        pixel_size_x = binned_data.metadata.get('pixelsize_x', 1.0)
        pixel_size_y = binned_data.metadata.get('pixelsize_y', 1.0)
        
        # Get bin centers
        x_gen = binned_data.metadata.get('x_gen', None)
        y_gen = binned_data.metadata.get('y_gen', None)
        
        # Get dimensions
        n_y = binned_data.metadata['ny']
        n_x = binned_data.metadata['nx']
        
        # Convert bin centers to physical units (arcsec)
        if x_gen is not None and y_gen is not None:
            x_gen_physical = (x_gen - n_x/2) * pixel_size_x
            y_gen_physical = (y_gen - n_y/2) * pixel_size_y
        else:
            # If bin centers not available, recreate from bin map
            x_gen_physical = []
            y_gen_physical = []
            try:
                bin_map = binned_data.bin_num.reshape(n_y, n_x)
                for i in range(np.max(bin_map) + 1):
                    mask = bin_map == i
                    if np.any(mask):
                        y_indices, x_indices = np.where(mask)
                        x_center = np.mean(x_indices)
                        y_center = np.mean(y_indices)
                        x_gen_physical.append((x_center - n_x/2) * pixel_size_x)
                        y_gen_physical.append((y_center - n_y/2) * pixel_size_y)
                x_gen_physical = np.array(x_gen_physical)
                y_gen_physical = np.array(y_gen_physical)
            except Exception as e:
                logger.warning(f"Could not recreate bin centers: {e}")
                x_gen_physical = np.array([])
                y_gen_physical = np.array([])
        
        
        # Create stellar kinematics plots
        if 'stellar_kinematics' in vnb_results and len(x_gen_physical) > 0:
            velocity = vnb_results['stellar_kinematics'].get('velocity', None)
            dispersion = vnb_results['stellar_kinematics'].get('dispersion', None)
            
            if velocity is not None and dispersion is not None:
                try:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Plot velocity field
                    valid_mask = np.isfinite(velocity)
                    if np.any(valid_mask):
                        # Get percentile range for color scale
                        vmin = np.percentile(velocity[valid_mask], 5)
                        vmax = np.percentile(velocity[valid_mask], 95)
                        
                        # For velocity, use symmetric color scale if appropriate
                        vabs = max(abs(vmin), abs(vmax))
                        if vmin < 0 and vmax > 0:
                            vmin, vmax = -vabs, vabs
                        
                        sc0 = axes[0].scatter(
                            x_gen_physical[valid_mask], 
                            y_gen_physical[valid_mask], 
                            c=velocity[valid_mask], 
                            cmap='coolwarm', 
                            s=50, 
                            edgecolor='k',
                            vmin=vmin,
                            vmax=vmax
                        )
                        cbar0 = plt.colorbar(sc0, ax=axes[0], label='Velocity (km/s)')
                        cbar0.ax.tick_params(labelsize=8)
                    else:
                        axes[0].text(0.5, 0.5, "No valid velocity data", 
                                   ha='center', va='center', transform=axes[0].transAxes)
                    
                    axes[0].set_xlabel('X (arcsec)')
                    axes[0].set_ylabel('Y (arcsec)')
                    axes[0].set_title('Stellar Velocity')
                    axes[0].set_aspect('equal')
                    axes[0].grid(True, alpha=0.3)
                    
                    # Plot dispersion field
                    valid_mask = np.isfinite(dispersion)
                    if np.any(valid_mask):
                        # Get percentile range for color scale
                        vmin = np.percentile(dispersion[valid_mask], 5)
                        vmax = np.percentile(dispersion[valid_mask], 95)
                        
                        sc1 = axes[1].scatter(
                            x_gen_physical[valid_mask], 
                            y_gen_physical[valid_mask], 
                            c=dispersion[valid_mask], 
                            cmap='viridis', 
                            s=50, 
                            edgecolor='k',
                            vmin=vmin,
                            vmax=vmax
                        )
                        cbar1 = plt.colorbar(sc1, ax=axes[1], label='Dispersion (km/s)')
                        cbar1.ax.tick_params(labelsize=8)
                    else:
                        axes[1].text(0.5, 0.5, "No valid dispersion data", 
                                   ha='center', va='center', transform=axes[1].transAxes)
                    
                    axes[1].set_xlabel('X (arcsec)')
                    axes[1].set_ylabel('Y (arcsec)')
                    axes[1].set_title('Stellar Velocity Dispersion')
                    axes[1].set_aspect('equal')
                    axes[1].grid(True, alpha=0.3)
                    
                    # Add overall title
                    plt.suptitle(f'Stellar Kinematics - {galaxy_name}', fontsize=14)
                    
                    # Save figure
                    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
                    plt.savefig(plots_dir / f"{galaxy_name}_vnb_kinematics.png", dpi=150)
                    plt.close(fig)
                    
                    # Create v/sigma map (useful kinematic indicator)
                    try:
                        fig, ax = plt.subplots(figsize=(8, 7))
                        
                        # Calculate v/sigma
                        valid_mask = np.isfinite(velocity) & np.isfinite(dispersion) & (dispersion > 0)
                        if np.any(valid_mask):
                            v_sigma = np.abs(velocity[valid_mask]) / dispersion[valid_mask]
                            
                            # Determine sensible scale
                            vmin = 0
                            vmax = min(2.0, np.percentile(v_sigma, 95))
                            
                            sc = ax.scatter(
                                x_gen_physical[valid_mask], 
                                y_gen_physical[valid_mask], 
                                c=v_sigma, 
                                cmap='magma', 
                                s=50, 
                                edgecolor='k',
                                vmin=vmin,
                                vmax=vmax
                            )
                            plt.colorbar(sc, ax=ax, label='|V|/σ')
                            
                            ax.set_xlabel('X (arcsec)')
                            ax.set_ylabel('Y (arcsec)')
                            ax.set_title(f'Rotational Support (|V|/σ) - {galaxy_name}')
                            ax.set_aspect('equal')
                            ax.grid(True, alpha=0.3)
                            
                            # Save figure
                            plt.tight_layout()
                            plt.savefig(plots_dir / f"{galaxy_name}_vnb_v_sigma.png", dpi=150)
                        else:
                            logger.warning("Insufficient data for v/sigma map")
                        plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Error creating v/sigma map: {e}")
                        plt.close('all')
                except Exception as e:
                    logger.warning(f"Error creating kinematics plots: {str(e)}")
                    plt.close('all')
        
                    
        # Create stellar population plots
        if 'stellar_population' in vnb_results and len(x_gen_physical) > 0:
            try:
                params = vnb_results['stellar_population']
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
                        
                        valid_mask = np.isfinite(param_values)
                        if np.any(valid_mask):
                            # Get appropriate range for color scale
                            vmin = np.percentile(param_values[valid_mask], 5)
                            vmax = np.percentile(param_values[valid_mask], 95)
                            
                            sc = axes[i].scatter(
                                x_gen_physical[valid_mask], 
                                y_gen_physical[valid_mask], 
                                c=param_values[valid_mask], 
                                cmap='plasma', 
                                s=50, 
                                edgecolor='k',
                                vmin=vmin, 
                                vmax=vmax
                            )
                            cbar = plt.colorbar(sc, ax=axes[i], label=label)
                            cbar.ax.tick_params(labelsize=8)
                        else:
                            axes[i].text(0.5, 0.5, f"No valid {param_name} data", 
                                       ha='center', va='center', transform=axes[i].transAxes)
                        
                        axes[i].set_xlabel('X (arcsec)')
                        axes[i].set_ylabel('Y (arcsec)')
                        axes[i].set_title(f'Stellar {label}')
                        axes[i].set_aspect('equal')
                        axes[i].grid(True, alpha=0.3)
                    
                    # Add overall title
                    plt.suptitle(f'Stellar Population - {galaxy_name}', fontsize=14)
                    
                    # Save figure
                    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
                    plt.savefig(plots_dir / f"{galaxy_name}_vnb_stellar_pop.png", dpi=150)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Error creating stellar population plots: {str(e)}")
                plt.close('all')
    except Exception as e:
                logger.warning(f"Error creating stellar population plots: {str(e)}")
                plt.close('all')


def create_binning_plots(binned_data, plots_dir, galaxy_name):
    """
    Create basic binning visualization plots with real physical coordinates
    
    Parameters
    ----------
    binned_data : BinnedSpectra
        Binned data object
    plots_dir : Path
        Directory to save plots
    galaxy_name : str
        Galaxy name
    """
    try:
        # Get pixel scale for coordinate conversion
        pixel_size_x = binned_data.metadata.get('pixelsize_x', 1.0)
        pixel_size_y = binned_data.metadata.get('pixelsize_y', 1.0)
        
        # Get bin centers and SNR values
        x_gen = binned_data.metadata.get('x_gen', None)
        y_gen = binned_data.metadata.get('y_gen', None)
        sn = binned_data.metadata.get('sn', None)
        
        # Get dimensions from metadata
        n_y = binned_data.metadata.get('ny', 1)
        n_x = binned_data.metadata.get('nx', 1)
        
        # Handle different bin_num formats
        bin_num = binned_data.bin_num
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a mapping from bin numbers to pixels
        x_coords = []
        y_coords = []
        bins = []
        
        # Check if we can reshape bin_num to 2D
        try:
            bin_map = bin_num.reshape(n_y, n_x)
            # Convert to physical coordinates using meshgrid
            y_indices, x_indices = np.indices((n_y, n_x))
            x_phys = (x_indices - n_x/2) * pixel_size_x
            y_phys = (y_indices - n_y/2) * pixel_size_y
            
            # Find all valid bins
            for bin_id in range(np.max(bin_map) + 1):
                mask = bin_map == bin_id
                if np.any(mask):
                    y_idx, x_idx = np.where(mask)
                    for i in range(len(y_idx)):
                        y_coords.append(y_phys[y_idx[i], x_idx[i]])
                        x_coords.append(x_phys[y_idx[i], x_idx[i]])
                        bins.append(bin_id)
        except:
            # If reshaping fails, try to get pixels from bin_indices
            if hasattr(binned_data, 'bin_indices') and binned_data.bin_indices:
                for bin_id, indices in enumerate(binned_data.bin_indices):
                    for idx in indices:
                        # Convert linear index to 2D coordinates
                        y = idx // n_x
                        x = idx % n_x
                        if 0 <= y < n_y and 0 <= x < n_x:
                            # Convert to physical coordinates
                            x_phys = (x - n_x/2) * pixel_size_x
                            y_phys = (y - n_y/2) * pixel_size_y
                            x_coords.append(x_phys)
                            y_coords.append(y_phys)
                            bins.append(bin_id)
            else:
                # No binning information, plot a placeholder
                ax.text(0.5, 0.5, "No valid binning data available", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Binning Map - {galaxy_name}')
                
                # Save figure and return
                plt.tight_layout()
                plt.savefig(plots_dir / f"{galaxy_name}_binning_map.png", dpi=150)
                plt.close(fig)
                return
        
        # Convert lists to arrays
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        bins = np.array(bins)
        
        # Check if we have any valid bins
        if len(bins) == 0:
            ax.text(0.5, 0.5, "No valid binning data available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Binning Map - {galaxy_name}')
                
            # Save figure and return
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_binning_map.png", dpi=150)
            plt.close(fig)
            return
        
        # Use scatter plot instead of pcolormesh
        cmap = plt.cm.get_cmap('tab20', max(20, np.max(bins)+1))
        unique_bins = np.unique(bins)
        
        # Plot each bin with a separate call to scatter for proper legend
        for bin_id in unique_bins:
            mask = bins == bin_id
            ax.scatter(x_coords[mask], y_coords[mask], 
                     color=cmap(bin_id % 20), s=15, alpha=0.7,
                     label=f'Bin {bin_id}' if bin_id < 5 else None)
        
        # Add bin centers if available
        if x_gen is not None and y_gen is not None:
            # Convert to physical coordinates
            x_gen_phys = [(x - n_x/2) * pixel_size_x for x in x_gen]
            y_gen_phys = [(y - n_y/2) * pixel_size_y for y in y_gen]
            
            # Plot bin centers with numbers
            for i, (x, y) in enumerate(zip(x_gen_phys, y_gen_phys)):
                if i < len(unique_bins):
                    ax.plot(x, y, 'ko', markersize=8)
                    ax.text(x, y, str(i), color='white', fontsize=8, 
                           ha='center', va='center',
                           bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.1'))
        
        # Add SNR values if available
        if sn is not None and x_gen is not None and y_gen is not None:
            # Create a separate scatter plot for SNR
            try:
                limit = min(len(x_gen_phys), len(y_gen_phys), len(sn))
                sc = ax.scatter(
                    x_gen_phys[:limit], 
                    y_gen_phys[:limit], 
                    c=sn[:limit], 
                    cmap='viridis', 
                    s=30, 
                    alpha=0.5,
                    edgecolor='k'
                )
                plt.colorbar(sc, ax=ax, label='Signal-to-Noise Ratio')
            except Exception as e:
                logger.warning(f"Error adding SNR colorbar: {e}")
        
        # Set labels and grid
        ax.set_xlabel('X (arcsec)')
        ax.set_ylabel('Y (arcsec)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Binning Map - {galaxy_name}')
        
        # Add legend for first few bins
        if len(unique_bins) > 0:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles[:min(5, len(handles))],
                    labels[:min(5, len(labels))],
                    loc='upper right', 
                    fontsize='small'
                )
        
        # Save figure
        plt.tight_layout()
        plt.savefig(plots_dir / f"{galaxy_name}_binning_map.png", dpi=150)
        plt.close(fig)
        
        # Create histograms for bin properties
        # SNR histogram
        if sn is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(sn, bins=20, color='skyblue', edgecolor='black')
            
            # Add target SNR line if available
            target_snr = binned_data.metadata.get('target_snr', None)
            if target_snr is not None:
                ax.axvline(x=target_snr, color='red', linestyle='--', 
                          label=f'Target SNR = {target_snr:.1f}')
            
            # Add median SNR line
            median_snr = np.nanmedian(sn)
            ax.axvline(x=median_snr, color='green', linestyle='-', 
                      label=f'Median SNR = {median_snr:.1f}')
            
            ax.set_title(f'Bin SNR Distribution - {galaxy_name}')
            ax.set_xlabel('Signal-to-Noise Ratio')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_snr_histogram.png", dpi=150)
            plt.close(fig)
        
        # Bin size histogram
        n_pixels = binned_data.metadata.get('n_pixels', None)
        if n_pixels is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(n_pixels, bins=20, color='lightgreen', edgecolor='black')
            
            median_pixels = np.nanmedian(n_pixels)
            ax.axvline(x=median_pixels, color='red', linestyle='-', 
                      label=f'Median Size = {median_pixels:.1f} pixels')
            
            ax.set_title(f'Bin Size Distribution - {galaxy_name}')
            ax.set_xlabel('Number of Pixels per Bin')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_binsize_histogram.png", dpi=150)
            plt.close(fig)
        
        # Bin area in physical units
        if n_pixels is not None and pixel_size_x > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            bin_areas = n_pixels * (pixel_size_x * pixel_size_y)  # arcsec²
            ax.hist(bin_areas, bins=20, color='salmon', edgecolor='black')
            
            median_area = np.nanmedian(bin_areas)
            ax.axvline(x=median_area, color='red', linestyle='-', 
                      label=f'Median Area = {median_area:.2f} arcsec²')
            
            ax.set_title(f'Bin Area Distribution - {galaxy_name}')
            ax.set_xlabel('Area (arcsec²)')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_binsize_histogram.png", dpi=150)
            plt.close(fig)
            
    except Exception as e:
        logger.error(f"Error creating binning plots: {str(e)}")
        logger.error(traceback.format_exc())
    """
    Create basic binning visualization plots with real physical coordinates
    
    Parameters
    ----------
    binned_data : BinnedSpectra
        Binned data object
    plots_dir : Path
        Directory to save plots
    galaxy_name : str
        Galaxy name
    """
    try:
        # Get pixel scale for coordinate conversion
        pixel_size_x = binned_data.metadata.get('pixelsize_x', 1.0)
        pixel_size_y = binned_data.metadata.get('pixelsize_y', 1.0)
        
        # Get bin centers and SNR values
        x_gen = binned_data.metadata.get('x_gen', None)
        y_gen = binned_data.metadata.get('y_gen', None)
        sn = binned_data.metadata.get('sn', None)
        
        # Get dimensions from metadata
        n_y = binned_data.metadata.get('ny', 1)
        n_x = binned_data.metadata.get('nx', 1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Handle different bin_num formats
        bin_num = binned_data.bin_num
        
        # Convert bin_num to appropriate dimensions
        if isinstance(bin_num, np.ndarray):
            if len(bin_num.shape) == 1:
                # 1D array
                if len(bin_num) == n_y * n_x:
                    # Can reshape to 2D
                    bin_map = bin_num.reshape(n_y, n_x)
                else:
                    # Can't reshape, create empty map
                    bin_map = np.full((n_y, n_x), -1)
                    
                    # Fill with bin numbers where possible
                    for bin_id in np.unique(bin_num):
                        if bin_id >= 0:
                            idx_list = np.where(bin_num == bin_id)[0]
                            for idx in idx_list:
                                if idx < n_y * n_x:
                                    row = idx // n_x
                                    col = idx % n_x
                                    if 0 <= row < n_y and 0 <= col < n_x:
                                        bin_map[row, col] = bin_id
            else:
                # Already 2D
                bin_map = bin_num
        else:
            # Not an array, create empty map
            bin_map = np.full((n_y, n_x), -1)
        
        # Create physical coordinate grid
        y_coords, x_coords = np.indices((n_y, n_x))
        x_physical = (x_coords - n_x/2) * pixel_size_x
        y_physical = (y_coords - n_y/2) * pixel_size_y
        
        # Plot bins using scatter plot instead of pcolormesh
        # Get unique bin numbers
        unique_bins = np.unique(bin_map)
        unique_bins = unique_bins[unique_bins >= 0]
        
        # Create colormap
        cmap = plt.cm.get_cmap('tab20', max(20, len(unique_bins)))
        
        for i, bin_id in enumerate(unique_bins):
            mask = bin_map == bin_id
            if np.any(mask):
                y_idx, x_idx = np.where(mask)
                x_vals = x_physical[y_idx, x_idx]
                y_vals = y_physical[y_idx, x_idx]
                color = cmap(i % 20)
                ax.scatter(x_vals, y_vals, color=color, s=15, alpha=0.7,
                          label=f'Bin {bin_id}' if i < 5 else None)
        
        # Add bin centers if available
        if x_gen is not None and y_gen is not None:
            x_gen_phys = np.array([(x - n_x/2) * pixel_size_x for x in x_gen])
            y_gen_phys = np.array([(y - n_y/2) * pixel_size_y for y in y_gen])
            
            # Plot centers with numbers
            for i, (x, y) in enumerate(zip(x_gen_phys, y_gen_phys)):
                if i < len(unique_bins):
                    ax.plot(x, y, 'ko', markersize=5)
                    ax.text(x, y, str(i), color='white', fontsize=8, ha='center', va='center',
                           bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.1'))
        
        # Add SNR colorbar if available
        if sn is not None and x_gen is not None and y_gen is not None:
            # Create separate scatter points for SNR values
            try:
                limit = min(len(x_gen_phys), len(y_gen_phys), len(sn))
                sc = ax.scatter(
                    x_gen_phys[:limit], 
                    y_gen_phys[:limit], 
                    c=sn[:limit], 
                    cmap='viridis', 
                    s=20, 
                    alpha=0.5,
                    edgecolor='k'
                )
                plt.colorbar(sc, ax=ax, label='Signal-to-Noise Ratio')
            except Exception as e:
                logger.warning(f"Error adding SNR colorbar: {e}")
        
        # Set labels and grid
        ax.set_xlabel('X (arcsec)')
        ax.set_ylabel('Y (arcsec)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Binning Map - {galaxy_name}')
        
        # Add legend for first few bins
        if len(unique_bins) > 0:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles[:min(5, len(handles))],
                    labels[:min(5, len(labels))],
                    loc='upper right',
                    fontsize='small'
                )
        
        # Save figure
        plt.tight_layout()
        plt.savefig(plots_dir / f"{galaxy_name}_binning_map.png", dpi=150)
        plt.close(fig)
        
        # Create histograms for bin properties
        # SNR histogram
        if sn is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(sn, bins=20, color='skyblue', edgecolor='black')
            
            # Add target SNR line if available
            target_snr = binned_data.metadata.get('target_snr', None)
            if target_snr is not None:
                ax.axvline(x=target_snr, color='red', linestyle='--', 
                          label=f'Target SNR = {target_snr:.1f}')
            
            # Add median SNR line
            median_snr = np.nanmedian(sn)
            ax.axvline(x=median_snr, color='green', linestyle='-', 
                      label=f'Median SNR = {median_snr:.1f}')
            
            ax.set_title(f'Bin SNR Distribution - {galaxy_name}')
            ax.set_xlabel('Signal-to-Noise Ratio')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_snr_histogram.png", dpi=150)
            plt.close(fig)
        
        # Bin size histogram
        n_pixels = binned_data.metadata.get('n_pixels', None)
        if n_pixels is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(n_pixels, bins=20, color='lightgreen', edgecolor='black')
            
            median_pixels = np.nanmedian(n_pixels)
            ax.axvline(x=median_pixels, color='red', linestyle='-', 
                      label=f'Median Size = {median_pixels:.1f} pixels')
            
            ax.set_title(f'Bin Size Distribution - {galaxy_name}')
            ax.set_xlabel('Number of Pixels per Bin')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_binsize_histogram.png", dpi=150)
            plt.close(fig)
        
        # Bin area in physical units
        if n_pixels is not None and pixel_size_x > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            bin_areas = n_pixels * (pixel_size_x * pixel_size_y)  # arcsec²
            ax.hist(bin_areas, bins=20, color='salmon', edgecolor='black')
            
            median_area = np.nanmedian(bin_areas)
            ax.axvline(x=median_area, color='red', linestyle='-', 
                      label=f'Median Area = {median_area:.2f} arcsec²')
            
            ax.set_title(f'Bin Area Distribution - {galaxy_name}')
            ax.set_xlabel('Area (arcsec²)')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_bin_area.png", dpi=150)
            plt.close(fig)
            
    except Exception as e:
        logger.error(f"Error creating binning plots: {str(e)}")
        logger.error(traceback.format_exc())
    """
    Create basic binning visualization plots with real physical coordinates
    
    Parameters
    ----------
    binned_data : BinnedSpectra
        Binned data object
    plots_dir : Path
        Directory to save plots
    galaxy_name : str
        Galaxy name
    """
    try:
        # Get pixel scale for coordinate conversion
        pixel_size_x = binned_data.metadata.get('pixelsize_x', 1.0)
        pixel_size_y = binned_data.metadata.get('pixelsize_y', 1.0)
        
        # Get bin centers and SNR values
        x_gen = binned_data.metadata.get('x_gen', None)
        y_gen = binned_data.metadata.get('y_gen', None)
        sn = binned_data.metadata.get('sn', None)
        
        # Get dimensions from metadata
        n_y = binned_data.metadata.get('ny', 1)
        n_x = binned_data.metadata.get('nx', 1)
        
        # Check if bin_num is 1D or 2D and reshape accordingly
        bin_num = binned_data.bin_num
        if len(bin_num.shape) == 1 and len(bin_num) == n_y * n_x:
            # Reshape 1D bin_num to 2D
            bin_map = bin_num.reshape(n_y, n_x)
        else:
            # Already 2D or incompatible shape
            try:
                bin_map = bin_num.reshape(n_y, n_x)
            except:
                # Create a simple bin map if we can't reshape
                bin_map = np.full((n_y, n_x), -1)
                # Fill with available bin numbers
                for bin_id in np.unique(bin_num):
                    if bin_id >= 0:
                        bin_indices = np.where(bin_num == bin_id)[0]
                        for idx in bin_indices:
                            row = idx // n_x if idx < n_y * n_x else 0
                            col = idx % n_x if idx < n_y * n_x else 0
                            if 0 <= row < n_y and 0 <= col < n_x:
                                bin_map[row, col] = bin_id
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get physical coordinates for each pixel
        y_coords, x_coords = np.indices((n_y, n_x))
        x_physical = (x_coords - n_x/2) * pixel_size_x
        y_physical = (y_coords - n_y/2) * pixel_size_y
        
        # Get unique bins
        unique_bins = np.unique(bin_map)
        unique_bins = unique_bins[unique_bins >= 0]
        n_bins = len(unique_bins)
        
        # Create colormap
        cmap = plt.cm.get_cmap('tab20', max(20, n_bins))
        
        # Plot each bin using scatter plot
        for i, bin_id in enumerate(unique_bins):
            mask = bin_map == bin_id
            if np.any(mask):
                y_idx, x_idx = np.where(mask)
                
                # Get physical coordinates for this bin
                x_vals = x_physical[y_idx, x_idx]
                y_vals = y_physical[y_idx, x_idx]
                
                # Plot bin points
                color = cmap(i % 20)
                ax.scatter(x_vals, y_vals, color=color, s=15, alpha=0.7,
                          label=f'Bin {bin_id}' if i < 5 else None)
        
        # Add bin centers if available
        if x_gen is not None and y_gen is not None:
            # Convert to physical coordinates
            x_gen_phys = np.array([(x - n_x/2) * pixel_size_x for x in x_gen])
            y_gen_phys = np.array([(y - n_y/2) * pixel_size_y for y in y_gen])
            
            # Plot centers with numbers
            for i, (x, y) in enumerate(zip(x_gen_phys, y_gen_phys)):
                if i < len(unique_bins):
                    ax.plot(x, y, 'ko', markersize=5)
                    ax.text(x, y, str(i), color='white', fontsize=8, ha='center', va='center',
                           bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.1'))
        
        # Add SNR colorbar if available
        if sn is not None and x_gen is not None and y_gen is not None:
            # Create separate scatter points for SNR values
            sc = ax.scatter(
                x_gen_phys[:len(sn)], 
                y_gen_phys[:len(sn)], 
                c=sn, 
                cmap='viridis', 
                s=20, 
                alpha=0.5,
                edgecolor='k'
            )
            plt.colorbar(sc, ax=ax, label='Signal-to-Noise Ratio')
        
        # Set labels and grid
        ax.set_xlabel('X (arcsec)')
        ax.set_ylabel('Y (arcsec)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add title
        ax.set_title(f'Binning Map - {galaxy_name}')
        
        # Add legend for first few bins
        if n_bins > 0:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles[:min(5, len(handles))],  # Show up to 5 bins
                    labels[:min(5, len(labels))],
                    loc='upper right',
                    fontsize='small'
                )
        
        # Save figure
        plt.tight_layout()
        plt.savefig(plots_dir / f"{galaxy_name}_binning_map.png", dpi=150)
        plt.close(fig)
        
        # Create histograms for bin properties
        # SNR histogram
        if sn is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(sn, bins=20, color='skyblue', edgecolor='black')
            
            # Add target SNR line if available
            target_snr = binned_data.metadata.get('target_snr', None)
            if target_snr is not None:
                ax.axvline(x=target_snr, color='red', linestyle='--', 
                          label=f'Target SNR = {target_snr:.1f}')
            
            # Add median SNR line
            median_snr = np.nanmedian(sn)
            ax.axvline(x=median_snr, color='green', linestyle='-', 
                      label=f'Median SNR = {median_snr:.1f}')
            
            ax.set_title(f'Bin SNR Distribution - {galaxy_name}')
            ax.set_xlabel('Signal-to-Noise Ratio')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_snr_histogram.png", dpi=150)
            plt.close(fig)
        
        # Bin size histogram
        n_pixels = binned_data.metadata.get('n_pixels', None)
        if n_pixels is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(n_pixels, bins=20, color='lightgreen', edgecolor='black')
            
            median_pixels = np.nanmedian(n_pixels)
            ax.axvline(x=median_pixels, color='red', linestyle='-', 
                      label=f'Median Size = {median_pixels:.1f} pixels')
            
            ax.set_title(f'Bin Size Distribution - {galaxy_name}')
            ax.set_xlabel('Number of Pixels per Bin')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_binsize_histogram.png", dpi=150)
            plt.close(fig)
        
        # Bin area in physical units
        if n_pixels is not None and pixel_size_x > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            bin_areas = n_pixels * (pixel_size_x * pixel_size_y)  # arcsec²
            ax.hist(bin_areas, bins=20, color='salmon', edgecolor='black')
            
            median_area = np.nanmedian(bin_areas)
            ax.axvline(x=median_area, color='red', linestyle='-', 
                      label=f'Median Area = {median_area:.2f} arcsec²')
            
            ax.set_title(f'Bin Area Distribution - {galaxy_name}')
            ax.set_xlabel('Area (arcsec²)')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_bin_area.png", dpi=150)
            plt.close(fig)
            
    except Exception as e:
        logger.error(f"Error creating binning plots: {str(e)}")
        logger.error(traceback.format_exc())
    """
    Create basic binning visualization plots with real physical coordinates
    
    Parameters
    ----------
    binned_data : BinnedSpectra
        Binned data object
    plots_dir : Path
        Directory to save plots
    galaxy_name : str
        Galaxy name
    """
    try:
        # Get pixel scale for coordinate conversion
        pixel_size_x = binned_data.metadata.get('pixelsize_x', 1.0)
        pixel_size_y = binned_data.metadata.get('pixelsize_y', 1.0)
        
        # Get bin centers and SNR values
        x_gen = binned_data.metadata.get('x_gen', None)
        y_gen = binned_data.metadata.get('y_gen', None)
        sn = binned_data.metadata.get('sn', None)
        
        # Create bin map
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
                # We need to ensure mask is 2D if bin_num is 1D
                if len(mask.shape) == 1 and mask.shape[0] == n_y * n_x:
                    mask = mask.reshape(n_y, n_x)
                bin_map[mask] = i
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create physical coordinate grid for display
        y_coords, x_coords = np.indices((n_y, n_x))
        
        # Convert to physical units (arcseconds)
        # Center the coordinates (center at 0,0)
        x_physical = (x_coords - n_x/2) * pixel_size_x
        y_physical = (y_coords - n_y/2) * pixel_size_y
        
        # Create colored bin map
        n_bins = len(unique_bins)
        cmap = plt.cm.get_cmap('tab20', max(20, n_bins))
        
        # Plot each bin using scatter plot
        for i in unique_bins:
            mask = bin_map == i
            if np.any(mask):
                color = cmap(i % 20)  # Cycle through colors if more than 20 bins
                
                # Use scatter for plot - extract the coordinates
                y_idx, x_idx = np.where(mask)
                ax.scatter(
                    x_physical[y_idx, x_idx], 
                    y_physical[y_idx, x_idx], 
                    color=color, 
                    s=15, 
                    alpha=0.7,
                    label=f'Bin {i}' if i < 5 else None  # Only label first few bins
                )
        
        # Plot bin centers with bin numbers
        if x_gen is not None and y_gen is not None:
            # Convert bin centers to physical coordinates
            x_gen_physical = (x_gen - n_x/2) * pixel_size_x
            y_gen_physical = (y_gen - n_y/2) * pixel_size_y
            
            for i, (x, y) in enumerate(zip(x_gen_physical, y_gen_physical)):
                ax.text(x, y, str(i), color='black', fontsize=8, 
                       ha='center', va='center', backgroundcolor='white')
            
            # Also show bin centers as points
            ax.scatter(x_gen_physical, y_gen_physical, color='black', s=20, alpha=0.5)
        
        # Add colorbar for SNR if available
        if sn is not None and x_gen is not None and y_gen is not None:
            # Create a separate scattered points with SNR values
            sc = ax.scatter(
                x_gen_physical, 
                y_gen_physical, 
                c=sn, 
                cmap='viridis', 
                s=30, 
                alpha=0.7,
                edgecolor='k'
            )
            cbar = plt.colorbar(sc, ax=ax, label='Signal-to-Noise Ratio')
            # Add colorbar ticks
            cbar.ax.tick_params(labelsize=8)
        
        # Set axis labels with physical units
        ax.set_xlabel('X (arcsec)')
        ax.set_ylabel('Y (arcsec)')
        
        # Set equal aspect ratio for proper spatial representation
        ax.set_aspect('equal')
        
        # Add grid for reference
        ax.grid(True, alpha=0.3)
        
        # Add title
        ax.set_title(f'Voronoi Binning Map - {galaxy_name}')
        
        # Add legend for first few bins
        if n_bins > 0:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles[:5],  # Only show first 5 bins in legend
                    labels[:5],
                    loc='upper right',
                    fontsize='small'
                )
        
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
            
        # Create bin spatial coverage
        if n_pixels is not None and pixel_size_x > 0:
            # Calculate physical area of each bin in square arcseconds
            bin_areas = n_pixels * (pixel_size_x * pixel_size_y)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot histogram of bin areas
            ax.hist(bin_areas, bins=20, color='salmon', edgecolor='black')
            
            # Add median area line
            median_area = np.median(bin_areas)
            ax.axvline(x=median_area, color='red', linestyle='-', 
                      label=f'Median Area = {median_area:.2f} arcsec²')
            
            ax.set_title(f'Bin Area Distribution - {galaxy_name}')
            ax.set_xlabel('Area (arcsec²)')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            # Save figure
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_vnb_bin_area.png", dpi=150)
            plt.close(fig)
    
    except Exception as e:
        logger.error(f"Error creating binning plots: {str(e)}")
        logger.error(traceback.format_exc())
    """
    Create basic binning visualization plots with real physical coordinates
    
    Parameters
    ----------
    binned_data : BinnedSpectra
        Binned data object
    plots_dir : Path
        Directory to save plots
    galaxy_name : str
        Galaxy name
    """
    try:
        # Get pixel scale for coordinate conversion
        pixel_size_x = binned_data.metadata.get('pixelsize_x', 1.0)
        pixel_size_y = binned_data.metadata.get('pixelsize_y', 1.0)
        
        # Get bin centers and SNR values
        x_gen = binned_data.metadata.get('x_gen', None)
        y_gen = binned_data.metadata.get('y_gen', None)
        sn = binned_data.metadata.get('sn', None)
        
        # Create bin map
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
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create physical coordinate grid for display
        y_coords, x_coords = np.indices((n_y, n_x))
        
        # Convert to physical units (arcseconds)
        # Center the coordinates (center at 0,0)
        x_physical = (x_coords - n_x/2) * pixel_size_x
        y_physical = (y_coords - n_y/2) * pixel_size_y
        
        # Create colored bin map
        n_bins = len(unique_bins)
        cmap = plt.cm.get_cmap('tab20', n_bins)
        
        # Plot each bin with proper physical coordinates
        for i in unique_bins:
            mask = bin_map == i
            if np.any(mask):
                color = cmap(i % 20)  # Cycle through colors if more than 20 bins
                
                # Use scatter instead of pcolormesh for simplicity and compatibility
                y_idx, x_idx = np.where(mask)
                x_vals = x_physical[y_idx, x_idx]
                y_vals = y_physical[y_idx, x_idx]
                
                ax.scatter(x_vals, y_vals, color=color, s=15, alpha=0.7)
        
        # Plot bin centers with bin numbers
        if x_gen is not None and y_gen is not None:
            # Convert bin centers to physical coordinates
            x_gen_physical = (x_gen - n_x/2) * pixel_size_x
            y_gen_physical = (y_gen - n_y/2) * pixel_size_y
            
            for i, (x, y) in enumerate(zip(x_gen_physical, y_gen_physical)):
                ax.text(x, y, str(i), color='black', fontsize=8, 
                       ha='center', va='center', backgroundcolor='white')
        
        # Add colorbar for SNR if available
        if sn is not None and x_gen is not None and y_gen is not None:
            sc = ax.scatter(x_gen_physical, y_gen_physical, c=sn, 
                          cmap='viridis', s=30, alpha=0.7)
            cbar = plt.colorbar(sc, ax=ax, label='Signal-to-Noise Ratio')
            # Add colorbar ticks
            cbar.ax.tick_params(labelsize=8)
        
        # Set axis labels with physical units
        ax.set_xlabel('X (arcsec)')
        ax.set_ylabel('Y (arcsec)')
        
        # Set equal aspect ratio for proper spatial representation
        ax.set_aspect('equal')
        
        # Add grid for reference
        ax.grid(True, alpha=0.3)
        
        # Add title
        ax.set_title(f'Voronoi Binning Map - {galaxy_name}')
        
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
            
        # Create bin spatial coverage
        if n_pixels is not None and pixel_size_x > 0:
            # Calculate physical area of each bin in square arcseconds
            bin_areas = n_pixels * (pixel_size_x * pixel_size_y)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot histogram of bin areas
            ax.hist(bin_areas, bins=20, color='salmon', edgecolor='black')
            
            # Add median area line
            median_area = np.median(bin_areas)
            ax.axvline(x=median_area, color='red', linestyle='-', 
                      label=f'Median Area = {median_area:.2f} arcsec²')
            
            ax.set_title(f'Bin Area Distribution - {galaxy_name}')
            ax.set_xlabel('Area (arcsec²)')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            # Save figure
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_vnb_bin_area.png", dpi=150)
            plt.close(fig)
    
    except Exception as e:
        logger.error(f"Error creating binning plots: {str(e)}")
        logger.error(traceback.format_exc())
    """
    Create basic binning visualization plots with real physical coordinates
    
    Parameters
    ----------
    binned_data : BinnedSpectra
        Binned data object
    plots_dir : Path
        Directory to save plots
    galaxy_name : str
        Galaxy name
    """
    try:
        # Get pixel scale for coordinate conversion
        pixel_size_x = binned_data.metadata.get('pixelsize_x', 1.0)
        pixel_size_y = binned_data.metadata.get('pixelsize_y', 1.0)
        
        # Get bin centers and SNR values
        x_gen = binned_data.metadata.get('x_gen', None)
        y_gen = binned_data.metadata.get('y_gen', None)
        sn = binned_data.metadata.get('sn', None)
        
        # Create bin map
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
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create physical coordinate grid for display
        y_coords, x_coords = np.indices((n_y, n_x))
        
        # Convert to physical units (arcseconds)
        # Center the coordinates (center at 0,0)
        x_physical = (x_coords - n_x/2) * pixel_size_x
        y_physical = (y_coords - n_y/2) * pixel_size_y
        
        # Create colored bin map
        n_bins = len(unique_bins)
        cmap = plt.cm.get_cmap('tab20', n_bins)
        
        # Plot each bin with proper physical coordinates
        for i in unique_bins:
            mask = bin_map == i
            if np.any(mask):
                color = cmap(i % 20)  # Cycle through colors if more than 20 bins
                # Use pcolormesh for exact coordinate mapping
                ax.pcolormesh(
                    x_physical[mask], 
                    y_physical[mask], 
                    np.ones_like(x_physical[mask]),
                    color=color, edgecolor='k', linewidth=0.1
                )
        
        # Plot bin centers with bin numbers
        if x_gen is not None and y_gen is not None:
            # Convert bin centers to physical coordinates
            x_gen_physical = (x_gen - n_x/2) * pixel_size_x
            y_gen_physical = (y_gen - n_y/2) * pixel_size_y
            
            for i, (x, y) in enumerate(zip(x_gen_physical, y_gen_physical)):
                ax.text(x, y, str(i), color='black', fontsize=8, 
                       ha='center', va='center', backgroundcolor='white')
        
        # Add colorbar for SNR if available
        if sn is not None and x_gen is not None and y_gen is not None:
            sc = ax.scatter(x_gen_physical, y_gen_physical, c=sn, 
                          cmap='viridis', s=30, alpha=0.7)
            cbar = plt.colorbar(sc, ax=ax, label='Signal-to-Noise Ratio')
            # Add colorbar ticks
            cbar.ax.tick_params(labelsize=8)
        
        # Set axis labels with physical units
        ax.set_xlabel('X (arcsec)')
        ax.set_ylabel('Y (arcsec)')
        
        # Set equal aspect ratio for proper spatial representation
        ax.set_aspect('equal')
        
        # Add grid for reference
        ax.grid(True, alpha=0.3)
        
        # Add title
        ax.set_title(f'Voronoi Binning Map - {galaxy_name}')
        
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
            
        # Create bin spatial coverage
        if n_pixels is not None and pixel_size_x > 0:
            # Calculate physical area of each bin in square arcseconds
            bin_areas = n_pixels * (pixel_size_x * pixel_size_y)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot histogram of bin areas
            ax.hist(bin_areas, bins=20, color='salmon', edgecolor='black')
            
            # Add median area line
            median_area = np.median(bin_areas)
            ax.axvline(x=median_area, color='red', linestyle='-', 
                      label=f'Median Area = {median_area:.2f} arcsec²')
            
            ax.set_title(f'Bin Area Distribution - {galaxy_name}')
            ax.set_xlabel('Area (arcsec²)')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            # Save figure
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_vnb_bin_area.png", dpi=150)
            plt.close(fig)
    
    except Exception as e:
        logger.error(f"Error creating binning plots: {str(e)}")
        logger.error(traceback.format_exc())



def create_robust_binning_plots(binned_data, plots_dir, galaxy_name):
    """
    Create binning visualization plots avoiding pcolormesh and dimension issues
    
    Parameters
    ----------
    binned_data : BinnedSpectra
        Binned data object
    plots_dir : Path
        Directory to save plots
    galaxy_name : str
        Galaxy name
    """
    try:
        # Get pixel scale for coordinate conversion
        pixel_size_x = binned_data.metadata.get('pixelsize_x', 1.0)
        pixel_size_y = binned_data.metadata.get('pixelsize_y', 1.0)
        
        # Get bin centers and SNR values
        x_gen = binned_data.metadata.get('x_gen', None)
        y_gen = binned_data.metadata.get('y_gen', None)
        sn = binned_data.metadata.get('sn', None)
        
        # Get dimensions from metadata
        n_y = binned_data.metadata.get('ny', 1)
        n_x = binned_data.metadata.get('nx', 1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a mapping from bin numbers to pixels
        x_coords = []
        y_coords = []
        bins = []
        
        # Check if we can reshape bin_num to 2D
        try:
            bin_map = binned_data.bin_num.reshape(n_y, n_x)
            # Convert to physical coordinates using meshgrid
            y_indices, x_indices = np.indices((n_y, n_x))
            x_phys = (x_indices - n_x/2) * pixel_size_x
            y_phys = (y_indices - n_y/2) * pixel_size_y
            
            # Find all valid bins
            for bin_id in range(np.max(bin_map) + 1):
                mask = bin_map == bin_id
                if np.any(mask):
                    y_idx, x_idx = np.where(mask)
                    for i in range(len(y_idx)):
                        y_coords.append(y_phys[y_idx[i], x_idx[i]])
                        x_coords.append(x_phys[y_idx[i], x_idx[i]])
                        bins.append(bin_id)
        except:
            # If reshaping fails, try to get pixels from bin_indices
            if hasattr(binned_data, 'bin_indices') and binned_data.bin_indices:
                for bin_id, indices in enumerate(binned_data.bin_indices):
                    for idx in indices:
                        # Convert linear index to 2D coordinates
                        y = idx // n_x
                        x = idx % n_x
                        if 0 <= y < n_y and 0 <= x < n_x:
                            # Convert to physical coordinates
                            x_phys = (x - n_x/2) * pixel_size_x
                            y_phys = (y - n_y/2) * pixel_size_y
                            x_coords.append(x_phys)
                            y_coords.append(y_phys)
                            bins.append(bin_id)
            else:
                # No binning information, plot a placeholder
                ax.text(0.5, 0.5, "No valid binning data available", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Binning Map - {galaxy_name}')
                
                # Save figure and return
                plt.tight_layout()
                plt.savefig(plots_dir / f"{galaxy_name}_binning_map.png", dpi=150)
                plt.close(fig)
                return
        
        # Convert lists to arrays
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        bins = np.array(bins)
        
        # Check if we have any valid bins
        if len(bins) == 0:
            ax.text(0.5, 0.5, "No valid binning data available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Binning Map - {galaxy_name}')
                
            # Save figure and return
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_binning_map.png", dpi=150)
            plt.close(fig)
            return
        
        # Use scatter plot instead of pcolormesh
        cmap = plt.cm.get_cmap('tab20', max(20, np.max(bins)+1))
        unique_bins = np.unique(bins)
        
        # Plot each bin with a separate call to scatter for proper legend
        for bin_id in unique_bins:
            mask = bins == bin_id
            ax.scatter(x_coords[mask], y_coords[mask], 
                     color=cmap(bin_id % 20), s=15, alpha=0.7,
                     label=f'Bin {bin_id}' if bin_id < 5 else None)
        
        # Add bin centers if available
        if x_gen is not None and y_gen is not None:
            # Convert to physical coordinates
            x_gen_phys = [(x - n_x/2) * pixel_size_x for x in x_gen]
            y_gen_phys = [(y - n_y/2) * pixel_size_y for y in y_gen]
            
            # Plot bin centers with numbers
            for i, (x, y) in enumerate(zip(x_gen_phys, y_gen_phys)):
                if i < len(unique_bins):
                    ax.plot(x, y, 'ko', markersize=8)
                    ax.text(x, y, str(i), color='white', fontsize=8, 
                           ha='center', va='center',
                           bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.1'))
        
        # Add SNR values if available
        if sn is not None and x_gen is not None and y_gen is not None:
            # Create a separate scatter plot for SNR
            try:
                limit = min(len(x_gen_phys), len(y_gen_phys), len(sn))
                sc = ax.scatter(
                    x_gen_phys[:limit], 
                    y_gen_phys[:limit], 
                    c=sn[:limit], 
                    cmap='viridis', 
                    s=30, 
                    alpha=0.5,
                    edgecolor='k'
                )
                plt.colorbar(sc, ax=ax, label='Signal-to-Noise Ratio')
            except Exception as e:
                logger.warning(f"Error adding SNR colorbar: {e}")
        
        # Set labels and grid
        ax.set_xlabel('X (arcsec)')
        ax.set_ylabel('Y (arcsec)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Binning Map - {galaxy_name}')
        
        # Add legend for first few bins
        if len(unique_bins) > 0:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles[:min(5, len(handles))],
                    labels[:min(5, len(labels))],
                    loc='upper right', 
                    fontsize='small'
                )
        
        # Save figure
        plt.tight_layout()
        plt.savefig(plots_dir / f"{galaxy_name}_binning_map.png", dpi=150)
        plt.close(fig)
        
        # Create histograms for bin properties
        # SNR histogram
        if sn is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(sn, bins=20, color='skyblue', edgecolor='black')
            
            # Add target SNR line if available
            target_snr = binned_data.metadata.get('target_snr', None)
            if target_snr is not None:
                ax.axvline(x=target_snr, color='red', linestyle='--', 
                          label=f'Target SNR = {target_snr:.1f}')
            
            # Add median SNR line
            median_snr = np.nanmedian(sn)
            ax.axvline(x=median_snr, color='green', linestyle='-', 
                      label=f'Median SNR = {median_snr:.1f}')
            
            ax.set_title(f'Bin SNR Distribution - {galaxy_name}')
            ax.set_xlabel('Signal-to-Noise Ratio')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_snr_histogram.png", dpi=150)
            plt.close(fig)
        
        # Bin size histogram
        n_pixels = binned_data.metadata.get('n_pixels', None)
        if n_pixels is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(n_pixels, bins=20, color='lightgreen', edgecolor='black')
            
            median_pixels = np.nanmedian(n_pixels)
            ax.axvline(x=median_pixels, color='red', linestyle='-', 
                      label=f'Median Size = {median_pixels:.1f} pixels')
            
            ax.set_title(f'Bin Size Distribution - {galaxy_name}')
            ax.set_xlabel('Number of Pixels per Bin')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_binsize_histogram.png", dpi=150)
            plt.close(fig)
        
        # Bin area in physical units
        if n_pixels is not None and pixel_size_x > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            bin_areas = n_pixels * (pixel_size_x * pixel_size_y)  # arcsec²
            ax.hist(bin_areas, bins=20, color='salmon', edgecolor='black')
            
            median_area = np.nanmedian(bin_areas)
            ax.axvline(x=median_area, color='red', linestyle='-', 
                      label=f'Median Area = {median_area:.2f} arcsec²')
            
            ax.set_title(f'Bin Area Distribution - {galaxy_name}')
            ax.set_xlabel('Area (arcsec²)')
            ax.set_ylabel('Number of Bins')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_binsize_histogram.png", dpi=150)
            plt.close(fig)
            
    except Exception as e:
        logger.error(f"Error creating robust binning plots: {str(e)}")
        logger.error(traceback.format_exc())