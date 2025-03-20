"""
Spectral Binning Tools - Support for Voronoi binning and radial binning
"""
import numpy as np
import warnings
import logging
from typing import Tuple, Dict, Optional, Union, List, Any
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

logger = logging.getLogger(__name__)

# Constants
C_KMS = 299792.458  # Speed of light in km/s

def make_bins(wavs):
    """Given wavelength points, find edges and widths of wavelength bins."""
    edges = np.zeros(wavs.shape[0]+1)
    widths = np.zeros(wavs.shape[0])
    edges[0] = wavs[0] - (wavs[1] - wavs[0])/2
    widths[-1] = (wavs[-1] - wavs[-2])
    edges[-1] = wavs[-1] + (wavs[-1] - wavs[-2])/2
    edges[1:-1] = (wavs[1:] + wavs[:-1])/2
    widths[:-1] = edges[1:-1] - edges[:-2]
    return edges, widths

def spectres(new_wavs, spec_wavs, spec_fluxes, spec_errs=None, fill=None, verbose=True):
    """
    Resamples spectra (and optionally associated uncertainties) onto a new wavelength basis.
    
    Parameters
    ----------
    new_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the spectrum or spectra.
    spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the spectrum or spectra.
    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in spec_wavs,
        last dimension must correspond to the shape of spec_wavs.
    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties.
    fill : float (optional)
        Value to use outside the original wavelength range.
    verbose : bool (optional)
        Whether to show warnings.

    Returns
    -------
    new_fluxes : numpy.ndarray
        Array of resampled flux values.
    new_errs : numpy.ndarray (optional)
        Array of uncertainties associated with fluxes in new_fluxes.
    """
    # Implementation borrowed from spectres package
    old_wavs = spec_wavs
    old_fluxes = spec_fluxes
    old_errs = spec_errs

    # Make arrays of edge positions and widths for the old and new bins
    old_edges, old_widths = make_bins(old_wavs)
    new_edges, new_widths = make_bins(new_wavs)

    # Generate output arrays to be populated
    new_fluxes = np.zeros(old_fluxes[..., 0].shape + new_wavs.shape)

    if old_errs is not None:
        if old_errs.shape != old_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape as spec_fluxes.")
        else:
            new_errs = np.copy(new_fluxes)

    start = 0
    stop = 0

    # Calculate new flux and uncertainty values, looping over new bins
    for j in range(new_wavs.shape[0]):
        # Add filler values if new_wavs extends outside of spec_wavs
        if (new_edges[j] < old_edges[0]) or (new_edges[j+1] > old_edges[-1]):
            new_fluxes[..., j] = fill

            if spec_errs is not None:
                new_errs[..., j] = fill
            continue

        # Find first old bin which is partially covered by the new bin
        while old_edges[start+1] <= new_edges[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin
        while old_edges[stop+1] < new_edges[j+1]:
            stop += 1

        # If new bin is fully inside an old bin, use the old bin value
        if stop == start:
            new_fluxes[..., j] = old_fluxes[..., start]
            if old_errs is not None:
                new_errs[..., j] = old_errs[..., start]
        else:
            # Calculate proportional overlap with first and last old bins
            start_factor = ((old_edges[start+1] - new_edges[j]) / 
                           (old_edges[start+1] - old_edges[start]))
            end_factor = ((new_edges[j+1] - old_edges[stop]) / 
                         (old_edges[stop+1] - old_edges[stop]))

            # Adjust old bin widths by overlap factor
            old_widths[start] *= start_factor
            old_widths[stop] *= end_factor

            # Populate new_fluxes spectrum and uncertainty arrays
            f_widths = old_widths[start:stop+1] * old_fluxes[..., start:stop+1]
            new_fluxes[..., j] = np.sum(f_widths, axis=-1)
            new_fluxes[..., j] /= np.sum(old_widths[start:stop+1])

            if old_errs is not None:
                e_wid = old_widths[start:stop+1] * old_errs[..., start:stop+1]
                new_errs[..., j] = np.sqrt(np.sum(e_wid**2, axis=-1))
                new_errs[..., j] /= np.sum(old_widths[start:stop+1])

            # Restore the old bin widths
            old_widths[start] /= start_factor
            old_widths[stop] /= end_factor

    # Return both flux and error if errors were provided
    if old_errs is not None:
        return new_fluxes, new_errs
    else:
        return new_fluxes

def apply_velocity_shift(spectrum, wavelength, velocity):
    """
    Apply a velocity shift to a spectrum.
    
    Parameters
    ----------
    spectrum : numpy.ndarray
        Spectrum flux values.
    wavelength : numpy.ndarray
        Wavelength array.
    velocity : float
        Velocity shift in km/s.
        
    Returns
    -------
    numpy.ndarray
        Velocity-shifted spectrum.
    """
    shifted_wavelength = wavelength / (1 + velocity/C_KMS)
    return spectres(wavelength, shifted_wavelength, spectrum, fill=0.0)

def run_voronoi_binning(x, y, signal, noise, target_snr, plot=False, quiet=True, cvt=True, min_snr=None):
    """
    Run Voronoi binning algorithm with enhanced error handling.
    
    Parameters
    ----------
    x, y : array_like
        Coordinates of the pixels.
    signal : array_like
        Signal at each pixel.
    noise : array_like
        Noise at each pixel.
    target_snr : float
        Target signal-to-noise ratio for the bins.
    plot : bool, optional
        Whether to plot the bins.
    quiet : bool, optional
        Whether to suppress information.
    cvt : bool, optional
        Whether to use centroidal Voronoi tessellation.
    min_snr : float, optional
        Minimum SNR to try if original target fails. Defaults to target_snr/2.
        
    Returns
    -------
    tuple
        (bin_num, x_gen, y_gen, sn, n_pixels, scale)
    """
    import logging
    import numpy as np
    logger = logging.getLogger(__name__)
    
    # 处理输入数据，确保没有无效值
    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(signal) & np.isfinite(noise) & (noise > 0)
    if np.sum(valid_mask) < 10:  # 需要至少10个有效点
        logger.error(f"Too few valid data points for Voronoi binning: {np.sum(valid_mask)}")
        # 创建一个简单的单bin解决方案
        bin_num = np.zeros_like(x, dtype=int)
        x_gen = np.array([np.mean(x[valid_mask])] if np.any(valid_mask) else [0])
        y_gen = np.array([np.mean(y[valid_mask])] if np.any(valid_mask) else [0])
        sn = np.array([np.mean(signal[valid_mask]/noise[valid_mask])] if np.any(valid_mask) else [1])
        n_pixels = np.array([np.sum(valid_mask)])
        scale = 1.0
        logger.warning("Created a single bin as fallback due to insufficient valid data")
        return bin_num, x_gen, y_gen, sn, n_pixels, scale
    
    # 使用有效点
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    signal_valid = signal[valid_mask]
    noise_valid = noise[valid_mask]
    
    # 如果没有指定最小SNR，默认为目标值的一半
    if min_snr is None:
        min_snr = target_snr / 2
    
    # 逐步降低SNR，直到成功
    current_snr = target_snr
    success = False
    
    while not success and current_snr >= min_snr:
        try:
            from vorbin.voronoi_2d_binning import voronoi_2d_binning
            result = voronoi_2d_binning(
                x_valid, y_valid, signal_valid, noise_valid, current_snr, 
                plot=plot, quiet=quiet, cvt=cvt
            )
            
            # 检查结果
            if isinstance(result, tuple):
                if len(result) >= 6:
                    # 获取有效结果
                    valid_bin_num, x_gen, y_gen, sn, n_pixels, scale = result[:6]
                    success = True
                else:
                    # 结果不完整
                    logger.warning(f"Voronoi binning returned insufficient values with SNR={current_snr}")
                    current_snr *= 0.8  # 降低SNR并重试
            else:
                # 单一返回值（可能只是bin_num）
                valid_bin_num = result
                # 创建合理的默认值
                unique_bins = np.unique(valid_bin_num)
                if len(unique_bins) > 0:  # 确保有bin
                    x_gen = np.zeros(len(unique_bins))
                    y_gen = np.zeros(len(unique_bins))
                    sn = np.ones(len(unique_bins)) * current_snr
                    n_pixels = np.ones(len(unique_bins))
                    
                    # 计算bin中心和统计信息
                    for i, bin_id in enumerate(unique_bins):
                        mask = valid_bin_num == bin_id
                        if np.any(mask):
                            x_gen[i] = np.mean(x_valid[mask])
                            y_gen[i] = np.mean(y_valid[mask])
                            n_pixels[i] = np.sum(mask)
                    
                    scale = 1.0
                    success = True
                else:
                    # 没有有效bin，降低SNR并重试
                    logger.warning(f"No valid bins created with SNR={current_snr}")
                    current_snr *= 0.8
        
        except Exception as e:
            logger.warning(f"Voronoi binning failed with SNR={current_snr}: {str(e)}")
            current_snr *= 0.8  # 降低SNR并重试
    
    # 如果所有尝试都失败，创建单一bin作为后备方案
    if not success:
        logger.warning(f"All Voronoi binning attempts failed, creating a single bin as fallback")
        valid_bin_num = np.zeros_like(x_valid, dtype=int)
        x_gen = np.array([np.mean(x_valid)])
        y_gen = np.array([np.mean(y_valid)])
        sn = np.array([np.mean(signal_valid/noise_valid)])
        n_pixels = np.array([len(x_valid)])
        scale = 1.0
    
    # 将bin编号映射回原始数组
    bin_num = -np.ones_like(x, dtype=int)  # 初始化为-1（无效值）
    bin_num[valid_mask] = valid_bin_num
    
    if current_snr < target_snr:
        logger.info(f"Voronoi binning succeeded with reduced SNR={current_snr} (target was {target_snr})")
    
    return bin_num, x_gen, y_gen, sn, n_pixels, scale

def calculate_radial_bins(x, y, center_x=0, center_y=0, pa=0, ellipticity=0, 
                         n_rings=10, log_spacing=False):
    """
    Calculate radial bins for a set of coordinates.
    
    Parameters
    ----------
    x, y : array_like
        Coordinates of the pixels.
    center_x, center_y : float, optional
        Center coordinates.
    pa : float, optional
        Position angle in degrees.
    ellipticity : float, optional
        Ellipticity (0-1).
    n_rings : int, optional
        Number of rings.
    log_spacing : bool, optional
        Whether to use logarithmic spacing.
        
    Returns
    -------
    tuple
        (bin_num, bin_edges, bin_radii)
    """
    # Convert PA to radians
    pa_rad = np.radians(pa)
    
    # Calculate relative coordinates
    x_rel = x - center_x
    y_rel = y - center_y
    
    # Rotate coordinates to align with position angle
    x_rot = x_rel * np.cos(pa_rad) + y_rel * np.sin(pa_rad)
    y_rot = -x_rel * np.sin(pa_rad) + y_rel * np.cos(pa_rad)
    
    # Apply ellipticity (convert to semi-major axis)
    if ellipticity > 0:
        a = np.sqrt(x_rot**2 + (y_rot / (1 - ellipticity))**2)
    else:
        a = np.sqrt(x_rot**2 + y_rot**2)
    
    # Determine bin edges
    max_radius = np.max(a)
    min_radius = np.min(a) if np.min(a) > 0 else max_radius / (n_rings * 10)  # Avoid zero
    
    if log_spacing:
        # Logarithmic spacing (better for galaxy centers)
        bin_edges = np.logspace(np.log10(min_radius), np.log10(max_radius), n_rings+1)
    else:
        # Linear spacing
        bin_edges = np.linspace(0, max_radius, n_rings+1)
    
    # Assign each pixel to a bin
    bin_num = np.zeros_like(a, dtype=int)
    bin_radii = []
    
    for i in range(n_rings):
        if i == 0:
            # First bin includes everything less than the first edge
            mask = (a < bin_edges[1])
        elif i == n_rings - 1:
            # Last bin includes everything greater than or equal to the last edge
            mask = (a >= bin_edges[i])
        else:
            # Middle bins include everything between edges
            mask = (a >= bin_edges[i]) & (a < bin_edges[i+1])
        
        bin_num[mask] = i
        if np.any(mask):
            bin_radii.append(np.mean(a[mask]))
        else:
            bin_radii.append((bin_edges[i] + bin_edges[i+1]) / 2)
    
    return bin_num, bin_edges, np.array(bin_radii)

class BinnedSpectra:
    """
    Class to hold binned spectra and related data.
    
    Attributes
    ----------
    bin_type : str
        Type of binning ('voronoi' or 'radial')
    bin_num : numpy.ndarray
        Bin assignment for each spaxel
    bin_indices : list
        List of arrays containing indices of spaxels in each bin
    spectra : numpy.ndarray
        Binned spectra, shape (n_wavelength, n_bins)
    wavelength : numpy.ndarray
        Wavelength array
    metadata : dict
        Additional metadata
    """
    
    def __init__(self, bin_type, bin_num, bin_indices, spectra, wavelength, metadata=None):
        """Initialize with binned data."""
        self.bin_type = bin_type
        self.bin_num = bin_num
        self.bin_indices = bin_indices
        self.spectra = spectra
        self.wavelength = wavelength
        self.metadata = metadata if metadata is not None else {}
        
        # Number of bins
        self.n_bins = len(bin_indices)
    
    def to_p2p_compatible(self, cube):
        """
        Convert to P2P-compatible format.
        
        Parameters
        ----------
        cube : MUSECube
            Original data cube
            
        Returns
        -------
        dict
            Dictionary containing pseudo-cube data that can be processed by P2P functions
        """
        # Create a fake cube with shape (n_wavelength, 1, n_bins)
        fake_cube = np.zeros((self.spectra.shape[0], 1, self.spectra.shape[1]))
        fake_cube[:, 0, :] = self.spectra
        
        # Create fake cube variance (constant for now)
        fake_variance = np.ones_like(fake_cube)
        
        # Create fake SNR map
        snr_map = np.ones((1, self.n_bins))
        if 'sn' in self.metadata:
            snr_map[0, :] = self.metadata['sn']
        
        # Create class with necessary attributes for P2P functions
        class PseudoCube:
            def __init__(self, orig_cube, fake_cube_data, spectra_obj):
                # Don't call parent initializer
                self._initialized = True
                
                # Copy needed attributes from original cube
                self._redshift = orig_cube._redshift
                self._wvl_air_angstrom_range = orig_cube._wvl_air_angstrom_range
                
                # Copy FWHM and pixel size attributes
                self._FWHM_gal = orig_cube._FWHM_gal if hasattr(orig_cube, '_FWHM_gal') else 1.0
                self._pxl_size_x = orig_cube._pxl_size_x if hasattr(orig_cube, '_pxl_size_x') else 0.2
                self._pxl_size_y = orig_cube._pxl_size_y if hasattr(orig_cube, '_pxl_size_y') else 0.2
                
                # Copy goodwavelength if available
                if hasattr(orig_cube, '_goodwavelength'):
                    self._goodwavelength = orig_cube._goodwavelength
                
                # Set data from fake cube using binned spectra's wavelength
                self.cube = fake_cube_data
                self.cubevar = fake_variance
                self.wave = spectra_obj.wavelength   # 使用BinnedSpectra的波长（已截取）
                self._lambda_gal = spectra_obj.wavelength  # 使用BinnedSpectra的波长（已截取）
                
                # Set dimensions
                self._n_y = self.cube.shape[1]
                self._n_x = self.cube.shape[2]
                
                # Setup for ppxf
                self._spectra = spectra_obj.spectra
                self._spectra_2d = spectra_obj.spectra  # Flattened version
                self._ln_lambda_gal = np.log(self.wave)
                
                # Calculate velocity scale
                c = 299792.458
                dlambda = np.min(np.diff(self.wave))
                self.velscale = c * dlambda / self.wave[0]
                self._vel_scale = self.velscale
                
                # These are needed when running ppxf
                self._n_wave_fit = len(self.wave)
                
                # Keep bin information
                self.bin_num = spectra_obj.bin_num
                self.bin_indices = spectra_obj.bin_indices
                self.bin_metadata = spectra_obj.metadata
                
                # Initialize result fields exactly as in MUSECube
                self._velocity_field = np.full((self._n_y, self._n_x), np.nan)
                self._dispersion_field = np.full((self._n_y, self._n_x), np.nan)
                self._bestfit_field = np.full((self._n_wave_fit, self._n_y, self._n_x), np.nan)
                self._optimal_tmpls = None
                self._template_weights = None
                self._poly_coeffs = None
                self._sps = None
                
                # For emission lines
                self._emission_flux = {}
                self._emission_vel = {}
                self._emission_sig = {}
                self._gas_bestfit_field = None
                self._emission_wavelength = {}
                
                # For spectral indices
                self._spectral_indices = {}
                self._index_calculators = {}
                
                # Save original cube reference for method calls
                self._orig_cube = orig_cube
                
            def fit_spectra(self, template_filename, ppxf_vel_init=0, ppxf_vel_disp_init=40, ppxf_deg=3, n_jobs=-1):
                """
                Fit the stellar continuum using pPXF.
                
                This version adapts the method to work with binned spectra. It processes each bin
                instead of each spaxel, using the mapping from bins to spaxels.
                
                Parameters
                ----------
                template_filename : str
                    Filename of the stellar template
                ppxf_vel_init : int, default=0
                    Initial guess for the velocity in pPXF
                ppxf_vel_disp_init : int, default=40
                    Initial guess for the velocity dispersion in pPXF
                ppxf_deg : int, default=3
                    Degree of the additive polynomial for pPXF
                n_jobs : int, default=-1
                    Number of parallel jobs to run (-1 means using all processors)
                    
                Returns
                -------
                tuple
                    (velocity_field, dispersion_field, bestfit_field, optimal_templates, polynomial_coefficients)
                """
                import os
                import logging
                import warnings
                import numpy as np
                from ppxf import ppxf_util
                from ppxf.ppxf import ppxf
                from ppxf.sps_util import sps_lib
                from joblib import Parallel, delayed
                
                logger = logging.getLogger(__name__)
                
                try:
                    # Check if template file exists
                    if not os.path.exists(template_filename):
                        raise FileNotFoundError(f"Template file not found: {template_filename}")
                    
                    # Load template
                    sps = sps_lib(
                        filename=template_filename,
                        velscale=self._vel_scale,
                        fwhm_gal=None,
                        norm_range=self._wvl_air_angstrom_range
                    )
                    self._sps = sps  # Store SPS object for later reference
                    sps.templates = sps.templates.reshape(sps.templates.shape[0], -1)
                    
                    # Normalize stellar template
                    sps.templates /= np.median(sps.templates)
                    tmpl_mask = ppxf_util.determine_mask(
                        ln_lam=self._ln_lambda_gal,
                        lam_range_temp=np.exp(sps.ln_lam_temp[[0, -1]]),
                        width=1000
                    )
                    
                    # Initialize storage for templates and weights
                    n_templates = sps.templates.shape[1]
                    n_wave_fit = self._n_wave_fit  # Length of the rebinned wavelength array
                    n_wave_temp = sps.templates.shape[0]  # Length of the template wavelength array
                    
                    # Important: Initialize fields with correct dimensions
                    # For optimal templates, use the template wavelength grid
                    self._optimal_tmpls = np.full((n_wave_temp, self._n_y, self._n_x), np.nan)
                    self._template_weights = np.full((n_templates, self._n_y, self._n_x), np.nan)
                    self._poly_coeffs = []  # Store polynomial coefficients
                    
                    # For observed galaxy wavelength grid
                    self._bestfit_field = np.full((n_wave_fit, self._n_y, self._n_x), np.nan)
                    
                    # Process each bin instead of each spaxel
                    n_bins = self._spectra.shape[1]  # Number of bins
                    
                    def fit_bin(bin_idx):
                        """Fit a single bin spectrum"""
                        try:
                            # For pseudo-cube, we map bin_idx to (i=0, j=bin_idx)
                            i, j = 0, bin_idx
                            galaxy_data = self._spectra[:, bin_idx]
                            
                            # Use constant noise for now
                            galaxy_noise = np.ones_like(galaxy_data)
                            
                            # Skip low SNR or invalid bins
                            if np.count_nonzero(galaxy_data) < 50 or np.count_nonzero(np.isfinite(galaxy_data)) < 50:
                                return i, j, None
                            
                            # Replace NaN values to avoid problems in ppxf
                            if np.any(~np.isfinite(galaxy_data)):
                                galaxy_data = np.nan_to_num(galaxy_data, nan=0.0, posinf=0.0, neginf=0.0)
                            if np.any(~np.isfinite(galaxy_noise)):
                                galaxy_noise = np.nan_to_num(galaxy_noise, nan=1.0, posinf=1.0, neginf=1.0)
                            
                            with warnings.catch_warnings():
                                warnings.filterwarnings(
                                    'ignore', category=RuntimeWarning,
                                    message='invalid value encountered in scalar divide'
                                )
                                try:
                                    pp = ppxf(
                                        sps.templates, galaxy_data, galaxy_noise,
                                        self._vel_scale, mask=tmpl_mask,
                                        start=[ppxf_vel_init, ppxf_vel_disp_init], degree=ppxf_deg,
                                        lam=self._lambda_gal, lam_temp=sps.lam_temp,
                                        quiet=True
                                    )
                                    
                                    # Ensure dispersion value is reasonable
                                    if pp.sol[1] < 0:
                                        pp.sol[1] = 10.0  # Set to a reasonable minimum value
                                    
                                    # Calculate polynomial coefficients for later use
                                    poly_coeff = np.polyfit(self._lambda_gal, pp.apoly, ppxf_deg)
                                    
                                    # Calculate optimal template directly from weights on TEMPLATE wavelength grid
                                    optimal_template = sps.templates @ pp.weights
                                    
                                    # Calculate best-fit on GALAXY wavelength grid
                                    bestfit = pp.bestfit
                                    
                                    return i, j, (
                                        pp.sol[0], pp.sol[1], bestfit,
                                        optimal_template,
                                        pp.weights,
                                        poly_coeff
                                    )
                                except Exception as e:
                                    # If fitting fails, try again with a simpler configuration
                                    try:
                                        pp = ppxf(
                                            sps.templates, galaxy_data, galaxy_noise,
                                            self._vel_scale, mask=tmpl_mask,
                                            start=[ppxf_vel_init, ppxf_vel_disp_init], degree=0,  # Simplify to constant polynomial
                                            lam=self._lambda_gal, lam_temp=sps.lam_temp,
                                            quiet=True
                                        )
                                        
                                        # Ensure dispersion value is reasonable
                                        if pp.sol[1] < 0:
                                            pp.sol[1] = 10.0
                                        
                                        # Calculate polynomial coefficients (constant term)
                                        poly_coeff = np.array([pp.apoly[0]])
                                        
                                        # Calculate optimal template directly from weights on TEMPLATE wavelength grid
                                        optimal_template = sps.templates @ pp.weights
                                        
                                        # Calculate best-fit on GALAXY wavelength grid
                                        bestfit = pp.bestfit
                                        
                                        return i, j, (
                                            pp.sol[0], pp.sol[1], bestfit,
                                            optimal_template,
                                            pp.weights,
                                            poly_coeff
                                        )
                                    except Exception as e:
                                        # Both attempts failed, return None
                                        if bin_idx % 100 == 0:  # Reduce log clutter by only logging every 100th failure
                                            logger.debug(f"Fitting failed for bin {bin_idx}: {str(e)}")
                                        return i, j, None
                        except Exception as e:
                            logger.error(f"Error in fit_bin for bin {bin_idx}: {str(e)}")
                            return 0, bin_idx, None
                    
                    # Process all bins
                    # Use Progress or simple Parallel based on availability
                    try:
                        from utils.parallel import ParallelTqdm
                        fit_results = ParallelTqdm(
                            n_jobs=n_jobs, desc='Fitting binned spectra', total_tasks=n_bins
                        )(delayed(fit_bin)(idx) for idx in range(n_bins))
                    except ImportError:
                        # Fallback to standard Parallel
                        fit_results = Parallel(n_jobs=n_jobs)(delayed(fit_bin)(idx) for idx in range(n_bins))
                    
                    # Process results
                    for fit_result in fit_results:
                        if fit_result[2] is None:
                            continue
                        
                        row, col, (vel, disp, bestfit, optimal_tmpl, weights, poly_coeff) = fit_result
                        
                        # For pseudocube, we store results in a flattened way where bin index = column index
                        self._velocity_field[row, col] = vel
                        self._dispersion_field[row, col] = disp
                        
                        # Store best-fit on GALAXY wavelength grid
                        self._bestfit_field[:, row, col] = bestfit
                        
                        # Store optimal template on TEMPLATE wavelength grid
                        self._optimal_tmpls[:, row, col] = optimal_tmpl
                            
                        # Store template weights
                        self._template_weights[:len(weights), row, col] = weights
                            
                        self._poly_coeffs.append((row, col, poly_coeff))
                    
                    return (self._velocity_field, self._dispersion_field,
                            self._bestfit_field, self._optimal_tmpls, self._poly_coeffs)
                
                except Exception as e:
                    logger.error(f"Error in fit_spectra: {str(e)}")
                    return (np.full((self._n_y, self._n_x), np.nan),
                            np.full((self._n_y, self._n_x), np.nan),
                            np.full((n_wave_fit, self._n_y, self._n_x), np.nan),
                            np.full((n_wave_temp if 'n_wave_temp' in locals() else n_wave_fit, 
                                    self._n_y, self._n_x), np.nan),
                            [])
            
            def fit_emission_lines(self, template_filename, line_names=None, ppxf_vel_init=None,
                                ppxf_sig_init=50.0, ppxf_deg=8, n_jobs=-1, verbose=True):
                """
                Fit emission line components based on stellar template.
                Adapted for binned spectra in PseudoCube.
                
                Parameters
                ----------
                template_filename : str
                    Filename of the stellar template
                line_names : List[str], optional
                    List of emission lines to fit, defaults to all available lines
                ppxf_vel_init : np.ndarray, optional
                    Initial velocity field, defaults to stellar velocity field
                ppxf_sig_init : float, default=50.0
                    Initial velocity dispersion in km/s for gas
                ppxf_deg : int, default=8
                    Degree of additive polynomial for pPXF
                n_jobs : int, default=-1
                    Number of parallel jobs to run
                verbose : bool, default=True
                    Whether to print verbose output
                    
                Returns
                -------
                Dict[str, Any]
                    Dictionary containing emission line fitting results
                """
                import os
                import logging
                import warnings
                import numpy as np
                from ppxf import ppxf_util
                from ppxf.ppxf import ppxf
                from joblib import Parallel, delayed

                logger = logging.getLogger(__name__)
                # Set log level
                original_level = logger.level
                if not verbose:
                    logger.setLevel(logging.WARNING)
                else:
                    logger.setLevel(logging.INFO)
                
                try:    
                    # Check if stellar fitting has already been performed
                    if self._sps is None or self._optimal_tmpls is None:
                        raise ValueError("Must run fit_spectra() before fit_emission_lines()")
                    
                    if ppxf_vel_init is None:
                        # Use stellar velocity field as initial value
                        ppxf_vel_init = self._velocity_field
                    
                    # Initialize result storage
                    self._emission_flux = {}
                    self._emission_vel = {}
                    self._emission_sig = {}
                    self._gas_bestfit_field = np.full((self._n_wave_fit, self._n_y, self._n_x), np.nan)
                    self._emission_wavelength = {}
                    
                    # Generate emission line templates using ppxf's emission_lines function
                    lam_range_gal = [np.min(self._lambda_gal), np.max(self._lambda_gal)]
                    
                    from ppxf.ppxf_util import emission_lines
                    gas_templates, gas_names, line_wave = emission_lines(
                        self._sps.ln_lam_temp, lam_range_gal, self._FWHM_gal
                    )

                    # Set up gas components - using 1 gas kinematic component
                    ngas_comp = 1
                    gas_templates = np.tile(gas_templates, ngas_comp)
                    gas_names = np.asarray([a + f"_({p+1})" for p in range(ngas_comp) for a in gas_names])
                    line_wave = np.tile(line_wave, ngas_comp)
                    
                    # Filter emission lines if specific ones are requested
                    if line_names is not None:
                        valid_indices = []
                        for i, name in enumerate(gas_names):
                            base_name = name.split('_(')[0] if '_(' in name else name
                            if any(requested.lower() in base_name.lower() for requested in line_names):
                                valid_indices.append(i)
                        
                        if valid_indices:
                            gas_templates = gas_templates[:, valid_indices]
                            gas_names = [gas_names[i] for i in valid_indices]
                            line_wave = [line_wave[i] for i in valid_indices]
                    
                    # Store emission line wavelengths for reference
                    self._emission_wavelength = dict(zip(gas_names, line_wave))
                    logger.info("Emission lines included in gas templates:")
                    logger.info(gas_names)
                    
                    # Initialize emission line storage
                    for name in gas_names:
                        base_name = name.split('_(')[0] if '_(' in name else name
                        if base_name not in self._emission_flux:
                            self._emission_flux[base_name] = np.full((self._n_y, self._n_x), np.nan)
                            self._emission_vel[base_name] = np.full((self._n_y, self._n_x), np.nan)
                            self._emission_sig[base_name] = np.full((self._n_y, self._n_x), np.nan)
                    
                    # Store ppxf results for each bin
                    self._ppxf_gas_results = []
                    
                    # Process each bin
                    n_bins = self._spectra.shape[1]
                    
                    def fit_bin_emission(bin_idx):
                        """Fit emission lines for a single bin"""
                        try:
                            # For pseudo-cube, we map bin_idx to (i=0, j=bin_idx)
                            i, j = 0, bin_idx
                            
                            # Skip if first-time fitting failed
                            if np.isnan(self._velocity_field[i, j]):
                                return i, j, None
                            
                            # Get bin spectrum
                            galaxy_data = self._spectra[:, bin_idx]
                            
                            # Use constant noise for now
                            galaxy_noise = np.ones_like(galaxy_data)
                            
                            # Replace NaN values to avoid problems in ppxf
                            if np.any(~np.isfinite(galaxy_data)):
                                galaxy_data = np.nan_to_num(galaxy_data, nan=0.0, posinf=0.0, neginf=0.0)
                            if np.any(~np.isfinite(galaxy_noise)):
                                galaxy_noise = np.nan_to_num(galaxy_noise, nan=1.0, posinf=1.0, neginf=1.0)
                            
                            # Get optimal stellar template for this bin
                            optimal_template = self._optimal_tmpls[:, i, j]
                            
                            # Get initial velocity value
                            vel_init = self._velocity_field[i, j] if not np.isnan(self._velocity_field[i, j]) else 0
                            
                            try:
                                # Load SPS for this bin
                                from ppxf.sps_util import sps_lib
                                sps = sps_lib(
                                    filename=template_filename,
                                    velscale=self._vel_scale,
                                    fwhm_gal=None,
                                    norm_range=self._wvl_air_angstrom_range
                                )
                                
                                # Combine stellar and gas templates
                                stars_gas_templates = np.column_stack([optimal_template, gas_templates])
                                
                                # Define component types - [0] for stellar, [1] for gas components
                                component = [0] + [1]*2  # This indicates 2 gas components
                                gas_component = np.array(component) > 0  # True for gas components
                                
                                # Define moments for each component type
                                moments = [-2, 2]  # -2 for stellar (fixed dispersion), 2 for gas (full kinematics)
                                ncomp = len(moments)  # Should be 2
                                tied = [['', ''] for _ in range(ncomp)]
                                
                                # Set initial parameters
                                start = [
                                    [vel_init, self._dispersion_field[i, j]],  # Stellar initial kinematics
                                    [vel_init, ppxf_sig_init]                  # Gas initial kinematics
                                ]
                                
                                # Set boundary conditions
                                vlim = lambda x: vel_init + x*np.array([-100, 100])
                                bounds = [
                                    [vlim(2), [20, 300]],  # Stellar bounds
                                    [vlim(2), [20, 100]]   # Gas bounds
                                ]
                                
                                # Call ppxf with appropriate parameters and warning suppression
                                try:
                                    # Suppress warnings for division operations
                                    with warnings.catch_warnings():
                                        warnings.filterwarnings('ignore', category=RuntimeWarning, 
                                                            message='invalid value encountered in scalar divide')
                                        warnings.filterwarnings('ignore', category=RuntimeWarning, 
                                                            message='divide by zero encountered')
                                        
                                        # Ensure noise values are not zero to prevent division issues
                                        galaxy_noise = np.maximum(galaxy_noise, 1e-10)
                                        
                                        pp = ppxf(
                                            stars_gas_templates, galaxy_data, galaxy_noise, 
                                            self._vel_scale, start,
                                            moments=moments, degree=ppxf_deg, mdegree=-1,
                                            component=component, 
                                            gas_component=gas_component, 
                                            gas_names=gas_names, 
                                            lam=self._lambda_gal,
                                            lam_temp=sps.lam_temp,
                                            tied=tied,
                                            bounds=bounds,
                                            quiet=True
                                        )
                                        
                                    # Extract results
                                    # Calculate best-fit models for stellar and gas components
                                    bestfit = pp.bestfit if hasattr(pp, 'bestfit') else np.zeros_like(galaxy_data)
                                    
                                    # Extract gas best-fit component
                                    gas_bestfit = np.zeros_like(bestfit)
                                    if hasattr(pp, 'gas_bestfit'):
                                        gas_bestfit = pp.gas_bestfit
                                    elif hasattr(pp, 'component') and hasattr(pp, 'bestfit'):
                                        # Try to extract gas component from the full model
                                        comp = pp.component
                                        if len(comp) > 0:
                                            gas_idx = np.where(comp > 0)[0]
                                            if len(gas_idx) > 0:
                                                gas_bestfit = np.sum(pp.matrix[:, gas_idx] @ pp.weights[gas_idx], axis=1)
                                    
                                    # Calculate stellar component (total - gas)
                                    stellar_bestfit = bestfit - gas_bestfit
                                    
                                    # Extract polynomial coefficients
                                    if hasattr(pp, 'apoly'):
                                        apoly = pp.apoly
                                    
                                    # Properly calculate optimal template with polynomial
                                    apoly_se_2 = np.polyfit(self._lambda_gal, pp.apoly, 3)
                                    NEL_cal_tmp = (stars_gas_templates[:,0] * pp.weights[0]) + np.poly1d(apoly_se_2)(sps.lam_temp)
                                    
                                    # Get stellar and gas kinematic solutions
                                    stellar_sol = [pp.sol[0][0],pp.sol[0][1]] if hasattr(pp, 'sol') else [vel_init, self._dispersion_field[i, j]]
                                    
                                    # Get gas kinematics
                                    gas_sol = None
                                    if hasattr(pp, 'gas_kinematics'):
                                        gas_sol = pp.gas_kinematics[0]  # Take first gas component
                                    else:
                                        # Try to extract from sol
                                        if hasattr(pp, 'sol') and hasattr(pp, 'ncomp') and pp.ncomp > 1:
                                            gas_sol = [pp.sol[1][0],pp.sol[1][1]]  # Take the second component's solution
                                    
                                    # Store results
                                    result = {
                                        'flux': pp.gas_flux if hasattr(pp, 'gas_flux') else None,
                                        'gas_bestfit': gas_bestfit,
                                        'stellar_bestfit': stellar_bestfit,
                                        'total_bestfit': bestfit,
                                        'sol': stellar_sol,  # Stellar kinematics
                                        'gas_sol': gas_sol,  # Gas kinematics
                                        'weights': pp.weights if hasattr(pp, 'weights') else None,
                                        'NEL_cal_tmp': NEL_cal_tmp
                                    }
                                    return i, j, result
                                    
                                except Exception as e:
                                    if verbose and bin_idx % 100 == 0:  # Reduce log clutter
                                        logger.warning(f"Gas fitting failed for bin {bin_idx}: {str(e)}")
                                    return i, j, None
                            
                            except Exception as e:
                                if verbose and bin_idx % 100 == 0:  # Reduce log clutter
                                    logger.warning(f"Error in emission line fitting for bin {bin_idx}: {str(e)}")
                                return i, j, None 
                        except Exception as e:
                            logger.error(f"Error in fit_bin_emission for bin {bin_idx}: {str(e)}")
                            return 0, bin_idx, None
                    
                    # Process all bins
                    # Use Progress or simple Parallel based on availability
                    try:
                        from utils.parallel import ParallelTqdm
                        fit_results = ParallelTqdm(
                            n_jobs=n_jobs, desc='Fitting emission lines', total_tasks=n_bins
                        )(delayed(fit_bin_emission)(idx) for idx in range(n_bins))
                    except ImportError:
                        # Fallback to standard Parallel
                        fit_results = Parallel(n_jobs=n_jobs)(delayed(fit_bin_emission)(idx) for idx in range(n_bins))
                    
                    # Process results
                    for fit_result in fit_results:
                        if fit_result[2] is None:
                            continue
                        
                        row, col, result = fit_result
                        
                        # Save ppxf result for this bin
                        self._ppxf_gas_results.append((row, col, result))
                        
                        # Save gas fitting result
                        if 'gas_bestfit' in result and result['gas_bestfit'] is not None:
                            self._gas_bestfit_field[:, row, col] = result['gas_bestfit']
                        
                        # Update optimal template if needed
                        if 'NEL_cal_tmp' in result and result['NEL_cal_tmp'] is not None:
                            self._optimal_tmpls[:, row, col] = result['NEL_cal_tmp']
                        
                        # Update kinematics if needed
                        if 'sol' in result and result['sol'] is not None:
                            self._velocity_field[row, col] = result['sol'][0]
                            self._dispersion_field[row, col] = result['sol'][1]
                        
                        # Save emission line flux and velocity information
                        if 'flux' in result and result['flux'] is not None:
                            # Process emission line fluxes
                            for k, full_name in enumerate(gas_names):
                                if k < len(result['flux']):
                                    # Get base name without component number
                                    base_name = full_name.split('_(')[0] if '_(' in full_name else full_name
                                    
                                    # Store flux
                                    self._emission_flux[base_name][row, col] = result['flux'][k]
                                    
                                    # Store kinematics if available
                                    if 'gas_sol' in result and result['gas_sol'] is not None:
                                        self._emission_vel[base_name][row, col] = result['gas_sol'][0]
                                        self._emission_sig[base_name][row, col] = result['gas_sol'][1]
                    
                    # Restore original log level
                    logger.setLevel(original_level)
                    
                    # Return result dictionary
                    return {
                        'emission_flux': self._emission_flux,
                        'emission_vel': self._emission_vel,
                        'emission_sig': self._emission_sig,
                        'gas_bestfit_field': self._gas_bestfit_field,
                        'emission_wavelength': self._emission_wavelength,
                        'optimal_tmpls': self._optimal_tmpls,
                        'velocity_field': self._velocity_field,
                        'dispersion_field': self._dispersion_field,
                    }
                
                except Exception as e:
                    logger.error(f"Error in fit_emission_lines: {str(e)}")
                    logger.setLevel(original_level)
                    return {
                        'emission_flux': {},
                        'emission_vel': {},
                        'emission_sig': {},
                        'gas_bestfit_field': np.full((self._n_wave_fit, self._n_y, self._n_x), np.nan),
                        'emission_wavelength': {}
                    }
            
            def calculate_spectral_indices(self, indices_list=None, n_jobs=-1, verbose=False):
                """
                Calculate spectral indices for each bin using LineIndexCalculator
                
                Parameters
                ----------
                indices_list : list of str, optional
                    List of spectral indices to calculate, None uses standard set
                n_jobs : int, default=-1
                    Number of parallel jobs
                verbose : bool, default=False
                    Whether to display detailed information
                        
                Returns
                -------
                dict
                    Dictionary of spectral indices
                """
                import os
                import logging
                import warnings
                import numpy as np
                from joblib import Parallel, delayed

                logger = logging.getLogger(__name__)
                
                # Set log level
                original_level = logger.level
                if not verbose:
                    logger.setLevel(logging.WARNING)
                else:
                    logger.setLevel(logging.INFO)
                
                try:
                    # Import LineIndexCalculator
                    try:
                        from spectral_indices import LineIndexCalculator
                    except ImportError:
                        logger.error("spectral_indices module not found. Please make sure it's in your Python path.")
                        return {}
                    
                    # Define standard spectral indices if not provided
                    if indices_list is None:
                        indices_list = ['Hbeta', 'Fe5015', 'Mgb']
                    
                    # Get min/max wavelength from original data
                    orig_wave_min = np.min(self._lambda_gal)
                    orig_wave_max = np.max(self._lambda_gal)
                    logger.info(f"Original wavelength range: {orig_wave_min:.2f} - {orig_wave_max:.2f} Å")
                    
                    # Complete index definitions including blue and red bands
                    full_index_definitions = {
                        'Hbeta': {
                            'blue': (4827.875, 4847.875),
                            'band': (4847.875, 4876.625),
                            'red': (4876.625, 4891.625)
                        },
                        'Mgb': {
                            'blue': (5142.625, 5161.375),
                            'band': (5160.125, 5192.625),
                            'red': (5191.375, 5206.375)
                        },
                        'Fe5015': {
                            'blue': (4946.500, 4977.750),
                            'band': (4977.750, 5054.000),
                            'red': (5054.000, 5065.250)
                        },
                    }
                    
                    valid_indices = []
                    for index_name in indices_list:
                        if index_name in full_index_definitions:
                            windows = full_index_definitions[index_name]
                            # Check if index overlaps with data wavelength range
                            if (orig_wave_min <= windows['red'][1] and orig_wave_max >= windows['blue'][0]):
                                valid_indices.append(index_name)
                                logger.info(f"Index {index_name} is within wavelength range")
                            else:
                                logger.warning(f"Index {index_name} outside wavelength range")
                        else:
                            # Include other indices without checking range
                            valid_indices.append(index_name)
                    
                    if not valid_indices:
                        logger.warning("No valid spectral indices to calculate within wavelength range")
                        return {}
                    
                    # Update indices list
                    indices_list = valid_indices
                    
                    # Initialize spectral indices
                    self._spectral_indices = {index_name: np.full((self._n_y, self._n_x), np.nan) for index_name in indices_list}
                    
                    # Check if we have emission line fitting results
                    has_emission_lines = (self._gas_bestfit_field is not None and 
                                    np.any(~np.isnan(self._gas_bestfit_field)))
                    
                    # Store calculators for later plotting if needed
                    self._index_calculators = {}
                    
                    # Process each bin
                    n_bins = self._spectra.shape[1]
                    
                    def calculate_index(bin_idx):
                        """
                        Calculate spectral indices for a single bin
                        """
                        try:
                            # For pseudo-cube, we map bin_idx to (i=0, j=bin_idx)
                            i, j = 0, bin_idx
                            
                            # Skip if first-time fitting failed
                            if np.isnan(self._velocity_field[i, j]):
                                return i, j, {index_name: np.nan for index_name in indices_list}
                            
                            # Get bin spectrum
                            bin_spectrum = self._spectra[:, bin_idx]
                            
                            # Get optimal template
                            optimal_template = self._optimal_tmpls[:, i, j]
                            
                            # Get velocity
                            velocity = self._velocity_field[i, j]
                            
                            # Get gas model if available
                            gas_model = None
                            if has_emission_lines:
                                gas_model = self._gas_bestfit_field[:, i, j]
                                # Verify gas model has valid values
                                if not np.any(np.isfinite(gas_model)) or np.all(gas_model == 0):
                                    gas_model = None
                            
                            # Create LineIndexCalculator with warning suppression
                            with warnings.catch_warnings():
                                warnings.filterwarnings('ignore', category=RuntimeWarning, 
                                                    message='invalid value encountered in')
                                warnings.filterwarnings('ignore', category=RuntimeWarning, 
                                                    message='divide by zero')
                                
                                try:
                                    calculator = LineIndexCalculator(
                                        wave=self._lambda_gal,           # Observation wavelength grid
                                        flux=bin_spectrum,               # Observed spectrum
                                        fit_wave=self._sps.lam_temp,     # Template wavelength grid
                                        fit_flux=optimal_template,       # Template spectrum
                                        em_wave=self._lambda_gal if gas_model is not None else None,  # Emission line wavelength grid
                                        em_flux_list=gas_model,          # Emission line spectrum
                                        velocity_correction=velocity,    # Velocity correction
                                        continuum_mode='auto'            # Auto select continuum mode
                                    )
                                except Exception as e:
                                    logger.debug(f"Error creating LineIndexCalculator for bin {bin_idx}: {str(e)}")
                                    return i, j, {index_name: np.nan for index_name in indices_list}
                                
                                # Calculate all indices at once to minimize function calls
                                indices_values = {}
                                for index_name in indices_list:
                                    try:
                                        # Use calculate_index method that handles pre-processing
                                        index_value = calculator.calculate_index(index_name)
                                        indices_values[index_name] = index_value
                                    except Exception as e:
                                        indices_values[index_name] = np.nan
                                
                                # Store calculator for the first bin for later plotting
                                if bin_idx == 0:
                                    self._index_calculators['first_bin'] = calculator
                                
                                return i, j, indices_values
                        except Exception as e:
                            logger.error(f"Error in calculate_index for bin {bin_idx}: {str(e)}")
                            return 0, bin_idx, {index_name: np.nan for index_name in indices_list}
                    
                    # Process all bins
                    # Use Progress or simple Parallel based on availability
                    try:
                        from utils.parallel import ParallelTqdm
                        index_results = ParallelTqdm(
                            n_jobs=n_jobs, desc='Calculating spectral indices', total_tasks=n_bins
                        )(delayed(calculate_index)(idx) for idx in range(n_bins))
                    except ImportError:
                        # Fallback to standard Parallel
                        index_results = Parallel(n_jobs=n_jobs)(delayed(calculate_index)(idx) for idx in range(n_bins))
                    
                    # Process results
                    for result in index_results:
                        if result is None or result[2] is None:
                            continue
                            
                        row, col, indices_values = result
                        for index_name, value in indices_values.items():
                            if index_name in self._spectral_indices:
                                self._spectral_indices[index_name][row, col] = value
                    
                    # Restore original log level
                    logger.setLevel(original_level)
                    
                    return self._spectral_indices
                
                except Exception as e:
                    logger.error(f"Error in calculate_spectral_indices: {str(e)}")
                    logger.setLevel(original_level)
                    return {}
        
        return PseudoCube(cube, fake_cube, self)
    
    def to_plot_format(self):
        """
        将binned数据转换为绘图友好的格式
        
        Returns
        -------
        dict
            包含适用于绘图函数的数据
        """
        import numpy as np
        
        # 确保我们有距离信息
        has_distance = False
        distances = []
        
        # 从元数据中提取距离信息
        if 'distance' in self.metadata:
            has_distance = True
            distances = self.metadata['distance']
        elif 'bin_distances' in self.metadata:
            has_distance = True
            distances = self.metadata['bin_distances']
        elif 'radii' in self.metadata:
            has_distance = True
            distances = self.metadata['radii']
        
        # 如果没有找到现成的距离信息，可以创建简单的序列
        if not has_distance:
            distances = np.arange(self.n_bins)
        
        # 为绘图准备数据结构
        plot_data = {
            # 基本信息
            'bin_type': self.bin_type,
            'n_bins': self.n_bins,
            
            # 距离信息
            'distance': {
                'bin_distances': distances
            },
            
            # 恒星运动学参数 - 设置一个默认结构
            'stellar_kinematics': {
                'velocity': np.zeros(self.n_bins),
                'dispersion': np.zeros(self.n_bins)
            },
            
            # 发射线参数 - 设置一个默认空结构
            'emission': {},
            
            # 谱指数 - 设置一个默认空结构
            'indices': {}
        }
        
        # 从元数据填充恒星运动学
        if 'stellar_vel' in self.metadata:
            plot_data['stellar_kinematics']['velocity'] = self.metadata['stellar_vel']
        if 'stellar_disp' in self.metadata:
            plot_data['stellar_kinematics']['dispersion'] = self.metadata['stellar_disp']
        elif 'stellar_sigma' in self.metadata:
            plot_data['stellar_kinematics']['dispersion'] = self.metadata['stellar_sigma']
        
        # 填充发射线参数
        line_prefixes = ['flux_', 'em_vel_', 'em_sig_']
        for key in self.metadata:
            for prefix in line_prefixes:
                if key.startswith(prefix):
                    line_name = key[len(prefix):]
                    plot_data['emission'][key] = self.metadata[key]
        
        # 填充谱指数
        for key in self.metadata:
            if key.startswith('index_'):
                index_name = key[6:]
                plot_data['indices'][index_name] = self.metadata[key]
        
        # 添加恒星物理参数
        stellar_pop_keys = ['log_age', 'age', 'metallicity']
        has_stellar_pop = False
        stellar_pop = {}
        
        for key in stellar_pop_keys:
            if key in self.metadata:
                stellar_pop[key] = self.metadata[key]
                has_stellar_pop = True
        
        if has_stellar_pop:
            plot_data['stellar_population'] = stellar_pop
        
        return plot_data
    
    def create_visualization_plots(self, output_dir, galaxy_name):
        """
        为binned数据创建可视化图表
        
        Parameters
        ----------
        output_dir : Path
            输出目录
        galaxy_name : str
            星系名称
        """
        import logging
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        logger = logging.getLogger(__name__)
        
        try:
            # 创建plots目录
            plots_dir = output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True, parents=True)
            
            # 转换为绘图格式
            plot_data = self.to_plot_format()
            
            # 导入绘图模块
            # 注意：这里假设p2p模块已经导入
            try:
                from analysis.p2p import create_radial_profile_plots
                
                # 创建径向剖面图
                create_radial_profile_plots(
                    plot_data, 
                    plots_dir=plots_dir, 
                    galaxy_name=galaxy_name, 
                    analysis_type=self.bin_type.upper()
                )
                
                logger.info(f"Created radial profile plots in {plots_dir}")
            except ImportError:
                logger.warning("p2p module not available, falling back to basic visualization")
                
                # 创建基本径向剖面图
                if 'stellar_kinematics' in plot_data and 'distance' in plot_data:
                    # 创建速度剖面图
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(plot_data['distance']['bin_distances'], 
                          plot_data['stellar_kinematics']['velocity'], 
                          'o-', label='Velocity')
                    ax.set_xlabel('Radius (arcsec)')
                    ax.set_ylabel('Velocity (km/s)')
                    ax.set_title(f'{galaxy_name} Velocity Profile')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    fig.savefig(plots_dir / f"{galaxy_name}_{self.bin_type}_velocity_profile.png", dpi=150)
                    plt.close(fig)
                    
                    # 创建弥散剖面图
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(plot_data['distance']['bin_distances'], 
                          plot_data['stellar_kinematics']['dispersion'], 
                          'o-', label='Dispersion')
                    ax.set_xlabel('Radius (arcsec)')
                    ax.set_ylabel('Velocity Dispersion (km/s)')
                    ax.set_title(f'{galaxy_name} Velocity Dispersion Profile')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    fig.savefig(plots_dir / f"{galaxy_name}_{self.bin_type}_dispersion_profile.png", dpi=150)
                    plt.close(fig)
            
            # 绘制各个bin的光谱（选择一部分bin）
            try:
                n_samples = min(4, self.n_bins)  # 最多绘制4个bin
                sample_indices = [0]  # 始终包括第一个bin
                
                # 添加其他bin的索引（均匀分布）
                if self.n_bins > 1:
                    sample_indices.extend([
                        i * (self.n_bins - 1) // (n_samples - 1)
                        for i in range(1, n_samples)
                    ])
                
                for i, bin_idx in enumerate(sample_indices):
                    if bin_idx < self.spectra.shape[1]:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(self.wavelength, self.spectra[:, bin_idx], 'k-')
                        ax.set_xlabel('Wavelength (Å)')
                        ax.set_ylabel('Flux')
                        ax.set_title(f'Bin {bin_idx} Spectrum')
                        plt.tight_layout()
                        fig.savefig(plots_dir / f"{galaxy_name}_{self.bin_type}_bin_{bin_idx}_spectrum.png", dpi=150)
                        plt.close(fig)
                
                logger.info(f"Created spectrum plots for {len(sample_indices)} bins")
                
            except Exception as e:
                logger.error(f"Error creating bin spectrum plots: {e}")
                plt.close('all')
            
        except Exception as e:
            logger.error(f"Error creating visualization plots: {e}")
            plt.close('all')
    
    def save(self, filename):
        """
        保存binned数据到文件
        
        Parameters
        ----------
        filename : str 或 Path
            输出文件名
        """
        import pickle
        import os
        import logging
        import numpy as np
        from pathlib import Path
        logger = logging.getLogger(__name__)
        
        try:
            # 将filename转换为Path对象
            if not isinstance(filename, Path):
                filename = Path(filename)
            
            # 基本文件名（不带扩展名）
            base_filename = filename.stem
            
            # 创建pickle文件名（.pkl扩展名）
            pickle_filename = filename.parent / f"{base_filename}.pkl"
            
            # 创建一个字典，包含所有需要保存的数据
            save_data = {
                'wavelength': self.wavelength,
                'spectra': self.spectra,
                'bin_num': self.bin_num,
                'bin_indices': self.bin_indices,
                'bin_type': self.bin_type,
                'n_bins': self.n_bins,
                'metadata': self.metadata
            }
            
            # 使用pickle保存所有数据，包括不规则形状的数据
            with open(pickle_filename, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"Saved binned data to {pickle_filename}")
            
            # 尝试另存为legacy格式
            try:
                # 创建legacy目录
                legacy_dir = filename.parent / f"{base_filename}_data"
                legacy_dir.mkdir(exist_ok=True, parents=True)
                
                # 保存波长（一维数组，直接保存）
                np.save(legacy_dir / "wavelength.npy", self.wavelength)
                
                # 保存bin号（一维数组，直接保存）
                np.save(legacy_dir / "bin_num.npy", self.bin_num)
                
                # 逐个保存每个bin的光谱
                for i in range(self.n_bins):
                    spec_file = legacy_dir / f"spectra_{i}.npy"
                    # 确保我们保存一维数组
                    spec = self.spectra[:, i].flatten() if self.spectra.ndim > 1 else self.spectra
                    np.save(spec_file, spec)
                
                # 保存元数据 - 转换为JSON可序列化格式
                import json
                
                # 递归转换函数
                def to_serializable(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist() if obj.ndim > 0 else float(obj)
                    elif isinstance(obj, list):
                        return [to_serializable(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {k: to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                        return obj.item()
                    else:
                        try:
                            json.dumps(obj)
                            return obj
                        except:
                            return str(obj)
                
                # 转换元数据
                meta_json = to_serializable(self.metadata)
                
                # 保存为JSON
                with open(legacy_dir / "metadata.json", 'w') as f:
                    json.dump(meta_json, f)
                
                logger.info(f"Saved legacy format to {legacy_dir}")
                
            except Exception as e:
                logger.error(f"Error saving legacy format: {str(e)}")
                # 继续执行，确保主要保存操作不受影响
        
        except Exception as e:
            logger.error(f"Error saving binned data: {str(e)}")
            raise

    @classmethod
    def load(cls, filename):
        """
        从文件加载binned数据
        
        Parameters
        ----------
        filename : str 或 Path
            输入文件名
                
        Returns
        -------
        BinnedData子类
            加载的数据
        """
        import pickle
        import os
        import logging
        import numpy as np
        from pathlib import Path
        logger = logging.getLogger(__name__)
        
        try:
            # 将filename转换为Path对象
            if not isinstance(filename, Path):
                filename = Path(filename)
            
            # 基本文件名（不带扩展名）
            base_filename = filename.stem
            
            # 创建pickle文件名（.pkl扩展名）
            pickle_filename = filename.parent / f"{base_filename}.pkl"
            
            # 如果pickle文件存在，使用它
            if pickle_filename.exists():
                with open(pickle_filename, 'rb') as f:
                    save_data = pickle.load(f)
                
                # 创建新对象
                obj = cls.__new__(cls)
                
                # 设置属性
                obj.wavelength = save_data['wavelength']
                obj.spectra = save_data['spectra']
                obj.bin_num = save_data['bin_num']
                obj.bin_indices = save_data['bin_indices']
                obj.bin_type = save_data['bin_type']
                obj.n_bins = save_data['n_bins']
                obj.metadata = save_data['metadata']
                
                logger.info(f"Loaded binned data from {pickle_filename}")
                return obj
            
            # 如果pickle文件不存在，尝试使用npz文件
            elif filename.exists():
                try:
                    data = np.load(filename, allow_pickle=True)
                    
                    # 创建新对象
                    obj = cls.__new__(cls)
                    
                    # 设置基本属性
                    obj.wavelength = data['wavelength']
                    obj.bin_num = data['bin_num']
                    obj.bin_type = str(data['bin_type']) if 'bin_type' in data else 'unknown'
                    obj.n_bins = int(data['n_bins'])
                    
                    # 加载光谱
                    if 'spectra' in data:
                        obj.spectra = data['spectra']
                    
                    # 加载元数据
                    if 'metadata' in data:
                        obj.metadata = data['metadata'].item()
                    else:
                        obj.metadata = {}
                    
                    # 加载bin_indices
                    if 'bin_indices_list' in data:
                        obj.bin_indices = []
                        for indices in data['bin_indices_list']:
                            obj.bin_indices.append(np.array(indices))
                    elif 'bin_indices' in data:
                        obj.bin_indices = data['bin_indices']
                    else:
                        obj.bin_indices = []
                    
                    logger.info(f"Loaded binned data from {filename}")
                    return obj
                except Exception as e:
                    logger.warning(f"Error loading npz file: {str(e)}, trying legacy format")
                
            # 尝试加载legacy格式
            legacy_dir = filename.parent / f"{base_filename}_data"
            
            if legacy_dir.exists():
                # 创建新对象
                obj = cls.__new__(cls)
                
                # 加载波长
                wavelength_file = legacy_dir / "wavelength.npy"
                if wavelength_file.exists():
                    obj.wavelength = np.load(wavelength_file)
                else:
                    raise FileNotFoundError(f"Wavelength file not found: {wavelength_file}")
                
                # 加载bin号
                bin_num_file = legacy_dir / "bin_num.npy"
                if bin_num_file.exists():
                    obj.bin_num = np.load(bin_num_file)
                else:
                    raise FileNotFoundError(f"Bin number file not found: {bin_num_file}")
                
                # 确定bins数量
                unique_bins = np.unique(obj.bin_num)
                obj.n_bins = len(unique_bins[unique_bins >= 0])  # 只计算有效bin
                obj.bin_type = 'unknown'
                
                # 加载光谱
                spectra_list = []
                for i in range(obj.n_bins):
                    spec_file = legacy_dir / f"spectra_{i}.npy"
                    if spec_file.exists():
                        spectra_list.append(np.load(spec_file))
                
                # 将光谱列表转换为数组
                if spectra_list:
                    obj.spectra = np.column_stack(spectra_list)
                else:
                    obj.spectra = np.array([])
                
                # 加载元数据
                import json
                metadata_file = legacy_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        obj.metadata = json.load(f)
                else:
                    obj.metadata = {}
                
                # 重建bin_indices
                obj.bin_indices = []
                for i, bin_id in enumerate(unique_bins):
                    if bin_id >= 0:
                        indices = np.where(obj.bin_num == bin_id)[0]
                        obj.bin_indices.append(indices)
                
                logger.info(f"Loaded legacy format from {legacy_dir}")
                return obj
            else:
                raise FileNotFoundError(f"No data file found at {filename} or legacy format at {legacy_dir}")
        
        except Exception as e:
            logger.error(f"Error loading binned data: {str(e)}")
            raise

class VoronoiBinnedData(BinnedSpectra):
    """
    Voronoi binned 数据类
    
    扩展BinnedSpectra类，专门用于Voronoi binning结果
    """
    
    def __init__(self, bin_num, bin_indices, spectra, wavelength, metadata=None):
        """
        初始化Voronoi binned数据
        
        Parameters
        ----------
        bin_num : numpy.ndarray
            每个像素对应的bin号
        bin_indices : list
            每个bin中包含的像素索引列表
        spectra : numpy.ndarray
            合并后的光谱数据，形状为(n_wavelength, n_bins)
        wavelength : numpy.ndarray
            波长数组
        metadata : dict, optional
            额外的元数据
        """
        super().__init__("voronoi", bin_num, bin_indices, spectra, wavelength, metadata)

class RadialBinnedData(BinnedSpectra):
    """
    径向binned数据类
    
    扩展BinnedSpectra类，专门用于径向binning结果
    """
    
    def __init__(self, bin_num, bin_indices, spectra, wavelength, bin_radii=None, metadata=None):
        """
        初始化径向binned数据
        
        Parameters
        ----------
        bin_num : numpy.ndarray
            每个像素对应的bin号
        bin_indices : list
            每个bin中包含的像素索引列表
        spectra : numpy.ndarray
            合并后的光谱数据，形状为(n_wavelength, n_bins)
        wavelength : numpy.ndarray
            波长数组
        bin_radii : numpy.ndarray, optional
            每个bin的半径
        metadata : dict, optional
            额外的元数据
        """
        if metadata is None:
            metadata = {}
            
        # 确保有径向距离信息
        if bin_radii is not None:
            metadata['bin_distances'] = bin_radii
            metadata['radii'] = bin_radii
            
        super().__init__("radial", bin_num, bin_indices, spectra, wavelength, metadata)

# Utility functions for plotting
def plot_binned_map(x, y, bin_num, values=None, title=None, cmap='viridis', 
                   vmin=None, vmax=None, savefile=None, figsize=(10, 8), 
                   equal_aspect=True):
    """
    Plot a map of bins, optionally colored by values.
    
    Parameters
    ----------
    x, y : array_like
        Coordinates of pixels
    bin_num : array_like
        Bin assignment for each pixel
    values : array_like, optional
        Values to use for coloring bins
    title : str, optional
        Plot title
    cmap : str, optional
        Colormap name
    vmin, vmax : float, optional
        Color scale limits
    savefile : str, optional
        Filename to save plot
    figsize : tuple, optional
        Figure size
    equal_aspect : bool, optional
        Whether to keep aspect ratio equal
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if values is not None:
        # Map bin values to pixels
        pixel_values = np.full_like(x, np.nan, dtype=float)
        for bin_id in np.unique(bin_num):
            mask = bin_num == bin_id
            if bin_id < len(values) and not np.isnan(values[bin_id]):
                pixel_values[mask] = values[bin_id]
        
        # Plot values
        sc = ax.scatter(x, y, c=pixel_values, cmap=cmap, s=5, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax)
    else:
        # Create a color map for bins
        unique_bins = np.unique(bin_num)
        cmap_obj = plt.cm.get_cmap(cmap, len(unique_bins))
        
        # Plot each bin with a different color
        for i, bin_id in enumerate(unique_bins):
            mask = bin_num == bin_id
            ax.scatter(x[mask], y[mask], c=[cmap_obj(i)], s=5, alpha=0.8, edgecolor='none')
    
    if equal_aspect:
        ax.set_aspect('equal')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    if title:
        ax.set_title(title)
    
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
    
    return fig

def plot_radial_profile(radii, values, yerr=None, title=None, xlabel='Radius', 
                       ylabel=None, savefile=None, figsize=(10, 6)):
    """
    Plot a radial profile.
    
    Parameters
    ----------
    radii : array_like
        Radii of the bins
    values : array_like
        Values to plot
    yerr : array_like, optional
        Error bars for values
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    savefile : str, optional
        Filename to save plot
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 确保radii和values是一维数组
    radii_flat = radii.flatten() if hasattr(radii, 'flatten') else radii
    values_flat = values.flatten() if hasattr(values, 'flatten') else values
    
    if yerr is not None:
        yerr_flat = yerr.flatten() if hasattr(yerr, 'flatten') else yerr
        ax.errorbar(radii_flat, values_flat, yerr=yerr_flat, fmt='o-', capsize=3)
    else:
        ax.plot(radii_flat, values_flat, 'o-')
    
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
    
    return fig