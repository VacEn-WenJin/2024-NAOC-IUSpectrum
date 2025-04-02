"""
MUSE Data Cube analysis class
Core data processing module for ISAPC
"""
import logging
logger = logging.getLogger(__name__)

import traceback
import time
import warnings
import os
from typing import Union, Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
from astropy import constants
from astropy.io import fits
from joblib import delayed
from ppxf import ppxf_util
from ppxf.ppxf import ppxf
from ppxf.sps_util import sps_lib
from scipy.interpolate import interp1d

from utils.parallel import ParallelTqdm
from utils.calc import apply_velocity_shift, spectres

# Get speed of light from astropy in km/s
SPEED_OF_LIGHT = constants.c.to('km/s').value

class MUSECube:
    def __init__(
            self,
            filename: str,
            redshift: float,
            wvl_air_angstrom_range: Optional[tuple[float, float]] = None,
            use_good_wavelength: bool = True
    ) -> None:
        """
        Read MUSE data cube, extract relevant information and preprocess it
        (de-redshift, ppxf log-rebin, and form coordinate grid).
        
        Parameters
        ----------
        filename : str
            Filename of the MUSE data cube
        redshift : float
            Redshift of the galaxy
        wvl_air_angstrom_range : tuple[float, float], optional
            Wavelength range to consider (Angstrom in air wavelength)
            If None and use_good_wavelength=True, will try to use goodwavelengthrange from data
        use_good_wavelength : bool
            Whether to use goodwavelengthrange from data if available
        """
        self._filename = filename
        self._wvl_air_angstrom_range = wvl_air_angstrom_range
        self._redshift = redshift
        self._use_good_wavelength = use_good_wavelength

        try:
            self._read_fits_file()
            self._preprocess_cube()
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

        # Initialize fields for storing results
        self._optimal_tmpls = None
        self._emission_flux = {}
        self._emission_sig = {}
        self._emission_vel = {}
        self._gas_bestfit_field = None
        self._spectral_indices = {}
        self._index_calculators = {}
        self._emission_wavelength = {}
        self._template_weights = None
        self._poly_coeffs = None
        self._sps = None

        # Initialize binning support fields
        self._is_binned = False
        self._bin_type = None
        self._bin_num = None
        self._bin_indices = None
        self._binned_spectra = None
        self._binned_wavelength = None
        self._bin_pixel_map = {}
        self._n_bins = 0
        self._bin_velocity = None
        self._bin_dispersion = None
        self._bin_bestfit = None
        self._bin_optimal_tmpls = None
        self._bin_weights = None
        self._bin_emission_flux = {}
        self._bin_emission_vel = {}
        self._bin_emission_sig = {}
        self._bin_gas_bestfit = None
        self._bin_indices_result = {}

    def _read_fits_file(self) -> None:
        """
        Read MUSE cube data, create a dummy noise cube, extract the wavelength axis,
        and obtain instrumental information from the FITS header.
        """
        try:
            cut_lhs, cut_rhs = 1, 1

            with fits.open(self._filename) as fits_hdu:
                # Load fits header info
                self._fits_hdu_header = fits_hdu[0].header
                
                # Enhanced handling of multiple FITS file types
                # First try the primary HDU
                if fits_hdu[0].data is not None and len(fits_hdu[0].data.shape) == 3:
                    self._raw_cube_data = fits_hdu[0].data[cut_lhs:-cut_rhs, :, :] * 1e18
                
                # If primary HDU doesn't have a data cube, check extension HDUs
                elif len(fits_hdu) > 1:
                    data_found = False
                    for ext in range(1, len(fits_hdu)):
                        if fits_hdu[ext].data is not None and len(fits_hdu[ext].data.shape) == 3:
                            self._raw_cube_data = fits_hdu[ext].data[cut_lhs:-cut_rhs, :, :] * 1e18
                            # Merge header information
                            for key in fits_hdu[ext].header:
                                if key not in ('XTENSION', 'BITPIX', 'NAXIS', 'PCOUNT', 'GCOUNT'):
                                    self._fits_hdu_header[key] = fits_hdu[ext].header[key]
                            data_found = True
                            break
                    
                    if not data_found:
                        raise ValueError("No valid 3D data found in the FITS file")
                else:
                    raise ValueError("Invalid FITS file structure: no data cube found")
                
                # Replace NaN values with zeros
                if np.any(~np.isfinite(self._raw_cube_data)):
                    self._raw_cube_data = np.nan_to_num(self._raw_cube_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Create a variance cube (dummy, as the file contains no errors)
                self._raw_cube_var = np.ones_like(self._raw_cube_data)

                # Calculate wavelength axis
                if 'CRVAL3' in self._fits_hdu_header and 'CD3_3' in self._fits_hdu_header:
                    self._obs_wvl_air_angstrom = (
                            self._fits_hdu_header['CRVAL3'] + self._fits_hdu_header['CD3_3'] *
                            (np.arange(self._raw_cube_data.shape[0]) + cut_lhs)
                    )
                else:
                    # If wavelength information not found, create a linear wavelength axis and issue warning
                    warnings.warn("Wavelength information not found in header. Using a linear scale.")
                    self._obs_wvl_air_angstrom = np.linspace(4000, 7000, self._raw_cube_data.shape[0])

                # Find and set goodwavelengthrange
                good_wvl_range = None
                
                # Try to get goodwavelengthrange from header
                if 'WAVGOOD0' in self._fits_hdu_header and 'WAVGOOD1' in self._fits_hdu_header:
                    good_wvl_range = (
                        float(self._fits_hdu_header['WAVGOOD0']) / (1 + self._redshift),
                        float(self._fits_hdu_header['WAVGOOD1']) / (1 + self._redshift)
                    )
                    logger.info(f"Found goodwavelengthrange in header with redshift correction: {good_wvl_range}")
                
                # Try to find goodwavelengthrange in other extensions and tables
                if good_wvl_range is None:
                    for ext in range(len(fits_hdu)):
                        if isinstance(fits_hdu[ext], fits.BinTableHDU):
                            if 'GOODWAVEMIN' in fits_hdu[ext].header and 'GOODWAVEMAX' in fits_hdu[ext].header:
                                good_wvl_range = (
                                    float(fits_hdu[ext].header['GOODWAVEMIN']) / (1 + self._redshift),
                                    float(fits_hdu[ext].header['GOODWAVEMAX']) / (1 + self._redshift)
                                )
                                logger.info(f"Found goodwavelengthrange in extension {ext}: {good_wvl_range}")
                                break
                            # Check columns in table
                            elif fits_hdu[ext].data is not None:
                                col_names = fits_hdu[ext].data.names if hasattr(fits_hdu[ext].data, 'names') else []
                                if 'goodwavelengthrange' in col_names:
                                    good_wvl_vals = fits_hdu[ext].data['goodwavelengthrange']
                                    good_wvl_range = (min(good_wvl_vals), max(good_wvl_vals))
                                    logger.info(f"Found goodwavelengthrange in table: {good_wvl_range}")
                                    break
                
                # If goodwavelengthrange found and use_good_wavelength is set
                if good_wvl_range is not None and self._use_good_wavelength:
                    # Check if wavelength range is reasonable, avoid extreme values
                    min_wvl = np.min(self._obs_wvl_air_angstrom) / (1 + self._redshift)
                    max_wvl = np.max(self._obs_wvl_air_angstrom) / (1 + self._redshift)
                    
                    # Ensure good_wvl_range is within actual data range
                    good_min = max(good_wvl_range[0], min_wvl)
                    good_max = min(good_wvl_range[1], max_wvl)
                    
                    # Set wavelength range, considering redshift correction
                    self._wvl_air_angstrom_range = (good_min, good_max)
                    logger.info(f"Using goodwavelengthrange (rest-frame): {self._wvl_air_angstrom_range}")
                elif self._wvl_air_angstrom_range is None:
                    # If no goodwavelengthrange found and no range provided, use full wavelength range
                    min_wvl = np.min(self._obs_wvl_air_angstrom) / (1 + self._redshift)
                    max_wvl = np.max(self._obs_wvl_air_angstrom) / (1 + self._redshift)
                    self._wvl_air_angstrom_range = (min_wvl, max_wvl)
                    logger.info(f"Using full wavelength range (rest-frame): {self._wvl_air_angstrom_range}")

                self._FWHM_gal = 1
                # Instrument specific parameters from ESO
                if 'CD1_1' in self._fits_hdu_header and 'CD2_1' in self._fits_hdu_header:
                    self._pxl_size_x = abs(np.sqrt(
                        self._fits_hdu_header['CD1_1'] ** 2 + self._fits_hdu_header['CD2_1'] ** 2
                    )) * 3600
                    self._pxl_size_y = abs(np.sqrt(
                        self._fits_hdu_header['CD1_2'] ** 2 + self._fits_hdu_header['CD2_2'] ** 2
                    )) * 3600
                else:
                    # Default pixel size
                    self._pxl_size_x = 0.2  # arcsec
                    self._pxl_size_y = 0.2  # arcsec
                    warnings.warn("Pixel size information not found in header. Using default value of 0.2 arcsec.")
        except Exception as e:
            logger.error(f"Error reading FITS file: {str(e)}")
            raise

    def _preprocess_cube(self):
        """Preprocess the data cube: apply redshift correction, select wavelength range, and rebin spectra"""
        try:
            # Apply redshift correction
            wvl_air_angstrom = self._obs_wvl_air_angstrom / (1 + self._redshift)

            # Select valid wavelength range
            valid_mask = (
                    (wvl_air_angstrom > self._wvl_air_angstrom_range[0]) &
                    (wvl_air_angstrom < self._wvl_air_angstrom_range[1])
            )
            self._wvl_air_angstrom = wvl_air_angstrom[valid_mask]
            self._cube_data = self._raw_cube_data[valid_mask, :, :]
            self._cube_var = self._raw_cube_var[valid_mask, :, :]

            # Derive signal and noise
            signal_2d = np.nanmedian(self._cube_data, axis=0)
            noise_2d = np.sqrt(np.nanmedian(self._cube_var, axis=0))
            self._signal = signal_2d.ravel()
            self._noise = noise_2d.ravel()

            # Create spatial coordinates for each spaxel using the image indices
            n_y, n_x = signal_2d.shape  # note: cube shape is (n_wave, n_y, n_x)
            rows, cols = np.indices((n_y, n_x))
            
            # Enhanced: Find brightest pixel, based on integrated flux over entire wavelength range
            flux_sum = np.nansum(self._cube_data, axis=0)
            brightest_idx = np.unravel_index(np.nanargmax(flux_sum), flux_sum.shape)
            brightest_y, brightest_x = brightest_idx
            
            # Centering coordinates on the brightest spaxel and scale to arcseconds
            self.x = (cols.ravel() - brightest_x) * self._pxl_size_x
            self.y = (rows.ravel() - brightest_y) * self._pxl_size_y

            # Reshape cube to 2D: each column corresponds to one spaxel spectrum
            n_wvl = self._cube_data.shape[0]
            self._spectra_2d = self._cube_data.reshape(n_wvl, -1)
            self._variance_2d = self._cube_var.reshape(n_wvl, -1)

            # Replace NaN values in spectra to avoid problems in log-rebinning
            if np.any(~np.isfinite(self._spectra_2d)):
                self._spectra_2d = np.nan_to_num(self._spectra_2d, nan=0.0, posinf=0.0, neginf=0.0)
            if np.any(~np.isfinite(self._variance_2d)):
                self._variance_2d = np.nan_to_num(self._variance_2d, nan=1.0, posinf=1.0, neginf=1.0)

            # Log-rebin of the spectra
            self._vel_scale = np.min(
                SPEED_OF_LIGHT * np.diff(np.log(self._wvl_air_angstrom))
            )

            self._spectra, self._ln_lambda_gal, _ = ppxf_util.log_rebin(
                lam=[np.min(self._wvl_air_angstrom), np.max(self._wvl_air_angstrom)],
                spec=self._spectra_2d,
                velscale=self._vel_scale
            )
            self._log_variance, _, _ = ppxf_util.log_rebin(
                lam=[np.min(self._wvl_air_angstrom), np.max(self._wvl_air_angstrom)],
                spec=self._variance_2d,
                velscale=self._vel_scale
            )
            self._lambda_gal = np.exp(self._ln_lambda_gal)
            self._FWHM_gal = self._FWHM_gal / (1 + self._redshift)

            self._row = rows.ravel() + 1
            self._col = cols.ravel() + 1

            # Initialize fields for storing results
            self._n_y, self._n_x = n_y, n_x
            self._velocity_field = np.full((n_y, n_x), np.nan)
            self._dispersion_field = np.full((n_y, n_x), np.nan)
            
            # Store the shape of the wavelength axis for later reference
            self._n_wave_fit = len(self._lambda_gal)
            self._bestfit_field = np.full((self._n_wave_fit, n_y, n_x), np.nan)

        except Exception as e:
            logger.error(f"Error preprocessing cube: {str(e)}")
            raise

    def setup_binning(self, bin_type, bin_data):
        """
        Set up binning information in the cube for binned analysis
        
        Parameters
        ----------
        bin_type : str
            Binning type: 'VNB' for Voronoi or 'RDB' for Radial
        bin_data : dict or BinnedSpectra object
            Binning information
        """
        # Store binning information
        self._bin_type = bin_type
        
        # Extract key binning data
        if hasattr(bin_data, 'bin_num') and hasattr(bin_data, 'bin_indices'):
            # If given a BinnedSpectra object
            self._bin_num = bin_data.bin_num
            self._bin_indices = bin_data.bin_indices
            self._binned_spectra = bin_data.spectra
            self._binned_wavelength = bin_data.wavelength
            
            # If bin_data has metadata, copy relevant values
            if hasattr(bin_data, 'metadata'):
                self._bin_metadata = bin_data.metadata
                
                # For radial binning, get bin_radii if available
                if bin_type == 'RDB' and hasattr(bin_data, 'bin_radii'):
                    self._bin_radii = bin_data.bin_radii
                    
        else:
            # If given a dictionary
            self._bin_num = bin_data.get('bin_num')
            self._bin_indices = bin_data.get('bin_indices')
            self._binned_spectra = bin_data.get('spectra')
            self._binned_wavelength = bin_data.get('wavelength')
            
            # Get metadata if available
            self._bin_metadata = bin_data.get('metadata', {})
            
            # For radial binning, get bin_radii if available
            if bin_type == 'RDB' and 'bin_radii' in bin_data:
                self._bin_radii = bin_data.get('bin_radii')
            
        # Create mapping between bins and pixels
        self._bin_pixel_map = {}
        for bin_idx, indices in enumerate(self._bin_indices):
            self._bin_pixel_map[bin_idx] = indices
        
        # Mark cube as using binned mode
        self._is_binned = True
        
        # Store number of bins
        self._n_bins = len(self._bin_indices)
        
        # Calculate logarithmic wavelength grid on the binned wavelength
        self._bin_ln_wavelength = np.log(self._binned_wavelength)
        
        logger.info(f"Cube set up for {bin_type} analysis with {self._n_bins} bins")

    def calculate_snr(self, continuum_range=None):
        """
        Calculate signal-to-noise ratio from the spectra and fits
        
        Parameters
        ----------
        continuum_range : tuple of float, optional
            Wavelength range to use for calculation (min, max)
            
        Returns
        -------
        dict
            Dictionary containing SNR maps 
        """
        # Check if we're using binned data
        if self._is_binned:
            return self._calculate_snr_binned(continuum_range)
        try:
            # Check if fitting has been performed
            if self._bestfit_field is None or np.all(np.isnan(self._bestfit_field)):
                logger.warning("No spectral fitting results available for SNR calculation")
                return None
            
            # Initialize result arrays
            n_y, n_x = self._velocity_field.shape
            snr_map = np.full((n_y, n_x), np.nan)
            signal_map = np.full((n_y, n_x), np.nan)
            noise_map = np.full((n_y, n_x), np.nan)
            
            # Get wavelength range for calculation
            if continuum_range is None:
                # Use a default range in rest-frame wavelength
                continuum_range = (5075, 5125)  # Standard continuum region
            
            # Find wavelength indices within range
            wave_mask = ((self._lambda_gal >= continuum_range[0]) & 
                        (self._lambda_gal <= continuum_range[1]))
            
            if not np.any(wave_mask):
                logger.warning(f"No wavelength points in range {continuum_range}")
                return None
            
            # Calculate SNR for each valid spaxel
            valid_mask = ~np.isnan(self._velocity_field)
            for y in range(n_y):
                for x in range(n_x):
                    if valid_mask[y, x]:
                        # Get spaxel index
                        spaxel_idx = y * n_x + x
                        
                        # Get observed and model spectra for continuum region
                        observed = self._spectra[wave_mask, spaxel_idx]
                        model = self._bestfit_field[wave_mask, y, x]
                        
                        # Calculate signal as median of model
                        signal = np.nanmedian(model)
                        if signal < 0:
                            signal = 0.1
                        
                        # Calculate noise as std of residuals
                        residual = observed - model
                        noise = np.nanstd(residual)
                        
                        # Calculate SNR
                        if noise < 1:
                            noise = 1
                        snr = signal / noise
                        snr_map[y, x] = snr
                        signal_map[y, x] = signal
                        noise_map[y, x] = noise
            
            return {
                'snr': snr_map,
                'signal': signal_map,
                'noise': noise_map,
                'wavelength_range': continuum_range
            }
        
        except Exception as e:
            logger.error(f"Error calculating SNR: {str(e)}")
            return None

    def _calculate_snr_binned(self, continuum_range=None):
        """
        Calculate signal-to-noise ratio for binned data
        
        Parameters
        ----------
        continuum_range : tuple of float, optional
            Wavelength range to use for calculation (min, max)
            
        Returns
        -------
        dict
            Dictionary containing SNR maps
        """
        try:
            # Check if bin fitting has been performed
            if not hasattr(self, '_bin_bestfit') or self._bin_bestfit is None:
                logger.warning("No binned spectral fitting results available for SNR calculation")
                return None
            
            # Initialize result arrays
            n_bins = self._n_bins
            bin_snr = np.full(n_bins, np.nan)
            bin_signal = np.full(n_bins, np.nan)
            bin_noise = np.full(n_bins, np.nan)
            
            # Create pixel maps for output consistency
            n_y, n_x = self._velocity_field.shape
            snr_map = np.full((n_y, n_x), np.nan)
            signal_map = np.full((n_y, n_x), np.nan)
            noise_map = np.full((n_y, n_x), np.nan)
            
            # Get wavelength range for calculation
            if continuum_range is None:
                # Use a default range in rest-frame wavelength
                continuum_range = (5075, 5125)  # Standard continuum region
            
            # Find wavelength indices within range
            wave_mask = ((self._binned_wavelength >= continuum_range[0]) & 
                        (self._binned_wavelength <= continuum_range[1]))
            
            if not np.any(wave_mask):
                logger.warning(f"No wavelength points in range {continuum_range}")
                return None
            
            # Calculate SNR for each bin
            for bin_idx in range(n_bins):
                # Skip invalid bins
                if np.isnan(self._bin_velocity[bin_idx]):
                    continue
                
                # Get observed and model spectra for continuum region
                observed = self._binned_spectra[wave_mask, bin_idx]
                model = self._bin_bestfit[wave_mask, bin_idx]
                
                # Skip invalid spectra
                if not np.any(np.isfinite(observed)) or not np.any(np.isfinite(model)):
                    continue
                
                # Calculate signal as median of model
                signal = np.nanmedian(model)
                if signal < 0 or not np.isfinite(signal):
                    signal = 0.1
                
                # Calculate noise as std of residuals
                residual = observed - model
                noise = np.nanstd(residual)
                if noise < 1 or not np.isfinite(noise):
                    noise = 1
                
                # Calculate SNR
                snr = signal / noise
                bin_snr[bin_idx] = snr
                bin_signal[bin_idx] = signal
                bin_noise[bin_idx] = noise
                
                # Map to pixels
                if bin_idx in self._bin_pixel_map:
                    for pixel_idx in self._bin_pixel_map[bin_idx]:
                        row = pixel_idx // self._n_x
                        col = pixel_idx % self._n_x
                        
                        if 0 <= row < self._n_y and 0 <= col < self._n_x:
                            snr_map[row, col] = snr
                            signal_map[row, col] = signal
                            noise_map[row, col] = noise
            
            return {
                'snr': snr_map,
                'signal': signal_map,
                'noise': noise_map,
                'bin_snr': bin_snr,
                'bin_signal': bin_signal,
                'bin_noise': bin_noise,
                'wavelength_range': continuum_range
            }
            
        except Exception as e:
            logger.error(f"Error calculating SNR for binned data: {str(e)}")
            return None

    def fit_spectra(
        self,
        template_filename: str,
        ppxf_vel_init: int = 0,
        ppxf_vel_disp_init: int = 40,
        ppxf_deg: int = 3,
        n_jobs: int = -1,
        use_binned: bool = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
        # Determine whether to use binned mode
        if use_binned is None:
            use_binned = hasattr(self, '_is_binned') and self._is_binned

        if use_binned and hasattr(self, '_binned_spectra') and self._binned_spectra is not None:
            return self._fit_spectra_binned(template_filename, ppxf_vel_init, 
                                        ppxf_vel_disp_init, ppxf_deg, n_jobs)
        else:
            # Original pixel-by-pixel implementation
            return self._fit_spectra_original(template_filename, ppxf_vel_init, 
                                            ppxf_vel_disp_init, ppxf_deg, n_jobs)

    def _fit_spectra_original(
            self,
            template_filename: str,
            ppxf_vel_init: int = 0,
            ppxf_vel_disp_init: int = 40,
            ppxf_deg: int = 3,
            n_jobs: int = -1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
        """
        Fit the stellar continuum in each spaxel using pPXF.
        
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
            n_wave_temp = sps.templates.shape[0]  # Length of the template wavelength array - KEEP FULL LENGTH
            
            # Important: Initialize fields with correct dimensions
            # For optimal templates, use the template wavelength grid
            self._optimal_tmpls = np.full((n_wave_temp, self._n_y, self._n_x), np.nan)
            self._template_weights = np.full((n_templates, self._n_y, self._n_x), np.nan)
            self._poly_coeffs = []  # Store polynomial coefficients
            
            # For observed galaxy wavelength grid
            self._bestfit_field = np.full((n_wave_fit, self._n_y, self._n_x), np.nan)

            n_wvl, n_spaxel = self._spectra.shape

            def fit_spaxel(idx):
                """Fit a single spaxel spectrum"""
                i, j = np.unravel_index(idx, (self._n_y, self._n_x))
                galaxy_data = self._spectra[:, idx]
                # Use the square root of the variance as the noise estimate
                galaxy_noise = np.sqrt(self._log_variance[:, idx])

                # Skip low SNR or invalid pixels
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
                            if idx % 100 == 0:  # Reduce log clutter by only logging every 100th failure
                                logger.debug(f"Fitting failed at ({i},{j}): {str(e)}")
                            return i, j, None

            fit_results = ParallelTqdm(
                n_jobs=n_jobs, desc='Fitting spectra', total_tasks=n_spaxel
            )(delayed(fit_spaxel)(idx) for idx in range(n_spaxel))
            
            for fit_result in fit_results:
                if fit_result[2] is None:
                    continue
                row, col, (vel, disp, bestfit, optimal_tmpl, weights, poly_coeff) = fit_result
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
            logger.error(f"Error fitting spectra: {str(e)}")
            # Return empty arrays in case of error
            return (np.full((self._n_y, self._n_x), np.nan),
                    np.full((self._n_y, self._n_x), np.nan),
                    np.full((self._n_wave_fit, self._n_y, self._n_x), np.nan),
                    np.full((n_wave_temp if 'n_wave_temp' in locals() else self._n_wave_fit, 
                             self._n_y, self._n_x), np.nan),
                    [])

    def _fit_spectra_binned(self, template_filename, ppxf_vel_init=0, 
                       ppxf_vel_disp_init=40, ppxf_deg=3, n_jobs=-1):
        """
        Fit the stellar continuum for binned spectra
        """
        try:
            # Check if template file exists
            if not os.path.exists(template_filename):
                raise FileNotFoundError(f"Template file not found: {template_filename}")
            
            # Load template
            sps = sps_lib(
                filename=template_filename,
                velscale=self._vel_scale,
                fwhm_gal=None,
                norm_range=[np.min(self._binned_wavelength), np.max(self._binned_wavelength)]
            )
            self._sps = sps  # Store SPS object for later reference
            sps.templates = sps.templates.reshape(sps.templates.shape[0], -1)
            
            # Normalize stellar template
            sps.templates /= np.median(sps.templates)
            
            # Create proper wavelength mask for binned data
            ln_binned_wavelength = np.log(self._binned_wavelength)
            tmpl_mask = ppxf_util.determine_mask(
                ln_lam=ln_binned_wavelength,
                lam_range_temp=np.exp(sps.ln_lam_temp[[0, -1]]),
                width=1000
            )
            
            # Initialize storage for results
            n_templates = sps.templates.shape[1]
            n_wave_fit = len(self._binned_wavelength)
            n_wave_temp = sps.templates.shape[0]
            n_bins = self._n_bins
            
            # Initialize fields with correct dimensions
            self._velocity_field = np.full((self._n_y, self._n_x), np.nan)
            self._dispersion_field = np.full((self._n_y, self._n_x), np.nan)
            
            # For optimal templates and bestfit, use bin dimensions first
            bin_velocity = np.full(n_bins, np.nan)
            bin_dispersion = np.full(n_bins, np.nan) 
            bin_bestfit = np.full((n_wave_fit, n_bins), np.nan)
            bin_optimal_tmpls = np.full((n_wave_temp, n_bins), np.nan)
            bin_weights = np.full((n_templates, n_bins), np.nan)
            bin_poly_coeffs = []
            
            # Define function to process a single bin
            def fit_bin(bin_idx):
                """Fit a single bin's spectrum"""
                bin_spectrum = self._binned_spectra[:, bin_idx]
                
                # Create dummy noise (use constant or estimate from spectrum)
                bin_noise = np.ones_like(bin_spectrum) * np.std(bin_spectrum) * 0.1
                
                # Skip low SNR or invalid bins
                if np.count_nonzero(bin_spectrum) < 50 or np.count_nonzero(np.isfinite(bin_spectrum)) < 50:
                    return bin_idx, None
                
                # Replace NaN values to avoid problems in ppxf
                if np.any(~np.isfinite(bin_spectrum)):
                    bin_spectrum = np.nan_to_num(bin_spectrum, nan=0.0, posinf=0.0, neginf=0.0)
                if np.any(~np.isfinite(bin_noise)):
                    bin_noise = np.nan_to_num(bin_noise, nan=1.0, posinf=1.0, neginf=1.0)
                
                # Call pPXF with error handling
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    try:
                        pp = ppxf(
                            sps.templates, bin_spectrum, bin_noise,
                            self._vel_scale, mask=tmpl_mask,
                            start=[ppxf_vel_init, ppxf_vel_disp_init], degree=ppxf_deg,
                            lam=self._binned_wavelength, lam_temp=sps.lam_temp,
                            quiet=True
                        )
                        
                        # Ensure dispersion value is reasonable
                        if pp.sol[1] < 0:
                            pp.sol[1] = 10.0  # Set to a reasonable minimum value
                        
                        # Calculate polynomial coefficients for later use
                        poly_coeff = np.polyfit(self._binned_wavelength, pp.apoly, ppxf_deg)
                        
                        # Calculate optimal template directly from weights on TEMPLATE wavelength grid
                        optimal_template = sps.templates @ pp.weights
                        
                        # Calculate best-fit on GALAXY wavelength grid
                        bestfit = pp.bestfit
                        
                        return bin_idx, (
                            pp.sol[0], pp.sol[1], bestfit,
                            optimal_template,
                            pp.weights,
                            poly_coeff
                        )
                    except Exception as e:
                        # Fall back to simpler fit if first attempt fails
                        try:
                            pp = ppxf(
                                sps.templates, bin_spectrum, bin_noise,
                                self._vel_scale, mask=tmpl_mask,
                                start=[ppxf_vel_init, ppxf_vel_disp_init], degree=0,
                                lam=self._binned_wavelength, lam_temp=sps.lam_temp,
                                quiet=True
                            )
                            
                            # Ensure dispersion value is reasonable
                            if pp.sol[1] < 0:
                                pp.sol[1] = 10.0
                            
                            # Calculate polynomial coefficients (constant term)
                            poly_coeff = np.array([pp.apoly[0]])
                            
                            # Calculate optimal template directly from weights
                            optimal_template = sps.templates @ pp.weights
                            
                            # Calculate best-fit
                            bestfit = pp.bestfit
                            
                            return bin_idx, (
                                pp.sol[0], pp.sol[1], bestfit,
                                optimal_template,
                                pp.weights,
                                poly_coeff
                            )
                        except Exception as e2:
                            logger.debug(f"Both fitting attempts failed for bin {bin_idx}: {e2}")
                            return bin_idx, None
            
            # Process bins in parallel
            from utils.parallel import ParallelTqdm
            
            fit_results = ParallelTqdm(
                n_jobs=n_jobs, desc='Fitting binned spectra', total_tasks=n_bins
            )(delayed(fit_bin)(bin_idx) for bin_idx in range(n_bins))
            
            # Process results
            for bin_idx, result in fit_results:
                if result is None:
                    continue
                    
                vel, disp, bestfit, optimal_tmpl, weights, poly_coeff = result
                
                # Store bin results
                bin_velocity[bin_idx] = vel
                bin_dispersion[bin_idx] = disp
                bin_bestfit[:, bin_idx] = bestfit
                bin_optimal_tmpls[:, bin_idx] = optimal_tmpl
                bin_weights[:, bin_idx] = weights
                bin_poly_coeffs.append((bin_idx, poly_coeff))
                
                # Map bin results to pixels
                if bin_idx in self._bin_pixel_map:
                    for pixel_idx in self._bin_pixel_map[bin_idx]:
                        # Convert linear index to 2D coordinates
                        row = pixel_idx // self._n_x
                        col = pixel_idx % self._n_x
                        
                        # Store in pixel-based fields
                        if 0 <= row < self._n_y and 0 <= col < self._n_x:
                            self._velocity_field[row, col] = vel
                            self._dispersion_field[row, col] = disp
            
            # Store bin-specific results
            self._bin_velocity = bin_velocity
            self._bin_dispersion = bin_dispersion
            self._bin_bestfit = bin_bestfit
            self._bin_optimal_tmpls = bin_optimal_tmpls
            self._bin_weights = bin_weights
            
            # Reshape bestfit and optimal templates to match original dimensions
            # but with bin values in each pixel
            self._bestfit_field = np.zeros((n_wave_fit, self._n_y, self._n_x))
            self._optimal_tmpls = np.zeros((n_wave_temp, self._n_y, self._n_x))
            
            # Map bin results to the 3D arrays
            for bin_idx in range(n_bins):
                if bin_idx in self._bin_pixel_map:
                    for pixel_idx in self._bin_pixel_map[bin_idx]:
                        row = pixel_idx // self._n_x
                        col = pixel_idx % self._n_x
                        
                        if 0 <= row < self._n_y and 0 <= col < self._n_x:
                            self._bestfit_field[:, row, col] = bin_bestfit[:, bin_idx]
                            self._optimal_tmpls[:, row, col] = bin_optimal_tmpls[:, bin_idx]
            
            return (self._velocity_field, self._dispersion_field,
                    self._bestfit_field, self._optimal_tmpls, bin_poly_coeffs)
        
        except Exception as e:
            logger.error(f"Error in binned spectra fitting: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return empty arrays in case of error
            return (np.full((self._n_y, self._n_x), np.nan),
                    np.full((self._n_y, self._n_x), np.nan),
                    np.full((n_wave_fit, self._n_y, self._n_x), np.nan),
                    np.full((n_wave_temp, self._n_y, self._n_x), np.nan),
                    [])

    def fit_emission_lines(
        self,
        template_filename: str,
        line_names: Optional[List[str]] = None,
        ppxf_vel_init: Optional[np.ndarray] = None,
        ppxf_sig_init: float = 50.0,
        ppxf_deg: int = 8,
        n_jobs: int = -1,
        verbose: bool = True,
        use_binned: bool = None
        ) -> Dict[str, Any]:
        """
        Universal emission line fitting for both pixel-by-pixel and binned data
        
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
        use_binned : bool, optional
            Whether to use binned data if available. If None, auto-detect.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing emission line fitting results
        """
        # Determine whether to use binned mode
        if use_binned is None:
            use_binned = hasattr(self, '_is_binned') and self._is_binned

        if use_binned and hasattr(self, '_binned_spectra') and self._binned_spectra is not None:
            return self._fit_emission_lines_binned(template_filename, line_names, ppxf_vel_init,
                                                ppxf_sig_init, ppxf_deg, n_jobs, verbose)
        else:
            # Original pixel-by-pixel implementation
            return self._fit_emission_lines_original(template_filename, line_names, ppxf_vel_init,
                                                ppxf_sig_init, ppxf_deg, n_jobs, verbose)

    def _fit_emission_lines_original(
        self,
        template_filename: str,
        line_names: Optional[List[str]] = None,
        ppxf_vel_init: Optional[np.ndarray] = None,
        ppxf_sig_init: float = 50.0,
        ppxf_deg: int = 8,
        n_jobs: int = -1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Fit emission line components based on stellar template.
        
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
                    
                    # Improved matching to handle special characters in line names
                    match_found = False
                    for requested in line_names:
                        # Clean up names for comparison by removing special characters
                        cleaned_base = base_name.replace('[', '').replace(']', '').replace('_d', '')
                        cleaned_requested = requested.replace('[', '').replace(']', '').replace('_d', '')
                        
                        if cleaned_requested.lower() in cleaned_base.lower():
                            match_found = True
                            break
                    
                    if match_found:
                        valid_indices.append(i)
                    
                if valid_indices:
                    gas_templates = gas_templates[:, valid_indices]
                    gas_names = [gas_names[i] for i in valid_indices]
                    line_wave = [line_wave[i] for i in valid_indices]
                else:
                    logger.warning(f"No matching emission lines found for: {line_names}")
                    logger.warning(f"Available lines: {[n.split('_(')[0] for n in gas_names]}")
            
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
            
            # Store ppxf results for each spaxel
            self._ppxf_gas_results = []
            
            n_wvl, n_spaxel = self._spectra.shape
            
            def fit_spaxel_emission(idx):
                """Fit emission lines for a single spaxel"""
                i, j = np.unravel_index(idx, (self._n_y, self._n_x))
                galaxy_data = self._spectra[:, idx]
                galaxy_noise = np.sqrt(self._log_variance[:, idx])
                
                # Skip if insufficient data or first-time fitting failed
                if (np.count_nonzero(galaxy_data) < 50 or 
                    np.count_nonzero(np.isfinite(galaxy_data)) < 50 or 
                    np.isnan(self._velocity_field[i, j])):
                    return i, j, None
                
                # Replace NaN values to avoid problems in ppxf
                if np.any(~np.isfinite(galaxy_data)):
                    galaxy_data = np.nan_to_num(galaxy_data, nan=0.0, posinf=0.0, neginf=0.0)
                if np.any(~np.isfinite(galaxy_noise)):
                    galaxy_noise = np.nan_to_num(galaxy_noise, nan=1.0, posinf=1.0, neginf=1.0)
                
                # Get optimal stellar template for this spaxel
                optimal_template = self._optimal_tmpls[:, i, j]
                
                # Get initial velocity value
                vel_init = self._velocity_field[i, j] if not np.isnan(self._velocity_field[i, j]) else 0
                
                try:
                    # Load SPS for this spaxel
                    sps = sps_lib(
                        filename=template_filename,
                        velscale=self._vel_scale,
                        fwhm_gal=None,
                        norm_range=self._wvl_air_angstrom_range
                    )
                    
                    # Combine stellar and gas templates
                    stars_gas_templates = np.column_stack([optimal_template, gas_templates])
                    
                    # Define component types - [0] for stellar, [1] for gas components
                    component = [0] + [1]*gas_templates.shape[1]  # Correct number based on actual templates
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
                        if verbose and idx % 100 == 0:  # Reduce log clutter
                            logger.warning(f"Gas fitting failed for pixel ({i}, {j}): {str(e)}")
                        return i, j, None
                
                except Exception as e:
                    if verbose and idx % 100 == 0:  # Reduce log clutter
                        logger.warning(f"Error in emission line fitting for pixel ({i},{j}): {str(e)}")
                    return i, j, None            
            # Run fits in parallel
            fit_results = ParallelTqdm(
                n_jobs=n_jobs, desc='Fitting emission lines', total_tasks=n_spaxel
            )(delayed(fit_spaxel_emission)(idx) for idx in range(n_spaxel))
            
            # Process results
            for fit_result in fit_results:
                if fit_result[2] is None:
                    continue
                
                row, col, result = fit_result
                
                # Save ppxf result for this spaxel
                self._ppxf_gas_results.append((row, col, result))
                
                # Save gas fitting result
                if 'gas_bestfit' in result and result['gas_bestfit'] is not None:
                    self._gas_bestfit_field[:, row, col] = result['gas_bestfit']
                
                
                # print(result['NEL_cal_tmp'].shape)
                # print(self._optimal_tmpls.shape)
                # Update optimal template if needed
                if 'NEL_cal_tmp' in result and result['NEL_cal_tmp'] is not None:
                    self._optimal_tmpls[:, row, col] = result['NEL_cal_tmp']
                # Update kinematics if needed
                if 'sol' in result and result['sol'] is not None:
                    if abs(result['sol'][0]) < 300:
                        self._velocity_field[row, col] = result['sol'][0]
                    else:
                        self._velocity_field[row, col] = 0
                    
                    if result['sol'][1] < 300:
                        self._dispersion_field[row, col] = result['sol'][1]
                    else:
                        self._dispersion_field[row, col] = 0
                
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
            
            # Post-process emission line results
            self._post_process_emission_results()
            
            # Restore original log level
            logger.setLevel(original_level)
            
            # Calculate SNR after emission line fitting (more accurate with full model)
            snr_info = self.calculate_snr()
            if snr_info is not None:
                # Add SNR information to the result
                result_dict = {
                    'emission_flux': self._emission_flux,
                    'emission_vel': self._emission_vel,
                    'emission_sig': self._emission_sig,
                    'gas_bestfit_field': self._gas_bestfit_field,
                    'emission_wavelength': self._emission_wavelength,
                    'optimal_tmpls': self._optimal_tmpls,
                    'velocity_field': self._velocity_field,
                    'dispersion_field': self._dispersion_field,
                    'signal': snr_info['signal'],
                    'noise': snr_info['noise'],
                    'snr': snr_info['snr']
                }
            else:
                # Use original result dictionary format
                result_dict = {
                    'emission_flux': self._emission_flux,
                    'emission_vel': self._emission_vel,
                    'emission_sig': self._emission_sig,
                    'gas_bestfit_field': self._gas_bestfit_field,
                    'emission_wavelength': self._emission_wavelength,
                    'optimal_tmpls': self._optimal_tmpls,
                    'velocity_field': self._velocity_field,
                    'dispersion_field': self._dispersion_field
                }

            return result_dict
        except Exception as e:
            logger.error(f"Error fitting emission lines: {str(e)}")
            logger.setLevel(original_level)
            return {
                'emission_flux': {},
                'emission_vel': {},
                'emission_sig': {},
                'gas_bestfit_field': np.full((self._n_wave_fit, self._n_y, self._n_x), np.nan),
                'emission_wavelength': {}
            }

    def _fit_emission_lines_binned(
        self,
        template_filename: str,
        line_names: Optional[List[str]] = None,
        ppxf_vel_init: Optional[np.ndarray] = None,
        ppxf_sig_init: float = None,
        ppxf_deg: int = None,
        n_jobs: int = -1,
        verbose: bool = False
        ) -> Dict[str, Any]:
        """ 
        Fit emission lines for binned spectra
        
        Parameters
        ----------
        template_filename : str
            Filename of the stellar template
        line_names : List[str], optional
            List of emission lines to fit, defaults to all available lines
        ppxf_vel_init : np.ndarray, optional
            Initial velocity field, defaults to stellar velocity field
        ppxf_sig_init : float, optional
            Initial velocity dispersion in km/s for gas, defaults to config value
        ppxf_deg : int, optional
            Degree of additive polynomial for pPXF, defaults to config value
        n_jobs : int, default=-1
            Number of parallel jobs to run
        verbose : bool, default=False
            Whether to print verbose output
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing emission line fitting results
        """
        # Import configuration manager for default values
        try:
            from config_manager import get_emission_lines, get_spectral_fitting_parameters
            
            # Get parameters from config if not specified
            fitting_params = get_spectral_fitting_parameters()
            
            if ppxf_sig_init is None:
                ppxf_sig_init = fitting_params['ppxf_vel_disp_init']
            
            if ppxf_deg is None:
                ppxf_deg = fitting_params['ppxf_gas_deg']
            
            # Use default emission lines from config if not specified
            if line_names is None:
                line_names = get_emission_lines()
                logger.info(f"Using default emission lines from config: {line_names}")
        except ImportError:
            # Fallback if config_manager not available
            if ppxf_sig_init is None:
                ppxf_sig_init = 50.0
            
            if ppxf_deg is None:
                ppxf_deg = 2
            
            # No default line_names - will use all available lines
        
        # Set log level
        original_level = logger.level
        if not verbose:
            logger.setLevel(logging.WARNING)
        
        try:
            # Check if stellar fitting has already been performed
            if not hasattr(self, '_bin_optimal_tmpls') or self._bin_optimal_tmpls is None:
                raise ValueError("Must run fit_spectra() before fit_emission_lines() for binned data")
            
            # Use bin velocities as initial values if not provided
            if ppxf_vel_init is None:
                ppxf_vel_init = self._bin_velocity if hasattr(self, '_bin_velocity') else np.zeros(self._n_bins)
            
            # Initialize result storage for bins
            n_bins = self._n_bins
            n_wave = len(self._binned_wavelength)
            
            self._bin_emission_flux = {}
            self._bin_emission_vel = {}
            self._bin_emission_sig = {}
            self._bin_gas_bestfit = np.full((n_wave, n_bins), np.nan)
            
            # Initialize pixel-based results
            self._emission_flux = {}
            self._emission_vel = {}
            self._emission_sig = {}
            self._gas_bestfit_field = np.full((n_wave, self._n_y, self._n_x), np.nan)
            
            # Store polynomial coefficients for each bin
            self._bin_poly_coeffs = []
            
            # Generate emission line templates
            from ppxf.ppxf_util import emission_lines
            
            lam_range_gal = [np.min(self._binned_wavelength), np.max(self._binned_wavelength)]
            FWHM_gal = getattr(self, '_FWHM_gal', 1.0)
            redshift = getattr(self, '_redshift', 0.0)
            
            # Generate gas templates
            gas_templates, gas_names, line_wave = emission_lines(
                self._sps.ln_lam_temp, lam_range_gal, FWHM_gal / (1 + redshift)
            )
            
            # Set up gas components
            ngas_comp = 1
            gas_templates = np.tile(gas_templates, ngas_comp)
            gas_names = np.asarray([a + f"_({p+1})" for p in range(ngas_comp) for a in gas_names])
            line_wave = np.tile(line_wave, ngas_comp)
            
            # Filter lines if requested
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
            
            # Initialize emission line storage
            for name in gas_names:
                base_name = name.split('_(')[0] if '_(' in name else name
                self._bin_emission_flux[base_name] = np.full(n_bins, np.nan)
                self._bin_emission_vel[base_name] = np.full(n_bins, np.nan)
                self._bin_emission_sig[base_name] = np.full(n_bins, np.nan)
                
                # Also initialize pixel arrays
                self._emission_flux[base_name] = np.full((self._n_y, self._n_x), np.nan)
                self._emission_vel[base_name] = np.full((self._n_y, self._n_x), np.nan)
                self._emission_sig[base_name] = np.full((self._n_y, self._n_x), np.nan)
            
            # Define function to process a single bin
            def fit_bin_emission(bin_idx):
                """Fit emission lines for a single bin"""
                # Skip bins with no valid velocity measurement
                if not hasattr(self, '_bin_velocity') or np.isnan(self._bin_velocity[bin_idx]):
                    return bin_idx, None
                
                # Get bin data
                bin_spectrum = self._binned_spectra[:, bin_idx]
                
                # Create noise estimate from the spectrum
                # Using variance = median(abs(spectrum - median(spectrum)))^2
                bin_noise = np.ones_like(bin_spectrum) # Default uniform noise
                
                # Estimate noise if enough valid points
                if np.count_nonzero(np.isfinite(bin_spectrum)) > 10:
                    # Remove extreme outliers for better noise estimation
                    clean_spectrum = bin_spectrum.copy()
                    median_val = np.nanmedian(clean_spectrum)
                    mad = np.nanmedian(np.abs(clean_spectrum - median_val))
                    
                    # Mark outliers as NaN
                    if mad > 0:
                        outliers = np.abs(clean_spectrum - median_val) > 5 * mad
                        clean_spectrum[outliers] = np.nan
                    
                    # Estimate noise robustly
                    noise_estimate = np.nanstd(clean_spectrum)
                    
                    # Use noise estimate if valid
                    if noise_estimate > 0 and np.isfinite(noise_estimate):
                        bin_noise = np.ones_like(bin_spectrum) * noise_estimate
                
                # Skip bins with insufficient data
                if np.count_nonzero(bin_spectrum) < 30 or np.count_nonzero(np.isfinite(bin_spectrum)) < 30:
                    return bin_idx, None
                
                # Replace NaN values
                bin_spectrum = np.nan_to_num(bin_spectrum, nan=0.0, posinf=0.0, neginf=0.0)
                bin_noise = np.nan_to_num(bin_noise, nan=1.0, posinf=1.0, neginf=1.0)
                
                # Get optimal stellar template for this bin
                optimal_template = self._bin_optimal_tmpls[:, bin_idx]
                
                # Get initial velocity
                vel_init = self._bin_velocity[bin_idx] if hasattr(self, '_bin_velocity') else ppxf_vel_init
                
                try:
                    # Combine stellar and gas templates
                    stars_gas_templates = np.column_stack([optimal_template, gas_templates])
                    
                    # Define component types
                    component = [0] + [1]*gas_templates.shape[1]
                    gas_component = np.array(component) > 0
                    
                    # Define moments for each component type
                    moments = [2, 2]  # 2 moments for both stellar and gas
                    ncomp = len(moments)
                    
                    # Set initial parameters
                    start = [
                        [vel_init, self._bin_dispersion[bin_idx]],  # Stellar initial kinematics
                        [vel_init, ppxf_sig_init]                 # Gas initial kinematics
                    ]
                    
                    # Set boundary conditions
                    vlim = lambda x: vel_init + x*np.array([-300, 300])
                    bounds = [
                        [vlim(1), [1, 300]],  # Stellar bounds
                        [vlim(1), [1, 200]]   # Gas bounds
                    ]
                    
                    # Call pPXF with error handling
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        
                        # Ensure noise values are not zero
                        bin_noise = np.maximum(bin_noise, 1e-10)
                        
                        pp = ppxf(
                            stars_gas_templates, bin_spectrum, bin_noise, 
                            self._vel_scale, start,
                            moments=moments, degree=ppxf_deg, mdegree=-1,
                            component=component, 
                            gas_component=gas_component, 
                            gas_names=gas_names, 
                            lam=self._binned_wavelength,
                            lam_temp=self._sps.lam_temp,
                            bounds=bounds,
                            quiet=True
                        )
                    
                    # Extract results
                    bestfit = pp.bestfit
                    
                    # Extract gas bestfit
                    gas_bestfit = np.zeros_like(bestfit)
                    if hasattr(pp, 'gas_bestfit'):
                        gas_bestfit = pp.gas_bestfit
                    elif hasattr(pp, 'matrix') and hasattr(pp, 'weights') and hasattr(pp, 'component'):
                        # Extract gas component from full model
                        comp = pp.component
                        gas_idx = np.where(comp > 0)[0]
                        if len(gas_idx) > 0:
                            gas_bestfit = np.sum(pp.matrix[:, gas_idx] @ pp.weights[gas_idx], axis=1)
                    
                    # Calculate stellar component
                    stellar_bestfit = bestfit - gas_bestfit
                    
                    # Get stellar and gas solutions
                    stellar_sol = pp.sol[0]
                    gas_sol = pp.sol[1]
                    
                    # Properly calculate and update optimal template with polynomial
                    updated_optimal_template = None
                    poly_coeff = None
                    try:
                        # Get the polynomial from ppxf
                        if hasattr(pp, 'apoly') and pp.apoly is not None and hasattr(self, '_sps'):
                            # Fit polynomial coefficients (use same degree as in ppxf)
                            poly_coeff = np.polyfit(self._binned_wavelength, pp.apoly, ppxf_deg)
                            
                            # Get the base stellar template with new weights
                            # First component (index 0) is the stellar template
                            updated_optimal_template = stars_gas_templates[:, 0] * pp.weights[0]
                            
                            # Add the polynomial evaluated on the template wavelength grid
                            template_poly = np.poly1d(poly_coeff)(self._sps.lam_temp)
                            updated_optimal_template += template_poly
                    except Exception as e:
                        logger.debug(f"Error updating optimal template for bin {bin_idx}: {e}")
                    
                    # Prepare results
                    result = {
                        'gas_bestfit': gas_bestfit,
                        'stellar_bestfit': stellar_bestfit,
                        'total_bestfit': bestfit,
                        'sol': stellar_sol,
                        'gas_sol': gas_sol,
                        'weights': pp.weights,
                        'updated_optimal_template': updated_optimal_template,
                        'poly_coeff': poly_coeff
                    }
                    
                    # Add gas flux if available
                    if hasattr(pp, 'gas_flux'):
                        result['gas_flux'] = pp.gas_flux
                        
                        # Process each emission line
                        result['emission_flux'] = {}
                        result['emission_vel'] = {}
                        result['emission_sig'] = {}
                        
                        for k, name in enumerate(gas_names):
                            base_name = name.split('_(')[0] if '_(' in name else name
                            result['emission_flux'][base_name] = pp.gas_flux[k]
                            result['emission_vel'][base_name] = gas_sol[0]
                            result['emission_sig'][base_name] = gas_sol[1]
                    
                    return bin_idx, result
                    
                except Exception as e:
                    logger.debug(f"Error fitting emission lines for bin {bin_idx}: {e}")
                    return bin_idx, None
            
            # Process bins in parallel
            from utils.parallel import ParallelTqdm
            
            fit_results = ParallelTqdm(
                n_jobs=n_jobs, desc='Fitting emission lines', total_tasks=n_bins
            )(delayed(fit_bin_emission)(bin_idx) for bin_idx in range(n_bins))
            
            # Process results
            for bin_idx, result in fit_results:
                if result is None:
                    continue
                
                # Store bin-level results
                self._bin_gas_bestfit[:, bin_idx] = result.get('gas_bestfit', np.zeros(n_wave))
                
                # Store updated optimal template if available
                if 'updated_optimal_template' in result and result['updated_optimal_template'] is not None:
                    # Initialize _bin_optimal_tmpls if needed
                    if not hasattr(self, '_bin_optimal_tmpls') or self._bin_optimal_tmpls is None:
                        n_wave_temp = len(self._sps.lam_temp) if hasattr(self, '_sps') else n_wave
                        self._bin_optimal_tmpls = np.zeros((n_wave_temp, n_bins))
                    
                    # Store the updated optimal template
                    self._bin_optimal_tmpls[:, bin_idx] = result['updated_optimal_template']
                
                # Store polynomial coefficients if available
                if 'poly_coeff' in result and result['poly_coeff'] is not None:
                    if not hasattr(self, '_bin_poly_coeffs'):
                        self._bin_poly_coeffs = []
                    self._bin_poly_coeffs.append((bin_idx, result['poly_coeff']))
                
                # Store emission line results
                if 'emission_flux' in result:
                    for base_name, flux in result['emission_flux'].items():
                        self._bin_emission_flux[base_name][bin_idx] = flux
                        self._bin_emission_vel[base_name][bin_idx] = result['emission_vel'][base_name]
                        self._bin_emission_sig[base_name][bin_idx] = result['emission_sig'][base_name]
                elif 'gas_flux' in result and gas_names:
                    for k, name in enumerate(gas_names):
                        base_name = name.split('_(')[0] if '_(' in name else name
                        self._bin_emission_flux[base_name][bin_idx] = result['gas_flux'][k]
                        self._bin_emission_vel[base_name][bin_idx] = result['gas_sol'][0]
                        self._bin_emission_sig[base_name][bin_idx] = result['gas_sol'][1]
                
                # Map results to pixels
                if bin_idx in self._bin_pixel_map:
                    for pixel_idx in self._bin_pixel_map[bin_idx]:
                        row = pixel_idx // self._n_x
                        col = pixel_idx % self._n_x
                        
                        if 0 <= row < self._n_y and 0 <= col < self._n_x:
                            # Fill pixel fields
                            self._gas_bestfit_field[:, row, col] = result.get('gas_bestfit', np.zeros(n_wave))
                            
                            if 'emission_flux' in result:
                                for base_name, flux in result['emission_flux'].items():
                                    self._emission_flux[base_name][row, col] = flux
                                    self._emission_vel[base_name][row, col] = result['emission_vel'][base_name]
                                    self._emission_sig[base_name][row, col] = result['emission_sig'][base_name]
                            elif 'gas_flux' in result and gas_names:
                                for k, name in enumerate(gas_names):
                                    base_name = name.split('_(')[0] if '_(' in name else name
                                    self._emission_flux[base_name][row, col] = result['gas_flux'][k]
                                    self._emission_vel[base_name][row, col] = result['gas_sol'][0]
                                    self._emission_sig[base_name][row, col] = result['gas_sol'][1]
            
            # Calculate SNR information
            snr_info = self.calculate_snr()
            
            # Prepare result dictionary
            if snr_info is not None:
                result_dict = {
                    'emission_flux': self._emission_flux,
                    'emission_vel': self._emission_vel,
                    'emission_sig': self._emission_sig,
                    'gas_bestfit_field': self._gas_bestfit_field,
                    'emission_wavelength': self._emission_wavelength,
                    'bin_emission_flux': self._bin_emission_flux,
                    'bin_emission_vel': self._bin_emission_vel,
                    'bin_emission_sig': self._bin_emission_sig,
                    'velocity_field': self._velocity_field,
                    'dispersion_field': self._dispersion_field,
                    'signal': snr_info.get('signal', None),
                    'noise': snr_info.get('noise', None),
                    'snr': snr_info.get('snr', None)
                }
            else:
                result_dict = {
                    'emission_flux': self._emission_flux,
                    'emission_vel': self._emission_vel,
                    'emission_sig': self._emission_sig,
                    'gas_bestfit_field': self._gas_bestfit_field,
                    'emission_wavelength': self._emission_wavelength,
                    'bin_emission_flux': self._bin_emission_flux,
                    'bin_emission_vel': self._bin_emission_vel,
                    'bin_emission_sig': self._bin_emission_sig,
                    'velocity_field': self._velocity_field,
                    'dispersion_field': self._dispersion_field
                }
            
            # Restore log level
            logger.setLevel(original_level)
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Error in binned emission line fitting: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Restore log level
            logger.setLevel(original_level)
            
            # Return empty results
            return {
                'emission_flux': {},
                'emission_vel': {},
                'emission_sig': {},
                'gas_bestfit_field': np.full((n_wave, self._n_y, self._n_x), np.nan),
                'emission_wavelength': {}
            }

    def calculate_spectral_indices(self, indices_list=None, n_jobs=-1, verbose=False, use_binned=None):
        """
        Universal spectral indices calculation for both pixel-by-pixel and binned data
        
        Parameters
        ----------
        indices_list : list of str, optional
            List of spectral indices to calculate, None uses standard set
        n_jobs : int, default=-1
            Number of parallel jobs
        verbose : bool, default=False
            Whether to display detailed information
        use_binned : bool, optional
            Whether to use binned data if available. If None, auto-detect.
                    
        Returns
        -------
        dict
            Dictionary of spectral indices
        """
        # Determine whether to use binned mode
        if use_binned is None:
            use_binned = hasattr(self, '_is_binned') and self._is_binned

        if use_binned and hasattr(self, '_binned_spectra') and self._binned_spectra is not None:
            return self._calculate_spectral_indices_binned(indices_list, n_jobs, verbose)
        else:
            # Original pixel-by-pixel implementation 
            return self._calculate_spectral_indices_original(indices_list, n_jobs, verbose)

    def _calculate_spectral_indices_original(self, indices_list=None, n_jobs=-1, verbose=False):
        """
        Calculate spectral indices for each spaxel using LineIndexCalculator
        
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
        # Import LineIndexCalculator 
        from spectral_indices import LineIndexCalculator
        
        # Set log level
        original_level = logger.level
        if not verbose:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
        
        try:
            # Define standard spectral indices if not provided
            # Default is now Hbeta, Fe5015, and Mgb as requested
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
                    # Relaxed condition: only need some overlap between index range and data range
                    if (orig_wave_min <= windows['red'][1] and orig_wave_max >= windows['blue'][0]):
                        valid_indices.append(index_name)
                        logger.info(f"Index {index_name} is within wavelength range")
                    else:
                        logger.warning(f"Index {index_name} outside wavelength range: blue={windows['blue']}, red={windows['red']} vs range={orig_wave_min:.2f}-{orig_wave_max:.2f}")
            
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
            
            n_wvl, n_spaxel = self._spectra.shape
            
            def calculate_index(idx):
                """
                Calculate spectral indices for a single spaxel
                Optimized for better multiprocessing performance
                """
                i, j = np.unravel_index(idx, (self._n_y, self._n_x))
                
                # Skip if first-time fitting failed - quick early return
                if np.isnan(self._velocity_field[i, j]):
                    return i, j, {index_name: np.nan for index_name in indices_list}
                
                # Get all data at once to minimize Python-level operations
                observed_spectrum = self._spectra[:, idx]
                optimal_template = self._optimal_tmpls[:, i, j]
                stellar_velocity = self._velocity_field[i, j]
                
                # Get gas velocity if available - separate from stellar velocity
                gas_velocity = None
                if has_emission_lines and hasattr(self, '_emission_vel'):
                    # Try to find gas velocity from available emission lines
                    for line_name, vel_map in self._emission_vel.items():
                        if np.isfinite(vel_map[i, j]):
                            gas_velocity = vel_map[i, j]
                            break
                
                # Get gas model if available - only once
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
                            flux=observed_spectrum,          # Observed spectrum
                            fit_wave=self._sps.lam_temp,     # Template wavelength grid
                            fit_flux=optimal_template,       # Template spectrum
                            em_wave=self._lambda_gal if gas_model is not None else None,  # Emission line wavelength grid
                            em_flux_list=gas_model,    # Emission line spectrum
                            velocity_correction=stellar_velocity,    # Stellar velocity correction
                            gas_velocity_correction=gas_velocity,    # Gas velocity correction - key change
                            continuum_mode='auto'            # Auto select continuum mode
                        )
                    except Exception as e:
                        logger.debug(f"Error creating LineIndexCalculator at ({i},{j}): {str(e)}")
                        return i, j, {index_name: np.nan for index_name in indices_list}
                    
                    # Calculate all indices at once to minimize function calls
                    indices_values = {}
                    for index_name in indices_list:
                        try:
                            index_value = calculator.calculate_index(index_name)
                            indices_values[index_name] = index_value
                        except Exception as e:
                            indices_values[index_name] = np.nan
                    
                    # Only store calculator for specific positions (central and some samples)
                    # to reduce memory usage
                    central_i, central_j = self._n_y // 2, self._n_x // 2
                    if i == central_i and j == central_j:
                        self._index_calculators['central'] = calculator
                    elif (i % (self._n_y // 4) == 0 and j % (self._n_x // 4) == 0):
                        key = f"sample_{i}_{j}"
                        self._index_calculators[key] = calculator
                    
                    return i, j, indices_values
            
            # Calculate optimal chunk size for better parallelization
            # This reduces thread management overhead
            chunksize = max(1, n_spaxel // (4 * max(1, n_jobs if n_jobs > 0 else os.cpu_count())))
            
            logger.info(f"Using chunk size {chunksize} for spectral indices calculation")
            
            # Run calculations in parallel with optimized backend and chunk size
            index_results = ParallelTqdm(
                n_jobs=n_jobs, 
                desc='Calculating spectral indices', 
                total_tasks=n_spaxel,
                backend='threading'  # 'threading' often works better for I/O bound tasks
            )(delayed(calculate_index)(idx) for idx in range(n_spaxel))
            
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
            logger.error(f"Error calculating spectral indices: {str(e)}")
            logger.setLevel(original_level)
            return {}
    
    def _calculate_spectral_indices_binned(self, indices_list=None, n_jobs=-1, verbose=False):
        """
        Calculate spectral indices for binned data using the same approach as pixel-to-pixel mode
        
        Parameters
        ----------
        indices_list : list of str, optional
            List of spectral indices to calculate, None uses the default set from config
        n_jobs : int, default=-1
            Number of parallel jobs
        verbose : bool, default=False
            Whether to display detailed information
                        
        Returns
        -------
        dict
            Dictionary of spectral indices
        """
        from spectral_indices import LineIndexCalculator, set_warnings
        import traceback
        import warnings
        
        # Import configuration manager for default indices
        try:
            from config_manager import get_spectral_indices
            
            # Define standard indices from config if not provided
            if indices_list is None:
                indices_list = get_spectral_indices()
                logger.info(f"Using default indices from config: {indices_list}")
        except ImportError:
            # Fallback if config_manager not available
            if indices_list is None:
                indices_list = ['Hbeta', 'Fe5015', 'Mgb']
                logger.info(f"Using hardcoded default indices: {indices_list}")
        
        # Control warnings
        set_warnings(verbose)
        
        try:
            # Initialize bin-level and pixel-level storage
            n_bins = self._n_bins
            self._bin_indices_result = {}
            
            for index_name in indices_list:
                self._bin_indices_result[index_name] = np.full(n_bins, np.nan)
            
            # Initialize pixel-based results
            self._spectral_indices = {}
            for index_name in indices_list:
                self._spectral_indices[index_name] = np.full((self._n_y, self._n_x), np.nan)
            
            # Check if we have emission-line fits
            has_emission_lines = (hasattr(self, '_bin_gas_bestfit') and 
                                self._bin_gas_bestfit is not None and 
                                np.any(np.isfinite(self._bin_gas_bestfit)))
            
            # Define function to calculate indices for a bin
            def process_bin(bin_idx):
                """Calculate spectral indices for a single bin, using the same logic as p2p"""
                # Skip bins without fits
                if not hasattr(self, '_bin_velocity') or np.isnan(self._bin_velocity[bin_idx]):
                    return bin_idx, {name: np.nan for name in indices_list}
                
                try:
                    # Get all data at once to minimize Python-level operations
                    observed_spectrum = self._binned_spectra[:, bin_idx]
                    
                    # Get optimal template - this should now include polynomial components
                    # if emission line fitting has been done
                    if hasattr(self, '_bin_optimal_tmpls') and self._bin_optimal_tmpls is not None:
                        optimal_template = self._bin_optimal_tmpls[:, bin_idx]
                    else:
                        # Fallback if no optimal template available
                        optimal_template = None
                            
                        # Try to compute it from weights if available (initial fitting)
                        if hasattr(self, '_bin_weights') and len(self._bin_weights) > bin_idx:
                            weights = self._bin_weights[bin_idx]
                            if hasattr(self, '_sps') and hasattr(self._sps, 'templates'):
                                # Basic optimal template from weights
                                optimal_template = np.dot(weights, self._sps.templates.T)
                                    
                                # Add polynomial if available
                                if hasattr(self, '_bin_poly_coeffs'):
                                    for b_idx, poly_coeff in self._bin_poly_coeffs:
                                        if b_idx == bin_idx:
                                            # Add polynomial evaluated on template wavelength grid
                                            template_poly = np.poly1d(poly_coeff)(self._sps.lam_temp)
                                            optimal_template += template_poly
                                            break
                        
                    stellar_velocity = self._bin_velocity[bin_idx]
                    
                    # Get gas velocity if available - this is the key change
                    gas_velocity = None
                    if hasattr(self, '_bin_emission_vel'):
                        # Try to find gas velocity from available emission lines
                        for line_name, vel_array in self._bin_emission_vel.items():
                            if bin_idx < len(vel_array) and np.isfinite(vel_array[bin_idx]):
                                gas_velocity = vel_array[bin_idx]
                                break
                    
                    # Get gas model if available - only once
                    gas_model = None
                    if has_emission_lines:
                        gas_model = self._bin_gas_bestfit[:, bin_idx]
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
                            # Use exactly the same parameters as in p2p method
                            calculator = LineIndexCalculator(
                                wave=self._binned_wavelength,           # Observation wavelength grid
                                flux=observed_spectrum,                 # Observed spectrum
                                fit_wave=self._sps.lam_temp if hasattr(self, '_sps') else self._binned_wavelength,   # Template wavelength grid
                                fit_flux=optimal_template,               # Template spectrum (now with polynomial)
                                em_wave=self._binned_wavelength if gas_model is not None else None,  # Emission line wavelength grid
                                em_flux_list=gas_model,                 # Emission line spectrum
                                velocity_correction=stellar_velocity,   # Stellar velocity correction
                                gas_velocity_correction=gas_velocity,   # Gas velocity correction - key change
                                continuum_mode='auto'                   # Auto select continuum mode
                            )
                        except Exception as e:
                            if verbose:
                                logger.warning(f"Error creating LineIndexCalculator for bin {bin_idx}: {e}")
                            return bin_idx, {index_name: np.nan for index_name in indices_list}
                        
                        # Calculate all required indices
                        result = {}
                        for index_name in indices_list:
                            try:
                                # Add extra logging for Hbeta
                                if index_name == 'Hbeta' and verbose:
                                    logger.info(f"Calculating Hbeta for bin {bin_idx}")
                                
                                index_value = calculator.calculate_index(index_name)
                                result[index_name] = index_value
                                
                                # Log the Hbeta result
                                if index_name == 'Hbeta' and verbose:
                                    logger.info(f"Bin {bin_idx} - Hbeta value: {index_value}")
                            except Exception as e:
                                if verbose:
                                    logger.debug(f"Error calculating {index_name} for bin {bin_idx}: {e}")
                                result[index_name] = np.nan
                                
                                # Special handling for Hbeta errors
                                if index_name == 'Hbeta':
                                    logger.warning(f"Failed to calculate Hbeta for bin {bin_idx}: {e}")
                        
                        return bin_idx, result
                        
                except Exception as e:
                    if verbose:
                        logger.debug(f"Error processing bin {bin_idx}: {e}\n{traceback.format_exc()}")
                    return bin_idx, {name: np.nan for name in indices_list}
            
            # Process bins in parallel
            from utils.parallel import ParallelTqdm
            
            results = ParallelTqdm(
                n_jobs=n_jobs, desc='Calculating spectral indices', total_tasks=n_bins
            )(delayed(process_bin)(bin_idx) for bin_idx in range(n_bins))
            
            # Process results
            for bin_idx, indices in results:
                if indices is None:
                    continue
                    
                # Store bin-level results
                for index_name, value in indices.items():
                    self._bin_indices_result[index_name][bin_idx] = value
                
                # Map to pixels
                if bin_idx in self._bin_pixel_map:
                    for pixel_idx in self._bin_pixel_map[bin_idx]:
                        row = pixel_idx // self._n_x
                        col = pixel_idx % self._n_x
                        
                        if 0 <= row < self._n_y and 0 <= col < self._n_x:
                            for index_name, value in indices.items():
                                self._spectral_indices[index_name][row, col] = value
            
            # Return combined results
            return {
                'bin_indices': self._bin_indices_result,
                'pixel_indices': self._spectral_indices
            }
            
        except Exception as e:
            logger.error(f"Error in binned spectral indices calculation: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def plot_spectral_indices(self, spaxel_position=None, mode="MUSE", number=0, save_path=None):
        """
        Plot spectral indices for a specific spaxel
        
        Parameters
        ----------
        spaxel_position : tuple of int, optional
            (row, col) for the spaxel to plot, defaults to central spaxel
        mode : str, default="MUSE"
            Mode identifier for plot title
        number : int, default=0
            Number identifier for the plot
        save_path : str, optional
            Path to save the plot
                
        Returns
        -------
        matplotlib.figure.Figure
            Figure with spectral indices plot
        """
        # Import LineIndexCalculator
        from spectral_indices import LineIndexCalculator
        
        try:
            # Check if spectral indices have been calculated
            if not hasattr(self, '_index_calculators') or not self._index_calculators:
                logger.warning("Spectral indices calculators not available. Run calculate_spectral_indices() first.")
                return None
            
            # Determine which calculator to use
            calculator = None
            if spaxel_position is None:
                # Use central position
                if 'central' in self._index_calculators:
                    calculator = self._index_calculators['central']
                else:
                    # Find first available calculator
                    calculator = next(iter(self._index_calculators.values()), None)
            else:
                # Try to find calculator for this position
                row, col = spaxel_position
                key = f"sample_{row}_{col}"
                if key in self._index_calculators:
                    calculator = self._index_calculators[key]
                else:
                    # If not found, create one
                    spaxel_idx = np.ravel_multi_index((row, col), (self._n_y, self._n_x))
                    
                    # Skip if position is invalid
                    if (row < 0 or row >= self._n_y or col < 0 or col >= self._n_x or
                        np.isnan(self._velocity_field[row, col])):
                        logger.warning(f"Invalid spaxel position: ({row}, {col})")
                        return None
                    
                    # Get observed spectrum
                    observed_spectrum = self._spectra[:, spaxel_idx]
                    
                    # Get optimal template on original wavelength grid
                    optimal_template = self._optimal_tmpls[:, row, col]
                    
                    # Get stellar velocity
                    stellar_velocity = self._velocity_field[row, col]
                    
                    # Get gas velocity if available - this is the key change
                    gas_velocity = None
                    if hasattr(self, '_emission_vel'):
                        # Try to find gas velocity from available emission lines
                        for line_name, vel_map in self._emission_vel.items():
                            if np.isfinite(vel_map[row, col]):
                                gas_velocity = vel_map[row, col]
                                break
                    
                    # Skip if template is not valid
                    if not np.any(np.isfinite(optimal_template)):
                        logger.warning(f"No valid template for position ({row}, {col})")
                        return None
                    
                    # Get gas model if available
                    gas_model = None
                    if hasattr(self, '_gas_bestfit_field') and self._gas_bestfit_field is not None:
                        gas_model = self._gas_bestfit_field[:, row, col]
                        # Verify gas model is valid
                        if not np.any(np.isfinite(gas_model)) or np.all(gas_model == 0):
                            gas_model = None
                    
                    # Create LineIndexCalculator for this position
                    try:
                        calculator = LineIndexCalculator(
                            wave=self._lambda_gal,  
                            flux=observed_spectrum,  
                            fit_wave=self._sps.lam_temp,
                            fit_flux=optimal_template,
                            em_wave=self._lambda_gal if gas_model is not None else None,
                            em_flux_list=gas_model,
                            velocity_correction=stellar_velocity,
                            gas_velocity_correction=gas_velocity,  # Pass gas velocity separately
                            continuum_mode='auto'
                        )
                    except Exception as e:
                        logger.warning(f"Error creating LineIndexCalculator for position ({row}, {col}): {str(e)}")
                        return None
            
            if calculator is None:
                logger.warning("Could not find or create a LineIndexCalculator for the specified position.")
                return None
            
            # Use the calculator's plot_all_lines method
            try:
                fig, axes = calculator.plot_all_lines(
                    mode=mode, 
                    number=number,
                    save_path=save_path,
                    show_index=True
                )
                
                return fig, axes
            except Exception as e:
                logger.error(f"Error plotting spectral lines: {str(e)}")
                return None
        
        except Exception as e:
            logger.error(f"Error in plot_spectral_indices: {str(e)}")
            return None

    def plot_emission_maps(self, emission_lines=None, cmap='viridis', figsize=(15, 10), dpi=100, save_path=None):
        """
        Plot 2D maps of emission line flux, velocity, and velocity dispersion
        
        Parameters
        ----------
        emission_lines : list of str, optional
            List of emission lines to plot, if None, use all available
        cmap : str, default='viridis'
            Colormap to use for the plots
        figsize : tuple, default=(15, 10)
            Figure size
        dpi : int, default=100
            Figure DPI
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure object
        """
        try:
            # Check if emission line fitting has been performed
            if self._emission_flux is None or not self._emission_flux:
                raise ValueError("No emission lines have been fitted. Run fit_emission_lines() first.")
            
            # Get list of available emission lines
            if emission_lines is None:
                emission_lines = list(self._emission_flux.keys())
            else:
                # Only keep lines that are available
                emission_lines = [line for line in emission_lines if line in self._emission_flux]
            
            if not emission_lines:
                raise ValueError("No valid emission lines to plot")
            
            # Create figure
            n_lines = len(emission_lines)
            n_rows = n_lines
            n_cols = 3  # flux, velocity, dispersion
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, 
                                   squeeze=False, constrained_layout=True)
            
            # Title information
            filename = Path(self._filename).name
            fig.suptitle(f"Emission Line Maps: {filename}", fontsize=16)
            
            # Column titles
            col_titles = ['Flux', 'Velocity [km/s]', 'Dispersion [km/s]']
            for col, title in enumerate(col_titles):
                axes[0, col].set_title(title, fontsize=14)
            
            # Plot each emission line
            for row, line_name in enumerate(emission_lines):
                # Get data
                flux = self._emission_flux.get(line_name, np.full((self._n_y, self._n_x), np.nan))
                vel = self._emission_vel.get(line_name, np.full((self._n_y, self._n_x), np.nan))
                sig = self._emission_sig.get(line_name, np.full((self._n_y, self._n_x), np.nan))
                
                # Row label
                wavelength = self._emission_wavelength.get(line_name, 0)
                if wavelength > 0:
                    row_label = f"{line_name} ({wavelength:.1f}Å)"
                else:
                    row_label = line_name
                axes[row, 0].set_ylabel(row_label, fontsize=12)
                
                # Create masks for valid data
                flux_mask = ~np.isnan(flux) & (flux > 0)
                vel_mask = ~np.isnan(vel) & flux_mask
                sig_mask = ~np.isnan(sig) & flux_mask
                
                # Plot flux (log scale)
                if np.any(flux_mask):
                    log_flux = np.log10(flux)
                    vmin = np.nanpercentile(log_flux[flux_mask], 5)
                    vmax = np.nanpercentile(log_flux[flux_mask], 95)
                    im0 = axes[row, 0].imshow(log_flux, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
                    # Add colorbar
                    plt.colorbar(im0, ax=axes[row, 0], label='log(Flux)')
                else:
                    axes[row, 0].text(0.5, 0.5, 'No Valid Data', horizontalalignment='center',
                                   verticalalignment='center', transform=axes[row, 0].transAxes)
                
                # Plot velocity
                if np.any(vel_mask):
                    # Use symmetric colormap for velocity
                    vel_cmap = 'coolwarm'
                    vel_abs_max = np.nanpercentile(np.abs(vel[vel_mask]), 95)
                    im1 = axes[row, 1].imshow(vel, origin='lower', cmap=vel_cmap, 
                                           vmin=-vel_abs_max, vmax=vel_abs_max)
                    plt.colorbar(im1, ax=axes[row, 1], label='Velocity [km/s]')
                else:
                    axes[row, 1].text(0.5, 0.5, 'No Valid Data', horizontalalignment='center',
                                   verticalalignment='center', transform=axes[row, 1].transAxes)
                
                # Plot sigma
                if np.any(sig_mask):
                    vmin = np.nanpercentile(sig[sig_mask], 5)
                    vmax = np.nanpercentile(sig[sig_mask], 95)
                    im2 = axes[row, 2].imshow(sig, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
                    plt.colorbar(im2, ax=axes[row, 2], label='Dispersion [km/s]')
                else:
                    axes[row, 2].text(0.5, 0.5, 'No Valid Data', horizontalalignment='center',
                                   verticalalignment='center', transform=axes[row, 2].transAxes)
            
            # Remove axis ticks for cleaner appearance
            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Save figure if path provided
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            
            return fig, axes
        
        except Exception as e:
            logger.error(f"Error plotting emission maps: {str(e)}")
            return None

    def plot_fit(self, row, col, plot_range=None, show_indices=True, figsize=(12, 8), dpi=100):
        """
        Plot fitting results for a specific spaxel
        
        Parameters
        ----------
        row : int
            Row index of spaxel
        col : int
            Column index of spaxel
        plot_range : tuple of float, optional
            Wavelength range to plot [min, max], None uses full range
        show_indices : bool, default=True
            Whether to show spectral indices values
        figsize : tuple of float, default=(12, 8)
            Figure size
        dpi : int, default=100
            Figure DPI
            
        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure object
        """
        try:
            # Check if fitting has been performed
            if self._velocity_field is None:
                raise ValueError("Must run fit_spectra() before plot_fit()")
            
            # Check if position is valid
            if row < 0 or row >= self._n_y or col < 0 or col >= self._n_x:
                raise ValueError(f"Invalid position: ({row}, {col}), shape is ({self._n_y}, {self._n_x})")
            
            # Check if velocity is valid at this position
            if np.isnan(self._velocity_field[row, col]):
                raise ValueError(f"No valid fit at position ({row}, {col})")
            
            # Get spectrum and fit
            spaxel_idx = np.ravel_multi_index((row, col), (self._n_y, self._n_x))
            spectrum = self._spectra[:, spaxel_idx]
            bestfit = self._bestfit_field[:, row, col]
            
            # Get gas model and stellar model if available
            has_gas_component = (self._gas_bestfit_field is not None and 
                                not np.all(np.isnan(self._gas_bestfit_field[:, row, col])))
                                
            if has_gas_component:
                gas_model = self._gas_bestfit_field[:, row, col]
                # Stellar model is the total best-fit minus gas
                stellar_model = bestfit - gas_model
                # Full model is bestfit (already includes both)
                full_model = bestfit
            else:
                gas_model = np.zeros_like(bestfit)
                stellar_model = bestfit
                full_model = bestfit
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            # Set wavelength range
            if plot_range is None:
                # Get min/max wavelength with some margin
                margin = 50  # Angstroms
                plot_range = [np.min(self._lambda_gal) + margin, np.max(self._lambda_gal) - margin]
                
                # Check if we have emission lines to include in view
                if hasattr(self, '_emission_wavelength') and self._emission_wavelength:
                    emission_waves = list(self._emission_wavelength.values())
                    if emission_waves:
                        min_em = min(emission_waves) - 50
                        max_em = max(emission_waves) + 50
                        if min_em > plot_range[0] and min_em < plot_range[1]:
                            plot_range[0] = min_em - 100
                        if max_em < plot_range[1] and max_em > plot_range[0]:
                            plot_range[1] = max_em + 100
            
            # Find wavelength indices within range
            wave_idx = np.where((self._lambda_gal >= plot_range[0]) & (self._lambda_gal <= plot_range[1]))[0]
            
            # Check if any wavelength points in range
            if len(wave_idx) == 0:
                raise ValueError(f"No wavelength points in range {plot_range}")
            
            # Calculate valid range for y-axis
            valid_flux = np.concatenate([
                spectrum[wave_idx][np.isfinite(spectrum[wave_idx])],
                full_model[wave_idx][np.isfinite(full_model[wave_idx])],
                stellar_model[wave_idx][np.isfinite(stellar_model[wave_idx])]
            ])
            
            if len(valid_flux) == 0:
                raise ValueError("No valid flux values to plot")
                
            ymin = np.percentile(valid_flux, 1)
            ymax = np.percentile(valid_flux, 99)
            yrange = ymax - ymin
            
            # Plot data
            ax.plot(self._lambda_gal[wave_idx], spectrum[wave_idx], 'k-', label='Data', lw=1.5)
            
            # Plot full model
            ax.plot(self._lambda_gal[wave_idx], full_model[wave_idx], 'r-', label='Best Fit', lw=1.5)
            
            # Plot stellar model
            ax.plot(self._lambda_gal[wave_idx], stellar_model[wave_idx], 'b-', label='Stellar Component', lw=1.5)
            
            # Plot gas component if available and not all zeros
            if has_gas_component and not np.all(gas_model[wave_idx] == 0):
                # Plot gas component
                ax.plot(self._lambda_gal[wave_idx], gas_model[wave_idx], 'g-', label='Gas Component', lw=1.5)
                
                # Annotate emission lines
                if hasattr(self, '_emission_wavelength') and self._emission_wavelength:
                    y_range = ax.get_ylim()
                    y_pos = y_range[0] + 0.1 * (y_range[1] - y_range[0])
                    
                    for name, wavelength in self._emission_wavelength.items():
                        if plot_range[0] <= wavelength <= plot_range[1]:
                            # Only show lines within plot range
                            ax.axvline(x=wavelength, color='g', linestyle='--', alpha=0.5)
                            ax.text(wavelength, y_pos, name, rotation=90, 
                                    verticalalignment='bottom', horizontalalignment='center',
                                    fontsize=8, alpha=0.7)
            
            # Plot residuals
            residual = spectrum - full_model
            ax.plot(self._lambda_gal[wave_idx], residual[wave_idx] + ymin - 0.2*yrange, 'c-', label='Residual', lw=1)
            ax.axhline(y=ymin - 0.2*yrange, color='k', linestyle=':', lw=0.5)
            
            # Add spectral indices if requested
            if show_indices and hasattr(self, '_spectral_indices') and self._spectral_indices:
                # Define bandpass definitions for display
                index_bands = {
                    'Hbeta': {'blue': (4827.9, 4847.9), 'band': (4847.9, 4876.6), 'red': (4876.6, 4891.6)},
                    'Mgb': {'blue': (5142.6, 5161.4), 'band': (5160.1, 5192.6), 'red': (5191.4, 5206.4)},
                    'Fe5270': {'blue': (5233.2, 5248.2), 'band': (5245.7, 5285.7), 'red': (5285.7, 5318.2)},
                    'Fe5335': {'blue': (5304.6, 5315.9), 'band': (5312.1, 5352.1), 'red': (5353.4, 5363.4)},
                    'Fe5015': {'blue': (4946.5, 4977.8), 'band': (4977.8, 5054.0), 'red': (5054.0, 5065.3)},
                    'D4000': {'blue': (3750.0, 3950.0), 'red': (4050.0, 4250.0)}
                }
                
                # Get spectral index values
                index_text = []
                ymin, ymax = ax.get_ylim()
                
                for index_name, index_map in self._spectral_indices.items():
                    index_value = index_map[row, col]
                    if not np.isnan(index_value):
                        # Format the value with two decimal places
                        index_text.append(f"{index_name}: {index_value:.2f}")
                        
                        # If the index bands are in the plotted range, highlight them
                        if index_name in index_bands:
                            bands = index_bands[index_name]
                            
                            # For regular indices (not D4000)
                            if 'band' in bands:
                                blue_band = bands['blue']
                                index_band = bands['band']
                                red_band = bands['red']
                                
                                # Check if bands are in plot range
                                if (blue_band[0] >= plot_range[0] and blue_band[1] <= plot_range[1] and
                                    index_band[0] >= plot_range[0] and index_band[1] <= plot_range[1] and
                                    red_band[0] >= plot_range[0] and red_band[1] <= plot_range[1]):
                                    
                                    # Highlight bands with semi-transparent rectangles
                                    ax.axvspan(blue_band[0], blue_band[1], alpha=0.2, color='blue')
                                    ax.axvspan(index_band[0], index_band[1], alpha=0.2, color='green')
                                    ax.axvspan(red_band[0], red_band[1], alpha=0.2, color='red')
                            
                            # For D4000 which uses two bands
                            elif 'blue' in bands and 'red' in bands:
                                blue_band = bands['blue']
                                red_band = bands['red']
                                
                                # Check if bands are in plot range
                                if (blue_band[0] >= plot_range[0] and blue_band[1] <= plot_range[1] and
                                    red_band[0] >= plot_range[0] and red_band[1] <= plot_range[1]):
                                    
                                    # Highlight bands with semi-transparent rectangles
                                    ax.axvspan(blue_band[0], blue_band[1], alpha=0.2, color='blue')
                                    ax.axvspan(red_band[0], red_band[1], alpha=0.2, color='red')
                
                # Add text with spectral index values
                if index_text:
                    ax.text(0.02, 0.95, '\n'.join(index_text), 
                            transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add kinematic information
            vel = self._velocity_field[row, col]
            disp = self._dispersion_field[row, col]
            
            # Get emission line velocities if available
            em_vel_text = ""
            if self._emission_vel:
                for line_name, vel_map in self._emission_vel.items():
                    line_vel = vel_map[row, col]
                    line_sigma = self._emission_sig.get(line_name, np.zeros_like(vel_map))[row, col]
                    line_flux = self._emission_flux.get(line_name, np.zeros_like(vel_map))[row, col]
                    
                    if not np.isnan(line_vel) and not np.isnan(line_flux) and line_flux > 0:
                        em_vel_text += f"{line_name}: v={line_vel:.1f}, σ={line_sigma:.1f}, flux={line_flux:.1e}\n"
            
            # Create info text
            info_text = f"Spaxel: ({row}, {col})\n"
            info_text += f"Stellar Vel: {vel:.1f} km/s\n"
            info_text += f"Stellar Disp: {disp:.1f} km/s\n"
            if em_vel_text:
                info_text += "\nEmission Lines:\n" + em_vel_text
            
            # Add text box with info
            ax.text(0.98, 0.95, info_text, 
                    transform=ax.transAxes, fontsize=10,
                    horizontalalignment='right', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Add labels and legend
            ax.set_xlabel('Wavelength (Å)', fontsize=12)
            ax.set_ylabel('Flux', fontsize=12)
            ax.set_title(f'Spectral Fit - Spaxel ({row}, {col})', fontsize=14)
            ax.legend(loc='lower right')
            
            # Set wavelength range
            ax.set_xlim(plot_range)
            
            # Set y-axis limits with safeguards
            ax.set_ylim(ymin - 0.3*yrange, ymax + 0.1*yrange)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting fit for spaxel ({row}, {col}): {str(e)}")
            return None

    def plot_bin_fit(self, bin_idx, wavelength_range=None, figsize=(12, 6), save_path=None):
        """
        Plot the spectrum and fit for a specific bin in binned mode
        
        Parameters
        ----------
        bin_idx : int
            Bin index to plot
        wavelength_range : tuple, optional
            Wavelength range to plot [min, max]
        figsize : tuple, default=(12, 6)
            Figure size
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        tuple
            (fig, axes) - Figure and axes objects
        """
        import visualization
        
        if not hasattr(self, '_is_binned') or not self._is_binned:
            logger.warning("Not in binned mode, can't plot bin fit")
            return None
        
        if not hasattr(self, '_bin_bestfit') or self._bin_bestfit is None:
            logger.warning("No bin fitting results available, run fit_spectra() first")
            return None
        
        if bin_idx < 0 or bin_idx >= self._n_bins:
            logger.warning(f"Invalid bin index {bin_idx}, valid range is 0-{self._n_bins-1}")
            return None
        
        try:
            # Get gas best fit if available
            bin_gas_bestfit = self._bin_gas_bestfit if hasattr(self, '_bin_gas_bestfit') else None
            
            # Get stellar best fit if directly available, or calculate as total - gas
            if hasattr(self, '_bin_stellar_bestfit'):
                bin_stellar_bestfit = self._bin_stellar_bestfit
            elif bin_gas_bestfit is not None:
                # Calculate stellar = total - gas
                bin_stellar_bestfit = self._bin_bestfit - bin_gas_bestfit
            else:
                bin_stellar_bestfit = None
            
            # Get wavelength range for plot
            if wavelength_range is None:
                # Default to full wavelength range
                wavelength_range = (
                    np.min(self._binned_wavelength), 
                    np.max(self._binned_wavelength)
                )
                
                # Try to find interesting spectral features to include
                if hasattr(self, '_emission_wavelength') and self._emission_wavelength:
                    # Find emission line wavelengths
                    emission_waves = list(self._emission_wavelength.values())
                    if emission_waves:
                        # Calculate range to include main emission lines
                        em_min = min(emission_waves) - 100
                        em_max = max(emission_waves) + 100
                        
                        # Use emission line range if it's within our data
                        if (em_min > wavelength_range[0] and 
                            em_max < wavelength_range[1] and 
                            em_max - em_min > 200):
                            wavelength_range = (em_min, em_max)
            
            # Create title with bin info
            title = f"Bin {bin_idx}"
            
            # Add velocity and dispersion if available
            if hasattr(self, '_bin_velocity') and len(self._bin_velocity) > bin_idx:
                vel = self._bin_velocity[bin_idx]
                disp = self._bin_dispersion[bin_idx]
                if np.isfinite(vel) and np.isfinite(disp):
                    title += f" - V={vel:.1f} km/s, σ={disp:.1f} km/s"
            
            # If this is a radial bin, add radius information
            if hasattr(self, '_bin_radii') and len(self._bin_radii) > bin_idx:
                radius = self._bin_radii[bin_idx]
                if np.isfinite(radius):
                    title += f" - Radius={radius:.1f} arcsec"
            
            # Call visualization function
            fig, axes = visualization.plot_bin_spectrum_fit(
                self._binned_wavelength,
                bin_idx,
                self._binned_spectra,
                self._bin_bestfit,
                bin_gas_bestfit=bin_gas_bestfit,
                bin_stellar_bestfit=bin_stellar_bestfit,
                title=title,
                plot_range=wavelength_range,
                figsize=figsize
            )
            
            # Save figure if path provided
            if save_path is not None:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved bin spectrum plot to {save_path}")
            
            return fig, axes
        
        except Exception as e:
            logger.error(f"Error plotting bin fit: {e}")
            return None

    def plot_bin_fits(self, bin_indices=None, n_bins=5, wavelength_range=None, 
                    figsize=(12, 6), save_dir=None):
        """
        Plot fits for multiple bins
        
        Parameters
        ----------
        bin_indices : list, optional
            List of bin indices to plot. If None, will use evenly spaced bins.
        n_bins : int, default=5
            Number of bins to plot if bin_indices is None
        wavelength_range : tuple, optional
            Wavelength range to plot [min, max]
        figsize : tuple, default=(12, 6)
            Figure size
        save_dir : str, optional
            Directory to save figures (will create if doesn't exist)
            
        Returns
        -------
        list
            List of (fig, axes) tuples
        """
        if not hasattr(self, '_is_binned') or not self._is_binned:
            logger.warning("Not in binned mode, can't plot bin fits")
            return []
        
        if not hasattr(self, '_bin_bestfit') or self._bin_bestfit is None:
            logger.warning("No bin fitting results available, run fit_spectra() first")
            return []
        
        # Determine which bins to plot
        if bin_indices is None:
            # Choose evenly spaced bins
            valid_bins = []
            for i in range(self._n_bins):
                if (hasattr(self, '_bin_velocity') and 
                    i < len(self._bin_velocity) and 
                    np.isfinite(self._bin_velocity[i])):
                    valid_bins.append(i)
            
            if len(valid_bins) == 0:
                logger.warning("No valid bins with fits found")
                return []
                
            # Take evenly spaced indices
            if len(valid_bins) <= n_bins:
                bin_indices = valid_bins
            else:
                step = len(valid_bins) // n_bins
                bin_indices = valid_bins[::step][:n_bins]
        
        # Create save directory if provided
        if save_dir is not None:
            import os
            from pathlib import Path
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
        
        # Plot each bin
        results = []
        for bin_idx in bin_indices:
            if bin_idx < 0 or bin_idx >= self._n_bins:
                continue
                
            # Create save path if needed
            if save_dir is not None:
                if hasattr(self, '_bin_type'):
                    # Include bin type in filename
                    save_file = f"{self._bin_type}_bin_{bin_idx}_fit.png"
                else:
                    save_file = f"bin_{bin_idx}_fit.png"
                save_path_bin = os.path.join(save_dir, save_file)
            else:
                save_path_bin = None
            
            # Plot bin fit
            result = self.plot_bin_fit(
                bin_idx, 
                wavelength_range=wavelength_range,
                figsize=figsize,
                save_path=save_path_bin
            )
            
            if result is not None:
                fig, axes = result
                results.append(result)
                
                # Close the figure after adding to results to prevent memory leaks
                plt.close(fig)
        
        return results

    def plot_bin_analysis_results(self, output_dir=None):
        """
        Create comprehensive plots for binned analysis results
        
        Parameters
        ----------
        output_dir : str or Path, optional
            Directory to save plots. If None, plots are displayed but not saved.
            
        Returns
        -------
        dict
            Dictionary of created figures
        """
        import visualization
        from pathlib import Path
        
        if not hasattr(self, '_is_binned') or not self._is_binned:
            logger.warning("Not in binned mode, can't create binned analysis plots")
            return {}
        
        # Create output directory if provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # Store created figures
        figures = {}
        
        try:
            # 1. Bin Map
            if hasattr(self, '_bin_num') and self._bin_num is not None:
                try:
                    # Create bin map figure
                    fig, ax = plt.subplots(figsize=(8, 7))
                    
                    # Create binning map
                    ax.imshow(self._bin_num.reshape(self._n_y, self._n_x), 
                            origin='lower', cmap='tab20', aspect='equal')
                    
                    ax.set_title(f"{self._bin_type} Binning Map" if hasattr(self, '_bin_type') else "Binning Map")
                    
                    if output_dir is not None:
                        fig.savefig(output_dir / "bin_map.png", dpi=150, bbox_inches='tight')
                    
                    figures['bin_map'] = (fig, ax)
                except Exception as e:
                    logger.warning(f"Error creating bin map: {e}")
            
            # 2. Kinematics
            if (hasattr(self, '_bin_velocity') and hasattr(self, '_bin_dispersion') and 
                self._bin_velocity is not None and self._bin_dispersion is not None and
                hasattr(self, '_bin_num') and self._bin_num is not None):
                try:
                    # Create kinematics figure
                    fig, axes = visualization.plot_bin_kinematics(
                        self._bin_num.reshape(self._n_y, self._n_x),
                        self._bin_velocity,
                        self._bin_dispersion,
                        title=self._bin_type if hasattr(self, '_bin_type') else None
                    )
                    
                    if output_dir is not None:
                        fig.savefig(output_dir / "kinematics.png", dpi=150, bbox_inches='tight')
                    
                    figures['kinematics'] = (fig, axes)
                except Exception as e:
                    logger.warning(f"Error creating kinematics plot: {e}")
            
            # 3. LIC Visualization
            if (hasattr(self, '_bin_velocity') and self._bin_velocity is not None and
                hasattr(self, '_bin_num') and self._bin_num is not None):
                try:
                    # Create LIC figure
                    fig, ax = visualization.plot_bin_lic(
                        self._bin_num.reshape(self._n_y, self._n_x),
                        self._bin_velocity,
                        title=f"{self._bin_type} Velocity Field" if hasattr(self, '_bin_type') else "Velocity Field"
                    )
                    
                    if output_dir is not None:
                        fig.savefig(output_dir / "velocity_lic.png", dpi=150, bbox_inches='tight')
                    
                    figures['velocity_lic'] = (fig, ax)
                except Exception as e:
                    logger.warning(f"Error creating LIC plot: {e}")
            
            # 4. Emission Lines
            if (hasattr(self, '_bin_emission_flux') and self._bin_emission_flux and
                hasattr(self, '_bin_num') and self._bin_num is not None):
                try:
                    # Create emission line figures
                    fig, axes = visualization.plot_bin_emission_lines(
                        self._bin_num.reshape(self._n_y, self._n_x),
                        self._bin_emission_flux,
                        title=f"{self._bin_type} Emission Lines" if hasattr(self, '_bin_type') else "Emission Lines"
                    )
                    
                    if fig is not None and output_dir is not None:
                        fig.savefig(output_dir / "emission_lines.png", dpi=150, bbox_inches='tight')
                    
                    figures['emission_lines'] = (fig, axes)
                except Exception as e:
                    logger.warning(f"Error creating emission line plots: {e}")
            
            # 5. Spectral Indices
            if ((hasattr(self, '_bin_indices_result') and self._bin_indices_result) or 
                (hasattr(self, 'bin_indices') and self.bin_indices)):
                try:
                    # Get indices dict - try both possible attributes
                    if hasattr(self, '_bin_indices_result') and self._bin_indices_result:
                        indices_dict = self._bin_indices_result
                    elif hasattr(self, 'bin_indices') and self.bin_indices:
                        indices_dict = self.bin_indices
                    else:
                        indices_dict = {}
                        
                    # Check if we have any indices to plot
                    if indices_dict and isinstance(indices_dict, dict):
                        # Create spectral indices figures
                        bin_map = self._bin_num.reshape(self._n_y, self._n_x)
                        
                        # Get index names
                        index_names = list(indices_dict.keys())
                        if not index_names:
                            raise ValueError("No spectral indices available")
                            
                        # Limit to 6 indices for display
                        index_names = index_names[:6]
                        n_indices = len(index_names)
                        
                        # Calculate figure layout
                        if n_indices <= 3:
                            nrows, ncols = 1, n_indices
                        else:
                            nrows = (n_indices + 2) // 3  # Ceiling division
                            ncols = min(3, n_indices)
                        
                        # Create figure
                        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
                        
                        # Handle different axes arrangements
                        if n_indices == 1:
                            axes = np.array([axes])
                        
                        # Ensure axes is array-like for iteration
                        axes = np.atleast_1d(axes)
                        
                        # Plot each index
                        for i, index_name in enumerate(index_names):
                            if i < axes.size:  # Ensure we have an axis for this index
                                # Get axis (handle both 1D and 2D axes arrays)
                                if axes.ndim == 1:
                                    ax = axes[i]
                                else:
                                    ax = axes.flat[i]
                                    
                                # Get values for this index
                                values = indices_dict[index_name]
                                
                                # Use safe plotting function
                                visualization.safe_plot_array(
                                    values, bin_map, ax=ax,
                                    title=index_name,
                                    cmap='plasma',
                                    label='Index Value'
                                )
                        
                        # Hide any unused axes
                        for i in range(n_indices, axes.size):
                            if axes.ndim == 1 and i < len(axes):
                                axes[i].axis('off')
                            elif axes.ndim > 1:
                                axes.flat[i].axis('off')
                        
                        # Add title
                        fig.suptitle(f"{self._bin_type} Spectral Indices" if hasattr(self, '_bin_type') else "Spectral Indices", 
                                fontsize=16)
                        
                        # Adjust layout
                        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for title
                        
                        # Save figure
                        if output_dir is not None:
                            fig.savefig(output_dir / "spectral_indices.png", dpi=150, bbox_inches='tight')
                        
                        figures['spectral_indices'] = (fig, axes)
                except Exception as e:
                    logger.warning(f"Error creating spectral indices plots: {e}")
            
            # 6. Sample of bin spectra
            if (hasattr(self, '_bin_bestfit') and self._bin_bestfit is not None):
                try:
                    # Create directory for bin spectra plots
                    if output_dir is not None:
                        bin_plots_dir = output_dir / "bin_spectra"
                        bin_plots_dir.mkdir(exist_ok=True, parents=True)
                    else:
                        bin_plots_dir = None
                    
                    # Plot a sample of bin spectra
                    bin_figs = self.plot_bin_fits(n_bins=min(5, self._n_bins), save_dir=bin_plots_dir)
                    
                    # Store figures
                    figures['bin_spectra'] = bin_figs
                except Exception as e:
                    logger.warning(f"Error creating bin spectra plots: {e}")
            
            # 7. Radial Profiles (for RDB only)
            if (hasattr(self, '_bin_type') and self._bin_type == 'RDB' and
                hasattr(self, '_bin_radii') and self._bin_radii is not None):
                try:
                    # Velocity and dispersion profiles
                    if hasattr(self, '_bin_velocity') and hasattr(self, '_bin_dispersion'):
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Plot velocity profile
                        if np.any(np.isfinite(self._bin_velocity)):
                            valid = np.isfinite(self._bin_velocity) & np.isfinite(self._bin_radii)
                            ax1.plot(self._bin_radii[valid], self._bin_velocity[valid], 'o-')
                            ax1.set_xlabel('Radius (arcsec)')
                            ax1.set_ylabel('Velocity (km/s)')
                            ax1.set_title('Radial Velocity Profile')
                            ax1.grid(True, alpha=0.3)
                        else:
                            ax1.text(0.5, 0.5, "No valid velocity data", 
                                ha='center', va='center', transform=ax1.transAxes)
                        
                        # Plot dispersion profile
                        if np.any(np.isfinite(self._bin_dispersion)):
                            valid = np.isfinite(self._bin_dispersion) & np.isfinite(self._bin_radii)
                            ax2.plot(self._bin_radii[valid], self._bin_dispersion[valid], 'o-')
                            ax2.set_xlabel('Radius (arcsec)')
                            ax2.set_ylabel('Dispersion (km/s)')
                            ax2.set_title('Radial Dispersion Profile')
                            ax2.grid(True, alpha=0.3)
                        else:
                            ax2.text(0.5, 0.5, "No valid dispersion data", 
                                ha='center', va='center', transform=ax2.transAxes)
                        
                        plt.tight_layout()
                        
                        if output_dir is not None:
                            fig.savefig(output_dir / "radial_kinematics.png", dpi=150, bbox_inches='tight')
                        
                        figures['radial_kinematics'] = (fig, (ax1, ax2))
                    
                    # Spectral indices profiles
                    if hasattr(self, '_bin_indices_result') and self._bin_indices_result:
                        # Get indices
                        index_names = list(self._bin_indices_result.keys())
                        
                        if index_names:
                            # Limit to 6 indices
                            index_names = index_names[:6]
                            n_indices = len(index_names)
                            
                            # Create figure
                            fig, axes = plt.subplots(n_indices, 1, figsize=(8, 3*n_indices))
                            if n_indices == 1:
                                axes = [axes]
                            
                            # Plot each index
                            for i, index_name in enumerate(index_names):
                                index_values = self._bin_indices_result[index_name]
                                valid = np.isfinite(index_values) & np.isfinite(self._bin_radii)
                                
                                if np.any(valid):
                                    axes[i].plot(self._bin_radii[valid], index_values[valid], 'o-')
                                    axes[i].set_xlabel('Radius (arcsec)')
                                    axes[i].set_ylabel('Index Value')
                                    axes[i].set_title(f'{index_name} Profile')
                                    axes[i].grid(True, alpha=0.3)
                                else:
                                    axes[i].text(0.5, 0.5, f"No valid {index_name} data", 
                                            ha='center', va='center', transform=axes[i].transAxes)
                            
                            plt.tight_layout()
                            
                            if output_dir is not None:
                                fig.savefig(output_dir / "radial_indices.png", dpi=150, bbox_inches='tight')
                            
                            figures['radial_indices'] = (fig, axes)
                except Exception as e:
                    logger.warning(f"Error creating radial profile plots: {e}")
            
            return figures
        
        except Exception as e:
            logger.error(f"Error creating binned analysis plots: {e}")
            return figures

    def plot_bin_index_calculation(self, bin_idx, save_dir=None):
        """
        Plot the spectral index calculation visualization for a bin
        
        Parameters
        ----------
        bin_idx : int
            Bin index to plot
        save_dir : str, optional
            Directory to save the plot
            
        Returns
        -------
        tuple
            (fig, axes) - Figure and axes objects
        """
        if not hasattr(self, '_is_binned') or not self._is_binned:
            logger.warning("Not in binned mode, can't plot bin indices")
            return None
        
        if bin_idx < 0 or bin_idx >= self._n_bins:
            logger.warning(f"Invalid bin index: {bin_idx}")
            return None
        
        try:
            # Get bin data
            bin_spectrum = self._binned_spectra[:, bin_idx]
            
            # Skip bins with insufficient data
            if not np.any(np.isfinite(bin_spectrum)):
                logger.warning(f"No valid data for bin {bin_idx}")
                return None
            
            # Get stellar velocity
            stellar_velocity = 0.0
            if hasattr(self, '_bin_velocity') and bin_idx < len(self._bin_velocity):
                stellar_velocity = self._bin_velocity[bin_idx]
                if not np.isfinite(stellar_velocity):
                    stellar_velocity = 0.0
            
            # Get gas velocity if available - this is the key change
            gas_velocity = None
            if hasattr(self, '_bin_emission_vel'):
                # Try to find gas velocity from available emission lines
                for line_name, vel_array in self._bin_emission_vel.items():
                    if bin_idx < len(vel_array) and np.isfinite(vel_array[bin_idx]):
                        gas_velocity = vel_array[bin_idx]
                        break
            
            # Get optimal template - CRITICAL: This must match how it's done in index calculation
            optimal_template = None
            if hasattr(self, '_bin_optimal_tmpls') and self._bin_optimal_tmpls is not None:
                optimal_template = self._bin_optimal_tmpls[:, bin_idx]
            else:
                # Try to compute it from weights if available (from initial fitting)
                if hasattr(self, '_bin_weights') and len(self._bin_weights) > bin_idx:
                    weights = self._bin_weights[bin_idx]
                    if hasattr(self, '_sps') and hasattr(self._sps, 'templates'):
                        # Basic optimal template from weights
                        optimal_template = np.dot(weights, self._sps.templates.T)
                        
                        # Add polynomial if available (from emission line fitting)
                        if hasattr(self, '_bin_poly_coeffs'):
                            for b_idx, poly_coeff in self._bin_poly_coeffs:
                                if b_idx == bin_idx:
                                    # Add polynomial evaluated on template wavelength grid
                                    template_poly = np.poly1d(poly_coeff)(self._sps.lam_temp)
                                    optimal_template += template_poly
                                    break
            
            # Get proper fit wave and flux
            if optimal_template is not None and hasattr(self, '_sps'):
                # Use SPS template wavelength and optimal template
                fit_wave = self._sps.lam_temp
                fit_flux = optimal_template
            else:
                # Fallback to bestfit if optimal template not available
                logger.info(f"Falling back to bestfit for bin {bin_idx}")
                fit_wave = self._binned_wavelength
                fit_flux = self._bin_bestfit[:, bin_idx] if hasattr(self, '_bin_bestfit') else bin_spectrum
            
            # Get emission line model if available
            em_wave = None
            em_flux = None
            if hasattr(self, '_bin_gas_bestfit') and self._bin_gas_bestfit is not None:
                gas_data = self._bin_gas_bestfit[:, bin_idx]
                if np.any(np.isfinite(gas_data)):
                    em_wave = self._binned_wavelength
                    em_flux = gas_data
            
            # Print diagnostic info
            logger.info(f"Bin {bin_idx} - Stellar Velocity: {stellar_velocity:.2f} km/s")
            if gas_velocity is not None:
                logger.info(f"Bin {bin_idx} - Gas Velocity: {gas_velocity:.2f} km/s")
            logger.info(f"Bin {bin_idx} - Observed wavelength range: {np.min(self._binned_wavelength):.1f}-{np.max(self._binned_wavelength):.1f}")
            if fit_wave is not None:
                logger.info(f"Bin {bin_idx} - Template wavelength range: {np.min(fit_wave):.1f}-{np.max(fit_wave):.1f}")
            
            # Create a LineIndexCalculator
            from spectral_indices import LineIndexCalculator
            
            calculator = LineIndexCalculator(
                wave=self._binned_wavelength,
                flux=bin_spectrum,
                fit_wave=fit_wave,
                fit_flux=fit_flux,
                em_wave=em_wave,
                em_flux_list=em_flux,
                velocity_correction=stellar_velocity,
                gas_velocity_correction=gas_velocity,  # Pass gas velocity separately
                continuum_mode='auto',
                show_warnings=False
            )
            
            # Create plot
            mode = self._bin_type if hasattr(self, '_bin_type') else "BIN"
            
            # Create save path if requested
            save_path = None
            if save_dir:
                from pathlib import Path
                save_dir = Path(save_dir)
                save_dir.mkdir(exist_ok=True, parents=True)
                save_path = str(save_dir)
            
            # Plot with index calculations
            fig, axes = calculator.plot_all_lines(
                mode=mode,
                number=bin_idx,
                save_path=save_path,
                show_index=True
            )
            
            return fig, axes
        
        except Exception as e:
            logger.error(f"Error plotting bin index calculation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


    def _post_process_emission_results(self):
        """
        Post-process emission line results by replacing NaN values with zeros
        where other pixels have data
        """
        # Check if we have emission line data
        if not hasattr(self, '_emission_flux') or not self._emission_flux:
            return
        
        # For each emission line
        for line_name in self._emission_flux:
            # Get the emission flux for this line
            flux_map = self._emission_flux[line_name]
            
            # Check if we have any valid values
            if np.any(np.isfinite(flux_map)):
                # Replace NaNs with zeros
                nan_mask = ~np.isfinite(flux_map)
                flux_map[nan_mask] = 0.0
                self._emission_flux[line_name] = flux_map
                
                # Also update velocity and dispersion maps
                if hasattr(self, '_emission_vel') and line_name in self._emission_vel:
                    vel_map = self._emission_vel[line_name]
                    vel_map[nan_mask] = 0.0
                    self._emission_vel[line_name] = vel_map
                    
                if hasattr(self, '_emission_sig') and line_name in self._emission_sig:
                    sig_map = self._emission_sig[line_name]
                    sig_map[nan_mask] = 0.0
                    self._emission_sig[line_name] = sig_map
        
        # Also fix gas_bestfit_field if available
        if hasattr(self, '_gas_bestfit_field') and self._gas_bestfit_field is not None:
            # Find pixels with NaN values
            nan_mask = ~np.isfinite(self._gas_bestfit_field)
            if np.any(nan_mask):
                # Replace NaNs with zeros
                self._gas_bestfit_field[nan_mask] = 0.0

    @property
    def redshift(self):
        """Return the galaxy redshift"""
        return self._redshift

    @property

    def raw_data(self):
        """Return the raw data"""
        return {
            'obs_wvl_air_angstrom': self._obs_wvl_air_angstrom,
            'raw_cube_data': self._raw_cube_data,
            'raw_cube_var': self._raw_cube_var,
        }

    @property
    def instrument_info(self):
        """Return instrument information"""
        return {
            'CD1_1': self._fits_hdu_header.get('CD1_1', 0),
            'CD1_2': self._fits_hdu_header.get('CD1_2', 0),
            'CD2_1': self._fits_hdu_header.get('CD2_1', 0),
            'CD2_2': self._fits_hdu_header.get('CD2_2', 0),
            'CRVAL1': self._fits_hdu_header.get('CRVAL1', 0),
            'CRVAL2': self._fits_hdu_header.get('CRVAL2', 0),
            'pxl_size_x': self._pxl_size_x,
            'pxl_size_y': self._pxl_size_y
        }

    @property
    def fits_hdu_header(self):
        """Return the FITS header"""
        return self._fits_hdu_header

    @property
    def fit_spectra_result(self):
        """Get stellar fitting results"""
        return {
            'velocity_field': self._velocity_field,
            'dispersion_field': self._dispersion_field,
            'bestfit_field': self._bestfit_field,
            'optimal_tmpls': self._optimal_tmpls,
            'template_weights': self._template_weights,
            'poly_coeffs': self._poly_coeffs
        }
        
    @property
    def fit_emission_result(self):
        """Get emission line fitting results"""
        # Check if emission line fitting has been performed
        if self._gas_bestfit_field is None:
            warnings.warn("Emission line fitting has not been performed")
            return None
            
        return {
            'emission_flux': self._emission_flux,
            'emission_vel': self._emission_vel,
            'emission_sig': self._emission_sig,
            'gas_bestfit_field': self._gas_bestfit_field,
            'emission_wavelength': self._emission_wavelength
        }
        
    @property
    def spectral_indices_result(self):
        """Get spectral indices results"""
        # Check if spectral indices have been calculated
        if not hasattr(self, '_spectral_indices') or not self._spectral_indices:
            warnings.warn("Spectral indices have not been calculated")
            return None
            
        return self._spectral_indices

    def get_grid_stat(
            self, x_idx: Union[int, list[int]], y_idx: Union[int, list[int]]
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get the velocity, dispersion, and bestfit for the given spaxel indices.
        
        Parameters
        ----------
        x_idx : int or list of int
            x index of the spaxel(s)
        y_idx : int or list of int
            y index of the spaxel(s)
            
        Returns
        -------
        dict
            Dictionary of velocity, dispersion, and bestfit for the given spaxel indices
        """
        if isinstance(x_idx, int):
            x_idx = [x_idx]
        if isinstance(y_idx, int):
            y_idx = [y_idx]

        results = {}
        for x in x_idx:
            for y in y_idx:
                if 0 <= y < self._n_y and 0 <= x < self._n_x:
                    results[(x, y)] = (
                        self._velocity_field[y, x],
                        self._dispersion_field[y, x],
                        self._bestfit_field[:, y, x]
                    )
        return results
        
    def save_results(self, output_dir: str, prefix: Optional[str] = None):
        """
        Save fitting results to FITS files
        
        Parameters
        ----------
        output_dir : str
            Output directory
        prefix : str, optional
            Filename prefix, defaults to the input filename
        """
        try:
            from astropy.io import fits
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Get filename
            if prefix is None:
                prefix = os.path.splitext(os.path.basename(self._filename))[0]
            
            # Create header
            hdr = fits.Header()
            for key, value in self.instrument_info.items():
                hdr[key] = value
            hdr['REDSHIFT'] = self._redshift
            
            # Save velocity field
            hdu = fits.PrimaryHDU(self._velocity_field, header=hdr)
            hdu.writeto(os.path.join(output_dir, f"{prefix}_velocity.fits"), overwrite=True)
            
            # Save dispersion field
            hdu = fits.PrimaryHDU(self._dispersion_field, header=hdr)
            hdu.writeto(os.path.join(output_dir, f"{prefix}_dispersion.fits"), overwrite=True)
            
            # Save best fit result
            hdu = fits.PrimaryHDU(self._bestfit_field, header=hdr)
            hdu.writeto(os.path.join(output_dir, f"{prefix}_bestfit.fits"), overwrite=True)
            
            # Save emission line results if available
            if self._gas_bestfit_field is not None:
                # Save gas fitting result
                hdu = fits.PrimaryHDU(self._gas_bestfit_field, header=hdr)
                hdu.writeto(os.path.join(output_dir, f"{prefix}_gas_bestfit.fits"), overwrite=True)
                
                # Save emission line flux
                for name, flux in self._emission_flux.items():
                    hdu = fits.PrimaryHDU(flux, header=hdr)
                    hdu.writeto(os.path.join(output_dir, f"{prefix}_{name}_flux.fits"), overwrite=True)
            
            # Save spectral indices if available
            if hasattr(self, '_spectral_indices') and self._spectral_indices:
                for name, index_map in self._spectral_indices.items():
                    hdu = fits.PrimaryHDU(index_map, header=hdr)
                    hdu.writeto(os.path.join(output_dir, f"{prefix}_{name}_index.fits"), overwrite=True)
        
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(filename={self._filename!r}, '
                f'redshift={self._redshift!r}, '
                f'wvl_air_angstrom_range={self._wvl_air_angstrom_range!r})')