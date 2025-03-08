"""
Pixel-to-Pixel (P2P) IFU Spectral Analysis Pipeline

This module performs pixel-by-pixel spectral fitting of IFU data using pPXF,
calculates emission line properties, and computes spectral indices.

Features:
- Multi-threaded pixel fitting
- Centralized configuration
- Efficient spectral index calculation with template substitution
- Robust error handling and recovery
- Customizable visualization
- Single pixel testing capability

Version 3.8.3    2025Mar08    Fixed spectral index calculation and improved error handling
"""

### ------------------------------------------------- ###
# Package Imports
### ------------------------------------------------- ###

import os
import sys
import time
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from scipy import interpolate, integrate
from scipy.optimize import curve_fit
from tqdm import tqdm

from astropy.io import fits
from astropy.table import Table, vstack
import astropy.units as units
import astropy.coordinates as coord

import ppxf as ppxf_package
from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from plotbin.display_bins import display_bins

# Suppress warnings during fitting
warnings.filterwarnings("ignore", category=RuntimeWarning)


### ------------------------------------------------- ###
# Helper Functions
### ------------------------------------------------- ###

def to_scalar(value):
    """
    Convert any value to a scalar for safe formatting.
    
    Parameters
    ----------
    value : any
        Input value that needs to be converted to scalar
        
    Returns
    -------
    float
        Scalar version of the input value
    """
    if value is None:
        return 0.0
    elif isinstance(value, (np.ndarray, list, tuple)):
        # If it's an array type, get the first element
        if len(value) > 0:
            return to_scalar(value[0])  # Recursively process
        return 0.0
    elif hasattr(value, 'item'):
        # If it's a numpy scalar, use item() to convert
        return value.item()
    else:
        # Make sure the return value is a float
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


### ------------------------------------------------- ###
# Configuration System
### ------------------------------------------------- ###

class P2PConfig:
    """Configuration class for P2P fitting parameters."""
    
    def __init__(self):
        # Galaxy parameters
        self.galaxy_name = 'VCC_1588'
        self.redshift = 0.0042
        self.vel_s = 0  # Initial guess for velocity
        self.vel_dis_s = 40  # Initial guess for velocity dispersion
        
        # File paths
        self.spectrum_filename = 'VCC1588_stack.fits'
        self.base_dir = Path('E:/ProGram/Dr.Zheng/2024NAOC-IUS/Wkp/2024-NAOC-IUSpectrum')
        self.data_dir = self.base_dir / 'Ori_Data'
        self.output_dir = self.base_dir / 'FitData' / f'Fit_DS_{datetime.now().strftime("%y%b%d")}_{self.galaxy_name}'
        self.plot_dir = self.base_dir / 'FitPlot' / f'Fit_{datetime.now().strftime("%y%b%d")}_{self.galaxy_name}'
        
        # Spectral range and indices
        self.lam_range_temp = [4800, 5250]  # Wavelength range in Angstroms
        self.line_indices = ['Hbeta', 'Fe5015', 'Mgb']
        
        # Good wavelength range (from FITS header or default)
        self.good_wavelength_range = [4851.01, 5230.06]  # Default values
        
        # pPXF fitting parameters
        self.degree = 3  # Polynomial degree for additive component
        self.mdegree = -1  # Polynomial degree for multiplicative component (disabled)
        self.moments = [4, 2]  # Moments to fit for stellar and gas components
        self.gas_names = ['Hbeta', '[OIII]5007']  # Gas lines to fit
        self.ngas_comp = 1  # Number of gas components
        self.fwhm_gas = 1.0  # FWHM for emission line templates (Angstroms)
        self.mask_width = 1000  # Width parameter for determine_mask function
        
        # Computational settings
        self.n_threads = os.cpu_count() // 2  # Default to half available cores
        self.max_memory_gb = 4  # Maximum memory usage in GB
        
        # Visualization settings
        self.make_plots = True
        self.plot_every_n = 500  # Only plot every n pixels
        self.save_plots = True
        self.dpi = 150
        
        # Flags for what to compute
        self.compute_emission_lines = True
        self.compute_spectral_indices = True
        self.compute_stellar_pops = False  # Not implemented yet
        
        # Error handling and recovery
        self.fallback_to_simple_fit = True  # Use simpler fit if primary fit fails
        self.retry_with_degree_zero = True  # Retry with degree=0 if fit fails
        self.skip_bad_pixels = True  # Skip pixels with insufficient data
        self.safe_mode = False  # Extra-safe settings for difficult data
        self.use_template_for_indices = True  # Use template for indices outside good wavelength range
        
        # Constants
        self.c = 299792.458  # Speed of light in km/s
    
    def create_directories(self):
        """Create all required directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        if self.make_plots:
            os.makedirs(self.plot_dir, exist_ok=True)
            os.makedirs(self.plot_dir / 'P2P_res', exist_ok=True)
    
    def get_data_path(self):
        """Get the full path to the data file."""
        return self.data_dir / self.spectrum_filename
    
    def save(self, filename=None):
        """Save configuration to file."""
        if filename is None:
            filename = self.output_dir / f"{self.galaxy_name}_config.json"
        
        # Convert Path objects to strings for JSON serialization
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in self.__dict__.items()}
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        return filename
    
    @classmethod
    def load(cls, filename):
        """Load configuration from file."""
        config = cls()
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert string paths back to Path objects
        for k, v in data.items():
            if k.endswith('_dir') or k.endswith('_path') or k == 'spectrum_filename':
                setattr(config, k, Path(v))
            else:
                setattr(config, k, v)
        
        return config
    
    def update_from_fits_header(self, header):
        """Update configuration parameters from FITS header."""
        if 'WAVGOOD0' in header and 'WAVGOOD1' in header:
            try:
                good_min = float(header['WAVGOOD0'])
                good_max = float(header['WAVGOOD1'])
                self.good_wavelength_range = [good_min, good_max]
                logging.info(f"Updated good wavelength range to {self.good_wavelength_range} from FITS header")
            except Exception as e:
                logging.warning(f"Failed to read good wavelength range from FITS header: {str(e)}")
                logging.warning("Using default good wavelength range")


### ------------------------------------------------- ###
# Debug Utilities
### ------------------------------------------------- ###

def check_array_compatibility(arrays_dict, step_name, pixel_coords=None):
    """
    Check compatibility of multiple arrays and output detailed debug info.
    
    Parameters
    ----------
    arrays_dict : dict
        Dictionary with array names as keys and arrays as values
    step_name : str
        Name of the current step being executed
    pixel_coords : tuple, optional
        Current pixel coordinates (i,j)
    
    Returns
    -------
    bool
        Whether all arrays are compatible
    """
    pixel_info = f" for pixel {pixel_coords}" if pixel_coords else ""
    logging.debug(f"===== ARRAY COMPATIBILITY CHECK AT {step_name}{pixel_info} =====")
    
    # Collect shape information
    shapes = {name: arr.shape for name, arr in arrays_dict.items()}
    lengths = {name: len(arr) for name, arr in arrays_dict.items()}
    
    # Output shape information
    for name, shape in shapes.items():
        logging.debug(f"  - {name}: shape={shape}, length={lengths[name]}")
    
    # Check if lengths are consistent
    unique_lengths = set(lengths.values())
    compatible = len(unique_lengths) == 1
    
    if not compatible:
        logging.debug(f"INCOMPATIBLE ARRAYS DETECTED in {step_name}{pixel_info}")
        logging.debug(f"  - Unique lengths: {unique_lengths}")
        # Output samples of array values
        for name, arr in arrays_dict.items():
            sample = arr[:5] if len(arr) > 5 else arr
            logging.debug(f"  - {name} sample values: {sample}")
    else:
        logging.debug(f"Arrays are compatible in {step_name}{pixel_info}")
    
    return compatible


### ------------------------------------------------- ###
# Spectrum Index Calculation
### ------------------------------------------------- ###

def make_bins(wavs):
    """
    Given a series of wavelength points, find the edges and widths
    of corresponding wavelength bins.
    
    Parameters
    ----------
    wavs : numpy.ndarray
        Array of wavelength points
        
    Returns
    -------
    edges : numpy.ndarray
        Array of bin edges
    widths : numpy.ndarray
        Array of bin widths
    """
    edges = np.zeros(wavs.shape[0] + 1)
    widths = np.zeros(wavs.shape[0])
    
    # Calculate the first edge
    edges[0] = wavs[0] - (wavs[1] - wavs[0]) / 2
    
    # Calculate the last width
    widths[-1] = (wavs[-1] - wavs[-2])
    
    # Calculate the last edge
    edges[-1] = wavs[-1] + (wavs[-1] - wavs[-2]) / 2
    
    # Calculate intermediate edges
    edges[1:-1] = (wavs[1:] + wavs[:-1]) / 2
    
    # Calculate widths except the last one
    widths[:-1] = edges[1:-1] - edges[:-2]
    
    return edges, widths


def spectres(new_wavs, spec_wavs, spec_fluxes, spec_errs=None, fill=None):
    """
    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.
    
    Parameters
    ----------
    new_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.
    
    spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.
    
    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        spec_wavs, last dimension must correspond to the shape of
        spec_wavs. Extra dimensions before this may be used to include
        multiple spectra.
    
    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.
    
    fill : float (optional)
        Where new_wavs extends outside the wavelength range in spec_wavs
        this value will be used as a filler in new_fluxes and new_errs.
    
    Returns
    -------
    new_fluxes : numpy.ndarray
        Array of resampled flux values
        
    new_errs : numpy.ndarray (optional)
        Array of resampled error values, only returned if spec_errs was provided
    """
    # Default fill value is zero
    if fill is None:
        fill = 0.0
    
    # Rename the input variables for clarity within the function
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

        # If new bin is fully inside an old bin start and stop are equal
        if stop == start:
            new_fluxes[..., j] = old_fluxes[..., start]
            if old_errs is not None:
                new_errs[..., j] = old_errs[..., start]

        # Otherwise multiply the first and last old bin widths by P_ij
        else:
            start_factor = ((old_edges[start+1] - new_edges[j])
                          / (old_edges[start+1] - old_edges[start]))

            end_factor = ((new_edges[j+1] - old_edges[stop])
                        / (old_edges[stop+1] - old_edges[stop]))

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

            # Put back the old bin widths to their initial values
            old_widths[start] /= start_factor
            old_widths[stop] /= end_factor

    # If errors were supplied return both new_fluxes and new_errs
    if old_errs is not None:
        return new_fluxes, new_errs

    # Otherwise just return the new_fluxes spectrum array
    else:
        return new_fluxes


class LineIndexCalculator:
    """
    Class for calculating spectral line indices from fitted spectra.
    
    Supports calculation of common spectral indices like Hbeta, Mgb, Fe5015.
    Handles velocity correction and emission line subtraction.
    Uses template data for continuum estimation while preserving original line data.
    """
    
    def __init__(self, wave, flux, fit_wave, fit_flux, em_wave=None, em_flux_list=None, 
                 velocity_correction=0, error=None, good_wavelength_range=None):
        """
        Initialize the line index calculator.
        
        Parameters
        ----------
        wave : array-like
            Original spectrum wavelength array
        flux : array-like
            Original spectrum flux array
        fit_wave : array-like
            Fitted template wavelength array for continuum calculation
        fit_flux : array-like
            Fitted template flux array for continuum calculation
        em_wave : array-like, optional
            Emission line wavelength array
        em_flux_list : array-like, optional
            Combined emission line spectrum
        velocity_correction : float, optional
            Velocity correction in km/s, default is 0
        error : array-like, optional
            Error array for the spectrum
        good_wavelength_range : list, optional
            Good wavelength range [min, max] for diagnostic purposes
        """
        self.c = 299792.458  # Speed of light in km/s
        self.velocity = velocity_correction
        
        # Apply velocity correction
        self.wave = self._apply_velocity_correction(wave)
        self.flux = flux.copy()  # Create a copy to avoid modifying original data
        
        # Store template data
        self.fit_wave = fit_wave
        self.fit_flux = fit_flux
        self.error = error if error is not None else np.ones_like(flux)
        self.good_wavelength_range = good_wavelength_range
        
        # Handle emission lines
        if em_wave is not None and em_flux_list is not None:
            self.em_wave = self._apply_velocity_correction(em_wave)
            self.em_flux_list = em_flux_list
            self._subtract_emission_lines()
    
    def _subtract_emission_lines(self):
        """
        Subtract emission lines from the original spectrum.
        The input em_flux_list is already the combined result.
        """
        # Resample emission line spectrum to original spectrum wavelength grid
        try:
            em_flux_resampled = spectres(self.wave, self.em_wave, self.em_flux_list)
            
            # Subtract emission lines from original spectrum
            self.flux -= em_flux_resampled
        except Exception as e:
            logging.warning(f"Error subtracting emission lines: {str(e)}")
            # Continue without subtracting emission lines
    
    def _apply_velocity_correction(self, wave):
        """
        Apply velocity correction to wavelength.
        
        Parameters
        ----------
        wave : array-like
            Original wavelength array
            
        Returns
        -------
        array-like
            Corrected wavelength array
        """
        return wave / (1 + (self.velocity / self.c))
    
    @staticmethod
    def define_line_windows(line_name):
        """
        Define absorption line and continuum windows.
        
        Parameters
        ----------
        line_name : str
            Absorption line name
            
        Returns
        -------
        dict
            Dictionary containing blue, line, and red window ranges
        """
        # Define standard Lick index windows
        windows = {
            'Hbeta': {
                'blue': (4827.875, 4847.875),
                'line': (4847.875, 4876.625),
                'red': (4876.625, 4891.625)
            },
            'Mgb': {
                'blue': (5142.625, 5161.375),
                'line': (5160.125, 5192.625),
                'red': (5191.375, 5206.375)
            },
            'Fe5015': {
                'blue': (4946.500, 4977.750),
                'line': (4977.750, 5054.000),
                'red': (5054.000, 5065.250)
            },
            'Fe5270': {
                'blue': (5233.150, 5248.150),
                'line': (5245.650, 5285.650),
                'red': (5285.650, 5318.150)
            }
        }
        return windows.get(line_name)

    def calculate_pseudo_continuum(self, wave_range, flux_range):
        """
        Calculate pseudo-continuum.
        
        Parameters
        ----------
        wave_range : array-like
            Wavelength range
        flux_range : array-like
            Corresponding flux values
            
        Returns
        -------
        float
            Pseudo-continuum value
        """
        return np.median(flux_range)

    def calculate_index(self, line_name, return_error=False):
        """
        Calculate absorption line index using template for continuum estimation.
        
        Parameters
        ----------
        line_name : str
            Absorption line name
        return_error : bool
            Whether to return error
            
        Returns
        -------
        float
            Absorption line index value
        float
            Error value (if return_error=True)
        """
        logging.debug(f"===== CALCULATING INDEX: {line_name} =====")
        
        # Get window definitions
        windows = self.define_line_windows(line_name)
        if windows is None:
            raise ValueError(f"Unknown absorption line: {line_name}")

        # Extract regions for calculations
        def get_region_data(region_name, from_template=False):
            """Get region data with possible template substitution"""
            region_bounds = windows[region_name]
            wave_in_region = (self.wave >= region_bounds[0]) & (self.wave <= region_bounds[1])
            
            if not np.any(wave_in_region):
                return np.array([]), np.array([]), np.array([])
            
            region_wave = self.wave[wave_in_region]
            
            # Check if region is outside good wavelength range
            if self.good_wavelength_range is not None and not from_template:
                outside_good = ((region_bounds[0] < self.good_wavelength_range[0]) or 
                               (region_bounds[1] > self.good_wavelength_range[1]))
                
                if outside_good:
                    logging.debug(f"  - {region_name} region partially outside good range, using template substitution")
                    
                    # Get template flux for this region (through interpolation)
                    template_flux = np.interp(region_wave, self.fit_wave, self.fit_flux)
                    
                    # Get original flux
                    region_flux = self.flux[wave_in_region]
                    region_err = self.error[wave_in_region]
                    
                    # Create mask for points inside/outside good range
                    in_good_range = ((region_wave >= self.good_wavelength_range[0]) & 
                                    (region_wave <= self.good_wavelength_range[1]))
                    
                    # For points outside good range, use normalized template
                    if np.any(~in_good_range) and np.any(in_good_range):
                        # Calculate normalization factor from points in good range
                        good_points = region_flux[in_good_range]
                        good_template = template_flux[in_good_range]
                        norm_factor = np.median(good_points) / np.median(good_template) if np.median(good_template) != 0 else 1.0
                        
                        # Create synthetic flux
                        synthetic_flux = region_flux.copy()
                        synthetic_flux[~in_good_range] = template_flux[~in_good_range] * norm_factor
                        
                        # Use synthetic flux
                        return region_wave, synthetic_flux, region_err
            
            # Default case - just use original data
            return region_wave, self.flux[wave_in_region], self.error[wave_in_region]
            
        # Get fit template regions (always from template)
        def get_fit_region(region):
            mask = (self.fit_wave >= windows[region][0]) & (self.fit_wave <= windows[region][1])
            return self.fit_wave[mask], self.fit_flux[mask]
        
        # Get data for each region
        blue_wave, blue_flux, blue_err = get_region_data('blue')
        line_wave, line_flux, line_err = get_region_data('line')
        red_wave, red_flux, red_err = get_region_data('red')
        
        # Get template data for continuum estimation
        blue_wave_fit, blue_flux_fit = get_fit_region('blue')
        red_wave_fit, red_flux_fit = get_fit_region('red')
        
        # Check if we have enough data points
        logging.debug(f"STEP: Checking if we have enough data points")
        logging.debug(f"  - Blue continuum points: {len(blue_flux)}")
        logging.debug(f"  - Line region points: {len(line_flux)}")
        logging.debug(f"  - Red continuum points: {len(red_flux)}")
        
        if len(blue_flux) < 3 or len(line_flux) < 3 or len(red_flux) < 3:
            logging.warning(f"Not enough points for index calculation: blue={len(blue_flux)}, line={len(line_flux)}, red={len(red_flux)}")
            return np.nan if not return_error else (np.nan, np.nan)

        # Calculate continuum using template data
        blue_cont = self.calculate_pseudo_continuum(blue_wave_fit, blue_flux_fit)
        red_cont = self.calculate_pseudo_continuum(red_wave_fit, red_flux_fit)
        
        wave_cont = np.array([np.mean(blue_wave_fit), np.mean(red_wave_fit)])
        flux_cont = np.array([blue_cont, red_cont])
        
        logging.debug(f"STEP: Calculating continuum")
        logging.debug(f"  - Blue continuum: {blue_cont:.4f} at λ={np.mean(blue_wave_fit):.2f}")
        logging.debug(f"  - Red continuum: {red_cont:.4f} at λ={np.mean(red_wave_fit):.2f}")
        
        # Linear interpolation for continuum
        f_interp = interpolate.interp1d(wave_cont, flux_cont)
        cont_at_line = f_interp(line_wave)

        # Calculate integral (use trapezoidal rule for integration)
        logging.debug(f"STEP: Calculating index using trapezoidal integration")
        index = np.trapz((1.0 - line_flux/cont_at_line), line_wave)
        logging.debug(f"  - Index value: {index:.6f}")
        
        # Store data for plotting
        self._last_calc = {
            'line_name': line_name,
            'windows': windows,
            'blue_wave_fit': blue_wave_fit,
            'blue_flux_fit': blue_flux_fit,
            'red_wave_fit': red_wave_fit,
            'red_flux_fit': red_flux_fit,
            'blue_wave': blue_wave,
            'blue_flux': blue_flux,
            'line_wave': line_wave,
            'line_flux': line_flux,
            'red_wave': red_wave,
            'red_flux': red_flux,
            'wave_cont': wave_cont,
            'flux_cont': flux_cont,
            'cont_at_line': cont_at_line,
            'index': index
        }
        
        if return_error:
            # Calculate error
            error = np.sqrt(np.trapz((line_err/cont_at_line)**2, line_wave))
            logging.debug(f"  - Index error: {error:.6f}")
            return index, error
        
        return index

    def plot_line_fit(self, line_name, output_path=None):
        """
        Plot absorption line fitting result matching the original implementation style.
        
        Parameters
        ----------
        line_name : str
            Absorption line name
        output_path : str, optional
            Path to save the plot
        """
        # Calculate index if not already done to populate the data
        if not hasattr(self, '_last_calc') or self._last_calc.get('line_name') != line_name:
            self.calculate_index(line_name)
            
        if not hasattr(self, '_last_calc'):
            logging.warning(f"No data available for plotting {line_name}")
            return
            
        data = self._last_calc
        windows = data['windows']
        
        # Set x-axis range: extend window range on both sides by 20Å
        x_min = windows['blue'][0] - 20
        x_max = windows['red'][1] + 20
        
        # Create plot with three panels using figure instead of gridspec for better tight_layout compatibility
        fig = plt.figure(figsize=(12, 10))
        
        # Main panel: Original spectrum and fit
        ax1 = fig.add_subplot(3, 1, 1)
        
        # Second panel: Zoomed view of line region
        ax2 = fig.add_subplot(3, 1, 2)
        
        # Third panel: Absorption profile
        ax3 = fig.add_subplot(3, 1, 3)
        
        # First panel: Full spectrum
        if hasattr(self, 'em_flux_list'):
            # Original data with emission lines
            if hasattr(self, 'em_wave') and hasattr(self, 'em_flux_list'):
                try:
                    ax1.plot(self.wave, self.flux + self.em_flux_list, 'k-', 
                           label='Original Spectrum', alpha=0.7)
                    ax1.plot(self.em_wave, self.em_flux_list, 'r-', 
                           label='Emission Lines', alpha=0.7)
                except Exception as e:
                    ax1.plot(self.wave, self.flux, 'k-', label='Original Spectrum', alpha=0.7)
                    logging.warning(f"Could not plot emission lines: {str(e)}")
        else:
            ax1.plot(self.wave, self.flux, 'k-', label='Original Spectrum', alpha=0.7)
            
        ax1.plot(self.fit_wave, self.fit_flux, 'b-', label='Template Fit', alpha=0.7)
        
        # Mark regions
        colors = {'blue': 'blue', 'line': 'green', 'red': 'red'}
        for region, (start, end) in windows.items():
            ax1.axvspan(start, end, alpha=0.2, color=colors[region], label=f'{region.capitalize()} Region')
            ax2.axvspan(start, end, alpha=0.2, color=colors[region])
            ax3.axvspan(start, end, alpha=0.2, color=colors[region])

        ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        # Mark good wavelength range if available
        if self.good_wavelength_range is not None:
            ax1.axvspan(self.good_wavelength_range[0], self.good_wavelength_range[1], 
                      color='lightgreen', alpha=0.15, label='Good λ Range')
            ax2.axvspan(self.good_wavelength_range[0], self.good_wavelength_range[1], 
                      color='lightgreen', alpha=0.15)
            ax3.axvspan(self.good_wavelength_range[0], self.good_wavelength_range[1], 
                      color='lightgreen', alpha=0.15)
        
        # Add continuum points and line
        if 'wave_cont' in data and 'flux_cont' in data:
            ax1.plot(data['wave_cont'], data['flux_cont'], 'ro', ms=8, label='Continuum Points')
            
            # Plot continuum line over index region
            if 'line_wave' in data and 'cont_at_line' in data:
                ax1.plot(data['line_wave'], data['cont_at_line'], 'r--', lw=2, label='Continuum Level')
                
        # Set y-axis limits using template range
        try:
            temp_mask = (self.fit_wave >= x_min) & (self.fit_wave <= x_max)
            y_max = np.max(self.fit_flux[temp_mask]) * 1.1
            y_min = np.min(self.fit_flux[temp_mask]) * 0.9
            ax1.set_ylim(y_min, y_max)
        except Exception as e:
            logging.warning(f"Error setting y-limits: {str(e)}")
        
        ax1.set_xlim(x_min, x_max)
        ax1.set_title(f'Spectral Index Measurement: {line_name}')
        ax1.set_ylabel('Flux')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Second panel: Zoomed view of index region
        # Plot data in each region with different markers
        if 'blue_wave' in data:
            ax2.plot(data['blue_wave'], data['blue_flux'], 'o', color='blue', ms=4, alpha=0.8, label='Blue Continuum')
        if 'line_wave' in data:
            ax2.plot(data['line_wave'], data['line_flux'], 'o', color='green', ms=4, alpha=0.8, label='Line Region')
        if 'red_wave' in data:
            ax2.plot(data['red_wave'], data['red_flux'], 'o', color='red', ms=4, alpha=0.8, label='Red Continuum')
        
        # Plot continuum line
        if 'line_wave' in data and 'cont_at_line' in data:
            ax2.plot(data['line_wave'], data['cont_at_line'], 'r--', lw=2)
        
        # Adjust y-axis limits for second panel
        padding = 10  # Å of padding around regions
        zoom_min = windows['blue'][0] - padding
        zoom_max = windows['red'][1] + padding
        ax2.set_xlim(zoom_min, zoom_max)
        
        ax2.set_ylabel('Flux')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        # Third panel: Absorption profile
        if 'line_wave' in data and 'line_flux' in data and 'cont_at_line' in data:
            # Calculate absorption profile
            abs_profile = 1.0 - data['line_flux'] / data['cont_at_line']
            
            # Plot absorption profile
            ax3.plot(data['line_wave'], abs_profile, 'g-', lw=2, label='Absorption Profile')
            
            # Fill area under the curve (represents the index)
            ax3.fill_between(data['line_wave'], 0, abs_profile, color='green', alpha=0.3)
            
            # Add index value as text
            index_val = data.get('index', np.nan)
            ax3.text(0.05, 0.85, f"{line_name} Index = {index_val:.4f} Å", 
                    transform=ax3.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            
            # Add horizontal line at y=0
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Set limits for absorption profile
            ax3.set_xlim(windows['line'][0] - 5, windows['line'][1] + 5)
            ax3.set_ylim(-0.05, max(abs_profile) * 1.1 + 0.05)
        
        ax3.set_xlabel('Wavelength (Å)')
        ax3.set_ylabel('1 - Flux/Continuum')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax3.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax3.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


### ------------------------------------------------- ###
# Data Handling
### ------------------------------------------------- ###

class IFUDataCube:
    """
    Class for handling IFU data cubes with associated methods for preprocessing.
    
    Provides functionality to:
    - Read and parse FITS files
    - De-redshift spectra
    - Rebin spectra
    - Compute coordinates
    """
    
    def __init__(self, filename, lam_range, redshift, config):
        """
        Initialize and load IFU data cube.
        
        Parameters
        ----------
        filename : str or Path
            Path to the FITS file
        lam_range : tuple
            Wavelength range to use [min, max]
        redshift : float
            Redshift of the galaxy
        config : P2PConfig
            Configuration object
        """
        self.config = config
        self.read_fits_file(filename)
        
        # Only use the specified rest-frame wavelength range
        wave = self.wave / (1 + redshift)  # De-redshift the spectrum
        w = (wave > lam_range[0]) & (wave < lam_range[1])
        wave = wave[w]
        cube = self.cube[w, ...]
        cubevar = self.cubevar[w, ...]
        
        # Calculate signal and noise
        signal = np.nanmedian(cube, 0)
        noise = np.sqrt(np.nanmedian(cubevar, 0))
        
        # Create coordinates centered on the brightest spaxel
        jm = np.argmax(signal)
        row, col = np.indices(cube.shape[-2:])
        row = row.ravel()
        col = col.ravel()
        x = (col - col[jm]) * self.pixsize_x
        y = (row - row[jm]) * self.pixsize_y
        
        # Transform cube into 2D array of spectra
        npix = cube.shape[0]
        spectra = cube.reshape(npix, -1)  # Create array of spectra [npix, nx*ny]
        variance = cubevar.reshape(npix, -1)  # Create array of variance [npix, nx*ny]
        
        # Calculate velocity scale and log-rebin
        velscale = np.min(config.c * np.diff(np.log(wave)))  # Preserve smallest velocity step
        lam_range_temp = [np.min(wave), np.max(wave)]
        spectra, ln_lam_gal, velscale = util.log_rebin(lam_range_temp, spectra, velscale=velscale)
        
        # Store all the processed data
        self.spectra = spectra
        self.variance = variance
        self.x = x
        self.y = y
        self.signal = signal.ravel()
        self.noise = noise.ravel()
        self.col = col + 1  # Start counting from 1
        self.row = row + 1
        self.velscale = velscale
        self.ln_lam_gal = ln_lam_gal
        self.lam_gal = np.exp(ln_lam_gal)
        self.fwhm_gal = self.fwhm_gal / (1 + redshift)
        
        # Initialize fields for results
        self.velfield = np.full(self.cube.shape[1:3], np.nan)
        self.sigfield = np.full(self.cube.shape[1:3], np.nan)
        self.bestfit_field = np.full((spectra.shape[0], self.cube.shape[1], self.cube.shape[2]), np.nan)
        self.optimal_templates = np.full((spectra.shape[0], self.cube.shape[1], self.cube.shape[2]), np.nan)
        
        # Index maps
        self.index_maps = {}
        for index_name in config.line_indices:
            self.index_maps[index_name] = np.full(self.cube.shape[1:3], np.nan)
        
        # Emission line maps
        self.el_flux_maps = {}
        self.el_snr_maps = {}
        for line_name in config.gas_names:
            self.el_flux_maps[line_name] = np.full(self.cube.shape[1:3], np.nan)
            self.el_snr_maps[line_name] = np.full(self.cube.shape[1:3], np.nan)
        
        logging.debug(f"IFUDataCube initialized - Shape: {self.cube.shape}, Wavelength range: [{self.lam_gal[0]:.2f}, {self.lam_gal[-1]:.2f}]")
        
    def read_fits_file(self, filename):
        """
        Read FITS file containing IFU data cube.
        
        Parameters
        ----------
        filename : str or Path
            Path to the FITS file
        """
        # Define data ranges to use
        cut_low = 1
        cut_high = 1
        
        try:
            hdu = fits.open(filename)
            head = hdu[0].header
            
            # Scale data if needed (convert to proper flux units)
            cube = hdu[0].data[cut_low:-cut_high, :, :] * (10**18)
            
            # Create empty error cube if not provided
            cubevar = np.ones_like(cube)  # Default to uniform variance
            
            # Calculate wavelength array
            wave = head['CRVAL3'] + head['CD3_3'] * np.arange(cube.shape[0]) + head['CD3_3'] * cut_low
            
            # Store the data
            self.cube = cube
            self.cubevar = cubevar
            self.wave = wave
            
            # Get spectral resolution (FWHM)
            self.fwhm_gal = 1.0  # Default value if not provided
            
            # Get pixel size
            self.pixsize_x = abs(np.sqrt((head['CD1_1'])**2 + (head['CD2_1'])**2)) * 3600
            self.pixsize_y = abs(np.sqrt((head['CD1_2'])**2 + (head['CD2_2'])**2)) * 3600
            
            # Store WCS information
            self.CD1_1 = head.get('CD1_1', 0)
            self.CD1_2 = head.get('CD1_2', 0)
            self.CD2_1 = head.get('CD2_1', 0)
            self.CD2_2 = head.get('CD2_2', 0)
            self.CRVAL1 = head.get('CRVAL1', 0)
            self.CRVAL2 = head.get('CRVAL2', 0)
            
            # Update good wavelength range from FITS header if available
            if self.config is not None:
                if 'WAVGOOD0' in head and 'WAVGOOD1' in head:
                    try:
                        good_min = float(head['WAVGOOD0'])
                        good_max = float(head['WAVGOOD1'])
                        self.config.good_wavelength_range = [good_min, good_max]
                        logging.info(f"Updated good wavelength range to {self.config.good_wavelength_range} from FITS header")
                    except Exception as e:
                        logging.warning(f"Failed to read good wavelength range from FITS header: {str(e)}")
                        logging.warning("Using default good wavelength range")
            
            # Close the FITS file
            hdu.close()
            
            logging.debug(f"FITS file read successfully: {filename}")
            logging.debug(f"Cube shape: {cube.shape}, Wavelength range: [{wave[0]:.2f}, {wave[-1]:.2f}]")
            
        except Exception as e:
            logging.error(f"Error reading FITS file {filename}: {str(e)}")
            raise


### ------------------------------------------------- ###
# pPXF Fitting Functions
### ------------------------------------------------- ###

def prepare_templates(config, velscale):
    """
    Prepare stellar and gas templates for pPXF fitting.
    
    Parameters
    ----------
    config : P2PConfig
        Configuration object
    velscale : float
        Velocity scale of the data
        
    Returns
    -------
    tuple
        (sps, gas_templates, gas_names, line_wave)
    """
    logging.debug("===== PREPARING TEMPLATES =====")
    
    # Set up paths
    ppxf_dir = Path(lib.__file__).parent
    sps_name = 'emiles'  # Use EMILES templates
    basename = f"spectra_{sps_name}_9.0.npz"
    filename = ppxf_dir / 'sps_models' / basename
    
    logging.debug(f"STEP: Loading stellar templates from {filename}")
    
    # Load stellar templates - use None to skip broadening
    FWHM_stellar = None  # Skip broadening for stellar templates
    sps = lib.sps_lib(filename, velscale, FWHM_stellar, norm_range=config.lam_range_temp)
    
    # Reshape and normalize templates
    npix, *reg_dim = sps.templates.shape
    sps.templates = sps.templates.reshape(npix, -1)
    sps.templates /= np.median(sps.templates)  # Normalize by scalar
    
    # Log template info for debugging
    logging.debug(f"STEP: Stellar templates prepared")
    logging.debug(f"  - Stellar template shape: {sps.templates.shape}")
    logging.debug(f"  - Template wavelength range: {np.exp(sps.ln_lam_temp[[0, -1]])}")
    
    # Prepare gas templates with a proper FWHM value
    logging.debug(f"STEP: Preparing gas templates with FWHM={config.fwhm_gas}")
    FWHM_gas = config.fwhm_gas  # Use the config value for gas templates
    lam_range_gal = [np.exp(sps.ln_lam_temp[0]), np.exp(sps.ln_lam_temp[-1])]
    
    try:
        # Ensure gas templates have same wavelength grid as stellar templates
        all_gas_templates, all_gas_names, all_line_wave = util.emission_lines(
            sps.ln_lam_temp, lam_range_gal, FWHM_gas)
        
        logging.debug(f"  - Generated gas templates: {len(all_gas_names)} emission lines")
        logging.debug(f"  - Gas templates shape: {all_gas_templates.shape}")
        
        # Normalize gas templates if needed
        if not np.all(np.isfinite(all_gas_templates)):
            logging.warning("Found non-finite values in gas templates, fixing...")
            all_gas_templates = np.nan_to_num(all_gas_templates, nan=0.0, posinf=0.0, neginf=0.0)
        
    except Exception as e:
        logging.error(f"Error generating gas templates: {str(e)}")
        # Create a minimal empty gas template as fallback
        all_gas_templates = np.zeros((sps.templates.shape[0], 1))
        all_gas_names = np.array(['Dummy'])
        all_line_wave = np.array([5000.0])
        logging.warning("Using dummy gas template as fallback")
    
    # Filter to only use requested emission lines if specified
    logging.debug(f"STEP: Filtering gas templates to requested lines: {config.gas_names}")
    if config.gas_names and len(config.gas_names) > 0:
        mask = np.zeros(len(all_gas_names), dtype=bool)
        for name in config.gas_names:
            for i, full_name in enumerate(all_gas_names):
                if name in full_name:
                    mask[i] = True
        
        if np.any(mask):
            gas_templates = all_gas_templates[:, mask]
            gas_names = all_gas_names[mask]
            line_wave = all_line_wave[mask]
            logging.debug(f"  - Using emission lines: {gas_names}")
        else:
            gas_templates = all_gas_templates
            gas_names = all_gas_names
            line_wave = all_line_wave
            logging.warning(f"No matches for requested lines {config.gas_names}. Using all available lines.")
    else:
        gas_templates = all_gas_templates
        gas_names = all_gas_names
        line_wave = all_line_wave
    
    # Handle multiple gas components
    if config.ngas_comp > 1:
        logging.debug(f"STEP: Creating {config.ngas_comp} gas components")
        gas_templates = np.tile(gas_templates, config.ngas_comp)
        gas_names = np.asarray([f"{a}_({p+1})" for p in range(config.ngas_comp) for a in gas_names])
        line_wave = np.tile(line_wave, config.ngas_comp)
        logging.debug(f"  - Final gas templates shape: {gas_templates.shape}")
    
    logging.debug(f"STEP: Template preparation complete")
    logging.debug(f"  - Stellar templates: {sps.templates.shape}")
    logging.debug(f"  - Gas templates: {gas_templates.shape}")
    logging.debug(f"  - Gas lines: {gas_names}")
    
    return sps, gas_templates, gas_names, line_wave


def fit_single_pixel(args):
    """
    Fit a single pixel with pPXF.
    
    Parameters
    ----------
    args : tuple
        (i, j, galaxy_data, sps, gas_templates, gas_names, line_wave, config)
        
    Returns
    -------
    tuple
        (i, j, results_dict or None)
    """
    i, j, galaxy_data, sps, gas_templates, gas_names, line_wave, config = args
    
    # Get index in the flattened array
    k_index = i * galaxy_data.cube.shape[2] + j
    
    logging.debug(f"===== FITTING PIXEL ({i},{j}) =====")
    
    # Skip if this is a bad pixel
    if np.count_nonzero(galaxy_data.spectra[:, k_index]) < 50 and config.skip_bad_pixels:
        logging.debug(f"Skipping pixel ({i},{j}) - insufficient data points")
        return i, j, None
    
    try:
        # Get spectrum data
        spectrum = galaxy_data.spectra[:, k_index]
        noise = np.ones_like(spectrum)  # Use uniform noise
        
        # Calculate wavelength ranges
        gal_wave_range = np.exp(galaxy_data.ln_lam_gal[[0, -1]])
        temp_wave_range = np.exp(sps.ln_lam_temp[[0, -1]])
        
        logging.debug(f"STEP: Wavelength ranges")
        logging.debug(f"  - Galaxy wave range: {gal_wave_range}")
        logging.debug(f"  - Template wave range: {temp_wave_range}")
        
        # Create wavelength mask for fitting
        logging.debug(f"STEP: Creating wavelength mask with width={config.mask_width}")
        
        # First limit to good wavelength range
        if hasattr(config, 'good_wavelength_range'):
            good_ln_lam_min = np.log(config.good_wavelength_range[0])
            good_ln_lam_max = np.log(config.good_wavelength_range[1])
            
            # Create template mask
            mask_template = util.determine_mask(galaxy_data.ln_lam_gal, 
                                            temp_wave_range, 
                                            width=config.mask_width)
            
            # Create good wavelength mask
            mask_good = (galaxy_data.ln_lam_gal >= good_ln_lam_min) & (galaxy_data.ln_lam_gal <= good_ln_lam_max)
            
            # Combine masks
            mask = mask_template & mask_good
            
            logging.debug(f"  - Template mask points: {np.sum(mask_template)}")
            logging.debug(f"  - Good wavelength mask points: {np.sum(mask_good)}")
            logging.debug(f"  - Combined mask points: {np.sum(mask)}")
        else:
            # Just use template mask
            mask = util.determine_mask(galaxy_data.ln_lam_gal, 
                                     temp_wave_range, 
                                     width=config.mask_width)
            logging.debug(f"  - Template mask points: {np.sum(mask)}")
        
        if not np.any(mask):
            logging.warning(f"Empty mask for pixel ({i},{j}). Wavelength ranges may not overlap or good range too restrictive.")
            return i, j, None
        
        # First stellar fit with error handling
        logging.debug(f"STEP: First stellar-only fit")
        try:
            pp_stars = ppxf(sps.templates, spectrum, noise, galaxy_data.velscale, 
                          [config.vel_s, config.vel_dis_s],
                          degree=config.degree,
                          plot=False, mask=mask, lam=galaxy_data.lam_gal, 
                          lam_temp=sps.lam_temp, quiet=True)
            logging.debug(f"  - First fit successful: v={pp_stars.sol[0]:.1f}, σ={pp_stars.sol[1]:.1f}")
        except Exception as e:
            if config.retry_with_degree_zero:
                logging.warning(f"Initial stellar fit failed for pixel ({i},{j}): {str(e)}")
                logging.debug(f"  - Retrying with simplified parameters: degree=0, mdegree=-1")
                # Try with simpler polynomial
                pp_stars = ppxf(sps.templates, spectrum, noise, galaxy_data.velscale, 
                              [config.vel_s, config.vel_dis_s],
                              degree=0, mdegree=-1, # Use simplest polynomial settings
                              plot=False, mask=mask, lam=galaxy_data.lam_gal, 
                              lam_temp=sps.lam_temp, quiet=True)
                logging.debug(f"  - Retry successful: v={pp_stars.sol[0]:.1f}, σ={pp_stars.sol[1]:.1f}")
            else:
                raise  # Re-raise the exception if we're not retrying
        
        # Get best-fit template
        logging.debug(f"STEP: Getting best-fit stellar template")
        logging.debug(f"  - sps.templates shape: {sps.templates.shape}")
        logging.debug(f"  - pp_stars.weights shape: {pp_stars.weights.shape}")
        
        # Ensure we have valid weights before matrix multiplication
        if pp_stars.weights is None or not np.any(np.isfinite(pp_stars.weights)):
            logging.warning(f"Invalid weights in stellar fit for pixel ({i},{j})")
            return i, j, None
            
        best_template = sps.templates @ pp_stars.weights
        
        logging.debug(f"  - Resulting best_template shape: {best_template.shape}")
        
        # Make sure best_template is 2D column vector
        if best_template.ndim == 1:
            logging.debug(f"STEP: Reshaping 1D template to 2D column vector")
            best_template = best_template.reshape(-1, 1)
            logging.debug(f"  - After reshape: best_template shape: {best_template.shape}")
        
        # Combine stellar and gas templates
        logging.debug(f"STEP: Combining stellar and gas templates")
        logging.debug(f"  - Best template shape: {best_template.shape}")
        logging.debug(f"  - Gas templates shape: {gas_templates.shape}")
        
        stars_gas_templates = np.column_stack([best_template, gas_templates])
        logging.debug(f"  - Combined template shape: {stars_gas_templates.shape}")
        
        # Set up component array correctly
        logging.debug(f"STEP: Setting up component array")
        n_templates = stars_gas_templates.shape[1]
        component = np.zeros(n_templates, dtype=int)
        component[1:] = 1  # Mark gas templates as component 1
        logging.debug(f"  - Component array: shape={component.shape}, unique values={np.unique(component)}")
        
        # Mark gas components
        gas_component = np.zeros_like(component, dtype=bool)
        gas_component[1:] = True
        logging.debug(f"  - Gas component mask: sum={np.sum(gas_component)}")
        
        # Set up moments
        logging.debug(f"STEP: Setting up moments array")
        moments = config.moments
        if isinstance(moments, int):
            moments = [moments, 2]  # Default: stellar moments and gas=2
        logging.debug(f"  - Moments: {moments}")
        
        # Set up start values
        vel_stars = to_scalar(pp_stars.sol[0])
        sigma_stars = to_scalar(pp_stars.sol[1]) 
        
        # Ensure sigma is positive
        if sigma_stars < 0:
            logging.warning(f"Negative velocity dispersion detected: {sigma_stars:.1f} km/s. Setting to 10 km/s.")
            sigma_stars = 10.0
            
        logging.debug(f"STEP: Setting up start values based on stellar fit: v={vel_stars:.1f}, σ={sigma_stars:.1f}")
        
        # Create properly sized start array based on moments
        start = []
        
        # Stellar component
        if moments[0] == 2:
            start.append([vel_stars, sigma_stars])
        elif moments[0] == 4:
            start.append([vel_stars, sigma_stars, 0, 0])
        else:
            start_stars = [vel_stars]
            if moments[0] >= 2:
                start_stars.append(sigma_stars)
            start_stars.extend([0] * (moments[0] - len(start_stars)))
            start.append(start_stars)
        
        # Gas component
        if len(moments) > 1:
            if moments[1] == 2:
                start.append([vel_stars, 50])  # Use stellar velocity, 50 km/s dispersion
            elif moments[1] == 1:
                start.append([vel_stars])
            else:
                start_gas = [vel_stars]
                if moments[1] >= 2:
                    start_gas.append(50)
                start_gas.extend([0] * (moments[1] - len(start_gas)))
                start.append(start_gas)
        
        logging.debug(f"  - Start values: {start}")
        
        # Set up bounds
        logging.debug(f"STEP: Setting up parameter bounds")
        bounds = []
        vel_range = [vel_stars - 100, vel_stars + 100]  # Velocity bounds
        
        # Stellar component bounds
        if moments[0] == 2:
            bounds.append([vel_range, [5, 300]])  # Minimum sigma = 5 km/s
        elif moments[0] == 4:
            bounds.append([vel_range, [5, 300], [-0.3, 0.3], [-0.3, 0.3]])
        else:
            bound_stars = [vel_range]
            if moments[0] >= 2:
                bound_stars.append([5, 300])  # Ensure minimum sigma is positive
            bound_stars.extend([[-0.3, 0.3]] * (moments[0] - len(bound_stars)))
            bounds.append(bound_stars)
        
        # Gas component bounds
        if len(moments) > 1:
            if moments[1] == 2:
                bounds.append([vel_range, [5, 100]])  # Minimum sigma = 5 km/s
            elif moments[1] == 1:
                bounds.append([vel_range])
            else:
                bound_gas = [vel_range]
                if moments[1] >= 2:
                    bound_gas.append([5, 100])  # Ensure minimum sigma is positive
                bound_gas.extend([[-0.3, 0.3]] * (moments[1] - len(bound_gas)))
                bounds.append(bound_gas)
        
        logging.debug(f"  - Bounds: {bounds}")
        
        # Extra debug info before fitting
        logging.debug(f"STEP: Final check before combined fit")
        logging.debug(f"  - Templates shape: {stars_gas_templates.shape}")
        logging.debug(f"  - Component: {component.shape}, first few values: {component[:5]}")
        logging.debug(f"  - Moments: {moments}")
        logging.debug(f"  - Start values compatibility: {len(start) == len(moments)}")
        logging.debug(f"  - Bounds compatibility: {len(bounds) == len(moments)}")
        
        # Run complete fit with safety measures
        logging.debug(f"STEP: Running combined stellar+gas fit")
        pp = None  # Initialize pp to ensure it exists even if errors occur
        
        try:
            pp = ppxf(stars_gas_templates, spectrum, noise, galaxy_data.velscale, start,
                    plot=False, moments=moments, degree=config.degree, mdegree=config.mdegree, 
                    component=component, gas_component=gas_component, gas_names=gas_names,
                    lam=galaxy_data.lam_gal, lam_temp=sps.lam_temp, 
                    bounds=bounds, mask=mask, quiet=True)
            
            # Check for negative velocity dispersion
            if hasattr(pp, 'sol') and len(pp.sol) > 1 and isinstance(pp.sol[1], (int, float)) and pp.sol[1] < 0:
                logging.warning(f"Negative velocity dispersion detected: {pp.sol[1]:.1f} km/s. Setting to 10 km/s.")
                pp.sol[1] = 10.0  # Force to a small positive value
                
            logging.debug(f"  - Combined fit successful: v={to_scalar(pp.sol[0]):.1f}, σ={to_scalar(pp.sol[1]):.1f}, χ²={to_scalar(pp.chi2):.3f}")
        except Exception as e:
            logging.warning(f"Gas+stellar fit failed for pixel ({i},{j}): {str(e)}")
            
            if config.fallback_to_simple_fit:
                logging.debug(f"  - Using fallback to stellar-only fit")
                # Fallback: just use stellar fit
                pp = pp_stars
                
                # Check if pp has valid solution
                if not hasattr(pp, 'sol') or pp.sol is None:
                    logging.warning(f"Fallback fit has no solution for pixel ({i},{j})")
                    return i, j, None
                
                # Add required gas attributes to make later code work
                pp.gas_bestfit = np.zeros_like(spectrum)
                if not hasattr(pp, 'gas_flux'):
                    pp.gas_flux = np.zeros(len(gas_names))
                
                # No gas template results
                pp.gas_bestfit_templates = np.zeros((spectrum.shape[0], len(gas_names)))
                logging.info(f"Used fallback stellar-only fit for pixel ({i},{j})")
            else:
                raise  # Re-raise the exception if we're not using fallback
        
        # Additional safety check after fitting
        if pp is None or not hasattr(pp, 'bestfit') or pp.bestfit is None:
            logging.warning(f"Missing valid fit results for pixel ({i},{j})")
            return i, j, None
            
        # Calculate S/N
        logging.debug(f"STEP: Calculating S/N and residuals")
        residuals = spectrum - pp.bestfit
        rms = robust_sigma(residuals[mask], zero=1)  # Only use masked region
        signal = np.median(spectrum[mask])
        snr = signal / rms if rms > 0 else 0
        logging.debug(f"  - Signal: {signal:.3f}, RMS: {rms:.3f}, S/N: {snr:.1f}")
        
        # Extract emission line fluxes and S/N
        logging.debug(f"STEP: Extracting emission line measurements")
        el_results = {}
        for name in config.gas_names:
            # Find matching gas names (allowing for prefixes in the template)
            matches = [idx for idx, gname in enumerate(gas_names) if name in gname]
            if matches:
                idx = matches[0]  # Use the first match
                
                # Extract the flux
                dlam = line_wave[idx] * galaxy_data.velscale / config.c
                
                # Safety check for gas_flux
                if hasattr(pp, 'gas_flux') and pp.gas_flux is not None and idx < len(pp.gas_flux):
                    flux = pp.gas_flux[idx] * dlam
                else:
                    flux = 0.0
                
                # Calculate A/N
                an = 0  # Default to 0
                if (hasattr(pp, 'gas_bestfit_templates') and 
                    pp.gas_bestfit_templates is not None and 
                    idx < pp.gas_bestfit_templates.shape[1]):
                    peak = np.max(pp.gas_bestfit_templates[:, idx])
                    an = peak / rms if rms > 0 else 0
                
                el_results[name] = {'flux': flux, 'an': an}
                logging.debug(f"  - {name}: flux={flux:.3e}, A/N={an:.1f}")
        
        # Calculate spectral indices
        logging.debug(f"STEP: Calculating spectral indices")
        indices = {}
        if config.compute_spectral_indices:
            # Process spectrum to remove emission lines if available
            if hasattr(pp, 'gas_bestfit') and pp.gas_bestfit is not None:
                clean_spectrum = spectrum - pp.gas_bestfit
            else:
                clean_spectrum = spectrum
            
            try:
                # Extract template flux and ensure it's 1D
                template_flux = best_template.flatten()
                
                # Create index calculator using the original approach
                # Important: Don't resample the entire template - let the calculator handle it per window
                calculator = LineIndexCalculator(
                    galaxy_data.lam_gal, clean_spectrum,
                    sps.lam_temp, template_flux,
                    velocity_correction=to_scalar(pp.sol[0]),
                    good_wavelength_range=config.good_wavelength_range if config.use_template_for_indices else None)
                
                # Calculate requested indices
                for index_name in config.line_indices:
                    try:
                        indices[index_name] = calculator.calculate_index(index_name)
                        logging.debug(f"  - {index_name}: {indices[index_name]:.4f}")
                    except Exception as e:
                        logging.warning(f"Failed to calculate index {index_name}: {str(e)}")
                        indices[index_name] = np.nan
            except Exception as e:
                logging.warning(f"Failed to initialize LineIndexCalculator: {str(e)}")
                import traceback
                logging.debug(traceback.format_exc())
                for index_name in config.line_indices:
                    indices[index_name] = np.nan
        
        # Generate plot if requested - BEFORE compiling results to catch any errors
        if config.make_plots and (i * j) % config.plot_every_n == 0:
            try:
                logging.debug(f"STEP: Generating diagnostic plot")
                plot_pixel_fit(i, j, pp, galaxy_data, config)
                logging.debug(f"  - Plot saved successfully")
            except Exception as e:
                logging.warning(f"Failed to create plot for pixel ({i},{j}): {str(e)}")
                import traceback
                logging.debug(traceback.format_exc())
                # Continue with result processing - don't let plot failure affect results
        
        # Compose results - ensure all values are properly handled
        logging.debug(f"STEP: Compiling final results")
        
        # Safely access pp attributes using to_scalar for numerical values
        sol_0 = 0.0
        sol_1 = 0.0
        if hasattr(pp, 'sol') and pp.sol is not None:
            if len(pp.sol) > 0:
                sol_0 = to_scalar(pp.sol[0])
            if len(pp.sol) > 1:
                sol_1 = to_scalar(pp.sol[1])
        
        # Ensure we have all required arrays
        bestfit = pp.bestfit if hasattr(pp, 'bestfit') and pp.bestfit is not None else np.zeros_like(spectrum)
        gas_bestfit = pp.gas_bestfit if hasattr(pp, 'gas_bestfit') and pp.gas_bestfit is not None else np.zeros_like(spectrum)
        opt_template = best_template.flatten() if best_template is not None else np.zeros_like(spectrum)
        
        results = {
            'success': True,
            'velocity': sol_0,
            'sigma': sol_1,
            'bestfit': bestfit,
            'weights': pp.weights if hasattr(pp, 'weights') and pp.weights is not None else np.zeros(1),
            'gas_bestfit': gas_bestfit,
            'optimal_template': opt_template,
            'rms': rms,
            'snr': snr,
            'el_results': el_results,
            'indices': indices,
            'apoly': pp.apoly if hasattr(pp, 'apoly') and pp.apoly is not None else None,
            'pp_obj': pp
        }
        
        logging.debug(f"===== PIXEL ({i},{j}) FIT COMPLETED SUCCESSFULLY =====")
        return i, j, results
    
    except Exception as e:
        logging.error(f"Error fitting pixel ({i},{j}): {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return i, j, None


def fit_pixel_grid(galaxy_data, sps, gas_templates, gas_names, line_wave, config):
    """
    Fit the entire grid of pixels using parallel processing.
    
    Parameters
    ----------
    galaxy_data : IFUDataCube
        Object containing the galaxy data
    sps : object
        Stellar population synthesis library
    gas_templates : array
        Gas emission line templates
    gas_names : array
        Names of gas emission lines
    line_wave : array
        Wavelengths of emission lines
    config : P2PConfig
        Configuration object
        
    Returns
    -------
    dict
        Fitting results for all pixels
    """
    # Create list of pixels to process
    ny, nx = galaxy_data.cube.shape[1:3]
    pixels = [(i, j, galaxy_data, sps, gas_templates, gas_names, line_wave, config) 
             for i in range(ny) for j in range(nx)]
    
    # Set up progress tracking
    total_pixels = len(pixels)
    completed = 0
    successful = 0
    
    logging.info(f"Starting P2P fitting for {config.galaxy_name} with {total_pixels} pixels")
    logging.info(f"Using {config.n_threads} parallel processes")
    
    # Use ProcessPoolExecutor for parallel processing
    start_time = time.time()
    
    results = {}
    
    with ProcessPoolExecutor(max_workers=config.n_threads) as executor:
        # Submit all pixels for processing
        future_to_pixel = {executor.submit(fit_single_pixel, pixel): pixel for pixel in pixels}
        
        # Process results as they complete
        with tqdm(total=total_pixels, desc="Fitting pixels") as pbar:
            for future in as_completed(future_to_pixel):
                i, j, result = future.result()
                completed += 1
                
                if result is not None:
                    results[(i, j)] = result
                    successful += 1
                
                # Update progress bar
                pbar.update(1)
                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_pixels - completed) / rate if rate > 0 else 0
                    logging.info(f"Processed {completed}/{total_pixels} pixels ({successful} successful). "
                                f"Rate: {rate:.1f} pix/s, ETA: {eta/60:.1f} minutes")
    
    end_time = time.time()
    logging.info(f"Completed P2P fitting in {end_time - start_time:.1f} seconds")
    logging.info(f"Successfully fitted {successful}/{total_pixels} pixels")
    
    return results


def process_results(galaxy_data, results, config):
    """
    Process and store the fitting results.
    
    Parameters
    ----------
    galaxy_data : IFUDataCube
        Object containing the galaxy data
    results : dict
        Fitting results from fit_pixel_grid
    config : P2PConfig
        Configuration object
    """
    # Extract dimensions
    ny, nx = galaxy_data.cube.shape[1:3]
    npix = galaxy_data.spectra.shape[0]
    
    # Initialize arrays for results if not already done
    galaxy_data.velfield = np.full((ny, nx), np.nan)
    galaxy_data.sigfield = np.full((ny, nx), np.nan)
    
    # Process each successful fit
    for (i, j), result in results.items():
        if result['success']:
            # Store kinematic measurements
            galaxy_data.velfield[i, j] = result['velocity']
            galaxy_data.sigfield[i, j] = result['sigma']
            
            # Store best fit spectra
            galaxy_data.bestfit_field[:, i, j] = result['bestfit']
            galaxy_data.optimal_templates[:, i, j] = result['optimal_template']
            
            # Store emission line results
            for name, data in result['el_results'].items():
                if name in galaxy_data.el_flux_maps:
                    galaxy_data.el_flux_maps[name][i, j] = data['flux']
                    galaxy_data.el_snr_maps[name][i, j] = data['an']
            
            # Store spectral indices
            for name, value in result['indices'].items():
                if name in galaxy_data.index_maps:
                    galaxy_data.index_maps[name][i, j] = value
    
    # Create dataframe for all results
    df_data = []
    
    # Calculate galaxy center
    center_x = galaxy_data.CRVAL1 + ((ny * galaxy_data.CD1_2 + nx * galaxy_data.CD1_1) / 2)
    center_y = galaxy_data.CRVAL2 + ((ny * galaxy_data.CD2_2 + nx * galaxy_data.CD2_1) / 2)
    
    for (i, j), result in results.items():
        if not result['success']:
            continue
            
        # Calculate position and radius
        ra = galaxy_data.CRVAL1 + (i * galaxy_data.CD1_2) + (j * galaxy_data.CD1_1)
        dec = galaxy_data.CRVAL2 + (i * galaxy_data.CD2_2) + (j * galaxy_data.CD2_1)
        radius = np.sqrt((ra - center_x)**2 + (dec - center_y)**2)
        
        # Create row for this pixel
        row = {
            'i': i,
            'j': j,
            'k_index': i * nx + j,
            'velocity': result['velocity'],
            'sigma': result['sigma'],
            'SNR': result['snr'],
            'R': radius,
        }
        
        # Add emission line data
        for name, data in result['el_results'].items():
            row[f'{name}_flux'] = data['flux']
            row[f'{name}_SNR'] = data['an']
        
        # Add spectral indices
        for name, value in result['indices'].items():
            row[f'{name}_index'] = value
        
        df_data.append(row)
    
    # Create dataframe and save to CSV
    if df_data:
        df = pd.DataFrame(df_data)
        csv_path = config.output_dir / f"{config.galaxy_name}_P2P_results.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved results to {csv_path}")
    else:
        logging.warning("No successful fits to save to CSV")


### ------------------------------------------------- ###
# Visualization Functions
### ------------------------------------------------- ###

def plot_pixel_fit(i, j, pp, galaxy_data, config):
    """
    Create diagnostic plot for a single pixel fit.
    
    Parameters
    ----------
    i, j : int
        Pixel coordinates
    pp : ppxf object
        pPXF fit result
    galaxy_data : IFUDataCube
        Object containing the galaxy data
    config : P2PConfig
        Configuration object
    """
    try:
        logging.debug(f"===== PLOT GENERATION DEBUG FOR PIXEL ({i},{j}) =====")
        
        # Create plot directory if it doesn't exist
        plot_dir = config.plot_dir / 'P2P_res'
        os.makedirs(plot_dir, exist_ok=True)
        
        # Get data
        k_index = i * galaxy_data.cube.shape[2] + j
        lam_gal = galaxy_data.lam_gal
        spectrum = galaxy_data.spectra[:, k_index]
        
        # Check data shapes
        logging.debug(f"STEP: Checking data shapes for plotting")
        logging.debug(f"  - lam_gal shape: {lam_gal.shape}")
        logging.debug(f"  - spectrum shape: {spectrum.shape}")
        logging.debug(f"  - pp.bestfit shape: {pp.bestfit.shape if hasattr(pp, 'bestfit') else 'Not available'}")
        
        if hasattr(pp, 'gas_bestfit'):
            logging.debug(f"  - pp.gas_bestfit shape: {pp.gas_bestfit.shape}")
        
        if hasattr(pp, 'gas_bestfit_templates'):
            logging.debug(f"  - pp.gas_bestfit_templates shape: {pp.gas_bestfit_templates.shape}")
        
        # Define spectral index windows for visualization
        index_windows = {
            'Hbeta': {'blue': (4827.875, 4847.875), 'line': (4847.875, 4876.625), 'red': (4876.625, 4891.625)},
            'Fe5015': {'blue': (4946.500, 4977.750), 'line': (4977.750, 5054.000), 'red': (5054.000, 5065.250)},
            'Mgb': {'blue': (5142.625, 5161.375), 'line': (5160.125, 5192.625), 'red': (5191.375, 5206.375)}
        }
        
        # Create figure with 3 panels - regular subplot layout for better compatibility
        fig = plt.figure(figsize=(16, 12), dpi=config.dpi)
        
        # Top panel: Original data and fit
        ax1 = fig.add_subplot(3, 1, 1)
        
        # Middle panel: Residuals
        ax2 = fig.add_subplot(3, 1, 2)
        
        # Bottom panel: Emission components
        ax3 = fig.add_subplot(3, 1, 3)
        
        # Mark good wavelength range if provided
        if hasattr(config, 'good_wavelength_range'):
            good_min, good_max = config.good_wavelength_range
            ax1.axvspan(good_min, good_max, color='lightgreen', alpha=0.1, label='Good Wavelength Range')
            ax2.axvspan(good_min, good_max, color='lightgreen', alpha=0.1)
            ax3.axvspan(good_min, good_max, color='lightgreen', alpha=0.1)
        
        # Ensure pp.bestfit exists - use spectrum if not
        bestfit = pp.bestfit if hasattr(pp, 'bestfit') and pp.bestfit is not None else spectrum
        
        # Plot original spectrum and best fit
        ax1.plot(lam_gal, spectrum, c='tab:blue', lw=1, alpha=.9, 
                label=f"{config.galaxy_name}\npixel:[{i},{j}]")
        ax1.plot(lam_gal, bestfit, '--', c='tab:red', alpha=.9)
        
        # Highlight spectral feature regions
        for index_name in ['Hbeta', 'Fe5015', 'Mgb']:
            if index_name in index_windows:
                windows = index_windows[index_name]
                
                # Line region (central bandpass)
                line_start, line_end = windows['line']
                ax1.axvspan(line_start, line_end, color='tab:gray', alpha=.5, zorder=0)
                ax2.axvspan(line_start, line_end, color='tab:gray', alpha=.5, zorder=0)
                ax3.axvspan(line_start, line_end, color='tab:gray', alpha=.5, zorder=0)
                
                # Blue continuum region
                blue_start, blue_end = windows['blue']
                ax1.axvspan(blue_start, blue_end, color='tab:gray', alpha=.3, zorder=0)
                ax2.axvspan(blue_start, blue_end, color='tab:gray', alpha=.3, zorder=0)
                ax3.axvspan(blue_start, blue_end, color='tab:gray', alpha=.3, zorder=0)
                
                # Red continuum region
                red_start, red_end = windows['red']
                ax1.axvspan(red_start, red_end, color='tab:gray', alpha=.3, zorder=0)
                ax2.axvspan(red_start, red_end, color='tab:gray', alpha=.3, zorder=0)
                ax3.axvspan(red_start, red_end, color='tab:gray', alpha=.3, zorder=0)
        
        # Plot emission line components if available
        if hasattr(pp, 'gas_bestfit_templates') and pp.gas_bestfit_templates is not None:
            if pp.gas_bestfit_templates.shape[1] > 0:
                ax1.plot(lam_gal, pp.gas_bestfit_templates[:, 0], color='tab:orange', zorder=1, alpha=.9)
            
            if pp.gas_bestfit_templates.shape[1] > 1:
                ax1.plot(lam_gal, pp.gas_bestfit_templates[:, 1], color='tab:purple', zorder=1, alpha=.9)
        
        ax1.plot(lam_gal, bestfit, '-', lw=.7, c='tab:red')
        
        # Set up the residuals plot
        ax2.plot(lam_gal, np.zeros(lam_gal.shape), '-', color='k', lw=.7, alpha=.9, zorder=0)
        
        # Calculate median and std of residuals
        residuals = spectrum - bestfit
        median_residual = np.median(residuals)
        std_residual = np.std(residuals)
        
        ax2.plot(lam_gal, [median_residual] * lam_gal.shape[0], '--', 
                color='tab:blue', lw=1, alpha=.9, zorder=1)
        
        # Plot residual range
        upper_bound = median_residual + std_residual
        lower_bound = median_residual - std_residual
        ax2.fill_between([min(lam_gal), max(lam_gal)], [upper_bound, upper_bound], 
                        [lower_bound, lower_bound], color='tab:gray', alpha=.2,
                        label=f'Range: {median_residual:.3f}±{std_residual:.3f}', zorder=1)
        
        # Plot residuals
        ax2.plot(lam_gal, residuals, '+', ms=2, mew=3, color='tab:green', alpha=.9, zorder=2)
        
        # Bottom panel: Raw residuals and emission components
        gas_bestfit = pp.gas_bestfit if hasattr(pp, 'gas_bestfit') and pp.gas_bestfit is not None else np.zeros_like(spectrum)
        stellar_bestfit = bestfit - gas_bestfit
        ax3.plot(lam_gal, spectrum - stellar_bestfit, '+', ms=2, mew=3, 
                color='tab:green', alpha=.9, zorder=2)
        
        # Plot emission components if available
        if hasattr(pp, 'gas_bestfit_templates') and pp.gas_bestfit_templates is not None:
            if pp.gas_bestfit_templates.shape[1] > 0:
                ax3.plot(lam_gal, pp.gas_bestfit_templates[:, 0], 
                      color='tab:orange', zorder=2, alpha=.9)
            
            if pp.gas_bestfit_templates.shape[1] > 1:
                ax3.plot(lam_gal, pp.gas_bestfit_templates[:, 1], 
                      color='tab:purple', zorder=2, alpha=.9)
            
            # Plot combined emission
            if hasattr(pp, 'gas_bestfit') and pp.gas_bestfit is not None:
                ax3.plot(lam_gal, pp.gas_bestfit, lw=.7, color='tab:red', zorder=2, alpha=.9)
        
        # Set up axis formatting
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.tick_params(axis='both', which='both', labelsize='x-small', 
                        right=True, top=True, direction='in')
            ax.set_xlim(min(lam_gal), max(lam_gal))
            ax.grid(True, alpha=0.3)
        
        # Set y-axis limits
        ax1.set_ylim(0, max(spectrum) * 1.1)
        ax2.set_ylim(min(residuals) * 1.2, max(residuals) * 1.2)
        
        emission_residuals = spectrum - stellar_bestfit
        ax3.set_ylim(min(emission_residuals) * 1.2, max(emission_residuals) * 1.2)
        
        # Set labels
        ax3.set_xlabel(r'Wavelength [$\AA$]', size=11)
        ax1.set_ylabel('Flux', size=11)
        ax2.set_ylabel('Residuals', size=11)
        ax3.set_ylabel('Emission Components', size=11)
        
        # Add legends
        ax1.legend()
        ax2.legend()
        
        # Check formatting parameters and convert to scalar
        logging.debug(f"STEP: Checking formatting parameters")
        
        # Use to_scalar helper function for safe conversion
        velocity = to_scalar(pp.sol[0]) if hasattr(pp, 'sol') and pp.sol is not None and len(pp.sol) > 0 else 0.0
        sigma = to_scalar(pp.sol[1]) if hasattr(pp, 'sol') and pp.sol is not None and len(pp.sol) > 1 else 0.0
        chi2 = to_scalar(pp.chi2) if hasattr(pp, 'chi2') and pp.chi2 is not None else 0.0
        
        logging.debug(f"  - After conversion: velocity={velocity}, sigma={sigma}, chi2={chi2}")
        
        # Add title with kinematic info using safe scalar values
        ax1.set_title(f"Velocity: {velocity:.1f} km/s, σ: {sigma:.1f} km/s, χ²: {chi2:.3f}")
        
        # Add more space between subplots and adjust layout
        plt.subplots_adjust(hspace=0.3)
        
        # Save the plot
        plot_path = plot_dir / f"{config.galaxy_name}_pixel_{i}_{j}.pdf"
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        
        logging.debug(f"Plot saved to {plot_path}")
        
    except Exception as e:
        logging.error(f"Error in plot generation for pixel ({i},{j}): {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())


def create_summary_plots(galaxy_data, config):
    """
    Create summary plots of the fitting results.
    
    Parameters
    ----------
    galaxy_data : IFUDataCube
        Object containing the galaxy data
    config : P2PConfig
        Configuration object
    """
    try:
        # Create plots directory
        os.makedirs(config.plot_dir, exist_ok=True)
        
        # 1. Kinematic maps
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Velocity map
        vmax = np.nanpercentile(np.abs(galaxy_data.velfield), 90)
        im0 = axes[0].imshow(galaxy_data.velfield, origin='lower', cmap='RdBu_r', 
                           vmin=-vmax, vmax=vmax)
        axes[0].set_title('Velocity [km/s]')
        plt.colorbar(im0, ax=axes[0])
        
        # Velocity dispersion map
        sigma_max = np.nanpercentile(galaxy_data.sigfield, 95)
        im1 = axes[1].imshow(galaxy_data.sigfield, origin='lower', cmap='viridis', 
                           vmin=0, vmax=sigma_max)
        axes[1].set_title('Velocity Dispersion [km/s]')
        plt.colorbar(im1, ax=axes[1])
        
        plt.suptitle(f"{config.galaxy_name} - Stellar Kinematics")
        plt.tight_layout()
        plt.savefig(config.plot_dir / f"{config.galaxy_name}_kinematics.png", dpi=150)
        plt.close()
        
        # 2. Emission line maps
        if config.compute_emission_lines and len(config.gas_names) > 0:
            n_lines = len(config.gas_names)
            fig, axes = plt.subplots(2, n_lines, figsize=(4*n_lines, 8))
            
            if n_lines == 1:  # Handle case with single emission line
                axes = np.array([[axes[0]], [axes[1]]])
            
            for i, name in enumerate(config.gas_names):
                # Flux map
                flux_map = galaxy_data.el_flux_maps[name]
                vmax = np.nanpercentile(flux_map, 95)
                im = axes[0, i].imshow(flux_map, origin='lower', cmap='inferno', vmin=0, vmax=vmax)
                axes[0, i].set_title(f"{name} Flux")
                plt.colorbar(im, ax=axes[0, i])
                
                # S/N map
                snr_map = galaxy_data.el_snr_maps[name]
                im = axes[1, i].imshow(snr_map, origin='lower', cmap='viridis', vmin=0, vmax=5)
                axes[1, i].set_title(f"{name} S/N")
                plt.colorbar(im, ax=axes[1, i])
            
            plt.suptitle(f"{config.galaxy_name} - Emission Lines")
            plt.tight_layout()
            plt.savefig(config.plot_dir / f"{config.galaxy_name}_emission_lines.png", dpi=150)
            plt.close()
        
        # 3. Spectral index maps
        if config.compute_spectral_indices and len(config.line_indices) > 0:
            n_indices = len(config.line_indices)
            n_cols = min(3, n_indices)
            n_rows = (n_indices + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
            axes = np.atleast_2d(axes)
            
            for i, name in enumerate(config.line_indices):
                row = i // n_cols
                col = i % n_cols
                
                index_map = galaxy_data.index_maps[name]
                vmin = np.nanpercentile(index_map, 5)
                vmax = np.nanpercentile(index_map, 95)
                
                im = axes[row, col].imshow(index_map, origin='lower', cmap='viridis', 
                                        vmin=vmin, vmax=vmax)
                axes[row, col].set_title(f"{name} Index")
                plt.colorbar(im, ax=axes[row, col])
            
            # Hide empty subplots
            for i in range(n_indices, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].axis('off')
            
            plt.suptitle(f"{config.galaxy_name} - Spectral Indices")
            plt.tight_layout()
            plt.savefig(config.plot_dir / f"{config.galaxy_name}_indices.png", dpi=150)
            plt.close()
    
    except Exception as e:
        logging.error(f"Error creating summary plots: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())


### ------------------------------------------------- ###
# Save Results
### ------------------------------------------------- ###

def save_results_to_fits(galaxy_data, config):
    """
    Save results to FITS files.
    
    Parameters
    ----------
    galaxy_data : IFUDataCube
        Object containing the galaxy data
    config : P2PConfig
        Configuration object
    """
    try:
        # Create a header based on the input data
        hdr = fits.Header()
        hdr['OBJECT'] = config.galaxy_name
        hdr['REDSHIFT'] = config.redshift
        hdr['CD1_1'] = galaxy_data.CD1_1
        hdr['CD1_2'] = galaxy_data.CD1_2
        hdr['CD2_1'] = galaxy_data.CD2_1
        hdr['CD2_2'] = galaxy_data.CD2_2
        hdr['CRVAL1'] = galaxy_data.CRVAL1
        hdr['CRVAL2'] = galaxy_data.CRVAL2
        
        # Add good wavelength range to header
        if hasattr(config, 'good_wavelength_range'):
            hdr['WAVGOOD0'] = config.good_wavelength_range[0]
            hdr['WAVGOOD1'] = config.good_wavelength_range[1]
        
        # Save kinematic maps
        hdu_vel = fits.PrimaryHDU(galaxy_data.velfield, header=hdr)
        hdu_vel.header['CONTENT'] = 'Stellar velocity field'
        hdu_vel.header['BUNIT'] = 'km/s'
        hdu_vel.writeto(config.output_dir / f"{config.galaxy_name}_velfield.fits", overwrite=True)
        
        hdu_sig = fits.PrimaryHDU(galaxy_data.sigfield, header=hdr)
        hdu_sig.header['CONTENT'] = 'Stellar velocity dispersion'
        hdu_sig.header['BUNIT'] = 'km/s'
        hdu_sig.writeto(config.output_dir / f"{config.galaxy_name}_sigfield.fits", overwrite=True)
        
        # Save emission line maps
        for name in config.gas_names:
            if name in galaxy_data.el_flux_maps:
                hdu = fits.PrimaryHDU(galaxy_data.el_flux_maps[name], header=hdr)
                hdu.header['CONTENT'] = f'{name} emission line flux'
                hdu.header['BUNIT'] = 'flux units'
                hdu.writeto(config.output_dir / f"{config.galaxy_name}_{name}_flux.fits", overwrite=True)
                
                hdu = fits.PrimaryHDU(galaxy_data.el_snr_maps[name], header=hdr)
                hdu.header['CONTENT'] = f'{name} emission line S/N'
                hdu.header['BUNIT'] = 'ratio'
                hdu.writeto(config.output_dir / f"{config.galaxy_name}_{name}_snr.fits", overwrite=True)
        
        # Save spectral index maps
        for name in config.line_indices:
            if name in galaxy_data.index_maps:
                hdu = fits.PrimaryHDU(galaxy_data.index_maps[name], header=hdr)
                hdu.header['CONTENT'] = f'{name} spectral index'
                hdu.header['BUNIT'] = 'Angstrom'
                hdu.writeto(config.output_dir / f"{config.galaxy_name}_{name}_index.fits", overwrite=True)
        
        # Save best-fit spectra cube
        wave_hdr = hdr.copy()
        wave_hdr['CRVAL3'] = galaxy_data.lam_gal[0]  # First wavelength value
        wave_hdr['CD3_3'] = np.mean(np.diff(galaxy_data.lam_gal))  # Average wavelength step
        
        hdu = fits.PrimaryHDU(galaxy_data.bestfit_field, header=wave_hdr)
        hdu.header['CONTENT'] = 'Best-fit spectra'
        hdu.header['BUNIT'] = 'flux units'
        hdu.writeto(config.output_dir / f"{config.galaxy_name}_bestfit.fits", overwrite=True)
    
    except Exception as e:
        logging.error(f"Error saving results to FITS: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())


### ------------------------------------------------- ###
# Error Handling & Optimization
### ------------------------------------------------- ###

def optimize_ppxf_params(config, problem_type=None):
    """
    Optimize pPXF parameters based on the specific problem encountered.
    
    Parameters
    ----------
    config : P2PConfig
        Configuration object
    problem_type : str, optional
        Type of problem to address ('mask_error', 'broadcast_error', etc.)
        
    Returns
    -------
    P2PConfig
        Updated configuration object
    """
    if problem_type == "mask_error":
        # Fix for issues with determine_mask function
        logging.info("Applying optimizations for mask errors")
        config.mask_width = 500  # Use narrower mask width 
        
    elif problem_type == "broadcast_error":
        # If encountering broadcast errors, simplify the fit
        logging.info("Applying optimizations for broadcast errors")
        config.degree = 0  # Use simpler additive polynomial
        config.mdegree = -1  # Disable multiplicative polynomial
        config.retry_with_degree_zero = True
        config.fallback_to_simple_fit = True
        config.safe_mode = True  # Enable extra safety measures
        
    elif problem_type == "slow_convergence":
        # If fits are taking too long
        logging.info("Applying optimizations for slow convergence")
        config.n_threads = max(2, config.n_threads // 2)  # Reduce thread count to limit memory usage
        config.moments = [2, 2]  # Simplify to just velocity and dispersion
        
    elif problem_type == "memory_error":
        # If running out of memory
        logging.info("Applying optimizations for memory errors")
        config.n_threads = max(1, config.n_threads // 4)  # Drastically reduce thread count
        config.make_plots = False  # Disable plot generation to save memory
        
    elif problem_type == "gas_fit_error":
        # If having problems with gas fitting
        logging.info("Applying optimizations for gas fitting errors")
        config.fwhm_gas = 2.0  # Use wider lines for better stability
        config.fallback_to_simple_fit = True  # Fall back to stellar-only fit if gas fit fails
    
    elif problem_type == "format_error":
        # Fix for format string errors
        logging.info("Applying optimizations for formatting errors")
        config.safe_mode = True
        config.fallback_to_simple_fit = True
        
    return config


### ------------------------------------------------- ###
# Single Pixel Testing
### ------------------------------------------------- ###

def test_single_pixel(config=None, i=0, j=0, debug_level=logging.DEBUG):
    """
    Test fitting for a single pixel, for debugging purposes.
    
    Parameters
    ----------
    config : P2PConfig or str, optional
        Configuration object or path to configuration file
    i, j : int
        Pixel coordinates (row, column)
    debug_level : int
        Debug logging level
        
    Returns
    -------
    dict
        Fitting results
    """
    # Set more detailed logging level
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(debug_level)
    
    try:
        # Set up configuration
        if config is None:
            config = P2PConfig()
        elif isinstance(config, str):
            config = P2PConfig.load(config)
        
        # Prepare test environment
        logging.info(f"=== STARTING SINGLE PIXEL TEST FOR PIXEL ({i},{j}) ===")
        logging.info(f"Using config: {config.galaxy_name}")
        
        # In debug mode, enable extra safety
        if debug_level == logging.DEBUG:
            logging.info("Enabling safe mode for debugging")
            config.safe_mode = True
            config.fallback_to_simple_fit = True
            config.retry_with_degree_zero = True
        
        # Create necessary directories
        config.create_directories()
        
        # 1. Load data
        logging.info("Loading data...")
        galaxy_data = IFUDataCube(config.get_data_path(), config.lam_range_temp, config.redshift, config)
        logging.info(f"Data shape: {galaxy_data.cube.shape}")
        logging.info(f"Wavelength range: [{galaxy_data.lam_gal[0]:.2f}, {galaxy_data.lam_gal[-1]:.2f}]")
        logging.info(f"Good wavelength range: {config.good_wavelength_range}")
        
        # 2. Prepare templates
        logging.info("Preparing templates...")
        sps, gas_templates, gas_names, line_wave = prepare_templates(config, galaxy_data.velscale)
        logging.info(f"Stellar templates shape: {sps.templates.shape}")
        logging.info(f"Gas templates shape: {gas_templates.shape}")
        
        # 3. Fit single pixel
        logging.info(f"Starting fit for pixel ({i},{j})...")
        args = (i, j, galaxy_data, sps, gas_templates, gas_names, line_wave, config)
        pixel_i, pixel_j, result = fit_single_pixel(args)
        
        # 4. If fit successful, generate diagnostic plot
        if result is not None and result.get('success', False):
            logging.info(f"Fit successful! Velocity: {result['velocity']:.1f} km/s, Sigma: {result['sigma']:.1f} km/s")
            
            # Get pPXF object for plotting
            pp = result.get('pp_obj')
            if pp is not None:
                # Generate diagnostic plot
                logging.info("Generating diagnostic plot...")
                try:
                    plot_pixel_fit(i, j, pp, galaxy_data, config)
                    logging.info("Plot generated successfully")
                except Exception as e:
                    logging.error(f"Error generating plot: {str(e)}")
            
            # If spectral indices were calculated, display results
            if config.compute_spectral_indices:
                logging.info("Spectral indices results:")
                for name, value in result['indices'].items():
                    # Check if index window is outside good wavelength range
                    windows = LineIndexCalculator.define_line_windows(name)
                    if windows:
                        line_min = windows['blue'][0]
                        line_max = windows['red'][1]
                        outside_range = ""
                        if (line_min < config.good_wavelength_range[0] or 
                            line_max > config.good_wavelength_range[1]):
                            outside_range = " (partially outside good wavelength range - using template substitution)"
                        logging.info(f"  - {name}: {value:.4f}{outside_range}")
                    else:
                        logging.info(f"  - {name}: {value:.4f}")
            
            # If emission lines were calculated, display results
            if config.compute_emission_lines:
                logging.info("Emission line results:")
                for name, data in result['el_results'].items():
                    logging.info(f"  - {name}: flux={data['flux']:.4e}, S/N={data['an']:.2f}")
            
            return result
        else:
            logging.error(f"Fit failed for pixel ({i},{j})")
            logging.info("Trying with even simpler settings...")
            
            # Try with ultra-simplified settings
            config.degree = -1  # No additive polynomials 
            config.moments = [2, 2]  # Just velocity and dispersion
            config.fwhm_gas = 2.0  # Wider gas lines
            
            # Second attempt
            pixel_i, pixel_j, result = fit_single_pixel(args)
            
            if result is not None and result.get('success', False):
                logging.info("Second attempt successful with simplified settings")
                return result
            else:
                logging.error("Failed even with simplified settings")
                return None
            
    except Exception as e:
        logging.error(f"Error during single pixel test: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None
    
    finally:
        # Restore original logging level
        logging.getLogger().setLevel(original_level)


def debug_line_index_calculator(wave, flux, fit_wave, fit_flux, line_name="Hbeta", velocity=0, 
                             good_wavelength_range=None):
    """
    Debug LineIndexCalculator functionality independently.
    
    Parameters
    ----------
    wave : array-like
        Original spectrum wavelength array
    flux : array-like
        Original spectrum flux array
    fit_wave : array-like
        Fitted spectrum wavelength array
    fit_flux : array-like
        Fitted spectrum flux array
    line_name : str
        Absorption line name to calculate
    velocity : float
        Velocity correction in km/s
    good_wavelength_range : list, optional
        Good wavelength range [min, max] for template substitution
        
    Returns
    -------
    float
        Absorption line index value
    """
    logging.info(f"=== TESTING LINE INDEX CALCULATOR FOR {line_name} ===")
    logging.info(f"Input shapes - wave: {wave.shape}, flux: {flux.shape}")
    logging.info(f"Template shapes - fit_wave: {fit_wave.shape}, fit_flux: {fit_flux.shape}")
    
    try:
        # Check input arrays
        if len(wave) != len(flux):
            logging.error(f"Wavelength and flux length mismatch: wave={len(wave)}, flux={len(flux)}")
            # Try to fix
            min_len = min(len(wave), len(flux))
            wave = wave[:min_len]
            flux = flux[:min_len]
            logging.info(f"Truncated to common length: {min_len}")
        
        if len(fit_wave) != len(fit_flux):
            logging.error(f"Template wavelength and flux length mismatch: fit_wave={len(fit_wave)}, fit_flux={len(fit_flux)}")
            # Try to fix
            min_len = min(len(fit_wave), len(fit_flux))
            fit_wave = fit_wave[:min_len]
            fit_flux = fit_flux[:min_len]
            logging.info(f"Truncated to common length: {min_len}")
        
        # Create calculator
        calculator = LineIndexCalculator(
            wave, flux, fit_wave, fit_flux, 
            velocity_correction=velocity,
            good_wavelength_range=good_wavelength_range
        )
        
        # Get index windows
        windows = calculator.define_line_windows(line_name)
        
        # Check if index is outside good wavelength range
        if good_wavelength_range is not None:
            line_min = windows['blue'][0]
            line_max = windows['red'][1]
            if line_min < good_wavelength_range[0] or line_max > good_wavelength_range[1]:
                logging.info(f"Index {line_name} ({line_min:.2f}-{line_max:.2f}) partially outside good wavelength range "
                           f"({good_wavelength_range[0]:.2f}-{good_wavelength_range[1]:.2f})")
                logging.info(f"Template substitution will be used for sections outside good wavelength range")
        
        # Calculate index
        index = calculator.calculate_index(line_name)
        logging.info(f"Successfully calculated {line_name} index: {index:.6f}")
        
        # Visualize result
        output_path = f"debug_{line_name}_index.png"
        calculator.plot_line_fit(line_name, output_path)
        logging.info(f"Diagnostic plot saved to {output_path}")
        
        return index
        
    except Exception as e:
        logging.error(f"Error calculating {line_name} index: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None


### ------------------------------------------------- ###
# Main Driver Functions
### ------------------------------------------------- ###

def run_p2p_analysis(config=None, problem_fixes=None):
    """
    Run the full P2P analysis pipeline.
    
    Parameters
    ----------
    config : P2PConfig or str, optional
        Configuration object or path to configuration file
    problem_fixes : str, optional
        Specific problems to fix (mask_error, broadcast_error, etc.)
        
    Returns
    -------
    IFUDataCube
        Object containing all processed data and results
    """
    # Set up configuration
    if config is None:
        config = P2PConfig()
    elif isinstance(config, str):
        config = P2PConfig.load(config)
    
    # Apply problem-specific optimizations if requested
    if problem_fixes:
        config = optimize_ppxf_params(config, problem_fixes)
    
    # Create output directories
    config.create_directories()
    
    # Set up logging
    log_path = config.output_dir / f"{config.galaxy_name}_p2p_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    # Save configuration
    config.save()
    
    # Log start of analysis
    logging.info(f"Starting P2P analysis for {config.galaxy_name}")
    logging.info(f"Data file: {config.get_data_path()}")
    logging.info(f"Configuration settings: stellar_moments={config.moments}, degree={config.degree}, "
               f"mdegree={config.mdegree}, fwhm_gas={config.fwhm_gas}, mask_width={config.mask_width}")
    if hasattr(config, 'good_wavelength_range'):
        logging.info(f"Good wavelength range: [{config.good_wavelength_range[0]:.2f}, {config.good_wavelength_range[1]:.2f}]")
    start_time = time.time()
    
    try:
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        galaxy_data = IFUDataCube(config.get_data_path(), config.lam_range_temp, config.redshift, config)
        
        # Check spectral index windows against good wavelength range
        if config.compute_spectral_indices and hasattr(config, 'good_wavelength_range'):
            for index_name in config.line_indices:
                windows = LineIndexCalculator.define_line_windows(index_name)
                if windows:
                    line_min = windows['blue'][0]  # Blue start
                    line_max = windows['red'][1]   # Red end
                    
                    if line_min < config.good_wavelength_range[0] or line_max > config.good_wavelength_range[1]:
                        logging.warning(f"Index {index_name} wavelength range ({line_min:.2f}-{line_max:.2f}) "
                                       f"partially outside good wavelength range "
                                       f"({config.good_wavelength_range[0]:.2f}-{config.good_wavelength_range[1]:.2f})")
                        if config.use_template_for_indices:
                            logging.info(f"Template substitution will be used for spectral index {index_name}")
        
        # Prepare templates
        logging.info("Preparing stellar and gas templates...")
        sps, gas_templates, gas_names, line_wave = prepare_templates(config, galaxy_data.velscale)
        
        # Run pixel fitting
        logging.info("Starting pixel-by-pixel fitting...")
        results = fit_pixel_grid(galaxy_data, sps, gas_templates, gas_names, line_wave, config)
        
        # Process results
        logging.info("Processing and storing results...")
        process_results(galaxy_data, results, config)
        
        # Create summary plots
        if config.make_plots:
            logging.info("Creating summary plots...")
            create_summary_plots(galaxy_data, config)
        
        # Save results to FITS files
        logging.info("Saving maps to FITS files...")
        save_results_to_fits(galaxy_data, config)
        
        # Log completion
        end_time = time.time()
        logging.info(f"P2P analysis completed in {end_time - start_time:.1f} seconds")
        
        return galaxy_data
        
    except Exception as e:
        logging.error(f"Error in P2P analysis: {str(e)}")
        logging.exception("Stack trace:")
        raise


### ------------------------------------------------- ###
# Command-line Interface
### ------------------------------------------------- ###

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="P2P spectral analysis of IFU data")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--create-config", action="store_true", 
                      help="Create a default configuration file")
    parser.add_argument("--config-output", type=str, default="p2p_config.json",
                      help="Output path for generated configuration file")
    parser.add_argument("--fix", type=str, 
                      choices=["mask_error", "broadcast_error", "slow_convergence", 
                               "memory_error", "gas_fit_error", "format_error"],
                      help="Apply specific problem fixes")
    
    # Add single pixel test parameters
    parser.add_argument("--test-pixel", action="store_true",
                      help="Run test on a single pixel")
    parser.add_argument("--pixel-i", type=int, default=0,
                      help="Row index of pixel to test")
    parser.add_argument("--pixel-j", type=int, default=0,
                      help="Column index of pixel to test")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    parser.add_argument("--no-template-substitution", action="store_true",
                      help="Disable template substitution for spectral indices")
    
    args = parser.parse_args()
    
    # Set logging level
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.create_config:
        config = P2PConfig()
        config.save(args.config_output)
        print(f"Default configuration saved to {args.config_output}")
    elif args.test_pixel:
        # Load configuration
        config = P2PConfig.load(args.config) if args.config else P2PConfig()
        
        # Apply problem fixes
        if args.fix:
            config = optimize_ppxf_params(config, args.fix)
        
        # Handle template substitution flag
        if args.no_template_substitution:
            config.use_template_for_indices = False
            
        # Run single pixel test
        result = test_single_pixel(config, args.pixel_i, args.pixel_j, 
                                 debug_level=log_level)
        
        if result:
            print(f"Single pixel test completed successfully, results saved")
        else:
            print(f"Single pixel test failed, check log file")
    else:
        # Handle template substitution flag
        if args.no_template_substitution and args.config:
            config = P2PConfig.load(args.config)
            config.use_template_for_indices = False
            run_p2p_analysis(config, args.fix)
        elif args.no_template_substitution:
            config = P2PConfig()
            config.use_template_for_indices = False
            run_p2p_analysis(config, args.fix)
        elif args.config:
            run_p2p_analysis(args.config, args.fix)
        else:
            run_p2p_analysis(problem_fixes=args.fix)