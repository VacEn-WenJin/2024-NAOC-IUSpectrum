"""
Pixel-to-Pixel (P2P) IFU Spectral Analysis Pipeline

This module performs pixel-by-pixel spectral fitting of IFU data using pPXF,
calculates emission line properties, and computes spectral indices.

Features:
- Multi-threaded pixel fitting
- Centralized configuration
- Efficient spectral index calculation
- Robust error handling
- Customizable visualization

Version 3.0.0    2025Mar07    Optimized Implementation
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
        self.base_dir = Path('.')
        self.data_dir = self.base_dir / 'Ori_Data'
        self.output_dir = self.base_dir / 'FitData' / f'Fit_DS_{datetime.now().strftime("%y%b%d")}_{self.galaxy_name}'
        self.plot_dir = self.base_dir / 'FitPlot' / f'Fit_{datetime.now().strftime("%y%b%d")}_{self.galaxy_name}'
        
        # Spectral range and indices
        self.lam_range_temp = [4800, 5250]  # Wavelength range in Angstroms
        self.line_indices = ['Hbeta', 'Fe5015', 'Mgb']
        
        # pPXF fitting parameters
        self.degree = 3  # Polynomial degree for additive component
        self.mdegree = -1  # Polynomial degree for multiplicative component (disabled)
        self.moments = [4, 2]  # Moments to fit for stellar and gas components
        self.gas_names = ['Hbeta', '[OIII]5007']  # Gas lines to fit
        self.ngas_comp = 1  # Number of gas components
        
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
    
    Supports calculation of common spectral indices like Hbeta, Mgb, Fe5015, etc.
    Handles velocity correction and emission line subtraction.
    """
    
    def __init__(self, wave, flux, fit_wave, fit_flux, em_wave=None, em_flux_list=None, 
                 velocity_correction=0, error=None):
        """
        Initialize the line index calculator.
        
        Parameters
        ----------
        wave : array-like
            Original spectrum wavelength array
        flux : array-like
            Original spectrum flux array
        fit_wave : array-like
            Fitted spectrum wavelength array for continuum calculation
        fit_flux : array-like
            Fitted spectrum flux array for continuum calculation
        em_wave : array-like, optional
            Emission line wavelength array
        em_flux_list : array-like, optional
            Combined emission line spectrum
        velocity_correction : float, optional
            Velocity correction in km/s, default is 0
        error : array-like, optional
            Error array for the spectrum
        """
        self.c = 299792.458  # Speed of light in km/s
        self.velocity = velocity_correction
        
        # Apply velocity correction
        self.wave = self._apply_velocity_correction(wave)
        self.flux = flux.copy()  # Create a copy to avoid modifying original data
        self.fit_wave = fit_wave
        self.fit_flux = fit_flux
        self.error = error if error is not None else np.ones_like(flux)
        
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
        em_flux_resampled = spectres(self.wave, self.em_wave, self.em_flux_list)
        
        # Subtract emission lines from original spectrum
        self.flux -= em_flux_resampled
    
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
        
    def define_line_windows(self, line_name):
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
        Calculate absorption line index.
        
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
        # Get window definitions
        windows = self.define_line_windows(line_name)
        if windows is None:
            raise ValueError(f"Unknown absorption line: {line_name}")

        # Extract regions for calculations
        def get_fit_region(region):
            mask = (self.fit_wave >= windows[region][0]) & (self.fit_wave <= windows[region][1])
            return self.fit_wave[mask], self.fit_flux[mask]

        def get_orig_region(region):
            mask = (self.wave >= windows[region][0]) & (self.wave <= windows[region][1])
            return self.wave[mask], self.flux[mask], self.error[mask]

        # Get data from regions
        blue_wave_fit, blue_flux_fit = get_fit_region('blue')
        red_wave_fit, red_flux_fit = get_fit_region('red')
        line_wave, line_flux, line_err = get_orig_region('line')

        # Check if we have enough data points
        if len(blue_flux_fit) < 3 or len(line_flux) < 3 or len(red_flux_fit) < 3:
            return np.nan if not return_error else (np.nan, np.nan)

        # Calculate continuum
        blue_cont = self.calculate_pseudo_continuum(blue_wave_fit, blue_flux_fit)
        red_cont = self.calculate_pseudo_continuum(red_wave_fit, red_flux_fit)
        
        wave_cont = np.array([np.mean(blue_wave_fit), np.mean(red_wave_fit)])
        flux_cont = np.array([blue_cont, red_cont])
        
        # Linear interpolation for continuum
        f_interp = interpolate.interp1d(wave_cont, flux_cont)
        cont_at_line = f_interp(line_wave)

        # Calculate integral (use trapezoidal rule for integration)
        index = np.trapz((1.0 - line_flux/cont_at_line), line_wave)
        
        if return_error:
            # Calculate error
            error = np.sqrt(np.trapz((line_err/cont_at_line)**2, line_wave))
            return index, error
        
        return index

    def plot_line_fit(self, line_name, output_path=None):
        """
        Plot absorption line fitting result with original input data comparison.
        
        Parameters
        ----------
        line_name : str
            Absorption line name
        output_path : str, optional
            Path to save the plot
        """
        windows = self.define_line_windows(line_name)
        
        # Set x-axis range: extend window range on both sides by 20Å
        x_min = windows['blue'][0] - 20
        x_max = windows['red'][1] + 20
        
        # Create two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1])
        
        # Calculate y-axis range
        # Ensure using same wavelength grid data
        wave_mask = (self.wave >= x_min) & (self.wave <= x_max)
        em_mask = (self.em_wave >= x_min) & (self.em_wave <= x_max) if hasattr(self, 'em_wave') else None
        fit_mask = (self.fit_wave >= x_min) & (self.fit_wave <= x_max)
        
        # Calculate range within this wavelength range
        if hasattr(self, 'em_flux_list'):
            flux_range = self.flux[wave_mask] + self.em_flux_list[wave_mask]
        else:
            flux_range = self.flux[wave_mask]
            
        fit_range = self.fit_flux[fit_mask]
        
        y_min = min(np.min(flux_range), np.min(fit_range)) * 0.9
        y_max = max(np.max(flux_range), np.max(fit_range)) * 1.1
        
        # First panel: original input data
        if hasattr(self, 'em_flux_list'):
            ax1.plot(self.wave, self.flux + self.em_flux_list, 'k-', label='Original Spectrum', alpha=0.7)
            ax1.plot(self.em_wave, self.em_flux_list, 'r-', label='Emission Lines', alpha=0.7)
        else:
            ax1.plot(self.wave, self.flux, 'k-', label='Original Spectrum', alpha=0.7)
            
        ax1.plot(self.fit_wave, self.fit_flux, 'b-', label='Template Fit', alpha=0.7)
        
        # Mark regions
        colors = {'blue': 'b', 'line': 'g', 'red': 'r'}
        for region, (start, end) in windows.items():
            ax1.axvspan(start, end, alpha=0.2, color=colors[region])

        ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_xlabel('Rest-frame Wavelength (Å)')
        ax1.set_ylabel('Flux')
        ax1.set_title(f'Original Data Comparison (v={self.velocity:.1f} km/s)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # For second panel, calculate y-axis range
        processed_flux = self.flux[wave_mask]
        fit_flux_range = self.fit_flux[fit_mask]
        y_min_processed = min(np.min(processed_flux), np.min(fit_flux_range)) * 0.9
        y_max_processed = max(np.max(processed_flux), np.max(fit_flux_range)) * 1.1
        
        # Second panel: processed spectrum
        ax2.plot(self.wave, self.flux, 'k-', label='Processed Spectrum')
        ax2.plot(self.fit_wave, self.fit_flux, 'b--', label='Template Fit', alpha=0.7)
        
        # Mark regions
        for region, (start, end) in windows.items():
            ax2.axvspan(start, end, alpha=0.2, color=colors[region])
        
        # Calculate and plot continuum
        def get_fit_region(region):
            mask = (self.fit_wave >= windows[region][0]) & (self.fit_wave <= windows[region][1])
            return self.fit_wave[mask], self.fit_flux[mask]
        
        blue_wave_fit, blue_flux_fit = get_fit_region('blue')
        red_wave_fit, red_flux_fit = get_fit_region('red')
        blue_cont = self.calculate_pseudo_continuum(blue_wave_fit, blue_flux_fit)
        red_cont = self.calculate_pseudo_continuum(red_wave_fit, red_flux_fit)
        wave_cont = np.array([np.mean(blue_wave_fit), np.mean(red_wave_fit)])
        flux_cont = np.array([blue_cont, red_cont])
        ax2.plot(wave_cont, flux_cont, 'r*', markersize=10, label='Continuum Points')
        
        # Plot continuum line
        f_interp = interpolate.interp1d(wave_cont, flux_cont)
        line_mask = (self.wave >= windows['line'][0]) & (self.wave <= windows['line'][1])
        line_wave = self.wave[line_mask]
        cont_at_line = f_interp(line_wave)
        ax2.plot(line_wave, cont_at_line, 'r--', label='Continuum Fit')

        ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min_processed, y_max_processed)
        ax2.set_xlabel('Rest-frame Wavelength (Å)')
        ax2.set_ylabel('Flux')
        ax2.set_title('Processed Spectrum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
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
            
            # Close the FITS file
            hdu.close()
            
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
    # Set up paths
    ppxf_dir = Path(lib.__file__).parent
    sps_name = 'emiles'  # Use EMILES templates
    basename = f"spectra_{sps_name}_9.0.npz"
    filename = ppxf_dir / 'sps_models' / basename
    
    # Load stellar templates
    FWHM_gal = None  # Skip templates broadening
    sps = lib.sps_lib(filename, velscale, FWHM_gal, norm_range=config.lam_range_temp)
    
    # Reshape and normalize templates
    npix, *reg_dim = sps.templates.shape
    sps.templates = sps.templates.reshape(npix, -1)
    sps.templates /= np.median(sps.templates)  # Normalize by scalar
    
    # Prepare gas templates
    lam_range_gal = [np.exp(sps.ln_lam_temp[0]), np.exp(sps.ln_lam_temp[-1])]
    gas_templates, gas_names, line_wave = util.emission_lines(
        sps.ln_lam_temp, lam_range_gal, FWHM_gal)
    
    # Handle multiple gas components if needed
    if config.ngas_comp > 1:
        gas_templates = np.tile(gas_templates, config.ngas_comp)
        gas_names = np.asarray([f"{a}_({p+1})" for p in range(config.ngas_comp) for a in gas_names])
        line_wave = np.tile(line_wave, config.ngas_comp)
    
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
    dict
        Fitting results for this pixel
    """
    i, j, galaxy_data, sps, gas_templates, gas_names, line_wave, config = args
    
    # Get index in the flattened array
    k_index = i * galaxy_data.cube.shape[2] + j
    
    # Skip if this is a bad pixel
    if np.count_nonzero(galaxy_data.spectra[:, k_index]) < 50:
        return i, j, None
    
    try:
        # Get spectrum data
        spectrum = galaxy_data.spectra[:, k_index]
        noise = np.ones_like(spectrum)  # Use uniform noise
        
        # First fit: stars only
        mask = util.determine_mask(galaxy_data.ln_lam_gal, 
                                   np.exp(sps.ln_lam_temp[[0, -1]]), width=1000)
        
        pp_stars = ppxf(sps.templates, spectrum, noise, galaxy_data.velscale, 
                      [config.vel_s, config.vel_dis_s],
                      degree=config.degree,
                      plot=False, mask=mask, lam=galaxy_data.lam_gal, 
                      lam_temp=sps.lam_temp, quiet=True)
        
        # Get best-fit template
        best_template = sps.templates @ pp_stars.weights
        
        # Second fit: stars + gas
        stars_gas_templates = np.column_stack([best_template, gas_templates])
        component = [0] + [1] * len(config.gas_names)
        gas_component = np.array(component) > 0
        moments = config.moments  # e.g., [4, 2] for 4 moments in stars, 2 in gas
        
        # Setup tied parameters and starting values
        ncomp = len(moments)
        tied = [['', ''] for _ in range(ncomp)]
        
        # Use first fit results as starting point
        start = [pp_stars.sol[:2]]
        for _ in range(ncomp-1):
            start.append([pp_stars.sol[0], 50])  # Use stellar velocity for gas
            
        # Set reasonable bounds
        vlim = lambda x: pp_stars.sol[0] + x * np.array([-100, 100])
        bounds = [[vlim(2), [20, 300]]]
        for _ in range(ncomp-1):
            bounds.append([vlim(2), [20, 100]])
        
        # Run the full fit
        pp = ppxf(stars_gas_templates, spectrum, noise, galaxy_data.velscale, start,
                plot=False, moments=moments, degree=config.degree, mdegree=config.mdegree, 
                component=component, gas_component=gas_component, gas_names=gas_names,
                lam=galaxy_data.lam_gal, lam_temp=sps.lam_temp, tied=tied,
                bounds=bounds, global_search=True, quiet=True)
        
        # Calculate S/N
        residuals = spectrum - pp.bestfit
        rms = robust_sigma(residuals, zero=1)
        signal = np.median(spectrum[np.isfinite(spectrum)])
        snr = signal / rms if rms > 0 else 0
        
        # Extract emission line fluxes and A/N
        el_results = {}
        for name in config.gas_names:
            kk = gas_names == f"{name}_({1})"  # First gas component
            if np.any(kk):
                dlam = line_wave[kk] * galaxy_data.velscale / config.c
                flux = (pp.gas_flux[kk] * dlam)[0]
                
                # Get amplitude/noise ratio
                gas_bestfit = pp.gas_bestfit_templates[:, kk]
                if gas_bestfit.size > 0:
                    peak = np.max(gas_bestfit)
                    an = peak / rms if rms > 0 else 0
                else:
                    an = 0
                    
                el_results[name] = {'flux': flux, 'an': an}
        
        # Calculate spectral indices if requested
        indices = {}
        if config.compute_spectral_indices:
            # Process spectrum to remove emission lines
            # Prepare data for index calculation
            clean_spectrum = spectrum - pp.gas_bestfit
            
            # Create index calculator
            calculator = LineIndexCalculator(
                galaxy_data.lam_gal, clean_spectrum,
                sps.lam_temp, best_template,
                velocity_correction=pp.sol[0])
            
            # Calculate requested indices
            for index_name in config.line_indices:
                indices[index_name] = calculator.calculate_index(index_name)
        
        # Compose results
        results = {
            'success': True,
            'velocity': pp.sol[0],
            'sigma': pp.sol[1],
            'bestfit': pp.bestfit,
            'weights': pp.weights,
            'gas_bestfit': pp.gas_bestfit,
            'optimal_template': best_template,
            'rms': rms,
            'snr': snr,
            'el_results': el_results,
            'indices': indices,
            'apoly': pp.apoly,
            'pp_obj': pp
        }
        
        # Generate diagnostic plot if requested
        if config.make_plots and (i * j) % config.plot_every_n == 0:
            plot_pixel_fit(i, j, pp, galaxy_data, config)
        
        return i, j, results
    
    except Exception as e:
        logging.error(f"Error fitting pixel ({i},{j}): {str(e)}")
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
                galaxy_data.el_flux_maps[name][i, j] = data['flux']
                galaxy_data.el_snr_maps[name][i, j] = data['an']
            
            # Store spectral indices
            for name, value in result['indices'].items():
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
    # Create plot directory if it doesn't exist
    plot_dir = config.plot_dir / 'P2P_res'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get data
    k_index = i * galaxy_data.cube.shape[2] + j
    lam_gal = galaxy_data.lam_gal
    spectrum = galaxy_data.spectra[:, k_index]
    
    # Define spectral index windows for visualization
    index_windows = {
        'Hbeta': {'blue': (4827.875, 4847.875), 'line': (4847.875, 4876.625), 'red': (4876.625, 4891.625)},
        'Fe5015': {'blue': (4946.500, 4977.750), 'line': (4977.750, 5054.000), 'red': (5054.000, 5065.250)},
        'Mgb': {'blue': (5142.625, 5161.375), 'line': (5160.125, 5192.625), 'red': (5191.375, 5206.375)}
    }
    
    # Create figure with gridspec
    fig = plt.figure(figsize=(16, 12), dpi=config.dpi, tight_layout=False)
    
    # Top panel: Original data and fit
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.05, right=.95, bottom=0.65, top=0.95, hspace=0.0)
    ax1 = plt.subplot(gs1[0])
    
    # Middle panel: Residuals
    gs2 = gridspec.GridSpec(1, 1)
    gs2.update(left=0.05, right=.95, bottom=0.35, top=0.65, hspace=0.0)
    ax2 = plt.subplot(gs2[0])
    
    # Bottom panel: Emission components
    gs3 = gridspec.GridSpec(1, 1)
    gs3.update(left=0.05, right=.95, bottom=0.05, top=0.35, hspace=0.0)
    ax3 = plt.subplot(gs3[0])
    
    # Plot original spectrum and best fit
    ax1.plot(lam_gal, spectrum, c='tab:blue', lw=1, alpha=.9, 
            label=f"{config.galaxy_name}\npixel:[{i},{j}]")
    ax1.plot(lam_gal, pp.bestfit, '--', c='tab:red', alpha=.9)
    
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
    
    # Plot emission line components
    if hasattr(pp, 'gas_bestfit_templates') and pp.gas_bestfit_templates is not None:
        if pp.gas_bestfit_templates.shape[1] > 0:
            ax1.plot(lam_gal, pp.gas_bestfit_templates[:, 0], color='tab:orange', zorder=1, alpha=.9)
        
        if pp.gas_bestfit_templates.shape[1] > 1:
            ax1.plot(lam_gal, pp.gas_bestfit_templates[:, 1], color='tab:purple', zorder=1, alpha=.9)
    
    ax1.plot(lam_gal, pp.bestfit, '-', lw=.7, c='tab:red')
    
    # Set up the residuals plot
    ax2.plot(lam_gal, np.zeros(lam_gal.shape), '-', color='k', lw=.7, alpha=.9, zorder=0)
    
    # Calculate median and std of residuals
    residuals = spectrum - pp.bestfit
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
    stellar_bestfit = pp.bestfit - pp.gas_bestfit
    ax3.plot(lam_gal, spectrum - stellar_bestfit, '+', ms=2, mew=3, 
            color='tab:green', alpha=.9, zorder=2)
    
    # Plot emission components
    if hasattr(pp, 'gas_bestfit_templates') and pp.gas_bestfit_templates is not None:
        if pp.gas_bestfit_templates.shape[1] > 0:
            ax3.plot(lam_gal, pp.gas_bestfit_templates[:, 0], 
                   color='tab:orange', zorder=2, alpha=.9)
        
        if pp.gas_bestfit_templates.shape[1] > 1:
            ax3.plot(lam_gal, pp.gas_bestfit_templates[:, 1], 
                   color='tab:purple', zorder=2, alpha=.9)
        
        # Plot combined emission
        ax3.plot(lam_gal, pp.gas_bestfit, lw=.7, color='tab:red', zorder=2, alpha=.9)
    
    # Set up axis formatting
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(axis='both', which='both', labelsize='x-small', 
                      right=True, top=True, direction='in')
        ax.set_xlim(min(lam_gal), max(lam_gal))
    
    # Set y-axis limits
    ax1.set_ylim(0, max(spectrum) * 1.1)
    ax2.set_ylim(min(residuals) * 1.2, max(residuals) * 1.2)
    
    if hasattr(pp, 'gas_bestfit'):
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
    
    # Add title with kinematic info
    ax1.set_title(f"Velocity: {pp.sol[0]:.1f} km/s, σ: {pp.sol[1]:.1f} km/s, χ²: {pp.chi2:.3f}")
    
    # Save the plot
    plot_path = plot_dir / f"{config.galaxy_name}_pixel_{i}_{j}.pdf"
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


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


### ------------------------------------------------- ###
# Main Driver Functions
### ------------------------------------------------- ###

def run_p2p_analysis(config=None):
    """
    Run the full P2P analysis pipeline.
    
    Parameters
    ----------
    config : P2PConfig or str, optional
        Configuration object or path to configuration file
        
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
    
    # Set up logging
    log_path = config.output_dir / f"{config.galaxy_name}_p2p_log.txt"
    os.makedirs(config.output_dir, exist_ok=True)
    
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
    
    # Create output directories
    config.create_directories()
    
    # Log start of analysis
    logging.info(f"Starting P2P analysis for {config.galaxy_name}")
    logging.info(f"Data file: {config.get_data_path()}")
    start_time = time.time()
    
    try:
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        galaxy_data = IFUDataCube(config.get_data_path(), config.lam_range_temp, config.redshift, config)
        
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
    # Save kinematic maps
    hdu = fits.PrimaryHDU(galaxy_data.velfield)
    hdu.header['OBJECT'] = config.galaxy_name
    hdu.header['CONTENT'] = 'Stellar velocity field'
    hdu.header['BUNIT'] = 'km/s'
    hdu.header.update(galaxy_data.cube.header)
    hdu.writeto(config.output_dir / f"{config.galaxy_name}_velfield.fits", overwrite=True)
    
    hdu = fits.PrimaryHDU(galaxy_data.sigfield)
    hdu.header['OBJECT'] = config.galaxy_name
    hdu.header['CONTENT'] = 'Stellar velocity dispersion'
    hdu.header['BUNIT'] = 'km/s'
    hdu.header.update(galaxy_data.cube.header)
    hdu.writeto(config.output_dir / f"{config.galaxy_name}_sigfield.fits", overwrite=True)
    
    # Save emission line maps
    for name in config.gas_names:
        hdu = fits.PrimaryHDU(galaxy_data.el_flux_maps[name])
        hdu.header['OBJECT'] = config.galaxy_name
        hdu.header['CONTENT'] = f'{name} emission line flux'
        hdu.header['BUNIT'] = 'flux units'
        hdu.header.update(galaxy_data.cube.header)
        hdu.writeto(config.output_dir / f"{config.galaxy_name}_{name}_flux.fits", overwrite=True)
    
    # Save spectral index maps
    for name in config.line_indices:
        hdu = fits.PrimaryHDU(galaxy_data.index_maps[name])
        hdu.header['OBJECT'] = config.galaxy_name
        hdu.header['CONTENT'] = f'{name} spectral index'
        hdu.header['BUNIT'] = 'Angstrom'
        hdu.header.update(galaxy_data.cube.header)
        hdu.writeto(config.output_dir / f"{config.galaxy_name}_{name}_index.fits", overwrite=True)


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
    
    args = parser.parse_args()
    
    if args.create_config:
        config = P2PConfig()
        config.save(args.config_output)
        print(f"Default configuration saved to {args.config_output}")
    else:
        if args.config:
            run_p2p_analysis(args.config)
        else:
            run_p2p_analysis()