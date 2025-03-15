#!/usr/bin/env python
# coding: utf-8

"""
ISAP - IFU Spectral Analysis Pipeline v4.2.0
Integrated Stellar and Gas Spectral Fitting Program.
Supports pixel-by-pixel fitting, Voronoi binning, and radial binning.
"""

import os
import sys
import time
import logging
import argparse
import warnings
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Third-party libraries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits

from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
import ppxf.sps_util as miles

# Use the newer plotbin package instead of vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from plotbin.display_bins import display_bins

# Ignore specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

# Utility functions
def to_scalar(array):
    """
    Convert numpy array, list, or iterable to a scalar.
    Returns the original if not convertible.
    
    Parameters
    ----------
    array : array_like
        Input array or scalar
        
    Returns
    -------
    scalar or array
        Converted scalar or original array
    """
    try:
        if hasattr(array, '__iter__') and not isinstance(array, str):
            if len(array) == 1:
                return array[0]
            else:
                return array
        return array
    except (TypeError, ValueError):
        return array

def Apply_velocity_correction(Wavelength, z=0):
    """
    Apply redshift velocity correction to wavelength
    
    Parameters
    ----------
    Wavelength : float or array_like
        Input wavelength
    z : float, optional
        Redshift
        
    Returns
    -------
    float or array_like
        Corrected wavelength
    """
    return Wavelength * (1+z)


### ------------------------------------------------- ###
# Configuration Class
### ------------------------------------------------- ###

class P2PConfig:
    """
    Configuration class: Contains all parameters needed for ISAP analysis
    """
    
    def __init__(self, args=None):
        # Constants
        self.c = 299792.458  # Speed of light (km/s)
        
        # ---- Path settings ----
        self.base_dir = Path("E:/ProGram/Dr.Zheng/2024NAOC-IUS/Wkp/2024-NAOC-IUSpectrum")
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "output"
        self.plot_dir = self.base_dir / "plots"
        
        # ---- File naming settings ----
        self.galaxy_name = "galaxy"
        self.data_file = None
        self.output_prefix = None  # Optional prefix for output files
        
        # ---- Processing settings ----
        self.n_threads = max(1, os.cpu_count() // 2)
        self.parallel_mode = 'grouped'  # 'grouped' or 'global'
        self.batch_size = 50  # Binning batch size for grouped mode
        
        # ---- Plot settings ----
        self.make_plots = True
        self.no_plots = False
        self.plot_count = 0
        self.max_plots = 50
        self.dpi = 120
        self.LICplot = True  # Whether to create line index comparison plots
        
        # ---- Template settings ----
        self.template_file = None
        self.template_dir = self.base_dir / "templates"
        self.use_miles = True
        self.library_dir = None
        self.ssp_template = "MILES_BASTI_CH_baseFe_BaSTI_T00-10.0"
        self.z_metal = [0.0]
        self.t_age = [8.0, 9.0, 10.0]
        
        # ---- Wavelength settings ----
        self.lam_range_gal = None
        self.lam_range_temp = [3540, 7410]
        self.good_wavelength_range = [3700, 6800]
        
        # ---- Fitting settings ----
        self.compute_errors = False
        self.compute_emission_lines = True
        self.compute_spectral_indices = True
        self.use_two_stage_fit = True
        self.retry_with_degree_zero = True
        self.global_search = False
        self.fallback_to_simple_fit = True
        
        # ---- Kinematics settings ----
        self.vel_s = 0.0            # Initial stellar velocity
        self.vel_dis_s = 100.0      # Initial stellar velocity dispersion
        self.vel_gas = 0.0          # Initial gas velocity
        self.vel_dis_gas = 100.0    # Initial gas velocity dispersion
        self.redshift = 0.0         # Galaxy redshift
        self.helio_vel = 0.0        # Heliocentric velocity correction
        self.moments = [2, 2]       # pPXF moments parameter
        
        # ---- Continuum mode ----
        self.continuum_mode = "Cubic"
        
        # ---- Gas emission lines ----
        self.el_wave = None
        self.gas_names = ['OII3726', 'OII3729', 'Hgamma', 'OIII4363', 'HeII4686', 'Hbeta', 
                          'OIII5007', 'HeI5876', 'OI6300', 'Halpha', 'NII6583', 'SII6716', 'SII6731']
        self.line_indices = ['Fe5015', 'Fe5270', 'Fe5335', 'Mgb', 'Hbeta', 'Halpha']
        
        # ---- Progress bar settings ----
        self.progress_bar = True
        
        # ---- ISAP Integration ----
        self.use_isap = False
        self.isap_file = None
        
        # If arguments are provided, update config
        if args is not None:
            self.update_from_args(args)
    
    def update_from_args(self, args):
        """
        Update configuration from command line arguments
        
        Parameters
        ----------
        args : argparse.Namespace
            Command line arguments
        """
        # Basic parameters
        if hasattr(args, 'data_dir') and args.data_dir:
            self.data_dir = Path(args.data_dir)
        
        if hasattr(args, 'output_dir') and args.output_dir:
            self.output_dir = Path(args.output_dir)
        
        if hasattr(args, 'galaxy_name') and args.galaxy_name:
            self.galaxy_name = args.galaxy_name
        
        if hasattr(args, 'data_file') and args.data_file:
            self.data_file = args.data_file
        
        if hasattr(args, 'output_prefix') and args.output_prefix:
            self.output_prefix = args.output_prefix
            
        # Thread count
        if hasattr(args, 'threads') and args.threads:
            self.n_threads = args.threads
            
        # Plot settings
        if hasattr(args, 'no_plots'):
            self.no_plots = args.no_plots
            if self.no_plots:
                self.make_plots = False
                
        if hasattr(args, 'max_plots') and args.max_plots is not None:
            self.max_plots = args.max_plots
            
        if hasattr(args, 'dpi') and args.dpi:
            self.dpi = args.dpi
        
        # Parallel mode settings
        if hasattr(args, 'parallel_mode') and args.parallel_mode:
            self.parallel_mode = args.parallel_mode
            
        if hasattr(args, 'batch_size') and args.batch_size:
            self.batch_size = args.batch_size
            
        # Template settings
        if hasattr(args, 'template_dir') and args.template_dir:
            self.template_dir = Path(args.template_dir)
            
        if hasattr(args, 'use_miles') and args.use_miles is not None:
            self.use_miles = args.use_miles
            
        if hasattr(args, 'template_file') and args.template_file:
            self.template_file = args.template_file
            
        # Redshift and velocity settings
        if hasattr(args, 'redshift') and args.redshift is not None:
            self.redshift = args.redshift
            
        # Process other possible arguments
        if hasattr(args, 'compute_emission_lines') and args.compute_emission_lines is not None:
            self.compute_emission_lines = args.compute_emission_lines
            
        if hasattr(args, 'compute_spectral_indices') and args.compute_spectral_indices is not None:
            self.compute_spectral_indices = args.compute_spectral_indices
            
        if hasattr(args, 'global_search') and args.global_search is not None:
            self.global_search = args.global_search
            
        # ISAP settings
        if hasattr(args, 'isap') and args.isap:
            self.use_isap = True
            
        if hasattr(args, 'fits_file') and args.fits_file:
            self.isap_file = args.fits_file
            
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
    
    def get_data_path(self):
        """
        Get the data file path
        
        Returns
        -------
        Path
            Data file path
        """
        if self.data_file:
            # 如果明确指定了数据文件，就直接使用它
            data_path = self.data_dir / self.data_file
            if not data_path.exists():
                raise FileNotFoundError(f"Specified data file not found: {data_path}")
            return data_path
        else:
            # 默认搜索与galaxy_name相关的文件
            # 先尝试精确匹配
            exact_pattern = f"{self.galaxy_name}.fits"
            exact_files = list(self.data_dir.glob(exact_pattern))
            if exact_files:
                return exact_files[0]
                
            # 然后尝试部分匹配
            wildcard = f"{self.galaxy_name}*.fits"
            matching_files = list(self.data_dir.glob(wildcard))
            
            if not matching_files:
                # 尝试一种更灵活的搜索 - 任何FITS文件
                wildcard = "*.fits"
                matching_files = list(self.data_dir.glob(wildcard))
                
            if matching_files:
                logging.info(f"Found data file: {matching_files[0]}")
                return matching_files[0]
            else:
                raise FileNotFoundError(f"Could not find FITS file matching {self.galaxy_name}")
                
    def get_template_path(self):
        """
        Get the template file path
        
        Returns
        -------
        Path
            Template file path
        """
        if self.template_file:
            return self.template_dir / self.template_file
        else:
            # Search for possible template files
            wildcard = "*.fits"
            matching_files = list(self.template_dir.glob(wildcard))
            
            if matching_files:
                return matching_files[0]
            else:
                if self.use_miles:
                    return None  # Use internal MILES library
                else:
                    raise FileNotFoundError("Could not find template file")
                    
    def get_output_filename(self, suffix, mode=None):
        """
        Generate standardized output filename with appropriate prefixes
        
        Parameters
        ----------
        suffix : str
            Filename suffix/identifier
        mode : str, optional
            Analysis mode (P2P, VNB, RDB)
            
        Returns
        -------
        Path
            Complete output file path
        """
        # Start with galaxy name
        name_parts = [self.galaxy_name]
        
        # Add custom prefix if defined
        if self.output_prefix:
            name_parts.insert(0, self.output_prefix)
            
        # Add mode indicator
        if mode is not None and mode in ["P2P", "VNB", "RDB"]:
            name_parts.append(mode)
            
        # Add suffix
        name_parts.append(suffix)
        
        # Join with underscores
        filename = "_".join(name_parts) + ".fits"
        
        return self.output_dir / filename
    
    def get_p2p_output_path(self, suffix):
        """
        Get path to a standard P2P output file. Automatically checks existence.
        
        Parameters
        ----------
        suffix : str
            File suffix/identifier (e.g., 'velfield')
            
        Returns
        -------
        tuple
            (Path to file, bool indicating if file exists)
        """
        path = self.get_output_filename(suffix, "P2P")
        exists = path.exists()
        return path, exists
                    
    def __str__(self):
        """
        Output configuration information
        
        Returns
        -------
        str
            Configuration information string
        """
        info = [
            f"Galaxy: {self.galaxy_name} (z={self.redshift})",
            f"Data Directory: {self.data_dir}",
            f"Output Directory: {self.output_dir}",
            f"Template: {'MILES (internal)' if self.use_miles else self.template_file}",
            f"Parallel Mode: {self.parallel_mode}",
            f"Threads: {self.n_threads}",
            f"Compute Emission Lines: {self.compute_emission_lines}",
            f"Compute Spectral Indices: {self.compute_spectral_indices}",
            f"Use Two-Stage Fit: {self.use_two_stage_fit}"
        ]
        return "\n".join(info)


### ------------------------------------------------- ###
# Data Loading Class
### ------------------------------------------------- ###

class IFUDataCube:
    """
    IFU data cube processing class.
    
    This class handles reading and preliminary processing of FITS format data cubes.
    """
    
    def __init__(self, filename, lam_range_temp, redshift, config):
        """
        Initialize IFU data cube.
        
        Parameters
        ----------
        filename : str or Path
            FITS file path
        lam_range_temp : list
            Template wavelength range
        redshift : float
            Galaxy redshift
        config : P2PConfig
            Configuration object
        """
        self.filename = str(filename)
        self.lam_range_temp = lam_range_temp
        self.redshift = redshift
        self.config = config
        
        # Initialize attributes
        self.cube = None
        self.header = None
        self.lam_gal = None
        self.spectra = None
        self.variance = None
        self.signal = None
        self.noise = None
        self.x = None
        self.y = None
        self.row = None
        self.col = None
        self.velscale = None
        self.pixsize_x = None  # Pixel size (X direction)
        self.pixsize_y = None  # Pixel size (Y direction)
        self.CD1_1 = 1.0
        self.CD1_2 = 0.0
        self.CD2_1 = 0.0
        self.CD2_2 = 1.0
        self.CRVAL1 = 0.0
        self.CRVAL2 = 0.0
        
        # Initialize result maps
        self.velfield = None
        self.sigfield = None
        self.bin_map = None
        self.el_flux_maps = {}
        self.el_snr_maps = {}
        self.index_maps = {}
        
        for name in config.gas_names:
            self.el_flux_maps[name] = None
            self.el_snr_maps[name] = None
            
        for name in config.line_indices:
            self.index_maps[name] = None
        
        # Read and process data
        self._read_data()
        self._preprocess_data()
    
    def _read_data(self):
        """
        Read FITS file data.
        """
        try:
            logging.info(f"Reading data file: {self.filename}")
            with fits.open(self.filename) as hdul:
                # Read primary HDU
                primary_hdu = hdul[0]
                self.header = primary_hdu.header
                
                # 定义数据范围
                cut_low = 1
                cut_high = 1
                
                # 处理FITS数据
                if primary_hdu.data is not None and len(primary_hdu.data.shape) == 3:
                    # 直接是数据立方体
                    # 与原始代码一致的处理方式
                    self.cube = primary_hdu.data[cut_low:-cut_high, :, :] * (10**18)
                elif len(hdul) > 1:
                    # 数据在扩展HDU中
                    for hdu in hdul[1:]:
                        if hdu.data is not None and len(hdu.data.shape) == 3:
                            # 与原始代码一致的处理方式
                            self.cube = hdu.data[cut_low:-cut_high, :, :] * (10**18)
                            # 合并头信息
                            for key in hdu.header:
                                if key not in self.header and key not in ('XTENSION', 'BITPIX', 'NAXIS', 'PCOUNT', 'GCOUNT'):
                                    self.header[key] = hdu.header[key]
                            break
                
                # 默认使用均匀方差
                if self.cube is not None:
                    self.variance = np.ones_like(self.cube)
                
                # 从头信息获取波长
                if self.cube is not None and 'CRVAL3' in self.header and 'CD3_3' in self.header:
                    head = self.header
                    wave = head['CRVAL3'] + head['CD3_3'] * np.arange(self.cube.shape[0]) + head['CD3_3'] * cut_low
                    self.lam_gal = wave
                
                # 如果没有找到数据立方体
                if self.cube is None:
                    raise ValueError("Could not find data cube in FITS file")
                
                # 从头信息提取基本信息，与原始代码一致
                if 'CD1_1' in self.header: self.CD1_1 = self.header['CD1_1']
                if 'CD1_2' in self.header: self.CD1_2 = self.header['CD1_2']
                if 'CD2_1' in self.header: self.CD2_1 = self.header['CD2_1']
                if 'CD2_2' in self.header: self.CD2_2 = self.header['CD2_2']
                if 'CRVAL1' in self.header: self.CRVAL1 = self.header['CRVAL1']
                if 'CRVAL2' in self.header: self.CRVAL2 = self.header['CRVAL2']
                
                # 提取像素大小 - 与原始代码一致
                self.pixsize_x = abs(np.sqrt((self.header['CD1_1'])**2 + (self.header['CD2_1'])**2)) * 3600
                self.pixsize_y = abs(np.sqrt((self.header['CD1_2'])**2 + (self.header['CD2_2'])**2)) * 3600
                
                logging.info(f"Data cube shape: {self.cube.shape}")
                
        except Exception as e:
            logging.error(f"Error reading FITS file: {str(e)}")
            if "Could not find data cube in FITS file" in str(e):
                logging.error("File structure does not match expected format. Please check the FITS file structure.")
            raise
    
    def _preprocess_data(self):
        """
        Preprocess data cube, extract spectra and calculate SNR.
        """
        try:
            # Get data dimensions
            nw, ny, nx = self.cube.shape
            
            # Organize spectral data
            self.spectra = np.reshape(self.cube, (nw, ny * nx))
            
            # Organize variance data
            if self.variance is not None:
                self.variance = np.reshape(self.variance, (nw, ny * nx))
            
            # Create coordinate arrays
            y_coords, x_coords = np.indices((ny, nx))
            self.x = (x_coords + 1).flatten()  # 1-indexed
            self.y = (y_coords + 1).flatten()
            self.row = self.y  # Row index equals y
            self.col = self.x  # Column index equals x
            
            # Calculate signal-to-noise ratio
            self._calculate_snr()
            
            logging.info(f"Wavelength range: {self.lam_gal[0]:.1f} - {self.lam_gal[-1]:.1f} Å")
            logging.info(f"Total pixels: {ny * nx}")
            
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def _calculate_snr(self):
        """
        Calculate signal-to-noise ratio for each pixel.
        """
        try:
            nw, npix = self.spectra.shape
            
            # Wavelength range restriction
            w1, w2 = self.lam_range_temp
            mask = (self.lam_gal > w1) & (self.lam_gal < w2)
            
            # Initialize signal and noise arrays
            self.signal = np.zeros(npix)
            self.noise = np.zeros(npix)
            
            for i in range(npix):
                # Get central region of spectrum
                spectrum = self.spectra[mask, i]
                
                # If variance available, use it directly
                if self.variance is not None:
                    pixel_var = self.variance[mask, i]
                    pixel_noise = np.sqrt(np.median(pixel_var[pixel_var > 0]))
                else:
                    # Otherwise estimate from spectrum
                    pixel_noise = robust_sigma(spectrum)
                
                # Calculate signal (median of spectrum)
                pixel_signal = np.median(spectrum)
                
                # Save results
                self.signal[i] = pixel_signal
                self.noise[i] = pixel_noise if pixel_noise > 0 else 1.0
                
        except Exception as e:
            logging.error(f"Error calculating SNR: {str(e)}")
            # Create default values
            npix = self.spectra.shape[1]
            self.signal = np.ones(npix)
            self.noise = np.ones(npix)


### ------------------------------------------------- ###
# ISAP Data Integration Functions
### ------------------------------------------------- ###

def read_isap_data(file_path, data_type='velocity'):
    """
    Read data from ISAP output FITS files.
    
    Parameters
    ----------
    file_path : str or Path
        ISAP output FITS file path
    data_type : str
        Data type, can be 'velocity', 'sigma', 'flux', etc.
        
    Returns
    -------
    tuple
        (data_array, header)
    """
    try:
        with fits.open(file_path) as hdul:
            # Try to get data from primary HDU
            if len(hdul[0].data.shape) == 2:  # 2D data
                data = hdul[0].data
                header = hdul[0].header
            elif len(hdul) > 1:
                # Look for extension that contains requested data type
                found = False
                for i, hdu in enumerate(hdul):
                    hdu_name = hdu.name.upper()
                    # Match data type
                    if (data_type.upper() in hdu_name or 
                        (data_type == 'velocity' and 'VEL' in hdu_name) or
                        (data_type == 'sigma' and ('SIGMA' in hdu_name or 'DISP' in hdu_name)) or
                        (data_type == 'flux' and 'FLUX' in hdu_name)):
                        if len(hdu.data.shape) == 2:  # Ensure 2D data
                            data = hdu.data
                            header = hdu.header
                            found = True
                            logging.info(f"Found {data_type} data in extension {i}: {hdu_name}")
                            break
                
                if not found:
                    # No specific name found, use first valid extension
                    for i, hdu in enumerate(hdul):
                        if hdu.data is not None and len(hdu.data.shape) == 2:
                            data = hdu.data
                            header = hdu.header
                            logging.warning(f"No extension matching {data_type} found, using extension {i}")
                            found = True
                            break
                    
                    if not found:
                        raise ValueError(f"No valid 2D {data_type} data found in file")
            else:
                raise ValueError("Unsupported FITS file structure")
                
        return data, header
    
    except Exception as e:
        logging.error(f"Error reading ISAP file: {str(e)}")
        raise

def extract_pixel_velocity(config, x, y, fits_file=None, isap_mode=False):
    """
    Extract velocity parameters for a single pixel from fitting results.
    Supports P2P, VNB, RDB and ISAP formats.
    
    Parameters
    ----------
    config : P2PConfig
        Configuration object
    x : int
        Pixel X coordinate (0-indexed)
    y : int
        Pixel Y coordinate (0-indexed)
    fits_file : str, optional
        FITS file path, if None use default path
    isap_mode : bool, optional
        Whether to read data from ISAP file
        
    Returns
    -------
    dict
        Dictionary containing pixel velocity parameters
    """
    result = {
        'coordinates': (x, y),
        'velocity': None,
        'sigma': None,
        'redshift': 0.0,
        'object': 'Unknown'
    }
    
    try:
        if isap_mode:
            # Read from ISAP file
            if fits_file is None:
                raise ValueError("ISAP mode requires specified FITS file path")
            
            # Read velocity field
            vel_data, vel_hdr = read_isap_data(fits_file, 'velocity')
            
            # Try to read dispersion field (may be in same file or different extension)
            try:
                # First try to read from the same file
                sig_data, _ = read_isap_data(fits_file, 'sigma')
            except:
                # Try to read from related file
                sig_file = Path(fits_file).parent / Path(fits_file).name.replace("velocity", "sigma").replace("vel", "sig")
                if sig_file.exists():
                    sig_data, _ = read_isap_data(sig_file, 'sigma')
                else:
                    sig_data = None
                    logging.warning(f"Could not find corresponding dispersion data file: {sig_file}")
            
            # Read header information
            result['redshift'] = vel_hdr.get('REDSHIFT', 0.0)
            result['object'] = vel_hdr.get('OBJECT', 'Unknown')
            result['analysis_type'] = 'ISAP'
            
            # Read other possible header information
            for key in ['INSTRUME', 'DATE-OBS', 'EXPTIME']:
                if key in vel_hdr:
                    result[key.lower()] = vel_hdr[key]
            
        else:
            # Read from P2P/VNB/RDB file
            # Determine FITS file path
            if fits_file is None:
                vel_file = config.get_output_filename("velfield", "P2P")
                sig_file = config.get_output_filename("sigfield", "P2P")
            else:
                # If specific file provided, use it and infer corresponding sig file
                vel_file = Path(fits_file)
                sig_file = vel_file.parent / vel_file.name.replace("velfield", "sigfield")
            
            # Check if files exist
            if not vel_file.exists():
                raise FileNotFoundError(f"Velocity field file does not exist: {vel_file}")
            
            # Read FITS files
            vel_data = fits.getdata(vel_file)
            vel_hdr = fits.getheader(vel_file)
            
            if sig_file.exists():
                sig_data = fits.getdata(sig_file)
            else:
                sig_data = None
                logging.warning(f"Velocity dispersion file does not exist: {sig_file}, returning velocity only")
            
            # Read header information
            result['redshift'] = vel_hdr.get('REDSHIFT', 0.0)
            result['object'] = vel_hdr.get('OBJECT', 'Unknown')
            
            # Determine analysis type
            if 'BINTYPE' in vel_hdr:
                bin_type = vel_hdr['BINTYPE']
                result['analysis_type'] = bin_type
                
                if bin_type == 'VNB':
                    # If Voronoi binning, read bin ID
                    try:
                        bin_map = fits.getdata(config.get_output_filename("binmap", "VNB"))
                        result['bin_id'] = bin_map[y, x]
                    except:
                        result['bin_id'] = -1
                elif bin_type == 'RDB':
                    # If radial binning, read radial distance
                    try:
                        rmap = fits.getdata(config.get_output_filename("radiusmap", "RDB"))
                        result['radius'] = rmap[y, x]
                    except:
                        pass
                    
                    # Find which ring it belongs to
                    try:
                        bin_map = fits.getdata(config.get_output_filename("binmap", "RDB"))
                        result['ring_id'] = bin_map[y, x]
                    except:
                        result['ring_id'] = -1
            else:
                result['analysis_type'] = "P2P"
                
            # Try to read emission line data
            emission_lines = {}
            for name in config.gas_names:
                try:
                    if result['analysis_type'] == "P2P":
                        flux_file = config.get_output_filename(f"{name}_flux", "P2P")
                    else:
                        flux_file = config.get_output_filename(f"{name}_flux", result['analysis_type'])
                    
                    if flux_file.exists():
                        flux_data = fits.getdata(flux_file)
                        emission_lines[name] = flux_data[y, x]
                except:
                    pass
            
            if emission_lines:
                result['emission_lines'] = emission_lines
                
            # Try to read spectral index data
            indices = {}
            for name in config.line_indices:
                try:
                    if result['analysis_type'] == "P2P":
                        index_file = config.get_output_filename(f"{name}_index", "P2P")
                    else:
                        index_file = config.get_output_filename(f"{name}_index", result['analysis_type'])
                        
                    if index_file.exists():
                        index_data = fits.getdata(index_file)
                        indices[name] = index_data[y, x]
                except:
                    pass
            
            if indices:
                result['spectral_indices'] = indices
        
        # Check if coordinates are within data range
        if y < 0 or y >= vel_data.shape[0] or x < 0 or x >= vel_data.shape[1]:
            raise ValueError(f"Coordinates ({x}, {y}) out of data range {vel_data.shape}")
        
        # Extract velocity and dispersion values
        result['velocity'] = vel_data[y, x]
        if sig_data is not None:
            result['sigma'] = sig_data[y, x]
        
        # If ISAP mode, try to read other possible data
        if isap_mode:
            # Try to read emission line data
            emission_lines = {}
            isap_dir = Path(fits_file).parent
            
            # Common emission line names
            common_lines = ['Halpha', 'Hbeta', 'OIII5007', 'NII6583', 'SII6716']
            
            for line in common_lines:
                try:
                    # Try various possible filename patterns
                    patterns = [
                        f"*{line}*flux*.fits",
                        f"*{line.lower()}*flux*.fits",
                        f"*flux*{line}*.fits",
                        f"*{line}*.fits"
                    ]
                    
                    found = False
                    for pattern in patterns:
                        matches = list(isap_dir.glob(pattern))
                        if matches:
                            line_file = matches[0]
                            line_data, _ = read_isap_data(line_file, 'flux')
                            emission_lines[line] = line_data[y, x]
                            found = True
                            break
                    
                    if not found and line in ['Halpha', 'Hbeta', 'OIII5007']:
                        logging.warning(f"Could not find {line} emission line data file")
                        
                except Exception as e:
                    logging.debug(f"Error reading {line} emission line data: {str(e)}")
            
            if emission_lines:
                result['emission_lines'] = emission_lines
        
        return result
        
    except Exception as e:
        logging.error(f"Error extracting pixel velocity parameters: {str(e)}")
        result['error'] = str(e)
        return result


### ------------------------------------------------- ###
# Template Preparation Functions
### ------------------------------------------------- ###

def prepare_templates(config, velscale):
    """
    Prepare stellar and gas templates.
    
    Parameters
    ----------
    config : P2PConfig
        Configuration object
    velscale : float
        Velocity scale (km/s/pixel)
        
    Returns
    -------
    tuple
        (sps, gas_templates, gas_names, line_wave)
    """
    # 1. Prepare stellar templates
    if config.use_miles:
        sps = prepare_miles_templates(config, velscale)
    else:
        sps = prepare_custom_templates(config, velscale)
    
    # 2. Prepare gas templates
    if config.compute_emission_lines:
        gas_templates, gas_names, line_wave = prepare_gas_templates(config, sps)
    else:
        gas_templates = np.zeros((sps.templates.shape[0], 0))
        gas_names = []
        line_wave = []
    
    return sps, gas_templates, gas_names, line_wave

def prepare_miles_templates(config, velscale):
    """
    Prepare MILES template library.
    
    Parameters
    ----------
    config : P2PConfig
        Configuration object
    velscale : float
        Velocity scale (km/s/pixel)
        
    Returns
    -------
    object
        Object with templates, ln_lam_temp and lam_temp attributes
    """
    logging.info("Using MILES stellar template library...")
    
    # Select MILES model
    miles_model = config.ssp_template
    
    # Metallicity and age grid
    z_metal = config.z_metal
    t_age = config.t_age
    
    # Create MILES instance
    try:
        miles_sps = miles.MilesSsp(miles_model, velscale, z_metal, t_age)
        logging.info(f"MILES template shape: {miles_sps.templates.shape}")
        return miles_sps
    except Exception as e:
        logging.error(f"Could not create MILES templates: {str(e)}")
        raise

def prepare_custom_templates(config, velscale):
    """
    Prepare custom template library.
    
    Parameters
    ----------
    config : P2PConfig
        Configuration object
    velscale : float
        Velocity scale (km/s/pixel)
        
    Returns
    -------
    object
        Object with templates, ln_lam_temp and lam_temp attributes
    """
    logging.info("Using custom stellar templates...")
    
    try:
        # Get template file path
        template_path = config.get_template_path()
        
        # Read template file
        with fits.open(template_path) as hdul:
            template_data = hdul[0].data
            template_header = hdul[0].header
        
        # Extract wavelength information
        nw = template_data.shape[0]
        if 'CRVAL1' in template_header and 'CDELT1' in template_header and 'CRPIX1' in template_header:
            crval1 = template_header['CRVAL1']
            cdelt1 = template_header['CDELT1']
            crpix1 = template_header['CRPIX1']
            lam_temp = np.arange(nw) * cdelt1 + (crval1 - crpix1 * cdelt1)
        else:
            # Use default wavelength range
            lam_temp = np.linspace(3500, 7500, nw)
            logging.warning("Wavelength information not found in template header, using default values")
        
        # Resample templates to constant log spacing (as required by PPXF)
        ln_lam_temp = np.log(lam_temp)
        new_ln_lam_temp = np.arange(ln_lam_temp[0], ln_lam_temp[-1], velscale/config.c)
        templates = util.log_rebin(lam_temp, template_data, velscale=velscale)[0]
        
        # Create object similar to MILES
        class CustomSps:
            pass
        
        custom_sps = CustomSps()
        custom_sps.templates = templates
        custom_sps.ln_lam_temp = new_ln_lam_temp
        custom_sps.lam_temp = np.exp(new_ln_lam_temp)
        
        logging.info(f"Custom template shape: {templates.shape}")
        return custom_sps
        
    except Exception as e:
        logging.error(f"Could not create custom templates: {str(e)}")
        raise

def prepare_gas_templates(config, sps):
    """
    Prepare gas emission line templates.
    
    Parameters
    ----------
    config : P2PConfig
        Configuration object
    sps : object
        Stellar template object, containing ln_lam_temp and lam_temp
        
    Returns
    -------
    tuple
        (gas_templates, gas_names, line_wave)
    """
    logging.info("Preparing gas emission line templates...")
    
    # Prepare gas emission lines
    try:
        # Emission line list
        line_names = config.gas_names
        
        # If custom emission line wavelengths specified
        if config.el_wave is not None:
            line_wave = config.el_wave
        else:
            # Use default emission line wavelengths
            line_wave = {
                'OII3726': 3726.03,
                'OII3729': 3728.82,
                'Hgamma': 4340.47,
                'OIII4363': 4363.21,
                'HeII4686': 4685.7,
                'Hbeta': 4861.33,
                'OIII4959': 4958.92,
                'OIII5007': 5006.84,
                'HeI5876': 5875.67,
                'OI6300': 6300.30,
                'Halpha': 6562.80,
                'NII6548': 6548.03,
                'NII6583': 6583.41,
                'SII6716': 6716.47,
                'SII6731': 6730.85
            }
            
            # Convert to array
            line_wave = np.array([line_wave[name] for name in line_names if name in line_wave])
            
            # If some names not in default dictionary, warn
            missing_lines = [name for name in line_names if name not in line_wave]
            if missing_lines:
                logging.warning(f"Unknown emission lines: {missing_lines}")
        
        # Use ppxf_util's emission_lines function to create gas templates
        gas_templates, gas_names, line_wave = util.emission_lines(
            sps.ln_lam_temp, line_wave, FWHM=config.vel_dis_gas)
        
        logging.info(f"Gas template shape: {gas_templates.shape}, Number of emission lines: {len(gas_names)}")
        return gas_templates, gas_names, line_wave
    
    except Exception as e:
        logging.error(f"Could not create gas templates: {str(e)}")
        # Return empty templates
        gas_templates = np.zeros((sps.templates.shape[0], 0))
        return gas_templates, [], []


### ------------------------------------------------- ###
# Line Index Calculator
### ------------------------------------------------- ###

class LineIndexCalculator:
    """
    Line index calculator.
    
    Calculates spectral indices for common spectral features like Lick indices.
    """
    
    def __init__(self, wavelength, flux, template_wave=None, template_flux=None, 
                 em_wave=None, em_flux_list=None, velocity_correction=0,
                 continuum_mode="Cubic"):
        """
        Initialize line index calculator.
        
        Parameters
        ----------
        wavelength : array
            Observed wavelength
        flux : array
            Observed flux
        template_wave : array, optional
            Template wavelength
        template_flux : array, optional
            Template flux
        em_wave : array, optional
            Emission line wavelength
        em_flux_list : array, optional
            Emission line flux
        velocity_correction : float, optional
            Velocity correction (applied to wavelengths defining bandpasses)
        continuum_mode : str, optional
            Continuum calculation mode, "Cubic" (default) or "Linear"
        """
        self.wavelength = wavelength
        self.flux = flux
        self.template_wave = template_wave
        self.template_flux = template_flux
        self.velocity_correction = velocity_correction
        self.continuum_mode = continuum_mode
        
        # Emission line information
        self.em_wave = em_wave
        self.em_flux = em_flux_list
        
        # Define spectral indices
        self.define_indices()
    
    def define_indices(self):
        """
        Define wavelength ranges for common spectral indices.
        """
        # Define some common spectral index definitions (blue, band, red continuum ranges)
        self.indices = {
            # Lick indices
            'Hbeta': {'blue': (4827.875, 4847.875), 'band': (4847.875, 4876.625), 'red': (4876.625, 4891.625)},
            'Mgb': {'blue': (5142.625, 5161.375), 'band': (5160.125, 5192.625), 'red': (5191.375, 5206.375)},
            'Fe5015': {'blue': (4946.500, 4977.750), 'band': (4977.750, 5054.000), 'red': (5054.000, 5065.250)},
            'Fe5270': {'blue': (5233.150, 5248.150), 'band': (5245.650, 5285.650), 'red': (5285.650, 5318.150)},
            'Fe5335': {'blue': (5304.625, 5315.875), 'band': (5312.125, 5352.125), 'red': (5353.375, 5363.375)},
            
            # Other common indices
            'Halpha': {'blue': (6510.0, 6540.0), 'band': (6554.0, 6568.0), 'red': (6575.0, 6585.0)},
            'Na D': {'blue': (5860.625, 5875.625), 'band': (5876.875, 5909.375), 'red': (5922.125, 5948.125)},
            'TiO 1': {'blue': (5936.625, 5994.125), 'band': (5937.875, 5994.875), 'red': (6038.625, 6103.625)},
            'TiO 2': {'blue': (6066.625, 6141.625), 'band': (6189.625, 6272.125), 'red': (6372.625, 6415.125)},
            'Ca H&K': {'blue': (3806.5, 3833.8), 'band': (3899.5, 4003.5), 'red': (4019.8, 4051.2)}
        }
    
    def calculate_index(self, index_name):
        """
        Calculate value for specified spectral index.
        
        Parameters
        ----------
        index_name : str
            Spectral index name
            
        Returns
        -------
        float
            Spectral index value
        """
        if index_name not in self.indices:
            raise ValueError(f"Unknown spectral index: {index_name}")
        
        # Get index definition
        index_def = self.indices[index_name]
        
        # Apply velocity correction
        vel_corr_factor = 1 + self.velocity_correction / 299792.458
        blue_range = (index_def['blue'][0] * vel_corr_factor, index_def['blue'][1] * vel_corr_factor)
        band_range = (index_def['band'][0] * vel_corr_factor, index_def['band'][1] * vel_corr_factor)
        red_range = (index_def['red'][0] * vel_corr_factor, index_def['red'][1] * vel_corr_factor)
        
        # Calculate continuum
        if self.continuum_mode.lower() == "cubic":
            continuum = self._calculate_cubic_continuum(blue_range, band_range, red_range)
        else:  # linear
            continuum = self._calculate_linear_continuum(blue_range, band_range, red_range)
        
        # Calculate bandpass region flux and continuum
        band_mask = (self.wavelength >= band_range[0]) & (self.wavelength <= band_range[1])
        
        if not np.any(band_mask):
            logging.warning(f"No data points in bandpass region {band_range}")
            return np.nan
        
        band_flux = self.flux[band_mask]
        band_wave = self.wavelength[band_mask]
        band_continuum = continuum[band_mask]
        
        # Calculate index value
        if index_name in ['Fe5015', 'Fe5270', 'Fe5335', 'Mgb', 'Na D', 'TiO 1', 'TiO 2']:
            # Equivalent width indices (EW)
            dwave = np.abs(np.diff(np.append(band_wave, 2*band_wave[-1]-band_wave[-2])))
            index_value = np.sum((1 - band_flux / band_continuum) * dwave)
        else:
            # Magnitude indices (like Hbeta)
            index_value = -2.5 * np.log10(np.mean(band_flux) / np.mean(band_continuum))
        
        return index_value
    
    def _calculate_linear_continuum(self, blue_range, band_range, red_range):
        """
        Calculate linear continuum.
        
        Parameters
        ----------
        blue_range : tuple
            Blue continuum range
        band_range : tuple
            Bandpass filter range
        red_range : tuple
            Red continuum range
            
        Returns
        -------
        array
            Continuum estimate over full wavelength range
        """
        # Blue continuum
        blue_mask = (self.wavelength >= blue_range[0]) & (self.wavelength <= blue_range[1])
        if not np.any(blue_mask):
            logging.warning(f"No data points in blue continuum region {blue_range}")
            return np.ones_like(self.wavelength) * np.nanmean(self.flux)
            
        blue_flux = np.nanmean(self.flux[blue_mask])
        blue_wave = np.nanmean(self.wavelength[blue_mask])
        
        # Red continuum
        red_mask = (self.wavelength >= red_range[0]) & (self.wavelength <= red_range[1])
        if not np.any(red_mask):
            logging.warning(f"No data points in red continuum region {red_range}")
            return np.ones_like(self.wavelength) * np.nanmean(self.flux)
            
        red_flux = np.nanmean(self.flux[red_mask])
        red_wave = np.nanmean(self.wavelength[red_mask])
        
        # Linear interpolation
        slope = (red_flux - blue_flux) / (red_wave - blue_wave)
        continuum = blue_flux + slope * (self.wavelength - blue_wave)
        
        return continuum
    
    def _calculate_cubic_continuum(self, blue_range, band_range, red_range):
        """
        Calculate cubic spline continuum.
        
        Parameters
        ----------
        blue_range : tuple
            Blue continuum range
        band_range : tuple
            Bandpass filter range
        red_range : tuple
            Red continuum range
            
        Returns
        -------
        array
            Continuum estimate over full wavelength range
        """
        # Blue continuum region
        blue_mask = (self.wavelength >= blue_range[0]) & (self.wavelength <= blue_range[1])
        
        # Red continuum region
        red_mask = (self.wavelength >= red_range[0]) & (self.wavelength <= red_range[1])
        
        # Check if there are enough data points
        if not np.any(blue_mask) or not np.any(red_mask):
            logging.warning(f"Not enough data points in continuum regions: blue={np.sum(blue_mask)}, red={np.sum(red_mask)}")
            # Fall back to linear continuum
            return self._calculate_linear_continuum(blue_range, band_range, red_range)
        
        # Combine blue and red continuum regions
        cont_mask = np.logical_or(blue_mask, red_mask)
        
        # Continuum wavelength and flux
        cont_wave = self.wavelength[cont_mask]
        cont_flux = self.flux[cont_mask]
        
        # Use cubic spline interpolation
        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(cont_wave, cont_flux)
            continuum = cs(self.wavelength)
        except:
            # Fall back to scipy.interpolate
            try:
                from scipy.interpolate import interp1d
                f = interp1d(cont_wave, cont_flux, kind='cubic', bounds_error=False,
                            fill_value=(cont_flux[0], cont_flux[-1]))
                continuum = f(self.wavelength)
            except:
                # Final fallback to linear interpolation
                logging.warning("Cubic interpolation failed, falling back to linear continuum")
                continuum = self._calculate_linear_continuum(blue_range, band_range, red_range)
        
        return continuum
    
    def plot_all_lines(self, mode='P2P', number=None):
        """
        Plot all defined spectral indices.
        
        Parameters
        ----------
        mode : str, optional
            Mode ("P2P", "VNB", or "RDB")
        number : int, optional
            Pixel or bin number
            
        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Create figure
        n_indices = len(self.indices)
        n_cols = min(3, n_indices)
        n_rows = (n_indices + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(4*n_cols, 3*n_rows), dpi=100)
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        # Plot each index
        for i, (name, index_def) in enumerate(self.indices.items()):
            # Calculate row and column position
            row = i // n_cols
            col = i % n_cols
            
            # Create subplot
            ax = fig.add_subplot(gs[row, col])
            
            # Apply velocity correction
            vel_corr_factor = 1 + self.velocity_correction / 299792.458
            blue_range = (index_def['blue'][0] * vel_corr_factor, index_def['blue'][1] * vel_corr_factor)
            band_range = (index_def['band'][0] * vel_corr_factor, index_def['band'][1] * vel_corr_factor)
            red_range = (index_def['red'][0] * vel_corr_factor, index_def['red'][1] * vel_corr_factor)
            
            # Calculate continuum
            if self.continuum_mode.lower() == "cubic":
                continuum = self._calculate_cubic_continuum(blue_range, band_range, red_range)
            else:  # linear
                continuum = self._calculate_linear_continuum(blue_range, band_range, red_range)
            
            # Determine wavelength range
            min_wave = min(blue_range[0], band_range[0], red_range[0])
            max_wave = max(blue_range[1], band_range[1], red_range[1])
            
            # Expand the range slightly
            range_width = max_wave - min_wave
            display_min = min_wave - 0.05 * range_width
            display_max = max_wave + 0.05 * range_width
            
            # Only show needed wavelength range
            plot_mask = (self.wavelength >= display_min) & (self.wavelength <= display_max)
            wave_plot = self.wavelength[plot_mask]
            flux_plot = self.flux[plot_mask]
            continuum_plot = continuum[plot_mask]
            
            # Check if there's valid data
            if len(wave_plot) == 0:
                ax.set_title(f"{name} (no data)")
                continue
            
            # Plot original spectrum
            ax.plot(wave_plot, flux_plot, color='black', lw=1, alpha=0.7, label='Observed')
            
            # Plot continuum
            ax.plot(wave_plot, continuum_plot, color='red', lw=1, ls='--', alpha=0.8, label='Continuum')
            
            # Plot template spectrum if available
            if self.template_wave is not None and self.template_flux is not None:
                mask_temp = (self.template_wave >= display_min) & (self.template_wave <= display_max)
                if np.any(mask_temp):
                    ax.plot(self.template_wave[mask_temp], self.template_flux[mask_temp], 
                            color='green', lw=1, alpha=0.5, label='Model')
            
            # Plot continuum regions
            ax.axvspan(blue_range[0], blue_range[1], color='blue', alpha=0.1)
            ax.axvspan(red_range[0], red_range[1], color='red', alpha=0.1)
            
            # Plot bandpass filter region
            ax.axvspan(band_range[0], band_range[1], color='green', alpha=0.1)
            
            # Calculate index value
            try:
                index_value = self.calculate_index(name)
                index_text = f"{name} = {index_value:.2f} Å"
            except:
                index_text = f"{name} = N/A"
            
            # Title and labels
            ax.set_title(index_text)
            ax.set_xlabel('Wavelength [Å]')
            ax.set_ylabel('Flux')
            
            if i == 0:  # Only add legend to first subplot
                ax.legend(loc='upper right', fontsize='small')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        if mode == 'P2P':
            save_path = f"plots/LIC_{mode}_pixel{number}.png"
        elif mode == 'VNB':
            save_path = f"plots/LIC_{mode}_bin{number}.png"
        elif mode == 'RDB':
            save_path = f"plots/LIC_{mode}_ring{number}.png"
        else:
            save_path = f"plots/LIC_indices.png"
            
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        
        logging.info(f"Line index comparison plot saved to: {save_path}")


### ------------------------------------------------- ###
# Pixel-by-Pixel Fitting Implementation
### ------------------------------------------------- ###

def fit_pixel(args):
    """
    Fit a single pixel's spectrum.
    
    Parameters
    ----------
    args : tuple
        (idx, k, galaxy_data, sps, gas_templates, gas_names, line_wave, config)
        
    Returns
    -------
    tuple
        (idx, results_dict or None)
    """
    idx, k, galaxy_data, sps, gas_templates, gas_names, line_wave, config = args
    
    try:
        # Get spectrum
        spectrum = galaxy_data.spectra[:, k]
        
        # Apply wavelength range restriction
        wave_range = [(Apply_velocity_correction(config.good_wavelength_range[0], config.redshift)),
                     (Apply_velocity_correction(config.good_wavelength_range[1], config.redshift))]
        
        # Wavelength range
        lam_gal = galaxy_data.lam_gal
        lam_range_temp = np.exp(sps.ln_lam_temp[[0, -1]])
        
        # Apply wavelength range filter
        spectrum = spectrum[np.where((lam_gal > wave_range[0]) & (lam_gal < wave_range[1]))]
        lam_gal = lam_gal[np.where((lam_gal > wave_range[0]) & (lam_gal < wave_range[1]))]
        
        # Use uniform noise
        noise = np.ones_like(spectrum)
        
        # Automatically calculate mask
        mask = util.determine_mask(np.log(lam_gal), lam_range_temp, width=1000)
        
        if not np.any(mask):
            return idx, None
        
        # First stage: fit stellar component only
        try:
            pp_stars = ppxf(sps.templates, spectrum, noise, galaxy_data.velscale, 
                           [config.vel_s, config.vel_dis_s],
                           degree=3,
                           plot=False, mask=mask, lam=lam_gal, 
                           lam_temp=sps.lam_temp, quiet=True)
        except Exception as e:
            if config.retry_with_degree_zero:
                # Try with simpler polynomial
                pp_stars = ppxf(sps.templates, spectrum, noise, galaxy_data.velscale, 
                               [config.vel_s, config.vel_dis_s],
                               degree=0, 
                               plot=False, mask=mask, lam=lam_gal, 
                               lam_temp=sps.lam_temp, quiet=True)
            else:
                raise
        
        # Create optimal stellar template
        if pp_stars.weights is None or not np.any(np.isfinite(pp_stars.weights)):
            return idx, None
        
        # Calculate optimal stellar template
        optimal_stellar_template = sps.templates @ pp_stars.weights
        
        # Record apoly
        apoly = pp_stars.apoly if hasattr(pp_stars, 'apoly') and pp_stars.apoly is not None else None
        
        # Save first stage results
        vel_stars = to_scalar(pp_stars.sol[0])
        sigma_stars = to_scalar(pp_stars.sol[1]) 
        bestfit_stars = pp_stars.bestfit
        
        # Ensure sigma value is reasonable
        if sigma_stars < 0:
            sigma_stars = 10.0
        
        # Second stage: fit combined stellar and gas templates
        if config.use_two_stage_fit and config.compute_emission_lines:
            # Define wavelength range
            wave_range = [(Apply_velocity_correction(config.good_wavelength_range[0], config.redshift)),
                          (Apply_velocity_correction(config.good_wavelength_range[1], config.redshift))]
            
            # Only use wavelength range of observation data
            wave_mask = (lam_gal >= wave_range[0]) & (lam_gal <= wave_range[1])
            galaxy_subset = spectrum[wave_mask]
            noise_subset = np.ones_like(galaxy_subset)
            
            # Ensure stellar template has correct shape
            if optimal_stellar_template.ndim > 1 and optimal_stellar_template.shape[1] == 1:
                optimal_stellar_template = optimal_stellar_template.flatten()
            
            # Combine stellar and gas templates
            stars_gas_templates = np.column_stack([optimal_stellar_template, gas_templates])
            
            # Set components array
            component = [0] + [1]*gas_templates.shape[1]
            gas_component = np.array(component) > 0
            
            # Set moments parameter
            moments = config.moments
            
            # Set starting values
            start = [
                [vel_stars, sigma_stars],  # Stellar component
                [vel_stars, 50]            # Gas component
            ]
            
            # Set bounds
            vlim = lambda x: vel_stars + x*np.array([-100, 100])
            bounds = [
                [vlim(2), [20, 300]],  # Stellar component
                [vlim(2), [20, 100]]   # Gas component
            ]
            
            # Set tied parameters
            ncomp = len(moments)
            tied = [['', ''] for _ in range(ncomp)]
            
            try:
                # Execute second stage fit
                pp = ppxf(stars_gas_templates, galaxy_subset, noise_subset, galaxy_data.velscale, start,
                         plot=False, moments=moments, degree=3, mdegree=-1, 
                         component=component, gas_component=gas_component, gas_names=gas_names,
                         lam=lam_gal[wave_mask], lam_temp=sps.lam_temp, 
                         tied=tied, bounds=bounds, quiet=True,
                         global_search=config.global_search)
                
                # Check if emission lines were successfully fit
                has_emission = False
                if hasattr(pp, 'gas_bestfit') and pp.gas_bestfit is not None:
                    has_emission = np.any(np.abs(pp.gas_bestfit) > 1e-10)
                
                # Create full bestfit
                full_bestfit = np.copy(bestfit_stars)
                
                # Calculate template
                Apoly_Params = np.polyfit(lam_gal[wave_mask], pp.apoly, 3)
                Temp_Calu = (stars_gas_templates[:,0] * pp.weights[0]) + np.poly1d(Apoly_Params)(sps.lam_temp)
                
                # Add full gas template
                if hasattr(pp, 'gas_bestfit') and pp.gas_bestfit is not None:
                    # Replace with second stage fit result within subset range
                    full_bestfit[wave_mask] = pp.bestfit
                    
                    # Create full range gas_bestfit
                    full_gas_bestfit = np.zeros_like(spectrum)
                    if has_emission:
                        full_gas_bestfit[wave_mask] = pp.gas_bestfit
                    
                    pp.full_gas_bestfit = full_gas_bestfit
                else:
                    pp.full_gas_bestfit = np.zeros_like(spectrum)
                
                pp.full_bestfit = full_bestfit
                
            except Exception as e:
                if config.fallback_to_simple_fit:
                    # Fallback: just use stellar fit
                    pp = pp_stars
                    pp.full_bestfit = bestfit_stars
                    pp.full_gas_bestfit = np.zeros_like(spectrum)
                    
                    # No gas template results
                    pp.gas_bestfit = np.zeros_like(spectrum[wave_mask]) if wave_mask.any() else np.zeros_like(spectrum)
                    if not hasattr(pp, 'gas_flux'):
                        pp.gas_flux = np.zeros(len(gas_names))
                    pp.gas_bestfit_templates = np.zeros((pp.gas_bestfit.shape[0], len(gas_names)))
                else:
                    raise
        else:
            # Don't use two-stage fitting, just use first stage results
            pp = pp_stars
            pp.full_bestfit = bestfit_stars
            pp.full_gas_bestfit = np.zeros_like(spectrum)
            
            # Add gas attributes manually 
            pp.gas_bestfit = np.zeros_like(spectrum)
            pp.gas_flux = np.zeros(len(gas_names)) if gas_names is not None else np.zeros(1)
            pp.gas_bestfit_templates = np.zeros((spectrum.shape[0], 
                                               len(gas_names) if gas_names is not None else 1))
        
        # Safety check
        if pp is None or not hasattr(pp, 'full_bestfit') or pp.full_bestfit is None:
            return idx, None
            
        # Calculate SNR
        residuals = spectrum - pp.full_bestfit
        rms = robust_sigma(residuals[mask], zero=1)
        signal = np.median(spectrum[mask])
        snr = signal / rms if rms > 0 else 0
        
        # Extract emission line information
        el_results = {}
        
        if config.compute_emission_lines:
            # Check if we have gas emission line results
            has_emission = False
            if hasattr(pp, 'full_gas_bestfit') and pp.full_gas_bestfit is not None:
                has_emission = np.any(np.abs(pp.full_gas_bestfit) > 1e-10)
                
            if has_emission:
                for name in config.gas_names:
                    # Find matching gas names
                    matches = [idx for idx, gname in enumerate(gas_names) if name in gname]
                    if matches:
                        idx = matches[0]
                        
                        # Extract the flux
                        dlam = line_wave[idx] * galaxy_data.velscale / config.c
                        
                        # Safety check for gas_flux
                        if hasattr(pp, 'gas_flux') and pp.gas_flux is not None and idx < len(pp.gas_flux):
                            flux = pp.gas_flux[idx] * dlam
                        else:
                            flux = 0.0
                        
                        # Calculate A/N
                        an = 0
                        if (hasattr(pp, 'gas_bestfit_templates') and 
                            pp.gas_bestfit_templates is not None and 
                            idx < pp.gas_bestfit_templates.shape[1]):
                            
                            peak = np.max(pp.gas_bestfit_templates[:, idx])
                            an = peak / rms if rms > 0 else 0
                        
                        el_results[name] = {'flux': flux, 'an': an}
            else:
                # Fill with empty results
                for name in config.gas_names:
                    el_results[name] = {'flux': 0.0, 'an': 0.0}
        
        # Save optimal template
        optimal_template = optimal_stellar_template
        
        # Calculate spectral indices
        indices = {}
        
        if config.compute_spectral_indices:
            try:
                # Create index calculator
                calculator = LineIndexCalculator(
                    lam_gal, spectrum,
                    sps.lam_temp, Temp_Calu,
                    em_wave=lam_gal,
                    em_flux_list=pp.full_gas_bestfit if hasattr(pp, 'full_gas_bestfit') else None,
                    velocity_correction=to_scalar(pp.sol[0]),
                    continuum_mode=config.continuum_mode)
                
                # Only generate LIC plot when needed
                if not config.no_plots and config.plot_count < config.max_plots and config.LICplot:
                    calculator.plot_all_lines(mode='P2P', number=k)
                    config.plot_count += 1
                
                # Calculate requested spectral indices
                for index_name in config.line_indices:
                    try:
                        indices[index_name] = calculator.calculate_index(index_name)
                    except Exception as e:
                        indices[index_name] = np.nan
            except Exception as e:
                for index_name in config.line_indices:
                    indices[index_name] = np.nan
        
        # Summarize results
        sol_0 = 0.0
        sol_1 = 0.0
        if hasattr(pp, 'sol') and pp.sol is not None:
            if len(pp.sol) > 0:
                sol_0 = to_scalar(pp.sol[0])
            if len(pp.sol) > 1:
                sol_1 = to_scalar(pp.sol[1])
        
        # Ensure all required arrays exist
        bestfit = pp.full_bestfit if hasattr(pp, 'full_bestfit') else pp.bestfit
        gas_bestfit = pp.full_gas_bestfit if hasattr(pp, 'full_gas_bestfit') else np.zeros_like(spectrum)
        
        results = {
            'success': True,
            'velocity': sol_0,
            'sigma': sol_1,
            'bestfit': bestfit,
            'weights': pp.weights if hasattr(pp, 'weights') and pp.weights is not None else np.zeros(1),
            'gas_bestfit': gas_bestfit,
            'optimal_template': optimal_template,
            'apoly': apoly,
            'rms': rms,
            'snr': snr,
            'el_results': el_results,
            'indices': indices,
            'stage1_bestfit': bestfit_stars,
            'pp_obj': pp
        }
        
        return idx, results
    
    except Exception as e:
        logging.error(f"Error fitting pixel {idx}: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return idx, None


def run_p2p_analysis(config):
    """
    Run pixel-by-pixel spectral fitting analysis.
    
    Parameters
    ----------
    config : P2PConfig
        Configuration object
        
    Returns
    -------
    tuple
        (galaxy_data, velfield, sigfield)
    """
    logging.info(f"===== Starting Pixel-by-Pixel Analysis (parallel mode={config.parallel_mode}) =====")
    
    # Start timing
    start_time = time.time()
    
    try:
        # 1. Load data
        logging.info("Loading data...")
        galaxy_data = IFUDataCube(config.get_data_path(), config.lam_range_temp, config.redshift, config)
        
        # 2. Prepare templates
        logging.info("Preparing stellar and gas templates...")
        sps, gas_templates, gas_names, line_wave = prepare_templates(config, galaxy_data.velscale)
        
        # 3. Get data dimensions
        nw, npix = galaxy_data.spectra.shape
        ny, nx = galaxy_data.cube.shape[1:3]
        
        # Create arrays for results
        velfield = np.full((ny, nx), np.nan)
        sigfield = np.full((ny, nx), np.nan)
        
        # Create maps for emission lines and indices
        el_flux_maps = {}
        el_snr_maps = {}
        index_maps = {}
        
        for name in config.gas_names:
            el_flux_maps[name] = np.full((ny, nx), np.nan)
            el_snr_maps[name] = np.full((ny, nx), np.nan)
            
        for name in config.line_indices:
            index_maps[name] = np.full((ny, nx), np.nan)
        
        # 4. Perform pixel-by-pixel fitting in parallel
        logging.info(f"Starting pixel-by-pixel fitting for {npix} pixels...")
        
        # Create list of pixels to process (filter by SNR)
        pixel_list = []
        for k in range(npix):
            if galaxy_data.signal[k] > 0 and galaxy_data.noise[k] > 0:
                pixel_list.append(k)
        
        logging.info(f"Processing {len(pixel_list)}/{npix} valid pixels")
        
        # Choose processing method based on parallel mode
        if config.parallel_mode == 'grouped':
            # Memory optimization: process pixels in batches
            batch_size = config.batch_size
            n_batches = (len(pixel_list) + batch_size - 1) // batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(pixel_list))
                
                logging.info(f"Processing batch {batch_idx+1}/{n_batches} (pixels {start_idx+1}-{end_idx})")
                
                # Create list of pixels in this batch
                batch_pixels = [pixel_list[i] for i in range(start_idx, end_idx)]
                
                with ProcessPoolExecutor(max_workers=config.n_threads) as executor:
                    futures = []
                    for i, k in enumerate(batch_pixels):
                        args = (i + start_idx, k, galaxy_data, sps, gas_templates, gas_names, line_wave, config)
                        futures.append(executor.submit(fit_pixel, args))
                    
                    # Process results
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {batch_idx+1}"):
                        idx, result = future.result()
                        if result is not None and result.get('success', False):
                            k = pixel_list[idx]
                            # Get position in 2D grid
                            y = (k // nx)
                            x = (k % nx)
                            
                            # Save velocity and dispersion
                            velfield[y, x] = result['velocity']
                            sigfield[y, x] = result['sigma']
                            
                            # Save emission line information
                            for name, data in result.get('el_results', {}).items():
                                if name in el_flux_maps:
                                    el_flux_maps[name][y, x] = data['flux']
                                    el_snr_maps[name][y, x] = data['an']
                            
                            # Save spectral indices
                            for name, value in result.get('indices', {}).items():
                                if name in index_maps:
                                    index_maps[name][y, x] = value
                
                # Force garbage collection
                import gc
                gc.collect()
        
        else:  # global mode
            logging.info(f"Using global parallel mode for all {len(pixel_list)} pixels")
            
            with ProcessPoolExecutor(max_workers=config.n_threads) as executor:
                futures = []
                for i, k in enumerate(pixel_list):
                    args = (i, k, galaxy_data, sps, gas_templates, gas_names, line_wave, config)
                    futures.append(executor.submit(fit_pixel, args))
                
                # Process results
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pixels"):
                    idx, result = future.result()
                    if result is not None and result.get('success', False):
                        k = pixel_list[idx]
                        # Get position in 2D grid
                        y = (k // nx)
                        x = (k % nx)
                        
                        # Save velocity and dispersion
                        velfield[y, x] = result['velocity']
                        sigfield[y, x] = result['sigma']
                        
                        # Save emission line information
                        for name, data in result.get('el_results', {}).items():
                            if name in el_flux_maps:
                                el_flux_maps[name][y, x] = data['flux']
                                el_snr_maps[name][y, x] = data['an']
                        
                        # Save spectral indices
                        for name, value in result.get('indices', {}).items():
                            if name in index_maps:
                                index_maps[name][y, x] = value
        
        # 5. Store results in galaxy_data object
        galaxy_data.velfield = velfield
        galaxy_data.sigfield = sigfield
        
        for name in config.gas_names:
            galaxy_data.el_flux_maps[name] = el_flux_maps[name]
            galaxy_data.el_snr_maps[name] = el_snr_maps[name]
            
        for name in config.line_indices:
            galaxy_data.index_maps[name] = index_maps[name]
        
        # 6. Save results to FITS files
        logging.info("Saving results to FITS files...")
        
        # Create header
        hdr = fits.Header()
        hdr['OBJECT'] = config.galaxy_name
        hdr['REDSHIFT'] = config.redshift
        hdr['CD1_1'] = galaxy_data.CD1_1
        hdr['CD1_2'] = galaxy_data.CD1_2
        hdr['CD2_1'] = galaxy_data.CD2_1
        hdr['CD2_2'] = galaxy_data.CD2_2
        hdr['CRVAL1'] = galaxy_data.CRVAL1
        hdr['CRVAL2'] = galaxy_data.CRVAL2
        
        # Add P2P information
        hdr['BINTYPE'] = 'P2P'
        hdr['PARMODE'] = config.parallel_mode
        
        # Save velocity field
        hdu_vel = fits.PrimaryHDU(velfield, header=hdr)
        hdu_vel.header['CONTENT'] = 'Stellar velocity field (P2P)'
        hdu_vel.header['BUNIT'] = 'km/s'
        hdu_vel.writeto(config.get_output_filename("velfield", "P2P"), overwrite=True)
        
        # Save velocity dispersion field
        hdu_sig = fits.PrimaryHDU(sigfield, header=hdr)
        hdu_sig.header['CONTENT'] = 'Stellar velocity dispersion (P2P)'
        hdu_sig.header['BUNIT'] = 'km/s'
        hdu_sig.writeto(config.get_output_filename("sigfield", "P2P"), overwrite=True)
        
        # Save emission line maps
        if config.compute_emission_lines:
            for name in config.gas_names:
                if name in el_flux_maps:
                    hdu = fits.PrimaryHDU(el_flux_maps[name], header=hdr)
                    hdu.header['CONTENT'] = f'{name} emission line flux (P2P)'
                    hdu.header['BUNIT'] = 'flux units'
                    hdu.writeto(config.get_output_filename(f"{name}_flux", "P2P"), overwrite=True)
                    
                    hdu = fits.PrimaryHDU(el_snr_maps[name], header=hdr)
                    hdu.header['CONTENT'] = f'{name} emission line S/N (P2P)'
                    hdu.header['BUNIT'] = 'ratio'
                    hdu.writeto(config.get_output_filename(f"{name}_snr", "P2P"), overwrite=True)
        
        # Save spectral index maps
        if config.compute_spectral_indices:
            for name in config.line_indices:
                if name in index_maps:
                    hdu = fits.PrimaryHDU(index_maps[name], header=hdr)
                    hdu.header['CONTENT'] = f'{name} spectral index (P2P)'
                    hdu.header['BUNIT'] = 'Angstrom'
                    hdu.writeto(config.get_output_filename(f"{name}_index", "P2P"), overwrite=True)
        
        # 7. Calculate completion time
        end_time = time.time()
        logging.info(f"P2P analysis completed in {end_time - start_time:.1f} seconds")
        
        return galaxy_data, velfield, sigfield
        
    except Exception as e:
        logging.error(f"Error in P2P analysis: {str(e)}")
        logging.exception("Stack trace:")
        raise


### ------------------------------------------------- ###
# Voronoi Binning Implementation
### ------------------------------------------------- ###

class VoronoiBinning:
    """
    Voronoi binning-based spectral analysis class.
    
    Groups pixels into regions with target signal-to-noise using Voronoi tessellation,
    then performs a single spectral fit for each bin.
    """
    
    def __init__(self, galaxy_data, config):
        """
        Initialize Voronoi binning.
        
        Parameters
        ----------
        galaxy_data : IFUDataCube
            Object containing galaxy data
        config : P2PConfig
            Configuration object
        """
        self.galaxy_data = galaxy_data
        self.config = config
        self.bin_data = None
        self.bin_results = {}
        
        # Arrays to store bin mapping and results
        ny, nx = galaxy_data.cube.shape[1:3]
        self.bin_map = np.full((ny, nx), -1)  # -1 indicates unbinned pixels
        self.bin_signal = np.full((ny, nx), np.nan)
        self.bin_noise = np.full((ny, nx), np.nan)
        self.bin_snr = np.full((ny, nx), np.nan)
        
        # Create results maps
        self.velfield = np.full((ny, nx), np.nan)
        self.sigfield = np.full((ny, nx), np.nan)
        
        # Save bin map to galaxy_data
        galaxy_data.bin_map = self.bin_map.copy()
        
        # Emission line and index maps
        self.el_flux_maps = {}
        self.el_snr_maps = {}
        self.index_maps = {}
        
        for name in config.gas_names:
            self.el_flux_maps[name] = np.full((ny, nx), np.nan)
            self.el_snr_maps[name] = np.full((ny, nx), np.nan)
            
        for name in config.line_indices:
            self.index_maps[name] = np.full((ny, nx), np.nan)
    
    def create_bins(self, target_snr=20, pixsize=None, cores=None):
        """
        Create Voronoi bins.
        
        Parameters
        ----------
        target_snr : float
            Target signal-to-noise ratio
        pixsize : float, optional
            Pixel size
        cores : int, optional
            Number of cores for parallel computation
            
        Returns
        -------
        int
            Number of bins created
        """
        logging.info(f"===== Creating Voronoi bins (target SNR: {target_snr}) =====")
        
        # Get data dimensions
        ny, nx = self.galaxy_data.cube.shape[1:3]
        
        # Extract coordinates and SNR data
        x = self.galaxy_data.x
        y = self.galaxy_data.y
        signal = self.galaxy_data.signal
        noise = self.galaxy_data.noise
        
        # Calculate SNR
        snr = np.divide(signal, noise, out=np.zeros_like(signal), where=noise>0)
        
        # Create mask for valid pixels (SNR>0)
        mask = (snr > 0) & np.isfinite(snr)
        x_good = x[mask]
        y_good = y[mask]
        signal_good = signal[mask]
        noise_good = noise[mask]
        
        # Ensure we have enough pixels for binning
        if len(x_good) < 10:
            logging.error("Not enough valid pixels for Voronoi binning")
            return 0
            
        logging.info(f"Using {len(x_good)}/{len(x)} valid pixels for Voronoi binning")
        
        try:
            # Perform Voronoi binning
            start_time = time.time()
            logging.info(f"Starting Voronoi binning computation...")
            
            # Set pixel size
            if pixsize is None:
                pixsize = 0.5 * (self.galaxy_data.pixsize_x + self.galaxy_data.pixsize_y)
                
            # Execute Voronoi binning
            bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale = voronoi_2d_binning(
                x_good, y_good, signal_good, noise_good, 
                target_snr, pixsize=pixsize, plot=False, quiet=True,
                cvt=True, wvt=True, cores=cores)
            
            # Calculate binning time
            end_time = time.time()
            logging.info(f"Voronoi binning complete: {np.max(bin_num)+1} bins created in {end_time - start_time:.1f} seconds")
            
            # Create complete bin mapping (including unused pixels)
            full_bin_map = np.full(len(x), -1)  # Default -1 (unused)
            full_bin_map[mask] = bin_num
            
            # Reconstruct 2D bin map
            bin_map_2d = np.full((ny, nx), -1)
            for i, (r, c) in enumerate(zip(self.galaxy_data.row, self.galaxy_data.col)):
                if full_bin_map[i] >= 0:  # Only map valid bins
                    # Note: coordinates are 1-indexed, need -1 for array indexing
                    bin_map_2d[r-1, c-1] = full_bin_map[i]
            
            # Save bin mapping
            self.bin_map = bin_map_2d
            self.galaxy_data.bin_map = bin_map_2d.copy()
            
            # Save bin information
            self.n_bins = np.max(bin_num) + 1
            self.x_bar = x_bar
            self.y_bar = y_bar
            self.bin_snr = sn
            self.n_pixels = n_pixels
            
            # Return bin count
            return self.n_bins
            
        except Exception as e:
            logging.error(f"Error during Voronoi binning: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return 0
    
    def extract_bin_spectra(self, p2p_velfield=None, isap_velfield=None):
        """
        Extract combined spectra from bins.
        
        Parameters
        ----------
        p2p_velfield : ndarray, optional
            P2P analysis velocity field, used for spectrum correction
        isap_velfield : ndarray, optional
            ISAP output velocity field, takes precedence over P2P field
            
        Returns
        -------
        dict
            Dictionary containing combined spectra
        """
        logging.info(f"===== Extracting combined spectra for {self.n_bins} bins =====")
        
        # Get data dimensions
        ny, nx = self.galaxy_data.cube.shape[1:3]
        npix = self.galaxy_data.spectra.shape[0]
        
        # First check if ISAP velocity field is provided
        if isap_velfield is not None:
            velfield = isap_velfield
            logging.info("Using ISAP velocity field for spectrum correction")
        # Then check if P2P velocity field is provided
        elif p2p_velfield is not None:
            velfield = p2p_velfield
            logging.info("Using P2P velocity field for spectrum correction")
        # Otherwise check if galaxy_data has velocity field
        elif hasattr(self.galaxy_data, 'velfield') and self.galaxy_data.velfield is not None:
            velfield = self.galaxy_data.velfield
            logging.info("Using galaxy_data velocity field for spectrum correction")
        else:
            # If no velocity field provided, try to automatically load P2P results
            p2p_vel_path, p2p_vel_exists = self.config.get_p2p_output_path("velfield")
            if p2p_vel_exists:
                try:
                    velfield = fits.getdata(p2p_vel_path)
                    logging.info(f"Automatically loaded P2P velocity field from {p2p_vel_path}")
                except Exception as e:
                    logging.warning(f"Failed to load P2P velocity field: {str(e)}")
                    velfield = None
            else:
                velfield = None
                logging.info("No velocity field found, not applying velocity correction")
        
        # Check if velfield is valid and shapes match
        if velfield is not None:
            if velfield.shape != (ny, nx):
                logging.warning(f"Velocity field shape {velfield.shape} doesn't match data {(ny, nx)}, not using velocity correction")
                velfield = None
            elif np.all(np.isnan(velfield)):
                logging.warning("Velocity field is all NaN, not using velocity correction")
                velfield = None
            else:
                apply_vel_correction = True
                logging.info("Velocity correction enabled")
        else:
            apply_vel_correction = False
            logging.info("No velocity correction - no valid velocity field found")
        
        # Create common wavelength grid (for resampling)
        lam_gal = self.galaxy_data.lam_gal
        
        # Initialize combined spectra dictionaries
        bin_spectra = {}
        bin_variances = {}
        bin_positions = {}
        
        # Extract spectrum for each bin
        for bin_id in range(self.n_bins):
            # Find all pixels belonging to this bin
            bin_mask = (self.bin_map == bin_id)
            
            if not np.any(bin_mask):
                logging.warning(f"Bin {bin_id} contains no pixels")
                continue
                
            # Get row and column indices for all pixels in this bin
            rows, cols = np.where(bin_mask)
            
            # Initialize accumulation variables
            coadded_spectrum = np.zeros(npix)
            coadded_variance = np.zeros(npix)
            total_weight = 0
            
            # Process each pixel
            for r, c in zip(rows, cols):
                k_index = r * nx + c
                
                # Get original spectrum
                pixel_spectrum = self.galaxy_data.spectra[:, k_index]
                
                # If variance data available, use it, otherwise create uniform variance
                if hasattr(self.galaxy_data, 'variance'):
                    pixel_variance = self.galaxy_data.variance[:, k_index]
                else:
                    pixel_variance = np.ones_like(pixel_spectrum)
                
                # Calculate weight for current pixel (using SNR)
                if hasattr(self.galaxy_data, 'signal') and hasattr(self.galaxy_data, 'noise'):
                    if k_index < len(self.galaxy_data.signal):
                        signal = self.galaxy_data.signal[k_index]
                        noise = self.galaxy_data.noise[k_index]
                        weight = (signal / noise)**2 if noise > 0 else 0
                    else:
                        weight = 1.0
                else:
                    weight = 1.0
                
                # Apply velocity correction (if available)
                if apply_vel_correction and not np.isnan(velfield[r, c]):
                    vel = velfield[r, c]
                    
                    # Shifted wavelength
                    lam_shifted = lam_gal * (1 + vel/self.config.c)
                    
                    # Resample to original wavelength grid
                    corrected_spectrum = np.interp(lam_gal, lam_shifted, pixel_spectrum,
                                                 left=0, right=0)
                    corrected_variance = np.interp(lam_gal, lam_shifted, pixel_variance,
                                                 left=np.inf, right=np.inf)
                    
                    # Accumulate corrected spectrum (weighted)
                    coadded_spectrum += corrected_spectrum * weight
                    coadded_variance += corrected_variance * weight**2
                else:
                    # No correction, accumulate directly
                    coadded_spectrum += pixel_spectrum * weight
                    coadded_variance += pixel_variance * weight**2
                
                total_weight += weight
            
            # Normalize accumulated spectrum
            if total_weight > 0:
                merged_spectrum = coadded_spectrum / total_weight
                merged_variance = coadded_variance / (total_weight**2)
            else:
                logging.warning(f"Bin {bin_id} has zero total weight, using simple average")
                merged_spectrum = coadded_spectrum / len(rows) if len(rows) > 0 else coadded_spectrum
                merged_variance = coadded_variance / (len(rows)**2) if len(rows) > 0 else coadded_variance
            
            # Store combined data
            bin_spectra[bin_id] = merged_spectrum
            bin_variances[bin_id] = merged_variance
            
            # Save bin position information
            bin_positions[bin_id] = {
                'x': np.mean(cols),  # columns correspond to x
                'y': np.mean(rows),  # rows correspond to y
                'n_pixels': len(rows)
            }
            
            # Add SNR information
            snr = np.median(merged_spectrum / np.sqrt(merged_variance))
            bin_positions[bin_id]['snr'] = snr
            
            # Log progress
            if bin_id % 50 == 0 or bin_id == self.n_bins - 1:
                logging.info(f"Extracted {bin_id+1}/{self.n_bins} bin spectra, SNR={snr:.1f}")
        
        # Save extracted data
        self.bin_data = {
            'spectra': bin_spectra,
            'variances': bin_variances,
            'positions': bin_positions
        }
        
        logging.info(f"Successfully extracted {len(bin_spectra)}/{self.n_bins} bin spectra")
        
        return self.bin_data
    
    def fit_bins(self, sps, gas_templates, gas_names, line_wave):
        """
        Fit combined spectra for each bin.
        
        Parameters
        ----------
        sps : object
            Stellar population synthesis library
        gas_templates : ndarray
            Gas emission line templates
        gas_names : array
            Gas emission line names
        line_wave : array
            Emission line wavelengths
            
        Returns
        -------
        dict
            Fitting results dictionary
        """
        logging.info(f"===== Starting fits for {self.n_bins} bin spectra (parallel mode={self.config.parallel_mode}) =====")
        
        if self.bin_data is None:
            logging.error("No bin data available")
            return {}
        
        # Prepare fitting parameters
        bin_ids = list(self.bin_data['spectra'].keys())
        
        # Use multiprocessing for parallel fitting
        start_time = time.time()
        results = {}
        
        # Choose processing method based on parallel mode
        if self.config.parallel_mode == 'grouped':
            # Memory optimization: process bins in batches
            batch_size = self.config.batch_size
            
            for batch_start in range(0, len(bin_ids), batch_size):
                batch_end = min(batch_start + batch_size, len(bin_ids))
                batch_bins = bin_ids[batch_start:batch_end]
                
                logging.info(f"Processing batch {batch_start//batch_size + 1}/{(len(bin_ids)-1)//batch_size + 1} "
                            f"(bins {batch_start+1}-{batch_end})")
                
                with ProcessPoolExecutor(max_workers=self.config.n_threads) as executor:
                    # Submit batch tasks
                    future_to_bin = {}
                    for bin_id in batch_bins:
                        # Prepare parameters
                        spectrum = self.bin_data['spectra'][bin_id]
                        position = self.bin_data['positions'][bin_id]
                        
                        # Create simulated single-pixel input
                        args = (bin_id, -1, self.galaxy_data, sps, gas_templates, gas_names, line_wave, self.config)
                        args[2].spectra = np.column_stack([spectrum])  # Replace with bin spectrum
                        
                        # Submit task
                        future = executor.submit(fit_bin, args)
                        future_to_bin[future] = bin_id
                    
                    # Process results
                    with tqdm(total=len(batch_bins), desc=f"Batch {batch_start//batch_size + 1}") as pbar:
                        for future in as_completed(future_to_bin):
                            bin_id, result = future.result()
                            if result is not None:
                                results[bin_id] = result
                            pbar.update(1)
                
                # Force garbage collection
                import gc
                gc.collect()
        
        else:  # global mode
            logging.info(f"Using global parallel mode for all {len(bin_ids)} bins")
            
            with ProcessPoolExecutor(max_workers=self.config.n_threads) as executor:
                # Submit all tasks
                future_to_bin = {}
                for bin_id in bin_ids:
                    # Prepare parameters
                    spectrum = self.bin_data['spectra'][bin_id]
                    position = self.bin_data['positions'][bin_id]
                    
                    # Create simulated single-pixel input
                    args = (bin_id, -1, self.galaxy_data, sps, gas_templates, gas_names, line_wave, self.config)
                    args[2].spectra = np.column_stack([spectrum])  # Replace with bin spectrum
                    
                    # Submit task
                    future = executor.submit(fit_bin, args)
                    future_to_bin[future] = bin_id
                
                # Process results
                with tqdm(total=len(bin_ids), desc="Processing bins") as pbar:
                    for future in as_completed(future_to_bin):
                        bin_id, result = future.result()
                        if result is not None:
                            results[bin_id] = result
                        pbar.update(1)
        
        # Calculate completion time
        end_time = time.time()
        successful = len(results)
        logging.info(f"Completed {successful}/{self.n_bins} bin fits in {end_time - start_time:.1f} seconds")
        
        # Save results
        self.bin_results = results
        
        return results
    
    def process_results(self):
        """
        Process fitting results and populate maps.
        
        Returns
        -------
        dict
            Processed results dictionary
        """
        logging.info(f"===== Processing results for {len(self.bin_results)} bins =====")
        
        if not self.bin_results:
            logging.error("No fitting results available")
            return {}
        
        # Get data dimensions
        ny, nx = self.galaxy_data.cube.shape[1:3]
        
        # Initialize results arrays
        velfield = np.full((ny, nx), np.nan)
        sigfield = np.full((ny, nx), np.nan)
        
        # Initialize emission line and index maps
        el_flux_maps = {}
        el_snr_maps = {}
        index_maps = {}
        
        for name in self.config.gas_names:
            el_flux_maps[name] = np.full((ny, nx), np.nan)
            el_snr_maps[name] = np.full((ny, nx), np.nan)
            
        for name in self.config.line_indices:
            index_maps[name] = np.full((ny, nx), np.nan)
        
        # Process results for each bin
        for bin_id, result in self.bin_results.items():
            if not result.get('success', False):
                continue
                
            # Extract result data
            velocity = result['velocity']
            sigma = result['sigma']
            
            # Extract emission line data
            el_results = result.get('el_results', {})
            
            # Extract index data
            indices = result.get('indices', {})
            
            # Find all pixels belonging to this bin
            bin_mask = (self.bin_map == bin_id)
            
            # Populate velocity and dispersion maps
            velfield[bin_mask] = velocity
            sigfield[bin_mask] = sigma
            
            # Populate emission line maps
            for name, data in el_results.items():
                if name in el_flux_maps:
                    el_flux_maps[name][bin_mask] = data['flux']
                    el_snr_maps[name][bin_mask] = data['an']
            
            # Populate index maps
            for name, value in indices.items():
                if name in index_maps:
                    index_maps[name][bin_mask] = value
        
        # Save processed maps
        self.velfield = velfield
        self.sigfield = sigfield
        self.el_flux_maps = el_flux_maps
        self.el_snr_maps = el_snr_maps
        self.index_maps = index_maps
        
        # Also update maps in galaxy_data
        self.galaxy_data.velfield = velfield.copy()
        self.galaxy_data.sigfield = sigfield.copy()
        
        for name in self.config.gas_names:
            self.galaxy_data.el_flux_maps[name] = el_flux_maps[name].copy()
            self.galaxy_data.el_snr_maps[name] = el_snr_maps[name].copy()
            
        for name in self.config.line_indices:
            self.galaxy_data.index_maps[name] = index_maps[name].copy()
        
        # Create CSV summary
        self.create_bin_summary()
        
        return {
            'velfield': velfield,
            'sigfield': sigfield,
            'el_flux_maps': el_flux_maps,
            'el_snr_maps': el_snr_maps,
            'index_maps': index_maps
        }
    
    def create_bin_summary(self):
        """
        Create bin results summary.
        
        Returns
        -------
        DataFrame
            Results summary
        """
        # Create data records list
        data = []
        
        for bin_id, result in self.bin_results.items():
            if not result.get('success', False):
                continue
                
            # Get bin position
            position = self.bin_data['positions'][bin_id]
            n_pixels = position['n_pixels']
            
            # Create basic record
            record = {
                'bin_id': bin_id,
                'x': position['x'],
                'y': position['y'],
                'n_pixels': n_pixels,
                'velocity': result['velocity'],
                'sigma': result['sigma'],
                'snr': result['snr']
            }
            
            # Add emission line data
            for name, data_dict in result.get('el_results', {}).items():
                record[f'{name}_flux'] = data_dict['flux']
                record[f'{name}_snr'] = data_dict['an']
            
            # Add index data
            for name, value in result.get('indices', {}).items():
                record[f'{name}_index'] = value
            
            data.append(record)
        
        # Create DataFrame
        if data:
            import pandas as pd
            df = pd.DataFrame(data)
            
            # Save CSV file
            csv_path = self.config.output_dir / f"{self.config.galaxy_name}_VNB_bins.csv"
            df.to_csv(csv_path, index=False)
            logging.info(f"Bin summary saved to {csv_path}")
            
            return df
        else:
            logging.warning("No bin results available to create summary")
            return None
    
    def plot_binning(self):
        """
        Plot Voronoi binning results.
        
        Returns
        -------
        None
        """
        if self.config.no_plots:
            return
            
        # If no bins successfully created, return
        if not hasattr(self, 'n_bins') or self.n_bins == 0:
            logging.warning("No bins available to plot")
            return
        
        try:
            with plt.rc_context({'figure.max_open_warning': False}):
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8), dpi=self.config.dpi)
                
                # Use vorbin's display_bins function to plot bins
                rnd_colors = display_bins(self.bin_map, ax=ax)
                
                # Labels and title
                ax.set_xlabel('X [pixels]')
                ax.set_ylabel('Y [pixels]')
                ax.set_title(f"{self.config.galaxy_name} - Voronoi Binning: {self.n_bins} bins")
                
                # Tight layout
                plt.tight_layout()
                
                # Save
                plot_path = self.config.plot_dir / f"{self.config.galaxy_name}_voronoi_bins.png"
                plt.savefig(plot_path, dpi=self.config.dpi)
                plt.close(fig)
                
                logging.info(f"Voronoi binning plot saved to {plot_path}")
        except Exception as e:
            logging.error(f"Error plotting binning: {str(e)}")
            plt.close('all')  # Ensure all figures are closed
    
    def plot_bin_results(self, bin_id):
        """
        Plot fitting results for a single bin.
        
        Parameters
        ----------
        bin_id : int
            Bin ID
            
        Returns
        -------
        None
        """
        if self.config.no_plots or self.config.plot_count >= self.config.max_plots:
            return
            
        if bin_id not in self.bin_results:
            logging.warning(f"No fitting results available for bin {bin_id}")
            return
            
        result = self.bin_results[bin_id]
        if not result.get('success', False):
            logging.warning(f"Fitting unsuccessful for bin {bin_id}")
            return
            
        try:
            # Get bin position information
            position = self.bin_data['positions'][bin_id]
            i, j = int(position['y']), int(position['x'])
            
            # Get pp object
            pp = result.get('pp_obj')
            if pp is None:
                logging.warning(f"No pp object available for bin {bin_id}")
                return
                
            # Set pp's additional attributes for plot_bin_fit function
            if hasattr(result, 'stage1_bestfit'):
                pp.stage1_bestfit = result['stage1_bestfit'] 
            else:
                pp.stage1_bestfit = pp.bestfit
                
            pp.optimal_stellar_template = result['optimal_template']
            pp.full_bestfit = result['bestfit']
            pp.full_gas_bestfit = result['gas_bestfit']
            
            # Call plotting function
            plot_bin_fit(bin_id, self.galaxy_data, pp, position, self.config)
            
            # Increment counter
            self.config.plot_count += 1
        except Exception as e:
            logging.error(f"Error plotting bin {bin_id} results: {str(e)}")
            plt.close('all')
    
    def create_summary_plots(self):
        """
        Create VNB results summary plots.
        
        Returns
        -------
        None
        """
        if self.config.no_plots:
            return
            
        try:
            # Create plots directory
            os.makedirs(self.config.plot_dir, exist_ok=True)
            
            # 1. Binning plot
            self.plot_binning()
            
            # 2. Kinematics plots
            with plt.rc_context({'figure.max_open_warning': False}):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.config.dpi)
                
                # Velocity map
                vmax = np.nanpercentile(np.abs(self.velfield), 90)
                im0 = axes[0].imshow(self.velfield, origin='lower', cmap='RdBu_r', 
                                  vmin=-vmax, vmax=vmax)
                axes[0].set_title('Velocity [km/s]')
                plt.colorbar(im0, ax=axes[0])
                
                # Velocity dispersion map
                sigma_max = np.nanpercentile(self.sigfield, 95)
                im1 = axes[1].imshow(self.sigfield, origin='lower', cmap='viridis', 
                                  vmin=0, vmax=sigma_max)
                axes[1].set_title('Velocity Dispersion [km/s]')
                plt.colorbar(im1, ax=axes[1])
                
                plt.suptitle(f"{self.config.galaxy_name} - VNB Stellar Kinematics")
                plt.tight_layout()
                plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_VNB_kinematics.png", dpi=self.config.dpi)
                plt.close(fig)
            
            # 3. Emission line plots
            if self.config.compute_emission_lines and len(self.config.gas_names) > 0:
                n_lines = len(self.config.gas_names)
                
                with plt.rc_context({'figure.max_open_warning': False}):
                    fig, axes = plt.subplots(2, n_lines, figsize=(4*n_lines, 8), dpi=self.config.dpi)
                    
                    if n_lines == 1:  # Handle single emission line case
                        axes = np.array([[axes[0]], [axes[1]]])
                    
                    for i, name in enumerate(self.config.gas_names):
                        # Flux map
                        flux_map = self.el_flux_maps[name]
                        vmax = np.nanpercentile(flux_map, 95)
                        im = axes[0, i].imshow(flux_map, origin='lower', cmap='inferno', vmin=0, vmax=vmax)
                        axes[0, i].set_title(f"{name} Flux")
                        plt.colorbar(im, ax=axes[0, i])
                        
                        # SNR map
                        snr_map = self.el_snr_maps[name]
                        im = axes[1, i].imshow(snr_map, origin='lower', cmap='viridis', vmin=0, vmax=5)
                        axes[1, i].set_title(f"{name} S/N")
                        plt.colorbar(im, ax=axes[1, i])
                    
                    plt.suptitle(f"{self.config.galaxy_name} - VNB Emission Lines")
                    plt.tight_layout()
                    plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_VNB_emission_lines.png", dpi=self.config.dpi)
                    plt.close(fig)
            
            # 4. Spectral indices plots
            if self.config.compute_spectral_indices and len(self.config.line_indices) > 0:
                n_indices = len(self.config.line_indices)
                n_cols = min(3, n_indices)
                n_rows = (n_indices + n_cols - 1) // n_cols
                
                with plt.rc_context({'figure.max_open_warning': False}):
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), dpi=self.config.dpi)
                    axes = np.atleast_2d(axes)
                    
                    for i, name in enumerate(self.config.line_indices):
                        row = i // n_cols
                        col = i % n_cols
                        
                        index_map = self.index_maps[name]
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
                    
                    plt.suptitle(f"{self.config.galaxy_name} - VNB Spectral Indices")
                    plt.tight_layout()
                    plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_VNB_indices.png", dpi=self.config.dpi)
                    plt.close(fig)
            
            # Force cleanup
            plt.close('all')
            import gc
            gc.collect()
        
        except Exception as e:
            logging.error(f"Error creating VNB summary plots: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
            plt.close('all')
    
    def save_results_to_fits(self):
        """
        Save VNB results to FITS files.
        
        Returns
        -------
        None
        """
        try:
            # Create header
            hdr = fits.Header()
            hdr['OBJECT'] = self.config.galaxy_name
            hdr['REDSHIFT'] = self.config.redshift
            hdr['CD1_1'] = self.galaxy_data.CD1_1
            hdr['CD1_2'] = self.galaxy_data.CD1_2
            hdr['CD2_1'] = self.galaxy_data.CD2_1
            hdr['CD2_2'] = self.galaxy_data.CD2_2
            hdr['CRVAL1'] = self.galaxy_data.CRVAL1
            hdr['CRVAL2'] = self.galaxy_data.CRVAL2
            
            # Add VNB information
            hdr['BINTYPE'] = 'VNB'
            hdr['NBINS'] = self.n_bins
            hdr['PARMODE'] = self.config.parallel_mode
            
            # Save bin map
            hdu_binmap = fits.PrimaryHDU(self.bin_map, header=hdr)
            hdu_binmap.header['CONTENT'] = 'Voronoi bin map'
            hdu_binmap.writeto(self.config.get_output_filename("binmap", "VNB"), overwrite=True)
            
            # Save velocity field
            hdu_vel = fits.PrimaryHDU(self.velfield, header=hdr)
            hdu_vel.header['CONTENT'] = 'Stellar velocity field (VNB)'
            hdu_vel.header['BUNIT'] = 'km/s'
            hdu_vel.writeto(self.config.get_output_filename("velfield", "VNB"), overwrite=True)
            
            # Save velocity dispersion field
            hdu_sig = fits.PrimaryHDU(self.sigfield, header=hdr)
            hdu_sig.header['CONTENT'] = 'Stellar velocity dispersion (VNB)'
            hdu_sig.header['BUNIT'] = 'km/s'
            hdu_sig.writeto(self.config.get_output_filename("sigfield", "VNB"), overwrite=True)
            
            # Save emission line maps
            for name in self.config.gas_names:
                if name in self.el_flux_maps:
                    hdu = fits.PrimaryHDU(self.el_flux_maps[name], header=hdr)
                    hdu.header['CONTENT'] = f'{name} emission line flux (VNB)'
                    hdu.header['BUNIT'] = 'flux units'
                    hdu.writeto(self.config.get_output_filename(f"{name}_flux", "VNB"), overwrite=True)
                    
                    hdu = fits.PrimaryHDU(self.el_snr_maps[name], header=hdr)
                    hdu.header['CONTENT'] = f'{name} emission line S/N (VNB)'
                    hdu.header['BUNIT'] = 'ratio'
                    hdu.writeto(self.config.get_output_filename(f"{name}_snr", "VNB"), overwrite=True)
            
            # Save spectral index maps
            for name in self.config.line_indices:
                if name in self.index_maps:
                    hdu = fits.PrimaryHDU(self.index_maps[name], header=hdr)
                    hdu.header['CONTENT'] = f'{name} spectral index (VNB)'
                    hdu.header['BUNIT'] = 'Angstrom'
                    hdu.writeto(self.config.get_output_filename(f"{name}_index", "VNB"), overwrite=True)
            
            logging.info(f"VNB results saved to FITS files")
            
        except Exception as e:
            logging.error(f"Error saving VNB results to FITS: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())


def fit_bin(args):
    """
    Fit a single Voronoi bin's combined spectrum.
    
    Parameters
    ----------
    args : tuple
        (bin_id, _, galaxy_data, sps, gas_templates, gas_names, line_wave, config)
        
    Returns
    -------
    tuple
        (bin_id, results_dict or None)
    """
    bin_id, _, galaxy_data, sps, gas_templates, gas_names, line_wave, config = args
    
    logging.debug(f"===== FITTING BIN {bin_id} =====")
    
    try:
        # Get bin spectrum
        spectrum = galaxy_data.spectra[:, 0]  # Already replaced with bin spectrum when submitting task
        
        wave_range = [(Apply_velocity_correction(config.good_wavelength_range[0], config.redshift)),
                      (Apply_velocity_correction(config.good_wavelength_range[1], config.redshift))]
        
        # Wavelength range
        lam_gal = galaxy_data.lam_gal
        lam_range_temp = np.exp(sps.ln_lam_temp[[0, -1]])
        
        # Apply wavelength range filter
        spectrum = spectrum[np.where((lam_gal > wave_range[0]) & (lam_gal < wave_range[1]))]
        lam_gal = lam_gal[np.where((lam_gal > wave_range[0]) & (lam_gal < wave_range[1]))]
        
        # Use uniform noise
        noise = np.ones_like(spectrum)
        
        # Automatically calculate mask
        mask = util.determine_mask(np.log(lam_gal), lam_range_temp, width=1000)
        
        if not np.any(mask):
            logging.warning(f"Empty mask for bin {bin_id}. Wavelength ranges may not overlap.")
            return bin_id, None
        
        # First stage: fit stellar component only
        logging.debug(f"STEP: FIRST STAGE - Stellar component only fit")
        
        try:
            pp_stars = ppxf(sps.templates, spectrum, noise, galaxy_data.velscale, 
                           [config.vel_s, config.vel_dis_s],
                           degree=3,
                           plot=False, mask=mask, lam=lam_gal, 
                           lam_temp=sps.lam_temp, quiet=True)
            
            logging.debug(f"  - First stage fit successful: v={pp_stars.sol[0]:.1f}, σ={pp_stars.sol[1]:.1f}")
        except Exception as e:
            if config.retry_with_degree_zero:
                logging.warning(f"Initial stellar fit failed for bin {bin_id}: {str(e)}")
                logging.debug(f"  - Retrying with simplified parameters: degree=0")
                # Try with simpler polynomial
                pp_stars = ppxf(sps.templates, spectrum, noise, galaxy_data.velscale, 
                               [config.vel_s, config.vel_dis_s],
                               degree=0, 
                               plot=False, mask=mask, lam=lam_gal, 
                               lam_temp=sps.lam_temp, quiet=True)
                logging.debug(f"  - Retry successful: v={pp_stars.sol[0]:.1f}, σ={pp_stars.sol[1]:.1f}")
            else:
                raise  # Re-raise the exception if we're not retrying
        
        # Create optimal stellar template
        if pp_stars.weights is None or not np.any(np.isfinite(pp_stars.weights)):
            logging.warning(f"Invalid weights in stellar fit for bin {bin_id}")
            return bin_id, None
        
        # Calculate optimal stellar template
        optimal_stellar_template = sps.templates @ pp_stars.weights
        
        # Record apoly
        apoly = pp_stars.apoly if hasattr(pp_stars, 'apoly') and pp_stars.apoly is not None else None
        
        # Save first stage results
        vel_stars = to_scalar(pp_stars.sol[0])
        sigma_stars = to_scalar(pp_stars.sol[1]) 
        bestfit_stars = pp_stars.bestfit
        
        # Ensure sigma value is reasonable
        if sigma_stars < 0:
            logging.warning(f"Negative velocity dispersion detected: {sigma_stars:.1f} km/s. Setting to 10 km/s.")
            sigma_stars = 10.0
        
        # Second stage: fit combined stellar and gas templates
        if config.use_two_stage_fit and config.compute_emission_lines:
            
            logging.debug(f"STEP: SECOND STAGE - Combined fit with optimal stellar template")
            
            # Define wavelength range
            wave_range = [(Apply_velocity_correction(config.good_wavelength_range[0], config.redshift)),
                          (Apply_velocity_correction(config.good_wavelength_range[1], config.redshift))]
            
            # Only use wavelength range of observation data
            wave_mask = (lam_gal >= wave_range[0]) & (lam_gal <= wave_range[1])
            galaxy_subset = spectrum[wave_mask]
            noise_subset = np.ones_like(galaxy_subset)
            
            # Ensure stellar template has correct shape
            if optimal_stellar_template.ndim > 1 and optimal_stellar_template.shape[1] == 1:
                optimal_stellar_template = optimal_stellar_template.flatten()
            
            # Combine stellar and gas templates
            stars_gas_templates = np.column_stack([optimal_stellar_template, gas_templates])
            
            # Set components array
            component = [0] + [1]*gas_templates.shape[1]
            gas_component = np.array(component) > 0
            
            # Set moments parameter
            moments = config.moments
            
            # Set starting values
            start = [
                [vel_stars, sigma_stars],  # Stellar component
                [vel_stars, 50]            # Gas component
            ]
            
            # Set bounds
            vlim = lambda x: vel_stars + x*np.array([-100, 100])
            bounds = [
                [vlim(2), [20, 300]],  # Stellar component
                [vlim(2), [20, 100]]   # Gas component
            ]
            
            # Set tied parameters
            ncomp = len(moments)
            tied = [['', ''] for _ in range(ncomp)]
            
            try:
                # Execute second stage fit
                pp = ppxf(stars_gas_templates, galaxy_subset, noise_subset, galaxy_data.velscale, start,
                         plot=False, moments=moments, degree=3, mdegree=-1, 
                         component=component, gas_component=gas_component, gas_names=gas_names,
                         lam=lam_gal[wave_mask], lam_temp=sps.lam_temp, 
                         tied=tied, bounds=bounds, quiet=True,
                         global_search=config.global_search)
                
                logging.debug(f"  - Combined fit successful: v={to_scalar(pp.sol[0]):.1f}, "
                             f"σ={to_scalar(pp.sol[1]):.1f}, χ²={to_scalar(pp.chi2):.3f}")
                
                # Check if emission lines were successfully fit
                has_emission = False
                if hasattr(pp, 'gas_bestfit') and pp.gas_bestfit is not None:
                    has_emission = np.any(np.abs(pp.gas_bestfit) > 1e-10)
                
                # Create full bestfit
                full_bestfit = np.copy(bestfit_stars)
                
                # Calculate template
                Apoly_Params = np.polyfit(lam_gal[wave_mask], pp.apoly, 3)
                Temp_Calu = (stars_gas_templates[:,0] * pp.weights[0]) + np.poly1d(Apoly_Params)(sps.lam_temp)
                
                # Add full gas template
                if hasattr(pp, 'gas_bestfit') and pp.gas_bestfit is not None:
                    # Replace with second stage fit result within subset range
                    full_bestfit[wave_mask] = pp.bestfit
                    
                    # Create full range gas_bestfit
                    full_gas_bestfit = np.zeros_like(spectrum)
                    if has_emission:
                        full_gas_bestfit[wave_mask] = pp.gas_bestfit
                    
                    pp.full_gas_bestfit = full_gas_bestfit
                else:
                    pp.full_gas_bestfit = np.zeros_like(spectrum)
                
                pp.full_bestfit = full_bestfit
                
            except Exception as e:
                logging.warning(f"Combined fit failed for bin {bin_id}: {str(e)}")
                
                if config.fallback_to_simple_fit:
                    logging.debug(f"  - Using fallback to stellar-only fit")
                    # Fallback: just use stellar fit
                    pp = pp_stars
                    pp.full_bestfit = bestfit_stars
                    pp.full_gas_bestfit = np.zeros_like(spectrum)
                    
                    # No gas template results
                    pp.gas_bestfit = np.zeros_like(spectrum[wave_mask]) if wave_mask.any() else np.zeros_like(spectrum)
                    if not hasattr(pp, 'gas_flux'):
                        pp.gas_flux = np.zeros(len(gas_names))
                    pp.gas_bestfit_templates = np.zeros((pp.gas_bestfit.shape[0], len(gas_names)))
                    
                    logging.info(f"Used fallback stellar-only fit for bin {bin_id}")
                else:
                    raise  # Re-raise the exception if we're not using fallback
        else:
            # Don't use two-stage fitting, just use first stage results
            logging.debug(f"STEP: Using single-stage fit (two-stage disabled)")
            pp = pp_stars
            pp.full_bestfit = bestfit_stars
            pp.full_gas_bestfit = np.zeros_like(spectrum)
            
            # Add gas attributes manually 
            pp.gas_bestfit = np.zeros_like(spectrum)
            pp.gas_flux = np.zeros(len(gas_names)) if gas_names is not None else np.zeros(1)
            pp.gas_bestfit_templates = np.zeros((spectrum.shape[0], 
                                               len(gas_names) if gas_names is not None else 1))
        
        # Safety check
        if pp is None or not hasattr(pp, 'full_bestfit') or pp.full_bestfit is None:
            logging.warning(f"Missing valid fit results for bin {bin_id}")
            return bin_id, None
            
        # Calculate SNR
        residuals = spectrum - pp.full_bestfit
        rms = robust_sigma(residuals[mask], zero=1)
        signal = np.median(spectrum[mask])
        snr = signal / rms if rms > 0 else 0
        
        # Extract emission line information
        el_results = {}
        
        if config.compute_emission_lines:
            # Check if we have gas emission line results
            has_emission = False
            if hasattr(pp, 'full_gas_bestfit') and pp.full_gas_bestfit is not None:
                has_emission = np.any(np.abs(pp.full_gas_bestfit) > 1e-10)
                
            if has_emission:
                for name in config.gas_names:
                    # Find matching gas names
                    matches = [idx for idx, gname in enumerate(gas_names) if name in gname]
                    if matches:
                        idx = matches[0]
                        
                        # Extract the flux
                        dlam = line_wave[idx] * galaxy_data.velscale / config.c
                        
                        # Safety check for gas_flux
                        if hasattr(pp, 'gas_flux') and pp.gas_flux is not None and idx < len(pp.gas_flux):
                            flux = pp.gas_flux[idx] * dlam
                        else:
                            flux = 0.0
                        
                        # Calculate A/N
                        an = 0
                        if (hasattr(pp, 'gas_bestfit_templates') and 
                            pp.gas_bestfit_templates is not None and 
                            idx < pp.gas_bestfit_templates.shape[1]):
                            
                            peak = np.max(pp.gas_bestfit_templates[:, idx])
                            an = peak / rms if rms > 0 else 0
                        
                        el_results[name] = {'flux': flux, 'an': an}
            else:
                # Fill with empty results
                for name in config.gas_names:
                    el_results[name] = {'flux': 0.0, 'an': 0.0}
        
        # Save optimal template
        optimal_template = optimal_stellar_template
        
        # Calculate spectral indices
        indices = {}
        
        if config.compute_spectral_indices:
            try:
                # Create index calculator
                calculator = LineIndexCalculator(
                    lam_gal, spectrum,
                    sps.lam_temp, Temp_Calu,
                    em_wave=lam_gal,
                    em_flux_list=pp.full_gas_bestfit if hasattr(pp, 'full_gas_bestfit') else None,
                    velocity_correction=to_scalar(pp.sol[0]),
                    continuum_mode=config.continuum_mode)
                
                # Only generate LIC plot when needed
                if config.LICplot and not config.no_plots and config.plot_count < config.max_plots:
                    calculator.plot_all_lines(mode='VNB', number=bin_id)
                    config.plot_count += 1
                
                # Calculate requested spectral indices
                for index_name in config.line_indices:
                    try:
                        indices[index_name] = calculator.calculate_index(index_name)
                    except Exception as e:
                        logging.warning(f"Failed to calculate index {index_name}: {str(e)}")
                        indices[index_name] = np.nan
            except Exception as e:
                logging.warning(f"Failed to initialize LineIndexCalculator: {str(e)}")
                for index_name in config.line_indices:
                    indices[index_name] = np.nan
        
        # Summarize results
        sol_0 = 0.0
        sol_1 = 0.0
        if hasattr(pp, 'sol') and pp.sol is not None:
            if len(pp.sol) > 0:
                sol_0 = to_scalar(pp.sol[0])
            if len(pp.sol) > 1:
                sol_1 = to_scalar(pp.sol[1])
        
        # Ensure all required arrays exist
        bestfit = pp.full_bestfit if hasattr(pp, 'full_bestfit') else pp.bestfit
        gas_bestfit = pp.full_gas_bestfit if hasattr(pp, 'full_gas_bestfit') else np.zeros_like(spectrum)
        
        results = {
            'success': True,
            'velocity': sol_0,
            'sigma': sol_1,
            'bestfit': bestfit,
            'weights': pp.weights if hasattr(pp, 'weights') and pp.weights is not None else np.zeros(1),
            'gas_bestfit': gas_bestfit,
            'optimal_template': optimal_template,
            'apoly': apoly,
            'rms': rms,
            'snr': snr,
            'el_results': el_results,
            'indices': indices,
            'stage1_bestfit': bestfit_stars,
            'pp_obj': pp
        }
        
        logging.debug(f"===== BIN {bin_id} FIT COMPLETED SUCCESSFULLY =====")
        return bin_id, results
    
    except Exception as e:
        logging.error(f"Error fitting bin {bin_id}: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return bin_id, None


def plot_bin_fit(bin_id, galaxy_data, pp, position, config):
    """
    Create diagnostic plot for bin fitting - memory optimized version
    
    Parameters
    ----------
    bin_id : int
        Bin ID
    galaxy_data : IFUDataCube
        Object containing galaxy data
    pp : ppxf object
        pPXF fitting result
    position : dict
        Dictionary containing bin position information
    config : P2PConfig
        Configuration object
    """
    # If all plots disabled, return immediately
    if config.no_plots or config.plot_count >= config.max_plots:
        return
    
    try:
        # Create plots directory
        plot_dir = config.plot_dir / 'VNB_res'
        os.makedirs(plot_dir, exist_ok=True)
        
        # Prepare filename and path
        plot_path_png = plot_dir / f"{config.galaxy_name}_bin_{bin_id}.png"
        
        # Get data
        lam_gal = galaxy_data.lam_gal
        
        # Here, we assume pp's spectra has been replaced with bin spectrum
        # So directly use first column of spectrum data
        spectrum = galaxy_data.spectra[:, 0]
        
        # Get fitting results
        bestfit = pp.full_bestfit if hasattr(pp, 'full_bestfit') else pp.bestfit
        stage1_bestfit = pp.stage1_bestfit if hasattr(pp, 'stage1_bestfit') else bestfit
        gas_bestfit = pp.full_gas_bestfit if hasattr(pp, 'full_gas_bestfit') else np.zeros_like(spectrum)
        
        # Extract needed attribute values
        velocity = to_scalar(pp.sol[0]) if hasattr(pp, 'sol') and pp.sol is not None and len(pp.sol) > 0 else 0.0
        sigma = to_scalar(pp.sol[1]) if hasattr(pp, 'sol') and pp.sol is not None and len(pp.sol) > 1 else 0.0
        chi2 = to_scalar(pp.chi2) if hasattr(pp, 'chi2') and pp.chi2 is not None else 0.0
        
        # Use with statement to create figure, ensuring resources are properly released
        with plt.rc_context({'figure.max_open_warning': False}):
            # Create figure, specify lower DPI to reduce memory usage
            fig = plt.figure(figsize=(12, 8), dpi=config.dpi)
            
            # Create subplots
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax3 = plt.subplot(gs[2])
            
            # First panel: original data and first stage fit
            ax1.plot(lam_gal, spectrum, c='k', lw=1, alpha=.8, 
                    label=f"{config.galaxy_name} bin:{bin_id} - Original")
            ax1.plot(lam_gal, stage1_bestfit, '-', c='r', alpha=.8, 
                    label='Stage 1: Stellar fit only')
            
            # Second panel: final fit result
            ax2.plot(lam_gal, spectrum, c='k', lw=1, alpha=.8, 
                    label='Original spectrum')
            ax2.plot(lam_gal, bestfit, '-', c='r', alpha=.8, 
                    label='Stage 2: Full fit')
            
            # Plot stellar component (full fit minus gas)
            stellar_comp = bestfit - gas_bestfit
            ax2.plot(lam_gal, stellar_comp, '-', c='g', alpha=.7, lw=0.7, 
                    label='Stellar component')
            
            # Third panel: emission lines and residuals
            residuals = spectrum - bestfit
            
            # Plot zero line
            ax3.axhline(0, color='k', lw=0.7, alpha=.5)
            
            # Plot residuals
            ax3.plot(lam_gal, residuals, 'g-', lw=0.8, alpha=.7, 
                    label='Residuals (data - full fit)')
            
            # Plot emission lines
            if np.any(gas_bestfit != 0):
                ax3.plot(lam_gal, gas_bestfit, 'r-', lw=1.2, alpha=0.8,
                      label='Gas component')
            
            # Define and plot spectral regions of interest
            spectral_regions = {
                'Hbeta': (4847.875, 4876.625),
                'Fe5015': (4977.750, 5054.000),
                'Mgb': (5160.125, 5192.625),
                '[OIII]': (4997, 5017)
            }
            
            # Mark spectral regions on all panels
            for name, (start, end) in spectral_regions.items():
                color = 'orange' if 'OIII' in name else 'lightgray'
                alpha = 0.3 if 'OIII' in name else 0.2
                for ax in [ax1, ax2, ax3]:
                    ax.axvspan(start, end, alpha=alpha, color=color)
                    # Add label at bottom
                    if ax == ax3:
                        y_pos = ax3.get_ylim()[0] + 0.1 * (ax3.get_ylim()[1] - ax3.get_ylim()[0])
                        ax.text((start + end)/2, y_pos, name, ha='center', va='bottom',
                              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Set all panel properties
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(4800, 5250)
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(axis='both', which='both', labelsize='x-small', 
                            right=True, top=True, direction='in')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize='small')
            
            # Set Y axis ranges
            y_min = np.min(spectrum) * 0.9
            y_max = np.max(spectrum) * 1.1
            ax1.set_ylim(y_min, y_max)
            ax2.set_ylim(y_min, y_max)
            
            # Set different Y axis range for third panel
            if np.any(gas_bestfit != 0):
                gas_max = np.max(np.abs(gas_bestfit)) * 3
                res_max = max(np.max(np.abs(residuals)), gas_max)
            else:
                res_max = np.max(np.abs(residuals)) * 3
            
            ax3.set_ylim(-res_max, res_max)
            
            # Set labels
            ax3.set_xlabel(r'Rest-frame Wavelength [$\AA$]', size=11)
            ax1.set_ylabel('Flux', size=11)
            ax2.set_ylabel('Flux', size=11)
            ax3.set_ylabel('Emission & Residuals', size=11)
            
            # Get bin position information
            x_pos = position.get('x', 0)
            y_pos = position.get('y', 0)
            n_pixels = position.get('n_pixels', 0)
            
            # Add title
            fig.suptitle(
                f"Bin {bin_id} - Two-stage Spectral Fit\n"
                f"Position: ({x_pos:.1f}, {y_pos:.1f}), {n_pixels} pixels\n"
                f"v={velocity:.1f} km/s, σ={sigma:.1f} km/s, χ²={chi2:.3f}", 
                fontsize=13
            )
            
            # Tight layout
            plt.tight_layout(rect=[0, 0, 1, 0.9])
            
            # Save image
            plt.savefig(plot_path_png, format='png', dpi=config.dpi, bbox_inches='tight')
            
            # Immediately close figure and release resources
            plt.close(fig)
            
            # Increment counter
            config.plot_count += 1
        
    except Exception as e:
        logging.error(f"Error plotting bin {bin_id} image: {str(e)}")
        # Ensure any failed figures are also closed
        plt.close('all')


### ------------------------------------------------- ###
# Radial Binning Implementation
### ------------------------------------------------- ###

class RadialBinning:
    """
    Radial binning-based spectral analysis class.
    
    Divides pixels into multiple radial rings based on distance from galaxy center,
    then performs a single spectral fit for each ring.
    """
    
    def __init__(self, galaxy_data, config):
        """
        Initialize radial binning.
        
        Parameters
        ----------
        galaxy_data : IFUDataCube
            Object containing galaxy data
        config : P2PConfig
            Configuration object
        """
        self.galaxy_data = galaxy_data
        self.config = config
        self.bin_data = None
        self.bin_results = {}
        
        # Arrays to store bin mapping and results
        ny, nx = galaxy_data.cube.shape[1:3]
        self.bin_map = np.full((ny, nx), -1)  # -1 indicates unbinned pixels
        self.rmap = np.full((ny, nx), np.nan)  # Radial distance map
        
        # Create results maps
        self.velfield = np.full((ny, nx), np.nan)
        self.sigfield = np.full((ny, nx), np.nan)
        
        # Save bin map to galaxy_data
        galaxy_data.bin_map = self.bin_map.copy()
        
        # Emission line and index maps
        self.el_flux_maps = {}
        self.el_snr_maps = {}
        self.index_maps = {}
        
        for name in config.gas_names:
            self.el_flux_maps[name] = np.full((ny, nx), np.nan)
            self.el_snr_maps[name] = np.full((ny, nx), np.nan)
            
        for name in config.line_indices:
            self.index_maps[name] = np.full((ny, nx), np.nan)
        
        # Radial binning specific parameters
        self.n_bins = 0
        self.bin_edges = None
        self.center_x = None
        self.center_y = None
        self.pa = 0.0  # Position angle (degrees)
        self.ellipticity = 0.0  # Ellipticity
    
    def create_bins(self, n_bins=10, min_radius=None, max_radius=None, 
                   center_x=None, center_y=None, pa=0.0, ellipticity=0.0,
                   log_spacing=True, snr_min=3.0, target_snr=None,
                   adaptive_bins=False):
        """
        Create radial bins.
        
        Parameters
        ----------
        n_bins : int
            Number of bins (if adaptive_bins=True, this is the maximum number)
        min_radius : float, optional
            Minimum radius (pixels)
        max_radius : float, optional
            Maximum radius (pixels)
        center_x : float, optional
            Center x coordinate
        center_y : float, optional
            Center y coordinate
        pa : float, optional
            Position angle (degrees)
        ellipticity : float, optional
            Ellipticity (0-1)
        log_spacing : bool, optional
            Whether to use logarithmic spacing (only when adaptive_bins=False)
        snr_min : float, optional
            Minimum SNR requirement
        target_snr : float, optional
            Target SNR (only when adaptive_bins=True)
        adaptive_bins : bool, optional
            Whether to use adaptive binning to balance SNR
            
        Returns
        -------
        int
            Number of bins created
        """
        if adaptive_bins:
            logging.info(f"===== Creating adaptive radial bins (target SNR={target_snr}) =====")
        else:
            logging.info(f"===== Creating uniform radial bins ({n_bins} rings) =====")
        
        # Get data dimensions
        ny, nx = self.galaxy_data.cube.shape[1:3]
        
        # Set center coordinates (if not provided)
        if center_x is None:
            center_x = nx / 2
        if center_y is None:
            center_y = ny / 2
            
        # Save parameters
        self.center_x = center_x
        self.center_y = center_y
        self.pa = pa
        self.ellipticity = ellipticity
        
        logging.info(f"Galaxy center: ({center_x:.1f}, {center_y:.1f}), PA: {pa:.1f}°, e: {ellipticity:.2f}")
        
        # Calculate radial distance map
        y_coords, x_coords = np.indices((ny, nx))
        x_diff = x_coords - center_x
        y_diff = y_coords - center_y
        
        # Apply position angle and ellipticity
        if ellipticity > 0 or pa != 0:
            # Convert position angle to radians
            pa_rad = np.radians(pa)
            
            # Rotate coordinate system to align major axis with PA
            x_rot = x_diff * np.cos(pa_rad) + y_diff * np.sin(pa_rad)
            y_rot = -x_diff * np.sin(pa_rad) + y_diff * np.cos(pa_rad)
            
            # Apply ellipticity
            b_to_a = 1 - ellipticity  # Minor-to-major axis ratio
            r_ell = np.sqrt((x_rot)**2 + (y_rot/b_to_a)**2)
        else:
            # Calculate simple Euclidean distance
            r_ell = np.sqrt(x_diff**2 + y_diff**2)
        
        # Save radial distance map
        self.rmap = r_ell
        
        # Determine radial range
        if min_radius is None:
            min_radius = 0.0
        if max_radius is None:
            max_radius = np.nanmax(r_ell)
        
        # Calculate SNR map
        if hasattr(self.galaxy_data, 'signal') and hasattr(self.galaxy_data, 'noise'):
            snr_map = np.zeros((ny, nx))
            for i in range(ny):
                for j in range(nx):
                    k_index = i * nx + j
                    if k_index < len(self.galaxy_data.signal):
                        signal = self.galaxy_data.signal[k_index]
                        noise = self.galaxy_data.noise[k_index]
                        snr_map[i, j] = signal / noise if noise > 0 else 0
        else:
            # If no SNR data, assume all pixels meet requirement
            snr_map = np.ones((ny, nx)) * snr_min * 2
        
        # Create valid pixel mask (apply SNR and radius constraints)
        valid_mask = (r_ell >= min_radius) & (r_ell <= max_radius) & (snr_map >= snr_min)
        
        if not np.any(valid_mask):
            logging.error("No valid pixels for radial binning")
            return 0
        
        # Create bins based on mode
        if adaptive_bins and target_snr is not None:
            # Adaptive mode: automatically adjust ring boundaries based on SNR
            return self._create_adaptive_bins(r_ell, valid_mask, snr_map, target_snr, 
                                             min_radius, max_radius, n_bins)
        else:
            # Equal width or logarithmic spacing mode
            if log_spacing:
                # Logarithmic spacing
                self.bin_edges = np.logspace(np.log10(max(min_radius, 0.5)), np.log10(max_radius), n_bins+1)
            else:
                # Linear spacing
                self.bin_edges = np.linspace(min_radius, max_radius, n_bins+1)
                
            return self._create_uniform_bins(r_ell, valid_mask, snr_map, n_bins)

    def _create_uniform_bins(self, r_ell, valid_mask, snr_map, n_bins):
        """
        Create uniform radial rings.
        
        Parameters
        ----------
        r_ell : ndarray
            Radial distance map
        valid_mask : ndarray
            Valid pixel mask
        snr_map : ndarray
            Signal-to-noise map
        n_bins : int
            Number of bins
            
        Returns
        -------
        int
            Number of bins created
        """
        logging.info(f"Radial range: {self.bin_edges[0]:.1f} - {self.bin_edges[-1]:.1f} pixels")
        logging.info(f"Bin edges: {self.bin_edges}")
        
        # Create bin mapping
        ny, nx = r_ell.shape
        bin_map = np.full((ny, nx), -1)
        
        # Assign pixels to bins
        for bin_id in range(n_bins):
            # Get inner and outer radius for this ring
            r_in = self.bin_edges[bin_id]
            r_out = self.bin_edges[bin_id+1]
            
            # Find all pixels that fall in this ring
            bin_mask = (r_ell >= r_in) & (r_ell < r_out) & valid_mask
            bin_map[bin_mask] = bin_id
            
            # Calculate average SNR for this ring
            bin_snr = np.mean(snr_map[bin_mask]) if np.any(bin_mask) else 0
            
            n_pixels = np.sum(bin_mask)
            logging.info(f"Ring {bin_id}: radius {r_in:.1f}-{r_out:.1f} pixels, contains {n_pixels} pixels, SNR~{bin_snr:.1f}")
            
            if n_pixels == 0:
                logging.warning(f"Ring {bin_id} contains no valid pixels")
        
        # Save bin mapping
        self.bin_map = bin_map
        self.galaxy_data.bin_map = bin_map.copy()
        self.n_bins = n_bins
        
        # Count binned pixels
        n_binned = np.sum(bin_map >= 0)
        n_total = ny * nx
        logging.info(f"Binning complete: {n_binned}/{n_total} pixels ({n_binned/n_total*100:.1f}%) assigned to {n_bins} rings")
        
        return n_bins

    def _create_adaptive_bins(self, r_ell, valid_mask, snr_map, target_snr, min_radius, max_radius, max_bins):
        """
        Create adaptive radial rings with roughly equal SNR.
        
        Parameters
        ----------
        r_ell : ndarray
            Radial distance map
        valid_mask : ndarray
            Valid pixel mask
        snr_map : ndarray
            Signal-to-noise map
        target_snr : float
            Target signal-to-noise ratio
        min_radius : float
            Minimum radius
        max_radius : float
            Maximum radius
        max_bins : int
            Maximum number of bins
            
        Returns
        -------
        int
            Number of bins created
        """
        logging.info(f"Adaptive binning: target SNR={target_snr}, max bins={max_bins}")
        
        # Get dimensions
        ny, nx = r_ell.shape
        
        # Create sorted index of valid pixels (by radial distance)
        valid_indices = np.where(valid_mask)
        radii = r_ell[valid_indices]
        snrs = snr_map[valid_indices]
        
        # Create array containing coordinates, radius and SNR
        pixel_data = np.array([(y, x, r, s) for y, x, r, s in 
                               zip(valid_indices[0], valid_indices[1], radii, snrs)],
                              dtype=[('y', int), ('x', int), ('r', float), ('snr', float)])
        
        # Sort by radius
        pixel_data.sort(order='r')
        
        # Adaptively create bins
        bin_map = np.full((ny, nx), -1)
        bin_edges = [min_radius]
        current_bin = 0
        start_idx = 0
        
        while start_idx < len(pixel_data) and current_bin < max_bins:
            # Start accumulating SNR
            cum_snr = 0
            cum_pixels = 0
            target_cum_snr = target_snr**2  # Need to accumulate SNR^2
            
            # Add pixels until target SNR reached or pixels exhausted
            end_idx = start_idx
            while end_idx < len(pixel_data) and cum_snr < target_cum_snr:
                y, x = pixel_data[end_idx]['y'], pixel_data[end_idx]['x']
                pixel_snr = pixel_data[end_idx]['snr']
                cum_snr += pixel_snr**2
                bin_map[y, x] = current_bin
                cum_pixels += 1
                end_idx += 1
                
                # If adding too many pixels but still not reaching target, force terminate
                if cum_pixels > len(pixel_data) // max_bins * 2:
                    break
            
            # Calculate actual SNR for this bin
            actual_snr = np.sqrt(cum_snr) if cum_pixels > 0 else 0
            
            # Record outer radius for this bin
            if end_idx < len(pixel_data):
                bin_edges.append(pixel_data[end_idx]['r'])
            else:
                bin_edges.append(max_radius)
            
            logging.info(f"Ring {current_bin}: radius {bin_edges[current_bin]:.1f}-{bin_edges[current_bin+1]:.1f} pixels, "
                        f"contains {cum_pixels} pixels, SNR={actual_snr:.1f}")
            
            # Advance to next bin
            start_idx = end_idx
            current_bin += 1
            
            # If no more pixels, exit loop
            if end_idx >= len(pixel_data):
                break
        
        # Save bin information
        self.bin_edges = np.array(bin_edges)
        self.bin_map = bin_map
        self.galaxy_data.bin_map = bin_map.copy()
        self.n_bins = current_bin
        
        # Count binned pixels
        n_binned = np.sum(bin_map >= 0)
        n_total = ny * nx
        logging.info(f"Adaptive binning complete: {n_binned}/{n_total} pixels ({n_binned/n_total*100:.1f}%) "
                   f"assigned to {current_bin} rings, target SNR={target_snr}")
        
        return current_bin
    
    def extract_bin_spectra(self, p2p_velfield=None, isap_velfield=None):
        """
        Extract combined spectra from each radial ring.
        
        Parameters
        ----------
        p2p_velfield : ndarray, optional
            P2P analysis velocity field, used for spectrum correction
        isap_velfield : ndarray, optional
            ISAP output velocity field, takes precedence over P2P field
            
        Returns
        -------
        dict
            Dictionary containing combined spectra
        """
        logging.info(f"===== Extracting combined spectra for {self.n_bins} radial rings =====")
        
        # Get data dimensions
        ny, nx = self.galaxy_data.cube.shape[1:3]
        npix = self.galaxy_data.spectra.shape[0]
        
        # First check if ISAP velocity field is provided
        if isap_velfield is not None:
            velfield = isap_velfield
            logging.info("Using ISAP velocity field for spectrum correction")
        # Then check if P2P velocity field is provided
        elif p2p_velfield is not None:
            velfield = p2p_velfield
            logging.info("Using P2P velocity field for spectrum correction")
        # Otherwise check if galaxy_data has velocity field
        elif hasattr(self.galaxy_data, 'velfield') and self.galaxy_data.velfield is not None:
            velfield = self.galaxy_data.velfield
            logging.info("Using galaxy_data velocity field for spectrum correction")
        else:
            # If no velocity field provided, try to automatically load P2P results
            p2p_vel_path, p2p_vel_exists = self.config.get_p2p_output_path("velfield")
            try:
                if p2p_vel_exists:
                    logging.info(f"Loading P2P velocity field: {p2p_vel_path}")
                    p2p_velfield = fits.getdata(p2p_vel_path)
                    if p2p_velfield is not None:
                        velfield = p2p_velfield
                        logging.info(f"Successfully loaded P2P velocity field, shape: {p2p_velfield.shape}")
            except Exception as e:
                logging.warning(f"Could not load P2P velocity field: {str(e)}")
                velfield = None
            
        # Create common wavelength grid (for resampling)
        lam_gal = self.galaxy_data.lam_gal
        
        # Initialize combined spectra dictionaries
        bin_spectra = {}
        bin_variances = {}
        bin_positions = {}
        
        # Extract spectrum for each bin
        for bin_id in range(self.n_bins):
            # Find all pixels belonging to this bin
            bin_mask = (self.bin_map == bin_id)
            
            if not np.any(bin_mask):
                logging.warning(f"Radial ring {bin_id} contains no pixels")
                continue
                
            # Get row and column indices for all pixels in this bin
            rows, cols = np.where(bin_mask)
            
            # Initialize accumulation variables
            coadded_spectrum = np.zeros(npix)
            coadded_variance = np.zeros(npix)
            total_weight = 0
            
            # Process each pixel
            for r, c in zip(rows, cols):
                k_index = r * nx + c
                
                # Get original spectrum
                pixel_spectrum = self.galaxy_data.spectra[:, k_index]
                
                # If variance data available, use it, otherwise create uniform variance
                if hasattr(self.galaxy_data, 'variance'):
                    pixel_variance = self.galaxy_data.variance[:, k_index]
                else:
                    pixel_variance = np.ones_like(pixel_spectrum)
                
                # Calculate weight for current pixel (using SNR)
                if hasattr(self.galaxy_data, 'signal') and hasattr(self.galaxy_data, 'noise'):
                    if k_index < len(self.galaxy_data.signal):
                        signal = self.galaxy_data.signal[k_index]
                        noise = self.galaxy_data.noise[k_index]
                        weight = (signal / noise)**2 if noise > 0 else 0
                    else:
                        weight = 1.0
                else:
                    weight = 1.0
                
                # Apply velocity correction (if available)
                if velfield is not None and r < velfield.shape[0] and c < velfield.shape[1] and not np.isnan(velfield[r, c]):
                    vel = velfield[r, c]
                    
                    # Shifted wavelength
                    lam_shifted = lam_gal * (1 + vel/self.config.c)
                    
                    # Resample to original wavelength grid
                    corrected_spectrum = np.interp(lam_gal, lam_shifted, pixel_spectrum,
                                                 left=0, right=0)
                    corrected_variance = np.interp(lam_gal, lam_shifted, pixel_variance,
                                                 left=np.inf, right=np.inf)
                    
                    # Accumulate corrected spectrum (weighted)
                    coadded_spectrum += corrected_spectrum * weight
                    coadded_variance += corrected_variance * weight**2
                else:
                    # No correction, accumulate directly
                    coadded_spectrum += pixel_spectrum * weight
                    coadded_variance += pixel_variance * weight**2
                
                total_weight += weight
            
            # Normalize accumulated spectrum
            if total_weight > 0:
                merged_spectrum = coadded_spectrum / total_weight
                merged_variance = coadded_variance / (total_weight**2)
            else:
                logging.warning(f"Ring {bin_id} has zero total weight, using simple average")
                merged_spectrum = coadded_spectrum / len(rows) if len(rows) > 0 else coadded_spectrum
                merged_variance = coadded_variance / (len(rows)**2) if len(rows) > 0 else coadded_variance
            
            # Store combined data
            bin_spectra[bin_id] = merged_spectrum
            bin_variances[bin_id] = merged_variance
            
            # Calculate average radius for the ring
            r_in = self.bin_edges[bin_id]
            r_out = self.bin_edges[bin_id+1]
            avg_radius = (r_in + r_out) / 2
            
            # Save bin position information
            bin_positions[bin_id] = {
                'radius': avg_radius,
                'r_in': r_in,
                'r_out': r_out,
                'n_pixels': len(rows)
            }
            
            # Add SNR information
            snr = np.median(merged_spectrum / np.sqrt(merged_variance))
            bin_positions[bin_id]['snr'] = snr
            
            # Log progress
            if bin_id % 5 == 0 or bin_id == self.n_bins - 1:
                logging.info(f"Extracted {bin_id+1}/{self.n_bins} radial ring spectra, "
                           f"radius={avg_radius:.1f}, SNR={snr:.1f}")
        
        # Save extracted data
        self.bin_data = {
            'spectra': bin_spectra,
            'variances': bin_variances,
            'positions': bin_positions
        }
        
        logging.info(f"Successfully extracted {len(bin_spectra)}/{self.n_bins} radial ring spectra")
        
        return self.bin_data
    
    def fit_bins(self, sps, gas_templates, gas_names, line_wave):
        """
        Fit combined spectra for each radial ring.
        
        Parameters
        ----------
        sps : object
            Stellar population synthesis library
        gas_templates : ndarray
            Gas emission line templates
        gas_names : array
            Gas emission line names
        line_wave : array
            Emission line wavelengths
            
        Returns
        -------
        dict
            Fitting results dictionary
        """
        logging.info(f"===== Starting fits for {self.n_bins} radial ring spectra (parallel mode={self.config.parallel_mode}) =====")
        
        if self.bin_data is None:
            logging.error("No bin data available")
            return {}
        
        # Prepare fitting parameters
        bin_ids = list(self.bin_data['spectra'].keys())
        
        # Use multiprocessing for parallel fitting
        start_time = time.time()
        results = {}
        
        # Radial rings usually have fewer bins, use smaller batch size
        rdb_batch_size = min(5, self.config.batch_size)
        
        # Choose processing method based on parallel mode
        if self.config.parallel_mode == 'grouped':
            # Memory optimization: process bins in batches
            for batch_start in range(0, len(bin_ids), rdb_batch_size):
                batch_end = min(batch_start + rdb_batch_size, len(bin_ids))
                batch_bins = bin_ids[batch_start:batch_end]
                
                logging.info(f"Processing batch {batch_start//rdb_batch_size + 1}/{(len(bin_ids)-1)//rdb_batch_size + 1} "
                            f"(rings {batch_start+1}-{batch_end})")
                
                with ProcessPoolExecutor(max_workers=self.config.n_threads) as executor:
                    # Submit batch tasks
                    future_to_bin = {}
                    for bin_id in batch_bins:
                        # Prepare parameters
                        spectrum = self.bin_data['spectra'][bin_id]
                        position = self.bin_data['positions'][bin_id]
                        
                        # Create simulated single-pixel input
                        args = (bin_id, -1, self.galaxy_data, sps, gas_templates, gas_names, line_wave, self.config)
                        args[2].spectra = np.column_stack([spectrum])  # Replace with bin spectrum
                        
                        # Submit task
                        future = executor.submit(fit_radial_bin, args)
                        future_to_bin[future] = bin_id
                    
                    # Process results
                    with tqdm(total=len(batch_bins), desc=f"Batch {batch_start//rdb_batch_size + 1}") as pbar:
                        for future in as_completed(future_to_bin):
                            bin_id, result = future.result()
                            if result is not None:
                                results[bin_id] = result
                            pbar.update(1)
                
                # Force garbage collection
                import gc
                gc.collect()
        
        else:  # global mode
            logging.info(f"Using global parallel mode for all {len(bin_ids)} rings")
            
            with ProcessPoolExecutor(max_workers=self.config.n_threads) as executor:
                # Submit all tasks
                future_to_bin = {}
                for bin_id in bin_ids:
                    # Prepare parameters
                    spectrum = self.bin_data['spectra'][bin_id]
                    position = self.bin_data['positions'][bin_id]
                    
                    # Create simulated single-pixel input
                    args = (bin_id, -1, self.galaxy_data, sps, gas_templates, gas_names, line_wave, self.config)
                    args[2].spectra = np.column_stack([spectrum])  # Replace with bin spectrum
                    
                    # Submit task
                    future = executor.submit(fit_radial_bin, args)
                    future_to_bin[future] = bin_id
                
                # Process results
                with tqdm(total=len(bin_ids), desc="Processing rings") as pbar:
                    for future in as_completed(future_to_bin):
                        bin_id, result = future.result()
                        if result is not None:
                            results[bin_id] = result
                        pbar.update(1)
        
        # Calculate completion time
        end_time = time.time()
        successful = len(results)
        logging.info(f"Completed {successful}/{self.n_bins} radial ring fits in {end_time - start_time:.1f} seconds")
        
        # Save results
        self.bin_results = results
        
        return results
    
    def process_results(self):
        """
        Process fitting results and populate maps.
        
        Returns
        -------
        dict
            Processed results dictionary
        """
        logging.info(f"===== Processing results for {len(self.bin_results)} radial rings =====")
        
        if not self.bin_results:
            logging.error("No fitting results available")
            return {}
        
        # Get data dimensions
        ny, nx = self.galaxy_data.cube.shape[1:3]
        
        # Initialize results arrays
        velfield = np.full((ny, nx), np.nan)
        sigfield = np.full((ny, nx), np.nan)
        
        # Initialize emission line and index maps
        el_flux_maps = {}
        el_snr_maps = {}
        index_maps = {}
        
        for name in self.config.gas_names:
            el_flux_maps[name] = np.full((ny, nx), np.nan)
            el_snr_maps[name] = np.full((ny, nx), np.nan)
            
        for name in self.config.line_indices:
            index_maps[name] = np.full((ny, nx), np.nan)
        
        # Process results for each bin
        for bin_id, result in self.bin_results.items():
            if not result.get('success', False):
                continue
                
            # Extract result data
            velocity = result['velocity']
            sigma = result['sigma']
            
            # Extract emission line data
            el_results = result.get('el_results', {})
            
            # Extract index data
            indices = result.get('indices', {})
            
            # Find all pixels belonging to this bin
            bin_mask = (self.bin_map == bin_id)
            
            # Populate velocity and dispersion maps
            velfield[bin_mask] = velocity
            sigfield[bin_mask] = sigma
            
            # Populate emission line maps
            for name, data in el_results.items():
                if name in el_flux_maps:
                    el_flux_maps[name][bin_mask] = data['flux']
                    el_snr_maps[name][bin_mask] = data['an']
            
            # Populate index maps
            for name, value in indices.items():
                if name in index_maps:
                    index_maps[name][bin_mask] = value
        
        # Save processed maps
        self.velfield = velfield
        self.sigfield = sigfield
        self.el_flux_maps = el_flux_maps
        self.el_snr_maps = el_snr_maps
        self.index_maps = index_maps
        
        # Also update maps in galaxy_data
        self.galaxy_data.velfield = velfield.copy()
        self.galaxy_data.sigfield = sigfield.copy()
        
        for name in self.config.gas_names:
            self.galaxy_data.el_flux_maps[name] = el_flux_maps[name].copy()
            self.galaxy_data.el_snr_maps[name] = el_snr_maps[name].copy()
            
        for name in self.config.line_indices:
            self.galaxy_data.index_maps[name] = index_maps[name].copy()
        
        # Create CSV summary
        self.create_bin_summary()
        
        return {
            'velfield': velfield,
            'sigfield': sigfield,
            'el_flux_maps': el_flux_maps,
            'el_snr_maps': el_snr_maps,
            'index_maps': index_maps
        }
    
    def create_bin_summary(self):
        """
        Create bin results summary.
        
        Returns
        -------
        DataFrame
            Results summary
        """
        # Create data records list
        data = []
        
        for bin_id, result in self.bin_results.items():
            if not result.get('success', False):
                continue
                
            # Get bin position
            position = self.bin_data['positions'][bin_id]
            n_pixels = position['n_pixels']
            radius = position['radius']
            r_in = position['r_in']
            r_out = position['r_out']
            
            # Create basic record
            record = {
                'bin_id': bin_id,
                'radius': radius,
                'r_in': r_in,
                'r_out': r_out,
                'n_pixels': n_pixels,
                'velocity': result['velocity'],
                'sigma': result['sigma'],
                'snr': result['snr']
            }
            
            # Add emission line data
            for name, data_dict in result.get('el_results', {}).items():
                record[f'{name}_flux'] = data_dict['flux']
                record[f'{name}_snr'] = data_dict['an']
            
            # Add index data
            for name, value in result.get('indices', {}).items():
                record[f'{name}_index'] = value
            
            data.append(record)
        
        # Create DataFrame
        if data:
            import pandas as pd
            df = pd.DataFrame(data)
            
            # Save CSV file
            csv_path = self.config.output_dir / f"{self.config.galaxy_name}_RDB_bins.csv"
            df.to_csv(csv_path, index=False)
            logging.info(f"Radial ring summary saved to {csv_path}")
            
            return df
        else:
            logging.warning("No ring results available to create summary")
            return None
    
    def plot_binning(self):
        """
        Plot radial binning results.
        
        Returns
        -------
        None
        """
        if self.config.no_plots:
            return
            
        # If no bins successfully created, return
        if not hasattr(self, 'n_bins') or self.n_bins == 0:
            logging.warning("No bins available to plot")
            return
        
        try:
            with plt.rc_context({'figure.max_open_warning': False}):
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8), dpi=self.config.dpi)
                
                # Use different colors to display different radial rings
                cmap = plt.cm.get_cmap('viridis', self.n_bins)
                im = ax.imshow(self.bin_map, origin='lower', cmap=cmap, 
                              vmin=-0.5, vmax=self.n_bins-0.5)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Radial Bin')
                
                # Draw circular markers for boundaries
                if self.ellipticity < 0.05:  # Nearly circular
                    # Draw circular boundaries
                    for r in self.bin_edges:
                        circle = plt.Circle((self.center_x, self.center_y), r, 
                                          fill=False, edgecolor='red', linestyle='--', alpha=0.7)
                        ax.add_patch(circle)
                else:
                    # Draw elliptical boundaries
                    for r in self.bin_edges:
                        # Ellipse semi-major and semi-minor axes
                        a = r
                        b = r * (1 - self.ellipticity)
                        
                        # Convert position angle to radians
                        pa_rad = np.radians(self.pa)
                        
                        ellipse = plt.matplotlib.patches.Ellipse(
                            (self.center_x, self.center_y), 2*a, 2*b,
                            angle=self.pa, fill=False, edgecolor='red', linestyle='--', alpha=0.7
                        )
                        ax.add_patch(ellipse)
                
                # Mark center
                ax.plot(self.center_x, self.center_y, 'r+', markersize=10)
                
                # Labels and title
                ax.set_xlabel('X [pixels]')
                ax.set_ylabel('Y [pixels]')
                ax.set_title(f"{self.config.galaxy_name} - Radial Binning: {self.n_bins} rings")
                
                # Tight layout
                plt.tight_layout()
                
                # Save
                plot_path = self.config.plot_dir / f"{self.config.galaxy_name}_radial_bins.png"
                plt.savefig(plot_path, dpi=self.config.dpi)
                plt.close(fig)
                
                logging.info(f"Radial binning plot saved to {plot_path}")
        except Exception as e:
            logging.error(f"Error plotting binning: {str(e)}")
            plt.close('all')  # Ensure all figures are closed
    
    def plot_bin_results(self, bin_id):
        """
        Plot fitting results for a single radial ring.
        
        Parameters
        ----------
        bin_id : int
            Bin ID
            
        Returns
        -------
        None
        """
        if self.config.no_plots or self.config.plot_count >= self.config.max_plots:
            return
            
        if bin_id not in self.bin_results:
            logging.warning(f"No fitting results available for radial ring {bin_id}")
            return
            
        result = self.bin_results[bin_id]
        if not result.get('success', False):
            logging.warning(f"Fitting unsuccessful for radial ring {bin_id}")
            return
            
        try:
            # Get bin position information
            position = self.bin_data['positions'][bin_id]
            
            # Get pp object
            pp = result.get('pp_obj')
            if pp is None:
                logging.warning(f"No pp object available for radial ring {bin_id}")
                return
                
            # Set pp's additional attributes for plot_bin_fit function
            if hasattr(result, 'stage1_bestfit'):
                pp.stage1_bestfit = result['stage1_bestfit'] 
            else:
                pp.stage1_bestfit = pp.bestfit
                
            pp.optimal_stellar_template = result['optimal_template']
            pp.full_bestfit = result['bestfit']
            pp.full_gas_bestfit = result['gas_bestfit']
            
            # Call plotting function - use the same function as VNB but with RDB mode
            plot_radial_bin_fit(bin_id, self.galaxy_data, pp, position, self.config)
            
            # Increment counter
            self.config.plot_count += 1
        except Exception as e:
            logging.error(f"Error plotting radial ring {bin_id} results: {str(e)}")
            plt.close('all')
    
    def plot_radial_profiles(self):
        """
        Plot radial profiles.
        
        Returns
        -------
        None
        """
        if self.config.no_plots:
            return
            
        # If no bins successfully created, return
        if not self.bin_results:
            logging.warning("No fitting results available to plot radial profiles")
            return
        
        try:
            # Prepare data
            radii = []
            velocities = []
            velocity_errs = []
            sigmas = []
            sigma_errs = []
            
            # Index data
            index_data = {name: [] for name in self.config.line_indices}
            
            # Emission line data
            flux_data = {name: [] for name in self.config.gas_names}
            
            # Extract data
            for bin_id, result in sorted(self.bin_results.items()):
                if not result.get('success', False):
                    continue
                    
                # Get bin radius
                radius = self.bin_data['positions'][bin_id]['radius']
                radii.append(radius)
                
                # Velocity and dispersion
                velocities.append(result['velocity'])
                velocity_errs.append(10.0)  # Assumed error
                sigmas.append(result['sigma'])
                sigma_errs.append(10.0)  # Assumed error
                
                # Indices
                for name in self.config.line_indices:
                    if name in result.get('indices', {}):
                        index_data[name].append(result['indices'][name])
                    else:
                        index_data[name].append(np.nan)
                
                # Emission line fluxes
                for name in self.config.gas_names:
                    if name in result.get('el_results', {}):
                        flux_data[name].append(result['el_results'][name]['flux'])
                    else:
                        flux_data[name].append(np.nan)
            
            with plt.rc_context({'figure.max_open_warning': False}):
                # Create velocity and dispersion plots
                fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=self.config.dpi, sharex=True)
                
                # Plot velocity profile
                axes[0].errorbar(radii, velocities, yerr=velocity_errs, fmt='o-', capsize=3)
                axes[0].set_ylabel('Velocity [km/s]')
                axes[0].set_title('Radial Velocity Profile')
                axes[0].grid(True, alpha=0.3)
                
                # Plot dispersion profile
                axes[1].errorbar(radii, sigmas, yerr=sigma_errs, fmt='o-', capsize=3)
                axes[1].set_ylabel('Velocity Dispersion [km/s]')
                axes[1].set_xlabel('Radius [pixels]')
                axes[1].set_title('Radial Velocity Dispersion Profile')
                axes[1].grid(True, alpha=0.3)
                
                plt.suptitle(f"{self.config.galaxy_name} - Kinematic Radial Profiles")
                plt.tight_layout()
                plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_RDB_kinematic_profiles.png", dpi=self.config.dpi)
                plt.close(fig)
                
                # Create indices profile plot
                if self.config.compute_spectral_indices and len(self.config.line_indices) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6), dpi=self.config.dpi)
                    
                    # Plot radial profile for each index
                    for name in self.config.line_indices:
                        ax.plot(radii, index_data[name], 'o-', label=name)
                    
                    ax.set_xlabel('Radius [pixels]')
                    ax.set_ylabel('Index Value [Å]')
                    ax.set_title(f"{self.config.galaxy_name} - Spectral Indices Radial Profiles")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_RDB_indices_profiles.png", dpi=self.config.dpi)
                    plt.close(fig)
                
                # Create emission line profile plot
                if self.config.compute_emission_lines and len(self.config.gas_names) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6), dpi=self.config.dpi)
                    
                    # Plot radial profile for each emission line
                    for name in self.config.gas_names:
                        ax.plot(radii, flux_data[name], 'o-', label=name)
                    
                    ax.set_xlabel('Radius [pixels]')
                    ax.set_ylabel('Flux')
                    ax.set_title(f"{self.config.galaxy_name} - Emission Line Flux Radial Profiles")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_RDB_emission_profiles.png", dpi=self.config.dpi)
                    plt.close(fig)
                
                logging.info(f"Radial profile plots saved to {self.config.plot_dir}")
        except Exception as e:
            logging.error(f"Error plotting radial profiles: {str(e)}")
            plt.close('all')
    
    def create_summary_plots(self):
        """
        Create RDB results summary plots.
        
        Returns
        -------
        None
        """
        if self.config.no_plots:
            return
            
        try:
            # Create plots directory
            os.makedirs(self.config.plot_dir, exist_ok=True)
            
            # 1. Binning plot
            self.plot_binning()
            
            # 2. Radial profiles plot
            self.plot_radial_profiles()
            
            # 3. Kinematics plots
            with plt.rc_context({'figure.max_open_warning': False}):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.config.dpi)
                
                # Velocity map
                vmax = np.nanpercentile(np.abs(self.velfield), 90)
                im0 = axes[0].imshow(self.velfield, origin='lower', cmap='RdBu_r', 
                                  vmin=-vmax, vmax=vmax)
                axes[0].set_title('Velocity [km/s]')
                plt.colorbar(im0, ax=axes[0])
                
                # Velocity dispersion map
                sigma_max = np.nanpercentile(self.sigfield, 95)
                im1 = axes[1].imshow(self.sigfield, origin='lower', cmap='viridis', 
                                  vmin=0, vmax=sigma_max)
                axes[1].set_title('Velocity Dispersion [km/s]')
                plt.colorbar(im1, ax=axes[1])
                
                plt.suptitle(f"{self.config.galaxy_name} - RDB Stellar Kinematics")
                plt.tight_layout()
                plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_RDB_kinematics.png", dpi=self.config.dpi)
                plt.close(fig)
            
            # 4. Emission line plots
            if self.config.compute_emission_lines and len(self.config.gas_names) > 0:
                n_lines = len(self.config.gas_names)
                
                with plt.rc_context({'figure.max_open_warning': False}):
                    fig, axes = plt.subplots(2, n_lines, figsize=(4*n_lines, 8), dpi=self.config.dpi)
                    
                    if n_lines == 1:  # Handle single emission line case
                        axes = np.array([[axes[0]], [axes[1]]])
                    
                    for i, name in enumerate(self.config.gas_names):
                        # Flux map
                        flux_map = self.el_flux_maps[name]
                        vmax = np.nanpercentile(flux_map, 95)
                        im = axes[0, i].imshow(flux_map, origin='lower', cmap='inferno', vmin=0, vmax=vmax)
                        axes[0, i].set_title(f"{name} Flux")
                        plt.colorbar(im, ax=axes[0, i])
                        
                        # SNR map
                        snr_map = self.el_snr_maps[name]
                        im = axes[1, i].imshow(snr_map, origin='lower', cmap='viridis', vmin=0, vmax=5)
                        axes[1, i].set_title(f"{name} S/N")
                        plt.colorbar(im, ax=axes[1, i])
                    
                    plt.suptitle(f"{self.config.galaxy_name} - RDB Emission Lines")
                    plt.tight_layout()
                    plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_RDB_emission_lines.png", dpi=self.config.dpi)
                    plt.close(fig)
            
            # 5. Spectral indices plots
            if self.config.compute_spectral_indices and len(self.config.line_indices) > 0:
                n_indices = len(self.config.line_indices)
                n_cols = min(3, n_indices)
                n_rows = (n_indices + n_cols - 1) // n_cols
                
                with plt.rc_context({'figure.max_open_warning': False}):
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), dpi=self.config.dpi)
                    axes = np.atleast_2d(axes)
                    
                    for i, name in enumerate(self.config.line_indices):
                        row = i // n_cols
                        col = i % n_cols
                        
                        index_map = self.index_maps[name]
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
                    
                    plt.suptitle(f"{self.config.galaxy_name} - RDB Spectral Indices")
                    plt.tight_layout()
                    plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_RDB_indices.png", dpi=self.config.dpi)
                    plt.close(fig)
            
            # Force cleanup
            plt.close('all')
            import gc
            gc.collect()
        
        except Exception as e:
            logging.error(f"Error creating RDB summary plots: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
            plt.close('all')
    
    def save_results_to_fits(self):
        """
        Save RDB results to FITS files.
        
        Returns
        -------
        None
        """
        try:
            # Create header
            hdr = fits.Header()
            hdr['OBJECT'] = self.config.galaxy_name
            hdr['REDSHIFT'] = self.config.redshift
            hdr['CD1_1'] = self.galaxy_data.CD1_1
            hdr['CD1_2'] = self.galaxy_data.CD1_2
            hdr['CD2_1'] = self.galaxy_data.CD2_1
            hdr['CD2_2'] = self.galaxy_data.CD2_2
            hdr['CRVAL1'] = self.galaxy_data.CRVAL1
            hdr['CRVAL2'] = self.galaxy_data.CRVAL2
            
            # Add RDB information
            hdr['BINTYPE'] = 'RDB'
            hdr['NBINS'] = self.n_bins
            hdr['CENTERX'] = self.center_x
            hdr['CENTERY'] = self.center_y
            hdr['PA'] = self.pa
            hdr['ELLIP'] = self.ellipticity
            hdr['PARMODE'] = self.config.parallel_mode
            
            # Save bin map and radial distance map
            hdu_binmap = fits.PrimaryHDU(self.bin_map, header=hdr)
            hdu_binmap.header['CONTENT'] = 'Radial bin map'
            hdu_binmap.writeto(self.config.get_output_filename("binmap", "RDB"), overwrite=True)
            
            hdu_rmap = fits.PrimaryHDU(self.rmap, header=hdr)
            hdu_rmap.header['CONTENT'] = 'Radial distance map'
            hdu_rmap.writeto(self.config.get_output_filename("radiusmap", "RDB"), overwrite=True)
            
            # Save velocity field
            hdu_vel = fits.PrimaryHDU(self.velfield, header=hdr)
            hdu_vel.header['CONTENT'] = 'Stellar velocity field (RDB)'
            hdu_vel.header['BUNIT'] = 'km/s'
            hdu_vel.writeto(self.config.get_output_filename("velfield", "RDB"), overwrite=True)
            
            # Save velocity dispersion field
            hdu_sig = fits.PrimaryHDU(self.sigfield, header=hdr)
            hdu_sig.header['CONTENT'] = 'Stellar velocity dispersion (RDB)'
            hdu_sig.header['BUNIT'] = 'km/s'
            hdu_sig.writeto(self.config.get_output_filename("sigfield", "RDB"), overwrite=True)
            
            # Save emission line maps
            for name in self.config.gas_names:
                if name in self.el_flux_maps:
                    hdu = fits.PrimaryHDU(self.el_flux_maps[name], header=hdr)
                    hdu.header['CONTENT'] = f'{name} emission line flux (RDB)'
                    hdu.header['BUNIT'] = 'flux units'
                    hdu.writeto(self.config.get_output_filename(f"{name}_flux", "RDB"), overwrite=True)
                    
                    hdu = fits.PrimaryHDU(self.el_snr_maps[name], header=hdr)
                    hdu.header['CONTENT'] = f'{name} emission line S/N (RDB)'
                    hdu.header['BUNIT'] = 'ratio'
                    hdu.writeto(self.config.get_output_filename(f"{name}_snr", "RDB"), overwrite=True)
            
            # Save spectral index maps
            for name in self.config.line_indices:
                if name in self.index_maps:
                    hdu = fits.PrimaryHDU(self.index_maps[name], header=hdr)
                    hdu.header['CONTENT'] = f'{name} spectral index (RDB)'
                    hdu.header['BUNIT'] = 'Angstrom'
                    hdu.writeto(self.config.get_output_filename(f"{name}_index", "RDB"), overwrite=True)
            
            logging.info(f"RDB results saved to FITS files")
            
        except Exception as e:
            logging.error(f"Error saving RDB results to FITS: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())


def fit_radial_bin(args):
    """
    Fit a single radial ring's combined spectrum. Uses same logic as fit_bin but optimized for radial rings.
    
    Parameters
    ----------
    args : tuple
        (bin_id, _, galaxy_data, sps, gas_templates, gas_names, line_wave, config)
        
    Returns
    -------
    tuple
        (bin_id, results_dict or None)
    """
    # Directly call fit_bin function, but use different log identifier
    bin_id, _, galaxy_data, sps, gas_templates, gas_names, line_wave, config = args
    
    logging.debug(f"===== FITTING RADIAL BIN {bin_id} =====")
    
    # Call generic fitting function
    result = fit_bin(args)
    
    if result[1] is not None:
        logging.debug(f"===== RADIAL BIN {bin_id} FIT COMPLETED SUCCESSFULLY =====")
    
    return result


def plot_radial_bin_fit(bin_id, galaxy_data, pp, position, config):
    """
    Create diagnostic plot for radial ring fitting - memory optimized version
    
    Parameters
    ----------
    bin_id : int
        Bin ID
    galaxy_data : IFUDataCube
        Object containing galaxy data
    pp : ppxf object
        pPXF fitting result
    position : dict
        Dictionary containing bin position information
    config : P2PConfig
        Configuration object
    """
    # If all plots disabled, return immediately
    if config.no_plots or config.plot_count >= config.max_plots:
        return
    
    try:
        # Create plots directory
        plot_dir = config.plot_dir / 'RDB_res'
        os.makedirs(plot_dir, exist_ok=True)
        
        # Prepare filename and path
        plot_path_png = plot_dir / f"{config.galaxy_name}_ring_{bin_id}.png"
        
        # Get data
        lam_gal = galaxy_data.lam_gal
        
        # Here, we assume pp's spectra has been replaced with bin spectrum
        # So directly use first column of spectrum data
        spectrum = galaxy_data.spectra[:, 0]
        
        # Get fitting results
        bestfit = pp.full_bestfit if hasattr(pp, 'full_bestfit') else pp.bestfit
        stage1_bestfit = pp.stage1_bestfit if hasattr(pp, 'stage1_bestfit') else bestfit
        gas_bestfit = pp.full_gas_bestfit if hasattr(pp, 'full_gas_bestfit') else np.zeros_like(spectrum)
        
        # Extract needed attribute values
        velocity = to_scalar(pp.sol[0]) if hasattr(pp, 'sol') and pp.sol is not None and len(pp.sol) > 0 else 0.0
        sigma = to_scalar(pp.sol[1]) if hasattr(pp, 'sol') and pp.sol is not None and len(pp.sol) > 1 else 0.0
        chi2 = to_scalar(pp.chi2) if hasattr(pp, 'chi2') and pp.chi2 is not None else 0.0
        
        # Use with statement to create figure, ensuring resources are properly released
        with plt.rc_context({'figure.max_open_warning': False}):
            # Create figure, specify lower DPI to reduce memory usage
            fig = plt.figure(figsize=(12, 8), dpi=config.dpi)
            
            # Create subplots
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax3 = plt.subplot(gs[2])
            
            # First panel: original data and first stage fit
            ax1.plot(lam_gal, spectrum, c='k', lw=1, alpha=.8, 
                    label=f"{config.galaxy_name} ring:{bin_id} - Original")
            ax1.plot(lam_gal, stage1_bestfit, '-', c='r', alpha=.8, 
                    label='Stellar component fit')
            
            # Second panel: final fit result
            ax2.plot(lam_gal, spectrum, c='k', lw=1, alpha=.8, 
                    label='Original spectrum')
            ax2.plot(lam_gal, bestfit, '-', c='r', alpha=.8, 
                    label='Full fit')
            
            # Plot stellar component (full fit minus gas)
            stellar_comp = bestfit - gas_bestfit
            ax2.plot(lam_gal, stellar_comp, '-', c='g', alpha=.7, lw=0.7, 
                    label='Stellar component')
            
            # Third panel: emission lines and residuals
            residuals = spectrum - bestfit
            
            # Plot zero line
            ax3.axhline(0, color='k', lw=0.7, alpha=.5)
            
            # Plot residuals
            ax3.plot(lam_gal, residuals, 'g-', lw=0.8, alpha=.7, 
                    label='Residuals (data - full fit)')
            
            # Plot emission lines
            if np.any(gas_bestfit != 0):
                ax3.plot(lam_gal, gas_bestfit, 'r-', lw=1.2, alpha=0.8,
                      label='Gas component')
            
            # Define and plot spectral regions of interest
            spectral_regions = {
                'Hbeta': (4847.875, 4876.625),
                'Fe5015': (4977.750, 5054.000),
                'Mgb': (5160.125, 5192.625),
                '[OIII]': (4997, 5017)
            }
            
            # Mark spectral regions on all panels
            for name, (start, end) in spectral_regions.items():
                color = 'orange' if 'OIII' in name else 'lightgray'
                alpha = 0.3 if 'OIII' in name else 0.2
                for ax in [ax1, ax2, ax3]:
                    ax.axvspan(start, end, alpha=alpha, color=color)
                    # Add label at bottom
                    if ax == ax3:
                        y_pos = ax3.get_ylim()[0] + 0.1 * (ax3.get_ylim()[1] - ax3.get_ylim()[0])
                        ax.text((start + end)/2, y_pos, name, ha='center', va='bottom',
                              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Set all panel properties
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(4800, 5250)
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(axis='both', which='both', labelsize='x-small', 
                            right=True, top=True, direction='in')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize='small')
            
            # Set Y axis ranges
            y_min = np.min(spectrum) * 0.9
            y_max = np.max(spectrum) * 1.1
            ax1.set_ylim(y_min, y_max)
            ax2.set_ylim(y_min, y_max)
            
            # Set different Y axis range for third panel
            if np.any(gas_bestfit != 0):
                gas_max = np.max(np.abs(gas_bestfit)) * 3
                res_max = max(np.max(np.abs(residuals)), gas_max)
            else:
                res_max = np.max(np.abs(residuals)) * 3
            
            ax3.set_ylim(-res_max, res_max)
            
            # Set labels
            ax3.set_xlabel(r'Rest-frame Wavelength [$\AA$]', size=11)
            ax1.set_ylabel('Flux', size=11)
            ax2.set_ylabel('Flux', size=11)
            ax3.set_ylabel('Emission & Residuals', size=11)
            
            # Get radial ring information
            radius = position.get('radius', 0)
            r_in = position.get('r_in', 0)
            r_out = position.get('r_out', 0)
            n_pixels = position.get('n_pixels', 0)
            
            # Add title
            fig.suptitle(
                f"Radial Ring {bin_id} - Two-stage Spectral Fit\n"
                f"Radius: {radius:.1f} pixels ({r_in:.1f}-{r_out:.1f}), {n_pixels} pixels\n"
                f"v={velocity:.1f} km/s, σ={sigma:.1f} km/s, χ²={chi2:.3f}", 
                fontsize=13
            )
            
            # Tight layout
            plt.tight_layout(rect=[0, 0, 1, 0.9])
            
            # Save image
            plt.savefig(plot_path_png, format='png', dpi=config.dpi, bbox_inches='tight')
            
            # Immediately close figure and release resources
            plt.close(fig)
            
            # Increment counter
            config.plot_count += 1
        
    except Exception as e:
        logging.error(f"Error plotting radial ring {bin_id} image: {str(e)}")
        # Ensure any failed figures are also closed
        plt.close('all')


### ------------------------------------------------- ###
# Analysis Runner Functions
### ------------------------------------------------- ###

def run_vnb_analysis(config, target_snr=20):
    """
    Run complete Voronoi binning analysis workflow.
    
    Parameters
    ----------
    config : P2PConfig
        Configuration object
    target_snr : float, optional
        Target signal-to-noise ratio, default is 20
        
    Returns
    -------
    tuple
        (galaxy_data, vnb)
    """
    logging.info(f"===== Starting Voronoi Binning Analysis (SNR={target_snr}, parallel mode={config.parallel_mode}) =====")
    
    # Start timing
    start_time = time.time()
    
    try:
        # 1. Load data
        logging.info("Loading data...")
        galaxy_data = IFUDataCube(config.get_data_path(), config.lam_range_temp, config.redshift, config)
        
        # 2. Prepare templates
        logging.info("Preparing stellar and gas templates...")
        sps, gas_templates, gas_names, line_wave = prepare_templates(config, galaxy_data.velscale)
        
        # 3. Initialize Voronoi binning
        vnb = VoronoiBinning(galaxy_data, config)
        
        # 4. Create bins
        n_bins = vnb.create_bins(target_snr=target_snr)
        if n_bins == 0:
            logging.error("Could not create Voronoi bins")
            return galaxy_data, vnb
        
        # 5. Try loading P2P velocity field for spectrum correction
        p2p_velfield = None
        p2p_path, p2p_exists = config.get_p2p_output_path("velfield")
        try:
            if p2p_exists:
                logging.info(f"Loading P2P velocity field: {p2p_path}")
                p2p_velfield = fits.getdata(p2p_path)
                logging.info(f"Successfully loaded P2P velocity field, shape: {p2p_velfield.shape}")
        except Exception as e:
            logging.warning(f"Could not load P2P velocity field: {str(e)}")
            p2p_velfield = None
            
        # Try to load ISAP velocity field if specified
        isap_velfield = None
        if config.use_isap and config.isap_file:
            try:
                logging.info(f"Loading ISAP velocity field: {config.isap_file}")
                isap_data, _ = read_isap_data(config.isap_file, 'velocity')
                if isap_data is not None:
                    isap_velfield = isap_data
                    logging.info(f"Successfully loaded ISAP velocity field, shape: {isap_velfield.shape}")
            except Exception as e:
                logging.warning(f"Could not load ISAP velocity field: {str(e)}")
            
        # Extract bin spectra
        bin_data = vnb.extract_bin_spectra(p2p_velfield, isap_velfield)
        
        # 6. Fit bins
        bin_results = vnb.fit_bins(sps, gas_templates, gas_names, line_wave)
        
        # 7. Process results
        vnb.process_results()
        
        # 8. Create summary plots
        if config.make_plots and not config.no_plots:
            logging.info("Creating summary plots...")
            vnb.create_summary_plots()
            
            # Create diagnostic plots for a few sample bins
            logging.info("Creating diagnostic plots for sample bins...")
            for bin_id in range(min(5, n_bins)):
                if bin_id in vnb.bin_results:
                    vnb.plot_bin_results(bin_id)
        
        # 9. Save results to FITS files
        logging.info("Saving results to FITS files...")
        vnb.save_results_to_fits()
        
        # Calculate completion time
        end_time = time.time()
        logging.info(f"VNB analysis completed in {end_time - start_time:.1f} seconds")
        
        return galaxy_data, vnb
        
    except Exception as e:
        logging.error(f"Error in VNB analysis: {str(e)}")
        logging.exception("Stack trace:")
        raise


def run_rdb_analysis(config, n_bins=10, center_x=None, center_y=None, 
                    pa=0.0, ellipticity=0.0, log_spacing=True,
                    adaptive_bins=False, target_snr=None):
    """
    Run complete radial binning analysis workflow.
    
    Parameters
    ----------
    config : P2PConfig
        Configuration object
    n_bins : int, optional
        Number of radial rings, default is 10
    center_x : float, optional
        Center x coordinate
    center_y : float, optional
        Center y coordinate
    pa : float, optional
        Position angle (degrees)
    ellipticity : float, optional
        Ellipticity (0-1)
    log_spacing : bool, optional
        Whether to use logarithmic spacing
    adaptive_bins : bool, optional
        Whether to use adaptive binning to balance SNR
    target_snr : float, optional
        Target SNR (only when adaptive_bins=True)
        
    Returns
    -------
    tuple
        (galaxy_data, rdb)
    """
    if adaptive_bins:
        logging.info(f"===== Starting adaptive radial binning analysis (target SNR={target_snr}, parallel mode={config.parallel_mode}) =====")
    else:
        logging.info(f"===== Starting uniform radial binning analysis (rings={n_bins}, parallel mode={config.parallel_mode}) =====")
    
    # Start timing
    start_time = time.time()
    
    try:
        # 1. Load data
        logging.info("Loading data...")
        galaxy_data = IFUDataCube(config.get_data_path(), config.lam_range_temp, config.redshift, config)
        
        # 2. Prepare templates
        logging.info("Preparing stellar and gas templates...")
        sps, gas_templates, gas_names, line_wave = prepare_templates(config, galaxy_data.velscale)
        
        # 3. Initialize radial binning
        rdb = RadialBinning(galaxy_data, config)
        
        # 4. Create bins
        ny, nx = galaxy_data.cube.shape[1:3]
        if center_x is None:
            center_x = nx / 2
        if center_y is None:
            center_y = ny / 2
        
        # Create bins based on mode
        if adaptive_bins:
            if target_snr is None:
                target_snr = 20.0  # Default target SNR
            n_bins = rdb.create_bins(n_bins=n_bins, center_x=center_x, center_y=center_y, 
                                    pa=pa, ellipticity=ellipticity, 
                                    adaptive_bins=True, target_snr=target_snr)
        else:    
            n_bins = rdb.create_bins(n_bins=n_bins, center_x=center_x, center_y=center_y, 
                                    pa=pa, ellipticity=ellipticity, log_spacing=log_spacing)
                                    
        if n_bins == 0:
            logging.error("Could not create radial bins")
            return galaxy_data, rdb
        
        # 5. Extract bin spectra
        # Try to use previous P2P results for velocity correction
        p2p_velfield = None
        p2p_path, p2p_exists = config.get_p2p_output_path("velfield")
        try:
            if p2p_exists:
                logging.info(f"Loading P2P velocity field: {p2p_path}")
                p2p_velfield = fits.getdata(p2p_path)
                logging.info(f"Successfully loaded P2P velocity field, shape: {p2p_velfield.shape}")
        except Exception as e:
            logging.warning(f"Could not load P2P velocity field: {str(e)}")
            p2p_velfield = None
            
        # Try to load ISAP velocity field if specified
        isap_velfield = None
        if config.use_isap and config.isap_file:
            try:
                logging.info(f"Loading ISAP velocity field: {config.isap_file}")
                isap_data, _ = read_isap_data(config.isap_file, 'velocity')
                if isap_data is not None:
                    isap_velfield = isap_data
                    logging.info(f"Successfully loaded ISAP velocity field, shape: {isap_velfield.shape}")
            except Exception as e:
                logging.warning(f"Could not load ISAP velocity field: {str(e)}")
        
        # Extract spectra (with velocity correction)
        bin_data = rdb.extract_bin_spectra(p2p_velfield, isap_velfield)
        
        # 6. Fit bins
        bin_results = rdb.fit_bins(sps, gas_templates, gas_names, line_wave)
        
        # 7. Process results
        rdb.process_results()
        
        # 8. Create summary plots
        if config.make_plots and not config.no_plots:
            logging.info("Creating summary plots...")
            rdb.create_summary_plots()
            
            # Create diagnostic plots for a few sample bins
            logging.info("Creating diagnostic plots for sample rings...")
            for bin_id in range(min(5, n_bins)):
                if bin_id in rdb.bin_results:
                    rdb.plot_bin_results(bin_id)
        
        # 9. Save results to FITS files
        logging.info("Saving results to FITS files...")
        rdb.save_results_to_fits()
        
        # Calculate completion time
        end_time = time.time()
        logging.info(f"RDB analysis completed in {end_time - start_time:.1f} seconds")
        
        return galaxy_data, rdb
        
    except Exception as e:
        logging.error(f"Error in RDB analysis: {str(e)}")
        logging.exception("Stack trace:")
        raise


### ------------------------------------------------- ###
# Main Function
### ------------------------------------------------- ###

def main():
    """
    Main function - parse command line arguments and run the program
    """
    # Create parser
    parser = argparse.ArgumentParser(description="ISAP v4.2.0 - IFU Spectral Analysis Pipeline")
    
    # Basic parameters
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Data directory path")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Output directory path")
    parser.add_argument("--galaxy-name", type=str, default=None,
                       help="Galaxy name")
    parser.add_argument("--data-file", type=str, default=None,
                       help="Data file name")
    parser.add_argument("--output-prefix", type=str, default=None,
                       help="Prefix for output files")
    parser.add_argument("--mode", type=str, choices=['P2P', 'VNB', 'RDB', 'ALL'], default=None,
                       help="Analysis mode: P2P (pixel-by-pixel), VNB (Voronoi binning), RDB (radial binning), ALL (all three in sequence)")
    
    # Parallel settings
    parser.add_argument("--threads", type=int, default=None,
                       help="Number of threads to use (default: half of CPU cores)")
    parser.add_argument("--parallel-mode", type=str, default='grouped', choices=['grouped', 'global'],
                       help="Parallel processing mode: 'grouped' for batch processing, 'global' for submitting all tasks at once (default: grouped)")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Number of bins per batch in grouped processing mode (default: 50)")
    
    # Plot settings
    parser.add_argument("--no-plots", action="store_true",
                       help="Do not create plots")
    parser.add_argument("--max-plots", type=int, default=50,
                       help="Maximum number of plots per type")
    parser.add_argument("--dpi", type=int, default=120,
                       help="Plot DPI")
    
    # Fitting settings
    parser.add_argument("--no-emission-lines", dest="compute_emission_lines", action="store_false",
                       help="Do not fit emission lines")
    parser.add_argument("--no-spectral-indices", dest="compute_spectral_indices", action="store_false",
                       help="Do not compute spectral indices")
    parser.add_argument("--global-search", action="store_true",
                       help="Use global search in pPXF fitting")
    
    # Template settings
    parser.add_argument("--template-dir", type=str, default="templates",
                       help="Template directory path")
    parser.add_argument("--use-miles", action="store_true", default=True,
                       help="Use MILES template library")
    parser.add_argument("--no-miles", dest="use_miles", action="store_false",
                       help="Do not use MILES template library")
    parser.add_argument("--template-file", type=str, default=None,
                       help="Custom template file name")
    
    # Voronoi binning parameters
    vnb_group = parser.add_argument_group('Voronoi Binning Options')
    vnb_group.add_argument("--target-snr", type=float, default=20.0,
                          help="Target SNR for Voronoi binning (default: 20)")
    
    # Radial binning parameters
    rdb_group = parser.add_argument_group('Radial Binning Options')
    rdb_group.add_argument("--n-rings", type=int, default=10,
                          help="Number of radial rings (default: 10)")
    rdb_group.add_argument("--center-x", type=float,
                          help="X coordinate of galaxy center (default: image center)")
    rdb_group.add_argument("--center-y", type=float,
                          help="Y coordinate of galaxy center (default: image center)")
    rdb_group.add_argument("--pa", type=float, default=0.0,
                          help="Position angle in degrees (default: 0)")
    rdb_group.add_argument("--ellipticity", type=float, default=0.0,
                          help="Ellipticity (e=1-b/a) for radial binning (default: 0)")
    rdb_group.add_argument("--linear-spacing", action="store_true",
                          help="Use linear spacing instead of logarithmic (for uniform bins mode)")
    rdb_group.add_argument("--adaptive-rdb", action="store_true",
                          help="Use adaptive radial binning to balance SNR")
    rdb_group.add_argument("--rdb-target-snr", type=float,
                          help="Target SNR for adaptive radial binning (default: same as VNB)")
    
    # Redshift parameters
    parser.add_argument("--redshift", type=float, default=0.0,
                       help="Galaxy redshift (default: 0)")
                       
    # ISAP integration
    parser.add_argument("--isap", action="store_true",
                       help="Use ISAP mode to read FITS files")
    parser.add_argument("--fits-file", type=str, default=None,
                       help="Specify FITS file path for ISAP data or pixel extraction")
    
    # Pixel extraction
    parser.add_argument("--extract-pixel", type=str, metavar="X,Y",
                       help="Extract velocity parameters for specified pixel (X,Y) from fitting results")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create configuration object
    config = P2PConfig(args)
    
    # Create file logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = config.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{config.galaxy_name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    
    logging.info(f"Starting ISAP v4.2.0 analysis - Log saved to: {log_path}")
    logging.info(f"Configuration parameters:\n{config}")
    
    # Handle pixel extraction if requested
    if args.extract_pixel:
        try:
            x, y = map(int, args.extract_pixel.split(','))
            pixel_data = extract_pixel_velocity(config, x, y, args.fits_file, isap_mode=args.isap)
            
            # Print formatted results
            print("\n========== Pixel Velocity Parameters ==========")
            print(f"Coordinates: ({x}, {y})")
            print(f"Galaxy: {pixel_data['object']}")
            
            if 'analysis_type' in pixel_data:
                print(f"Analysis type: {pixel_data['analysis_type']}")
            
            if pixel_data['velocity'] is not None and not np.isnan(pixel_data['velocity']):
                print(f"Velocity: {pixel_data['velocity']:.2f} km/s")
            else:
                print("Velocity: NaN (not fitted)")
            
            if pixel_data['sigma'] is not None and not np.isnan(pixel_data['sigma']):
                print(f"Velocity dispersion: {pixel_data['sigma']:.2f} km/s")
            else:
                print("Velocity dispersion: NaN (not fitted)")
            
            # Show binning information
            if 'bin_id' in pixel_data and pixel_data['bin_id'] >= 0:
                print(f"Voronoi bin ID: {pixel_data['bin_id']}")
            elif 'ring_id' in pixel_data and pixel_data['ring_id'] >= 0:
                print(f"Radial ring ID: {pixel_data['ring_id']}")
                if 'radius' in pixel_data:
                    print(f"Radial distance: {pixel_data['radius']:.2f} pixels")
            
            # Show additional ISAP information
            if args.isap:
                for key in ['instrume', 'date-obs', 'exptime']:
                    if key in pixel_data:
                        if key == 'exptime':
                            print(f"Exposure time: {pixel_data[key]:.1f} seconds")
                        else:
                            print(f"{key.capitalize()}: {pixel_data[key]}")
            
            # Show emission line information
            if 'emission_lines' in pixel_data and pixel_data['emission_lines']:
                print("\nEmission line fluxes:")
                for name, flux in pixel_data['emission_lines'].items():
                    if flux is not None and not np.isnan(flux):
                        print(f"  {name}: {flux:.4e}")
                    else:
                        print(f"  {name}: NaN (not detected)")
            
            # Show spectral index information
            if 'spectral_indices' in pixel_data and pixel_data['spectral_indices']:
                print("\nSpectral indices:")
                for name, value in pixel_data['spectral_indices'].items():
                    if value is not None and not np.isnan(value):
                        print(f"  {name}: {value:.4f}")
                    else:
                        print(f"  {name}: NaN (not measured)")
            
            # Show possible errors
            if 'error' in pixel_data:
                print(f"\nError: {pixel_data['error']}")
                
            print("===========================================\n")
            
            # If only extracting pixel data and not running analysis, exit
            if args.mode is None:
                return 0
                
        except Exception as e:
            print(f"Error extracting pixel data: {str(e)}")
            logging.error(f"Error extracting pixel data: {str(e)}")
            if args.mode is None:
                return 1
    
    # Run analysis based on mode
    try:
        # P2P analysis
        p2p_done = False
        galaxy_data = None
        p2p_velfield = None
        p2p_sigfield = None
        
        if args.mode == "P2P" or args.mode == "ALL":
            # Run P2P analysis first
            print(f"Running Pixel-by-Pixel analysis for {config.galaxy_name}")
            galaxy_data, p2p_velfield, p2p_sigfield = run_p2p_analysis(config)
            p2p_done = True

        # VNB analysis
        if args.mode == "VNB" or args.mode == "ALL":
            # If P2P already done, use its results
            if p2p_done:
                print(f"Running Voronoi binning with target SNR={args.target_snr}, using P2P results")
                run_vnb_analysis(config, target_snr=args.target_snr)
            else:
                # Run VNB analysis independently
                print(f"Running Voronoi binning with target SNR={args.target_snr}")
                run_vnb_analysis(config, target_snr=args.target_snr)
            
        # RDB analysis
        if args.mode == "RDB" or args.mode == "ALL":
            # RDB mode
            log_spacing = not args.linear_spacing
            spacing_type = "logarithmic" if log_spacing else "linear"
            
            # Set adaptive binning parameters
            if args.adaptive_rdb:
                # Use same target SNR as VNB unless explicitly specified
                rdb_target_snr = args.rdb_target_snr if args.rdb_target_snr else args.target_snr
                
                if p2p_done:
                    print(f"Running Adaptive Radial binning with target SNR={rdb_target_snr}, max bins={args.n_rings}, using P2P results")
                else:
                    print(f"Running Adaptive Radial binning with target SNR={rdb_target_snr}, max bins={args.n_rings}")
            else:
                if p2p_done:
                    print(f"Running Uniform Radial binning with {args.n_rings} rings, {spacing_type} spacing, using P2P results")
                else:
                    print(f"Running Uniform Radial binning with {args.n_rings} rings, {spacing_type} spacing")
            
            if args.center_x is not None and args.center_y is not None:
                print(f"Using specified center: ({args.center_x}, {args.center_y})")
            if args.pa != 0 or args.ellipticity > 0:
                print(f"Using PA={args.pa}°, ellipticity={args.ellipticity}")
                
            run_rdb_analysis(config, n_bins=args.n_rings, 
                           center_x=args.center_x, center_y=args.center_y,
                           pa=args.pa, ellipticity=args.ellipticity,
                           log_spacing=log_spacing,
                           adaptive_bins=args.adaptive_rdb, 
                           target_snr=rdb_target_snr if args.adaptive_rdb else None)
        
        logging.info("Analysis completed")
        print(f"Analysis completed. Results in {config.output_dir}, logs in {log_path}")
        
    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
        logging.exception("Stack trace:")
        print(f"Error during execution: {str(e)}")
        print(f"See log for details: {log_path}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())