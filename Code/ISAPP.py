"""
Pixel-to-Pixel (P2P) IFU Spectral Analysis Pipeline

This module performs pixel-by-pixel spectral fitting of IFU data using pPXF,
calculates emission line properties, and computes spectral indices.

Features:
- Multi-threaded pixel fitting
- Two-stage fitting strategy (stellar first, then emission lines)
- Efficient spectral index calculation 
- Robust error handling and recovery
- Customizable visualization
- Single pixel testing capability

Version 3.9.9    2025Mar10    Memory optimization for plotting functions
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
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from scipy import interpolate, integrate
from scipy.optimize import curve_fit
from tqdm import tqdm

from contextlib import redirect_stdout, redirect_stderr

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

# 设置matplotlib的非交互式后端，减少内存使用
mpl.use('Agg')

# 抑制警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
        self.moments = [-2, 2]  # Moments to fit for stellar and gas components
        self.gas_names = ['Hbeta', '[OIII]5007']  # Gas lines to fit
        self.ngas_comp = 1  # Number of gas components
        self.fwhm_gas = 1.0  # FWHM for emission line templates (Angstroms)
        self.mask_width = 1000  # Width parameter for determine_mask function
        
        # Two-stage fitting parameters
        self.use_two_stage_fit = True  # Use two-stage fitting strategy
        self.global_search = True  # Use global search in second stage fitting
        
        # Computational settings
        self.n_threads = os.cpu_count() // 2  # Default to half available cores
        self.max_memory_gb = 12  # Maximum memory usage in GB
        
        # 优化后的图形控制选项
        self.make_plots = True          # 是否生成图
        self.plot_every_n = 300         # 每n个像素生成一次图
        self.save_plots = True          # 是否保存图
        self.dpi = 100                  # 图像DPI，降低以减少内存使用
        self.no_plots = False           # 完全禁用所有图形，优先级最高
        self.max_plots = 20             # 最大生成图数量
        self.plot_count = 0             # 当前图数量计数器
        self.LICplot = False            # 是否绘制LineIndexCalculator的图

        # Flags for what to compute
        self.compute_emission_lines = True
        self.compute_spectral_indices = True
        self.compute_stellar_pops = False  # Not implemented yet
        
        # Error handling and recovery
        self.fallback_to_simple_fit = True  # Use simpler fit if primary fit fails
        self.retry_with_degree_zero = True  # Retry with degree=0 if fit fails
        self.skip_bad_pixels = True  # Skip pixels with insufficient data
        self.safe_mode = False  # Extra-safe settings for difficult data
        
        # Spectral index calculation
        self.continuum_mode = 'auto'  # 'auto', 'fit', or 'original'
        
        # Constants
        self.c = 299792.458  # Speed of light in km/s
    
    def create_directories(self):
        """Create all required directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        if self.make_plots and not self.no_plots:
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

def Apply_velocity_correction(wave, z):
        """
        应用速度修正到波长
        
        Parameters:
        -----------
        wave : array-like
            原始波长数组
            
        Returns:
        --------
        array-like : 修正后的波长数组
        """
        return wave / (1 + (z))

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
                 velocity_correction=0, error=None, continuum_mode='auto'):
        """
        初始化吸收线指数计算器
        
        Parameters:
        -----------
        wave : array-like
            原始光谱的波长数组
        flux : array-like
            原始光谱的流量数组
        fit_wave : array-like
            拟合光谱的波长数组，用于计算连续谱
        fit_flux : array-like
            拟合光谱的流量数组，用于计算连续谱
        em_wave : array-like, optional
            发射线的波长数组
        em_flux_list : array-like, optional
            合并后的发射线光谱
        velocity_correction : float, optional
            速度修正值，单位为km/s，默认为0
        error : array-like, optional
            误差数组
        continuum_mode : str, optional
            连续谱计算模式
            'auto': 仅在原始谱数据不足时使用拟合谱
            'fit': 始终使用拟合谱
            'original': 尽可能使用原始谱（数据不足时报警）
        """
        self.c = 299792.458  # 光速，单位为km/s
        self.velocity = velocity_correction
        self.continuum_mode = continuum_mode
        
        # 进行速度修正
        if wave is not None:
            self.wave = self._apply_velocity_correction(wave)
            self.flux = flux.copy()  # 创建副本以避免修改原始数据
        else:
            self.wave = None
            self.flux = None

        self.fit_wave = fit_wave
        self.fit_flux = fit_flux
        self.error = error if error is not None else (np.ones_like(flux) if flux is not None else None)
        
        # 处理发射线
        if em_wave is not None and em_flux_list is not None:
            self.em_wave = self._apply_velocity_correction(em_wave)
            # self.em_wave = em_wave
            self.em_flux_list = em_flux_list
            self._subtract_emission_lines()
    
    def _subtract_emission_lines(self):
        """
        从原始光谱中减去发射线
        输入的em_flux_list已经是合并后的结果
        """
        if self.wave is None or self.flux is None:
            return
            
        try:
            # 将发射线光谱重采样到原始光谱的波长网格上
            em_flux_resampled = spectres(self.wave, self.em_wave, self.em_flux_list)
            
            # 从原始光谱中减去发射线
            self.flux -= em_flux_resampled
        except Exception as e:
            logging.warning(f"Error subtracting emission lines: {str(e)}")
            # 出错时继续，不减去发射线
    
    def _apply_velocity_correction(self, wave):
        """
        应用速度修正到波长
        
        Parameters:
        -----------
        wave : array-like
            原始波长数组
            
        Returns:
        --------
        array-like : 修正后的波长数组
        """
        return wave / (1 + (self.velocity/self.c))

    def _check_data_coverage(self, wave_range):
        """
        检查原始数据是否完整覆盖给定波长范围
        
        Parameters:
        -----------
        wave_range : tuple
            (min_wave, max_wave)
            
        Returns:
        --------
        bool : 是否完整覆盖
        """
        if self.wave is None:
            return False
        return (wave_range[0] >= np.min(self.wave)) and (wave_range[1] <= np.max(self.wave))
        
    def define_line_windows(self, line_name):
        """
        定义吸收线和连续谱窗口
        
        Parameters:
        -----------
        line_name : str
            吸收线名称
            
        Returns:
        --------
        dict : 包含蓝端、中心和红端窗口的字典
        """
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
            }
        }
        return windows.get(line_name)

    def calculate_pseudo_continuum(self, wave_range, flux_range, region_type):
        """
        计算伪连续谱
        
        Parameters:
        -----------
        wave_range : tuple or array-like
            波长范围
        flux_range : array-like or None
            对应的流量值（如果使用拟合谱则不需要）
        region_type : str
            区域类型('blue' 或 'red')
            
        Returns:
        --------
        float : 伪连续谱值
        """
        if self.continuum_mode == 'fit':
            # 使用拟合谱
            mask = (self.fit_wave >= wave_range[0]) & (self.fit_wave <= wave_range[1])
            return np.median(self.fit_flux[mask])
        
        elif self.continuum_mode == 'auto':
            # 检查原始数据覆盖
            if self._check_data_coverage(wave_range):
                mask = (self.wave >= wave_range[0]) & (self.wave <= wave_range[1])
                return np.median(self.flux[mask])
            else:
                # 数据不足时使用拟合谱
                mask = (self.fit_wave >= wave_range[0]) & (self.fit_wave <= wave_range[1])
                return np.median(self.fit_flux[mask])
        
        else:  # 'original'
            if not self._check_data_coverage(wave_range):
                raise ValueError(f"原始数据不足以覆盖{region_type}端连续谱区域")
            mask = (self.wave >= wave_range[0]) & (self.wave <= wave_range[1])
            return np.median(self.flux[mask])

    def calculate_index(self, line_name, return_error=False):
        """
        计算吸收线指数
        
        Parameters:
        -----------
        line_name : str
            吸收线名称 ('Hbeta', 'Mgb', 或 'Fe5015')
        return_error : bool
            是否返回误差
            
        Returns:
        --------
        float : 吸收线指数值
        float : 误差值（如果return_error=True）
        """
        if self.wave is None or self.flux is None:
            return np.nan if not return_error else (np.nan, np.nan)
            
        # 获取窗口定义
        windows = self.define_line_windows(line_name)
        if windows is None:
            raise ValueError(f"未知的吸收线: {line_name}")

        # 获取线心区域数据
        line_mask = (self.wave >= windows['line'][0]) & (self.wave <= windows['line'][1])
        line_wave = self.wave[line_mask]
        line_flux = self.flux[line_mask]
        line_err = self.error[line_mask]

        # 检查数据点数
        if len(line_flux) < 3:
            return np.nan if not return_error else (np.nan, np.nan)

        # 计算连续谱
        blue_cont = self.calculate_pseudo_continuum(windows['blue'], None, 'blue')
        red_cont = self.calculate_pseudo_continuum(windows['red'], None, 'red')
        
        wave_cont = np.array([
            np.mean(windows['blue']), 
            np.mean(windows['red'])
        ])
        flux_cont = np.array([blue_cont, red_cont])
        
        # 线性插值得到连续谱
        f_interp = interpolate.interp1d(wave_cont, flux_cont)
        cont_at_line = f_interp(line_wave)

        # 计算积分
        index = np.trapz((1.0 - line_flux/cont_at_line), line_wave)
        
        # 存储计算结果供绘图使用
        self._last_calc = {
            'line_name': line_name,
            'windows': windows,
            'blue_cont': blue_cont,
            'red_cont': red_cont,
            'wave_cont': wave_cont,
            'flux_cont': flux_cont,
            'line_wave': line_wave,
            'line_flux': line_flux,
            'cont_at_line': cont_at_line,
            'index': index
        }
        
        if return_error:
            # 计算误差
            error = np.sqrt(np.trapz((line_err/cont_at_line)**2, line_wave))
            return index, error
        
        return index

    def plot_line_fit(self, line_name, output_path=None):
        """
        绘制单个吸收线拟合结果 - 内存优化版本
        
        Parameters:
        -----------
        line_name : str
            吸收线名称
        output_path : str, optional
            保存图像的路径
        """
        if self.wave is None or self.flux is None:
            logging.warning(f"No data available for plotting {line_name}")
            return
            
        # 如果还没有计算过指数，先计算一次
        if not hasattr(self, '_last_calc') or self._last_calc.get('line_name') != line_name:
            self.calculate_index(line_name)
            
        if not hasattr(self, '_last_calc'):
            logging.warning(f"没有可用于绘图的数据: {line_name}")
            return
            
        data = self._last_calc
        windows = data['windows']
        
        # 设置X轴范围：窗口两侧各延伸20Å
        x_min = windows['blue'][0] - 20
        x_max = windows['red'][1] + 20
        
        with plt.rc_context({'figure.max_open_warning': False}):
            # 创建图形和子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=100)
            
            # 第一个面板：原始光谱和拟合
            # 根据需要绘制发射线
            if hasattr(self, 'em_flux_list'):
                try:
                    # 绘制带有发射线的原始光谱
                    org_flux = self.flux.copy()
                    if hasattr(self, 'em_wave') and hasattr(self, 'em_flux_list'):
                        em_flux_resampled = spectres(self.wave, self.em_wave, self.em_flux_list)
                        orig_with_em = org_flux + em_flux_resampled
                        ax1.plot(self.wave, orig_with_em, 'k-', label='Original+Emission', alpha=0.7)
                        ax1.plot(self.em_wave, self.em_flux_list, 'r-', label='Emission Lines', alpha=0.7)
                except Exception as e:
                    ax1.plot(self.wave, self.flux, 'k-', label='Original Spectrum', alpha=0.7)
                    logging.warning(f"无法绘制发射线: {str(e)}")
            else:
                ax1.plot(self.wave, self.flux, 'k-', label='Original Spectrum', alpha=0.7)
                
            # 绘制拟合模板
            ax1.plot(self.fit_wave, self.fit_flux, 'b-', label='Template Fit', alpha=0.7)
            
            # 标记吸收线区域
            colors = {'blue': 'b', 'line': 'g', 'red': 'r'}
            for region, (start, end) in windows.items():
                ax1.axvspan(start, end, alpha=0.2, color=colors[region], 
                           label=f'{region.capitalize()} Window')
                ax2.axvspan(start, end, alpha=0.2, color=colors[region])

            # 第二个面板：吸收指数测量
            ax2.plot(self.wave, self.flux, 'k-', label='Processed Spectrum')
            
            # 绘制连续谱点和线
            ax2.plot(data['wave_cont'], data['flux_cont'], 'r*', markersize=10, label='Continuum Points')
            ax2.plot(data['line_wave'], data['cont_at_line'], 'r--', label='Continuum Fit')
            
            # 绘制填充的吸收区域（指数）
            ax2.fill_between(data['line_wave'], data['line_flux'], data['cont_at_line'], 
                            alpha=0.3, color='g', label='Absorption Index')
            
            # 添加文本标签显示指数值
            ax2.text(0.05, 0.9, f"{line_name} Index = {data['index']:.4f} Å", 
                    transform=ax2.transAxes, fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.7))
            
            # 设置坐标轴属性
            for ax in [ax1, ax2]:
                ax.set_xlim(x_min, x_max)
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(axis='both', which='both', labelsize='x-small', 
                              right=True, top=True, direction='in')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
            
            # 设置标题和标签
            ax1.set_title(f'Original Spectrum (v={self.velocity:.1f} km/s)')
            ax2.set_title(f'Index Measurement: {line_name}')
            ax2.set_xlabel('Rest-frame Wavelength (Å)')
            ax1.set_ylabel('Flux')
            ax2.set_ylabel('Flux')
            
            plt.tight_layout()
            
            # 保存或显示图像
            if output_path:
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)

    def plot_all_lines(self, mode=None, number=None, save_path=None, show_index=False):
        """
        绘制包含所有谱线的完整图谱 - 内存优化版本
        
        Parameters:
        -----------
        mode : str, optional
            图像的模式，必须是'P2P'、'VNB'或'RNB'之一
        number : int, optional
            图像的编号，必须是整数
        save_path : str, optional
            图像保存的路径。如果提供，图像将被保存到该路径下
        show_index : bool, optional
            是否显示谱指数参数，默认为False
        """
        if self.wave is None or self.flux is None:
            logging.warning("No data available for plotting all lines")
            return
            
        # 验证mode和number参数
        if mode is not None and number is not None:
            valid_modes = ['P2P', 'VNB', 'RNB']
            if mode not in valid_modes:
                raise ValueError(f"Mode must be one of {valid_modes}")
            if not isinstance(number, int):
                raise ValueError("Number must be an integer")
            mode_title = f"{mode}{number}"
        else:
            mode_title = None
        
        # 获取所有定义的谱线
        all_windows = {
            'Hbeta': self.define_line_windows('Hbeta'),
            'Mgb': self.define_line_windows('Mgb'),
            'Fe5015': self.define_line_windows('Fe5015')
        }
        
        # 设置固定的X轴范围
        min_wave = 4800
        max_wave = 5250
        
        with plt.rc_context({'figure.max_open_warning': False}):
            # 创建图形和整体标题，降低DPI
            fig = plt.figure(figsize=(15, 12), dpi=100)
            if mode_title:
                fig.suptitle(mode_title, fontsize=16, y=0.95)
            
            # 创建子图，调整高度比例以适应整体标题
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.2)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            
            # 设置统一的颜色方案
            colors = {
                'blue': 'tab:blue',
                'line': 'tab:green',
                'red': 'tab:red',
                'orig_cont': 'tab:orange',   # 原始光谱连续谱颜色（橙色）
                'fit_cont': 'tab:green',     # 拟合光谱连续谱颜色（绿色）
                'inactive_cont': 'gray'      # 未使用的连续谱颜色
            }
            
            # 第一个面板：原始数据对比
            wave_mask = (self.wave >= min_wave) & (self.wave <= max_wave)
            fit_mask = (self.fit_wave >= min_wave) & (self.fit_wave <= max_wave)
            
            # 计算y轴范围
            if hasattr(self, 'em_flux_list'):
                try:
                    em_flux_resampled = spectres(self.wave, self.em_wave, self.em_flux_list)
                    flux_range = self.flux[wave_mask] + em_flux_resampled[wave_mask]
                except:
                    flux_range = self.flux[wave_mask]
            else:
                flux_range = self.flux[wave_mask]
            fit_range = self.fit_flux[fit_mask]
            
            y_min = min(np.min(flux_range), np.min(fit_range)) * 0.9
            y_max = max(np.max(flux_range), np.max(fit_range)) * 1.1
            
            # 绘制光谱
            if hasattr(self, 'em_flux_list'):
                try:
                    em_flux_resampled = spectres(self.wave, self.em_wave, self.em_flux_list)
                    ax1.plot(self.wave, self.flux + em_flux_resampled, color='tab:blue', 
                            label='Original Spectrum', alpha=0.8)
                    ax1.plot(self.em_wave, self.em_flux_list, color='tab:orange', 
                            label='Emission Lines', alpha=0.8)
                except:
                    ax1.plot(self.wave, self.flux, color='tab:blue', 
                            label='Original Spectrum', alpha=0.8)
            else:
                ax1.plot(self.wave, self.flux, color='tab:blue', 
                        label='Original Spectrum', alpha=0.8)
            ax1.plot(self.fit_wave, self.fit_flux, color='tab:red', 
                    label='Template Fit', alpha=0.8)
            
            # 为第二个面板计算y轴范围
            processed_flux = self.flux[wave_mask]
            fit_flux_range = self.fit_flux[fit_mask]
            y_min_processed = min(np.min(processed_flux), np.min(fit_flux_range)) * 0.9
            y_max_processed = max(np.max(processed_flux), np.max(fit_flux_range)) * 1.1
            
            # 第二个面板：处理后的光谱
            ax2.plot(self.wave, self.flux, color='tab:blue', 
                    label='Processed Spectrum', alpha=0.8)
            ax2.plot(self.fit_wave, self.fit_flux, '--', color='tab:red',
                    label='Template Fit', alpha=0.8)
            
            # 在两个面板中标记所有谱线区域
            for line_name, windows in all_windows.items():
                for panel in [ax1, ax2]:
                    # 标记蓝端、线心和红端区域
                    alpha = 0.2  # 恢复原来的透明度
                    # 只在图例中显示一次每种区域类型
                    if line_name == 'Hbeta':  # 第一个谱线用于图例
                        panel.axvspan(windows['blue'][0], windows['blue'][1], 
                                    alpha=alpha, color=colors['blue'], label='Blue window')
                        panel.axvspan(windows['line'][0], windows['line'][1], 
                                    alpha=alpha, color=colors['line'], label='Line region')
                        panel.axvspan(windows['red'][0], windows['red'][1], 
                                    alpha=alpha, color=colors['red'], label='Red window')
                    else:
                        panel.axvspan(windows['blue'][0], windows['blue'][1], 
                                    alpha=alpha, color=colors['blue'])
                        panel.axvspan(windows['line'][0], windows['line'][1], 
                                    alpha=alpha, color=colors['line'])
                        panel.axvspan(windows['red'][0], windows['red'][1], 
                                    alpha=alpha, color=colors['red'])
                    
                    # 添加文字标注到底部
                    if panel == ax1:
                        y_text = y_min + 0.05 * (y_max - y_min)
                    else:
                        y_text = y_min_processed + 0.05 * (y_max_processed - y_min_processed)
                    
                    # 基础标签
                    panel.text(np.mean(windows['line']), y_text, line_name,
                            horizontalalignment='center', verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                    
                    # 在第二个面板添加连续谱点
                    if panel == ax2:
                        # 计算连续谱点
                        blue_cont_orig = None
                        red_cont_orig = None
                        blue_cont_fit = None
                        red_cont_fit = None
                        
                        # 检查是否可以使用原始光谱
                        if self._check_data_coverage(windows['blue']):
                            mask = (self.wave >= windows['blue'][0]) & (self.wave <= windows['blue'][1])
                            blue_cont_orig = np.median(self.flux[mask])
                        if self._check_data_coverage(windows['red']):
                            mask = (self.wave >= windows['red'][0]) & (self.wave <= windows['red'][1])
                            red_cont_orig = np.median(self.flux[mask])
                        
                        # 计算拟合光谱的连续谱点
                        mask_blue = (self.fit_wave >= windows['blue'][0]) & (self.fit_wave <= windows['blue'][1])
                        mask_red = (self.fit_wave >= windows['red'][0]) & (self.fit_wave <= windows['red'][1])
                        blue_cont_fit = np.median(self.fit_flux[mask_blue])
                        red_cont_fit = np.median(self.fit_flux[mask_red])
                        
                        wave_cont = np.array([
                            np.mean(windows['blue']), 
                            np.mean(windows['red'])
                        ])

                        # 根据计算模式决定哪个是活动的连续谱
                        is_orig_active = (self.continuum_mode == 'original' or 
                                        (self.continuum_mode == 'auto' and 
                                        blue_cont_orig is not None and 
                                        red_cont_orig is not None))
                        
                        # 绘制原始光谱连续谱点和线（如果存在）
                        if blue_cont_orig is not None and red_cont_orig is not None:
                            flux_cont_orig = np.array([blue_cont_orig, red_cont_orig])
                            if not is_orig_active:
                                # 非活动状态
                                panel.plot(wave_cont, flux_cont_orig, '*', color=colors['inactive_cont'], 
                                        markersize=10, alpha=0.5,
                                        label='Original spectrum continuum (inactive)' if line_name == 'Hbeta' else '')
                                panel.plot(wave_cont, flux_cont_orig, '--', color=colors['inactive_cont'], 
                                        alpha=0.5)
                            else:
                                # 活动状态
                                panel.plot(wave_cont, flux_cont_orig, '*', color=colors['orig_cont'], 
                                        markersize=10, alpha=0.8,
                                        label='Original spectrum continuum (orange)' if line_name == 'Hbeta' else '')
                                panel.plot(wave_cont, flux_cont_orig, '--', color=colors['orig_cont'], 
                                        alpha=0.8)

                        # 绘制拟合光谱连续谱点和线
                        flux_cont_fit = np.array([blue_cont_fit, red_cont_fit])
                        if is_orig_active:
                            # 非活动状态
                            panel.plot(wave_cont, flux_cont_fit, '*', color=colors['inactive_cont'], 
                                    markersize=10, alpha=0.5,
                                    label='Template continuum (inactive)' if line_name == 'Hbeta' else '')
                            panel.plot(wave_cont, flux_cont_fit, '--', color=colors['inactive_cont'], 
                                    alpha=0.5)
                        else:
                            # 活动状态
                            panel.plot(wave_cont, flux_cont_fit, '*', color=colors['fit_cont'], 
                                    markersize=10, alpha=0.8,
                                    label='Template continuum (green)' if line_name == 'Hbeta' else '')
                            panel.plot(wave_cont, flux_cont_fit, '--', color=colors['fit_cont'], 
                                    alpha=0.8)

                        # 如果需要显示谱指数参数
                        if show_index:
                            try:
                                # 保存当前的continuum_mode
                                original_mode = self.continuum_mode
                                
                                # 计算原始光谱的指数值
                                self.continuum_mode = 'original'
                                try:
                                    orig_index = self.calculate_index(line_name)
                                    if np.isnan(orig_index):
                                        orig_index = None
                                except ValueError:
                                    orig_index = None
                                
                                # 计算拟合光谱的指数值
                                self.continuum_mode = 'fit'
                                fit_index = self.calculate_index(line_name)
                                if np.isnan(fit_index):
                                    fit_index = None
                                
                                # 恢复原始的continuum_mode
                                self.continuum_mode = original_mode
                                
                                # 计算文本位置
                                base_y_text = y_text + 0.05 * (y_max_processed - y_min_processed)
                                
                                # 构建显示文本
                                if orig_index is not None and fit_index is not None:
                                    # 分别显示两个值
                                    y_offset = 0.1 * (y_max_processed - y_min_processed)
                                    
                                    # 显示原始光谱的值（上面）
                                    panel.text(np.mean(windows['line']), 
                                            base_y_text + y_offset,
                                            f"{orig_index:.3f}", 
                                            color=colors['orig_cont'], 
                                            horizontalalignment='center',
                                            verticalalignment='bottom', 
                                            fontsize='x-small',
                                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                                    
                                    # 显示拟合光谱的值（下面）
                                    panel.text(np.mean(windows['line']), 
                                            base_y_text + y_offset/2,
                                            f"{fit_index:.3f}", 
                                            color=colors['fit_cont'], 
                                            horizontalalignment='center',
                                            verticalalignment='bottom', 
                                            fontsize='x-small',
                                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                                    
                                elif fit_index is not None:
                                    # 只显示拟合光谱的值
                                    fit_text = f"{fit_index:.3f}"
                                    panel.text(np.mean(windows['line']), 
                                            base_y_text + 0.02 * (y_max_processed - y_min_processed),
                                            fit_text, 
                                            color=colors['fit_cont'], 
                                            horizontalalignment='center',
                                            verticalalignment='bottom', 
                                            fontsize='x-small',
                                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                            
                            except Exception as e:
                                logging.warning(f"Error calculating index for {line_name}: {str(e)}")
            
            # 设置两个面板的属性
            ax1.set_xlim(min_wave, max_wave)
            ax1.set_ylim(y_min, y_max)
            ax2.set_xlim(min_wave, max_wave)
            ax2.set_ylim(y_min_processed, y_max_processed)
            
            # 设置两个面板的共同属性
            for ax in [ax1, ax2]:
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(axis='both', which='both', labelsize='x-small', 
                            right=True, top=True, direction='in')
                ax.set_xlabel('Rest-frame Wavelength (Å)')
                ax.set_ylabel('Flux')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
            
            ax1.set_title(f'Original Data Comparison (v={self.velocity:.1f} km/s)')
            ax2.set_title('Processed Spectrum with Continuum Fits')
            
            # 调整布局
            if mode_title:
                plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.3)
            else:
                plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9, hspace=0.3)
            
            # 如果提供了保存路径，保存图像
            if save_path and mode_title:
                # 确保save_path存在
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                # 构建完整的文件路径
                filepath = os.path.join(save_path, f"{mode_title}.png")
                
                # 保存图像为PNG而不是PDF，减少文件大小
                plt.savefig(filepath, format='png', dpi=100, bbox_inches='tight')
                logging.info(f"Figure saved as: {filepath}")
                plt.close(fig)
            else:
                plt.show()
                plt.close(fig)


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
        
        # Filter to good wavelength range
        spectra = spectra[ np.where((np.exp(ln_lam_gal) > Apply_velocity_correction(config.good_wavelength_range[0],config.redshift))
                                    & (np.exp(ln_lam_gal) < Apply_velocity_correction(config.good_wavelength_range[1],config.redshift))) ]
        ln_lam_gal = ln_lam_gal[ np.where((np.exp(ln_lam_gal) > Apply_velocity_correction(config.good_wavelength_range[0],config.redshift))
                                    & (np.exp(ln_lam_gal) < Apply_velocity_correction(config.good_wavelength_range[1],config.redshift))) ]
        
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
        # Fix for optimal_templates size
        self.optimal_templates = np.full((93974, self.cube.shape[1], self.cube.shape[2]), np.nan)
        
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
    使用两阶段拟合策略拟合单个像素。
    
    阶段1: 仅使用恒星模板拟合
    阶段2: 使用第一阶段最优恒星模板和气体模板一起拟合
    
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
        wave_range = [(Apply_velocity_correction(config.good_wavelength_range[0], config.redshift)),
                            (Apply_velocity_correction(config.good_wavelength_range[1], config.redshift))]
        # Get spectrum data
        spectrum = galaxy_data.spectra[:, k_index]
        noise = np.ones_like(spectrum)  
        
        # 波长范围
        lam_gal = galaxy_data.lam_gal
        lam_range_temp = np.exp(sps.ln_lam_temp[[0, -1]])
        
        spectrum = spectrum[ np.where((lam_gal > wave_range[0]) & (lam_gal < wave_range[1])) ]
        lam_gal = lam_gal[ np.where((lam_gal > wave_range[0]) & (lam_gal < wave_range[1])) ]

        noise = np.ones_like(spectrum)# Use uniform noise

        # 自动计算掩码 - 与原始代码一致
        mask = util.determine_mask(np.log(lam_gal), lam_range_temp, width=1000)
        
        logging.debug(f"STEP: Automatically calculated mask with width=1000")
        logging.debug(f"  - Mask has {np.sum(mask)} points")
        
        if not np.any(mask):
            logging.warning(f"Empty mask for pixel ({i},{j}). Wavelength ranges may not overlap.")
            return i, j, None
        
        #################################################
        # 第一阶段：仅拟合恒星成分 (与原始代码Cube_sol一致)
        #################################################
        logging.debug(f"STEP: FIRST STAGE - Stellar component only fit")
        
        try:
            
            
            # 使用与原始代码相同的参数
            pp_stars = ppxf(sps.templates, spectrum, noise, galaxy_data.velscale, 
                          [config.vel_s, config.vel_dis_s],
                          degree=3,  # 原始代码使用degree=3
                          plot=False, mask=mask, lam=lam_gal, 
                          lam_temp=sps.lam_temp, quiet=True)
            
            logging.debug(f"  - First stage fit successful: v={pp_stars.sol[0]:.1f}, σ={pp_stars.sol[1]:.1f}")
        except Exception as e:
            if config.retry_with_degree_zero:
                logging.warning(f"Initial stellar fit failed for pixel ({i},{j}): {str(e)}")
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
        
        # 创建最优恒星模板 - 与原始代码一致，只做权重和模板相乘
        logging.debug(f"STEP: Creating optimal stellar template")
        
        # Ensure we have valid weights
        if pp_stars.weights is None or not np.any(np.isfinite(pp_stars.weights)):
            logging.warning(f"Invalid weights in stellar fit for pixel ({i},{j})")
            return i, j, None
        
        # 计算最优恒星模板 - 与原始代码一致: 只用权重和模板相乘
        optimal_stellar_template = sps.templates @ pp_stars.weights
        
        # 记录apoly，但不添加到模板中
        apoly = pp_stars.apoly if hasattr(pp_stars, 'apoly') and pp_stars.apoly is not None else None
        
        # 保存第一阶段结果
        vel_stars = to_scalar(pp_stars.sol[0])
        sigma_stars = to_scalar(pp_stars.sol[1]) 
        bestfit_stars = pp_stars.bestfit
        
        # 确保sigma值合理
        if sigma_stars < 0:
            logging.warning(f"Negative velocity dispersion detected: {sigma_stars:.1f} km/s. Setting to 10 km/s.")
            sigma_stars = 10.0
        
        #################################################
        # 第二阶段：使用恒星模板和气体模板一起拟合 (与原始STF部分一致)
        #################################################
        if config.use_two_stage_fit and config.compute_emission_lines:
            
            logging.debug(f"STEP: SECOND STAGE - Combined fit with optimal stellar template")
            
            # 定义波长范围：从Hbeta蓝端到Mgb红端 - 与原始代码一致
            wave_range = [(Apply_velocity_correction(config.good_wavelength_range[0], config.redshift)),
                            (Apply_velocity_correction(config.good_wavelength_range[1], config.redshift))]
            
            # 只截取观测数据的波长范围 - 与原始代码一致
            wave_mask = (lam_gal >= wave_range[0]) & (lam_gal <= wave_range[1])
            galaxy_subset = spectrum[wave_mask]
            noise_subset = np.ones_like(galaxy_subset)
            
            # 确保恒星模板是正确的形状
            if optimal_stellar_template.ndim > 1 and optimal_stellar_template.shape[1] == 1:
                optimal_stellar_template = optimal_stellar_template.flatten()
            
            # 合并恒星和气体模板 - 与原始代码一致
            stars_gas_templates = np.column_stack([optimal_stellar_template, gas_templates])
            
            # 设置成分数组 - 与原始代码一致，是气体模板的数量而不是固定为2
            component = [0] + [1]*gas_templates.shape[1]  # 第一个是恒星模板，其余是气体模板
            gas_component = np.array(component) > 0
            
            # 设置moments参数 - 与原始代码一致
            moments = config.moments  # 使用配置中的值，默认为[-2, 2]
            
            # 设置起始值 - 与原始代码一致
            start = [
                [vel_stars, sigma_stars],  # 恒星成分
                [vel_stars, 50]            # 气体成分
            ]
            
            # 设置边界 - 与原始代码一致
            vlim = lambda x: vel_stars + x*np.array([-100, 100])
            bounds = [
                [vlim(2), [20, 300]],  # 恒星成分
                [vlim(2), [20, 100]]   # 气体成分
            ]
            
            # 设置tied参数 - 与原始代码一致
            ncomp = len(moments)
            tied = [['', ''] for _ in range(ncomp)]
            
            try:
                # 执行第二阶段拟合 - 与原始代码一致
                with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                    pp = ppxf(stars_gas_templates, galaxy_subset, noise_subset, galaxy_data.velscale, start,
                        plot=False, moments=moments, degree=3, mdegree=-1, 
                        component=component, gas_component=gas_component, gas_names=gas_names,
                        lam=lam_gal[wave_mask], lam_temp=sps.lam_temp, 
                        tied=tied, bounds=bounds, quiet=True,
                        global_search=config.global_search)
                
                logging.debug(f"  - Combined fit successful: v={to_scalar(pp.sol[0]):.1f}, "
                            f"σ={to_scalar(pp.sol[1]):.1f}, χ²={to_scalar(pp.chi2):.3f}")
                
                # 检查是否成功拟合到发射线
                has_emission = False
                if hasattr(pp, 'gas_bestfit') and pp.gas_bestfit is not None:
                    has_emission = np.any(np.abs(pp.gas_bestfit) > 1e-10)
                
                if has_emission:
                    logging.debug(f"  - Gas emission detected in second stage fit")
                else:
                    logging.debug(f"  - No significant emission detected in second stage fit")
                
                # 创建完整的bestfit（波长范围可能只是子集）
                full_bestfit = np.copy(bestfit_stars)  # 先用第一阶段结果填充

                # Add calculate template
                Apoly_Params = np.polyfit(lam_gal[wave_mask], pp.apoly, 3)
                Temp_Calu = (stars_gas_templates[:,0] * pp.weights[0]) + np.poly1d(Apoly_Params)(sps.lam_temp)
                
                # 添加完整的气体模板 (如果有)
                if hasattr(pp, 'gas_bestfit') and pp.gas_bestfit is not None:
                    # 先在子集范围内替换为第二阶段的拟合结果
                    full_bestfit[wave_mask] = pp.bestfit
                    
                    # 为了方便后续处理，也创建完整范围的gas_bestfit
                    full_gas_bestfit = np.zeros_like(spectrum)
                    if has_emission:
                        # 只在子集范围内设置gas_bestfit
                        full_gas_bestfit[wave_mask] = pp.gas_bestfit
                    
                    # 设置pp的gas_bestfit为完整范围版本
                    pp.full_gas_bestfit = full_gas_bestfit
                else:
                    pp.full_gas_bestfit = np.zeros_like(spectrum)
                
                # 设置pp的bestfit为完整范围版本
                pp.full_bestfit = full_bestfit
                
            except Exception as e:
                logging.warning(f"Combined fit failed for pixel ({i},{j}): {str(e)}")
                
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
                    
                    logging.info(f"Used fallback stellar-only fit for pixel ({i},{j})")
                else:
                    raise  # Re-raise the exception if we're not using fallback
        else:
            # 不使用两阶段拟合，只使用第一阶段结果
            logging.debug(f"STEP: Using single-stage fit (two-stage disabled)")
            pp = pp_stars
            pp.full_bestfit = bestfit_stars
            pp.full_gas_bestfit = np.zeros_like(spectrum)
            
            # Add gas attributes manually 
            pp.gas_bestfit = np.zeros_like(spectrum)
            pp.gas_flux = np.zeros(len(gas_names)) if gas_names is not None else np.zeros(1)
            pp.gas_bestfit_templates = np.zeros((spectrum.shape[0], 
                                               len(gas_names) if gas_names is not None else 1))
        
        # Additional safety check after fitting
        if pp is None or not hasattr(pp, 'full_bestfit') or pp.full_bestfit is None:
            logging.warning(f"Missing valid fit results for pixel ({i},{j})")
            return i, j, None
            
        # Calculate S/N
        logging.debug(f"STEP: Calculating S/N and residuals")
        residuals = spectrum - pp.full_bestfit
        rms = robust_sigma(residuals[mask], zero=1)  # Only use masked region
        signal = np.median(spectrum[mask])
        snr = signal / rms if rms > 0 else 0
        logging.debug(f"  - Signal: {signal:.3f}, RMS: {rms:.3f}, S/N: {snr:.1f}")
        
        # 提取发射线信息
        logging.debug(f"STEP: Extracting emission line measurements")
        el_results = {}
        
        if config.compute_emission_lines:
            # 检查是否有气体发射线结果
            has_emission = False
            if hasattr(pp, 'full_gas_bestfit') and pp.full_gas_bestfit is not None:
                has_emission = np.any(np.abs(pp.full_gas_bestfit) > 1e-10)
                
            if has_emission:
                logging.debug(f"  - Gas emission detected")
                
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
                            
                            # 注意：gas_bestfit_templates可能只在子集波长范围内
                            peak = np.max(pp.gas_bestfit_templates[:, idx])
                            an = peak / rms if rms > 0 else 0
                        
                        el_results[name] = {'flux': flux, 'an': an}
                        logging.debug(f"  - {name}: flux={flux:.3e}, A/N={an:.1f}")
            else:
                logging.debug(f"  - No significant gas emission detected")
                
                # 填充空结果
                for name in config.gas_names:
                    el_results[name] = {'flux': 0.0, 'an': 0.0}
        
        # 保存最优模板 - 与原始代码一致
        optimal_template = optimal_stellar_template
        
        # 计算光谱指数
        logging.debug(f"STEP: Calculating spectral indices")
        indices = {}
        
        if config.compute_spectral_indices:
            try:
                # 创建指数计算器
                calculator = LineIndexCalculator(
                    lam_gal, spectrum,
                    sps.lam_temp, Temp_Calu,
                    em_wave=lam_gal,
                    em_flux_list=pp.full_gas_bestfit if hasattr(pp, 'full_gas_bestfit') else None,
                    velocity_correction=to_scalar(pp.sol[0]),
                    continuum_mode=config.continuum_mode)
                
                # 只在真正需要时生成LIC图
                if config.LICplot and not config.no_plots and config.plot_count < config.max_plots:
                    calculator.plot_all_lines()
                    config.plot_count += 1
                
                # 计算请求的光谱指数
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
        
        # 生成诊断图 - 内存优化版本
        if config.make_plots and not config.no_plots and (i * j) % config.plot_every_n == 0 and config.plot_count < config.max_plots:
            try:
                logging.debug(f"STEP: Generating diagnostic plot")
                # 保存第一阶段和第二阶段的结果
                pp.stage1_bestfit = bestfit_stars
                pp.optimal_stellar_template = optimal_stellar_template
                pp.full_bestfit = pp.full_bestfit
                pp.full_gas_bestfit = pp.full_gas_bestfit
                
                # 调用优化版绘图函数
                plot_pixel_fit(i, j, pp, galaxy_data, config)
                
                # 删除临时属性以节省内存
                del pp.stage1_bestfit
                del pp.optimal_stellar_template
                
                # 强制垃圾回收
                import gc
                gc.collect()
                
                logging.debug(f"  - Plot saved successfully")
            except Exception as e:
                logging.warning(f"Failed to create plot for pixel ({i},{j}): {str(e)}")
                import traceback
                logging.debug(traceback.format_exc())
                # 确保清理所有图形
                plt.close('all')
        
        # 汇总结果
        logging.debug(f"STEP: Compiling final results")
        
        # 安全访问pp属性
        sol_0 = 0.0
        sol_1 = 0.0
        if hasattr(pp, 'sol') and pp.sol is not None:
            if len(pp.sol) > 0:
                sol_0 = to_scalar(pp.sol[0])
            if len(pp.sol) > 1:
                sol_1 = to_scalar(pp.sol[1])
        
        # 确保所有需要的数组都存在
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
            'apoly': apoly,  # 保存apoly
            'rms': rms,
            'snr': snr,
            'el_results': el_results,
            'indices': indices,
            'stage1_bestfit': bestfit_stars,  # 保存第一阶段结果
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
    
    # 批处理以减少内存使用
    batch_size = 100  # 每批处理的像素数量
    
    for batch_start in range(0, len(pixels), batch_size):
        batch_end = min(batch_start + batch_size, len(pixels))
        batch_pixels = pixels[batch_start:batch_end]
        
        logging.info(f"Processing batch {batch_start//batch_size + 1}/{(len(pixels)-1)//batch_size + 1} ({len(batch_pixels)} pixels)")
        
        with ProcessPoolExecutor(max_workers=config.n_threads) as executor:
            # Submit batch for processing
            future_to_pixel = {executor.submit(fit_single_pixel, pixel): pixel for pixel in batch_pixels}
            
            # Process results as they complete
            with tqdm(total=len(batch_pixels), desc=f"Batch {batch_start//batch_size + 1}") as pbar:
                for future in as_completed(future_to_pixel):
                    i, j, result = future.result()
                    completed += 1
                    
                    if result is not None:
                        results[(i, j)] = result
                        successful += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    
        # 报告批次完成情况
        batch_elapsed = time.time() - start_time
        rate = completed / batch_elapsed if batch_elapsed > 0 else 0
        eta = (total_pixels - completed) / rate if rate > 0 else 0
        
        logging.info(f"Completed {completed}/{total_pixels} pixels ({successful} successful). "
                    f"Rate: {rate:.1f} pix/s, ETA: {eta/60:.1f} minutes")
        
        # 批次处理完成后强制垃圾回收
        import gc
        gc.collect()
    
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
    创建像素拟合的诊断图，显示两阶段拟合结果 - 内存优化版本
    
    Parameters
    ----------
    i, j : int
        像素坐标
    pp : ppxf object
        pPXF拟合结果
    galaxy_data : IFUDataCube
        包含星系数据的对象
    config : P2PConfig
        配置对象
    """
    # 如果禁用了所有图形或已达到最大绘图数量限制，直接返回
    if config.no_plots or config.plot_count >= config.max_plots:
        return
    
    try:
        logging.debug(f"===== PLOT GENERATION FOR PIXEL ({i},{j}) =====")
        
        # 创建绘图目录
        plot_dir = config.plot_dir / 'P2P_res'
        os.makedirs(plot_dir, exist_ok=True)
        
        # 准备文件名和路径
        plot_path_png = plot_dir / f"{config.galaxy_name}_pixel_{i}_{j}.png"
        
        # 获取数据 - 在创建图像前提前获取所有需要的数据
        k_index = i * galaxy_data.cube.shape[2] + j
        lam_gal = galaxy_data.lam_gal
        spectrum = galaxy_data.spectra[:, k_index]
        
        # 获取拟合结果
        bestfit = pp.full_bestfit if hasattr(pp, 'full_bestfit') else pp.bestfit
        stage1_bestfit = pp.stage1_bestfit if hasattr(pp, 'stage1_bestfit') else bestfit
        gas_bestfit = pp.full_gas_bestfit if hasattr(pp, 'full_gas_bestfit') else np.zeros_like(spectrum)
        
        # 提取需要的属性值 - 避免在图中绘制时再次访问对象
        velocity = to_scalar(pp.sol[0]) if hasattr(pp, 'sol') and pp.sol is not None and len(pp.sol) > 0 else 0.0
        sigma = to_scalar(pp.sol[1]) if hasattr(pp, 'sol') and pp.sol is not None and len(pp.sol) > 1 else 0.0
        chi2 = to_scalar(pp.chi2) if hasattr(pp, 'chi2') and pp.chi2 is not None else 0.0
        
        # 使用with语句创建图形，确保资源正确释放
        with plt.rc_context({'figure.max_open_warning': False}):
            # 创建图形，指定较低的DPI以减少内存使用
            fig = plt.figure(figsize=(12, 8), dpi=config.dpi)
            
            # 创建子图
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax3 = plt.subplot(gs[2])
            
            # 第一个面板：原始数据和第一阶段拟合
            ax1.plot(lam_gal, spectrum, c='k', lw=1, alpha=.8, 
                    label=f"{config.galaxy_name} pixel:[{i},{j}] - Original")
            ax1.plot(lam_gal, stage1_bestfit, '-', c='r', alpha=.8, 
                    label='Stage 1: Stellar fit only')
            
            # 第二个面板：最终拟合结果
            ax2.plot(lam_gal, spectrum, c='k', lw=1, alpha=.8, 
                    label='Original spectrum')
            ax2.plot(lam_gal, bestfit, '-', c='r', alpha=.8, 
                    label='Stage 2: Full fit')
            
            # 绘制恒星成分（总拟合减去气体）
            stellar_comp = bestfit - gas_bestfit
            ax2.plot(lam_gal, stellar_comp, '-', c='g', alpha=.7, lw=0.7, 
                    label='Stellar component')
            
            # 第三个面板：只显示发射线和残差
            # 计算残差
            residuals = spectrum - bestfit
            
            # 绘制零线
            ax3.axhline(0, color='k', lw=0.7, alpha=.5)
            
            # 绘制残差
            ax3.plot(lam_gal, residuals, 'g-', lw=0.8, alpha=.7, 
                    label='Residuals (data - full fit)')
            
            # 绘制发射线
            if np.any(gas_bestfit != 0):
                ax3.plot(lam_gal, gas_bestfit, 'r-', lw=1.2, alpha=0.8,
                      label='Gas component')
            
            # 定义并绘制感兴趣的光谱区域
            spectral_regions = {
                'Hbeta': (4847.875, 4876.625),
                'Fe5015': (4977.750, 5054.000),
                'Mgb': (5160.125, 5192.625),
                '[OIII]': (4997, 5017)
            }
            
            # 在所有面板上标记光谱区域
            for name, (start, end) in spectral_regions.items():
                color = 'orange' if 'OIII' in name else 'lightgray'
                alpha = 0.3 if 'OIII' in name else 0.2
                for ax in [ax1, ax2, ax3]:
                    ax.axvspan(start, end, alpha=alpha, color=color)
                    # 在底部添加标签
                    if ax == ax3:
                        y_pos = ax3.get_ylim()[0] + 0.1 * (ax3.get_ylim()[1] - ax3.get_ylim()[0])
                        ax.text((start + end)/2, y_pos, name, ha='center', va='bottom',
                              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # 设置所有面板的属性
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(4800, 5250)  # 固定X轴范围，减少计算
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(axis='both', which='both', labelsize='x-small', 
                            right=True, top=True, direction='in')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize='small')
            
            # 设置Y轴范围
            y_min = np.min(spectrum) * 0.9
            y_max = np.max(spectrum) * 1.1
            ax1.set_ylim(y_min, y_max)
            ax2.set_ylim(y_min, y_max)
            
            # 为第三个面板设置不同的Y轴范围 - 重点显示发射线和残差
            if np.any(gas_bestfit != 0):
                # 如果有发射线，用发射线的scale
                gas_max = np.max(np.abs(gas_bestfit)) * 3
                res_max = max(np.max(np.abs(residuals)), gas_max)
            else:
                # 否则用残差的scale
                res_max = np.max(np.abs(residuals)) * 3
            
            ax3.set_ylim(-res_max, res_max)
            
            # 设置标签
            ax3.set_xlabel(r'Rest-frame Wavelength [$\AA$]', size=11)
            ax1.set_ylabel('Flux', size=11)
            ax2.set_ylabel('Flux', size=11)
            ax3.set_ylabel('Emission & Residuals', size=11)
            
            # 添加标题
            fig.suptitle(f"Pixel ({i}, {j}) - Two-stage Spectral Fit\nv={velocity:.1f} km/s, σ={sigma:.1f} km/s, χ²={chi2:.3f}", 
                        fontsize=14)
            
            # 紧凑布局以减少图形大小
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # 只保存PNG格式，更节省空间
            plt.savefig(plot_path_png, format='png', dpi=config.dpi, bbox_inches='tight')
            
            # 立即关闭图形并释放资源
            plt.close(fig)
            
            # 增加计数器
            config.plot_count += 1
            
            logging.debug(f"已绘制并保存像素 ({i},{j}) 的图像，当前总数: {config.plot_count}/{config.max_plots}")
        
    except Exception as e:
        logging.error(f"绘制像素 ({i},{j}) 图像时出错: {str(e)}")
        # 确保任何失败的图形也会被关闭
        plt.close('all')


def create_summary_plots(galaxy_data, config):
    """
    创建汇总图，内存优化版本。
    
    Parameters
    ----------
    galaxy_data : IFUDataCube
        Object containing the galaxy data
    config : P2PConfig
        Configuration object
    """
    # 如果禁用了所有图形，直接返回
    if config.no_plots:
        return
    
    try:
        # 创建plots目录
        os.makedirs(config.plot_dir, exist_ok=True)
        
        # 1. 运动学图
        with plt.rc_context({'figure.max_open_warning': False}):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=config.dpi)
            
            # 速度图
            vmax = np.nanpercentile(np.abs(galaxy_data.velfield), 90)
            im0 = axes[0].imshow(galaxy_data.velfield, origin='lower', cmap='RdBu_r', 
                               vmin=-vmax, vmax=vmax)
            axes[0].set_title('Velocity [km/s]')
            plt.colorbar(im0, ax=axes[0])
            
            # 速度弥散度图
            sigma_max = np.nanpercentile(galaxy_data.sigfield, 95)
            im1 = axes[1].imshow(galaxy_data.sigfield, origin='lower', cmap='viridis', 
                               vmin=0, vmax=sigma_max)
            axes[1].set_title('Velocity Dispersion [km/s]')
            plt.colorbar(im1, ax=axes[1])
            
            plt.suptitle(f"{config.galaxy_name} - Stellar Kinematics")
            plt.tight_layout()
            plt.savefig(config.plot_dir / f"{config.galaxy_name}_kinematics.png", dpi=config.dpi)
            plt.close(fig)
        
        # 2. 发射线图
        if config.compute_emission_lines and len(config.gas_names) > 0:
            n_lines = len(config.gas_names)
            
            with plt.rc_context({'figure.max_open_warning': False}):
                fig, axes = plt.subplots(2, n_lines, figsize=(4*n_lines, 8), dpi=config.dpi)
                
                if n_lines == 1:  # 处理单个发射线的情况
                    axes = np.array([[axes[0]], [axes[1]]])
                
                for i, name in enumerate(config.gas_names):
                    # 流量图
                    flux_map = galaxy_data.el_flux_maps[name]
                    vmax = np.nanpercentile(flux_map, 95)
                    im = axes[0, i].imshow(flux_map, origin='lower', cmap='inferno', vmin=0, vmax=vmax)
                    axes[0, i].set_title(f"{name} Flux")
                    plt.colorbar(im, ax=axes[0, i])
                    
                    # 信噪比图
                    snr_map = galaxy_data.el_snr_maps[name]
                    im = axes[1, i].imshow(snr_map, origin='lower', cmap='viridis', vmin=0, vmax=5)
                    axes[1, i].set_title(f"{name} S/N")
                    plt.colorbar(im, ax=axes[1, i])
                
                plt.suptitle(f"{config.galaxy_name} - Emission Lines")
                plt.tight_layout()
                plt.savefig(config.plot_dir / f"{config.galaxy_name}_emission_lines.png", dpi=config.dpi)
                plt.close(fig)
        
        # 3. 谱指数图
        if config.compute_spectral_indices and len(config.line_indices) > 0:
            n_indices = len(config.line_indices)
            n_cols = min(3, n_indices)
            n_rows = (n_indices + n_cols - 1) // n_cols
            
            with plt.rc_context({'figure.max_open_warning': False}):
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), dpi=config.dpi)
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
                
                # 隐藏空的子图
                for i in range(n_indices, n_rows * n_cols):
                    row = i // n_cols
                    col = i % n_cols
                    axes[row, col].axis('off')
                
                plt.suptitle(f"{config.galaxy_name} - Spectral Indices")
                plt.tight_layout()
                plt.savefig(config.plot_dir / f"{config.galaxy_name}_indices.png", dpi=config.dpi)
                plt.close(fig)
        
        # 强制清理
        plt.close('all')
        import gc
        gc.collect()
    
    except Exception as e:
        logging.error(f"创建汇总图时出错: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        plt.close('all')


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
        config.no_plots = True     # Completely disable all plotting
        
    elif problem_type == "gas_fit_error":
        # If having problems with gas fitting
        logging.info("Applying optimizations for gas fitting errors")
        config.fwhm_gas = 2.0  # Use wider lines for better stability
        config.fallback_to_simple_fit = True  # Fall back to stellar-only fit if gas fit fails
        config.use_two_stage_fit = True  # Enable two-stage fitting
        
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
    对单个像素进行测试拟合，用于调试目的。
    
    Parameters
    ----------
    config : P2PConfig or str, optional
        配置对象或配置文件的路径
    i, j : int
        像素坐标（行、列）
    debug_level : int
        调试日志级别
        
    Returns
    -------
    dict
        拟合结果
    """
    # 设置更详细的日志级别
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(debug_level)
    
    try:
        # 初始化配置
        if config is None:
            config = P2PConfig()
        elif isinstance(config, str):
            config = P2PConfig.load(config)
        
        # 启用调试配置
        if debug_level == logging.DEBUG:
            config.safe_mode = True
            config.fallback_to_simple_fit = True
            config.retry_with_degree_zero = True
            config.plot_every_n = 1  # 确保每个像素都绘图
        
        # 创建必要的目录
        config.create_directories()
        
        # 显示测试信息
        logging.info(f"=== 开始测试像素 ({i},{j}) ===")
        logging.info(f"使用配置: {config.galaxy_name}")
        
        # 1. 加载数据 - 与主流程相同
        logging.info("加载数据...")
        galaxy_data = IFUDataCube(config.get_data_path(), config.lam_range_temp, config.redshift, config)
        logging.info(f"数据形状: {galaxy_data.cube.shape}")
        logging.info(f"波长范围: [{galaxy_data.lam_gal[0]:.2f}, {galaxy_data.lam_gal[-1]:.2f}]")
        logging.info(f"有效波长范围: {config.good_wavelength_range}")
        
        # 2. 准备模板 - 与主流程相同
        logging.info("准备恒星和气体模板...")
        sps, gas_templates, gas_names, line_wave = prepare_templates(config, galaxy_data.velscale)
        logging.info(f"恒星模板形状: {sps.templates.shape}")
        logging.info(f"气体模板形状: {gas_templates.shape}")
        logging.info(f"气体发射线: {gas_names}")
        
        # 3. 拟合单个像素 - 直接调用实际使用的函数
        logging.info(f"开始拟合像素 ({i},{j})...")
        args = (i, j, galaxy_data, sps, gas_templates, gas_names, line_wave, config)
        
        # 确保使用相同的拟合代码
        pixel_i, pixel_j, result = fit_single_pixel(args)
        
        # 4. 结果处理
        if result is not None and result.get('success', False):
            logging.info(f"拟合成功! 速度: {result['velocity']:.1f} km/s, 弥散度: {result['sigma']:.1f} km/s")
            
            # 存储结果到数据对象中 - 模拟process_results的部分功能
            galaxy_data.velfield[i, j] = result['velocity']
            galaxy_data.sigfield[i, j] = result['sigma']
            galaxy_data.bestfit_field[:, i, j] = result['bestfit']
            galaxy_data.optimal_templates[:, i, j] = result['optimal_template']
            
            # 存储谱指数
            if config.compute_spectral_indices:
                for name, value in result['indices'].items():
                    if name in galaxy_data.index_maps:
                        galaxy_data.index_maps[name][i, j] = value
                        logging.info(f"  - {name}: {value:.4f}")
            
            # 存储发射线结果
            if config.compute_emission_lines:
                logging.info("发射线结果:")
                for name, data in result['el_results'].items():
                    if name in galaxy_data.el_flux_maps:
                        galaxy_data.el_flux_maps[name][i, j] = data['flux']
                        galaxy_data.el_snr_maps[name][i, j] = data['an']
                        logging.info(f"  - {name}: flux={data['flux']:.4e}, S/N={data['an']:.2f}")
            
            # 保存测试结果到CSV
            test_output_dir = config.output_dir / 'test_results'
            os.makedirs(test_output_dir, exist_ok=True)
            
            # 创建DataFrame并保存为CSV
            row = {
                'i': i, 'j': j, 
                'velocity': result['velocity'], 
                'sigma': result['sigma'],
                'SNR': result['snr']
            }
            
            # 添加发射线数据
            for name, data in result['el_results'].items():
                row[f'{name}_flux'] = data['flux']
                row[f'{name}_SNR'] = data['an']
            
            # 添加光谱指数
            for name, value in result['indices'].items():
                row[f'{name}_index'] = value
            
            df = pd.DataFrame([row])
            csv_path = test_output_dir / f"{config.galaxy_name}_pixel_{i}_{j}_results.csv"
            df.to_csv(csv_path, index=False)
            logging.info(f"测试结果保存到 {csv_path}")
            
            return result
        else:
            logging.error(f"像素 ({i},{j}) 拟合失败")
            logging.info("使用更简单的设置重试...")
            
            # 简化设置并重试
            old_degree = config.degree
            old_moments = config.moments
            old_fwhm_gas = config.fwhm_gas
            
            config.degree = -1  # No additive polynomials 
            config.moments = [2, 2]  # Just velocity and dispersion
            config.fwhm_gas = 2.0  # Wider gas lines
            
            # 重新准备模板
            logging.info("使用简化设置准备模板...")
            sps, gas_templates, gas_names, line_wave = prepare_templates(config, galaxy_data.velscale)
            
            # 再次尝试
            args = (i, j, galaxy_data, sps, gas_templates, gas_names, line_wave, config)
            pixel_i, pixel_j, result = fit_single_pixel(args)
            
            # 恢复原始设置
            config.degree = old_degree
            config.moments = old_moments
            config.fwhm_gas = old_fwhm_gas
            
            if result is not None and result.get('success', False):
                logging.info("使用简化设置的第二次尝试成功")
                return result
            else:
                logging.error("即使使用简化设置也失败了")
                return None
            
    except Exception as e:
        logging.error(f"单像素测试过程中出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None
    
    finally:
        # 恢复原始日志级别
        logging.getLogger().setLevel(original_level)


### ------------------------------------------------- ###
# Main Driver Functions
### ------------------------------------------------- ###

def run_p2p_analysis(config=None, problem_fixes=None, test_mode=False, test_row=None, test_col=None, only_col=None):
    """
    运行完整的P2P分析流程。
    
    Parameters
    ----------
    config : P2PConfig or str, optional
        配置对象或配置文件的路径
    problem_fixes : str, optional
        指定需要修复的问题(mask_error, broadcast_error等)
    test_mode : bool, optional
        是否运行测试模式，仅处理部分数据
    test_row : int, optional
        测试行索引。如果提供，仅处理此行的像素
    test_col : int, optional
        测试列索引。如果与test_row一起提供，仅处理单个像素
    only_col : int, optional
        测试单列模式，仅处理指定列的所有像素
        
    Returns
    -------
    IFUDataCube
        包含所有处理数据和结果的对象
    """
    # 设置配置
    if config is None:
        config = P2PConfig()
    elif isinstance(config, str):
        config = P2PConfig.load(config)
    
    # 应用问题特定优化（如果请求）
    if problem_fixes:
        config = optimize_ppxf_params(config, problem_fixes)
    
    # 创建输出目录
    config.create_directories()
    
    # 设置日志
    log_path = config.output_dir / f"{config.galaxy_name}_p2p_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    # 保存配置
    config.save()
    
    # 记录分析开始信息
    logging.info(f"开始P2P分析: {config.galaxy_name}")
    logging.info(f"数据文件: {config.get_data_path()}")
    logging.info(f"配置设置: stellar_moments={config.moments}, degree={config.degree}, "
               f"mdegree={config.mdegree}, fwhm_gas={config.fwhm_gas}, mask_width={config.mask_width}")
    if hasattr(config, 'good_wavelength_range'):
        logging.info(f"有效波长范围: [{config.good_wavelength_range[0]:.2f}, {config.good_wavelength_range[1]:.2f}]")
    
    # 记录绘图设置
    if config.no_plots:
        logging.info("所有绘图功能已禁用，以减少内存使用")
    else:
        logging.info(f"绘图设置: max_plots={config.max_plots}, plot_every_n={config.plot_every_n}, dpi={config.dpi}")
    
    # 记录测试模式信息
    if test_mode:
        logging.info("运行测试模式 - 仅处理部分数据")
        if test_row is not None and test_col is not None:
            logging.info(f"测试单个像素 ({test_row}, {test_col})")
        elif test_row is not None:
            logging.info(f"测试第 {test_row} 行的所有像素")
    elif only_col is not None:
        logging.info(f"运行单列测试模式 - 仅处理第 {only_col} 列的所有像素")
    
    start_time = time.time()
    
    try:
        # 加载和预处理数据
        logging.info("加载和预处理数据...")
        galaxy_data = IFUDataCube(config.get_data_path(), config.lam_range_temp, config.redshift, config)
        
        # 准备模板
        logging.info("准备恒星和气体模板...")
        sps, gas_templates, gas_names, line_wave = prepare_templates(config, galaxy_data.velscale)
        
        # 运行像素拟合
        logging.info("开始逐像素拟合...")
        ny, nx = galaxy_data.cube.shape[1:3]
        
        # 确定要处理的像素
        if test_mode:
            if test_row is not None and test_col is not None:
                # 测试单个像素
                logging.info(f"测试模式: 仅拟合像素 ({test_row}, {test_col})")
                pixels = [(test_row, test_col, galaxy_data, sps, gas_templates, gas_names, line_wave, config)]
            elif test_row is not None:
                # 测试一行像素
                logging.info(f"测试模式: 仅拟合第 {test_row} 行")
                pixels = [(test_row, j, galaxy_data, sps, gas_templates, gas_names, line_wave, config) 
                         for j in range(nx)]
            else:
                # 默认测试：中心行
                test_row = ny // 2
                logging.info(f"测试模式: 仅拟合中心行 {test_row}")
                pixels = [(test_row, j, galaxy_data, sps, gas_templates, gas_names, line_wave, config) 
                         for j in range(nx)]
                
            # 设置单线程处理测试像素
            results = {}
            for pixel_args in tqdm(pixels, desc="拟合测试像素"):
                i, j, result = fit_single_pixel(pixel_args)
                if result is not None:
                    results[(i, j)] = result
                    
        elif only_col is not None:
            # 单列测试模式：处理指定列的所有像素
            logging.info(f"单列测试模式: 仅拟合第 {only_col} 列")
            col = min(only_col, nx-1)  # 确保列索引在范围内
            pixels = [(i, col, galaxy_data, sps, gas_templates, gas_names, line_wave, config) 
                     for i in range(ny)]
            
            # 设置单线程处理测试像素
            results = {}
            for pixel_args in tqdm(pixels, desc=f"拟合第 {col} 列像素"):
                i, j, result = fit_single_pixel(pixel_args)
                if result is not None:
                    results[(i, j)] = result
        else:
            # 正常运行：处理所有像素
            results = fit_pixel_grid(galaxy_data, sps, gas_templates, gas_names, line_wave, config)
        
        # 处理结果
        logging.info("处理并存储结果...")
        process_results(galaxy_data, results, config)
        
        # 创建汇总图
        if config.make_plots and not config.no_plots:
            logging.info("创建汇总图...")
            create_summary_plots(galaxy_data, config)
        
        # 保存结果到FITS文件
        logging.info("保存图到FITS文件...")
        save_results_to_fits(galaxy_data, config)
        
        # 记录完成信息
        end_time = time.time()
        logging.info(f"P2P分析在 {end_time - start_time:.1f} 秒内完成")
        
        return galaxy_data
        
    except Exception as e:
        logging.error(f"P2P分析中出错: {str(e)}")
        logging.exception("堆栈跟踪:")
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
    
    # 测试模式参数
    test_group = parser.add_argument_group('Test Mode Options')
    test_group.add_argument("--test-mode", action="store_true",
                      help="Run in test mode (limited processing)")
    test_group.add_argument("--test-row", type=int, 
                      help="Row index to test (if omitted, central row is used)")
    test_group.add_argument("--test-col", type=int,
                      help="Column index to test (requires --test-row, tests single pixel)")
    
    # 单列测试模式参数
    test_group.add_argument("--only-col", type=int, 
                      help="Test fitting on a single column (all rows)")
    
    # 单像素测试模式
    test_group.add_argument("--test-pixel", action="store_true",
                      help="Run test on a single pixel")
    test_group.add_argument("--pixel-i", type=int, default=0,
                      help="Row index of pixel to test")
    test_group.add_argument("--pixel-j", type=int, default=0,
                      help="Column index of pixel to test")
    
    # 调试选项
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    # 参数选项
    parser.add_argument("--continuum-mode", type=str, 
                      choices=["auto", "fit", "original"],
                      default="auto",
                      help="Continuum calculation mode for spectral indices")
    parser.add_argument("--two-stage", action="store_true",
                        default="--two-stage",
                      help="Use two-stage fitting strategy (stellar first, then gas)")
    parser.add_argument("--global-search", action="store_true",
                        default="--global-search",
                      help="Use global search in second stage fitting")
    
    # 绘图控制选项
    plot_group = parser.add_argument_group('Plot Control Options')
    plot_group.add_argument("--no-plots", action="store_true",
                      help="Disable all plot generation to save memory")
    plot_group.add_argument("--max-plots", type=int, default=20,
                      help="Maximum number of plots to generate (default: 20)")
    plot_group.add_argument("--plot-dpi", type=int, default=100,
                      help="DPI value for plots (lower values use less memory, default: 100)")
    plot_group.add_argument("--plot-every", type=int, 
                      help="Generate plots every N pixels (default: 300)")
    
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
        # 支持旧的测试方式
        # Load configuration
        config = P2PConfig.load(args.config) if args.config else P2PConfig()
        
        # Apply problem fixes
        if args.fix:
            config = optimize_ppxf_params(config, args.fix)
        
        # 设置绘图选项
        config.no_plots = args.no_plots
        if args.max_plots > 0:
            config.max_plots = args.max_plots
        config.dpi = args.plot_dpi
        if args.plot_every:
            config.plot_every_n = args.plot_every
        config.plot_count = 0  # 重置计数器
        
        # Set continuum mode
        config.continuum_mode = args.continuum_mode
        print(f"Using continuum mode: {config.continuum_mode}")
        
        # Set two-stage fitting mode
        config.use_two_stage_fit = args.two_stage
        print(f"Two-stage fitting: {'Enabled' if config.use_two_stage_fit else 'Disabled'}")
        
        # Set global search option
        config.global_search = args.global_search
        print(f"Global search: {'Enabled' if config.global_search else 'Disabled'}")
            
        # Run single pixel test
        result = test_single_pixel(config, args.pixel_i, args.pixel_j, 
                                 debug_level=log_level)
        
        if result:
            print(f"Single pixel test completed successfully, results saved")
        else:
            print(f"Single pixel test failed, check log file")
    elif args.only_col is not None:
        # 单列测试模式
        if args.config:
            config = P2PConfig.load(args.config)
        else:
            config = P2PConfig()
            
        # Apply problem fixes
        if args.fix:
            config = optimize_ppxf_params(config, args.fix)
            
        # 设置绘图选项
        config.no_plots = args.no_plots
        if args.max_plots > 0:
            config.max_plots = args.max_plots
        config.dpi = args.plot_dpi
        if args.plot_every:
            config.plot_every_n = args.plot_every
        config.plot_count = 0  # 重置计数器
            
        # 设置参数
        config.continuum_mode = args.continuum_mode
        config.use_two_stage_fit = args.two_stage
        config.global_search = args.global_search
        
        print(f"Running single column test for column {args.only_col}")
        print(f"Continuum mode: {config.continuum_mode}")
        print(f"Two-stage fitting: {'Enabled' if config.use_two_stage_fit else 'Disabled'}")
        print(f"Global search: {'Enabled' if config.global_search else 'Disabled'}")
        print(f"Plotting: {'Disabled' if config.no_plots else f'Enabled (max {config.max_plots} plots)'}")
        
        # 运行单列测试
        run_p2p_analysis(config, args.fix, only_col=args.only_col)
    elif args.test_mode:
        # 测试模式
        # Handle configuration
        if args.config:
            config = P2PConfig.load(args.config)
        else:
            config = P2PConfig()
        
        # Apply problem fixes
        if args.fix:
            config = optimize_ppxf_params(config, args.fix)
        
        # 设置绘图选项
        config.no_plots = args.no_plots
        if args.max_plots > 0:
            config.max_plots = args.max_plots
        config.dpi = args.plot_dpi
        if args.plot_every:
            config.plot_every_n = args.plot_every
        config.plot_count = 0  # 重置计数器
        
        # 设置其他选项
        config.continuum_mode = args.continuum_mode
        config.use_two_stage_fit = args.two_stage
        config.global_search = args.global_search
        
        print(f"Running in test mode")
        print(f"Continuum mode: {config.continuum_mode}")
        print(f"Two-stage fitting: {'Enabled' if config.use_two_stage_fit else 'Disabled'}")
        print(f"Global search: {'Enabled' if config.global_search else 'Disabled'}")
        print(f"Plotting: {'Disabled' if config.no_plots else f'Enabled (max {config.max_plots} plots)'}")
        
        # 在测试模式下运行分析
        run_p2p_analysis(config, args.fix, test_mode=True, 
                        test_row=args.test_row, test_col=args.test_col)
    else:
        # Normal full run
        # Handle configuration
        if args.config:
            config = P2PConfig.load(args.config)
        else:
            config = P2PConfig()
        
        # Apply problem fixes
        if args.fix:
            config = optimize_ppxf_params(config, args.fix)
            
        # 设置绘图选项
        config.no_plots = args.no_plots
        if args.max_plots > 0:
            config.max_plots = args.max_plots
        config.dpi = args.plot_dpi
        if args.plot_every:
            config.plot_every_n = args.plot_every
        config.plot_count = 0  # 重置计数器
        
        # Set continuum mode
        config.continuum_mode = args.continuum_mode
        print(f"Using continuum mode: {config.continuum_mode}")
        
        # Set two-stage fitting mode
        config.use_two_stage_fit = args.two_stage
        print(f"Two-stage fitting: {'Enabled' if config.use_two_stage_fit else 'Disabled'}")
        
        # Set global search option
        config.global_search = args.global_search
        print(f"Global search: {'Enabled' if config.global_search else 'Disabled'}")
        print(f"Plotting: {'Disabled' if config.no_plots else f'Enabled (max {config.max_plots} plots)'}")
        
        # Run analysis
        run_p2p_analysis(config, args.fix)