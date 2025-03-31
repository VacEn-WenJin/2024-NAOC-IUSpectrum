"""
Spectral indices calculation module
Includes Lick indices, D4000, etc.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import warnings
import os
from scipy import interpolate
from scipy.interpolate import interp1d

# Import spectres from utils
from utils.calc import spectres

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 添加警告消息控制
SHOW_WARNINGS = False

def set_warnings(show=True):
    """
    控制是否显示警告消息
    
    Parameters:
    -----------
    show : bool
        如果为True，显示警告；如果为False，抑制警告
    """
    global SHOW_WARNINGS
    SHOW_WARNINGS = show

def warn(message, category=UserWarning):
    """
    根据全局设置发出警告
    
    Parameters:
    -----------
    message : str
        警告消息
    category : Warning, optional
        警告类别，默认为UserWarning
    """
    if SHOW_WARNINGS:
        warnings.warn(message, category)


class LineIndexCalculator:
    def __init__(self, wave, flux, fit_wave, fit_flux, em_wave=None, em_flux_list=None, 
                 velocity_correction=0, error=None, continuum_mode='auto', show_warnings=True):
        """
        Initialize the absorption line index calculator
        
        Parameters:
        -----------
        wave : array-like
            Wavelength array of the original spectrum
        flux : array-like
            Flux array of the original spectrum
        fit_wave : array-like
            Wavelength array of the fitted spectrum, used for continuum calculation
        fit_flux : array-like
            Flux array of the fitted spectrum, used for continuum calculation
        em_wave : array-like, optional
            Wavelength array for emission lines
        em_flux_list : array-like, optional
            Combined emission line spectrum
        velocity_correction : float, optional
            Velocity correction value in km/s, default is 0
        error : array-like, optional
            Error array
        continuum_mode : str, optional
            Continuum calculation mode
            'auto': Use fitted spectrum only when original data is insufficient
            'fit': Always use fitted spectrum
            'original': Use original spectrum when possible (warn when insufficient)
        show_warnings : bool, optional
            Whether to show warnings from this calculator instance
        """

        self.Test_Mode = True # test 1 pixel.
        self.show_warnings = show_warnings

        self.c = 299792.458  # Speed of light in km/s
        self.velocity = velocity_correction
        self.continuum_mode = continuum_mode
        
        # Create copies and ensure finite values
        self.wave = self._apply_velocity_correction(wave)
        self.flux = np.array(flux, copy=True)
        
        # Replace any NaN or Inf values with zeros
        if np.any(~np.isfinite(self.flux)):
            self.flux[~np.isfinite(self.flux)] = 0
            self._warn("Non-finite values in flux array replaced with zeros")
        
        self.fit_wave = fit_wave
        self.fit_flux = fit_flux
        
        # Handle NaNs in fitted flux
        if np.any(~np.isfinite(self.fit_flux)):
            self.fit_flux = np.array(self.fit_flux, copy=True)
            self.fit_flux[~np.isfinite(self.fit_flux)] = 0
            self._warn("Non-finite values in fitted flux array replaced with zeros")
        
        self.error = error if error is not None else np.ones_like(flux)
        
        # Process emission lines
        if em_wave is not None and em_flux_list is not None:
            self.em_wave = self._apply_velocity_correction(em_wave)
            self.em_flux_list = em_flux_list
            
            # Handle NaNs in emission line flux
            if np.any(~np.isfinite(self.em_flux_list)):
                self.em_flux_list = np.array(self.em_flux_list, copy=True)
                self.em_flux_list[~np.isfinite(self.em_flux_list)] = 0
                self._warn("Non-finite values in emission flux array replaced with zeros")
                
            self._subtract_emission_lines()
    
    def _warn(self, message, category=UserWarning):
        """根据实例设置发出警告"""
        if self.show_warnings and SHOW_WARNINGS:
            warnings.warn(message, category)
    
    def _subtract_emission_lines(self):
        """
        Subtract emission lines from the original spectrum
        The input em_flux_list is already a combined result
        """
        # Resample emission line spectrum to the original wavelength grid
        try:
            em_flux_resampled = spectres(self.wave, self.em_wave, self.em_flux_list)
            
            # Verify the result is valid before subtraction
            if np.any(~np.isfinite(em_flux_resampled)):
                self._warn("Non-finite values in resampled emission flux, replacing with zeros")
                em_flux_resampled[~np.isfinite(em_flux_resampled)] = 0
                
            # Subtract emission lines from the original spectrum
            self.flux -= em_flux_resampled
        except Exception as e:
            self._warn(f"Error subtracting emission lines: {str(e)}. Continuing without emission line subtraction.")
    
    def _apply_velocity_correction(self, wave):
        """
        Apply velocity correction to the wavelength
        
        Parameters:
        -----------
        wave : array-like
            Original wavelength array
            
        Returns:
        --------
        array-like : Corrected wavelength array
        """
        return wave / (1 + (self.velocity/self.c))

    def _check_data_coverage(self, wave_range):
        """
        Check if the original data completely covers the given wavelength range
        
        Parameters:
        -----------
        wave_range : tuple
            (min_wave, max_wave)
            
        Returns:
        --------
        bool : Whether the range is fully covered
        """
        return (wave_range[0] >= np.min(self.wave)) and (wave_range[1] <= np.max(self.wave))
        
    def define_line_windows(self, line_name):
        """
        Define absorption line and continuum windows
        
        Parameters:
        -----------
        line_name : str
            Absorption line name
                
        Returns:
        --------
        dict : Dictionary with blue, line, and red windows
        """
        # Try to get definition from config file first
        try:
            from config_manager import get_spectral_line_definition
            windows = get_spectral_line_definition(line_name)
            if windows is not None:
                if self.show_warnings:
                    logger.debug(f"Using line definition from config for {line_name}")
                return windows
        except ImportError:
            if self.show_warnings:
                logger.debug("Config manager not available, using built-in line definitions")
        except Exception as e:
            if self.show_warnings:
                logger.debug(f"Error getting line definition from config: {e}")
        
        # Fall back to hardcoded definitions
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
                'blue': (5233.2, 5248.2),
                'line': (5245.7, 5285.7),
                'red': (5285.7, 5318.2)
            },
            'Fe5335': {
                'blue': (5304.6, 5315.9),
                'line': (5312.1, 5352.1),
                'red': (5353.4, 5363.4)
            }
        }
        
        return windows.get(line_name)

    def calculate_pseudo_continuum(self, wave_range, flux_range, region_type):
        """
        Calculate pseudo-continuum
        
        Parameters:
        -----------
        wave_range : tuple or array-like
            Wavelength range
        flux_range : array-like or None
            Corresponding flux values (not needed if using fitted spectrum)
        region_type : str
            Region type ('blue' or 'red')
            
        Returns:
        --------
        float : Pseudo-continuum value
        """
        if self.continuum_mode == 'fit':
            # Use fitted spectrum
            try:
                mask = (self.fit_wave >= wave_range[0]) & (self.fit_wave <= wave_range[1])
                if np.any(mask):
                    return np.nanmedian(self.fit_flux[mask])
                else:
                    self._warn(f"No fitted data points in {region_type} continuum range")
                    return 0
            except Exception as e:
                self._warn(f"Error calculating continuum from fitted spectrum: {str(e)}")
                return 0
        
        elif self.continuum_mode == 'auto':
            # Check original data coverage
            if self._check_data_coverage(wave_range):
                mask = (self.wave >= wave_range[0]) & (self.wave <= wave_range[1])
                if np.any(mask):
                    return np.nanmedian(self.flux[mask])
                else:
                    self._warn(f"No data points in {region_type} continuum range")
                    return 0
            else:
                # Use fitted spectrum when data is insufficient
                mask = (self.fit_wave >= wave_range[0]) & (self.fit_wave <= wave_range[1])
                if np.any(mask):
                    return np.nanmedian(self.fit_flux[mask])
                else:
                    self._warn(f"No fitted data points in {region_type} continuum range")
                    return 0
        
        else:  # 'original'
            if not self._check_data_coverage(wave_range):
                self._warn(f"Original data insufficient to cover {region_type} continuum region, returning 0")
                return 0
            mask = (self.wave >= wave_range[0]) & (self.wave <= wave_range[1])
            if np.any(mask):
                return np.nanmedian(self.flux[mask])
            else:
                self._warn(f"No data points in {region_type} continuum range")
                return 0

    def calculate_index(self, line_name, return_error=False):
        """
        Calculate absorption line index
        
        Parameters:
        -----------
        line_name : str
            Absorption line name ('Hbeta', 'Mgb', 'Fe5015', etc.)
        return_error : bool
            Whether to return error
            
        Returns:
        --------
        float : Absorption line index value
        float : Error value (if return_error=True)
        """
        # Get window definition
        windows = self.define_line_windows(line_name)
        if windows is None:
            self._warn(f"Unknown absorption line: {line_name}")
            return np.nan if not return_error else (np.nan, np.nan)

        # Check for required keys in window definition
        if 'blue' not in windows or 'red' not in windows:
            self._warn(f"Incomplete window definition for {line_name}. Missing blue or red bands.")
            return np.nan if not return_error else (np.nan, np.nan)
            
        # Handle both 'line' and 'band' keys for compatibility
        # Config system uses 'band', older code uses 'line'
        if 'band' in windows:
            line_range = windows['band']
        elif 'line' in windows:
            line_range = windows['line']
        else:
            self._warn(f"Window definition for {line_name} has no line/band range.")
            return np.nan if not return_error else (np.nan, np.nan)

        # Get line region data
        try:
            line_mask = (self.wave >= line_range[0]) & (self.wave <= line_range[1])
            line_wave = self.wave[line_mask]
            line_flux = self.flux[line_mask]
            line_err = self.error[line_mask]

            # Check data points
            if len(line_flux) < 3:
                self._warn(f"Insufficient data points for {line_name} line region")
                return np.nan if not return_error else (np.nan, np.nan)

            # Calculate continuum
            blue_cont = self.calculate_pseudo_continuum(windows['blue'], None, 'blue')
            red_cont = self.calculate_pseudo_continuum(windows['red'], None, 'red')
            
            # Check for valid continuum values
            if blue_cont <= 0 or red_cont <= 0 or not np.isfinite(blue_cont) or not np.isfinite(red_cont):
                self._warn(f"Invalid continuum values for {line_name}: blue={blue_cont}, red={red_cont}")
                return np.nan if not return_error else (np.nan, np.nan)
            
            wave_cont = np.array([
                np.mean(windows['blue']), 
                np.mean(windows['red'])
            ])
            flux_cont = np.array([blue_cont, red_cont])
            
            # Linear interpolation for continuum
            f_interp = interpolate.interp1d(wave_cont, flux_cont)
            cont_at_line = f_interp(line_wave)

            # Check for division by zero
            if np.any(cont_at_line <= 0):
                self._warn(f"Zero or negative continuum values at line wavelengths for {line_name}")
                return np.nan if not return_error else (np.nan, np.nan)

            # Calculate integral
            index = np.trapz((1.0 - line_flux/cont_at_line), line_wave)
            
            if return_error:
                # Calculate error
                error = np.sqrt(np.trapz((line_err/cont_at_line)**2, line_wave))
                return index, error
            
            return index
        except Exception as e:
            self._warn(f"Error calculating {line_name} index: {str(e)}")
            return np.nan if not return_error else (np.nan, np.nan)

    def calculate_all_indices(self):
        """
        Calculate all defined spectral indices
        
        Returns:
        --------
        dict : Dictionary of index values
        """
        result = {}
        for line_name in ['Hbeta', 'Mgb', 'Fe5015'
                        # , 'Fe5270', 'Fe5335'
                        ]:
            try:
                index = self.calculate_index(line_name)
                if not np.isnan(index):
                    result[line_name] = index
            except Exception as e:
                logger.debug(f"Error calculating {line_name}: {str(e)}")
        
        return result

    def plot_all_lines(self, mode=None, number=None, save_path=None, show_index=False):
        """
        Plot all spectral lines in a complete figure
        
        Parameters:
        -----------
        mode : str, optional
            Figure mode, must be one of 'P2P', 'VNB', 'RNB', or 'MUSE'
        number : int, optional
            Figure number, must be an integer
        save_path : str, optional
            Path to save the figure. If provided, the figure will be saved there
        show_index : bool, optional
            Whether to show index parameters, default is False
                
        Returns:
        --------
        fig, axes : Figure and Axes objects for further customization
        """
        # Validate mode and number parameters
        if mode is not None and number is not None:
            valid_modes = ['P2P', 'VNB', 'RNB', 'MUSE']
            if mode not in valid_modes:
                self._warn(f"Mode must be one of {valid_modes}, got {mode}")
                mode = None
            if not isinstance(number, int):
                self._warn(f"Number must be an integer, got {type(number)}")
                number = 0
            mode_title = f"{mode}{number}" if mode is not None else None
        else:
            mode_title = None

        # Get all defined spectral lines
        all_windows = {name: self.define_line_windows(name) 
                    for name in ['Hbeta', 'Mgb', 'Fe5015', 'Fe5270', 'Fe5335']
                    if self.define_line_windows(name) is not None}
        
        # Set fixed X-axis range
        min_wave = 4800
        max_wave = 5250
        
        # Create figure and overall title
        fig = plt.figure(figsize=(15, 12))
        if mode_title:
            fig.suptitle(mode_title, fontsize=16, y=0.95)
        
        # Create subplots, adjust height ratios to accommodate overall title
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.2)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Set unified color scheme
        colors = {
            'blue': 'tab:blue',
            'line': 'tab:green',  # Used for 'band' or 'line' window
            'band': 'tab:green',  # Add this key for compatibility 
            'red': 'tab:red',
            'orig_cont': 'tab:orange',   # Original spectrum continuum color (orange)
            'fit_cont': 'tab:green',     # Fitted spectrum continuum color (green)
            'inactive_cont': 'gray'      # Inactive continuum color
        }
        
        # First panel: Original data comparison
        wave_mask = (self.wave >= min_wave) & (self.wave <= max_wave)
        fit_mask = (self.fit_wave >= min_wave) & (self.fit_wave <= max_wave)
        
        # Calculate y-axis range with improved error handling
        if hasattr(self, 'em_flux_list'):
            try:
                em_flux_resampled = spectres(self.wave, self.em_wave, self.em_flux_list)
                flux_range = self.flux[wave_mask] + em_flux_resampled[wave_mask]
            except Exception as e:
                flux_range = self.flux[wave_mask]
                self._warn(f"Error resampling emission flux: {str(e)}")
        else:
            flux_range = self.flux[wave_mask]
        fit_range = self.fit_flux[fit_mask]
        
        # Ensure we have valid flux values
        valid_flux = flux_range[np.isfinite(flux_range)]
        valid_fit = fit_range[np.isfinite(fit_range)]
        
        if len(valid_flux) > 0 and len(valid_fit) > 0:
            y_min = min(np.nanmin(valid_flux), np.nanmin(valid_fit)) * 0.9
            y_max = max(np.nanmax(valid_flux), np.nanmax(valid_fit)) * 1.1
        else:
            # Default y-axis range if valid data cannot be determined
            y_min = -1
            y_max = 1
            self._warn("Could not determine valid flux range for plotting. Using default range.")
        
        # Ensure y_min and y_max are valid (not NaN or Inf)
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min >= y_max:
            y_min = -1
            y_max = 1
            self._warn("Invalid y-axis limits detected. Using default range.")
        
        # Plot spectra
        if hasattr(self, 'em_flux_list'):
            try:
                em_flux_resampled = spectres(self.wave, self.em_wave, self.em_flux_list)
                if not np.all(np.isfinite(em_flux_resampled)):
                    em_flux_resampled = np.zeros_like(self.wave)
                ax1.plot(self.wave, self.flux + em_flux_resampled, color='tab:blue', 
                        label='Original Spectrum', alpha=0.8)
                ax1.plot(self.wave, em_flux_resampled, color='tab:orange', 
                        label='Emission Lines', alpha=0.8)
            except Exception as e:
                ax1.plot(self.wave, self.flux, color='tab:blue', 
                        label='Original Spectrum', alpha=0.8)
                self._warn(f"Error plotting emission components: {str(e)}")
        else:
            ax1.plot(self.wave, self.flux, color='tab:blue', 
                    label='Original Spectrum', alpha=0.8)
        ax1.plot(self.fit_wave, self.fit_flux, color='tab:red', 
                label='Template Fit', alpha=0.8)
        
        # Calculate y-axis range for second panel with error handling
        processed_flux = self.flux[wave_mask]
        fit_flux_range = self.fit_flux[fit_mask]
        
        # Ensure valid data for plot limits
        valid_proc = processed_flux[np.isfinite(processed_flux)]
        valid_fit = fit_flux_range[np.isfinite(fit_flux_range)]
        
        if len(valid_proc) > 0 and len(valid_fit) > 0:
            y_min_processed = min(np.nanmin(valid_proc), np.nanmin(valid_fit)) * 0.9
            y_max_processed = max(np.nanmax(valid_proc), np.nanmax(valid_fit)) * 1.1
        else:
            y_min_processed = -1
            y_max_processed = 1
            self._warn("Could not determine valid flux range for processed panel. Using default range.")
        
        # Ensure valid limits
        if not np.isfinite(y_min_processed) or not np.isfinite(y_max_processed) or y_min_processed >= y_max_processed:
            y_min_processed = -1
            y_max_processed = 1
            self._warn("Invalid y-axis limits for processed panel. Using default range.")
        
        # Second panel: Processed spectrum
        ax2.plot(self.wave, self.flux, color='tab:blue', 
                label='Processed Spectrum', alpha=0.8)
        ax2.plot(self.fit_wave, self.fit_flux, '--', color='tab:red',
                label='Template Fit', alpha=0.8)
        
        # Mark all spectral line regions in both panels
        for line_name, windows in all_windows.items():
            if windows is None:
                continue
                
            for panel in [ax1, ax2]:
                # Mark blue, line/band, and red regions
                alpha = 0.2  # Transparency
                
                # Check for all required keys in windows
                if 'blue' not in windows or ('line' not in windows and 'band' not in windows) or 'red' not in windows:
                    self._warn(f"Incomplete window definition for {line_name}. Skipping.")
                    continue
                    
                # Get line/band range - handle both 'line' and 'band' key for compatibility
                band_range = windows.get('band', windows.get('line', None))
                if band_range is None:
                    self._warn(f"Missing band/line range for {line_name}. Skipping.")
                    continue
                    
                # Only include in legend once for each region type
                if line_name == list(all_windows.keys())[0]:  # First spectral line for legend
                    panel.axvspan(windows['blue'][0], windows['blue'][1], 
                                alpha=alpha, color=colors['blue'], label='Blue window')
                    panel.axvspan(band_range[0], band_range[1], 
                                alpha=alpha, color=colors['line'], label='Line region')
                    panel.axvspan(windows['red'][0], windows['red'][1], 
                                alpha=alpha, color=colors['red'], label='Red window')
                else:
                    panel.axvspan(windows['blue'][0], windows['blue'][1], 
                                alpha=alpha, color=colors['blue'])
                    panel.axvspan(band_range[0], band_range[1], 
                                alpha=alpha, color=colors['line'])
                    panel.axvspan(windows['red'][0], windows['red'][1], 
                                alpha=alpha, color=colors['red'])
                
                # Add text annotation at the bottom
                if panel == ax1:
                    y_text = y_min + 0.05 * (y_max - y_min)
                else:
                    y_text = y_min_processed + 0.05 * (y_max_processed - y_min_processed)
                
                # Basic label
                panel.text(np.mean(band_range), y_text, line_name,
                        horizontalalignment='center', verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
                # Add continuum points in second panel
                if panel == ax2:
                    # Calculate continuum points
                    blue_cont_orig = None
                    red_cont_orig = None
                    blue_cont_fit = None
                    red_cont_fit = None
                    
                    # Check if original spectrum can be used
                    try:
                        if self._check_data_coverage(windows['blue']):
                            mask = (self.wave >= windows['blue'][0]) & (self.wave <= windows['blue'][1])
                            if np.any(mask):
                                blue_cont_orig = np.nanmedian(self.flux[mask])
                        if self._check_data_coverage(windows['red']):
                            mask = (self.wave >= windows['red'][0]) & (self.wave <= windows['red'][1])
                            if np.any(mask):
                                red_cont_orig = np.nanmedian(self.flux[mask])
                    except Exception as e:
                        self._warn(f"Error calculating original continuum: {str(e)}")
                    
                    # Calculate fitted spectrum continuum points
                    try:
                        mask_blue = (self.fit_wave >= windows['blue'][0]) & (self.fit_wave <= windows['blue'][1])
                        mask_red = (self.fit_wave >= windows['red'][0]) & (self.fit_wave <= windows['red'][1])
                        if np.any(mask_blue):
                            blue_cont_fit = np.nanmedian(self.fit_flux[mask_blue])
                        if np.any(mask_red):
                            red_cont_fit = np.nanmedian(self.fit_flux[mask_red])
                    except Exception as e:
                        self._warn(f"Error calculating fitted continuum: {str(e)}")
                    
                    # Skip if we don't have valid continuum points
                    if blue_cont_fit is None or red_cont_fit is None or not np.isfinite(blue_cont_fit) or not np.isfinite(red_cont_fit):
                        continue
                    
                    wave_cont = np.array([
                        np.mean(windows['blue']), 
                        np.mean(windows['red'])
                    ])

                    # Determine which continuum is active based on calculation mode
                    is_orig_active = (self.continuum_mode == 'original' or 
                                    (self.continuum_mode == 'auto' and 
                                    blue_cont_orig is not None and red_cont_orig is not None and
                                    np.isfinite(blue_cont_orig) and np.isfinite(red_cont_orig)))
                    
                    # Plot original spectrum continuum points and line (if available)
                    if blue_cont_orig is not None and red_cont_orig is not None and np.isfinite(blue_cont_orig) and np.isfinite(red_cont_orig):
                        flux_cont_orig = np.array([blue_cont_orig, red_cont_orig])
                        if not is_orig_active:
                            # Inactive state
                            panel.plot(wave_cont, flux_cont_orig, '*', color=colors['inactive_cont'], 
                                    markersize=10, alpha=0.5,
                                    label='Original spectrum continuum (inactive)' if line_name == list(all_windows.keys())[0] else '')
                            panel.plot(wave_cont, flux_cont_orig, '--', color=colors['inactive_cont'], 
                                    alpha=0.5)
                        else:
                            # Active state
                            panel.plot(wave_cont, flux_cont_orig, '*', color=colors['orig_cont'], 
                                    markersize=10, alpha=0.8,
                                    label='Original spectrum continuum (orange)' if line_name == list(all_windows.keys())[0] else '')
                            panel.plot(wave_cont, flux_cont_orig, '--', color=colors['orig_cont'], 
                                    alpha=0.8)

                    # Plot fitted spectrum continuum points and line
                    if blue_cont_fit is not None and red_cont_fit is not None and np.isfinite(blue_cont_fit) and np.isfinite(red_cont_fit):
                        flux_cont_fit = np.array([blue_cont_fit, red_cont_fit])
                        if is_orig_active:
                            # Inactive state
                            panel.plot(wave_cont, flux_cont_fit, '*', color=colors['inactive_cont'], 
                                    markersize=10, alpha=0.5,
                                    label='Template continuum (inactive)' if line_name == list(all_windows.keys())[0] else '')
                            panel.plot(wave_cont, flux_cont_fit, '--', color=colors['inactive_cont'], 
                                    alpha=0.5)
                        else:
                            # Active state
                            panel.plot(wave_cont, flux_cont_fit, '*', color=colors['fit_cont'], 
                                    markersize=10, alpha=0.8,
                                    label='Template continuum (green)' if line_name == list(all_windows.keys())[0] else '')
                            panel.plot(wave_cont, flux_cont_fit, '--', color=colors['fit_cont'], 
                                    alpha=0.8)

                    # Show index parameters if requested
                    if show_index:
                        try:
                            # Save current continuum_mode
                            original_mode = self.continuum_mode
                            
                            # Calculate index using original spectrum
                            self.continuum_mode = 'original'
                            try:
                                orig_index = self.calculate_index(line_name)
                                if np.isnan(orig_index):
                                    orig_index = None
                            except Exception:
                                orig_index = None
                            
                            # Calculate index using fitted spectrum
                            self.continuum_mode = 'fit'
                            try:
                                fit_index = self.calculate_index(line_name)
                                if np.isnan(fit_index):
                                    fit_index = None
                            except Exception:
                                fit_index = None
                            
                            # Restore original continuum_mode
                            self.continuum_mode = original_mode
                            
                            # Calculate text position
                            base_y_text = y_text + 0.05 * (y_max_processed - y_min_processed)
                            
                            # Build display text
                            if orig_index is not None and fit_index is not None:
                                # Show both values separately
                                y_offset = 0.1 * (y_max_processed - y_min_processed)
                                
                                # Show original spectrum value (top)
                                panel.text(np.mean(band_range), 
                                        base_y_text + y_offset,
                                        f"{orig_index:.3f}", 
                                        color=colors['orig_cont'], 
                                        horizontalalignment='center',
                                        verticalalignment='bottom', 
                                        fontsize='x-small',
                                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                                
                                # Show fitted spectrum value (bottom)
                                panel.text(np.mean(band_range), 
                                        base_y_text + y_offset/2,
                                        f"{fit_index:.3f}", 
                                        color=colors['fit_cont'], 
                                        horizontalalignment='center',
                                        verticalalignment='bottom', 
                                        fontsize='x-small',
                                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                                
                            elif fit_index is not None:
                                # Only show fitted spectrum value
                                fit_text = f"{fit_index:.3f}"
                                panel.text(np.mean(band_range), 
                                        base_y_text + 0.02 * (y_max_processed - y_min_processed),
                                        fit_text, 
                                        color=colors['fit_cont'], 
                                        horizontalalignment='center',
                                        verticalalignment='bottom', 
                                        fontsize='x-small',
                                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                        
                        except Exception as e:
                            self._warn(f"Error calculating index for {line_name}: {e}")
        
        # Set panel properties
        ax1.set_xlim(min_wave, max_wave)
        ax1.set_ylim(y_min, y_max)
        ax2.set_xlim(min_wave, max_wave)
        ax2.set_ylim(y_min_processed, y_max_processed)
        
        # Set common properties for both panels
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
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path and mode_title:
            # Ensure save_path exists
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            # Build complete file path
            filepath = os.path.join(save_path, f"{mode_title}.pdf")
            
            try:
                # Save figure
                plt.savefig(filepath, format='pdf', bbox_inches='tight')
                print(f"Figure saved as: {filepath}")
            except Exception as e:
                self._warn(f"Error saving figure: {str(e)}")
        
        return fig, [ax1, ax2]

def calculate_lick_indices(wave, flux, index_definitions=None, indices_list=None, show_warnings=True):
    """
    Calculate Lick indices for a single spectrum
    
    Parameters:
    -----------
    wave : ndarray
        Wavelength array
    flux : ndarray
        Flux array
    index_definitions : dict, optional
        Index definition dictionary
    indices_list : list of str, optional
        List of indices to calculate
    show_warnings : bool, optional
        Whether to show warnings
        
    Returns:
    --------
    dict : Dictionary of index values
    """
    # Create LineIndexCalculator
    calculator = LineIndexCalculator(
        wave=wave, 
        flux=flux, 
        fit_wave=wave, 
        fit_flux=flux, 
        continuum_mode='original',
        show_warnings=show_warnings
    )
    
    # Calculate all indices
    return calculator.calculate_all_indices()


def calculate_D4000(wave, flux):
    """
    Calculate 4000 Å break strength
    
    Parameters:
    -----------
    wave : ndarray
        Wavelength array
    flux : ndarray
        Flux array
        
    Returns:
    --------
    float : D4000 value
    """
    # Define blue and red regions
    blue_band = (3750.0, 3950.0)
    red_band = (4050.0, 4250.0)
    
    # Find wavelength indices
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        blue_idx = np.where((wave >= blue_band[0]) & (wave <= blue_band[1]))[0]
        red_idx = np.where((wave >= red_band[0]) & (wave <= red_band[1]))[0]
        
        if len(blue_idx) == 0 or len(red_idx) == 0:
            return np.nan
        
        # Calculate mean flux in blue and red regions
        blue_flux = np.nanmean(flux[blue_idx])
        red_flux = np.nanmean(flux[red_idx])
        
        # Calculate D4000 as ratio of red to blue flux
        if blue_flux <= 0 or not np.isfinite(blue_flux) or not np.isfinite(red_flux):
            return np.nan
        
        return red_flux / blue_flux