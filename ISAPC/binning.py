"""
Spectral Binning Tools - Support for Voronoi binning and radial binning
"""
import numpy as np
import warnings
from typing import Tuple, Dict, Optional, Union, List

from vorbin.voronoi_2d_binning import voronoi_2d_binning
from utils.parallel import ParallelTqdm
from joblib import delayed
from utils.calc import resample_spectrum, spectres

class VoronoiBinning:
    """Class for Voronoi binning algorithm-based spectral binning"""
    
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        signal: np.ndarray,
        noise: np.ndarray,
        wavelength: np.ndarray,
        spectra: np.ndarray,
        shape: Tuple[int, int],
        pixelsize: float = 1.0
    ):
        """
        Initialize Voronoi binning
        
        Parameters
        ----------
        x : ndarray
            Pixel x coordinate array
        y : ndarray
            Pixel y coordinate array
        signal : ndarray
            Signal strength array
        noise : ndarray
            Noise strength array
        wavelength : ndarray
            Wavelength array
        spectra : ndarray
            All pixel spectra array (wavelength, n_pixels) 
        shape : tuple
            Original image shape (n_y, n_x)
        pixelsize : float, default=1.0
            Pixel size (arcsec)
        """
        self.x = x
        self.y = y
        self.signal = signal
        self.noise = noise
        self.wavelength = wavelength
        self.spectra = spectra
        self.shape = shape
        self.pixelsize = pixelsize
        
        # Calculate SNR
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.snr = np.zeros_like(signal)
            valid_mask = noise > 0
            self.snr[valid_mask] = signal[valid_mask] / noise[valid_mask]
        
        # Initialize result storage
        self.bin_number = None
        self.bin_signal = None
        self.bin_noise = None
        self.bin_spectrum = None
        self.bin_snr = None
        self.bin_x = None
        self.bin_y = None
        self.bin_npixels = None
        self.binned_spectra = None
    
    def compute_bins(self, target_snr: float, quiet: bool = True) -> Dict:
        """
        Compute Voronoi binning
        
        Parameters
        ----------
        target_snr : float
            Target signal-to-noise ratio
        quiet : bool, default=True
            Whether to display detailed information
            
        Returns
        -------
        dict
            Binning result dictionary
        """
        # Use only valid pixels
        valid_mask = (self.snr > 0) & np.isfinite(self.snr)
        x = self.x[valid_mask]
        y = self.y[valid_mask]
        signal = self.signal[valid_mask]
        noise = self.noise[valid_mask]
        
        if len(x) < 10:
            raise ValueError("Not enough valid pixels for binning")
        
        # Perform Voronoi binning
        try:
            bin_number, x_gen, y_gen, bin_x, bin_y, bin_snr, bin_npixels, scale = voronoi_2d_binning(
                x, y, signal, noise, target_snr,
                pixelsize=self.pixelsize, plot=False, quiet=quiet,
                cvt=True, wvt=True
            )
        except Exception as e:
            # Handle errors in the Voronoi algorithm
            warnings.warn(f"Error in Voronoi binning: {str(e)}. Using simple binning.")
            
            # Simple radial binning as fallback
            bin_number = np.zeros_like(x, dtype=int)
            bin_x = np.array([np.mean(x)])
            bin_y = np.array([np.mean(y)])
            bin_snr = np.array([np.mean(signal/noise)])
            bin_npixels = np.array([len(x)])
        
        # Save binning results
        self.bin_number = bin_number
        self.bin_x = bin_x
        self.bin_y = bin_y
        self.bin_snr = bin_snr
        self.bin_npixels = bin_npixels
        
        # Create binning index mapping
        bin_map = np.full(self.shape, -1)
        valid_idx = np.where(valid_mask)[0]
        
        # Map bin numbers back to original 2D grid
        row = self.y.astype(int)
        col = self.x.astype(int)
        
        for i, bin_idx in enumerate(bin_number):
            pixel_idx = valid_idx[i]
            r = row[pixel_idx]
            c = col[pixel_idx]
            
            # Ensure indices are within valid range
            if 0 <= r < self.shape[0] and 0 <= c < self.shape[1]:
                bin_map[r, c] = bin_idx
        
        # Return bin results
        return {
            'bin_map': bin_map,
            'n_bins': int(np.max(bin_number)) + 1 if len(bin_number) > 0 else 0,
            'bin_x': bin_x,
            'bin_y': bin_y,
            'bin_snr': bin_snr,
            'bin_npixels': bin_npixels
        }
    
    def extract_binned_spectra(
        self, 
        bin_map: np.ndarray,
        velocity_field: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """
        Extract combined spectra after binning
        
        Parameters
        ----------
        bin_map : ndarray
            Binning index mapping
        velocity_field : ndarray, optional
            Velocity field, for spectrum alignment correction
            
        Returns
        -------
        dict
            Dictionary of binned spectra, keys are bin indices
        """
        # Calculate bin pixels
        n_bins = int(np.max(bin_map)) + 1
        binned_spectra = {}
        binned_errors = {}
        
        # Recalculate row/column indices
        n_y, n_x = self.shape
        row = (self.y + 0.5).astype(int)  # +0.5 for rounding
        col = (self.x + 0.5).astype(int)
        
        # Collect spectra for each bin
        for bin_idx in range(n_bins):
            # Find all pixels in this bin
            pixel_indices = []
            for i, (r, c) in enumerate(zip(row, col)):
                # Ensure indices are within valid range
                if 0 <= r < n_y and 0 <= c < n_x and bin_map[r, c] == bin_idx:
                    pixel_indices.append(i)
            
            if not pixel_indices:
                continue
                
            # Get spectra
            bin_spectra = self.spectra[:, pixel_indices]
            
            # If velocity field provided, align spectra
            if velocity_field is not None:
                try:
                    # Extract bin velocities
                    bin_velocities = [velocity_field[r, c] if 0 <= r < n_y and 0 <= c < n_x and np.isfinite(velocity_field[r, c]) else 0
                                     for r, c in zip(row[pixel_indices], col[pixel_indices])]
                    
                    # Align and accumulate spectra
                    aligned_spectra = np.zeros_like(bin_spectra)
                    
                    for i, vel in enumerate(bin_velocities):
                        if np.isnan(vel):
                            aligned_spectra[:, i] = bin_spectra[:, i]
                        else:
                            # Calculate wavelength shift
                            z = vel / 299792.458  # c in km/s
                            shifted_wave = self.wavelength * (1 + z)
                            
                            # Interpolate to original wavelength
                            try:
                                aligned_spectra[:, i] = spectres(
                                    self.wavelength, shifted_wave, bin_spectra[:, i],
                                    fill=0
                                )
                            except Exception as e:
                                # Fall back to original spectrum if alignment fails
                                warnings.warn(f"Error aligning spectrum: {str(e)}")
                                aligned_spectra[:, i] = bin_spectra[:, i]
                    
                    # Use aligned spectra
                    bin_spectra = aligned_spectra
                except Exception as e:
                    warnings.warn(f"Error during velocity alignment: {str(e)}. Using original spectra.")
            
            # Calculate bin spectrum (simple average)
            try:
                # First check for NaN values
                valid_mask = np.all(np.isfinite(bin_spectra), axis=0)
                if np.any(valid_mask):
                    binned_spectra[bin_idx] = np.nanmean(bin_spectra[:, valid_mask], axis=1)
                else:
                    # If all spectra have NaNs, create a placeholder spectrum
                    warnings.warn(f"Bin {bin_idx} has no valid spectra")
                    binned_spectra[bin_idx] = np.zeros_like(self.wavelength)
            except Exception as e:
                warnings.warn(f"Error averaging bin {bin_idx}: {str(e)}")
                continue
        
        self.binned_spectra = binned_spectra
        return binned_spectra


class RadialBinning:
    """Radial binning class"""
    
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        signal: np.ndarray,
        noise: np.ndarray,
        wavelength: np.ndarray,
        spectra: np.ndarray,
        shape: Tuple[int, int]
    ):
        """
        Initialize radial binning
        
        Parameters
        ----------
        x : ndarray
            Pixel x coordinate array
        y : ndarray
            Pixel y coordinate array
        signal : ndarray
            Signal strength array
        noise : ndarray
            Noise strength array
        wavelength : ndarray
            Wavelength array
        spectra : ndarray
            All pixel spectra array (wavelength, n_pixels)
        shape : tuple
            Original image shape (n_y, n_x)
        """
        self.x = x
        self.y = y
        self.signal = signal
        self.noise = noise
        self.wavelength = wavelength
        self.spectra = spectra
        self.shape = shape
        
        # Calculate SNR
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.snr = np.zeros_like(signal)
            valid_mask = noise > 0
            self.snr[valid_mask] = signal[valid_mask] / noise[valid_mask]
        
        # Initialize result storage
        self.bin_edges = None
        self.binned_spectra = None
        self.bin_map = None
        self.radial_map = None
    
    def compute_bins(
        self,
        n_bins: int = 10,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        pa: float = 0.0,
        ellipticity: float = 0.0,
        log_spacing: bool = True,
        min_radius: float = 0.0,
        max_radius: Optional[float] = None,
        min_snr: float = 0.0
    ) -> Dict:
        """
        Compute radial binning
        
        Parameters
        ----------
        n_bins : int, default=10
            Number of radial bins
        center_x : float, optional
            Center x coordinate, default is image center
        center_y : float, optional
            Center y coordinate, default is image center
        pa : float, default=0.0
            Position angle (degrees)
        ellipticity : float, default=0.0
            Ellipticity (0-1)
        log_spacing : bool, default=True
            Whether to use logarithmic spacing
        min_radius : float, default=0.0
            Minimum radius
        max_radius : float, optional
            Maximum radius, default uses maximum distance
        min_snr : float, default=0.0
            Minimum SNR requirement
            
        Returns
        -------
        dict
            Binning result dictionary
        """
        n_y, n_x = self.shape
        
        # Set default center
        if center_x is None:
            center_x = n_x / 2
        if center_y is None:
            center_y = n_y / 2
        
        # Calculate distance to center
        x_rel = self.x - center_x
        y_rel = self.y - center_y
        
        # Apply position angle and ellipticity
        try:
            if ellipticity > 0 or pa != 0:
                # Convert to radians
                pa_rad = np.radians(pa)
                
                # Rotate coordinate system
                x_rot = x_rel * np.cos(pa_rad) + y_rel * np.sin(pa_rad)
                y_rot = -x_rel * np.sin(pa_rad) + y_rel * np.cos(pa_rad)
                
                # Apply ellipticity
                b_to_a = 1 - ellipticity
                radius = np.sqrt(x_rot**2 + (y_rot/b_to_a)**2)
            else:
                # Simple Euclidean distance
                radius = np.sqrt(x_rel**2 + y_rel**2)
        except Exception as e:
            warnings.warn(f"Error calculating radii: {str(e)}. Using simple distance.")
            radius = np.sqrt(x_rel**2 + y_rel**2)
        
        # Create radius map
        radial_map = np.full(self.shape, np.nan)
        
        # Recalculate row/column indices
        row = (self.y + 0.5).astype(int)  # +0.5 for rounding
        col = (self.x + 0.5).astype(int)
        
        for i, r in enumerate(radius):
            if 0 <= row[i] < n_y and 0 <= col[i] < n_x:
                radial_map[row[i], col[i]] = r
        
        # Save radius map
        self.radial_map = radial_map
        
        # Set maximum radius
        if max_radius is None:
            valid_radius = radius[np.isfinite(radius)]
            if len(valid_radius) > 0:
                max_radius = np.nanmax(valid_radius)
            else:
                max_radius = 10 * min_radius if min_radius > 0 else 100
        
        # Create radial bin edges
        try:
            if log_spacing:
                # Logarithmic spacing
                bin_edges = np.logspace(
                    np.log10(max(min_radius, 0.5)), 
                    np.log10(max_radius), 
                    n_bins + 1
                )
            else:
                # Linear spacing
                bin_edges = np.linspace(min_radius, max_radius, n_bins + 1)
        except Exception as e:
            warnings.warn(f"Error creating bin edges: {str(e)}. Using default linear spacing.")
            bin_edges = np.linspace(min_radius, max_radius, n_bins + 1)
        
        self.bin_edges = bin_edges
        
        # Create bin mapping
        bin_map = np.full(self.shape, -1)
        valid_mask = (self.snr >= min_snr) & np.isfinite(radius)
        
        # Assign pixels to bins
        for bin_idx in range(n_bins):
            try:
                # Get inner and outer radius
                r_in = bin_edges[bin_idx]
                r_out = bin_edges[bin_idx + 1]
                
                # Assign pixels to bin
                for i, r in enumerate(radius):
                    if r >= r_in and r < r_out and valid_mask[i]:
                        r_idx = row[i]
                        c_idx = col[i]
                        if 0 <= r_idx < n_y and 0 <= c_idx < n_x:
                            bin_map[r_idx, c_idx] = bin_idx
            except Exception as e:
                warnings.warn(f"Error assigning bin {bin_idx}: {str(e)}")
                continue
        
        # Save bin mapping
        self.bin_map = bin_map
        
        # Calculate pixels per bin
        bin_counts = []
        for bin_idx in range(n_bins):
            count = np.sum(bin_map == bin_idx)
            bin_counts.append(count)
        
        # Return bin information
        return {
            'bin_map': bin_map,
            'radial_map': radial_map,
            'bin_edges': bin_edges,
            'bin_counts': bin_counts,
            'center': (center_x, center_y),
            'pa': pa,
            'ellipticity': ellipticity
        }
    
    def extract_binned_spectra(
        self, 
        bin_map: np.ndarray,
        velocity_field: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """
        Extract radial bin combined spectra
        
        Parameters
        ----------
        bin_map : ndarray
            Binning index mapping
        velocity_field : ndarray, optional
            Velocity field, for spectrum alignment correction
            
        Returns
        -------
        dict
            Dictionary of binned spectra, keys are bin indices
        """
        # Calculate bin pixels
        n_bins = int(np.max(bin_map)) + 1
        binned_spectra = {}
        
        # Recalculate row/column indices
        n_y, n_x = self.shape
        row = (self.y + 0.5).astype(int)  # +0.5 for rounding
        col = (self.x + 0.5).astype(int)
        
        # Collect spectra for each bin
        for bin_idx in range(n_bins):
            try:
                # Find all pixels in this bin
                pixel_indices = []
                for i, (r, c) in enumerate(zip(row, col)):
                    # Ensure indices are within valid range
                    if 0 <= r < n_y and 0 <= c < n_x and bin_map[r, c] == bin_idx:
                        pixel_indices.append(i)
                
                if not pixel_indices:
                    continue
                    
                # Get spectra
                bin_spectra = self.spectra[:, pixel_indices]
                
                # If velocity field provided, align spectra
                if velocity_field is not None:
                    try:
                        # Extract bin velocities
                        bin_velocities = [velocity_field[r, c] if 0 <= r < n_y and 0 <= c < n_x and np.isfinite(velocity_field[r, c]) else 0
                                        for r, c in zip(row[pixel_indices], col[pixel_indices])]
                        
                        # Align and accumulate spectra
                        aligned_spectra = np.zeros_like(bin_spectra)
                        
                        for i, vel in enumerate(bin_velocities):
                            if np.isnan(vel):
                                aligned_spectra[:, i] = bin_spectra[:, i]
                            else:
                                # Calculate wavelength shift
                                z = vel / 299792.458  # c in km/s
                                shifted_wave = self.wavelength * (1 + z)
                                
                                try:
                                    # Interpolate to original wavelength using spectres
                                    aligned_spectra[:, i] = spectres(
                                        self.wavelength, shifted_wave, bin_spectra[:, i],
                                        fill=0
                                    )
                                except Exception as e:
                                    # Fall back to original spectrum if alignment fails
                                    warnings.warn(f"Error aligning spectrum: {str(e)}")
                                    aligned_spectra[:, i] = bin_spectra[:, i]
                        
                        # Use aligned spectra
                        bin_spectra = aligned_spectra
                    except Exception as e:
                        warnings.warn(f"Error during velocity alignment: {str(e)}. Using original spectra.")
                
                # Calculate bin spectrum (simple average)
                valid_mask = np.all(np.isfinite(bin_spectra), axis=0)
                if np.any(valid_mask):
                    binned_spectra[bin_idx] = np.nanmean(bin_spectra[:, valid_mask], axis=1)
                else:
                    warnings.warn(f"Bin {bin_idx} has no valid spectra")
                    binned_spectra[bin_idx] = np.zeros_like(self.wavelength)
            except Exception as e:
                warnings.warn(f"Error extracting spectra for bin {bin_idx}: {str(e)}")
                continue
        
        self.binned_spectra = binned_spectra
        return binned_spectra