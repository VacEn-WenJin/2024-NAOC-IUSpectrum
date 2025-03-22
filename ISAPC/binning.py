"""
Spectral Binning Tools - Support for Voronoi binning and radial binning
"""
import numpy as np
import warnings
import logging
from typing import Tuple, Dict, Optional, Union, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path
import time
import os
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
from scipy.interpolate import interp1d
# Import utilities
from utils.calc import apply_velocity_shift, spectres

logger = logging.getLogger(__name__)

# Speed of light in km/s
C_KMS = 299792.458

def to_p2p_compatible(self, original_cube=None):
    """Convert binned data to p2p compatible format with proper velocity scale"""
    # Get dimensions
    n_wave = len(self.wavelength)
    n_bins = len(self.bin_indices)
    
    # Create pseudo-cube with shape [n_wave, 1, n_bins]
    pseudo_cube = np.zeros((n_wave, 1, n_bins))
    for i in range(n_bins):
        pseudo_cube[:, 0, i] = self.spectra[:, i]
    
    # Create variance cube (simple estimate)
    pseudo_variance = np.ones_like(pseudo_cube) * 0.01
    
    # Create x, y coordinates (just bin indices)
    x = np.arange(n_bins)
    y = np.zeros(n_bins)
    
    # Properly calculate velocity scale (matching ppxf requirements)
    c = 299792.458  # Speed of light in km/s
    ln_lambda = np.log(self.wavelength)
    dln_lambda = np.diff(ln_lambda)
    if len(dln_lambda) > 0:
        # Use median to be robust
        velscale = c * np.median(dln_lambda)
    else:
        # Fallback
        velscale = 50.0
    
    # Create result dict with all necessary cube attributes
    result = {
        'cube': pseudo_cube,
        'variance': pseudo_variance,
        'wavelength': self.wavelength,
        'bin_num': self.bin_num,
        'bin_indices': self.bin_indices,
        'x': x,
        'y': y,
        'metadata': self.metadata,
        'velscale': velscale  # Add the properly calculated velscale
    }
    
    # Add useful information from original cube if available
    if original_cube is not None:
        if hasattr(original_cube, '_redshift'):
            result['redshift'] = original_cube._redshift
        if hasattr(original_cube, '_pxl_size_x'):
            result['pxl_size_x'] = original_cube._pxl_size_x
            result['pxl_size_y'] = original_cube._pxl_size_y
    
    return result

@dataclass
class BinnedSpectra:
    """Base class for binned spectra data"""
    bin_num: np.ndarray          # Bin number for each pixel
    bin_indices: List[np.ndarray] # List of arrays containing pixel indices for each bin
    spectra: np.ndarray          # 2D array of binned spectra [n_wavelength, n_bins]
    wavelength: np.ndarray       # 1D array of wavelength
    metadata: Dict               # Metadata dictionary
    
    def save(self, filename: Union[str, Path]) -> None:
        """Save binned data to file"""
        np.savez_compressed(
            filename, 
            bin_num=self.bin_num,
            bin_indices=self.bin_indices,
            spectra=self.spectra,
            wavelength=self.wavelength,
            metadata=self.metadata
        )
        logger.info(f"Saved binned data to {filename}")
    
    @classmethod
    def load(cls, filename: Union[str, Path]):
        """Load binned data from file"""
        data = np.load(filename, allow_pickle=True)
        return cls(
            bin_num=data['bin_num'],
            bin_indices=data['bin_indices'].tolist(),
            spectra=data['spectra'],
            wavelength=data['wavelength'],
            metadata=data['metadata'].item()
        )
    
    def to_p2p_compatible(self, original_cube=None):
        """Convert to p2p compatible format"""
        # Get dimensions
        n_wave = len(self.wavelength)
        n_bins = len(self.bin_indices)
        
        # Create pseudo-cube with shape [n_wave, 1, n_bins]
        pseudo_cube = np.zeros((n_wave, 1, n_bins))
        for i in range(n_bins):
            pseudo_cube[:, 0, i] = self.spectra[:, i]
        
        # Create variance cube (simple estimate)
        pseudo_variance = np.ones_like(pseudo_cube) * 0.01
        
        # Create x, y coordinates (just bin indices)
        x = np.arange(n_bins)
        y = np.zeros(n_bins)
        
        # Create result dict with all necessary cube attributes
        result = {
            'cube': pseudo_cube,
            'variance': pseudo_variance,
            'wavelength': self.wavelength,
            'bin_num': self.bin_num,
            'bin_indices': self.bin_indices,
            'x': x,
            'y': y,
            'metadata': self.metadata
        }
        
        # Add useful information from original cube if available
        if original_cube is not None:
            if hasattr(original_cube, '_redshift'):
                result['redshift'] = original_cube._redshift
            if hasattr(original_cube, '_pxl_size_x'):
                result['pxl_size_x'] = original_cube._pxl_size_x
                result['pxl_size_y'] = original_cube._pxl_size_y
        
        # Create a proper cube object if the original cube has a constructor
        if original_cube is not None and hasattr(original_cube, '__class__'):
            try:
                # Try to create a new cube of the same class
                cube_class = original_cube.__class__
                # Create a new instance and copy over attributes
                new_cube = cube_class.__new__(cube_class)
                
                # Copy over basic attributes
                new_cube._lambda_gal = self.wavelength.copy()
                new_cube._spectra = pseudo_cube.reshape(n_wave, n_bins)
                new_cube._variance = pseudo_variance.reshape(n_wave, n_bins)
                new_cube._n_y = 1
                new_cube._n_x = n_bins
                
                # Copy over key attributes from original cube
                if hasattr(original_cube, '_redshift'):
                    new_cube._redshift = original_cube._redshift
                if hasattr(original_cube, '_pxl_size_x'):
                    new_cube._pxl_size_x = original_cube._pxl_size_x
                    new_cube._pxl_size_y = original_cube._pxl_size_y
                if hasattr(original_cube, '_goodwavelength'):
                    new_cube._goodwavelength = original_cube._goodwavelength
                
                # Copy fit_spectra and other methods
                new_cube.fit_spectra = original_cube.fit_spectra
                new_cube.fit_emission_lines = original_cube.fit_emission_lines
                new_cube.calculate_spectral_indices = original_cube.calculate_spectral_indices
                
                # Store the bin info for reference
                new_cube._bin_num = self.bin_num
                new_cube._bin_indices = self.bin_indices
                
                return new_cube
            except Exception as e:
                logger.warning(f"Failed to create cube object: {e}. Using dict format instead.")
        
        # Fallback to the original format
        return result
    
    def create_visualization_plots(self, output_dir, galaxy_name):
        """Create basic visualization plots for binned data"""
        plots_dir = Path(output_dir) / 'plots'
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Plot bin map
        try:
            # Get dimensions from bin_num
            if 'ny' in self.metadata and 'nx' in self.metadata:
                ny, nx = self.metadata['ny'], self.metadata['nx']
            else:
                # Estimate from bin_num shape
                ny, nx = self.bin_num.shape if hasattr(self.bin_num, 'shape') else (1, len(self.bin_num))
            
            # Create 2D bin map
            bin_map_2d = self.bin_num.reshape(ny, nx)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(bin_map_2d, cmap='tab20', interpolation='nearest')
            plt.colorbar(im, ax=ax, label='Bin number')
            ax.set_title(f"{galaxy_name} - Binning Map")
            
            # Save
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_bin_map.png")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create bin map visualization: {e}")
        
        # 2. Plot sample spectra
        try:
            # Choose a few bins to plot
            n_bins = self.spectra.shape[1]
            bins_to_plot = [0]  # Always plot first bin
            
            # Add some bins throughout the range
            if n_bins > 1:
                bins_to_plot.append(n_bins // 2)
            if n_bins > 2:
                bins_to_plot.append(n_bins - 1)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for bin_id in bins_to_plot:
                ax.plot(self.wavelength, self.spectra[:, bin_id], 
                       label=f"Bin {bin_id}")
            
            ax.set_xlabel('Wavelength (Å)')
            ax.set_ylabel('Flux')
            ax.set_title(f"{galaxy_name} - Sample Bin Spectra")
            ax.legend()
            
            # Save
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_sample_spectra.png")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create sample spectra visualization: {e}")


@dataclass
class VoronoiBinnedData(BinnedSpectra):
    """Class for Voronoi binned spectra"""
    
    def create_visualization_plots(self, output_dir, galaxy_name):
        """Create Voronoi specific visualization plots"""
        # Call parent method first
        super().create_visualization_plots(output_dir, galaxy_name)
        
        plots_dir = Path(output_dir) / 'plots'
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Add Voronoi specific plots
        try:
            # Plot SNR distribution
            if 'sn' in self.metadata:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(self.metadata['sn'], bins=20, alpha=0.7)
                ax.axvline(x=self.metadata.get('target_snr', 0), 
                          color='r', linestyle='--', 
                          label=f"Target SNR: {self.metadata.get('target_snr', 0)}")
                ax.set_xlabel('Signal-to-Noise Ratio')
                ax.set_ylabel('Number of Bins')
                ax.set_title(f"{galaxy_name} - Voronoi Bin SNR Distribution")
                ax.legend()
                
                # Save
                plt.tight_layout()
                plt.savefig(plots_dir / f"{galaxy_name}_voronoi_snr.png")
                plt.close(fig)
                
            # Plot bin size distribution
            if 'n_pixels' in self.metadata:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(self.metadata['n_pixels'], bins=20, alpha=0.7)
                ax.set_xlabel('Number of Pixels per Bin')
                ax.set_ylabel('Number of Bins')
                ax.set_title(f"{galaxy_name} - Voronoi Bin Size Distribution")
                
                # Save
                plt.tight_layout()
                plt.savefig(plots_dir / f"{galaxy_name}_voronoi_bin_size.png")
                plt.close(fig)
                
        except Exception as e:
            logger.warning(f"Failed to create Voronoi visualization: {e}")


@dataclass
class RadialBinnedData(BinnedSpectra):
    """Class for radial binned spectra"""
    bin_radii: np.ndarray        # Radius of each bin
    
    def create_visualization_plots(self, output_dir, galaxy_name):
        """Create radial specific visualization plots with physical coordinates"""
        # Call parent method first
        super().create_visualization_plots(output_dir, galaxy_name)
        
        plots_dir = Path(output_dir) / 'plots'
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Add radial specific plots
        try:
            # Get pixel size for coordinate conversion
            pixel_size_x = self.metadata.get('pixelsize_x', 1.0)
            pixel_size_y = self.metadata.get('pixelsize_y', 1.0)
            
            # Plot radius vs. bin number
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(np.arange(len(self.bin_radii)), self.bin_radii, 'o-')
            ax.set_xlabel('Bin Number')
            ax.set_ylabel('Radius (arcsec)')
            ax.set_title(f"{galaxy_name} - Radial Bin Distribution")
            ax.grid(True, alpha=0.3)
            
            # Save
            plt.tight_layout()
            plt.savefig(plots_dir / f"{galaxy_name}_radial_bins.png")
            plt.close(fig)
            
            # Plot bin map with circles representing bins using physical coordinates
            try:
                # Get dimensions from bin_num
                if 'ny' in self.metadata and 'nx' in self.metadata:
                    ny, nx = self.metadata['ny'], self.metadata['nx']
                else:
                    # Estimate from bin_num shape
                    bin_shape = getattr(self.bin_num, 'shape', None)
                    if bin_shape is not None:
                        ny, nx = bin_shape
                    else:
                        ny, nx = 1, len(self.bin_num)
                
                # Create 2D bin map
                bin_map_2d = self.bin_num.reshape(ny, nx)
                
                # Get center coordinates
                center_x = self.metadata.get('center_x', nx // 2)
                center_y = self.metadata.get('center_y', ny // 2)
                
                # Get ellipticity and PA
                ellipticity = self.metadata.get('ellipticity', 0)
                pa = self.metadata.get('pa', 0)
                
                # Create physical coordinate grid
                y_coords, x_coords = np.indices((ny, nx))
                
                # Convert to physical units (arcseconds)
                # Center coordinates based on image center
                x_physical = (x_coords - nx/2) * pixel_size_x
                y_physical = (y_coords - ny/2) * pixel_size_y
                
                # Convert center to physical units relative to center
                center_x_phys = (center_x - nx/2) * pixel_size_x
                center_y_phys = (center_y - ny/2) * pixel_size_y
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 10))
                
                # Create colored bin map
                unique_bins = np.unique(bin_map_2d)
                unique_bins = unique_bins[unique_bins >= 0]  # Remove negative values
                cmap = plt.cm.get_cmap('tab20', len(unique_bins))
                
                # Plot each bin with proper physical coordinates
                for i in unique_bins:
                    mask = bin_map_2d == i
                    color = cmap(i % 20)  # Cycle through colors
                    x_bin = x_physical[mask]
                    y_bin = y_physical[mask]
                    
                    if len(x_bin) > 0:
                        # Use scatter for visualization
                        ax.scatter(x_bin, y_bin, color=color, s=10, alpha=0.7, label=f'Bin {i}')
                
                # Add center marker
                ax.plot(center_x_phys, center_y_phys, 'r+', markersize=10, label='Center')
                
                # Add rings representing bin edges in physical coordinates
                if 'bin_edges' in self.metadata:
                    bin_edges = self.metadata['bin_edges']
                    
                    for radius in bin_edges:
                        # Draw circle or ellipse in physical coordinates
                        if ellipticity == 0 or not np.isfinite(ellipticity):
                            # Draw circle
                            circle = plt.Circle(
                                (center_x_phys, center_y_phys), 
                                radius,  # Already in arcsec
                                fill=False, 
                                color='white', 
                                linestyle='--', 
                                alpha=0.7
                            )
                            ax.add_patch(circle)
                        else:
                            # Draw ellipse
                            from matplotlib.patches import Ellipse
                            width = 2 * radius
                            height = 2 * radius * (1 - ellipticity)
                            ellipse = Ellipse(
                                (center_x_phys, center_y_phys), 
                                width, 
                                height,
                                angle=pa, 
                                fill=False, 
                                color='white', 
                                linestyle='--', 
                                alpha=0.7
                            )
                            ax.add_patch(ellipse)
                
                # Add ring labels
                if 'bin_radii' in dir(self):
                    for i, radius in enumerate(self.bin_radii):
                        # Label position along positive x-axis
                        label_x = center_x_phys + radius * np.cos(np.radians(pa))
                        label_y = center_y_phys + radius * np.sin(np.radians(pa))
                        
                        ax.text(
                            label_x, 
                            label_y, 
                            f'{i}', 
                            color='white', 
                            fontsize=8,
                            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2')
                        )
                
                # Set axis labels with physical units
                ax.set_xlabel('X (arcsec)')
                ax.set_ylabel('Y (arcsec)')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                
                # Set title
                ax.set_title(f"{galaxy_name} - Radial Bin Map (PA={pa:.1f}°, e={ellipticity:.2f})")
                
                # Create legend for first few bins
                if len(unique_bins) > 0:
                    max_legend = min(5, len(unique_bins))
                    handles, labels = ax.get_legend_handles_labels()
                    # Filter to show only first few bins and center
                    center_idx = labels.index('Center') if 'Center' in labels else -1
                    bin_indices = [i for i, label in enumerate(labels) if label.startswith('Bin') and int(label.split()[1]) < max_legend]
                    
                    if center_idx >= 0:
                        bin_indices.append(center_idx)
                    
                    filtered_handles = [handles[i] for i in bin_indices]
                    filtered_labels = [labels[i] for i in bin_indices]
                    
                    ax.legend(
                        filtered_handles, 
                        filtered_labels, 
                        loc='upper right', 
                        fontsize='small'
                    )
                
                # Save
                plt.tight_layout()
                plt.savefig(plots_dir / f"{galaxy_name}_radial_bin_map.png", dpi=150)
                plt.close(fig)
                
                # Plot radial bin profile with areas
                try:
                    # Calculate bin areas
                    bin_areas = []
                    for i, radius in enumerate(self.bin_radii):
                        if i == 0:
                            inner_edge = 0
                        else:
                            inner_edge = self.bin_radii[i-1]
                        
                        # Area of annular region, accounting for ellipticity
                        area = np.pi * (radius**2 - inner_edge**2)
                        if ellipticity > 0 and ellipticity < 1:
                            area *= (1 - ellipticity)
                            
                        bin_areas.append(area)
                    
                    # Plot bin radius vs. area
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(self.bin_radii, bin_areas, 'o-')
                    ax.set_xlabel('Radius (arcsec)')
                    ax.set_ylabel('Bin Area (arcsec²)')
                    ax.set_title(f"{galaxy_name} - Radial Bin Areas")
                    ax.grid(True, alpha=0.3)
                    
                    # Save
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"{galaxy_name}_radial_bin_areas.png", dpi=150)
                    plt.close(fig)
                except Exception as e:
                    logger.warning(f"Error creating bin area plot: {e}")
                    plt.close('all')
                
            except Exception as e:
                logger.warning(f"Failed to create radial bin map: {e}")
                plt.close('all')
            
        except Exception as e:
            logger.warning(f"Failed to create radial visualization: {e}")
            plt.close('all')

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


def combine_spectra_efficiently(spectra, wavelength, bin_indices, velocity_field=None, n_x=None):
    """
    Efficiently combine spectra into bins with improved velocity correction.
    
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
        
    Returns
    -------
    numpy.ndarray
        Combined bin spectra array [n_wave, n_bins]
    """
    n_wave = len(wavelength)
    n_bins = len(bin_indices)
    
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
        
        # Skip bins with invalid indices
        if np.max(indices) >= spectra.shape[1]:
            logger.warning(f"Bin {i} has indices beyond spectra shape, skipping")
            bin_spectra[:, i] = np.nan
            continue
            
        try:
            # Extract spectra for this bin
            bin_spectra_list = []
            
            # Try alternate velocity correction methods if one fails
            velocity_correction_success = False
            
            # First attempt: Calculate median velocity for the bin
            if do_correction:
                try:
                    # Get velocity field for this bin
                    bin_velocities = []
                    for idx in indices:
                        row = idx // n_x
                        col = idx % n_x
                        if row < velocity_field.shape[0] and col < velocity_field.shape[1]:
                            vel = velocity_field[row, col]
                            if np.isfinite(vel):
                                bin_velocities.append(vel)
                    
                    # Use median velocity for the bin if we have valid velocities
                    if bin_velocities:
                        median_velocity = np.median(bin_velocities)
                        
                        # Only apply correction for non-zero velocities
                        if abs(median_velocity) > 1.0:  # Minimum 1 km/s to apply correction
                            # Apply velocity shift using Doppler formula
                            lam_shifted = wavelength * (1 + median_velocity/C_KMS)
                            
                            # Combine all spectra first, then apply correction once
                            combined_spectrum = np.zeros(n_wave)
                            weight_sum = 0
                            
                            for idx in indices:
                                spec = spectra[:, idx]
                                if not np.all(~np.isfinite(spec)):
                                    combined_spectrum += spec
                                    weight_sum += 1
                            
                            if weight_sum > 0:
                                combined_spectrum /= weight_sum
                                
                                # Use spectres for resampling the combined spectrum
                                try:
                                    corrected_spectrum = spectres(wavelength, lam_shifted, combined_spectrum, fill=0.0)
                                    bin_spectra[:, i] = corrected_spectrum
                                    velocity_correction_success = True
                                    continue  # Skip to next bin
                                except Exception as e:
                                    logger.debug(f"Velocity correction failed: {e}")
                except Exception as e:
                    logger.debug(f"Bin velocity correction failed: {e}")
            
            # Second attempt: Process each spectrum individually
            if not velocity_correction_success:
                # Process each spectrum individually
                for idx in indices:
                    # Get spectrum
                    spectrum = spectra[:, idx]
                    
                    # Skip if all NaN
                    if np.all(~np.isfinite(spectrum)):
                        continue
                    
                    # Apply velocity correction if available
                    if do_correction:
                        try:
                            row = idx // n_x
                            col = idx % n_x
                            
                            # Check bounds
                            if row < velocity_field.shape[0] and col < velocity_field.shape[1]:
                                vel = velocity_field[row, col]
                                
                                # Only correct if velocity is valid and non-negligible
                                if np.isfinite(vel) and abs(vel) > 1.0:
                                    # Apply velocity shift using Doppler formula
                                    lam_shifted = wavelength * (1 + vel/C_KMS)
                                    
                                    try:
                                        # Use spectres for resampling
                                        corrected_spectrum = spectres(wavelength, lam_shifted, spectrum, fill=0.0)
                                        bin_spectra_list.append(corrected_spectrum)
                                        continue  # Skip the default append
                                    except Exception as e:
                                        # Fall through to add original spectrum
                                        pass
                        except Exception as e:
                            # Fall through to add original spectrum
                            pass
                    
                    # Add the original spectrum if no velocity correction
                    bin_spectra_list.append(spectrum)
                
                # Combine spectra if any valid
                if bin_spectra_list:
                    # Convert to array for easier operations
                    spectra_array = np.array(bin_spectra_list)
                    
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

def create_grid_binning(x, y, signal, noise, nx=3, ny=3, xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Create a grid-based binning as a reliable fallback
    
    Parameters
    ----------
    x, y : ndarray
        Coordinate arrays
    signal, noise : ndarray
        Signal and noise arrays
    nx, ny : int
        Number of bins in x and y directions
    xmin, xmax, ymin, ymax : float, optional
        Coordinate bounds
        
    Returns
    -------
    tuple or None
        (bin_num, x_gen, y_gen, sn, n_pixels, scale) or None if failed
    """
    try:
        # Ensure at least one bin
        nx = max(1, nx)
        ny = max(1, ny)
        
        # Get bounds if not provided
        if xmin is None:
            xmin = np.min(x)
        if xmax is None:
            xmax = np.max(x)
        if ymin is None:
            ymin = np.min(y)
        if ymax is None:
            ymax = np.max(y)
            
        # Add small margin to avoid edge issues
        eps = 1e-6
        xmin -= eps
        xmax += eps
        ymin -= eps
        ymax += eps
        
        # Create bin edges
        x_edges = np.linspace(xmin, xmax, nx + 1)
        y_edges = np.linspace(ymin, ymax, ny + 1)
        
        # Initialize arrays
        bin_num = np.full(len(x), -1, dtype=int)
        x_gen = []
        y_gen = []
        sn_values = []
        n_pixels = []
        
        # Assign bins
        bin_id = 0
        for i in range(nx):
            for j in range(ny):
                # Get bin boundaries
                x_min, x_max = x_edges[i], x_edges[i+1]
                y_min, y_max = y_edges[j], y_edges[j+1]
                
                # Select points in this bin
                mask = ((x >= x_min) & (x < x_max) & 
                        (y >= y_min) & (y < y_max))
                
                # Only create bin if it has data points
                if np.any(mask):
                    bin_num[mask] = bin_id
                    x_gen.append(np.mean(x[mask]))
                    y_gen.append(np.mean(y[mask]))
                    
                    # Calculate SNR
                    s = np.mean(signal[mask])
                    n = np.mean(noise[mask])
                    sn = s/n if n > 0 else 1.0
                    
                    sn_values.append(sn)
                    n_pixels.append(np.sum(mask))
                    bin_id += 1
        
        # Check if we created any bins
        if bin_id == 0:
            return None
            
        return (bin_num, np.array(x_gen), np.array(y_gen), 
                np.array(sn_values), np.array(n_pixels), 1.0)
    
    except Exception as e:
        logger.error(f"Error in grid binning: {e}")
        return None


def calculate_snr(flux, template=None, observed=None, continuum_mask=None):
    """
    Calculate signal-to-noise ratio using template fitting residuals
    
    Parameters
    ----------
    flux : ndarray
        Flux values
    template : ndarray, optional
        Template (model) spectrum
    observed : ndarray, optional
        Observed spectrum
    continuum_mask : ndarray, optional
        Boolean mask for continuum regions without emission lines
        
    Returns
    -------
    float
        Signal-to-noise ratio
    """
    # If template and observed are available, use residuals
    if template is not None and observed is not None:
        # Apply continuum mask if provided
        if continuum_mask is not None:
            template_cont = template[continuum_mask]
            observed_cont = observed[continuum_mask]
        else:
            template_cont = template
            observed_cont = observed
            
        # Remove NaN values
        valid = np.isfinite(template_cont) & np.isfinite(observed_cont)
        if np.sum(valid) < 5:  # Need at least a few points
            return 1.0  # Default minimum SNR
            
        # Calculate SNR using template and residuals
        # SNR = mean(template) / std(template - observed)
        signal = np.mean(template_cont[valid])
        noise = np.std(template_cont[valid] - observed_cont[valid])
        
        if noise > 0:
            return signal / noise
        return 1.0
    
    # If only flux is available, use simple estimation
    valid = np.isfinite(flux)
    if not np.any(valid):
        return 1.0
    
    # Use simple SNR estimation
    signal = np.mean(flux[valid])
    noise = np.std(flux[valid])
    
    if noise > 0:
        return signal / noise
    return 1.0


def run_voronoi_binning(x, y, signal, noise, target_snr, plot=0, quiet=False, cvt=True, min_snr=0.0):
    """
    Run Voronoi binning on input data with better handling of target SNR values
    
    Parameters
    ----------
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    signal : ndarray
        Signal data
    noise : ndarray
        Noise data
    target_snr : float
        Target signal-to-noise ratio. Recommended to be between:
        - Minimum: 1.5 × maximum pixel SNR
        - Maximum: 50 × maximum pixel SNR
    plot : int, default=0
        Plotting flag (0=no plot, 1=plot)
    quiet : bool, default=False
        Quiet mode
    cvt : bool, default=True
        Use CVT optimization
    min_snr : float, default=0.0
        Minimum SNR for valid bins
        
    Returns
    -------
    tuple
        Binning results (bin_num, x_gen, y_gen, sn, n_pixels, scale)
    """
    try:
        # Try to import vorbin
        from vorbin.voronoi_2d_binning import voronoi_2d_binning
        import warnings
        
        # Silence numpy warnings during binning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            # Ensure data is valid
            valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(signal) & np.isfinite(noise) & (noise > 0)
            if not quiet:
                logger.info(f"Using {np.sum(valid_mask)} valid pixels out of {len(x)} for Voronoi binning")
                
            if np.sum(valid_mask) < 3:
                raise ValueError(f"Too few valid pixels ({np.sum(valid_mask)}) for Voronoi binning")
                
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            signal_valid = signal[valid_mask]
            noise_valid = noise[valid_mask]
            
            # Calculate current median and max SNR
            pixel_snr = signal_valid / noise_valid
            median_snr = np.median(pixel_snr)
            max_pixel_snr = np.max(pixel_snr)
            
            if not quiet:
                logger.info(f"Median SNR in data: {median_snr:.1f}, Maximum SNR: {max_pixel_snr:.1f}")
            
            # Check if the target SNR is reasonable based on data quality
            # Recommended range: 1.5× to 50× maximum pixel SNR
            min_recommended = max_pixel_snr * 1.5
            max_recommended = max_pixel_snr * 50.0
            
            if target_snr < min_recommended:
                if not quiet:
                    logger.warning(f"Target SNR ({target_snr:.1f}) is lower than recommended minimum ({min_recommended:.1f})")
                    logger.info(f"Recommended range is {min_recommended:.1f} to {max_recommended:.1f}")
            elif target_snr > max_recommended:
                if not quiet:
                    logger.warning(f"Target SNR ({target_snr:.1f}) is higher than recommended maximum ({max_recommended:.1f})")
                    logger.info(f"Recommended range is {min_recommended:.1f} to {max_recommended:.1f}")
            
            # First try with the user's specified target_snr
            success = False
            error_message = ""
            
            # Try the main target first
            try:
                if not quiet:
                    logger.info(f"Trying Voronoi binning with target SNR = {target_snr:.1f}")
                
                for use_cvt in [cvt, False]:
                    try:
                        # Set exceptionally high max_iterations for better convergence
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            
                            bin_num, x_gen, y_gen, sn, n_pixels, scale = voronoi_2d_binning(
                                x_valid, y_valid, signal_valid, noise_valid, target_snr, 
                                cvt=use_cvt, plot=0, quiet=True, wvt=True)
                        
                        # Check if we have any valid bins
                        if len(x_gen) > 0:
                            success = True
                            if not quiet and use_cvt != cvt:
                                logger.warning(f"Using CVT={use_cvt} for better results")
                            break
                    except Exception as e:
                        error_message = str(e)
                        if "index 0 is out of bounds for axis 0 with size 0" in error_message:
                            # Specific error that requires different approach
                            continue
                        elif "zero-size array to reduction operation" in error_message:
                            # Another empty array error
                            continue
                        elif "too many values to unpack" in error_message:
                            # Error with voronoi_2d_binning unpacking
                            continue
            except Exception as e:
                error_message = str(e)
                
            # If the specified target failed, try an alternative value within recommended range
            if not success:
                if not quiet:
                    logger.warning(f"Failed with target SNR = {target_snr:.1f}: {error_message}")
                
                # Choose a moderate value within the recommended range
                alternative_target = min(max(max_pixel_snr * 3.0, min_recommended), max_recommended * 0.5)
                
                if not quiet:
                    logger.info(f"Trying alternative target SNR = {alternative_target:.1f}")
                
                try:
                    for use_cvt in [cvt, False]:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                
                                bin_num, x_gen, y_gen, sn, n_pixels, scale = voronoi_2d_binning(
                                    x_valid, y_valid, signal_valid, noise_valid, alternative_target,
                                    cvt=use_cvt, plot=0, quiet=True, wvt=True)
                                
                                if len(x_gen) > 0:
                                    success = True
                                    if not quiet:
                                        logger.warning(f"Using alternative SNR = {alternative_target:.1f}, CVT={use_cvt}")
                                    break
                        except Exception as e:
                            continue
                except Exception as e:
                    pass
            
            # If still no success, use create_grid_binning as a more reliable fallback
            if not success:
                if not quiet:
                    logger.warning(f"All Voronoi binning attempts failed with error: {error_message}")
                    logger.warning("Using fallback grid binning scheme")
                
                # Get additional information for grid binning
                xmin, xmax = np.min(x_valid), np.max(x_valid)
                ymin, ymax = np.min(y_valid), np.max(y_valid)
                
                # Create a grid-based binning - more reliable than quadrants for arbitrary shapes
                bin_result = create_grid_binning(
                    x_valid, y_valid, signal_valid, noise_valid, 
                    nx=min(4, int(np.sqrt(len(x_valid) / 5))),  # Auto-adjust grid size
                    ny=min(4, int(np.sqrt(len(x_valid) / 5))),
                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
                )
                
                if bin_result is not None:
                    bin_num, x_gen, y_gen, sn, n_pixels, scale = bin_result
                    success = True
                    if not quiet:
                        logger.info(f"Created {len(x_gen)} grid bins as fallback")
                else:
                    # Fall back to _create_fallback_binning
                    return _create_fallback_binning(x, y)
            
            # Map bin numbers back to the original arrays
            full_bin_num = np.full_like(x, -1, dtype=int)
            full_bin_num[valid_mask] = bin_num
            
            # Filter out low SNR bins if needed, but don't filter ALL bins
            if min_snr > 0:
                # Find bins with SNR below minimum
                bad_bins = np.where(sn < min_snr)[0]
                
                # Don't filter all bins
                if len(bad_bins) < len(sn):  # Only if some bins remain
                    # Mark these bins as unbinned (-1)
                    for bad_bin in bad_bins:
                        full_bin_num[full_bin_num == bad_bin] = -1
                    
                    if not quiet and len(bad_bins) > 0:
                        logger.info(f"Filtered out {len(bad_bins)} bins with SNR < {min_snr}")
                else:
                    if not quiet:
                        logger.warning(f"All bins have SNR < {min_snr}, but keeping them to avoid empty result")
            
            return full_bin_num, x_gen, y_gen, sn, n_pixels, scale
        
    except ImportError:
        logger.error("vorbin package not found. Please install with 'pip install vorbin'")
        raise
    except ValueError as ve:
        # For specific value errors, fall back to simple binning
        logger.error(f"Voronoi binning value error: {ve}")
        logger.info("Using simple radial binning as fallback")
        return _create_fallback_binning(x, y)
    except Exception as e:
        logger.error(f"Error in Voronoi binning: {e}")
        logger.info("Using simple radial binning as fallback")
        return _create_fallback_binning(x, y)


def _create_fallback_binning(x, y, n_bins=5):
    """
    Create a simple fallback binning when Voronoi binning fails
    
    Parameters
    ----------
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    n_bins : int, default=5
        Number of bins to create
        
    Returns
    -------
    tuple
        Simple binning results compatible with Voronoi binning return
    """
    # Calculate distance from center for each point
    x_center = np.median(x[np.isfinite(x)])
    y_center = np.median(y[np.isfinite(y)])
    
    r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    
    # Create simple radial bins
    valid_mask = np.isfinite(r)
    if np.sum(valid_mask) < 1:
        # If no valid points, return a simple error state
        bin_num = np.full_like(x, -1, dtype=int)
        return bin_num, np.array([0]), np.array([0]), np.array([1.0]), np.array([1]), 1.0
    
    r_valid = r[valid_mask]
    max_r = np.max(r_valid)
    
    # Create bin edges
    bin_edges = np.linspace(0, max_r, n_bins + 1)
    
    # Assign bins
    bin_num = np.full_like(x, -1, dtype=int)
    x_gen = []
    y_gen = []
    sn = []
    n_pixels = []
    
    for i in range(n_bins):
        bin_mask = valid_mask & (r >= bin_edges[i]) & (r < bin_edges[i+1])
        if np.any(bin_mask):
            bin_num[bin_mask] = i
            x_gen.append(np.mean(x[bin_mask]))
            y_gen.append(np.mean(y[bin_mask]))
            n_pixels.append(np.sum(bin_mask))
            sn.append(5.0)  # Use reasonable SNR value instead of placeholder
    
    # If no bins were created (unlikely but possible), create at least one bin
    if not x_gen:
        bin_num[valid_mask] = 0
        x_gen = [np.mean(x[valid_mask])]
        y_gen = [np.mean(y[valid_mask])]
        n_pixels = [np.sum(valid_mask)]
        sn = [5.0]
    
    logger.warning(f"Created simple fallback binning with {len(x_gen)} bins")
    
    return bin_num, np.array(x_gen), np.array(y_gen), np.array(sn), np.array(n_pixels), 1.0


def calculate_radial_bins(x, y, center_x=0, center_y=0, pa=0, ellipticity=0, n_rings=10, log_spacing=False):
    """
    Calculate radial bins with improved error handling
    
    Parameters
    ----------
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    center_x : float, default=0
        x coordinate of center
    center_y : float, default=0
        y coordinate of center
    pa : float, default=0
        Position angle (degrees)
    ellipticity : float, default=0
        Ellipticity (0-1)
    n_rings : int, default=10
        Number of radial rings
    log_spacing : bool, default=False
        Use logarithmic spacing
        
    Returns
    -------
    tuple
        (bin_num, bin_edges, bin_radii)
    """
    try:
        # Convert position angle to radians
        pa_rad = np.radians(pa)
        
        # Calculate semi-major axis
        dx = x - center_x
        dy = y - center_y
        
        # Rotate coordinates
        x_rot = dx * np.cos(pa_rad) + dy * np.sin(pa_rad)
        y_rot = -dx * np.sin(pa_rad) + dy * np.cos(pa_rad)
        
        # Apply ellipticity
        y_rot_scaled = y_rot / (1 - ellipticity) if ellipticity < 1 else y_rot
        
        # Calculate radius
        radius = np.sqrt(x_rot**2 + y_rot_scaled**2)
        
        # Check for valid data points
        valid_mask = np.isfinite(radius)
        if np.sum(valid_mask) < 1:
            logger.warning("No valid data points for binning")
            # Return a simple single bin with all pixels
            bin_num = np.zeros_like(radius, dtype=int)
            bin_edges = np.array([np.nanmax(radius)])
            bin_radii = np.array([0.0])
            return bin_num, bin_edges, bin_radii
        
        # Calculate bin edges
        max_radius = np.nanmax(radius[valid_mask])
        min_radius = np.nanmin(radius[valid_mask])
        
        # Ensure we have a reasonable range
        if max_radius <= min_radius or not np.isfinite(max_radius) or not np.isfinite(min_radius):
            logger.warning("Invalid radius range, using simple binning")
            bin_num = np.zeros_like(radius, dtype=int)
            bin_edges = np.array([max_radius])
            bin_radii = np.array([0.0])
            return bin_num, bin_edges, bin_radii
        
        # Adjust n_rings if we have very few points
        n_points = np.sum(valid_mask)
        if n_points < 2 * n_rings:
            new_n_rings = max(1, n_points // 2)
            logger.warning(f"Too few points ({n_points}) for {n_rings} rings. Adjusting to {new_n_rings} rings.")
            n_rings = new_n_rings
        
        if log_spacing:
            # Logarithmic spacing (avoid log(0) issues)
            min_log = np.log10(max(min_radius, max_radius/1000))
            max_log = np.log10(max_radius)
            bin_edges = np.logspace(min_log, max_log, n_rings+1)[1:]
        else:
            # Linear spacing
            bin_edges = np.linspace(min_radius, max_radius, n_rings + 1)[1:]
        
        # Assign bin numbers
        bin_num = np.zeros_like(radius, dtype=int)
        
        for i in range(n_rings):
            if i == 0:
                # First bin: all points with radius <= first edge
                mask = radius <= bin_edges[i]
            else:
                # Other bins: points with radius between previous and current edge
                mask = (radius > bin_edges[i-1]) & (radius <= bin_edges[i])
            
            bin_num[mask] = i
        
        # Check for empty bins and handle them
        # Count pixels in each bin
        bin_counts = np.bincount(bin_num[valid_mask], minlength=n_rings)
        empty_bins = np.where(bin_counts == 0)[0]
        
        if len(empty_bins) > 0:
            logger.warning(f"Found {len(empty_bins)} empty bins. Adjusting binning.")
            
            # Determine which bins have data
            non_empty_bins = np.where(bin_counts > 0)[0]
            
            if len(non_empty_bins) == 0:
                # All bins are empty (shouldn't happen, but just in case)
                logger.error("All bins are empty. Using single bin.")
                bin_num = np.zeros_like(radius, dtype=int)
                bin_edges = np.array([max_radius])
                bin_radii = np.array([np.nanmean(radius[valid_mask])])
                return bin_num, bin_edges, bin_radii
            
            # Create a mapping from old bin numbers to new contiguous ones
            new_bin_map = np.full(n_rings, -1, dtype=int)
            for i, bin_id in enumerate(non_empty_bins):
                new_bin_map[bin_id] = i
            
            # Remap the bin numbers
            new_bin_num = np.full_like(bin_num, -1)
            for old_bin, new_bin in enumerate(new_bin_map):
                if new_bin >= 0:  # Only map non-empty bins
                    new_bin_num[bin_num == old_bin] = new_bin
            
            # Create new bin_edges by only using edges that define non-empty bins
            if log_spacing:
                # Recreate with fewer bins but same range
                new_bin_edges = np.logspace(min_log, max_log, len(non_empty_bins) + 1)[1:]
            else:
                new_bin_edges = np.linspace(min_radius, max_radius, len(non_empty_bins) + 1)[1:]
            
            # Update values
            bin_num = new_bin_num
            bin_edges = new_bin_edges
            n_rings = len(non_empty_bins)
        
        # Calculate bin radii (at middle of each bin)
        bin_radii = np.zeros(n_rings)
        bin_edges_full = np.concatenate(([0], bin_edges))
        
        for i in range(n_rings):
            mask = bin_num == i
            if np.any(mask & valid_mask):
                # Use mean radius of points in bin
                bin_radii[i] = np.nanmean(radius[mask & valid_mask])
            else:
                # Use middle of bin edges if no points (shouldn't happen after our fixes)
                bin_radii[i] = 0.5 * (bin_edges_full[i] + bin_edges_full[i+1])
        
        return bin_num, bin_edges, bin_radii
        
    except Exception as e:
        logger.error(f"Error in radial binning: {e}")
        # Return a simple single bin as fallback
        bin_num = np.zeros_like(x, dtype=int)
        bin_edges = np.array([np.nanmax(np.sqrt(x**2 + y**2))])
        bin_radii = np.array([0.0])
        return bin_num, bin_edges, bin_radii


def plot_binned_map(x, y, bin_num, values=None, title=None, cmap='tab20', 
                  vmin=None, vmax=None, equal_aspect=True, savefile=None):
    """
    Plot binned map
    
    Parameters
    ----------
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    bin_num : ndarray
        Bin numbers
    values : ndarray, optional
        Values to plot
    title : str, optional
        Plot title
    cmap : str, default='tab20'
        Colormap
    vmin : float, optional
        Minimum value for colormap
    vmax : float, optional
        Maximum value for colormap
    equal_aspect : bool, default=True
        Equal aspect ratio
    savefile : str, optional
        File to save plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot with bins
    if values is None:
        # Show bin numbers
        valid_mask = bin_num >= 0
        scatter = ax.scatter(x[valid_mask], y[valid_mask], c=bin_num[valid_mask], 
                           cmap=cmap, s=10, vmin=vmin, vmax=vmax)
        plt.colorbar(scatter, ax=ax, label='Bin')
    else:
        # Show values
        valid_mask = (bin_num >= 0) & np.isfinite(values)

        # Fix array size mismatch (x, y, bin_num may be 2D arrays while values is 1D)
        if np.sum(valid_mask) > 0:
            # Ensure proper mapping even if array shapes differ
            values_fixed = np.zeros_like(bin_num, dtype=float)
            values_fixed.fill(np.nan)
            
            # Map values to bin numbers
            unique_bins = np.unique(bin_num[bin_num >= 0])
            for i, bin_id in enumerate(unique_bins):
                if i < len(values):
                    bin_mask = bin_num == bin_id
                    value = values[i]
                    # Handle scalar and array values
                    if np.isscalar(value) or (hasattr(value, 'size') and value.size == 1):
                        values_fixed[bin_mask] = value
                    elif hasattr(value, '__len__') and len(value) > 0:
                        values_fixed[bin_mask] = value[0]
            
            scatter = ax.scatter(x[valid_mask], y[valid_mask], c=values_fixed[valid_mask], 
                               cmap=cmap, s=10, vmin=vmin, vmax=vmax)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(x, y, c='gray', s=5, alpha=0.5)
            ax.text(0.5, 0.5, "No valid data to plot", 
                   transform=ax.transAxes, ha='center', va='center')
    
    # Set title
    if title:
        ax.set_title(title)
    
    # Set axes parameters
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    if equal_aspect:
        ax.set_aspect('equal')
    
    # Save figure if requested
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
    
    return fig


def plot_radial_profile(radius, values, errors=None, title=None, xlabel='Radius', 
                       ylabel=None, savefile=None):
    """
    Plot radial profile
    
    Parameters
    ----------
    radius : ndarray
        Radial distance
    values : ndarray
        Values to plot
    errors : ndarray, optional
        Error values
    title : str, optional
        Plot title
    xlabel : str, default='Radius'
        X-axis label
    ylabel : str, optional
        Y-axis label
    savefile : str, optional
        File to save plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to arrays and ensure compatibility
    radius = np.asarray(radius)
    values = np.asarray(values)
    
    # Check if data is valid
    valid_mask = np.isfinite(radius) & np.isfinite(values)
    if not np.any(valid_mask):
        ax.text(0.5, 0.5, "No valid data to plot", 
               transform=ax.transAxes, ha='center', va='center')
    else:
        # Plot data
        if errors is not None:
            errors = np.asarray(errors)
            valid_mask = valid_mask & np.isfinite(errors)
            ax.errorbar(radius[valid_mask], values[valid_mask], yerr=errors[valid_mask], 
                       fmt='o-', capsize=3)
        else:
            ax.plot(radius[valid_mask], values[valid_mask], 'o-')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set labels
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    # Set title
    if title:
        ax.set_title(title)
    
    # Save figure if requested
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
    
    return fig