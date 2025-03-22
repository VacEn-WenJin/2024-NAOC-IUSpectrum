"""
Adapter module to use P2P processing with binned data
"""
import numpy as np
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional, Callable, Union
import traceback

logger = logging.getLogger(__name__)

def convert_flat_index_to_coordinates(idx, n_x, n_y):
    """
    Convert a flattened (linear) index to 2D coordinates
    
    Parameters
    ----------
    idx : int
        Flattened index
    n_x : int
        Number of columns
    n_y : int
        Number of rows
        
    Returns
    -------
    tuple
        (row, col) coordinates
    """
    if idx < 0 or idx >= n_x * n_y:
        return None  # Invalid index
    
    row = idx // n_x
    col = idx % n_x
    
    if row >= n_y or col >= n_x:
        return None  # Out of bounds
    
    return row, col

def create_bin_map_from_indices(bin_indices, n_x, n_y):
    """
    Create a 2D bin map from bin indices
    
    Parameters
    ----------
    bin_indices : list
        List of arrays with pixel indices for each bin
    n_x : int
        Number of columns
    n_y : int
        Number of rows
        
    Returns
    -------
    numpy.ndarray
        2D bin map
    """
    # Create empty bin map
    bin_map = np.full((n_y, n_x), -1)
    
    # Fill with bin numbers
    for bin_id, indices in enumerate(bin_indices):
        for idx in indices:
            coords = convert_flat_index_to_coordinates(idx, n_x, n_y)
            if coords:
                row, col = coords
                bin_map[row, col] = bin_id
    
    return bin_map

class BinnedDataAdapter:
    """
    Adapter class to make binned data compatible with P2P pipeline.
    
    This class takes binned data and creates a compatible interface
    for P2P processing functions to work with.
    """
    
    def __init__(self, binned_data):
        """
        Initialize adapter with binned data.
        
        Parameters
        ----------
        binned_data : BinnedSpectra
            Binned spectra and related data
        """
        try:
            # Get P2P compatible format
            p2p_data = binned_data.to_p2p_compatible()
            
            # Set attributes that P2P expects
            self.cube = p2p_data['cube']
            self.variance = p2p_data['variance']
            self.wave = p2p_data['wavelength']
            self._wave = p2p_data['wavelength']  # Some code might use this instead
            self._lambda_gal = p2p_data['wavelength']  # Required by some functions
            self._spectra = p2p_data['cube'].reshape(len(p2p_data['wavelength']), -1)
            self._log_variance = p2p_data['variance'].reshape(len(p2p_data['wavelength']), -1)
            
            # Store original binning information
            self.bin_num = p2p_data['bin_num']
            self.bin_indices = p2p_data['bin_indices']
            self.metadata = p2p_data['metadata']
            
            # Calculate logarithmic wavelength grid as some P2P code might expect it
            # Properly calculate velocity scale per Cappellari 2017 equation 8
            self._ln_lam_gal = np.log(self.wave)
            
            # Calculate velocity scale properly:
            # velscale = c * delta[ln(lambda)]
            c = 299792.458  # Speed of light in km/s
            dln_lambda = np.diff(self._ln_lam_gal)
            if len(dln_lambda) > 0:
                # Use median to be robust against potential irregularities
                self._vel_scale = c * np.median(dln_lambda)
            else:
                # Fallback if wavelength array is too short
                self._vel_scale = 50.0  # Default value
                logger.warning("Could not calculate velocity scale from wavelength grid. Using default value.")
            
            # Set both versions of attribute for compatibility
            self.velscale = self._vel_scale
            
            # Create fake spatial dimensions for P2P
            self.ny, self.nx = 1, self.cube.shape[2]
            self._n_y, self._n_x = 1, self.cube.shape[2]
            self.x = p2p_data['x']
            self.y = p2p_data['y']
            
            # Always set pixel size attributes (both standard and underscore versions)
            # Extract from metadata if available
            self.pxl_size_x = self.metadata.get('pixelsize_x', 1.0)
            self.pxl_size_y = self.metadata.get('pixelsize_y', 1.0)
            self._pxl_size_x = self.pxl_size_x  # Also set the underscore version
            self._pxl_size_y = self.pxl_size_y  # Also set the underscore version
            
            # Add required fields for emission line analysis
            n_wave = len(self._lambda_gal)
            self._gas_bestfit_field = np.full((n_wave, self._n_y, self._n_x), np.nan)
            self._bestfit_field = np.full((n_wave, self._n_y, self._n_x), np.nan)
            self._optimal_tmpls = np.full((n_wave, self._n_y, self._n_x), np.nan)

            # Copy additional properties from metadata if available
            if 'redshift' in p2p_data:
                self._redshift = p2p_data['redshift']
            elif 'redshift' in self.metadata:
                self._redshift = self.metadata['redshift']
            else:
                self._redshift = 0.0  # Default value
            
            # Add a flag to indicate this is an adapter
            self._is_adapter = True
            
        except Exception as e:
            logger.error(f"Error initializing BinnedDataAdapter: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    # Add missing methods to properly mimic MUSECube
    def fit_spectra(self, template_filename=None, ppxf_vel_init=0, ppxf_vel_disp_init=40, ppxf_deg=3, n_jobs=-1):
        """
        Adapter implementation of fit_spectra - simplified for binned data
        
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
            Number of parallel jobs to run
            
        Returns
        -------
        tuple
            (velocity_field, dispersion_field, bestfit_field, optimal_templates, poly_coefficients)
        """
        logger.info("Running fit_spectra on binned data adapter")
        from ppxf import ppxf_util
        from ppxf.ppxf import ppxf
        from ppxf.sps_util import sps_lib
        import warnings
        
        try:
            # Load template
            sps = sps_lib(
                filename=template_filename,
                velscale=self._vel_scale,
                fwhm_gal=None,
                norm_range=[np.min(self._lambda_gal), np.max(self._lambda_gal)]
            )
            self._sps = sps  # Store SPS object for later reference
            sps.templates = sps.templates.reshape(sps.templates.shape[0], -1)
            
            # Normalize stellar template
            sps.templates /= np.median(sps.templates)
            tmpl_mask = ppxf_util.determine_mask(
                ln_lam=self._ln_lam_gal,
                lam_range_temp=np.exp(sps.ln_lam_temp[[0, -1]]),
                width=1000
            )
            
            # Initialize storage for templates and weights
            n_templates = sps.templates.shape[1]
            n_wave_fit = len(self._lambda_gal)
            n_wave_temp = sps.templates.shape[0]
            n_bins = self._n_x  # Number of bins
            
            # Important: Initialize fields with correct dimensions
            self._velocity_field = np.full((self._n_y, self._n_x), np.nan)
            self._dispersion_field = np.full((self._n_y, self._n_x), np.nan)
            self._bestfit_field = np.full((n_wave_fit, self._n_y, self._n_x), np.nan)
            self._optimal_tmpls = np.full((n_wave_temp, self._n_y, self._n_x), np.nan)
            self._template_weights = np.full((n_templates, self._n_y, self._n_x), np.nan)
            self._poly_coeffs = []
            
            # Process each bin
            for bin_idx in range(n_bins):
                galaxy_data = self._spectra[:, bin_idx]
                galaxy_noise = np.sqrt(self._log_variance[:, bin_idx])
                
                # Skip low SNR or invalid bins
                if np.count_nonzero(galaxy_data) < 50 or np.count_nonzero(np.isfinite(galaxy_data)) < 50:
                    continue
                
                # Replace NaN values to avoid problems in ppxf
                if np.any(~np.isfinite(galaxy_data)):
                    galaxy_data = np.nan_to_num(galaxy_data, nan=0.0, posinf=0.0, neginf=0.0)
                if np.any(~np.isfinite(galaxy_noise)):
                    galaxy_noise = np.nan_to_num(galaxy_noise, nan=1.0, posinf=1.0, neginf=1.0)
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    try:
                        pp = ppxf(
                            sps.templates, galaxy_data, galaxy_noise,
                            self._vel_scale, mask=tmpl_mask,
                            start=[ppxf_vel_init, ppxf_vel_disp_init], degree=ppxf_deg,
                            lam=self._lambda_gal, lam_temp=sps.lam_temp,
                            quiet=True
                        )
                        
                        # Store results
                        self._velocity_field[0, bin_idx] = pp.sol[0]
                        self._dispersion_field[0, bin_idx] = max(pp.sol[1], 10.0)  # Ensure minimum dispersion
                        self._bestfit_field[:, 0, bin_idx] = pp.bestfit
                        
                        # Calculate optimal template and weights
                        optimal_template = sps.templates @ pp.weights
                        self._optimal_tmpls[:, 0, bin_idx] = optimal_template
                        self._template_weights[:len(pp.weights), 0, bin_idx] = pp.weights
                        
                        # Store polynomial coefficients
                        poly_coeff = np.polyfit(self._lambda_gal, pp.apoly, ppxf_deg)
                        self._poly_coeffs.append((0, bin_idx, poly_coeff))
                        
                    except Exception as e:
                        logger.warning(f"Error fitting bin {bin_idx}: {e}")
                        # Try with simpler parameters
                        try:
                            pp = ppxf(
                                sps.templates, galaxy_data, galaxy_noise,
                                self._vel_scale, mask=tmpl_mask,
                                start=[ppxf_vel_init, ppxf_vel_disp_init], degree=0,  # Simplify
                                lam=self._lambda_gal, lam_temp=sps.lam_temp,
                                quiet=True
                            )
                            
                            # Store results
                            self._velocity_field[0, bin_idx] = pp.sol[0]
                            self._dispersion_field[0, bin_idx] = max(pp.sol[1], 10.0)
                            self._bestfit_field[:, 0, bin_idx] = pp.bestfit
                            
                            # Calculate optimal template and weights
                            optimal_template = sps.templates @ pp.weights
                            self._optimal_tmpls[:, 0, bin_idx] = optimal_template
                            self._template_weights[:len(pp.weights), 0, bin_idx] = pp.weights
                            
                            # Store polynomial coefficients
                            poly_coeff = np.array([pp.apoly[0]])
                            self._poly_coeffs.append((0, bin_idx, poly_coeff))
                            
                        except Exception as e2:
                            logger.debug(f"Both fitting attempts failed for bin {bin_idx}: {e2}")
            
            return (self._velocity_field, self._dispersion_field,
                    self._bestfit_field, self._optimal_tmpls, self._poly_coeffs)
                    
        except Exception as e:
            logger.error(f"Error in adapter fit_spectra: {str(e)}")
            logger.error(traceback.format_exc())
            return (np.full((self._n_y, self._n_x), np.nan),
                    np.full((self._n_y, self._n_x), np.nan),
                    np.full((n_wave_fit, self._n_y, self._n_x), np.nan),
                    np.full((n_wave_temp, self._n_y, self._n_x), np.nan),
                    [])
    
    def fit_emission_lines(self, template_filename=None, ppxf_vel_init=None, 
                          ppxf_sig_init=50.0, ppxf_deg=2, n_jobs=-1):
        """
        Adapter implementation of fit_emission_lines
        
        Parameters
        ----------
        template_filename : str
            Template filename
        ppxf_vel_init : array or None
            Initial velocity values
        ppxf_sig_init : float
            Initial dispersion value
        ppxf_deg : int
            Polynomial degree
        n_jobs : int
            Number of jobs
            
        Returns
        -------
        dict
            Dictionary of emission line results
        """
        logger.info("Running fit_emission_lines on binned data adapter")
        
        try:
            # Create simplified emission results structure
            n_wave_fit = len(self._lambda_gal)
            
            # Initialize empty emission line results
            result = {
                'emission_flux': {},
                'emission_vel': {},
                'emission_sig': {},
                'gas_bestfit_field': np.full((n_wave_fit, self._n_y, self._n_x), np.nan),
                'gas_bestfit': np.full((n_wave_fit, self._n_y, self._n_x), np.nan),
                'weights': self._template_weights if hasattr(self, '_template_weights') else None,
            }
            
            # Return minimal result - emission line fitting could be added later if needed
            return result
            
        except Exception as e:
            logger.error(f"Error in adapter fit_emission_lines: {str(e)}")
            return {}
    
    def calculate_spectral_indices(self, n_jobs=-1):
        """
        Adapter implementation of calculate_spectral_indices
        
        Parameters
        ----------
        n_jobs : int
            Number of jobs
            
        Returns
        -------
        dict
            Dictionary of spectral indices
        """
        from spectral_indices import LineIndexCalculator
        
        logger.info("Running calculate_spectral_indices on binned data adapter")
        
        try:
            # Initialize result dictionary
            indices_result = {}
            
            # Define indices to calculate
            indices_list = ['Hbeta', 'Mgb', 'Fe5015']
            
            # Initialize arrays
            for index_name in indices_list:
                indices_result[index_name] = np.full((self._n_y, self._n_x), np.nan)
            
            # Process each bin
            for bin_idx in range(self._n_x):
                # Skip bins without fits
                if np.isnan(self._velocity_field[0, bin_idx]):
                    continue
                
                try:
                    # Get data for this bin
                    galaxy_data = self._spectra[:, bin_idx]
                    fit_data = self._bestfit_field[:, 0, bin_idx]
                    
                    # Skip bins with insufficient data
                    if not np.any(np.isfinite(galaxy_data)) or not np.any(np.isfinite(fit_data)):
                        continue
                    
                    # Calculate indices
                    calculator = LineIndexCalculator(
                        wave=self._lambda_gal,
                        flux=galaxy_data,
                        fit_wave=self._sps.lam_temp if hasattr(self, '_sps') else self._lambda_gal,
                        fit_flux=self._optimal_tmpls[:, 0, bin_idx] if hasattr(self, '_optimal_tmpls') else fit_data,
                        velocity_correction=self._velocity_field[0, bin_idx],
                        continuum_mode='auto',
                        show_warnings=False
                    )
                    
                    # Calculate each index
                    for index_name in indices_list:
                        try:
                            index_value = calculator.calculate_index(index_name)
                            indices_result[index_name][0, bin_idx] = index_value
                        except Exception as e:
                            logger.debug(f"Error calculating {index_name} for bin {bin_idx}: {e}")
                            
                except Exception as e:
                    logger.debug(f"Error processing bin {bin_idx} for indices: {e}")
            
            return indices_result
            
        except Exception as e:
            logger.error(f"Error in adapter calculate_spectral_indices: {str(e)}")
            return {}
    
    @classmethod
    def from_p2p_results(cls, p2p_results, bin_num, bin_indices, spectra, wavelength, metadata=None):
        """
        Create adapter directly from P2P results and binning data.
        
        Parameters
        ----------
        p2p_results : dict
            P2P analysis results
        bin_num : ndarray
            Bin numbers for each pixel
        bin_indices : list
            List of arrays with pixel indices for each bin
        spectra : ndarray
            Binned spectra
        wavelength : ndarray
            Wavelength array
        metadata : dict, optional
            Additional metadata
            
        Returns
        -------
        BinnedDataAdapter
            Adapter object for P2P processing
        """
        from binning import BinnedSpectra
        
        # First create a BinnedSpectra object
        if metadata is None:
            metadata = {}
        
        # Add key information from P2P results to metadata
        if 'global_kinematics' in p2p_results:
            metadata.update(p2p_results['global_kinematics'])
        
        # Add pixel size information if available
        if 'distance' in p2p_results:
            if 'pixelsize_x' in p2p_results['distance']:
                metadata['pixelsize_x'] = p2p_results['distance']['pixelsize_x']
                metadata['pixelsize_y'] = p2p_results['distance']['pixelsize_y']
        
        # Create BinnedSpectra object
        binned_data = BinnedSpectra(
            bin_num=bin_num,
            bin_indices=bin_indices,
            spectra=spectra, 
            wavelength=wavelength,
            metadata=metadata
        )
        
        # Create adapter
        adapter = cls(binned_data)
        
        # Add velocity field directly from P2P results
        if 'stellar_kinematics' in p2p_results and 'velocity_field' in p2p_results['stellar_kinematics']:
            adapter.velocity_field = p2p_results['stellar_kinematics']['velocity_field']
        
        # Add dispersion field directly from P2P results
        if 'stellar_kinematics' in p2p_results and 'dispersion_field' in p2p_results['stellar_kinematics']:
            adapter.dispersion_field = p2p_results['stellar_kinematics']['dispersion_field']
            
        # Add emission lines if available
        if 'emission' in p2p_results:
            adapter.emission = p2p_results['emission']
            
            # Add gas velocity if available
            if 'velocity_field' in p2p_results['emission']:
                adapter.gas_velocity_field = p2p_results['emission']['velocity_field']
                
            # Add gas dispersion if available
            if 'dispersion_field' in p2p_results['emission']:
                adapter.gas_dispersion_field = p2p_results['emission']['dispersion_field']
        
        return adapter


def create_p2p_processor(p2p_function: Callable) -> Callable:
    """
    Create a function that processes binned data using a P2P function.
    
    Parameters
    ----------
    p2p_function : callable
        Original P2P processing function
        
    Returns
    -------
    callable
        Function that can process binned data
    """
    def processor(args, binned_data, p2p_results=None):
        """
        Process binned data using P2P function.
        
        Parameters
        ----------
        args : argparse.Namespace
            Command line arguments
        binned_data : BinnedSpectra or BinnedDataAdapter
            Binned data
        p2p_results : dict, optional
            P2P analysis results
            
        Returns
        -------
        dict
            P2P processing results
        """
        # If binned_data is not already an adapter, create one
        if not isinstance(binned_data, BinnedDataAdapter):
            try:
                if p2p_results is not None:
                    # Use more detailed constructor with P2P results
                    adapter = BinnedDataAdapter.from_p2p_results(
                        p2p_results=p2p_results,
                        bin_num=binned_data.bin_num,
                        bin_indices=binned_data.bin_indices,
                        spectra=binned_data.spectra,
                        wavelength=binned_data.wavelength,
                        metadata=binned_data.metadata
                    )
                else:
                    # Use standard constructor
                    adapter = BinnedDataAdapter(binned_data)
            except Exception as e:
                logger.error(f"Failed to create BinnedDataAdapter: {str(e)}")
                logger.error(traceback.format_exc())
                # Fallback to original adapter creation
                adapter = BinnedDataAdapter(binned_data)
        else:
            adapter = binned_data
        
        # Add velocity field from P2P results if available and not already present
        if p2p_results is not None:
            if not hasattr(adapter, 'velocity_field') and 'stellar_kinematics' in p2p_results:
                if 'velocity_field' in p2p_results['stellar_kinematics']:
                    adapter.velocity_field = p2p_results['stellar_kinematics']['velocity_field']
            
            if not hasattr(adapter, 'dispersion_field') and 'stellar_kinematics' in p2p_results:
                if 'dispersion_field' in p2p_results['stellar_kinematics']:
                    adapter.dispersion_field = p2p_results['stellar_kinematics']['dispersion_field']
        
        # Run P2P function
        return p2p_function(args, adapter)
    
    return processor


def extract_bin_results(p2p_results: Dict[str, Any], 
                        binned_data_adapter: BinnedDataAdapter,
                        result_type: str = 'both') -> Dict[str, Any]:
    """
    Extract results for binned data from P2P results.
    
    Parameters
    ----------
    p2p_results : dict
        Results from P2P processing
    binned_data_adapter : BinnedDataAdapter
        Adapter that was used for processing
    result_type : str
        Type of results to extract ('both', 'vnb', or 'rdb')
        
    Returns
    -------
    dict
        Results transformed for binned data
    """
    # Extract key information
    bin_indices = binned_data_adapter.bin_indices
    n_bins = len(bin_indices)
    
    # Create result container
    bin_results = {}
    
    # Copy direct results
    if 'velocity_field' in p2p_results:
        bin_results['velocity_field'] = p2p_results['velocity_field'].copy()
    
    if 'sigma_field' in p2p_results:
        bin_results['sigma_field'] = p2p_results['sigma_field'].copy()
    
    # Copy emission line results
    if 'emission_lines' in p2p_results:
        bin_results['emission_lines'] = {}
        for line_name, line_data in p2p_results['emission_lines'].items():
            bin_results['emission_lines'][line_name] = {
                k: v.copy() for k, v in line_data.items() if isinstance(v, np.ndarray)
            }
    
    # Copy spectral indices
    if 'spectral_indices' in p2p_results:
        bin_results['spectral_indices'] = {
            k: v.copy() for k, v in p2p_results['spectral_indices'].items()
        }
    
    # Copy stellar population parameters
    if 'stellar_pop' in p2p_results:
        bin_results['stellar_pop'] = {
            k: v.copy() for k, v in p2p_results['stellar_pop'].items()
        }
    
    # Add binning metadata
    bin_results['bin_metadata'] = {
        'n_bins': n_bins,
        'bin_indices': bin_indices
    }
    
    return bin_results


def load_p2p_results_for_galaxy(galaxy_name, output_dir):
    """
    Load P2P results for a given galaxy with enhanced search and validation.
    
    Parameters
    ----------
    galaxy_name : str
        Galaxy name
    output_dir : str or Path
        Output directory
        
    Returns
    -------
    dict or None
        P2P analysis results, or None if not found
    """
    from utils.io import load_results_from_npz
    
    output_dir = Path(output_dir)
    galaxy_dir = output_dir / galaxy_name
    data_dir = galaxy_dir / 'Data'
    
    # Create paths list to try in order
    search_paths = [
        data_dir / f"{galaxy_name}_P2P_results.npz",
        data_dir / f"{galaxy_name}_P2P_standardized.npz",
        data_dir / f"P2P_{galaxy_name}.npz",
        galaxy_dir / f"{galaxy_name}_P2P_results.npz",
        data_dir / "P2P_results.npz"
    ]
    
    # Additional search for any P2P-related files
    try:
        for pattern in [f"*{galaxy_name}*P2P*.npz", f"*P2P*{galaxy_name}*.npz", "*P2P*.npz"]:
            search_paths.extend(list(data_dir.glob(pattern)))
    except Exception:
        pass
    
    # Try each path
    for p2p_results_path in search_paths:
        if p2p_results_path.exists():
            try:
                logger.info(f"Attempting to load P2P results from {p2p_results_path}")
                p2p_results = load_results_from_npz(p2p_results_path)
                
                # Validate the loaded data
                if p2p_results is not None:
                    # Check for key fields
                    has_velocity = False
                    
                    # Check standard format
                    if 'stellar_kinematics' in p2p_results and 'velocity_field' in p2p_results['stellar_kinematics']:
                        velocity_field = p2p_results['stellar_kinematics']['velocity_field']
                        has_velocity = velocity_field is not None and not np.all(np.isnan(velocity_field))
                    
                    # Check alternate format
                    elif 'velocity_field' in p2p_results:
                        velocity_field = p2p_results['velocity_field']
                        has_velocity = velocity_field is not None and not np.all(np.isnan(velocity_field))
                    
                    if has_velocity:
                        logger.info(f"Successfully loaded valid P2P results from {p2p_results_path}")
                        return p2p_results
                    else:
                        logger.warning(f"Loaded P2P results from {p2p_results_path} but velocity field is missing or invalid")
            except Exception as e:
                logger.warning(f"Failed to load P2P results from {p2p_results_path}: {str(e)}")
    
    # If we reach here, no valid results were found
    logger.warning(f"No valid P2P results found for galaxy {galaxy_name}")
    return None