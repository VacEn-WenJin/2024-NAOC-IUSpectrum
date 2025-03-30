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
        Adapter implementation of fit_spectra - optimized with ParallelTqdm for binned data
        
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
        from utils.parallel import ParallelTqdm
        from joblib import delayed
        from ppxf import ppxf_util
        from ppxf.ppxf import ppxf
        from ppxf.sps_util import sps_lib
        import warnings
        
        logger.info("Running fit_spectra on binned data adapter")
        
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
            
            # Define function to process a single bin
            def fit_bin(bin_idx):
                """Fit a single bin's spectrum"""
                galaxy_data = self._spectra[:, bin_idx]
                galaxy_noise = np.sqrt(self._log_variance[:, bin_idx])
                
                # Skip low SNR or invalid bins
                if np.count_nonzero(galaxy_data) < 50 or np.count_nonzero(np.isfinite(galaxy_data)) < 50:
                    return bin_idx, None
                
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
                        
                        # Calculate polynomial coefficients for later use
                        poly_coeff = np.polyfit(self._lambda_gal, pp.apoly, ppxf_deg)
                        
                        # Calculate optimal template directly from weights on TEMPLATE wavelength grid
                        optimal_template = sps.templates @ pp.weights
                        
                        # Calculate best-fit on GALAXY wavelength grid
                        bestfit = pp.bestfit
                        
                        # Return successful result
                        return bin_idx, (
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
                                start=[ppxf_vel_init, ppxf_vel_disp_init], degree=0,  # Simplify
                                lam=self._lambda_gal, lam_temp=sps.lam_temp,
                                quiet=True
                            )
                            
                            # Calculate polynomial coefficients (constant term)
                            poly_coeff = np.array([pp.apoly[0]])
                            
                            # Calculate optimal template directly from weights on TEMPLATE wavelength grid
                            optimal_template = sps.templates @ pp.weights
                            
                            # Calculate best-fit on GALAXY wavelength grid
                            bestfit = pp.bestfit
                            
                            # Return successful result
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
            fit_results = ParallelTqdm(
                n_jobs=n_jobs, desc='Fitting spectra', total_tasks=n_bins
            )(delayed(fit_bin)(bin_idx) for bin_idx in range(n_bins))
            
            # Process results
            for bin_idx, result in fit_results:
                if result is None:
                    continue
                    
                vel, disp, bestfit, optimal_tmpl, weights, poly_coeff = result
                
                # Store results with validation
                self._velocity_field[0, bin_idx] = vel
                
                # Ensure dispersion value is reasonable
                self._dispersion_field[0, bin_idx] = max(disp, 10.0)  # Minimum dispersion value
                
                # Store best-fit on GALAXY wavelength grid
                self._bestfit_field[:, 0, bin_idx] = bestfit
                
                # Store optimal template on TEMPLATE wavelength grid
                self._optimal_tmpls[:, 0, bin_idx] = optimal_tmpl
                    
                # Store template weights
                self._template_weights[:len(weights), 0, bin_idx] = weights
                    
                self._poly_coeffs.append((0, bin_idx, poly_coeff))
            
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
    
    def fit_emission_lines(self, template_filename=None, line_names=None,
                          ppxf_vel_init=None, ppxf_sig_init=50.0, ppxf_deg=2, n_jobs=-1, verbose=False):
        """
        Adapter implementation of fit_emission_lines for binned data
        
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
        ppxf_deg : int, default=2
            Degree of additive polynomial for pPXF
        n_jobs : int, default=-1
            Number of parallel jobs to run
        verbose : bool, default=False
            Whether to print verbose output
                
        Returns
        -------
        Dict[str, Any]
            Dictionary containing emission line fitting results
        """
        # Set log level
        logger_level = logging.getLogger().level
        if not verbose:
            logging.getLogger().setLevel(logging.WARNING)
        
        try:
            from utils.parallel import ParallelTqdm
            from joblib import delayed
            import warnings
            from ppxf.ppxf import ppxf
            from ppxf.ppxf_util import emission_lines
            
            logger.info("Running fit_emission_lines on binned data adapter")
            
            # Check if stellar fitting has been performed
            if not hasattr(self, '_sps') or self._sps is None or not hasattr(self, '_optimal_tmpls'):
                logger.warning("Must run fit_spectra() before fit_emission_lines()")
                return {
                    'emission_flux': {},
                    'emission_vel': {},
                    'emission_sig': {},
                    'gas_bestfit_field': np.zeros_like(self._spectra).reshape(self._spectra.shape[0], 1, -1)
                }
            
            # Set up velocity initialization
            if ppxf_vel_init is None:
                # Use stellar velocity field as initial value
                ppxf_vel_init = self._velocity_field
            
            # Initialize result storage
            n_wave = len(self._lambda_gal)
            n_bins = self._n_x
            
            self._emission_flux = {}
            self._emission_vel = {}
            self._emission_sig = {}
            self._gas_bestfit_field = np.full((n_wave, self._n_y, n_bins), np.nan)
            self._emission_wavelength = {}
            
            # Generate emission line templates using ppxf's emission_lines function
            lam_range_gal = [np.min(self._lambda_gal), np.max(self._lambda_gal)]
            FWHM_gal = getattr(self, '_FWHM_gal', 1.0)  # Default FWHM if not available
            redshift = getattr(self, '_redshift', 0.0)  # Default redshift if not available
            
            # Generate gas templates
            gas_templates, gas_names, line_wave = emission_lines(
                self._sps.ln_lam_temp, lam_range_gal, FWHM_gal / (1 + redshift) 
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
            logger.info(f"Emission lines included in gas templates: {gas_names}")
            
            # Initialize emission line storage
            for name in gas_names:
                base_name = name.split('_(')[0] if '_(' in name else name
                if base_name not in self._emission_flux:
                    self._emission_flux[base_name] = np.full((self._n_y, n_bins), np.nan)
                    self._emission_vel[base_name] = np.full((self._n_y, n_bins), np.nan)
                    self._emission_sig[base_name] = np.full((self._n_y, n_bins), np.nan)
            
            # Store ppxf results for each bin
            self._ppxf_gas_results = []
            
            # Define function to process a single bin
            def fit_bin_emission(bin_idx):
                """Fit emission lines for a single bin"""
                # Skip bins with no valid velocity measurement
                if np.isnan(self._velocity_field[0, bin_idx]):
                    return bin_idx, None
                
                # Get bin data
                galaxy_data = self._spectra[:, bin_idx]
                galaxy_noise = np.sqrt(self._log_variance[:, bin_idx])
                
                # Skip bins with insufficient data
                if (np.count_nonzero(galaxy_data) < 50 or 
                    np.count_nonzero(np.isfinite(galaxy_data)) < 50):
                    return bin_idx, None
                    
                # Replace NaN values
                galaxy_data = np.nan_to_num(galaxy_data, nan=0.0, posinf=0.0, neginf=0.0)
                galaxy_noise = np.nan_to_num(galaxy_noise, nan=1.0, posinf=1.0, neginf=1.0)
                
                # Get optimal stellar template for this bin
                optimal_template = self._optimal_tmpls[:, 0, bin_idx]
                
                # Get initial velocity
                vel_init = self._velocity_field[0, bin_idx]
                
                try:
                    # Combine stellar and gas templates
                    stars_gas_templates = np.column_stack([optimal_template, gas_templates])
                    
                    # Define component types - [0] for stellar, [1] for gas components
                    component = [0] + [1]*gas_templates.shape[1]  # Stellar + gas components
                    gas_component = np.array(component) > 0  # True for gas components
                    
                    # Define moments for each component type
                    moments = [2, 2]  # 2 moments for both stellar and gas (vel, sigma)
                    ncomp = len(moments)  # Should be 2
                    
                    # Set initial parameters
                    start = [
                        [vel_init, self._dispersion_field[0, bin_idx]],  # Stellar initial kinematics
                        [vel_init, ppxf_sig_init]                      # Gas initial kinematics
                    ]
                    
                    # Set boundary conditions
                    vlim = lambda x: vel_init + x*np.array([-300, 300])  # Wider range for binned data
                    bounds = [
                        [vlim(1), [1, 300]],  # Stellar bounds
                        [vlim(1), [1, 200]]   # Gas bounds
                    ]
                    
                    # Call ppxf with appropriate parameters
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        
                        # Ensure noise values are not zero
                        galaxy_noise = np.maximum(galaxy_noise, 1e-10)
                        
                        pp = ppxf(
                            stars_gas_templates, galaxy_data, galaxy_noise, 
                            self._vel_scale, start,
                            moments=moments, degree=ppxf_deg, mdegree=-1,
                            component=component, 
                            gas_component=gas_component, 
                            gas_names=gas_names, 
                            lam=self._lambda_gal,
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
                        # Try to extract gas component from the full model
                        comp = pp.component
                        gas_idx = np.where(comp > 0)[0]
                        if len(gas_idx) > 0:
                            gas_bestfit = np.sum(pp.matrix[:, gas_idx] @ pp.weights[gas_idx], axis=1)
                    
                    # Calculate stellar component
                    stellar_bestfit = bestfit - gas_bestfit
                    
                    # Get stellar and gas kinematics
                    stellar_sol = pp.sol[0]  # Stellar kinematics
                    gas_sol = pp.sol[1]      # Gas kinematics
                    
                    # Prepare results
                    result = {
                        'gas_bestfit': gas_bestfit,
                        'stellar_bestfit': stellar_bestfit,
                        'total_bestfit': bestfit,
                        'sol': stellar_sol,
                        'gas_sol': gas_sol,
                        'weights': pp.weights
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
            fit_results = ParallelTqdm(
                n_jobs=n_jobs, desc='Fitting emission lines', total_tasks=n_bins
            )(delayed(fit_bin_emission)(bin_idx) for bin_idx in range(n_bins))
            
            # Process results
            for bin_idx, result in fit_results:
                if result is None:
                    continue
                
                # Store gas bestfit
                self._gas_bestfit_field[:, 0, bin_idx] = result['gas_bestfit']
                
                # Store emission line results
                if 'emission_flux' in result:
                    for base_name, flux in result['emission_flux'].items():
                        self._emission_flux[base_name][0, bin_idx] = flux
                        self._emission_vel[base_name][0, bin_idx] = result['emission_vel'][base_name]
                        self._emission_sig[base_name][0, bin_idx] = result['emission_sig'][base_name]
                elif 'gas_flux' in result:
                    for k, name in enumerate(gas_names):
                        base_name = name.split('_(')[0] if '_(' in name else name
                        self._emission_flux[base_name][0, bin_idx] = result['gas_flux'][k]
                        self._emission_vel[base_name][0, bin_idx] = result['gas_sol'][0]
                        self._emission_sig[base_name][0, bin_idx] = result['gas_sol'][1]
                
                # Store the full result
                self._ppxf_gas_results.append((0, bin_idx, result))
            
            # Calculate SNR information for each bin
            snr_info = self.calculate_snr() if hasattr(self, 'calculate_snr') else None
            
            # Prepare result dictionary
            if snr_info is not None:
                result_dict = {
                    'emission_flux': self._emission_flux,
                    'emission_vel': self._emission_vel,
                    'emission_sig': self._emission_sig,
                    'gas_bestfit_field': self._gas_bestfit_field,
                    'emission_wavelength': self._emission_wavelength,
                    'optimal_tmpls': self._optimal_tmpls,
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
                    'optimal_tmpls': self._optimal_tmpls,
                    'velocity_field': self._velocity_field,
                    'dispersion_field': self._dispersion_field
                }
            
            # Restore log level
            logging.getLogger().setLevel(logger_level)
            return result_dict
            
        except Exception as e:
            logger.error(f"Error in adapter fit_emission_lines: {str(e)}")
            logging.getLogger().setLevel(logger_level)
            return {
                'emission_flux': {},
                'emission_vel': {},
                'emission_sig': {},
                'gas_bestfit_field': np.full((n_wave, self._n_y, n_bins), np.nan),
                'emission_wavelength': {}
            }
    
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
        try:
            # Check if fitting has been performed
            if not hasattr(self, '_bestfit_field') or self._bestfit_field is None:
                logger.warning("No spectral fitting results available for SNR calculation")
                return None
            
            # Initialize result arrays
            n_bins = self._n_x
            snr_map = np.full((self._n_y, n_bins), np.nan)
            signal_map = np.full((self._n_y, n_bins), np.nan)
            noise_map = np.full((self._n_y, n_bins), np.nan)
            
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
            
            # Calculate SNR for each valid bin
            valid_mask = ~np.isnan(self._velocity_field[0])
            for bin_idx in range(n_bins):
                if valid_mask[bin_idx]:
                    # Get observed and model spectra for continuum region
                    observed = self._spectra[wave_mask, bin_idx]
                    model = self._bestfit_field[wave_mask, 0, bin_idx]
                    
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
                    snr_map[0, bin_idx] = snr
                    signal_map[0, bin_idx] = signal
                    noise_map[0, bin_idx] = noise
            
            return {
                'snr': snr_map,
                'signal': signal_map,
                'noise': noise_map,
                'wavelength_range': continuum_range
            }
        
        except Exception as e:
            logger.error(f"Error calculating SNR: {str(e)}")
            return None
    
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
        from spectral_indices import LineIndexCalculator, set_warnings
        from utils.parallel import ParallelTqdm
        from joblib import delayed
        
        # Control warnings
        set_warnings(verbose)
        
        logger.info("Running calculate_spectral_indices on binned data adapter")
        
        try:
            # Initialize result dictionary
            indices_result = {}
            
            # Define indices to calculate
            if indices_list is None:
                indices_list = ['Hbeta', 'Mgb', 'Fe5015']
            
            # Initialize arrays
            for index_name in indices_list:
                indices_result[index_name] = np.full((self._n_y, self._n_x), np.nan)
            
            # Define function to process a single bin
            def process_bin(bin_idx):
                # Skip bins without fits
                if not hasattr(self, '_velocity_field') or np.isnan(self._velocity_field[0, bin_idx]):
                    return bin_idx, {name: np.nan for name in indices_list}
                
                try:
                    # Get data for this bin
                    galaxy_data = self._spectra[:, bin_idx]
                    
                    # Skip bins with insufficient data
                    if not np.any(np.isfinite(galaxy_data)):
                        return bin_idx, {name: np.nan for name in indices_list}
                    
                    # Get corresponding fit data if available
                    fit_data = None
                    fit_wave = None
                    
                    if hasattr(self, '_sps') and self._sps is not None:
                        fit_wave = self._sps.lam_temp
                        if hasattr(self, '_optimal_tmpls') and self._optimal_tmpls is not None:
                            fit_data = self._optimal_tmpls[:, 0, bin_idx]
                    else:
                        # Use wavelength grid as fallback
                        fit_wave = self._lambda_gal
                        
                    if fit_data is None or not np.any(np.isfinite(fit_data)):
                        # Try using bestfit if available
                        if hasattr(self, '_bestfit_field') and self._bestfit_field is not None:
                            fit_data = self._bestfit_field[:, 0, bin_idx]
                            fit_wave = self._lambda_gal  # Use observed wavelength grid
                    
                    if fit_data is None or not np.any(np.isfinite(fit_data)):
                        # Use observed data as a fallback
                        fit_data = galaxy_data
                        fit_wave = self._lambda_gal
                    
                    # Get emission line data if available
                    em_wave = None
                    em_flux = None
                    if hasattr(self, '_gas_bestfit_field') and self._gas_bestfit_field is not None:
                        gas_data = self._gas_bestfit_field[:, 0, bin_idx]
                        if np.any(np.isfinite(gas_data)):
                            em_wave = self._lambda_gal
                            em_flux = gas_data
                    
                    # Get velocity value (ensure it's finite)
                    velocity = self._velocity_field[0, bin_idx]
                    if not np.isfinite(velocity):
                        velocity = 0.0
                        
                    # Create a copy of galaxy data to avoid modifying original
                    galaxy_data_copy = galaxy_data.copy()
                    
                    # Handle NaNs in data
                    galaxy_data_copy[~np.isfinite(galaxy_data_copy)] = 0.0
                    fit_data = np.nan_to_num(fit_data, nan=0.0)
                    
                    # Create more robust wavelength range check
                    # Define standard index wavelength windows
                    index_windows = {
                        'Hbeta': (4800, 4900),    # Slightly wider range than exact index definition
                        'Mgb': (5100, 5250),      # Slightly wider range
                        'Fe5015': (4900, 5100)    # Slightly wider range
                    }
                    
                    # Check if necessary wavelength ranges are available
                    can_calculate = {}
                    for index_name in indices_list:
                        window = index_windows.get(index_name)
                        if window is None:
                            can_calculate[index_name] = True  # No specific check
                            continue
                            
                        # Check if wavelength range is covered
                        min_wave = np.min(self._lambda_gal)
                        max_wave = np.max(self._lambda_gal)
                        
                        if min_wave <= window[0] and max_wave >= window[1]:
                            can_calculate[index_name] = True
                        else:
                            can_calculate[index_name] = False
                            logger.debug(f"Wavelength range for {index_name} ({window}) not covered by data range ({min_wave:.1f}-{max_wave:.1f})")
                    
                    # Skip if none of the indices can be calculated
                    if not any(can_calculate.values()):
                        return bin_idx, {name: np.nan for name in indices_list}
                    
                    # Calculate indices
                    try:
                        calculator = LineIndexCalculator(
                            wave=self._lambda_gal,
                            flux=galaxy_data_copy,
                            fit_wave=fit_wave,
                            fit_flux=fit_data,
                            em_wave=em_wave,
                            em_flux_list=em_flux,
                            velocity_correction=velocity,
                            continuum_mode='auto',
                            show_warnings=verbose
                        )
                        
                        # Calculate each index
                        result = {}
                        for index_name in indices_list:
                            if can_calculate[index_name]:
                                try:
                                    index_value = calculator.calculate_index(index_name)
                                    result[index_name] = index_value
                                except Exception as e:
                                    if verbose:
                                        logger.debug(f"Error calculating {index_name} for bin {bin_idx}: {e}")
                                    result[index_name] = np.nan
                            else:
                                result[index_name] = np.nan
                                
                        return bin_idx, result
                        
                    except Exception as e:
                        if verbose:
                            logger.debug(f"Failed to create LineIndexCalculator for bin {bin_idx}: {e}")
                        return bin_idx, {name: np.nan for name in indices_list}
                    
                except Exception as e:
                    if verbose:
                        logger.debug(f"Error processing bin {bin_idx} for indices: {e}")
                    return bin_idx, {name: np.nan for name in indices_list}
                    
            # Process bins in parallel
            if n_jobs == 1:
                # Sequential processing for debugging
                results = []
                for bin_idx in range(self._n_x):
                    results.append(process_bin(bin_idx))
            else:
                # Parallel processing
                results = ParallelTqdm(
                    n_jobs=n_jobs, desc='Calculating spectral indices', total_tasks=self._n_x
                )(delayed(process_bin)(bin_idx) for bin_idx in range(self._n_x))
            
            # Process results
            for bin_idx, indices in results:
                for index_name, value in indices.items():
                    indices_result[index_name][0, bin_idx] = value
            
            return indices_result
            
        except Exception as e:
            logger.error(f"Error in adapter calculate_spectral_indices: {str(e)}")
            logger.error(traceback.format_exc())
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
        # print(adapter)
        # Add velocity field from P2P results if available and not already present
        if p2p_results is not None:
            print('TS TS')
            if not hasattr(adapter, 'velocity_field') and 'stellar_kinematics' in p2p_results:
                if 'velocity_field' in p2p_results['stellar_kinematics']:
                    adapter.velocity_field = p2p_results['stellar_kinematics']['velocity_field']
            
            if not hasattr(adapter, 'dispersion_field') and 'stellar_kinematics' in p2p_results:
                if 'dispersion_field' in p2p_results['stellar_kinematics']:
                    adapter.dispersion_field = p2p_results['stellar_kinematics']['dispersion_field']
        
        # IMPORTANT: Create a modified args object to prevent saving P2P results when running on binned data
        import copy
        modified_args = copy.deepcopy(args)
        
        # Add a flag to indicate this is for binned data processing
        modified_args._is_binned_analysis = True
        
        # Set a special argument to disable automatic saving for non-P2P analysis
        modified_args.no_save = True
        
        # Run P2P function with modified args
        return p2p_function(modified_args, adapter)
    
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
    
    # Calculate mean distance for each bin if distance field is available
    if 'distance' in p2p_results and 'field' in p2p_results['distance']:
        distance_field = p2p_results['distance']['field']
        # Initialize bin distances array
        bin_distances = np.full(n_bins, np.nan)
        
        # Get dimensions for converting bin indices to 2D coordinates
        n_x = binned_data_adapter._n_x
        n_y = distance_field.shape[0]  # This should match the actual field dimensions
        
        # Calculate mean distance for each bin
        for bin_idx, indices in enumerate(bin_indices):
            if len(indices) > 0:
                # Get coordinates for these pixels
                pixel_distances = []
                for idx in indices:
                    if idx < n_x * n_y:
                        row = idx // n_x
                        col = idx % n_x
                        if 0 <= row < distance_field.shape[0] and 0 <= col < distance_field.shape[1]:
                            dist = distance_field[row, col]
                            if np.isfinite(dist):
                                pixel_distances.append(dist)
                
                # Calculate mean distance if we have valid distances
                if pixel_distances:
                    bin_distances[bin_idx] = np.mean(pixel_distances)
        
        # Add distance information to results
        bin_results['distance'] = {
            'bin_distances': bin_distances,
            'pixelsize_x': p2p_results['distance'].get('pixelsize_x', 1.0),
            'pixelsize_y': p2p_results['distance'].get('pixelsize_y', 1.0)
        }
    
    # Copy stellar kinematics results (velocity and dispersion)
    if 'stellar_kinematics' in p2p_results:
        bin_results['stellar_kinematics'] = {}
        
        # Handle velocity field
        if 'velocity_field' in p2p_results['stellar_kinematics']:
            velocity_field = p2p_results['stellar_kinematics']['velocity_field']
            bin_velocity = np.full(n_bins, np.nan)
            
            # Get dimensions for converting bin indices to 2D coordinates
            n_x = binned_data_adapter._n_x
            n_y = velocity_field.shape[0]  # This should match the actual field dimensions
            
            # Calculate mean velocity for each bin
            for bin_idx, indices in enumerate(bin_indices):
                if len(indices) > 0:
                    # Get velocities for all pixels in this bin
                    pixel_velocities = []
                    for idx in indices:
                        if idx < n_x * n_y:
                            row = idx // n_x
                            col = idx % n_x
                            if 0 <= row < velocity_field.shape[0] and 0 <= col < velocity_field.shape[1]:
                                vel = velocity_field[row, col]
                                if np.isfinite(vel):
                                    pixel_velocities.append(vel)
                    
                    # Calculate mean velocity if we have valid velocities
                    if pixel_velocities:
                        bin_velocity[bin_idx] = np.mean(pixel_velocities)
            
            bin_results['stellar_kinematics']['velocity'] = bin_velocity
        
        # Handle dispersion field
        if 'dispersion_field' in p2p_results['stellar_kinematics']:
            dispersion_field = p2p_results['stellar_kinematics']['dispersion_field']
            bin_dispersion = np.full(n_bins, np.nan)
            
            # Get dimensions for converting bin indices to 2D coordinates
            n_x = binned_data_adapter._n_x
            n_y = dispersion_field.shape[0]  # This should match the actual field dimensions
            
            # Calculate mean dispersion for each bin
            for bin_idx, indices in enumerate(bin_indices):
                if len(indices) > 0:
                    # Get dispersions for all pixels in this bin
                    pixel_dispersions = []
                    for idx in indices:
                        if idx < n_x * n_y:
                            row = idx // n_x
                            col = idx % n_x
                            if 0 <= row < dispersion_field.shape[0] and 0 <= col < dispersion_field.shape[1]:
                                disp = dispersion_field[row, col]
                                if np.isfinite(disp):
                                    pixel_dispersions.append(disp)
                    
                    # Calculate mean dispersion if we have valid dispersions
                    if pixel_dispersions:
                        bin_dispersion[bin_idx] = np.mean(pixel_dispersions)
            
            bin_results['stellar_kinematics']['dispersion'] = bin_dispersion
    
    # Copy emission line results
    if 'emission' in p2p_results:
        bin_results['emission'] = {}
        
        # Get dimensions for converting bin indices to 2D coordinates
        n_x = binned_data_adapter._n_x
        
        # Process each emission parameter
        for key, value in p2p_results['emission'].items():
            if isinstance(value, np.ndarray) and len(value.shape) == 2:
                # 2D parameter map like flux or velocity
                bin_values = np.full(n_bins, np.nan)
                n_y = value.shape[0]  # This should match the actual field dimensions
                
                # Calculate mean value for each bin
                for bin_idx, indices in enumerate(bin_indices):
                    if len(indices) > 0:
                        # Get values for all pixels in this bin
                        pixel_values = []
                        for idx in indices:
                            if idx < n_x * n_y:
                                row = idx // n_x
                                col = idx % n_x
                                if 0 <= row < value.shape[0] and 0 <= col < value.shape[1]:
                                    val = value[row, col]
                                    if np.isfinite(val):
                                        pixel_values.append(val)
                        
                        # Calculate mean value if we have valid values
                        if pixel_values:
                            bin_values[bin_idx] = np.mean(pixel_values)
                
                bin_results['emission'][key] = bin_values
            else:
                # Other parameters (e.g., dictionaries)
                bin_results['emission'][key] = value
    
    # Copy spectral indices
    if 'indices' in p2p_results:
        bin_results['indices'] = {}
        
        # Get dimensions for converting bin indices to 2D coordinates
        n_x = binned_data_adapter._n_x
        
        # Process each index
        for index_name, index_map in p2p_results['indices'].items():
            if isinstance(index_map, np.ndarray) and len(index_map.shape) == 2:
                # 2D index map
                bin_values = np.full(n_bins, np.nan)
                n_y = index_map.shape[0]  # This should match the actual field dimensions
                
                # Calculate mean value for each bin
                for bin_idx, indices in enumerate(bin_indices):
                    if len(indices) > 0:
                        # Get values for all pixels in this bin
                        pixel_values = []
                        for idx in indices:
                            if idx < n_x * n_y:
                                row = idx // n_x
                                col = idx % n_x
                                if 0 <= row < index_map.shape[0] and 0 <= col < index_map.shape[1]:
                                    val = index_map[row, col]
                                    if np.isfinite(val):
                                        pixel_values.append(val)
                        
                        # Calculate mean value if we have valid values
                        if pixel_values:
                            bin_values[bin_idx] = np.mean(pixel_values)
                
                bin_results['indices'][index_name] = bin_values
            else:
                # Other index data formats
                bin_results['indices'][index_name] = index_map
    
    # Copy stellar population parameters
    if 'stellar_population' in p2p_results:
        bin_results['stellar_population'] = {}
        
        # Get dimensions for converting bin indices to 2D coordinates
        n_x = binned_data_adapter._n_x
        
        # Process each parameter
        for param_name, param_map in p2p_results['stellar_population'].items():
            if isinstance(param_map, np.ndarray) and len(param_map.shape) == 2:
                # 2D parameter map
                bin_values = np.full(n_bins, np.nan)
                n_y = param_map.shape[0]  # This should match the actual field dimensions
                
                # Calculate mean value for each bin
                for bin_idx, indices in enumerate(bin_indices):
                    if len(indices) > 0:
                        # Get values for all pixels in this bin
                        pixel_values = []
                        for idx in indices:
                            if idx < n_x * n_y:
                                row = idx // n_x
                                col = idx % n_x
                                if 0 <= row < param_map.shape[0] and 0 <= col < param_map.shape[1]:
                                    val = param_map[row, col]
                                    if np.isfinite(val):
                                        pixel_values.append(val)
                        
                        # Calculate mean value if we have valid values
                        if pixel_values:
                            bin_values[bin_idx] = np.mean(pixel_values)
                
                bin_results['stellar_population'][param_name] = bin_values
            else:
                # Other parameter data formats
                bin_results['stellar_population'][param_name] = param_map
    
    # Add binning metadata
    bin_results['bin_metadata'] = {
        'n_bins': n_bins,
        'bin_indices': bin_indices
    }
    
    # Include global kinematics if available
    if 'global_kinematics' in p2p_results:
        bin_results['global_kinematics'] = p2p_results['global_kinematics']
    
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