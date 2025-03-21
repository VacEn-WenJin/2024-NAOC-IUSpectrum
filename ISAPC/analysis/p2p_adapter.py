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
            
            # Store original binning information
            self.bin_num = p2p_data['bin_num']
            self.bin_indices = p2p_data['bin_indices']
            self.metadata = p2p_data['metadata']
            
            # Calculate logarithmic wavelength grid as some P2P code might expect it
            c = 299792.458  # Speed of light in km/s
            dlambda = np.min(np.diff(self.wave))
            self.velscale = c * dlambda / self.wave[0]
            self._ln_lam_gal = np.log(self.wave)
            
            # Create fake spatial dimensions for P2P
            self.ny, self.nx = 1, self.cube.shape[2]
            self.x = p2p_data['x']
            self.y = p2p_data['y']
            
            # Copy additional properties from metadata if available
            if 'redshift' in p2p_data:
                self._redshift = p2p_data['redshift']
            if 'pxl_size_x' in p2p_data:
                self._pxl_size_x = p2p_data['pxl_size_x']
                self._pxl_size_y = p2p_data['pxl_size_y']
            
        except Exception as e:
            logger.error(f"Error initializing BinnedDataAdapter: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
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