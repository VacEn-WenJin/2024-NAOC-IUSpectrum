"""
Adapter module to use P2P processing with binned data
"""
import numpy as np
import logging
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional, Callable, Union

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
    def processor(args, binned_data):
        """
        Process binned data using P2P function.
        
        Parameters
        ----------
        args : argparse.Namespace
            Command line arguments
        binned_data : BinnedSpectra or BinnedDataAdapter
            Binned data
            
        Returns
        -------
        dict
            P2P processing results
        """
        # If binned_data is not already an adapter, create one
        if not isinstance(binned_data, BinnedDataAdapter):
            adapter = BinnedDataAdapter(binned_data)
        else:
            adapter = binned_data
        
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