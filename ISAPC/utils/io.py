"""
Input/output utility functions for ISAPC
"""
import os
import numpy as np
import logging
from astropy.io import fits
from pathlib import Path
from typing import Optional, Dict, Union, List, Any

logger = logging.getLogger(__name__)

def save_results_to_npz(output_file: Union[str, Path], data_dict: Dict[str, Any]) -> None:
    """
    Save results dictionary to NPZ file with enhanced handling of variable-length arrays.
    
    Parameters
    ----------
    output_file : str or Path
        Path to save the NPZ file
    data_dict : dict
        Dictionary of results to save
    """
    # Convert Path to string
    output_file = str(output_file)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Filter out objects that can't be saved
    filtered_dict = {}
    for key, value in data_dict.items():
        try:
            # Convert pandas DataFrame to dictionary
            if hasattr(value, 'to_dict'):
                filtered_dict[key] = value.to_dict()
            # Handle lists of arrays (like bin_indices) that can cause issues
            elif key == 'bin_indices' and isinstance(value, list):
                # For bin_indices, store as object array to handle variable lengths
                indices_array = np.empty(len(value), dtype=object)
                for i, indices in enumerate(value):
                    indices_array[i] = indices
                filtered_dict[key] = indices_array
            # Process nested dictionaries
            elif isinstance(value, dict):
                nested_dict = {}
                for nested_key, nested_value in value.items():
                    if hasattr(nested_value, 'to_dict'):
                        nested_dict[nested_key] = nested_value.to_dict()
                    else:
                        nested_dict[nested_key] = nested_value
                filtered_dict[key] = nested_dict
            else:
                filtered_dict[key] = value
        except Exception as e:
            logger.warning(f"Could not save '{key}': {e}")
    
    try:
        np.savez_compressed(output_file, **filtered_dict)
        logger.info(f"Saved results to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")
        
        # Try to save with pickle protocol for object arrays (fallback method)
        try:
            # Use allow_pickle=True explicitly
            np.savez(output_file, allow_pickle=True, **filtered_dict)
            logger.info(f"Saved results to {output_file} using pickle protocol")
        except Exception as e2:
            logger.error(f"Error saving results even with pickle: {e2}")


def load_results_from_npz(input_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load results dictionary from NPZ file.
    
    Parameters
    ----------
    input_file : str or Path
        Path to the NPZ file
        
    Returns
    -------
    dict
        Dictionary of results
    """
    # Convert Path to string
    input_file = str(input_file)
    
    try:
        data = np.load(input_file, allow_pickle=True)
        
        # Convert to regular dictionary
        result_dict = {}
        for key in data.files:
            result_dict[key] = data[key].item() if data[key].dtype == np.dtype('O') else data[key]
        
        logger.info(f"Loaded results from {input_file}")
        return result_dict
    except Exception as e:
        logger.error(f"Error loading results from {input_file}: {e}")
        return {}


def load_fits_data(input_file: Union[str, Path], hdu_index: int = 0) -> np.ndarray:
    """
    Load data from FITS file.
    
    Parameters
    ----------
    input_file : str or Path
        Path to the FITS file
    hdu_index : int, default=0
        HDU index to load
        
    Returns
    -------
    ndarray
        Data array
    """
    # Convert Path to string
    input_file = str(input_file)
    
    try:
        with fits.open(input_file) as hdul:
            data = hdul[hdu_index].data
        logger.info(f"Loaded FITS data from {input_file}, HDU {hdu_index}")
        return data
    except Exception as e:
        logger.error(f"Error loading FITS data from {input_file}: {e}")
        return None


def save_fits_data(output_file: Union[str, Path], data: np.ndarray, header: Optional[fits.Header] = None) -> None:
    """
    Save data to FITS file.
    
    Parameters
    ----------
    output_file : str or Path
        Path to save the FITS file
    data : ndarray
        Data array to save
    header : fits.Header, optional
        Header to include in the FITS file
    """
    # Convert Path to string
    output_file = str(output_file)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        hdu = fits.PrimaryHDU(data, header=header)
        hdu.writeto(output_file, overwrite=True)
        logger.info(f"Saved FITS data to {output_file}")
    except Exception as e:
        logger.error(f"Error saving FITS data to {output_file}: {e}")


def find_template(search_paths: Optional[List[str]] = None) -> Optional[str]:
    """
    Find a suitable stellar template file.
    
    Parameters
    ----------
    search_paths : list of str, optional
        Paths to search for template files
        
    Returns
    -------
    str or None
        Path to the template file, or None if not found
    """
    if search_paths is None:
        search_paths = [
            '.',
            './templates',
            '../templates',
            os.path.expanduser('~/templates'),
            os.path.expanduser('~/.isapc/templates')
        ]
    
    # Add ISAPC_TEMPLATES env var if it exists
    if 'ISAPC_TEMPLATES' in os.environ:
        search_paths.insert(0, os.environ['ISAPC_TEMPLATES'])
    
    # File patterns to search for
    patterns = [
        '*.fits',
        '*.npz',
        'templates*.npz',
        'miles*.npz',
        'MILES*.npz'
    ]
    
    # Search for template files
    for path in search_paths:
        if not os.path.exists(path):
            continue
            
        for pattern in patterns:
            files = list(Path(path).glob(pattern))
            if files:
                return str(files[0])
    
    return None


def save_standardized_results(galaxy_name, analysis_type, results, base_dir):
    """
    Save results in a standardized format and directory structure.
    
    Parameters
    ----------
    galaxy_name : str
        Name of the galaxy
    analysis_type : str
        Analysis type ('P2P', 'VNB', or 'RDB')
    results : dict
        Results dictionary
    base_dir : Path or str
        Base output directory
    """
    base_dir = Path(base_dir)
    
    # Create standardized directory structure
    galaxy_dir = base_dir / galaxy_name
    data_dir = galaxy_dir / 'Data'
    
    galaxy_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True)
    
    # Main results file
    main_file = data_dir / f"{galaxy_name}_{analysis_type}_results.npz"
    save_results_to_npz(main_file, results)
    
    # Save key components separately for convenience
    components = {
        'stellar_kinematics': ['velocity_field', 'dispersion_field'],
        'emission': ['velocity_field', 'dispersion_field', 'emission_flux'],
        'indices': None,  # Save all indices
        'stellar_population': ['log_age', 'age', 'metallicity']
    }
    
    for component, fields in components.items():
        if component in results:
            component_data = {}
            
            if fields is None:
                # Save entire component
                component_data = results[component]
            else:
                # Extract specified fields
                for field in fields:
                    if field in results[component]:
                        component_data[field] = results[component][field]
            
            if component_data:
                component_file = data_dir / f"{galaxy_name}_{analysis_type}_{component}.npz"
                save_results_to_npz(component_file, component_data)
    
    # Save binning info for VNB and RDB
    if analysis_type in ['VNB', 'RDB'] and 'bin_info' in results:
        binning_file = data_dir / f"{galaxy_name}_{analysis_type}_binning.npz"
        save_results_to_npz(binning_file, results['bin_info'])
    
    logger.info(f"Saved standardized {analysis_type} results for {galaxy_name}")



