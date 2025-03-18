"""
Input/output utility functions for ISAPC
"""
import os
import numpy as np
from astropy.io import fits
from pathlib import Path
from typing import Optional, Dict, Any


def find_template(template: Optional[str] = None) -> Optional[str]:
    """
    Find a template file, either using provided path or searching in default locations.
    
    Parameters
    ----------
    template : str, optional
        Path to template file, if already known
    
    Returns
    -------
    str or None
        Path to template file if found, None otherwise
    """
    if template is not None:
        if os.path.exists(template):
            return template
    
    # Check default locations
    default_dirs = ['templates', 'data/templates', '../templates']
    
    for directory in default_dirs:
        if os.path.exists(directory):
            templates = []
            for ext in ['.fits', '.npz']:
                templates.extend(Path(directory).glob(f"*{ext}"))
            
            if templates:
                return str(templates[0])
    
    return None


def save_results_to_fits(
    output_dir: str, 
    filename: str, 
    data_dict: Dict[str, Any], 
    header_dict: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save results to FITS files.
    
    Parameters
    ----------
    output_dir : str
        Output directory
    filename : str
        Base filename without extension
    data_dict : dict
        Dictionary of data to save
    header_dict : dict, optional
        Dictionary of header information
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create header
    hdr = fits.Header()
    if header_dict:
        for key, value in header_dict.items():
            hdr[key] = value
    
    # Save each array as a separate FITS file
    for name, data in data_dict.items():
        try:
            output_file = os.path.join(output_dir, f"{filename}_{name}.fits")
            hdu = fits.PrimaryHDU(data, header=hdr)
            hdu.writeto(output_file, overwrite=True)
        except Exception as e:
            print(f"Error saving {name} to FITS: {str(e)}")


def save_results_to_npz(
    output_file: str, 
    data_dict: Dict[str, Any]
) -> None:
    """
    Save results to NPZ file.
    
    Parameters
    ----------
    output_file : str
        Output filename with .npz extension
    data_dict : dict
        Dictionary of data to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    try:
        # Save data
        np.savez(output_file, **data_dict)
    except Exception as e:
        print(f"Error saving NPZ file: {str(e)}")
        
        # Try saving individual arrays to handle large data
        base_file = output_file.replace('.npz', '')
        for key, value in data_dict.items():
            try:
                np.save(f"{base_file}_{key}.npy", value)
            except Exception as inner_e:
                print(f"Error saving {key}: {str(inner_e)}")