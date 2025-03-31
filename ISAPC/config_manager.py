"""
Configuration management for ISAPC
Handles loading and parsing of configuration settings
"""
import os
import sys
import logging
import configparser
from pathlib import Path

logger = logging.getLogger(__name__)

# Default configuration file locations to try (in order)
CONFIG_LOCATIONS = [
    # Current working directory
    "config.ini",
    # User's home directory
    os.path.join(str(Path.home()), ".isapc", "config.ini"),
    # Installation directory
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"),
]

# Global configuration object
_config = None

def get_config():
    """Get the global configuration object, loading it if necessary"""
    global _config
    if _config is None:
        _config = load_config()
    return _config

def load_config(config_path=None):
    """
    Load configuration from file
    
    Parameters
    ----------
    config_path : str, optional
        Path to configuration file. If None, searches default locations
        
    Returns
    -------
    configparser.ConfigParser
        Loaded configuration
    """
    config = configparser.ConfigParser()
    
    # Add default values for critical settings
    config['SpectralIndices'] = {'default_indices': 'Hbeta, Fe5015, Mgb'}
    config['EmissionLines'] = {'default_lines': 'Hbeta, OIII_4959, OIII_5007'}
    config['SNR'] = {'min_wavelength': '5075', 'max_wavelength': '5125'}
    
    # Load configuration from file
    config_loaded = False
    
    if config_path is not None:
        # Try user-specified path
        if os.path.exists(config_path):
            try:
                config.read(config_path)
                logger.info(f"Loaded configuration from {config_path}")
                config_loaded = True
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {str(e)}")
        else:
            logger.warning(f"Configuration file not found: {config_path}")
    
    # If no config loaded yet, try default locations
    if not config_loaded:
        for path in CONFIG_LOCATIONS:
            if os.path.exists(path):
                try:
                    config.read(path)
                    logger.info(f"Loaded configuration from {path}")
                    config_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Error loading config from {path}: {str(e)}")
    
    if not config_loaded:
        logger.warning("No configuration file found, using default values")
    
    return config

def get_spectral_indices():
    """Get the list of default spectral indices to calculate"""
    config = get_config()
    indices_str = config.get('SpectralIndices', 'default_indices', 
                            fallback='Hbeta, Fe5015, Mgb')
    
    # Parse comma-separated list
    indices = [idx.strip() for idx in indices_str.split(',')]
    
    # Ensure Hbeta is included
    if 'Hbeta' not in indices:
        indices.append('Hbeta')
        logger.info("Added Hbeta to spectral indices list")
    
    return indices

def get_emission_lines():
    """Get the list of default emission lines to fit"""
    config = get_config()
    lines_str = config.get('EmissionLines', 'default_lines', 
                          fallback='Hbeta, OIII_4959, OIII_5007')
    
    # Parse comma-separated list
    lines = [line.strip() for line in lines_str.split(',')]
    
    return lines

def get_snr_wavelength_range():
    """Get the default wavelength range for SNR calculation"""
    config = get_config()
    min_wave = config.getfloat('SNR', 'min_wavelength', fallback=5075.0)
    max_wave = config.getfloat('SNR', 'max_wavelength', fallback=5125.0)
    
    return (min_wave, max_wave)

def get_voronoi_parameters():
    """Get default parameters for Voronoi binning"""
    config = get_config()
    
    params = {
        'target_snr': config.getfloat('VoronoiBinning', 'target_snr', fallback=30.0),
        'min_snr': config.getfloat('VoronoiBinning', 'min_snr', fallback=1.0),
        'use_cvt': config.getboolean('VoronoiBinning', 'use_cvt', fallback=True)
    }
    
    return params

def get_radial_parameters():
    """Get default parameters for Radial binning"""
    config = get_config()
    
    params = {
        'n_rings': config.getint('RadialBinning', 'n_rings', fallback=10),
        'log_spacing': config.getboolean('RadialBinning', 'log_spacing', fallback=False),
        'min_snr': config.getfloat('RadialBinning', 'min_snr', fallback=1.0)
    }
    
    return params

def get_spectral_fitting_parameters():
    """Get default parameters for spectral fitting"""
    config = get_config()
    
    params = {
        'ppxf_vel_init': config.getfloat('SpectralFitting', 'ppxf_vel_init', fallback=0.0),
        'ppxf_vel_disp_init': config.getfloat('SpectralFitting', 'ppxf_vel_disp_init', fallback=40.0),
        'ppxf_deg': config.getint('SpectralFitting', 'ppxf_deg', fallback=4),
        'ppxf_gas_deg': config.getint('SpectralFitting', 'ppxf_gas_deg', fallback=2),
        'ppxf_mdeg': config.getint('SpectralFitting', 'ppxf_mdeg', fallback=-1)
    }
    
    return params

def get_spectral_line_definition(line_name):
    """
    Get the definition for a specific spectral line
    
    Parameters
    ----------
    line_name : str
        Name of the spectral line
        
    Returns
    -------
    dict or None
        Dictionary with 'blue', 'band', 'red' tuples, or None if not found
    """
    config = get_config()
    
    if not config.has_section('SpectralLineDefinitions'):
        return None
    
    if not config.has_option('SpectralLineDefinitions', line_name):
        return None
    
    try:
        # Get definition string (blue_min, blue_max, band_min, band_max, red_min, red_max)
        definition = config.get('SpectralLineDefinitions', line_name)
        
        # Parse comma-separated values
        values = [float(val.strip()) for val in definition.split(',')]
        
        if len(values) != 6:
            logger.warning(f"Invalid spectral line definition for {line_name}: expected 6 values, got {len(values)}")
            return None
        
        # Create dictionary with ranges
        return {
            'blue': (values[0], values[1]),
            'band': (values[2], values[3]),
            'red': (values[4], values[5])
        }
    except Exception as e:
        logger.warning(f"Error parsing spectral line definition for {line_name}: {str(e)}")
        return None