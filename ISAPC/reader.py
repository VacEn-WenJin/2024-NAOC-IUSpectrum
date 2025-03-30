import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import traceback

logger = logging.getLogger(__name__)

def load_results_from_npz(file_path):
    """Load results from npz file directly"""
    import numpy as np
    data = np.load(file_path, allow_pickle=True)
    
    # Check if data contains a 'results' key which holds the actual data
    if 'results' in data:
        return data['results'].item()
    
    # Otherwise, get the first array
    keys = list(data.keys())
    if keys:
        first_key = keys[0]
        if isinstance(data[first_key], np.ndarray) and data[first_key].dtype == np.dtype('O'):
            return data[first_key].item()
    
    # As a fallback, create a dict from all keys
    return {k: data[k] for k in data}

def Read_otp(Galaxy_name, mode_name = 'P2P'):
    file_path = "./output/"+Galaxy_name+"/"+Galaxy_name+"_stack/Data/"+Galaxy_name+"_stack_"+mode_name+"_results.npz"
    df = load_results_from_npz(file_path)
    return df