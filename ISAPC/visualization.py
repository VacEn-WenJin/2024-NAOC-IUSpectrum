"""
Visualization module for IFU data analysis
Contains functions for plotting spectra, kinematic maps, binning, and more
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Ellipse
import warnings
from scipy.ndimage import gaussian_filter

import logging
logger = logging.getLogger(__name__)

def plot_spectrum(wavelength, flux, ax=None, title='Spectrum', xlabel='Wavelength (Å)', 
                 ylabel='Flux', color='k', linewidth=1, alpha=1, label=None):
    """
    Plot a single spectrum.
    
    Parameters
    ----------
    wavelength : ndarray
        Wavelength array
    flux : ndarray
        Flux array
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, default='Spectrum'
        Title for the plot
    xlabel : str, default='Wavelength (Å)'
        X-axis label
    ylabel : str, default='Flux'
        Y-axis label
    color : str, default='k'
        Line color
    linewidth : float, default=1
        Line width
    alpha : float, default=1
        Line transparency
    label : str, optional
        Label for the legend
        
    Returns
    -------
    matplotlib.axes.Axes
        Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    # Handle NaN values
    valid_mask = np.isfinite(wavelength) & np.isfinite(flux)
    if not np.any(valid_mask):
        ax.text(0.5, 0.5, "No valid data to plot", 
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    wave = wavelength[valid_mask]
    fl = flux[valid_mask]
    
    ax.plot(wave, fl, color=color, linewidth=linewidth, alpha=alpha, label=label)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if label is not None:
        ax.legend()
    
    return ax


def plot_spectrum_fit(wavelength, observed_flux, model_flux, stellar_flux=None, 
                     gas_flux=None, residual=None, mask=None, ranges=None, 
                     title='Spectrum Fit', figsize=(12, 8)):
    """
    Plot observed spectrum with fitted model components.
    
    Parameters
    ----------
    wavelength : ndarray
        Wavelength array
    observed_flux : ndarray
        Observed flux array
    model_flux : ndarray
        Model flux array
    stellar_flux : ndarray, optional
        Stellar component flux
    gas_flux : ndarray, optional
        Gas component flux
    residual : ndarray, optional
        Residual flux (observed - model)
    mask : ndarray, optional
        Boolean mask for regions to highlight
    ranges : list of tuples, optional
        List of wavelength ranges to highlight
    title : str, default='Spectrum Fit'
        Title for the plot
    figsize : tuple, default=(12, 8)
        Figure size
        
    Returns
    -------
    tuple
        (figure, axes) tuple
    """
    # Create residual if not provided
    if residual is None:
        residual = observed_flux - model_flux
    
    # Handle NaN values
    valid_mask = np.isfinite(wavelength) & np.isfinite(observed_flux) & np.isfinite(model_flux)
    if not np.any(valid_mask):
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid data to plot", 
                ha='center', va='center', transform=ax.transAxes)
        return fig, (ax,)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot observed and fitted spectrum
    ax1.plot(wavelength, observed_flux, 'k-', lw=1.5, label='Observed')
    ax1.plot(wavelength, model_flux, 'r-', lw=1.5, label='Model')
    
    # Plot components if provided and valid
    if stellar_flux is not None and np.any(np.isfinite(stellar_flux)):
        ax1.plot(wavelength, stellar_flux, 'b-', lw=1.5, label='Stellar')
    
    if gas_flux is not None and np.any(np.isfinite(gas_flux)):
        ax1.plot(wavelength, gas_flux, 'g-', lw=1.5, label='Gas')
    
    # Highlight masked regions if provided
    if mask is not None:
        masked_regions = np.ma.masked_where(~mask, observed_flux)
        ax1.plot(wavelength, masked_regions, 'y-', lw=1.5, label='Masked')
    
    # Highlight specific wavelength ranges if provided
    if ranges is not None:
        for i, (wmin, wmax) in enumerate(ranges):
            ax1.axvspan(wmin, wmax, color=f'C{i}', alpha=0.2)
    
    # Plot residuals
    ax2.plot(wavelength, residual, 'k-', lw=1.5)
    ax2.axhline(y=0, color='r', linestyle='-', lw=1.0)
    
    # Set y-axis limits with safeguards
    try:
        # Calculate y-axis limits for main plot
        valid_flux = np.concatenate([
            observed_flux[valid_mask],
            model_flux[valid_mask]
        ])
        if stellar_flux is not None and np.any(np.isfinite(stellar_flux)):
            valid_flux = np.concatenate([valid_flux, stellar_flux[valid_mask]])
        if gas_flux is not None and np.any(np.isfinite(gas_flux)):
            valid_flux = np.concatenate([valid_flux, gas_flux[valid_mask]])
        
        ymin = np.nanpercentile(valid_flux, 1)
        ymax = np.nanpercentile(valid_flux, 99)
        yrange = ymax - ymin
        
        # Add 10% padding
        ax1.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        
        # Calculate y-axis limits for residual plot
        valid_residual = residual[valid_mask]
        res_ymin = np.nanpercentile(valid_residual, 1)
        res_ymax = np.nanpercentile(valid_residual, 99)
        res_yrange = max(res_ymax - res_ymin, 1e-10)  # Avoid empty range
        
        # Add 10% padding
        ax2.set_ylim(res_ymin - 0.1*res_yrange, res_ymax + 0.1*res_yrange)
    except Exception as e:
        # Fall back to automatic scaling if percentile calculation fails
        warnings.warn(f"Error calculating plot limits: {str(e)}")
    
    # Add labels and legends
    ax1.set_ylabel('Flux')
    ax1.legend(loc='upper right')
    ax1.set_title(title)
    
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Residual')
    
    # Remove horizontal space between subplots
    plt.subplots_adjust(hspace=0)
    
    # Hide x-labels for top subplot
    for label in ax1.get_xticklabels():
        label.set_visible(False)
    
    return fig, (ax1, ax2)


def plot_velocity_field(velocity_field, mask=None, ax=None, 
                       title='Velocity Field', equal_aspect=False):
    """
    Plot the velocity field.
    
    Parameters
    ----------
    velocity_field : ndarray
        2D array of velocity values
    mask : ndarray, optional
        Boolean mask for values to exclude
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, default='Velocity Field'
        Title for the plot
    equal_aspect : bool, default=False
        Whether to keep aspect ratio equal
        
    Returns
    -------
    matplotlib.axes.Axes
        Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create masked array if mask provided
    if mask is not None:
        vel_plot = np.ma.array(velocity_field, mask=mask)
    else:
        vel_plot = np.ma.array(velocity_field, mask=~np.isfinite(velocity_field))
    
    # Check if there are any valid values
    if np.all(vel_plot.mask):
        ax.text(0.5, 0.5, "No valid velocity data", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return ax
    
    # Get symmetric color range
    valid_values = vel_plot.compressed()
    if len(valid_values) > 0:
        vabs = np.nanpercentile(np.abs(valid_values), 95)
        vmin, vmax = -vabs, vabs
    else:
        vmin, vmax = -100, 100  # Default range if no valid data
    
    # Plot velocity field
    im = ax.imshow(vel_plot, origin='lower', cmap='RdBu_r', 
                  vmin=vmin, vmax=vmax, 
                  aspect='equal' if equal_aspect else 'auto')
    
    plt.colorbar(im, ax=ax, label='Velocity (km/s)')
    
    ax.set_title(title)
    
    return ax


def plot_dispersion_field(dispersion_field, mask=None, ax=None, 
                         title='Velocity Dispersion', equal_aspect=False):
    """
    Plot the velocity dispersion field.
    
    Parameters
    ----------
    dispersion_field : ndarray
        2D array of velocity dispersion values
    mask : ndarray, optional
        Boolean mask for values to exclude
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, default='Velocity Dispersion'
        Title for the plot
    equal_aspect : bool, default=False
        Whether to keep aspect ratio equal
        
    Returns
    -------
    matplotlib.axes.Axes
        Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create masked array if mask provided or for NaN values
    if mask is not None:
        disp_plot = np.ma.array(dispersion_field, mask=mask)
    else:
        disp_plot = np.ma.array(dispersion_field, mask=~np.isfinite(dispersion_field))
    
    # Check if there are any valid values
    if np.all(disp_plot.mask):
        ax.text(0.5, 0.5, "No valid dispersion data", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return ax
    
    # Get colormap limits
    valid_values = disp_plot.compressed()
    if len(valid_values) > 0:
        vmin = max(0, np.nanpercentile(valid_values, 5))
        vmax = np.nanpercentile(valid_values, 95)
    else:
        vmin, vmax = 0, 100  # Default range if no valid data
    
    # Ensure valid range
    if vmin >= vmax:
        vmin = 0
        vmax = max(100, np.nanmax(disp_plot))
    
    # Plot dispersion field
    im = ax.imshow(disp_plot, origin='lower', cmap='viridis', 
                  vmin=vmin, vmax=vmax, 
                  aspect='equal' if equal_aspect else 'auto')
    
    plt.colorbar(im, ax=ax, label='Velocity Dispersion (km/s)')
    
    ax.set_title(title)
    
    return ax


def plot_binning_map(bin_map, snr_map=None, ax=None, title='Binning Map', 
                    equal_aspect=False, cmap='tab20'):
    """
    Plot binning map.
    
    Parameters
    ----------
    bin_map : ndarray
        2D array of bin indices
    snr_map : ndarray, optional
        2D array of SNR values
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, default='Binning Map'
        Title for the plot
    equal_aspect : bool, default=False
        Whether to keep aspect ratio equal
    cmap : str or matplotlib.colors.Colormap, default='tab20'
        Colormap to use
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the plot
    matplotlib.axes.Axes
        Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure
    
    # Create masked array for unbinned pixels
    masked_bin_map = np.ma.array(bin_map, mask=(bin_map < 0))
    
    # Check if any valid bins exist
    if np.all(masked_bin_map.mask):
        ax.text(0.5, 0.5, "No valid binning data", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig, ax
    
    # Get number of bins
    try:
        n_bins = int(np.max(bin_map)) + 1
    except:
        n_bins = 1  # Default if calculation fails
    
    # Plot bin map
    im = ax.imshow(masked_bin_map, origin='lower', cmap=cmap, 
                  aspect='equal' if equal_aspect else 'auto',
                  vmin=-0.5, vmax=min(n_bins, 20) - 0.5)
    
    # If SNR map provided, add it as contours
    if snr_map is not None and np.any(np.isfinite(snr_map)):
        try:
            # Smooth SNR map for better visualization
            smoothed_snr = gaussian_filter(np.nan_to_num(snr_map), sigma=1)
            
            # Create contour levels
            valid_snr = smoothed_snr[np.isfinite(smoothed_snr)]
            if len(valid_snr) > 0:
                snr_min = np.nanmin(valid_snr)
                snr_max = np.nanmax(valid_snr)
                if snr_min < snr_max:
                    snr_levels = np.linspace(snr_min, snr_max, 5)
                    
                    # Plot contours
                    ct = ax.contour(smoothed_snr, levels=snr_levels, 
                                   colors='white', alpha=0.5)
                    
                    # Add contour labels
                    ax.clabel(ct, inline=True, fontsize=8, fmt='%.1f')
        except Exception as e:
            warnings.warn(f"Error plotting SNR contours: {str(e)}")
    
    ax.set_title(title)
    
    return fig, ax


def plot_rotation_curve(rotation_curve, plot_model=True, vmax=None, pa=None, 
                       title='Rotation Curve', ax=None):
    """
    Plot rotation curve.
    
    Parameters
    ----------
    rotation_curve : ndarray
        Array with [radius, velocity] pairs
    plot_model : bool, default=True
        Whether to plot the model curve
    vmax : float, optional
        Maximum rotation velocity
    pa : float, optional
        Position angle in degrees
    title : str, default='Rotation Curve'
        Title for the plot
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the plot
    matplotlib.axes.Axes
        Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Extract radius and velocity
    try:
        radius = rotation_curve[:, 0]
        velocity = rotation_curve[:, 1]
        
        # Filter out NaN values
        valid_mask = np.isfinite(radius) & np.isfinite(velocity)
        if not np.any(valid_mask):
            ax.text(0.5, 0.5, "No valid rotation curve data", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig, ax
            
        radius = radius[valid_mask]
        velocity = velocity[valid_mask]
        
        # Plot data points
        ax.plot(radius, velocity, 'ko', label='Data')
        
        # Plot model curve if requested
        if plot_model and vmax is not None and np.isfinite(vmax):
            # Create a dense radius array for smooth curve
            if len(radius) > 0:
                r_model = np.linspace(0, np.max(radius) * 1.1, 100)
                
                # Arctan rotation curve model
                v_model = 2 * vmax / np.pi * np.arctan(r_model / 5)
                
                # Plot model
                ax.plot(r_model, v_model, 'r-', label='Model')
        
        # Add annotations
        if vmax is not None and np.isfinite(vmax):
            ax.axhline(y=vmax, color='b', linestyle='--', 
                      label=f'$V_{{max}}$ = {vmax:.1f} km/s')
    except Exception as e:
        warnings.warn(f"Error plotting rotation curve: {str(e)}")
        ax.text(0.5, 0.5, "Error plotting rotation curve", 
                ha='center', va='center', transform=ax.transAxes)
    
    # Add legend and labels
    ax.legend(loc='best')
    ax.set_xlabel('Radius (pixels)')
    ax.set_ylabel('Rotation Velocity (km/s)')
    
    # Add PA information if provided
    if pa is not None and np.isfinite(pa):
        ax.text(0.05, 0.95, f'PA = {pa:.1f}°', 
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top')
    
    ax.set_title(title)
    
    return fig, ax


def plot_rotation_model(velocity_field, mask=None, center_x=None, center_y=None, 
                       pa=None, model_field=None, ax=None, title='Rotation Model',
                       equal_aspect=False):
    """
    Plot rotation model with velocity field.
    
    Parameters
    ----------
    velocity_field : ndarray
        2D array of velocity values
    mask : ndarray, optional
        Boolean mask for values to exclude
    center_x : float, optional
        X-coordinate of rotation center
    center_y : float, optional
        Y-coordinate of rotation center
    pa : float, optional
        Position angle in degrees
    model_field : ndarray, optional
        2D array of model velocity values
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, default='Rotation Model'
        Title for the plot
    equal_aspect : bool, default=False
        Whether to keep aspect ratio equal
        
    Returns
    -------
    matplotlib.axes.Axes
        Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create masked array for NaN values
    if mask is not None:
        vel_plot = np.ma.array(velocity_field, mask=mask)
    else:
        vel_plot = np.ma.array(velocity_field, mask=~np.isfinite(velocity_field))
    
    # Check if there are any valid values
    if np.all(vel_plot.mask):
        ax.text(0.5, 0.5, "No valid velocity data", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return ax
    
    # Get symmetric color range
    valid_values = vel_plot.compressed()
    if len(valid_values) > 0:
        vabs = np.nanpercentile(np.abs(valid_values), 95)
        vmin, vmax = -vabs, vabs
    else:
        vmin, vmax = -100, 100  # Default range if no valid data
    
    # Plot velocity field
    im = ax.imshow(vel_plot, origin='lower', cmap='RdBu_r', 
                  vmin=vmin, vmax=vmax, 
                  aspect='equal' if equal_aspect else 'auto')
    
    # Add model contours if provided
    if model_field is not None and np.any(np.isfinite(model_field)):
        try:
            # Create contour levels
            levels = np.linspace(vmin, vmax, 11)
            
            # Plot contours
            ct = ax.contour(model_field, levels=levels, 
                           colors='white', alpha=0.7)
            
            # Add contour labels
            ax.clabel(ct, inline=True, fontsize=8, fmt='%.1f')
        except Exception as e:
            warnings.warn(f"Error plotting model contours: {str(e)}")
    
    # Add rotation center if provided
    if center_x is not None and center_y is not None and np.isfinite(center_x) and np.isfinite(center_y):
        ax.plot(center_x, center_y, 'wo', markersize=10, markeredgecolor='k')
    
    # Add rotation axis if PA provided
    if pa is not None and center_x is not None and center_y is not None:
        if np.isfinite(pa) and np.isfinite(center_x) and np.isfinite(center_y):
            try:
                # Convert PA to radians
                pa_rad = np.radians(pa)
                
                # Get image dimensions
                ny, nx = velocity_field.shape
                radius = min(nx, ny) // 2
                
                # Calculate endpoints of rotation axis line
                x1 = center_x + radius * np.cos(pa_rad)
                y1 = center_y + radius * np.sin(pa_rad)
                x2 = center_x - radius * np.cos(pa_rad)
                y2 = center_y - radius * np.sin(pa_rad)
                
                # Plot rotation axis
                ax.plot([x1, x2], [y1, y2], 'w--', lw=2)
                
                # Add PA annotation
                ax.text(0.05, 0.95, f'PA = {pa:.1f}°', 
                       transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', color='w',
                       bbox=dict(facecolor='k', alpha=0.5))
            except Exception as e:
                warnings.warn(f"Error plotting rotation axis: {str(e)}")
    
    plt.colorbar(im, ax=ax, label='Velocity (km/s)')
    
    ax.set_title(title)
    
    return ax


def plot_parameter_map(data, bin_map, ax=None, title=None, cmap='viridis', 
                      label=None, vmin=None, vmax=None, equal_aspect=True):
    """
    Plot parameter map with robust handling of different array dimensions
    
    Parameters
    ----------
    data : ndarray
        Parameter values (can be 1D or 2D)
    bin_map : ndarray
        2D bin map
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, optional
        Plot title
    cmap : str, default='viridis'
        Colormap to use
    label : str, optional
        Colorbar label
    vmin, vmax : float, optional
        Minimum and maximum values for colorbar
    equal_aspect : bool, default=True
        Whether to use equal aspect ratio
        
    Returns
    -------
    matplotlib.axes.Axes
        Axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))  # Square figure size
    
    try:
        # Create parameter map
        param_map = np.full_like(bin_map, np.nan, dtype=float)
        
        # Check if data is 1D or 2D
        if isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                # For each bin, fill map with corresponding value
                for i, value in enumerate(data):
                    if i < len(data) and np.isfinite(value):
                        param_map[bin_map == i] = value
            elif len(data.shape) == 2:
                # Check if shapes match
                if data.shape == bin_map.shape:
                    # Direct 2D array, just copy where bin_map is valid
                    valid_mask = (bin_map >= 0)
                    param_map[valid_mask] = data[valid_mask]
                else:
                    # Reshape if possible, otherwise just fill where we can
                    try:
                        data_reshaped = data.reshape(bin_map.shape)
                        valid_mask = (bin_map >= 0)
                        param_map[valid_mask] = data_reshaped[valid_mask]
                    except:
                        # Go back to bin-based filling
                        for i in range(np.max(bin_map) + 1):
                            mask = (bin_map == i)
                            if np.any(mask) and i < data.size:
                                param_map[mask] = data.flat[i]
            else:
                # Higher dimensional - try to flatten and use what we can
                flat_data = data.ravel()
                for i in range(min(np.max(bin_map) + 1, len(flat_data))):
                    mask = (bin_map == i)
                    if np.any(mask) and i < len(flat_data):
                        param_map[mask] = flat_data[i]
        
        # Plot parameter map
        valid_param = param_map[np.isfinite(param_map)]
        if len(valid_param) > 0:
            # Determine color scale
            if vmin is None:
                vmin = np.nanpercentile(valid_param, 5)
            if vmax is None:
                vmax = np.nanpercentile(valid_param, 95)
            
            # Ensure valid range
            if vmin >= vmax:
                vmin = np.nanmin(valid_param)
                vmax = np.nanmax(valid_param)
                # If still invalid, use default range
                if vmin >= vmax:
                    vmin, vmax = 0, 1
            
            # Create masked array for better visualization
            masked_param = np.ma.array(param_map, mask=~np.isfinite(param_map))
            
            # Plot the map
            im = ax.imshow(masked_param, origin='lower', cmap=cmap, 
                          vmin=vmin, vmax=vmax, aspect='equal' if equal_aspect else 'auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            if label:
                cbar.set_label(label)
        else:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
        
        if title:
            ax.set_title(title)
        
        # Set equal aspect ratio if requested
        if equal_aspect:
            ax.set_aspect('equal')
            
        # Add grid for reference
        ax.grid(True, alpha=0.3)
        
        return ax
    
    except Exception as e:
        logger.warning(f"Error plotting parameter map: {e}")
        ax.text(0.5, 0.5, f"Error plotting parameter: {str(e)}", 
               ha='center', va='center', transform=ax.transAxes)
        if title:
            ax.set_title(title)
        return ax


def safe_plot_array(data, bin_map, ax=None, title=None, cmap='viridis', label=None):
    """
    Plot array data safely handling different dimensions and shapes
    
    Parameters
    ----------
    data : ndarray or list
        Data to plot (1D, 2D, or other)
    bin_map : ndarray
        Bin map with bin numbers
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    title : str, optional
        Plot title
    cmap : str, default='viridis'
        Colormap name
    label : str, optional
        Colorbar label
        
    Returns
    -------
    matplotlib.axes.Axes
        Axis with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    try:
        # Convert to numpy array if not already
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Create parameter map
        param_map = np.full_like(bin_map, np.nan, dtype=float)
        
        # Different approaches based on data dimensions
        if data.ndim == 0:  # Scalar
            # Use scalar value for all valid bins
            param_map[bin_map >= 0] = float(data)
            
        elif data.ndim == 1:  # 1D array
            # Map each value to corresponding bin
            for i, val in enumerate(data):
                if i < len(data) and np.isfinite(val):
                    param_map[bin_map == i] = val
            
        elif data.ndim == 2 and data.shape == bin_map.shape:  # Matching 2D array
            # Direct copy to valid bins
            valid = (bin_map >= 0)
            param_map[valid] = data[valid]
            
        else:  # Other dimensions or shapes
            # Try to flatten and use what we can
            flat_data = data.flatten()
            for i in range(min(np.max(bin_map) + 1, len(flat_data))):
                mask = (bin_map == i)
                if np.any(mask):
                    param_map[mask] = flat_data[i]
        
        # Check if we have valid data
        valid_data = param_map[np.isfinite(param_map)]
        if len(valid_data) > 0:
            # Calculate color scale
            vmin = np.nanpercentile(valid_data, 5)
            vmax = np.nanpercentile(valid_data, 95)
            
            # Handle equal values
            if vmin >= vmax:
                vmin = np.nanmin(valid_data) - 0.1
                vmax = np.nanmax(valid_data) + 0.1
            
            # Plot with imshow
            im = ax.imshow(param_map, origin='lower', cmap=cmap, 
                         vmin=vmin, vmax=vmax, aspect='equal')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            if label:
                cbar.set_label(label)
        else:
            ax.text(0.5, 0.5, "No valid data to plot", 
                  ha='center', va='center', transform=ax.transAxes)
        
        # Add title
        if title:
            ax.set_title(title)
            
        # Add grid
        ax.grid(True, alpha=0.3)
            
    except Exception as e:
        logger.warning(f"Error in safe_plot_array: {e}")
        ax.text(0.5, 0.5, f"Error plotting: {str(e)}", 
              ha='center', va='center', transform=ax.transAxes)
        if title:
            ax.set_title(title)
    
    return ax


def plot_bin_map(bin_num, values, ax=None, cmap='viridis', title=None, 
                vmin=None, vmax=None, log_scale=False, colorbar_label=None):
    """
    Plot values for binned data, handling both 1D and 2D bin formats
    
    Parameters
    ----------
    bin_num : ndarray
        Bin map or bin indices
    values : ndarray
        Values for each bin
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    cmap : str, default='viridis'
        Colormap name
    title : str, optional
        Plot title
    vmin, vmax : float, optional
        Minimum and maximum color values
    log_scale : bool, default=False
        Whether to use log scale for values
    colorbar_label : str, optional
        Label for colorbar
        
    Returns
    -------
    matplotlib.axes.Axes
        Axis with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    try:
        # Convert inputs to numpy arrays if not already
        bin_num_np = np.asarray(bin_num)
        values_np = np.asarray(values)
        
        # Check for 1D vs 2D bin_num
        if bin_num_np.ndim == 1:
            # For 1D arrays, create a bar plot
            return plot_binned_values(bin_num_np, values_np, ax=ax, 
                                     title=title, cmap=cmap, 
                                     colorbar_label=colorbar_label,
                                     vmin=vmin, vmax=vmax, 
                                     log_scale=log_scale)
        else:
            # For 2D arrays, use safe_plot_array
            return safe_plot_array(values_np, bin_num_np, ax=ax, 
                                  title=title, cmap=cmap, 
                                  label=colorbar_label)
    
    except Exception as e:
        logger.warning(f"Error in plot_bin_map: {e}")
        ax.text(0.5, 0.5, f"Error plotting: {str(e)}", 
              ha='center', va='center', transform=ax.transAxes)
        if title:
            ax.set_title(title)
        return ax


def plot_binned_values(bin_indices, values, ax=None, title=None, cmap='viridis',
                      colorbar_label=None, vmin=None, vmax=None, log_scale=False):
    """
    Plot values for binned data using a bar plot approach that works well with any bin shape.
    
    Parameters
    ----------
    bin_indices : array-like
        Bin indices (can be 1D array or list of bin indices)
    values : array-like
        Values for each bin
    title : str, optional
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    cmap : str, default='viridis'
        Colormap name
    colorbar_label : str, optional
        Label for colorbar
    vmin, vmax : float, optional
        Minimum and maximum values for colorbar
    log_scale : bool, default=False
        Whether to use log scale
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        # Convert inputs to numpy arrays if not already
        values_array = np.asarray(values).flatten()  # Ensure 1D
        
        # Filter out invalid values
        valid_indices = np.isfinite(values_array)
        if np.sum(valid_indices) == 0:
            ax.text(0.5, 0.5, "No valid data to plot", 
                   ha='center', va='center', transform=ax.transAxes)
            if title:
                ax.set_title(title)
            return ax
        
        # Create plotting indices (just sequential numbers)
        x = np.arange(len(values_array))
        
        # Filter to valid data points
        x_valid = x[valid_indices]
        values_valid = values_array[valid_indices]
        
        # Handle log scale if requested
        if log_scale and np.any(values_valid > 0):
            # Find positive values
            positive_mask = values_valid > 0
            if np.any(positive_mask):
                # Get minimum positive value
                min_positive = np.min(values_valid[positive_mask])
                
                # Replace non-positive values with a small value
                values_valid[~positive_mask] = min_positive * 0.1
                
                # Apply log10
                values_valid = np.log10(values_valid)
                
                # Adjust colorbar label
                if colorbar_label and not colorbar_label.startswith('Log'):
                    colorbar_label = f"Log10({colorbar_label})"
        
        # Get color limits
        if len(values_valid) > 0:
            if vmin is None:
                vmin = np.min(values_valid)
            if vmax is None:
                vmax = np.max(values_valid)
            
            # Ensure valid range
            if vmin >= vmax:
                vmin = vmin * 0.9 if vmin != 0 else -1
                vmax = vmax * 1.1 if vmax != 0 else 1
                
            # Create colormap
            norm = Normalize(vmin=vmin, vmax=vmax)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            
            # Get colors for valid values
            colors = sm.to_rgba(values_valid)
            
            # Create bar plot
            ax.bar(x_valid, values_valid, color=colors, alpha=0.8)
            
            # Set labels
            ax.set_xlabel('Bin Index')
            ax.set_ylabel(colorbar_label if colorbar_label else 'Value')
            
            # Set reasonable number of x ticks
            max_ticks = min(20, len(x_valid))
            if len(x_valid) > max_ticks:
                step = len(x_valid) // max_ticks
                idx = np.arange(0, len(x_valid), step)
                ax.set_xticks(x_valid[idx])
                ax.set_xticklabels([str(int(i)) for i in x_valid[idx]])
            
            # Add colorbar
            plt.colorbar(sm, ax=ax, label=colorbar_label)
        else:
            ax.text(0.5, 0.5, "No valid data to plot", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Add title
        if title:
            ax.set_title(title)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return ax
    
    except Exception as e:
        logger.warning(f"Error plotting binned values: {e}")
        ax.text(0.5, 0.5, f"Error: {str(e)}", 
               ha='center', va='center', transform=ax.transAxes)
        if title:
            ax.set_title(title)
        return ax


def add_rotation_markers(ax, center_x, center_y, pa, radius=None, color='w'):
    """
    Add markers for rotation center and axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to add markers to
    center_x : float
        X-coordinate of rotation center
    center_y : float
        Y-coordinate of rotation center
    pa : float
        Position angle in degrees
    radius : float, optional
        Length of rotation axis
    color : str, default='w'
        Color of markers
        
    Returns
    -------
    matplotlib.axes.Axes
        Axis with markers
    """
    try:
        # Check parameters
        if not np.isfinite(center_x) or not np.isfinite(center_y) or not np.isfinite(pa):
            return ax
        
        # Add rotation center
        ax.plot(center_x, center_y, 'o', color=color, markersize=10, markeredgecolor='k')
        
        # Add rotation axis if radius provided
        if radius is not None:
            # Convert PA to radians
            pa_rad = np.radians(pa)
            
            # Calculate endpoints of rotation axis line
            x1 = center_x + radius * np.cos(pa_rad)
            y1 = center_y + radius * np.sin(pa_rad)
            x2 = center_x - radius * np.cos(pa_rad)
            y2 = center_y - radius * np.sin(pa_rad)
            
            # Plot rotation axis
            ax.plot([x1, x2], [y1, y2], '--', color=color, lw=2)
            
            # Add PA annotation
            ax.text(0.05, 0.95, f'PA = {pa:.1f}°', 
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', color=color,
                   bbox=dict(facecolor='k', alpha=0.5))
    except Exception as e:
        warnings.warn(f"Error adding rotation markers: {str(e)}")
    
    return ax


def plot_kinematics_summary(velocity_field, dispersion_field, bin_map=None, 
                          rotation_curve=None, params=None, equal_aspect=False):
    """
    Create a summary plot of kinematic analysis with robust error handling.
    
    Parameters
    ----------
    velocity_field : ndarray
        2D array of velocity values
    dispersion_field : ndarray
        2D array of velocity dispersion values
    bin_map : ndarray, optional
        2D array of bin indices
    rotation_curve : ndarray, optional
        Array with [radius, velocity] pairs
    params : dict, optional
        Dictionary of kinematic parameters
    equal_aspect : bool, default=False
        Whether to keep aspect ratio equal
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with the plot
    """
    # Create figure
    if rotation_curve is not None and np.any(np.isfinite(rotation_curve)):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax3 = None
    
    try:
        # Create mask for bad values
        mask = ~np.isfinite(velocity_field) | ~np.isfinite(dispersion_field)
        
        # If bin_map provided, add to mask
        if bin_map is not None:
            mask = mask | (bin_map < 0)
        
        # Plot velocity field with error handling
        try:
            # Check shape of velocity field
            if len(velocity_field.shape) == 2:
                # Regular 2D velocity field
                valid_vel = velocity_field[~mask]
                if len(valid_vel) > 0:
                    vmin = np.nanpercentile(valid_vel, 5)
                    vmax = np.nanpercentile(valid_vel, 95)
                    
                    # For velocity, use symmetric color scale
                    vabs = max(abs(vmin), abs(vmax))
                    if vmin < 0 and vmax > 0:
                        vmin, vmax = -vabs, vabs
                    
                    im1 = ax1.imshow(
                        np.ma.array(velocity_field, mask=mask),
                        origin='lower',
                        cmap='coolwarm',
                        vmin=vmin,
                        vmax=vmax,
                        aspect='equal' if equal_aspect else 'auto'
                    )
                    plt.colorbar(im1, ax=ax1, label='Velocity (km/s)')
                else:
                    ax1.text(0.5, 0.5, "No valid velocity data",
                           ha='center', va='center', transform=ax1.transAxes)
            elif velocity_field.shape[0] == 1:
                # Special case for 1D adapter - create a simple colored plot
                ax1.text(0.5, 0.5, "Binned velocity data (non-spatial)",
                       ha='center', va='center', transform=ax1.transAxes)
                
                # Create a simple bar plot of velocity values
                bin_indices = np.arange(velocity_field.shape[1])
                ax1.bar(bin_indices, velocity_field[0], color='blue', alpha=0.7)
                ax1.set_xlabel('Bin Index')
                ax1.set_ylabel('Velocity (km/s)')
            else:
                ax1.text(0.5, 0.5, "Incompatible velocity field shape",
                       ha='center', va='center', transform=ax1.transAxes)
        except Exception as e:
            logger.error(f"Error plotting velocity field: {e}")
            ax1.text(0.5, 0.5, "Error plotting velocity field",
                   ha='center', va='center', transform=ax1.transAxes)
        
        ax1.set_title('Velocity Field')
        
        # Plot dispersion field with error handling
        try:
            # Check shape of dispersion field
            if len(dispersion_field.shape) == 2:
                # Regular 2D dispersion field
                valid_disp = dispersion_field[~mask]
                if len(valid_disp) > 0:
                    vmin = np.nanpercentile(valid_disp, 5)
                    vmax = np.nanpercentile(valid_disp, 95)
                    
                    im2 = ax2.imshow(
                        np.ma.array(dispersion_field, mask=mask),
                        origin='lower',
                        cmap='viridis',
                        vmin=vmin,
                        vmax=vmax,
                        aspect='equal' if equal_aspect else 'auto'
                    )
                    plt.colorbar(im2, ax=ax2, label='Dispersion (km/s)')
                else:
                    ax2.text(0.5, 0.5, "No valid dispersion data",
                           ha='center', va='center', transform=ax2.transAxes)
            elif dispersion_field.shape[0] == 1:
                # Special case for 1D adapter - create a simple colored plot
                ax2.text(0.5, 0.5, "Binned dispersion data (non-spatial)",
                       ha='center', va='center', transform=ax2.transAxes)
                
                # Create a simple bar plot of dispersion values
                bin_indices = np.arange(dispersion_field.shape[1])
                ax2.bar(bin_indices, dispersion_field[0], color='green', alpha=0.7)
                ax2.set_xlabel('Bin Index')
                ax2.set_ylabel('Dispersion (km/s)')
            else:
                ax2.text(0.5, 0.5, "Incompatible dispersion field shape",
                       ha='center', va='center', transform=ax2.transAxes)
        except Exception as e:
            logger.error(f"Error plotting dispersion field: {e}")
            ax2.text(0.5, 0.5, "Error plotting dispersion field",
                   ha='center', va='center', transform=ax2.transAxes)
        
        ax2.set_title('Velocity Dispersion')
        
        # Add model contours and rotation parameters if available
        if params is not None:
            try:
                # Extract parameters
                center_x = params.get('center_x', None)
                center_y = params.get('center_y', None)
                pa = params.get('pa', None)
                
                # Check parameter validity
                if (center_x is not None and center_y is not None and pa is not None and
                    np.isfinite(center_x) and np.isfinite(center_y) and np.isfinite(pa)):
                    
                    # Add rotation center and axis to velocity plot
                    add_rotation_markers(
                        ax1, center_x, center_y, pa, 
                        radius=min(velocity_field.shape) // 3
                    )
            except Exception as e:
                logger.warning(f"Error adding rotation markers: {e}")
        
        # Plot rotation curve if provided in the third panel
        if rotation_curve is not None and ax3 is not None:
            try:
                plot_rotation_curve(
                    rotation_curve, 
                    plot_model=params is not None, 
                    vmax=params.get('vmax', None) if params else None,
                    pa=params.get('pa', None) if params else None,
                    title='Rotation Curve',
                    ax=ax3
                )
                
                # Add parameters if provided
                if params is not None:
                    # Format parameter values as text
                    param_text = []
                    if 'vmax' in params and np.isfinite(params['vmax']):
                        param_text.append(f"$V_{{max}}$ = {params['vmax']:.1f} km/s")
                    if 'pa' in params and np.isfinite(params['pa']):
                        param_text.append(f"PA = {params['pa']:.1f}°")
                    if 'vsys' in params and np.isfinite(params['vsys']):
                        param_text.append(f"$V_{{sys}}$ = {params['vsys']:.1f} km/s")
                    if 'center_x' in params and 'center_y' in params:
                        if np.isfinite(params['center_x']) and np.isfinite(params['center_y']):
                            param_text.append(f"Center = ({params['center_x']:.1f}, {params['center_y']:.1f})")
                    if 'sigma_mean' in params and np.isfinite(params['sigma_mean']):
                        param_text.append(f"$\\sigma_{{mean}}$ = {params['sigma_mean']:.1f} km/s")
                    
                    # Add text box
                    if param_text:
                        ax3.text(0.05, 0.05, '\n'.join(param_text), 
                               transform=ax3.transAxes, fontsize=10,
                               verticalalignment='bottom', horizontalalignment='left',
                               bbox=dict(facecolor='white', alpha=0.7))
            except Exception as e:
                logger.warning(f"Error plotting rotation curve: {e}")
                if ax3:
                    ax3.text(0.5, 0.5, "Error plotting rotation curve", 
                           ha='center', va='center', transform=ax3.transAxes)
    
    except Exception as e:
        logger.error(f"Error in plot_kinematics_summary: {e}")
        # Ensure we still return a figure even after error
        for ax in [ax1, ax2, ax3]:
            if ax:
                ax.text(0.5, 0.5, "Error generating plot", 
                       ha='center', va='center', transform=ax.transAxes)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig