"""
Calculation utility functions for ISAPC
"""
from typing import Union, Optional

import numpy as np


def apply_velocity_shift(
        wvl: Union[list[float], np.ndarray],
        z: float,
) -> np.ndarray:
    """
    Apply velocity shift to the given wavelength.
    
    Parameters
    ----------
    wvl : array-like
        Original wavelength
    z : float
        Redshift in km/s
        
    Returns
    -------
    ndarray
        Shifted wavelength
    """
    return np.array(np.asarray(wvl) * (1 + z))


def make_bins(wavs):
    """ 
    Given a series of wavelength points, find the edges and widths
    of corresponding wavelength bins.
    
    Parameters
    ----------
    wavs : ndarray
        Wavelength array
    
    Returns
    -------
    tuple
        (edges, widths) arrays
    """
    edges = np.zeros(wavs.shape[0]+1)
    widths = np.zeros(wavs.shape[0])
    edges[0] = wavs[0] - (wavs[1] - wavs[0])/2
    widths[-1] = (wavs[-1] - wavs[-2])
    edges[-1] = wavs[-1] + (wavs[-1] - wavs[-2])/2
    edges[1:-1] = (wavs[1:] + wavs[:-1])/2
    widths[:-1] = edges[1:-1] - edges[:-2]

    return edges, widths


def spectres(new_wavs, spec_wavs, spec_fluxes, spec_errs=None, fill=None,
             verbose=True):
    """
    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.

    Parameters
    ----------
    new_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.

    spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.

    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        spec_wavs, last dimension must correspond to the shape of
        spec_wavs. Extra dimensions before this may be used to include
        multiple spectra.

    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.

    fill : float (optional)
        Where new_wavs extends outside the wavelength range in spec_wavs
        this value will be used as a filler in new_fluxes and new_errs.

    verbose : bool (optional)
        Setting verbose to False will suppress the default warning about
        new_wavs extending outside spec_wavs and "fill" being used.

    Returns
    -------
    new_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same
        length as new_wavs, other dimensions are the same as
        spec_fluxes.

    new_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in new_fluxes.
        Only returned if spec_errs was specified.
    """

    # Rename the input variables for clarity within the function.
    old_wavs = spec_wavs
    old_fluxes = spec_fluxes
    old_errs = spec_errs

    # Make arrays of edge positions and widths for the old and new bins
    old_edges, old_widths = make_bins(old_wavs)
    new_edges, new_widths = make_bins(new_wavs)

    # Generate output arrays to be populated
    new_fluxes = np.zeros(old_fluxes[..., 0].shape + new_wavs.shape)

    if old_errs is not None:
        if old_errs.shape != old_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape "
                             "as spec_fluxes.")
        else:
            new_errs = np.copy(new_fluxes)

    start = 0
    stop = 0

    # Calculate new flux and uncertainty values, looping over new bins
    for j in range(new_wavs.shape[0]):

        # Add filler values if new_wavs extends outside of spec_wavs
        if (new_edges[j] < old_edges[0]) or (new_edges[j+1] > old_edges[-1]):
            new_fluxes[..., j] = fill

            if spec_errs is not None:
                new_errs[..., j] = fill

            # if (j == 0 or j == new_wavs.shape[0]-1) and verbose:
            #     warnings.warn(
            #         "Spectres: new_wavs contains values outside the range "
            #         "in spec_wavs, new_fluxes and new_errs will be filled "
            #         "with the value set in the 'fill' keyword argument "
            #         "(by default 0).",
            #         category=RuntimeWarning,
            #     )
            continue

        # Find first old bin which is partially covered by the new bin
        while old_edges[start+1] <= new_edges[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin
        while old_edges[stop+1] < new_edges[j+1]:
            stop += 1

        # If new bin is fully inside an old bin start and stop are equal
        if stop == start:
            new_fluxes[..., j] = old_fluxes[..., start]
            if old_errs is not None:
                new_errs[..., j] = old_errs[..., start]

        # Otherwise multiply the first and last old bin widths by P_ij
        else:
            start_factor = ((old_edges[start+1] - new_edges[j])
                            / (old_edges[start+1] - old_edges[start]))

            end_factor = ((new_edges[j+1] - old_edges[stop])
                          / (old_edges[stop+1] - old_edges[stop]))

            old_widths[start] *= start_factor
            old_widths[stop] *= end_factor

            # Populate new_fluxes spectrum and uncertainty arrays
            f_widths = old_widths[start:stop+1]*old_fluxes[..., start:stop+1]
            new_fluxes[..., j] = np.sum(f_widths, axis=-1)
            new_fluxes[..., j] /= np.sum(old_widths[start:stop+1])

            if old_errs is not None:
                e_wid = old_widths[start:stop+1]*old_errs[..., start:stop+1]

                new_errs[..., j] = np.sqrt(np.sum(e_wid**2, axis=-1))
                new_errs[..., j] /= np.sum(old_widths[start:stop+1])

            # Put back the old bin widths to their initial values
            old_widths[start] /= start_factor
            old_widths[stop] /= end_factor

    # If errors were supplied return both new_fluxes and new_errs.
    if old_errs is not None:
        return new_fluxes, new_errs

    # Otherwise just return the new_fluxes spectrum array
    else:
        return new_fluxes


def resample_spectrum(
        new_wvl: Union[list[float], np.ndarray],
        src_wvl: Union[list[float], np.ndarray],
        src_flux: Union[list[float], np.ndarray],
        src_flux_err: Optional[Union[list[float], np.ndarray]] = None,
        fill: float = 0.0,
) -> Union[
    tuple[np.ndarray, np.ndarray], np.ndarray
]:
    """
    Resample spectrum to a new wavelength grid.
    
    Parameters
    ----------
    new_wvl : array-like
        New wavelength grid
    src_wvl : array-like
        Source wavelength grid
    src_flux : array-like
        Source flux values
    src_flux_err : array-like, optional
        Source flux error values
    fill : float, default=0.0
        Fill value for wavelengths outside source range
        
    Returns
    -------
    ndarray or tuple
        Resampled flux or (resampled_flux, resampled_error)
    """
    # Convert input to numpy array
    new_wvl = np.asarray(new_wvl)
    src_wvl = np.asarray(src_wvl)
    src_flux = np.asarray(src_flux)

    if src_wvl.shape != src_flux.shape:
        raise ValueError(
            f'src_wvl and src_flux must have the same shape. '
            f'Got {src_wvl.shape} and {src_flux.shape}'
        )

    if src_flux_err is not None:
        src_flux_err = np.asarray(src_flux_err)
        if src_flux_err.shape != src_flux.shape:
            raise ValueError(
                f'src_flux_err must have the same shape as src_flux. '
                f'Got {src_flux_err.shape} and {src_flux.shape}'
            )

    # Enhanced robustness check
    if len(new_wvl) == 0 or len(src_wvl) == 0:
        if src_flux_err is not None:
            return np.array([]), np.array([])
        return np.array([])
        
    # Handle NaN values
    valid_mask = ~np.isnan(src_flux)
    if not np.any(valid_mask):
        if src_flux_err is not None:
            return np.full(new_wvl.shape, fill), np.full(new_wvl.shape, fill)
        return np.full(new_wvl.shape, fill)
    
    # If there are NaNs, use valid data for interpolation
    if not np.all(valid_mask):
        src_wvl = src_wvl[valid_mask]
        src_flux = src_flux[valid_mask]
        if src_flux_err is not None:
            src_flux_err = src_flux_err[valid_mask]

    # Compute bin edges for source and new wavelength grids
    src_edges, src_widths = make_bins(src_wvl)
    new_edges, _ = make_bins(new_wvl)

    new_flux = np.full(new_wvl.shape, fill, dtype=src_flux.dtype)
    if src_flux_err is not None:
        new_flux_err = np.full(new_wvl.shape, fill, dtype=src_flux.dtype)
    else:
        new_flux_err = None

    for i in range(len(new_wvl)):
        if new_edges[i] < src_edges[0] or new_edges[i + 1] > src_edges[-1]:
            continue

        # Identify source bins overlapping with new bin
        start_idx = np.searchsorted(src_edges, new_edges[i], side='right') - 1
        stop_idx = np.searchsorted(src_edges, new_edges[i + 1], side='left') - 1

        # If new bin is fully contained within a single source bin
        if start_idx == stop_idx:
            new_flux[i] = src_flux[start_idx]
            if new_flux_err is not None:
                new_flux_err[i] = src_flux_err[start_idx]
            continue

        # For multiple overlapping bins, adjust first and last bin contributions
        partial_widths = src_widths[start_idx:stop_idx + 1].copy()
        # Fraction of first source bin that overlaps new bin
        start_factor = (
                (src_edges[start_idx + 1] - new_edges[i]) /
                (src_edges[start_idx + 1] - src_edges[start_idx])
        )
        partial_widths[0] *= start_factor
        # Fraction of last source bin that overlaps new bin
        end_factor = (
                (new_edges[i + 1] - src_edges[stop_idx]) /
                (src_edges[stop_idx + 1] - src_edges[stop_idx])
        )
        partial_widths[-1] *= end_factor

        # Calculate weighted flux for new bin
        flux_slice = src_flux[start_idx:stop_idx + 1]
        total_width = np.sum(partial_widths)
        new_flux[i] = np.sum(flux_slice * partial_widths) / total_width

        # Propagate uncertainties if provided
        if new_flux_err is not None:
            err_slice = src_flux_err[start_idx:stop_idx + 1]
            weighted_err_sq = np.sum((err_slice * partial_widths) ** 2)
            new_flux_err[i] = np.sqrt(weighted_err_sq) / total_width

    if new_flux_err is not None:
        return new_flux, new_flux_err
    return new_flux