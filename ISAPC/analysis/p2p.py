"""
Pixel-to-pixel analysis module for ISAPC
"""
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import spectral_indices
import galaxy_params
import visualization

from utils.io import save_results_to_npz

logger = logging.getLogger(__name__)


def run_p2p_analysis(args, cube):
    """
    Run pixel-to-pixel analysis
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cube : MUSECube
        MUSE data cube object
        
    Returns
    -------
    dict
        Analysis results
    """
    logger.info("Starting pixel-to-pixel analysis...")
    start_time = time.time()
    
    # Fit stellar continuum
    result = cube.fit_spectra(
        template_filename=args.template,
        ppxf_vel_init=args.vel_init,
        ppxf_vel_disp_init=args.sigma_init,
        ppxf_deg=args.poly_degree if hasattr(args, 'poly_degree') else 3,
        n_jobs=args.n_jobs
    )
    
    velocity_field, dispersion_field, bestfit_field, optimal_tmpls, poly_coeffs = result
    
    logger.info(f"Stellar component fitting completed in {time.time() - start_time:.1f} seconds")
    
    # Fit emission lines
    emission_result = None
    if not args.no_emission:
        start_time = time.time()
        emission_result = cube.fit_emission_lines(
            ppxf_vel_init=velocity_field,  # Use stellar velocity field as initial guess
            ppxf_sig_init=args.sigma_init,
            ppxf_deg=2,  # Simpler polynomial for emission lines
            n_jobs=args.n_jobs
        )
        logger.info(f"Emission line fitting completed in {time.time() - start_time:.1f} seconds")
    
    # Calculate spectral indices
    indices_result = None
    if not args.no_indices:
        start_time = time.time()
        indices_result = cube.calculate_spectral_indices(
            n_jobs=args.n_jobs
        )
        logger.info(f"Spectral indices calculation completed in {time.time() - start_time:.1f} seconds")
    
    # Calculate galaxy parameters
    start_time = time.time()
    gp = galaxy_params.GalaxyParameters(
        velocity_field=velocity_field,
        dispersion_field=dispersion_field,
        pixelsize=cube._pxl_size_x
    )
    
    rotation_result = gp.fit_rotation_curve()
    kinematics_result = gp.calculate_kinematics()
    
    logger.info(f"Galaxy parameters calculation completed in {time.time() - start_time:.1f} seconds")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    galaxy_name = Path(args.filename).stem
    
    # Create results dictionary
    p2p_results = {
        'velocity_field': velocity_field,
        'dispersion_field': dispersion_field,
        'bestfit_field': bestfit_field,
        'optimal_tmpls': optimal_tmpls,
        'kinematics': {**rotation_result, **kinematics_result}
    }
    
    if emission_result is not None:
        p2p_results['emission'] = emission_result
    
    if indices_result is not None:
        p2p_results['indices'] = indices_result
    
    # Save as NPZ file
    save_results_to_npz(
        output_file=output_dir / f"{galaxy_name}_P2P_results.npz",
        data_dict=p2p_results
    )
    
    # Create visualizations
    if not args.no_plots:
        create_p2p_plots(args, cube, p2p_results, galaxy_name)
    
    logger.info("Pixel-to-pixel analysis completed")
    return p2p_results


def create_p2p_plots(args, cube, p2p_results, galaxy_name):
    """
    Create plots for pixel-to-pixel analysis
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cube : MUSECube
        MUSE data cube object
    p2p_results : dict
        Analysis results
    galaxy_name : str
        Galaxy name for file naming
    """
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract results
    velocity_field = p2p_results['velocity_field']
    dispersion_field = p2p_results['dispersion_field'] 
    bestfit_field = p2p_results['bestfit_field']
    optimal_tmpls = p2p_results['optimal_tmpls']
    
    rotation_result = p2p_results['kinematics']
    
    # Create kinematics plot
    fig = visualization.plot_kinematics_summary(
        velocity_field=velocity_field,
        dispersion_field=dispersion_field,
        rotation_curve=rotation_result['rotation_curve'],
        params=rotation_result,
        equal_aspect=args.equal_aspect
    )
    
    fig.savefig(plots_dir / f"{galaxy_name}_P2P_kinematics.png", dpi=150)
    plt.close(fig)
    
    # Create sample pixel spectrum fits
    create_sample_fits(cube, p2p_results, plots_dir, galaxy_name)
    
    # Create spectral index plots
    if 'indices' in p2p_results:
        indices_result = p2p_results['indices']
        create_indices_plots(cube, indices_result, plots_dir, galaxy_name)
        
        # Use LineIndexCalculator to create detailed index plots for central pixel
        n_y, n_x = velocity_field.shape
        central_y, central_x = n_y // 2, n_x // 2
        
        # Check if central pixel has valid data
        if np.isnan(velocity_field[central_y, central_x]) or np.isnan(dispersion_field[central_y, central_x]):
            # Find a valid pixel
            valid_mask = ~np.isnan(velocity_field) & ~np.isnan(dispersion_field)
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)
                # Use the first valid pixel
                central_y, central_x = valid_indices[0][0], valid_indices[1][0]
                logger.info(f"Central pixel invalid, using alternative pixel at ({central_x}, {central_y})")
            else:
                logger.warning("No valid pixels found for spectral index plotting. Skipping.")
                return
        
        # Get data for central pixel
        central_idx = central_y * n_x + central_x
        observed_spectrum = cube._spectra[:, central_idx]
        model_spectrum = bestfit_field[:, central_y, central_x]
        
        # Get gas model if available
        gas_model = None
        if 'emission' in p2p_results and 'gas_bestfit_field' in p2p_results['emission']:
            gas_model = p2p_results['emission']['gas_bestfit_field'][:, central_y, central_x]
            # Verify it's a valid array
            if not np.any(np.isfinite(gas_model)):
                gas_model = None
                logger.warning("Gas model contains only non-finite values. Using None instead.")
        
        try:
            # Create LIC with error handling
            calculator = spectral_indices.LineIndexCalculator(
                wave=cube._lambda_gal,
                flux=observed_spectrum,
                fit_wave=cube._sps.lam_temp,
                fit_flux=optimal_tmpls[:, central_y, central_x],
                em_wave=cube._lambda_gal if gas_model is not None else None,
                em_flux_list=gas_model,
                velocity_correction=velocity_field[central_y, central_x],
                continuum_mode='auto'
            )
            
            # Plot spectral lines with indices
            fig, axes = calculator.plot_all_lines(
                mode="P2P", 
                number=0,
                save_path=str(plots_dir),
                show_index=True
            )
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating spectral line plots: {str(e)}")


def create_sample_fits(cube, p2p_results, plots_dir, galaxy_name):
    """
    Create spectrum fits plots for sample pixels
    
    Parameters
    ----------
    cube : MUSECube
        MUSE data cube object
    p2p_results : dict
        Analysis results 
    plots_dir : Path
        Path to save plots
    galaxy_name : str
        Galaxy name for file naming
    """
    # Extract needed data
    velocity_field = p2p_results['velocity_field']
    bestfit_field = p2p_results['bestfit_field']
    emission_result = p2p_results.get('emission', None)
    
    n_y, n_x = velocity_field.shape
    
    # Select sample positions
    center_y, center_x = n_y // 2, n_x // 2
    sample_positions = [
        (center_y, center_x),  # Center
        (center_y, center_x + n_x//4),  # Right
        (center_y + n_y//4, center_x),  # Top
        (center_y - n_y//4, center_x - n_x//4)  # Bottom-left
    ]
    
    # Filter sample positions to ensure they're valid
    valid_positions = []
    for y, x in sample_positions:
        if 0 <= y < n_y and 0 <= x < n_x and np.isfinite(velocity_field[y, x]):
            valid_positions.append((y, x))
    
    # If no valid positions, try to find at least one valid point
    if not valid_positions:
        valid_mask = np.isfinite(velocity_field)
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)
            valid_positions = [(valid_indices[0][0], valid_indices[1][0])]
        else:
            logger.warning("No valid pixels found for spectrum plots. Skipping.")
            return
    
    for i, (y, x) in enumerate(valid_positions):
        # Get spaxel index
        idx = y * n_x + x
        
        # Get observed spectrum
        observed = cube._spectra[:, idx]
        
        # Get model spectrum
        model = bestfit_field[:, y, x]
        
        # Get gas component
        gas_comp = None
        if emission_result is not None and 'gas_bestfit_field' in emission_result:
            gas_comp = emission_result['gas_bestfit_field'][:, y, x]
            # Verify it's a valid array
            if not np.any(np.isfinite(gas_comp)):
                gas_comp = None
        
        # Create stellar component by subtracting gas
        stellar_comp = model.copy()
        if gas_comp is not None:
            stellar_comp -= gas_comp
        
        try:
            # Create plot with error handling
            fig, axes = visualization.plot_spectrum_fit(
                wavelength=cube._lambda_gal,
                observed_flux=observed,
                model_flux=model,
                stellar_flux=stellar_comp,
                gas_flux=gas_comp,
                title=f"Pixel ({x}, {y})"
            )
            
            fig.savefig(plots_dir / f"{galaxy_name}_P2P_spectrum_{i}.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating spectrum plot for pixel ({x}, {y}): {str(e)}")


def create_indices_plots(cube, indices_result, plots_dir, galaxy_name):
    """
    Create spectral indices plots
    
    Parameters
    ----------
    cube : MUSECube
        MUSE data cube object
    indices_result : dict
        Spectral indices results
    plots_dir : Path
        Path to save plots
    galaxy_name : str
        Galaxy name for file naming
    """
    # Plot maps for each index
    for name, index_map in indices_result.items():
        try:
            fig, ax = plt.subplots(figsize=(8, 7))
            
            # Calculate valid range
            valid_values = index_map[~np.isnan(index_map)]
            if len(valid_values) > 0:
                vmin = np.percentile(valid_values, 5)
                vmax = np.percentile(valid_values, 95)
                
                # Check for valid range
                if vmin < vmax and np.isfinite(vmin) and np.isfinite(vmax):
                    # Plot index map
                    im = ax.imshow(index_map, origin='lower', cmap='viridis', 
                                 vmin=vmin, vmax=vmax, aspect='auto')
                    plt.colorbar(im, ax=ax)
                    ax.set_title(f"{name} Index")
                    
                    fig.savefig(plots_dir / f"{galaxy_name}_P2P_{name}_index.png", dpi=150)
                else:
                    logger.warning(f"Invalid value range for {name} index map: vmin={vmin}, vmax={vmax}")
            else:
                logger.warning(f"No valid values in {name} index map")
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating index map for {name}: {str(e)}")
            plt.close('all')  # Ensure all figures are closed