"""
Pixel-to-pixel analysis module for ISAPC
Version 5.0.0
"""
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors
import spectral_indices
import galaxy_params
import visualization
from stellar_population import WeightParser
from utils.io import save_results_to_npz, save_standardized_results

logger = logging.getLogger(__name__)

# Speed of light in km/s
C_KMS = 299792.458


def calculate_distance_to_center(x, y, pxl_size_x, pxl_size_y=None):
    """
    Calculate physical distance to center
    
    Parameters
    ----------
    x : numpy.ndarray
        x coordinates in pixels
    y : numpy.ndarray
        y coordinates in pixels
    pxl_size_x : float
        x pixel size in arcsec
    pxl_size_y : float, optional
        y pixel size in arcsec, default same as x
        
    Returns
    -------
    numpy.ndarray
        Distance to center in arcsec
    """
    if pxl_size_y is None:
        pxl_size_y = pxl_size_x
        
    # Convert to physical units (arcsec)
    phys_x = x * pxl_size_x
    phys_y = y * pxl_size_y
    
    # Calculate Euclidean distance
    distance = np.sqrt(phys_x**2 + phys_y**2)
    
    return distance


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
        Analysis results with key physical parameters
    """
    logger.info("Starting pixel-to-pixel analysis...")
    start_time = time.time()
    
    # Disable warnings for spectral indices
    spectral_indices.set_warnings(False)
    
    # Extract galaxy name from filename
    galaxy_name = Path(args.filename).stem
    
    # Create standardized output directories
    output_dir = Path(args.output_dir)
    galaxy_dir = output_dir / galaxy_name
    data_dir = galaxy_dir / 'Data'
    plots_dir = galaxy_dir / 'Plots' / 'P2P'
    
    galaxy_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Fit stellar continuum
    result = cube.fit_spectra(
        template_filename=args.template,
        ppxf_vel_init=args.vel_init,
        ppxf_vel_disp_init=args.sigma_init,
        ppxf_deg=args.poly_degree if hasattr(args, 'poly_degree') else 3,
        n_jobs=args.n_jobs
    )
    
    stellar_velocity_field, stellar_dispersion_field, bestfit_field, optimal_tmpls, poly_coeffs = result
    
    logger.info(f"Stellar component fitting completed in {time.time() - start_time:.1f} seconds")
    
    # Fit emission lines
    emission_result = None
    if not args.no_emission:
        start_time = time.time()
        emission_result = cube.fit_emission_lines(
            template_filename=args.template,
            ppxf_vel_init=stellar_velocity_field,  # Use stellar velocity field as initial guess
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
    
    # Prepare emission line data (if available)
    gas_velocity_field = None
    gas_dispersion_field = None
    
    if emission_result is not None:
        try:
            # 1. Try to directly get from emission_result dictionary
            if 'velocity_field' in emission_result and emission_result['velocity_field'] is not None:
                gas_velocity_field = emission_result['velocity_field']
                logger.info("Extracted gas velocity field from emission_result")
                
            if 'dispersion_field' in emission_result and emission_result['dispersion_field'] is not None:
                gas_dispersion_field = emission_result['dispersion_field']
                logger.info("Extracted gas dispersion field from emission_result")
            
            # 2. Try to extract from emission_vel and emission_sig in emission_result
            if gas_velocity_field is None and 'emission_vel' in emission_result and emission_result['emission_vel']:
                # Get first available emission line
                for line_name, vel_map in emission_result['emission_vel'].items():
                    if not np.all(np.isnan(vel_map)):
                        gas_velocity_field = vel_map
                        logger.info(f"Using velocity field from emission line: {line_name}")
                        break
                
                # Get corresponding dispersion field
                if gas_velocity_field is not None and 'emission_sig' in emission_result and emission_result['emission_sig']:
                    for line_name, disp_map in emission_result['emission_sig'].items():
                        if not np.all(np.isnan(disp_map)):
                            gas_dispersion_field = disp_map
                            logger.info(f"Using dispersion field from emission line: {line_name}")
                            break
            
            # 3. Try to get from cube object
            if gas_velocity_field is None and hasattr(cube, '_emission_vel') and cube._emission_vel:
                # Get first available emission line
                for line_name, vel_map in cube._emission_vel.items():
                    if not np.all(np.isnan(vel_map)):
                        gas_velocity_field = vel_map
                        logger.info(f"Using velocity field from cube's emission line: {line_name}")
                        break
                
                # Get corresponding dispersion field
                if gas_velocity_field is not None and hasattr(cube, '_emission_sig') and cube._emission_sig:
                    for line_name, disp_map in cube._emission_sig.items():
                        if not np.all(np.isnan(disp_map)):
                            gas_dispersion_field = disp_map
                            logger.info(f"Using dispersion field from cube's emission line: {line_name}")
                            break
            
            # 4. Try to extract from cube._ppxf_gas_results if available
            if gas_velocity_field is None and hasattr(cube, '_ppxf_gas_results') and cube._ppxf_gas_results:
                # Initialize arrays as NaN
                gas_velocity_field = np.full((cube._n_y, cube._n_x), np.nan)
                gas_dispersion_field = np.full((cube._n_y, cube._n_x), np.nan)
                
                # Fill with values from each pixel's fit
                for row, col, result in cube._ppxf_gas_results:
                    if 'gas_sol' in result and result['gas_sol'] is not None:
                        gas_sol = result['gas_sol']
                        try:
                            # Try to access gas_sol as array
                            if isinstance(gas_sol, (list, np.ndarray)) and len(gas_sol) >= 2:
                                gas_velocity_field[row, col] = gas_sol[0]
                                gas_dispersion_field[row, col] = gas_sol[1]
                            elif gas_sol is not None:
                                # If gas_sol is scalar, assume it's the velocity
                                gas_velocity_field[row, col] = float(gas_sol)
                                # Use default value for dispersion
                                gas_dispersion_field[row, col] = args.sigma_init
                        except (TypeError, IndexError) as e:
                            logger.debug(f"Error extracting gas kinematics at ({row},{col}): {e}")
                
                logger.info("Extracted gas velocity and dispersion fields from ppxf_gas_results")
            
            # Validate extracted gas kinematic fields
            if gas_velocity_field is not None:
                valid_pixels = np.count_nonzero(~np.isnan(gas_velocity_field))
                if valid_pixels < 10:  # Assume fewer than 10 valid pixels is not useful for analysis
                    logger.warning(f"Too few valid pixels in gas velocity field ({valid_pixels}), using stellar field instead")
                    gas_velocity_field = None
                    gas_dispersion_field = None
                else:
                    logger.info(f"Found {valid_pixels} valid pixels in gas velocity field")
            
        except Exception as e:
            logger.error(f"Failed to extract gas velocity and dispersion fields: {e}")
            gas_velocity_field = None
            gas_dispersion_field = None
    
    # Decide which velocity and dispersion field to use for kinematic analysis
    # Prioritize emission line data (gas kinematics)
    if gas_velocity_field is not None and gas_dispersion_field is not None:
        # Check gas velocity field quality/coverage
        # Calculate ratio of valid pixels
        valid_gas_pixels = np.sum(~np.isnan(gas_velocity_field))
        valid_stellar_pixels = np.sum(~np.isnan(stellar_velocity_field))
        total_pixels = gas_velocity_field.size
        
        gas_coverage = valid_gas_pixels / total_pixels
        stellar_coverage = valid_stellar_pixels / total_pixels
        
        # If gas coverage is reasonable (at least 30% or more than 80% of stellar coverage)
        if gas_coverage > 0.3 or gas_coverage > 0.8 * stellar_coverage:
            logger.info(f"Using emission line velocity field for kinematics (coverage: {gas_coverage:.2f})")
            velocity_field = gas_velocity_field
            dispersion_field = gas_dispersion_field
            using_emission = True
        else:
            logger.info(f"Insufficient emission line coverage ({gas_coverage:.2f}), using stellar velocity field")
            velocity_field = stellar_velocity_field
            dispersion_field = stellar_dispersion_field
            using_emission = False
    else:
        logger.info("No emission line data available, using stellar velocity field")
        velocity_field = stellar_velocity_field
        dispersion_field = stellar_dispersion_field
        using_emission = False
    
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
    
    # Calculate distance to center for each pixel
    n_y, n_x = stellar_velocity_field.shape
    y_indices, x_indices = np.indices((n_y, n_x))
    
    # Use image center as default center
    center_y, center_x = n_y // 2, n_x // 2
    
    # Calculate coordinates relative to center
    rel_x = x_indices - center_x
    rel_y = y_indices - center_y
    
    # Calculate distance (arcsec)
    distance_field = calculate_distance_to_center(rel_x, rel_y, cube._pxl_size_x, cube._pxl_size_y)
    
    # Extract stellar population parameters using WeightParser
    stellar_pop_params = None
    if hasattr(cube, '_template_weights') and cube._template_weights is not None:
        try:
            logger.info("Extracting stellar population parameters...")
            start_time = time.time()
            
            # Initialize weight parser
            weight_parser = WeightParser(args.template)
            
            # Prepare arrays for physical parameters
            n_y, n_x = stellar_velocity_field.shape
            stellar_pop_params = {
                'log_age': np.full((n_y, n_x), np.nan),
                'age': np.full((n_y, n_x), np.nan),
                'metallicity': np.full((n_y, n_x), np.nan)
            }
            
            # Process weights based on shape
            weights = cube._template_weights
            
            if len(weights.shape) == 3:  # [n_templates, n_y, n_x]
                # Calculate for each valid pixel
                valid_mask = ~np.isnan(stellar_velocity_field)
                valid_indices = np.where(valid_mask)
                
                for i in range(len(valid_indices[0])):
                    y, x = valid_indices[0][i], valid_indices[1][i]
                    
                    try:
                        pixel_weights = weights[:, y, x]
                        if np.sum(pixel_weights) > 0:
                            params = weight_parser.get_physical_params(pixel_weights)
                            for param_name, value in params.items():
                                stellar_pop_params[param_name][y, x] = value
                    except Exception as e:
                        logger.debug(f"Error calculating stellar params for pixel ({x}, {y}): {e}")
            
            logger.info(f"Stellar population parameters extracted in {time.time() - start_time:.1f} seconds")
        except Exception as e:
            logger.error(f"Failed to extract stellar population parameters: {e}")
    elif emission_result is not None and 'weights' in emission_result and emission_result['weights'] is not None:
        try:
            logger.info("Extracting stellar population parameters from emission_result...")
            start_time = time.time()
            
            # Initialize weight parser
            weight_parser = WeightParser(args.template)
            
            # Prepare arrays for physical parameters
            n_y, n_x = stellar_velocity_field.shape
            stellar_pop_params = {
                'log_age': np.full((n_y, n_x), np.nan),
                'age': np.full((n_y, n_x), np.nan),
                'metallicity': np.full((n_y, n_x), np.nan)
            }
            
            # Use weights from emission_result
            weights = emission_result['weights']
            
            if len(weights.shape) == 2:  # [n_spectra, n_templates]
                # Calculate for each valid pixel
                valid_mask = ~np.isnan(stellar_velocity_field)
                valid_indices = np.where(valid_mask)
                
                for i in range(len(valid_indices[0])):
                    y, x = valid_indices[0][i], valid_indices[1][i]
                    idx = y * n_x + x
                    
                    try:
                        if idx < len(weights):
                            pixel_weights = weights[idx]
                            if np.sum(pixel_weights) > 0:
                                params = weight_parser.get_physical_params(pixel_weights)
                                for param_name, value in params.items():
                                    stellar_pop_params[param_name][y, x] = value
                    except Exception as e:
                        logger.debug(f"Error calculating stellar params for pixel ({x}, {y}): {e}")
            
            logger.info(f"Stellar population parameters extracted in {time.time() - start_time:.1f} seconds")
        except Exception as e:
            logger.error(f"Failed to extract stellar population parameters from emission_result: {e}")
    else:
        logger.warning("No weights found for stellar population analysis")
    
    # Create standardized results dictionary
    p2p_results = {
        'analysis_type': 'P2P',
        'stellar_kinematics': {
            'velocity_field': stellar_velocity_field,
            'dispersion_field': stellar_dispersion_field
        },
        'global_kinematics': {
            **rotation_result, 
            **kinematics_result,
            'based_on_emission': using_emission
        },
        'distance': {
            'field': distance_field,
            'pixelsize_x': cube._pxl_size_x,
            'pixelsize_y': cube._pxl_size_y
        }
    }
    
    # Add stellar population parameters if available
    if stellar_pop_params is not None:
        p2p_results['stellar_population'] = stellar_pop_params
    
    # Add emission line information
    if emission_result is not None:
        # Extract emission line parameters
        emission_params = {}
        
        # 1. Save gas kinematic fields
        if gas_velocity_field is not None:
            emission_params['velocity_field'] = gas_velocity_field
        if gas_dispersion_field is not None:
            emission_params['dispersion_field'] = gas_dispersion_field
        
        # 2. Extract emission line flux from emission_result
        if 'emission_flux' in emission_result and emission_result['emission_flux']:
            # emission_flux is a dictionary of different emission lines
            for line_name, flux_map in emission_result['emission_flux'].items():
                if not np.all(np.isnan(flux_map)):
                    emission_params[f'flux_{line_name}'] = flux_map
        
        # 3. If emission_result doesn't have emission_flux, try other sources
        if 'flux_' not in ''.join(emission_params.keys()):
            # Try to get directly from emission_result['flux']
            if 'flux' in emission_result and emission_result['flux'] is not None:
                emission_params['flux'] = emission_result['flux']
            
            # Try to get from cube._emission_flux
            if hasattr(cube, '_emission_flux') and cube._emission_flux:
                for line_name, flux_map in cube._emission_flux.items():
                    if not np.all(np.isnan(flux_map)):
                        emission_params[f'flux_{line_name}'] = flux_map
        
        # 4. Calculate line ratios
        try:
            line_ratios = {}
            
            # Check for Hbeta and [OIII]5007_d for calculation
            hb_key = None
            oiii_key = None
            
            # Find keys for Hbeta and OIII
            for key in emission_params.keys():
                if 'flux_Hbeta' in key:
                    hb_key = key
                elif 'flux_[OIII]5007' in key or 'flux_OIII_5007' in key:
                    oiii_key = key
            
            # If both keys found, calculate ratio
            if hb_key is not None and oiii_key is not None:
                hb_flux = emission_params[hb_key]
                oiii_flux = emission_params[oiii_key]
                
                # Calculate ratio, ensuring division by zero is handled
                valid_mask = ~np.isnan(hb_flux) & ~np.isnan(oiii_flux) & (hb_flux > 0)
                
                if np.any(valid_mask):
                    oiii_hb = np.full_like(hb_flux, np.nan)
                    oiii_hb[valid_mask] = oiii_flux[valid_mask] / hb_flux[valid_mask]
                    line_ratios['OIII_Hb'] = oiii_hb
                    logger.info("Calculated OIII/Hb line ratio")
            
            if line_ratios:
                emission_params['line_ratios'] = line_ratios
                
        except Exception as e:
            logger.warning(f"Could not calculate line ratios: {e}")
        
        # 5. Save gas best-fit (for plotting)
        if 'gas_bestfit_field' in emission_result and emission_result['gas_bestfit_field'] is not None:
            emission_params['gas_bestfit'] = emission_result['gas_bestfit_field']
        elif 'gas_bestfit' in emission_result and emission_result['gas_bestfit'] is not None:
            emission_params['gas_bestfit'] = emission_result['gas_bestfit']
        elif hasattr(cube, '_gas_bestfit_field') and cube._gas_bestfit_field is not None:
            emission_params['gas_bestfit'] = cube._gas_bestfit_field
        
        # 6. Save NEL_cal_tmp (if available)
        if 'NEL_cal_tmp' in emission_result and emission_result['NEL_cal_tmp'] is not None:
            emission_params['NEL_cal_tmp'] = emission_result['NEL_cal_tmp']
        
        # Only add emission key if we have valid data
        if emission_params:
            p2p_results['emission'] = emission_params
        else:
            logger.warning("No valid emission line data found despite successful fitting")
    
    # Add spectral indices
    if indices_result is not None:
        p2p_results['indices'] = indices_result
    
    # Save results
    save_standardized_results(galaxy_name, 'P2P', p2p_results, output_dir)
    
    # Create visualizations
    if not args.no_plots:
        create_p2p_plots(args, cube, p2p_results, galaxy_name, bestfit_field, optimal_tmpls, 
                         emission_result, using_emission)
        # Create radial profile plots
        create_radial_profile_plots(p2p_results, plots_dir=plots_dir, galaxy_name=galaxy_name, analysis_type="P2P")
    
    logger.info("Pixel-to-pixel analysis completed")
    return p2p_results


def create_p2p_plots(args, cube, p2p_results, galaxy_name, bestfit_field, optimal_tmpls, 
                      emission_result, using_emission):
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
    bestfit_field : ndarray
        Best-fit spectra (used for plotting only)
    optimal_tmpls : ndarray
        Optimal templates (used for plotting only)
    emission_result : dict
        Full emission line results (used for plotting only)
    using_emission : bool
        Whether emission lines were used for kinematics
    """
    # Set up plots directory
    output_dir = Path(args.output_dir)
    galaxy_dir = output_dir / galaxy_name
    plots_dir = galaxy_dir / 'Plots' / 'P2P'
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract results
    rotation_result = p2p_results['global_kinematics']
    
    # Determine which velocity field to use for primary display based on global kinematics
    if using_emission and 'emission' in p2p_results and 'velocity_field' in p2p_results['emission']:
        velocity_field = p2p_results['emission']['velocity_field']
        dispersion_field = p2p_results['emission']['dispersion_field'] if 'dispersion_field' in p2p_results['emission'] else None
        kinematics_type = "gas"  # For filename label
    else:
        velocity_field = p2p_results['stellar_kinematics']['velocity_field']
        dispersion_field = p2p_results['stellar_kinematics']['dispersion_field']
        kinematics_type = "stellar"  # For filename label
        using_emission = False  # Ensure flag is correct
    
    # If dispersion field not found, create NaN array with velocity field shape
    if dispersion_field is None:
        dispersion_field = np.full_like(velocity_field, np.nan)
        logger.warning("Dispersion field not found, using NaN array")
    
    # Create kinematics plot
    try:
        fig = visualization.plot_kinematics_summary(
            velocity_field=velocity_field,
            dispersion_field=dispersion_field,
            rotation_curve=rotation_result['rotation_curve'],
            params=rotation_result,
            equal_aspect=args.equal_aspect
        )
        
        # Use filename to distinguish gas/stellar kinematics
        fig.savefig(plots_dir / f"{galaxy_name}_P2P_{kinematics_type}_kinematics.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating kinematics plot: {str(e)}")
        plt.close('all')
    
    # Create stellar population parameter plots
    if 'stellar_population' in p2p_results:
        create_stellar_pop_plots(p2p_results['stellar_population'], plots_dir, galaxy_name)
    
    # If using stellar kinematics but emission line data is available, create gas kinematics plot
    if not using_emission and 'emission' in p2p_results and 'velocity_field' in p2p_results['emission']:
        gas_vel = p2p_results['emission']['velocity_field']
        gas_disp = p2p_results['emission']['dispersion_field'] if 'dispersion_field' in p2p_results['emission'] else np.full_like(gas_vel, np.nan)
        
        # Create gas kinematics plot (without rotation curve fit)
        try:
            # Check if plot_gas_kinematics function exists
            if hasattr(visualization, 'plot_gas_kinematics'):
                fig = visualization.plot_gas_kinematics(
                    velocity_field=gas_vel,
                    dispersion_field=gas_disp,
                    equal_aspect=args.equal_aspect
                )
            else:
                # If function doesn't exist, create basic plot
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Calculate valid data range
                valid_vel = gas_vel[~np.isnan(gas_vel)]
                valid_disp = gas_disp[~np.isnan(gas_disp)]
                
                if len(valid_vel) > 0:
                    vmin_vel = np.percentile(valid_vel, 5)
                    vmax_vel = np.percentile(valid_vel, 95)
                    im0 = axes[0].imshow(gas_vel, origin='lower', cmap='RdBu_r', 
                                      vmin=vmin_vel, vmax=vmax_vel, 
                                      aspect='auto' if not args.equal_aspect else 1)
                    plt.colorbar(im0, ax=axes[0], label='Velocity [km/s]')
                    axes[0].set_title('Gas Velocity Field')
                
                if len(valid_disp) > 0:
                    vmin_disp = np.percentile(valid_disp, 5)
                    vmax_disp = np.percentile(valid_disp, 95)
                    im1 = axes[1].imshow(gas_disp, origin='lower', cmap='viridis', 
                                      vmin=vmin_disp, vmax=vmax_disp, 
                                      aspect='auto' if not args.equal_aspect else 1)
                    plt.colorbar(im1, ax=axes[1], label='Dispersion [km/s]')
                    axes[1].set_title('Gas Velocity Dispersion')
                
                fig.suptitle('Gas Kinematics')
                plt.tight_layout()
            
            fig.savefig(plots_dir / f"{galaxy_name}_P2P_gas_kinematics.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating gas kinematics plot: {str(e)}")
            plt.close('all')
    
    # Create sample pixel spectrum fits
    create_sample_fits(cube, velocity_field, bestfit_field, emission_result, plots_dir, galaxy_name)
    
    # Create emission line maps if available
    if 'emission' in p2p_results:
        create_emission_maps(p2p_results['emission'], plots_dir, galaxy_name)
    
    # Create spectral index plots
    if 'indices' in p2p_results:
        create_indices_plots(cube, p2p_results['indices'], plots_dir, galaxy_name)
        
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
        
        try:
            # cube.plot_spectral_indices(mode="P2P", 
            #     spaxel_position=[central_x,central_y],
            #     save_path=str(plots_dir))
            # Get spectral data
            observed_spectrum = cube._spectra[:, central_idx]
            model_spectrum = bestfit_field[:, central_y, central_x]
            
            # Get gas model if available
            gas_model = None
            # if emission_result is not None:
                # Try to get gas model from emission_result
                # if 'gas_bestfit' in emission_result:
                #     gas_bestfit = emission_result['gas_bestfit']
                #     if gas_bestfit is not None:
                #         # Check shape of gas_bestfit and extract appropriately
                #         if len(gas_bestfit.shape) == 3:  # [n_wave, n_y, n_x]
                #             gas_model = gas_bestfit[:, central_y, central_x]
                #         elif len(gas_bestfit.shape) == 2:  # [n_wave, n_spectra]
                #             gas_model = gas_bestfit[:, central_idx]
                
                # # If not in emission_result, try cube
                # if gas_model is None and hasattr(cube, '_gas_bestfit_field') and cube._gas_bestfit_field is not None:
            gas_model = cube._gas_bestfit_field[:, central_y, central_x]
                    
                # Verify it's a valid array
                # if gas_model is not None and not np.any(np.isfinite(gas_model)):
                #     gas_model = None
                #     logger.warning("Gas model contains only non-finite values. Using None instead.")
            # Create LIC with error handling
            calculator = spectral_indices.LineIndexCalculator(
                wave=cube._lambda_gal,
                flux=observed_spectrum,
                fit_wave=cube._sps.lam_temp,
                fit_flux=optimal_tmpls[:, central_y, central_x],
                em_wave=cube._lambda_gal if gas_model is not None else None,
                em_flux_list=gas_model,
                velocity_correction=velocity_field[central_y, central_x],
                continuum_mode='auto',
                show_warnings=False
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
            plt.close('all')


def create_stellar_pop_plots(stellar_pop_params, plots_dir, galaxy_name):
    """
    Create plots for stellar population parameters
    
    Parameters
    ----------
    stellar_pop_params : dict
        Dictionary containing stellar population parameters
    plots_dir : Path
        Path to save plots
    galaxy_name : str
        Galaxy name for file naming
    """
    # Create plots for each parameter
    param_info = {
        'log_age': {
            'title': 'Log Age [yr]',
            'cmap': 'plasma',
            'vmin_percentile': 5,
            'vmax_percentile': 95
        },
        'age': {
            'title': 'Age [Gyr]',
            'cmap': 'plasma',
            'vmin_percentile': 5,
            'vmax_percentile': 95,
            'scale_factor': 1e-9  # Convert to Gyr
        },
        'metallicity': {
            'title': 'Metallicity [Z/H]',
            'cmap': 'viridis',
            'vmin_percentile': 5,
            'vmax_percentile': 95
        }
    }
    
    for param_name, info in param_info.items():
        if param_name in stellar_pop_params:
            try:
                param_map = stellar_pop_params[param_name]
                
                # Apply scale factor if defined
                if 'scale_factor' in info:
                    param_map = param_map * info['scale_factor']
                
                # Check data validity
                valid_values = param_map[~np.isnan(param_map)]
                if len(valid_values) > 0:
                    fig, ax = plt.subplots(figsize=(8, 7))
                    
                    # Calculate display range
                    vmin = np.percentile(valid_values, info['vmin_percentile'])
                    vmax = np.percentile(valid_values, info['vmax_percentile'])
                    
                    im = ax.imshow(param_map, origin='lower', cmap=info['cmap'], 
                                 vmin=vmin, vmax=vmax, aspect='auto')
                    plt.colorbar(im, ax=ax, label=info['title'])
                    ax.set_title(f"Stellar {info['title']}")
                    
                    fig.savefig(plots_dir / f"{galaxy_name}_P2P_stellar_{param_name}.png", dpi=150)
                    plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating {param_name} map: {str(e)}")
                plt.close('all')


def create_emission_maps(emission_params, plots_dir, galaxy_name):
    """
    Create emission line flux and ratio maps
    
    Parameters
    ----------
    emission_params : dict
        Emission line parameters
    plots_dir : Path
        Path to save plots
    galaxy_name : str
        Galaxy name for file naming
    """
    # Find all emission line flux maps
    flux_maps = {}
    
    # Collect all flux maps
    for key, value in emission_params.items():
        if key.startswith('flux_') and isinstance(value, np.ndarray):
            line_name = key[5:]  # Remove 'flux_' prefix
            flux_maps[line_name] = value
    
    # If no 'flux_' prefixed keys found, check for 'flux'
    if not flux_maps and 'flux' in emission_params:
        # If flux is a dictionary, it might contain line fluxes
        if isinstance(emission_params['flux'], dict):
            for line_name, flux in emission_params['flux'].items():
                flux_maps[line_name] = flux
        # If flux is an array, assume it's a single emission line
        elif isinstance(emission_params['flux'], np.ndarray):
            flux_maps['Combined'] = emission_params['flux']
    
    # Create maps for each emission line
    for line_name, flux_map in flux_maps.items():
        try:
            # Check data validity
            valid_values = flux_map[~np.isnan(flux_map) & (flux_map > 0)]
            if len(valid_values) > 0:
                fig, ax = plt.subplots(figsize=(8, 7))
                
                # Use log scale for display
                norm = mcolors.LogNorm(
                    vmin=np.percentile(valid_values, 1),
                    vmax=np.percentile(valid_values, 99)
                )
                
                im = ax.imshow(flux_map, origin='lower', cmap='inferno', 
                             norm=norm, aspect='auto')
                plt.colorbar(im, ax=ax, label='Flux')
                ax.set_title(f"{line_name} Flux")
                
                fig.savefig(plots_dir / f"{galaxy_name}_P2P_{line_name}_flux.png", dpi=150)
                plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating flux map for {line_name}: {str(e)}")
            plt.close('all')
    
    # Create BPT-style diagnostic plots if line ratios available
    if ('line_ratios' in emission_params and 
        'OIII_Hb' in emission_params['line_ratios']):
        
        try:
            # Get line ratio
            oiii_hb = emission_params['line_ratios']['OIII_Hb']
            
            # Check data validity
            valid_values = oiii_hb[~np.isnan(oiii_hb) & (oiii_hb > 0)]
            if len(valid_values) > 0:
                fig, ax = plt.subplots(figsize=(8, 7))
                
                # Use log scale for display
                norm = mcolors.LogNorm(
                    vmin=np.percentile(valid_values, 1),
                    vmax=np.percentile(valid_values, 99)
                )
                
                im = ax.imshow(oiii_hb, origin='lower', cmap='viridis', 
                             norm=norm, aspect='auto')
                plt.colorbar(im, ax=ax, label='Ratio')
                ax.set_title("OIII/Hβ Ratio")
                
                fig.savefig(plots_dir / f"{galaxy_name}_P2P_OIII_Hb_ratio.png", dpi=150)
                plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating OIII/Hb ratio map: {str(e)}")
            plt.close('all')


def create_sample_fits(cube, velocity_field, bestfit_field, emission_result, plots_dir, galaxy_name):
    """
    Create spectrum fits plots for sample pixels
    
    Parameters
    ----------
    cube : MUSECube
        MUSE data cube object
    velocity_field : ndarray
        Velocity field
    bestfit_field : ndarray
        Best-fit spectra
    emission_result : dict
        Emission line results 
    plots_dir : Path
        Path to save plots
    galaxy_name : str
        Galaxy name for file naming
    """
    n_y, n_x = velocity_field.shape
    
    # Select sample positions
    center_y, center_x = n_y // 2, n_x // 2
    sample_positions = [
        (center_y, center_x),  # Center
        (center_y, min(center_x + n_x//4, n_x-1)),  # Right
        (min(center_y + n_y//4, n_y-1), center_x),  # Top
        (max(center_y - n_y//4, 0), max(center_x - n_x//4, 0))  # Bottom-left
    ]
    
    # Filter to ensure positions are valid
    valid_positions = []
    for y, x in sample_positions:
        if 0 <= y < n_y and 0 <= x < n_x and np.isfinite(velocity_field[y, x]):
            valid_positions.append((y, x))
    
    # If no valid positions found, try to find at least one
    if not valid_positions:
        valid_mask = np.isfinite(velocity_field)
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)
            valid_positions = [(valid_indices[0][0], valid_indices[1][0])]
        else:
            logger.warning("No valid pixels found for spectrum plots. Skipping.")
            return
    
    for i, (y, x) in enumerate(valid_positions):
        try:
            # Get spaxel index
            idx = y * n_x + x
            
            # Get observed spectrum
            observed = cube._spectra[:, idx]
            
            # Get model spectrum
            model = bestfit_field[:, y, x]
            
            # Get gas component if available
            gas_comp = None
            if emission_result is not None:
                # Try to get gas component from emission_result
                if 'gas_bestfit' in emission_result:
                    gas_bestfit = emission_result['gas_bestfit']
                    if gas_bestfit is not None:
                        # Check shape and extract appropriately
                        if len(gas_bestfit.shape) == 3:  # [n_wave, n_y, n_x]
                            gas_comp = gas_bestfit[:, y, x]
                        elif len(gas_bestfit.shape) == 2:  # [n_wave, n_spectra]
                            gas_comp = gas_bestfit[:, idx]
                elif 'gas_bestfit_field' in emission_result:
                    gas_bestfit = emission_result['gas_bestfit_field']
                    if gas_bestfit is not None:
                        if len(gas_bestfit.shape) == 3:
                            gas_comp = gas_bestfit[:, y, x]
                        elif len(gas_bestfit.shape) == 2:
                            gas_comp = gas_bestfit[:, idx]
                
                # If not found in emission_result, try cube
                if gas_comp is None and hasattr(cube, '_gas_bestfit_field'):
                    gas_comp = cube._gas_bestfit_field[:, y, x]
                    
                # Verify it's a valid array
                if gas_comp is not None and not np.any(np.isfinite(gas_comp)):
                    gas_comp = None
            
            # Create stellar component by subtracting gas
            stellar_comp = model.copy()
            if gas_comp is not None:
                stellar_comp -= gas_comp
            
            # Create plot with error handling
            try:
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
                logger.error(f"Error in plot_spectrum_fit for pixel ({x}, {y}): {str(e)}")
                plt.close('all')
        
        except Exception as e:
            logger.error(f"Error creating spectrum plot for pixel ({x}, {y}): {str(e)}")
            plt.close('all')


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


def create_radial_profile_plots(results, plots_dir, galaxy_name, analysis_type="P2P"):
    """
    Create radial profile plots showing parameters vs radius
    
    Parameters
    ----------
    results : dict
        Analysis results dictionary
    plots_dir : Path
        Directory to save plots
    galaxy_name : str
        Galaxy name
    analysis_type : str
        Analysis type ("P2P", "VNB", "RDB")
    """
    # Extract distance information
    if analysis_type == "P2P":
        # For P2P, need radial averaging
        distance_field = results['distance']['field']
        # Replace NaN values with large values so they'll be ignored
        valid_mask = np.isfinite(distance_field)
        
        # Create distance bins
        max_dist = np.nanmax(distance_field)
        r_bins = np.linspace(0, max_dist, 15)
        r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
        
        # Prepare dictionary to store radial parameters
        radial_params = {}
        
        # Process stellar kinematics parameters
        if 'stellar_kinematics' in results:
            velocity = results['stellar_kinematics']['velocity_field']
            dispersion = results['stellar_kinematics']['dispersion_field']
            
            # Calculate average for each radial bin
            vel_profile = []
            disp_profile = []
            vel_err = []
            disp_err = []
            
            for i in range(len(r_bins) - 1):
                r_min, r_max = r_bins[i], r_bins[i+1]
                r_mask = (distance_field >= r_min) & (distance_field < r_max) & valid_mask
                
                if np.any(r_mask & ~np.isnan(velocity)):
                    vel_values = velocity[r_mask & ~np.isnan(velocity)]
                    vel_profile.append(np.nanmean(vel_values))
                    vel_err.append(np.nanstd(vel_values) / np.sqrt(len(vel_values)))
                else:
                    vel_profile.append(np.nan)
                    vel_err.append(np.nan)
                
                if np.any(r_mask & ~np.isnan(dispersion)):
                    disp_values = dispersion[r_mask & ~np.isnan(dispersion)]
                    disp_profile.append(np.nanmean(disp_values))
                    disp_err.append(np.nanstd(disp_values) / np.sqrt(len(disp_values)))
                else:
                    disp_profile.append(np.nan)
                    disp_err.append(np.nan)
            
            radial_params['velocity'] = (r_centers, np.array(vel_profile), np.array(vel_err))
            radial_params['dispersion'] = (r_centers, np.array(disp_profile), np.array(disp_err))
        
        # Process stellar population parameters
        if 'stellar_population' in results:
            for param_name, param_map in results['stellar_population'].items():
                param_profile = []
                param_err = []
                
                for i in range(len(r_bins) - 1):
                    r_min, r_max = r_bins[i], r_bins[i+1]
                    r_mask = (distance_field >= r_min) & (distance_field < r_max) & valid_mask
                    
                    if np.any(r_mask & ~np.isnan(param_map)):
                        param_values = param_map[r_mask & ~np.isnan(param_map)]
                        param_profile.append(np.nanmean(param_values))
                        param_err.append(np.nanstd(param_values) / np.sqrt(len(param_values)))
                    else:
                        param_profile.append(np.nan)
                        param_err.append(np.nan)
                
                radial_params[param_name] = (r_centers, np.array(param_profile), np.array(param_err))
        
        # Process emission line parameters
        if 'emission' in results:
            # Process emission line fluxes
            for key, flux_map in results['emission'].items():
                if key.startswith('flux_') and isinstance(flux_map, np.ndarray):
                    flux_profile = []
                    flux_err = []
                    
                    for i in range(len(r_bins) - 1):
                        r_min, r_max = r_bins[i], r_bins[i+1]
                        r_mask = (distance_field >= r_min) & (distance_field < r_max) & valid_mask
                        
                        if np.any(r_mask & ~np.isnan(flux_map)):
                            flux_values = flux_map[r_mask & ~np.isnan(flux_map)]
                            flux_profile.append(np.nanmean(flux_values))
                            flux_err.append(np.nanstd(flux_values) / np.sqrt(len(flux_values)))
                        else:
                            flux_profile.append(np.nan)
                            flux_err.append(np.nan)
                    
                    radial_params[key] = (r_centers, np.array(flux_profile), np.array(flux_err))
            
            # Process line ratios
            if 'line_ratios' in results['emission']:
                for ratio_name, ratio_map in results['emission']['line_ratios'].items():
                    ratio_profile = []
                    ratio_err = []
                    
                    for i in range(len(r_bins) - 1):
                        r_min, r_max = r_bins[i], r_bins[i+1]
                        r_mask = (distance_field >= r_min) & (distance_field < r_max) & valid_mask
                        
                        if np.any(r_mask & ~np.isnan(ratio_map)):
                            ratio_values = ratio_map[r_mask & ~np.isnan(ratio_map)]
                            ratio_profile.append(np.nanmean(ratio_values))
                            ratio_err.append(np.nanstd(ratio_values) / np.sqrt(len(ratio_values)))
                        else:
                            ratio_profile.append(np.nan)
                            ratio_err.append(np.nan)
                    
                    radial_params[f"ratio_{ratio_name}"] = (r_centers, np.array(ratio_profile), np.array(ratio_err))
        
        # Process spectral indices
        if 'indices' in results:
            for index_name, index_map in results['indices'].items():
                index_profile = []
                index_err = []
                
                for i in range(len(r_bins) - 1):
                    r_min, r_max = r_bins[i], r_bins[i+1]
                    r_mask = (distance_field >= r_min) & (distance_field < r_max) & valid_mask
                    
                    if np.any(r_mask & ~np.isnan(index_map)):
                        index_values = index_map[r_mask & ~np.isnan(index_map)]
                        index_profile.append(np.nanmean(index_values))
                        index_err.append(np.nanstd(index_values) / np.sqrt(len(index_values)))
                    else:
                        index_profile.append(np.nan)
                        index_err.append(np.nan)
                
                radial_params[f"index_{index_name}"] = (r_centers, np.array(index_profile), np.array(index_err))
    
    else:  # For VNB and RDB, use directly provided bin distances and parameters
        if 'distance' not in results or 'bin_distances' not in results['distance']:
            logger.warning(f"No distance information in {analysis_type} results")
            return
        
        r_centers = results['distance']['bin_distances']
        
        # Prepare radial parameters dictionary
        radial_params = {}
        
        # Process stellar kinematics parameters
        if 'stellar_kinematics' in results:
            velocity = results['stellar_kinematics']['velocity']
            dispersion = results['stellar_kinematics']['dispersion']
            
            # For VNB and RDB, no direct error estimates, use zero error array
            zero_err = np.zeros_like(r_centers)
            
            radial_params['velocity'] = (r_centers, velocity, zero_err)
            radial_params['dispersion'] = (r_centers, dispersion, zero_err)
        
        # Process stellar population parameters
        if 'stellar_population' in results:
            for param_name, param_values in results['stellar_population'].items():
                zero_err = np.zeros_like(param_values)
                radial_params[param_name] = (r_centers, param_values, zero_err)
        
        # Process emission line parameters
        if 'emission' in results:
            # Process line fluxes
            for key, flux_values in results['emission'].items():
                if key.startswith('flux_') and isinstance(flux_values, np.ndarray):
                    zero_err = np.zeros_like(flux_values)
                    radial_params[key] = (r_centers, flux_values, zero_err)
            
            # Process line ratios
            if 'line_ratios' in results['emission']:
                for ratio_name, ratio_values in results['emission']['line_ratios'].items():
                    zero_err = np.zeros_like(ratio_values)
                    radial_params[f"ratio_{ratio_name}"] = (r_centers, ratio_values, zero_err)
        
        # Process spectral indices
        if 'indices' in results:
            for index_name, index_values in results['indices'].items():
                zero_err = np.zeros_like(index_values)
                radial_params[f"index_{index_name}"] = (r_centers, index_values, zero_err)
    
    # Create radial profile plots
    # 1. Stellar kinematics
    if 'velocity' in radial_params and 'dispersion' in radial_params:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Velocity profile
            r, vel, vel_err = radial_params['velocity']
            ax1.errorbar(r, vel, yerr=vel_err, fmt='o-', capsize=3)
            ax1.set_xlabel('Radius (arcsec)')
            ax1.set_ylabel('Velocity (km/s)')
            ax1.set_title('Stellar Velocity Profile')
            ax1.grid(True, alpha=0.3)
            
            # Dispersion profile
            r, disp, disp_err = radial_params['dispersion']
            ax2.errorbar(r, disp, yerr=disp_err, fmt='o-', capsize=3)
            ax2.set_xlabel('Radius (arcsec)')
            ax2.set_ylabel('Velocity Dispersion (km/s)')
            ax2.set_title('Stellar Velocity Dispersion Profile')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(plots_dir / f"{galaxy_name}_{analysis_type}_kinematics_profile.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating kinematics profile plot: {e}")
            plt.close('all')
    
    # 2. Stellar population parameters
    stellar_params = ['log_age', 'age', 'metallicity']
    present_params = [p for p in stellar_params if p in radial_params]
    
    if present_params:
        try:
            n_plots = len(present_params)
            fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
            
            for i, param_name in enumerate(present_params):
                r, values, errors = radial_params[param_name]
                
                # For age, convert to Gyr
                if param_name == 'age':
                    values = values * 1e-9  # Convert to Gyr
                    errors = errors * 1e-9  # Convert to Gyr
                    param_title = 'Age (Gyr)'
                elif param_name == 'log_age':
                    param_title = 'Log Age (yr)'
                elif param_name == 'metallicity':
                    param_title = 'Metallicity [Z/H]'
                else:
                    param_title = param_name
                
                axes[i].errorbar(r, values, yerr=errors, fmt='o-', capsize=3)
                axes[i].set_xlabel('Radius (arcsec)')
                axes[i].set_ylabel(param_title)
                axes[i].set_title(f'Stellar {param_title} Profile')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(plots_dir / f"{galaxy_name}_{analysis_type}_stellar_pop_profile.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating stellar population profile plot: {e}")
            plt.close('all')
    
    # 3. Emission line flux profiles
    flux_params = [p for p in radial_params if p.startswith('flux_')]
    
    if flux_params:
        try:
            n_plots = min(len(flux_params), 3)  # Maximum 3 lines
            fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
            
            for i, param_name in enumerate(flux_params[:n_plots]):
                r, values, errors = radial_params[param_name]
                line_name = param_name[5:]  # Remove 'flux_' prefix
                
                axes[i].errorbar(r, values, yerr=errors, fmt='o-', capsize=3)
                axes[i].set_xlabel('Radius (arcsec)')
                axes[i].set_ylabel('Flux')
                axes[i].set_title(f'{line_name} Flux Profile')
                axes[i].grid(True, alpha=0.3)
                
                # Try log scale
                try:
                    if np.all(values[~np.isnan(values)] > 0):
                        axes[i].set_yscale('log')
                except:
                    pass
            
            plt.tight_layout()
            fig.savefig(plots_dir / f"{galaxy_name}_{analysis_type}_emission_flux_profile.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating emission flux profile plot: {e}")
            plt.close('all')
    
    # 4. Line ratio profiles
    ratio_params = [p for p in radial_params if p.startswith('ratio_')]
    
    if ratio_params:
        try:
            n_plots = len(ratio_params)
            fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
            
            for i, param_name in enumerate(ratio_params):
                r, values, errors = radial_params[param_name]
                ratio_name = param_name[6:]  # Remove 'ratio_' prefix
                
                axes[i].errorbar(r, values, yerr=errors, fmt='o-', capsize=3)
                axes[i].set_xlabel('Radius (arcsec)')
                axes[i].set_ylabel('Ratio')
                axes[i].set_title(f'{ratio_name} Ratio Profile')
                axes[i].grid(True, alpha=0.3)
                
                # Try log scale
                try:
                    if np.all(values[~np.isnan(values)] > 0):
                        axes[i].set_yscale('log')
                except:
                    pass
            
            plt.tight_layout()
            fig.savefig(plots_dir / f"{galaxy_name}_{analysis_type}_line_ratios_profile.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating line ratios profile plot: {e}")
            plt.close('all')
    
    # 5. Spectral indices profiles
    index_params = [p for p in radial_params if p.startswith('index_')]
    
    if index_params:
        try:
            n_plots = min(len(index_params), 3)  # Maximum 3 indices
            fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
            
            for i, param_name in enumerate(index_params[:n_plots]):
                r, values, errors = radial_params[param_name]
                index_name = param_name[6:]  # Remove 'index_' prefix
                
                axes[i].errorbar(r, values, yerr=errors, fmt='o-', capsize=3)
                axes[i].set_xlabel('Radius (arcsec)')
                axes[i].set_ylabel('Index Value')
                axes[i].set_title(f'{index_name} Index Profile')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(plots_dir / f"{galaxy_name}_{analysis_type}_indices_profile.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating indices profile plot: {e}")
            plt.close('all')