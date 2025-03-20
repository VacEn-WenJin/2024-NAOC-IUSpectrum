"""
Voronoi binning analysis module for ISAPC
"""
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.colors as mcolors
import spectral_indices
import galaxy_params
import visualization
from utils.io import save_results_to_npz
from binning import (
    run_voronoi_binning, apply_velocity_shift, BinnedSpectra,
    plot_binned_map, plot_radial_profile
)

logger = logging.getLogger(__name__)


def run_vnb_analysis(args, cube, p2p_results=None):
    """
    Run Voronoi binning analysis on MUSE data cube

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cube : MUSECube
        MUSE data cube object
    p2p_results : dict, optional
        Results from P2P analysis, used to get velocity field for correction
        
    Returns
    -------
    dict
        Analysis results with binned data and physical parameters
    """
    logger.info("Starting Voronoi binning analysis...")
    start_time = time.time()
    
    # Disable warnings for spectral indices
    spectral_indices.set_warnings(False)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Extract galaxy name from filename
    galaxy_name = Path(args.filename).stem
    
    # Get target SNR from args
    target_snr = args.target_snr if hasattr(args, 'target_snr') else 20.0
    
    # Step 1: Calculate signal and noise for binning
    # ---------------------------------------------
    try:
        # Use median of a continuum region to estimate signal
        continuum_region = (cube._lambda_gal > 5000) & (cube._lambda_gal < 5200)
        
        # Calculate signal and noise for each spectrum
        signal = np.nanmedian(cube._spectra[continuum_region, :], axis=0)
        noise = np.nanstd(cube._spectra[continuum_region, :], axis=0)
        
        # Convert to 2D arrays
        signal_map = np.zeros((cube._n_y, cube._n_x))
        noise_map = np.zeros((cube._n_y, cube._n_x))
        snr_map = np.zeros((cube._n_y, cube._n_x))
        
        # Fill arrays
        for i in range(len(cube._spectra[0])):
            row = (i // cube._n_x)
            col = (i % cube._n_x)
            if row < cube._n_y and col < cube._n_x:
                signal_map[row, col] = signal[i]
                noise_map[row, col] = noise[i]
                if noise[i] > 0:
                    snr_map[row, col] = signal[i] / noise[i]
        
        # Get coordinates
        x = cube.x if hasattr(cube, 'x') else np.arange(len(signal)) % cube._n_x
        y = cube.y if hasattr(cube, 'y') else np.arange(len(signal)) // cube._n_x
        
        logger.info(f"Calculated signal and noise for {len(signal)} spaxels")
    except Exception as e:
        logger.error(f"Error calculating signal and noise: {e}")
        raise
    
    # Step 2: Run Voronoi binning
    # --------------------------
    try:
        logger.info(f"Running Voronoi binning with target SNR = {target_snr}")
        bin_num, x_gen, y_gen, sn, n_pixels, scale = run_voronoi_binning(
            x, y, signal, noise, target_snr, 
            plot=0, quiet=True, cvt=True
        )
        
        # Get unique bin numbers
        unique_bins = np.unique(bin_num)
        n_bins = len(unique_bins)
        logger.info(f"Created {n_bins} Voronoi bins")
        
        # Create arrays to store bin results
        bin_indices = []
        bin_spectra = np.zeros((len(cube._lambda_gal), n_bins))
        
        # Get velocity field for correction if available from P2P results
        velocity_field = None
        if p2p_results is not None and 'stellar_kinematics' in p2p_results:
            if 'velocity_field' in p2p_results['stellar_kinematics']:
                velocity_field = p2p_results['stellar_kinematics']['velocity_field']
                logger.info("Using P2P velocity field for bin spectral correction")
        
        # Combine spectra in each bin
        logger.info("Combining spectra in each bin...")
        for i, bin_id in enumerate(unique_bins):
            # Get indices of spectra in this bin
            mask = bin_num == bin_id
            indices = np.where(mask)[0]
            bin_indices.append(indices)
            
            # Combine spectra
            bin_spectra_list = []
            
            for idx in indices:
                # Get spectrum
                spectrum = cube._spectra[:, idx]
                
                # Apply velocity correction if available
                if velocity_field is not None:
                    # Convert 1D index to 2D position
                    row = idx // cube._n_x
                    col = idx % cube._n_x
                    
                    # Check if position is valid
                    if row < velocity_field.shape[0] and col < velocity_field.shape[1]:
                        vel = velocity_field[row, col]
                        
                        # Apply correction only if velocity is valid
                        if np.isfinite(vel):
                            spectrum = apply_velocity_shift(spectrum, cube._lambda_gal, vel)
                
                bin_spectra_list.append(spectrum)
            
            # Combine spectra (median)
            bin_spectra[:, i] = np.nanmedian(np.array(bin_spectra_list), axis=0)
        
        # Create BinnedSpectra object
        binned_data = BinnedSpectra(
            bin_type='voronoi',
            bin_num=bin_num,
            bin_indices=bin_indices,
            spectra=bin_spectra,
            wavelength=cube._lambda_gal,
            metadata={
                'x_gen': x_gen,
                'y_gen': y_gen,
                'sn': sn,
                'n_pixels': n_pixels,
                'scale': scale,
                'target_snr': target_snr
            }
        )
        
        # Save binned data
        binned_data.save(output_dir / f"{galaxy_name}_VNB_binned_data.npz")
        logger.info(f"Saved binned data to {galaxy_name}_VNB_binned_data.npz")
        
        # Create bin visualization
        if not args.no_plots:
            fig = plot_binned_map(
                x, y, bin_num, title=f"{galaxy_name} - Voronoi Bins (SNR={target_snr})",
                equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                savefile=plots_dir / f"{galaxy_name}_VNB_bins.png"
            )
            plt.close(fig)
            
            # SNR map
            fig = plot_binned_map(
                x, y, bin_num, values=sn, title=f"{galaxy_name} - Bin S/N",
                cmap='viridis', savefile=plots_dir / f"{galaxy_name}_VNB_snr.png"
            )
            plt.close(fig)
    except Exception as e:
        logger.error(f"Error in Voronoi binning: {e}")
        raise
    
    # Step 3: Create pseudo-cube for processing
    # ---------------------------------------
    try:
        # Convert binned data to P2P-compatible format
        pseudo_cube = binned_data.to_p2p_compatible(cube)
        
        # Process with stellar fitting code
        logger.info("Fitting stellar components to binned spectra...")
        stellar_fit_result = pseudo_cube.fit_spectra(
            template_filename=args.template,
            ppxf_vel_init=args.vel_init,
            ppxf_vel_disp_init=args.sigma_init,
            ppxf_deg=args.poly_degree if hasattr(args, 'poly_degree') else 3,
            n_jobs=args.n_jobs
        )
        
        stellar_velocity_field, stellar_dispersion_field, bestfit_field, optimal_tmpls, poly_coeffs = stellar_fit_result
        
        logger.info(f"Stellar component fitting completed in {time.time() - start_time:.1f} seconds")
        
        # Fit emission lines
        emission_result = None
        if not args.no_emission:
            start_time_em = time.time()
            emission_result = pseudo_cube.fit_emission_lines(
                template_filename=args.template,
                ppxf_vel_init=stellar_velocity_field,  # Use stellar velocity field as initial guess
                ppxf_sig_init=args.sigma_init,
                ppxf_deg=2,  # Simpler polynomial for emission lines
                n_jobs=args.n_jobs
            )
            logger.info(f"Emission line fitting completed in {time.time() - start_time_em:.1f} seconds")
        
        # Calculate spectral indices
        indices_result = None
        if not args.no_indices:
            start_time_ind = time.time()
            indices_result = pseudo_cube.calculate_spectral_indices(
                n_jobs=args.n_jobs
            )
            logger.info(f"Spectral indices calculation completed in {time.time() - start_time_ind:.1f} seconds")
    except Exception as e:
        logger.error(f"Error in spectral fitting: {e}")
        raise
    
    # Step 4: Process stellar population parameters
    # -------------------------------------------
    stellar_pop_params = None
    try:
        if hasattr(pseudo_cube, '_template_weights') and pseudo_cube._template_weights is not None:
            logger.info("Extracting stellar population parameters...")
            start_time_sp = time.time()
            
            # Initialize weight parser
            from stellar_population import WeightParser
            weight_parser = WeightParser(args.template)
            
            # Prepare arrays for physical parameters
            n_y, n_x = 1, n_bins  # Pseudo-cube dimensions
            stellar_pop_params = {
                'log_age': np.full((n_y, n_x), np.nan),
                'age': np.full((n_y, n_x), np.nan),
                'metallicity': np.full((n_y, n_x), np.nan)
            }
            
            # Process weights
            weights = pseudo_cube._template_weights
            
            if len(weights.shape) == 3:  # [n_templates, n_y, n_x]
                for x in range(n_x):
                    try:
                        pixel_weights = weights[:, 0, x]  # Using y=0 always
                        if np.sum(pixel_weights) > 0:
                            params = weight_parser.get_physical_params(pixel_weights)
                            for param_name, value in params.items():
                                stellar_pop_params[param_name][0, x] = value
                    except Exception as e:
                        logger.debug(f"Error calculating stellar params for bin {x}: {e}")
            
            logger.info(f"Stellar population parameters extracted in {time.time() - start_time_sp:.1f} seconds")
    except Exception as e:
        logger.error(f"Failed to extract stellar population parameters: {e}")
    
    # Step 5: Prepare results
    # ---------------------
    
    # Calculate radial distance for each bin
    bin_distances = np.sqrt(x_gen**2 + y_gen**2) * cube._pxl_size_x
    
    # Create results dictionary
    vnb_results = {
        # Bin information
        'bin_info': {
            'bin_num': bin_num,
            'bin_indices': bin_indices,
            'x_gen': x_gen,
            'y_gen': y_gen,
            'sn': sn,
            'n_pixels': n_pixels
        },
        
        # Distance information
        'distance': {
            'bin_distances': bin_distances,
            'pixelsize_x': cube._pxl_size_x,
            'pixelsize_y': cube._pxl_size_y
        },
        
        # Stellar kinematics
        'stellar_kinematics': {
            'velocity_field': stellar_velocity_field.reshape(1, n_bins),
            'dispersion_field': stellar_dispersion_field.reshape(1, n_bins),
            'velocity': stellar_velocity_field,  # 1D array for easy access
            'dispersion': stellar_dispersion_field  # 1D array for easy access
        }
    }
    
    # Add stellar population parameters if available
    if stellar_pop_params is not None:
        # Extract 1D arrays from 2D maps (pseudo-cube has shape [1, n_bins])
        vnb_results['stellar_population'] = {
            'log_age': stellar_pop_params['log_age'][0, :],
            'age': stellar_pop_params['age'][0, :],
            'metallicity': stellar_pop_params['metallicity'][0, :]
        }
    
    # Add emission line results if available
    if emission_result is not None:
        emission_params = {}
        
        # Add velocity and dispersion fields if available
        if 'emission_vel' in emission_result:
            for line_name, vel_map in emission_result['emission_vel'].items():
                if not np.all(np.isnan(vel_map)):
                    emission_params['velocity_field'] = vel_map.reshape(1, n_bins)
                    emission_params['velocity'] = vel_map  # 1D array
                    break
        
        if 'emission_sig' in emission_result:
            for line_name, disp_map in emission_result['emission_sig'].items():
                if not np.all(np.isnan(disp_map)):
                    emission_params['dispersion_field'] = disp_map.reshape(1, n_bins)
                    emission_params['dispersion'] = disp_map  # 1D array
                    break
        
        # Add emission line fluxes
        if 'emission_flux' in emission_result:
            for line_name, flux_map in emission_result['emission_flux'].items():
                emission_params[f'flux_{line_name}'] = flux_map
        
        # Calculate line ratios
        try:
            line_ratios = {}
            
            # Check if Hbeta and [OIII]5007 are available
            hb_key = None
            oiii_key = None
            
            for key in emission_params.keys():
                if 'flux_Hbeta' in key:
                    hb_key = key
                elif 'flux_[OIII]5007' in key or 'flux_OIII_5007' in key:
                    oiii_key = key
            
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
        
        # Only add emission key if we have valid data
        if emission_params:
            vnb_results['emission'] = emission_params
    
    # Add spectral indices if available
    if indices_result is not None:
        vnb_results['indices'] = indices_result
    
    # Save results
    save_results_to_npz(
        output_file=output_dir / f"{galaxy_name}_VNB_results.npz",
        data_dict=vnb_results
    )
    
    # Save legacy format CSV file
    try:
        legacy_df = pd.DataFrame()
        
        # Format bin indices for compatibility with older code
        bin_indices_str = []
        for indices in bin_indices:
            bin_indices_str.append(str(indices.tolist()).replace(',', ''))
        
        # Prepare velocity and dispersion columns
        if 'stellar_kinematics' in vnb_results:
            vel = vnb_results['stellar_kinematics']['velocity']
            disp = vnb_results['stellar_kinematics']['dispersion']
            component_sol = [f'[{v}, {d}]' for v, d in zip(vel, disp)]
        else:
            component_sol = ['[0, 0]'] * n_bins
        
        # Add emission line fluxes if available
        h_beta_el_value = np.full(n_bins, np.nan)
        h_beta_el_anr = np.full(n_bins, np.nan)
        o3_5007_el_value = np.full(n_bins, np.nan)
        o3_5007_el_anr = np.full(n_bins, np.nan)
        
        if 'emission' in vnb_results:
            emission = vnb_results['emission']
            
            # Look for Hbeta flux
            for key in emission:
                if key.startswith('flux_') and 'Hbeta' in key:
                    h_beta_el_value = emission[key]
                    break
            
            # Look for OIII flux
            for key in emission:
                if key.startswith('flux_') and ('[OIII]5007' in key or 'OIII_5007' in key):
                    o3_5007_el_value = emission[key]
                    break
        
        # Add spectral indices if available
        h_beta_si = np.full(n_bins, np.nan)
        mg_b_si = np.full(n_bins, np.nan)
        fe_5015_si = np.full(n_bins, np.nan)
        
        if 'indices' in vnb_results:
            indices = vnb_results['indices']
            
            if 'Hbeta' in indices:
                h_beta_si = indices['Hbeta']
            
            if 'Mgb' in indices:
                mg_b_si = indices['Mgb']
            
            if 'Fe5015' in indices:
                fe_5015_si = indices['Fe5015']
        
        # Create final dataframe
        legacy_df = pd.DataFrame({
            'H_beta_EL_value': h_beta_el_value,
            'H_beta_EL_ANR': h_beta_el_anr,
            'O_3_5007_EL_value': o3_5007_el_value,
            'O_3_5007_EL_ANR': o3_5007_el_anr,
            'Component_Sol': component_sol,
            'H_beta_SI': h_beta_si,
            'Mg_b_SI': mg_b_si,
            'Fe_5015_SI': fe_5015_si,
            'SNR': sn,
            'K_index': bin_indices_str
        })
        
        # Save to CSV
        legacy_df.to_csv(output_dir / f"{galaxy_name}_VNB_SFR.csv", index=False)
        logger.info(f"Saved legacy format results to {galaxy_name}_VNB_SFR.csv")
    except Exception as e:
        logger.error(f"Error saving legacy format: {e}")
    
    # Step 6: Create visualizations
    # ---------------------------
    if not args.no_plots:
        try:
            # Create kinematics plots
            if 'stellar_kinematics' in vnb_results:
                velocity_field = vnb_results['stellar_kinematics']['velocity']
                dispersion_field = vnb_results['stellar_kinematics']['dispersion']
                
                # Create 2D maps using bin positions
                fig = plot_binned_map(
                    x_gen, y_gen, np.arange(n_bins), values=velocity_field,
                    title=f"{galaxy_name} - Stellar Velocity",
                    cmap='RdBu_r', 
                    vmin=-100, vmax=100,
                    equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                    savefile=plots_dir / f"{galaxy_name}_VNB_velocity.png"
                )
                plt.close(fig)
                
                fig = plot_binned_map(
                    x_gen, y_gen, np.arange(n_bins), values=dispersion_field,
                    title=f"{galaxy_name} - Stellar Dispersion",
                    cmap='viridis',
                    vmin=0, vmax=200,
                    equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                    savefile=plots_dir / f"{galaxy_name}_VNB_dispersion.png"
                )
                plt.close(fig)
                
                # Create radial profiles
                fig = plot_radial_profile(
                    bin_distances, velocity_field,
                    title=f"{galaxy_name} - Stellar Velocity Profile",
                    xlabel="Radius (arcsec)",
                    ylabel="Velocity (km/s)",
                    savefile=plots_dir / f"{galaxy_name}_VNB_velocity_profile.png"
                )
                plt.close(fig)
                
                fig = plot_radial_profile(
                    bin_distances, dispersion_field,
                    title=f"{galaxy_name} - Stellar Dispersion Profile",
                    xlabel="Radius (arcsec)",
                    ylabel="Dispersion (km/s)",
                    savefile=plots_dir / f"{galaxy_name}_VNB_dispersion_profile.png"
                )
                plt.close(fig)
            
            # Create stellar population plots
            if 'stellar_population' in vnb_results:
                for param, values in vnb_results['stellar_population'].items():
                    if param == 'age':
                        # Convert to Gyr for plotting
                        values_gyr = values * 1e-9
                        title = f"{galaxy_name} - Stellar Age (Gyr)"
                        
                        fig = plot_binned_map(
                            x_gen, y_gen, np.arange(n_bins), values=values_gyr,
                            title=title, cmap='plasma',
                            equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                            savefile=plots_dir / f"{galaxy_name}_VNB_{param}.png"
                        )
                        plt.close(fig)
                        
                        fig = plot_radial_profile(
                            bin_distances, values_gyr,
                            title=f"{galaxy_name} - Stellar Age Profile",
                            xlabel="Radius (arcsec)",
                            ylabel="Age (Gyr)",
                            savefile=plots_dir / f"{galaxy_name}_VNB_{param}_profile.png"
                        )
                        plt.close(fig)
                    else:
                        title = f"{galaxy_name} - Stellar {param.capitalize()}"
                        
                        fig = plot_binned_map(
                            x_gen, y_gen, np.arange(n_bins), values=values,
                            title=title, cmap='viridis',
                            equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                            savefile=plots_dir / f"{galaxy_name}_VNB_{param}.png"
                        )
                        plt.close(fig)
                        
                        fig = plot_radial_profile(
                            bin_distances, values,
                            title=f"{galaxy_name} - Stellar {param.capitalize()} Profile",
                            xlabel="Radius (arcsec)",
                            ylabel=param.capitalize(),
                            savefile=plots_dir / f"{galaxy_name}_VNB_{param}_profile.png"
                        )
                        plt.close(fig)
            
            # Create emission line plots
            if 'emission' in vnb_results:
                emission = vnb_results['emission']
                
                # Plot emission line velocities if available
                if 'velocity' in emission:
                    fig = plot_binned_map(
                        x_gen, y_gen, np.arange(n_bins), values=emission['velocity'],
                        title=f"{galaxy_name} - Gas Velocity",
                        cmap='RdBu_r',
                        vmin=-100, vmax=100,
                        equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                        savefile=plots_dir / f"{galaxy_name}_VNB_gas_velocity.png"
                    )
                    plt.close(fig)
                    
                    fig = plot_radial_profile(
                        bin_distances, emission['velocity'],
                        title=f"{galaxy_name} - Gas Velocity Profile",
                        xlabel="Radius (arcsec)",
                        ylabel="Velocity (km/s)",
                        savefile=plots_dir / f"{galaxy_name}_VNB_gas_velocity_profile.png"
                    )
                    plt.close(fig)
                
                # Plot emission line fluxes
                for key, values in emission.items():
                    if key.startswith('flux_'):
                        line_name = key[5:]  # Remove 'flux_' prefix
                        
                        # Only plot if we have valid values
                        if np.any(~np.isnan(values)):
                            fig = plot_binned_map(
                                x_gen, y_gen, np.arange(n_bins), values=values,
                                title=f"{galaxy_name} - {line_name} Flux",
                                cmap='inferno',
                                equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                                savefile=plots_dir / f"{galaxy_name}_VNB_{line_name}_flux.png"
                            )
                            plt.close(fig)
                            
                            fig = plot_radial_profile(
                                bin_distances, values,
                                title=f"{galaxy_name} - {line_name} Flux Profile",
                                xlabel="Radius (arcsec)",
                                ylabel="Flux",
                                savefile=plots_dir / f"{galaxy_name}_VNB_{line_name}_flux_profile.png"
                            )
                            plt.close(fig)
                
                # Plot line ratios
                if 'line_ratios' in emission:
                    for ratio_name, values in emission['line_ratios'].items():
                        if np.any(~np.isnan(values)):
                            fig = plot_binned_map(
                                x_gen, y_gen, np.arange(n_bins), values=values,
                                title=f"{galaxy_name} - {ratio_name} Ratio",
                                cmap='viridis',
                                equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                                savefile=plots_dir / f"{galaxy_name}_VNB_{ratio_name}_ratio.png"
                            )
                            plt.close(fig)
                            
                            fig = plot_radial_profile(
                                bin_distances, values,
                                title=f"{galaxy_name} - {ratio_name} Ratio Profile",
                                xlabel="Radius (arcsec)",
                                ylabel="Ratio",
                                savefile=plots_dir / f"{galaxy_name}_VNB_{ratio_name}_ratio_profile.png"
                            )
                            plt.close(fig)
            
            # Create spectral indices plots
            if 'indices' in vnb_results:
                indices = vnb_results['indices']
                
                for index_name, values in indices.items():
                    # Only plot if we have valid values
                    if np.any(~np.isnan(values)):
                        fig = plot_binned_map(
                            x_gen, y_gen, np.arange(n_bins), values=values,
                            title=f"{galaxy_name} - {index_name} Index",
                            cmap='viridis',
                            equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                            savefile=plots_dir / f"{galaxy_name}_VNB_{index_name}_index.png"
                        )
                        plt.close(fig)
                        
                        fig = plot_radial_profile(
                            bin_distances, values,
                            title=f"{galaxy_name} - {index_name} Index Profile",
                            xlabel="Radius (arcsec)",
                            ylabel=f"{index_name} Index",
                            savefile=plots_dir / f"{galaxy_name}_VNB_{index_name}_index_profile.png"
                        )
                        plt.close(fig)
            
            logger.info("Generated visualization plots")
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
    
    logger.info(f"Voronoi binning analysis completed in {time.time() - start_time:.1f} seconds")
    return vnb_results