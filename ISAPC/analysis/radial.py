"""
Radial binning analysis module for ISAPC
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
    calculate_radial_bins, apply_velocity_shift, BinnedSpectra, RadialBinnedData,
    plot_binned_map, plot_radial_profile
)

logger = logging.getLogger(__name__)


def run_rdb_analysis(args, cube, p2p_results=None):
    """
    Run Radial binning analysis on MUSE data cube

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
    logger.info("Starting Radial binning analysis...")
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
    
    # Get radial binning parameters
    n_rings = args.n_rings if hasattr(args, 'n_rings') else 10
    center_x = args.center_x if hasattr(args, 'center_x') and args.center_x is not None else 0
    center_y = args.center_y if hasattr(args, 'center_y') and args.center_y is not None else 0
    pa = args.pa if hasattr(args, 'pa') else 0
    ellipticity = args.ellipticity if hasattr(args, 'ellipticity') else 0
    log_spacing = args.log_spacing if hasattr(args, 'log_spacing') else False
    
    # Step 1: Extract coordinates for binning
    # ---------------------------------------------
    try:
        # 检查是否有goodwavelength可用于截取光谱
        wave_mask = None
        good_lambda = None

        # 首先检查cube对象是否已经有_goodwavelength属性
        if hasattr(cube, '_goodwavelength') and cube._goodwavelength is not None:
            good_lambda = cube._goodwavelength
            wave_mask = (cube._lambda_gal >= good_lambda[0]) & (cube._lambda_gal <= good_lambda[1])
            logger.info(f"Using goodwavelength range from cube object: {good_lambda[0]:.1f} - {good_lambda[1]:.1f} Å")
        # 如果没有，尝试从FITS头中读取
        elif hasattr(cube, '_fits_hdu_header'):
            if 'WAVGOOD0' in cube._fits_hdu_header and 'WAVGOOD1' in cube._fits_hdu_header:
                good_lambda = (
                    float(cube._fits_hdu_header['WAVGOOD0']) / (1 + cube._redshift),
                    float(cube._fits_hdu_header['WAVGOOD1']) / (1 + cube._redshift)
                )
                wave_mask = (cube._lambda_gal >= good_lambda[0]) & (cube._lambda_gal <= good_lambda[1])
                logger.info(f"Found goodwavelength range in header with redshift correction: {good_lambda[0]:.1f} - {good_lambda[1]:.1f} Å")
            else:
                # 如果在FITS头中也找不到
                wave_mask = np.ones_like(cube._lambda_gal, dtype=bool)
                logger.info("No goodwavelength range found in header, using full wavelength range")
        else:
            # 如果没有goodwavelength，使用全部波长
            wave_mask = np.ones_like(cube._lambda_gal, dtype=bool)
            logger.info("No goodwavelength range found, using full wavelength range")

        # 截取波长范围
        wavelength = cube._lambda_gal[wave_mask]
        
        # Get coordinates
        x = np.zeros(cube._n_y * cube._n_x)
        y = np.zeros(cube._n_y * cube._n_x)
        
        # Create a grid of pixel indices
        y_indices, x_indices = np.indices((cube._n_y, cube._n_x))
        
        # Get center in pixel coordinates if not provided
        if center_x == 0 and center_y == 0:
            center_x = cube._n_x // 2
            center_y = cube._n_y // 2
            logger.info(f"Using image center as default: ({center_x}, {center_y})")
        
        # Convert indices to physical coordinates (relative to center)
        x = (x_indices.ravel() - center_x) * cube._pxl_size_x
        y = (y_indices.ravel() - center_y) * cube._pxl_size_y
        
        logger.info(f"Extracted coordinates for {len(x)} spaxels")
    except Exception as e:
        logger.error(f"Error extracting coordinates: {e}")
        raise
    
    # Step 2: Run Radial binning
    # --------------------------
    try:
        logger.info(f"Running Radial binning with {n_rings} rings")
        logger.info(f"Parameters: center=({center_x}, {center_y}), PA={pa}, ellipticity={ellipticity}")
        logger.info(f"Using {'logarithmic' if log_spacing else 'linear'} spacing")
        
        bin_num, bin_edges, bin_radii = calculate_radial_bins(
            x, y, center_x=0, center_y=0,  # Already centered in physical units
            pa=pa, ellipticity=ellipticity,
            n_rings=n_rings, log_spacing=log_spacing
        )
        
        # Get unique bin numbers
        unique_bins = np.unique(bin_num)
        n_bins = len(unique_bins)
        logger.info(f"Created {n_bins} radial bins")
        
        # Create arrays to store bin results
        bin_indices = []
        bin_spectra = np.zeros((len(wavelength), n_bins))  # 使用截取后的波长长度
        
        # Get velocity field for correction if available from P2P results
        velocity_field = None
        if p2p_results is not None and 'stellar_kinematics' in p2p_results:
            if 'velocity_field' in p2p_results['stellar_kinematics']:
                velocity_field = p2p_results['stellar_kinematics']['velocity_field']
                logger.info("Using P2P velocity field for bin spectral correction")
        
        # Combine spectra in each bin
        logger.info("Combining spectra in each bin...")
        bin_snr = np.zeros(n_bins)
        
        for i, bin_id in enumerate(unique_bins):
            # Get indices of spectra in this bin
            mask = bin_num == bin_id
            # Convert to flat indices
            flat_indices = np.where(mask)[0]
            bin_indices.append(flat_indices)
            
            # Convert flat indices to 2D indices
            y_idx = flat_indices // cube._n_x
            x_idx = flat_indices % cube._n_x
            
            # Combine spectra
            spectra_list = []
            
            for j in range(len(flat_indices)):
                row, col = y_idx[j], x_idx[j]
                
                # Get spectrum - need to convert to indices that match spectral array
                spaxel_idx = row * cube._n_x + col
                
                # 获取并截取光谱
                full_spectrum = cube._spectra[:, spaxel_idx]
                spectrum = full_spectrum[wave_mask]
                
                # Apply velocity correction if available
                if velocity_field is not None:
                    # Check if position is valid
                    if row < velocity_field.shape[0] and col < velocity_field.shape[1]:
                        vel = velocity_field[row, col]
                        
                        # Apply correction only if velocity is valid
                        if np.isfinite(vel):
                            spectrum = apply_velocity_shift(spectrum, wavelength, vel)
                
                spectra_list.append(spectrum)
            
            # Stack spectra for this bin
            stacked_spectra = np.array(spectra_list)
            
            # Combine spectra (median)
            bin_spectra[:, i] = np.nanmedian(stacked_spectra, axis=0)
            
            # Calculate SNR
            # 使用截取的波长区域内找合适的连续谱区域
            if np.any((wavelength > 5000) & (wavelength < 5200)):
                continuum_region = (wavelength > 5000) & (wavelength < 5200)
            else:
                # 如果截取后的波长区域不包含5000-5200区间，选择中间1/3作为连续谱
                total_len = len(wavelength)
                start_idx = total_len // 3
                end_idx = 2 * total_len // 3
                continuum_region = np.arange(start_idx, end_idx)
            
            signal = np.nanmedian(bin_spectra[continuum_region, i])
            noise = np.nanstd(bin_spectra[continuum_region, i])
            if noise > 0:
                bin_snr[i] = signal / noise
            else:
                bin_snr[i] = 0
        
        # Create metadata
        metadata = {
            'bin_edges': bin_edges,
            'bin_radii': bin_radii,
            'center_x': center_x,
            'center_y': center_y,
            'pa': pa,
            'ellipticity': ellipticity,
            'log_spacing': log_spacing,
            'sn': bin_snr
        }
        
        # Create RadialBinnedData object
        binned_data = RadialBinnedData(
            bin_num=bin_num,
            bin_indices=bin_indices,
            spectra=bin_spectra,
            wavelength=wavelength,  # 使用截取后的波长
            bin_radii=bin_radii,
            metadata=metadata
        )
        
        # Save binned data
        binned_data.save(output_dir / f"{galaxy_name}_RDB_binned_data.npz")
        logger.info(f"Saved binned data to {galaxy_name}_RDB_binned_data.npz")
        
        # 创建可视化图表
        binned_data.create_visualization_plots(output_dir, galaxy_name)
        
        # Create bin visualization
        if not args.no_plots:
            # Get 2D indices for plotting
            y_2d = y_indices.ravel()
            x_2d = x_indices.ravel()
            
            fig = plot_binned_map(
                x_2d, y_2d, bin_num, title=f"{galaxy_name} - Radial Bins",
                equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                savefile=plots_dir / f"{galaxy_name}_RDB_bins.png"
            )
            plt.close(fig)
            
            # SNR map
            snr_values = np.zeros_like(bin_num, dtype=float)
            for i, bin_id in enumerate(unique_bins):
                mask = bin_num == bin_id
                snr_values[mask] = bin_snr[i]
            
            fig = plot_binned_map(
                x_2d, y_2d, bin_num, values=snr_values, title=f"{galaxy_name} - Bin S/N",
                cmap='viridis', savefile=plots_dir / f"{galaxy_name}_RDB_snr.png"
            )
            plt.close(fig)
    except Exception as e:
        logger.error(f"Error in Radial binning: {e}")
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
    
    # Create results dictionary
    rdb_results = {
        # Bin information
        'bin_info': {
            'bin_num': bin_num,
            'bin_indices': bin_indices,
            'bin_edges': bin_edges,
            'n_rings': n_rings
        },
        
        # Distance information
        'distance': {
            'bin_distances': bin_radii,  # Physical radii in arcseconds
            'pixelsize_x': cube._pxl_size_x,
            'pixelsize_y': cube._pxl_size_y,
            'center_x': center_x,
            'center_y': center_y,
            'pa': pa,
            'ellipticity': ellipticity
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
        rdb_results['stellar_population'] = {
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
            rdb_results['emission'] = emission_params
    
    # Add spectral indices if available
    if indices_result is not None:
        rdb_results['indices'] = indices_result
    
    # Save results
    save_results_to_npz(
        output_file=output_dir / f"{galaxy_name}_RDB_results.npz",
        data_dict=rdb_results
    )
    
    # Save legacy format CSV file
    try:
        legacy_df = pd.DataFrame()
        
        # Format bin indices for compatibility with older code
        bin_indices_str = []
        for indices in bin_indices:
            bin_indices_str.append(str(indices.tolist()).replace(',', ''))
        
        # Prepare velocity and dispersion columns
        if 'stellar_kinematics' in rdb_results:
            vel = rdb_results['stellar_kinematics']['velocity']
            disp = rdb_results['stellar_kinematics']['dispersion']
            
            # 确保一维数组格式
            vel = vel.flatten() if hasattr(vel, 'flatten') else vel
            disp = disp.flatten() if hasattr(disp, 'flatten') else disp
            
            component_sol = [f'[{v}, {d}]' for v, d in zip(vel, disp)]
        else:
            component_sol = ['[0, 0]'] * n_bins
        
        # Add emission line fluxes if available
        h_beta_el_value = np.full(n_bins, np.nan).flatten()
        h_beta_el_anr = np.full(n_bins, np.nan).flatten()
        o3_5007_el_value = np.full(n_bins, np.nan).flatten()
        o3_5007_el_anr = np.full(n_bins, np.nan).flatten()
        
        if 'emission' in rdb_results:
            emission = rdb_results['emission']
            
            # Look for Hbeta flux
            for key in emission:
                if key.startswith('flux_') and 'Hbeta' in key:
                    h_beta_el_value = emission[key].flatten() if hasattr(emission[key], 'flatten') else emission[key]
                    break
            
            # Look for OIII flux
            for key in emission:
                if key.startswith('flux_') and ('[OIII]5007' in key or 'OIII_5007' in key):
                    o3_5007_el_value = emission[key].flatten() if hasattr(emission[key], 'flatten') else emission[key]
                    break
        
        # Add spectral indices if available
        h_beta_si = np.full(n_bins, np.nan).flatten()
        mg_b_si = np.full(n_bins, np.nan).flatten()
        fe_5015_si = np.full(n_bins, np.nan).flatten()
        
        if 'indices' in rdb_results:
            indices = rdb_results['indices']
            
            if 'Hbeta' in indices:
                h_beta_si = indices['Hbeta'].flatten() if hasattr(indices['Hbeta'], 'flatten') else indices['Hbeta']
            
            if 'Mgb' in indices:
                mg_b_si = indices['Mgb'].flatten() if hasattr(indices['Mgb'], 'flatten') else indices['Mgb']
            
            if 'Fe5015' in indices:
                fe_5015_si = indices['Fe5015'].flatten() if hasattr(indices['Fe5015'], 'flatten') else indices['Fe5015']
        
        # 确保所有数组都是一维的
        bin_radii_flat = bin_radii.flatten() if hasattr(bin_radii, 'flatten') else bin_radii
        bin_snr_flat = bin_snr.flatten() if hasattr(bin_snr, 'flatten') else bin_snr
        
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
            'R': bin_radii_flat,  # Add radius information (crucial for RDB)
            'SNR': bin_snr_flat,
            'K_index': bin_indices_str
        })
        
        # Save to CSV
        legacy_df.to_csv(output_dir / f"{galaxy_name}_RDB_SFR.csv", index=False)
        logger.info(f"Saved legacy format results to {galaxy_name}_RDB_SFR.csv")
    except Exception as e:
        logger.error(f"Error saving legacy format: {e}")
    
    # Step 6: Create visualizations
    # ---------------------------
    if not args.no_plots:
        try:
            # Create rotation curve and velocity field plots
            
            # For radial plots, we simply plot values against radius
            fig = plot_radial_profile(
                bin_radii, stellar_velocity_field,
                title=f"{galaxy_name} - Stellar Velocity Profile",
                xlabel="Radius (arcsec)",
                ylabel="Velocity (km/s)",
                savefile=plots_dir / f"{galaxy_name}_RDB_velocity_profile.png"
            )
            plt.close(fig)
            
            fig = plot_radial_profile(
                bin_radii, stellar_dispersion_field,
                title=f"{galaxy_name} - Stellar Dispersion Profile",
                xlabel="Radius (arcsec)",
                ylabel="Dispersion (km/s)",
                savefile=plots_dir / f"{galaxy_name}_RDB_dispersion_profile.png"
            )
            plt.close(fig)
            
            # Create 2D maps
            velocity_values = np.zeros_like(bin_num, dtype=float)
            dispersion_values = np.zeros_like(bin_num, dtype=float)
            
            for i, bin_id in enumerate(unique_bins):
                mask = bin_num == bin_id
                velocity_values[mask] = stellar_velocity_field[i]
                dispersion_values[mask] = stellar_dispersion_field[i]
            
            fig = plot_binned_map(
                x_2d, y_2d, bin_num, values=velocity_values,
                title=f"{galaxy_name} - Stellar Velocity",
                cmap='RdBu_r',
                vmin=-100, vmax=100,
                equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                savefile=plots_dir / f"{galaxy_name}_RDB_velocity_map.png"
            )
            plt.close(fig)
            
            fig = plot_binned_map(
                x_2d, y_2d, bin_num, values=dispersion_values,
                title=f"{galaxy_name} - Stellar Dispersion",
                cmap='viridis',
                vmin=0, vmax=200,
                equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                savefile=plots_dir / f"{galaxy_name}_RDB_dispersion_map.png"
            )
            plt.close(fig)
            
            # Create stellar population plots
            if 'stellar_population' in rdb_results:
                for param, values in rdb_results['stellar_population'].items():
                    if param == 'age':
                        # Convert to Gyr for plotting
                        values_gyr = values * 1e-9
                        
                        fig = plot_radial_profile(
                            bin_radii, values_gyr,
                            title=f"{galaxy_name} - Stellar Age Profile",
                            xlabel="Radius (arcsec)",
                            ylabel="Age (Gyr)",
                            savefile=plots_dir / f"{galaxy_name}_RDB_{param}_profile.png"
                        )
                        plt.close(fig)
                        
                        # Create 2D map
                        param_values = np.zeros_like(bin_num, dtype=float)
                        for i, bin_id in enumerate(unique_bins):
                            mask = bin_num == bin_id
                            param_values[mask] = values_gyr[i]
                        
                        fig = plot_binned_map(
                            x_2d, y_2d, bin_num, values=param_values,
                            title=f"{galaxy_name} - Stellar Age (Gyr)",
                            cmap='plasma',
                            equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                            savefile=plots_dir / f"{galaxy_name}_RDB_{param}_map.png"
                        )
                        plt.close(fig)
                    else:
                        fig = plot_radial_profile(
                            bin_radii, values,
                            title=f"{galaxy_name} - Stellar {param.capitalize()} Profile",
                            xlabel="Radius (arcsec)",
                            ylabel=param.capitalize(),
                            savefile=plots_dir / f"{galaxy_name}_RDB_{param}_profile.png"
                        )
                        plt.close(fig)
                        
                        # Create 2D map
                        param_values = np.zeros_like(bin_num, dtype=float)
                        for i, bin_id in enumerate(unique_bins):
                            mask = bin_num == bin_id
                            param_values[mask] = values[i]
                        
                        fig = plot_binned_map(
                            x_2d, y_2d, bin_num, values=param_values,
                            title=f"{galaxy_name} - Stellar {param.capitalize()}",
                            cmap='viridis',
                            equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                            savefile=plots_dir / f"{galaxy_name}_RDB_{param}_map.png"
                        )
                        plt.close(fig)
            
            # Create emission line plots
            if 'emission' in rdb_results:
                emission = rdb_results['emission']
                
                # Plot emission line velocities if available
                if 'velocity' in emission:
                    emission_vel = emission['velocity'].flatten() if hasattr(emission['velocity'], 'flatten') else emission['velocity']
                    
                    fig = plot_radial_profile(
                        bin_radii, emission_vel,
                        title=f"{galaxy_name} - Gas Velocity Profile",
                        xlabel="Radius (arcsec)",
                        ylabel="Velocity (km/s)",
                        savefile=plots_dir / f"{galaxy_name}_RDB_gas_velocity_profile.png"
                    )
                    plt.close(fig)
                    
                    # Create 2D map
                    gas_vel_values = np.zeros_like(bin_num, dtype=float)
                    for i, bin_id in enumerate(unique_bins):
                        mask = bin_num == bin_id
                        gas_vel_values[mask] = emission_vel[i]
                    
                    fig = plot_binned_map(
                        x_2d, y_2d, bin_num, values=gas_vel_values,
                        title=f"{galaxy_name} - Gas Velocity",
                        cmap='RdBu_r',
                        vmin=-100, vmax=100,
                        equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                        savefile=plots_dir / f"{galaxy_name}_RDB_gas_velocity_map.png"
                    )
                    plt.close(fig)
                
                # Plot emission line fluxes
                for key, values in emission.items():
                    if key.startswith('flux_'):
                        line_name = key[5:]  # Remove 'flux_' prefix
                        
                        # 确保是一维数组
                        line_values = values.flatten() if hasattr(values, 'flatten') else values
                        
                        # Only plot if we have valid values
                        if np.any(~np.isnan(line_values)):
                            fig = plot_radial_profile(
                                bin_radii, line_values,
                                title=f"{galaxy_name} - {line_name} Flux Profile",
                                xlabel="Radius (arcsec)",
                                ylabel="Flux",
                                savefile=plots_dir / f"{galaxy_name}_RDB_{line_name}_flux_profile.png"
                            )
                            plt.close(fig)
                            
                            # Create 2D map
                            flux_values = np.zeros_like(bin_num, dtype=float)
                            for i, bin_id in enumerate(unique_bins):
                                mask = bin_num == bin_id
                                flux_values[mask] = line_values[i] if i < len(line_values) else np.nan
                            
                            fig = plot_binned_map(
                                x_2d, y_2d, bin_num, values=flux_values,
                                title=f"{galaxy_name} - {line_name} Flux",
                                cmap='inferno',
                                equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                                savefile=plots_dir / f"{galaxy_name}_RDB_{line_name}_flux_map.png"
                            )
                            plt.close(fig)
                
                # Plot line ratios
                if 'line_ratios' in emission:
                    for ratio_name, values in emission['line_ratios'].items():
                        # 确保是一维数组
                        ratio_values = values.flatten() if hasattr(values, 'flatten') else values
                        
                        if np.any(~np.isnan(ratio_values)):
                            fig = plot_radial_profile(
                                bin_radii, ratio_values,
                                title=f"{galaxy_name} - {ratio_name} Ratio Profile",
                                xlabel="Radius (arcsec)",
                                ylabel="Ratio",
                                savefile=plots_dir / f"{galaxy_name}_RDB_{ratio_name}_ratio_profile.png"
                            )
                            plt.close(fig)
                            
                            # Create 2D map
                            ratio_map_values = np.zeros_like(bin_num, dtype=float)
                            for i, bin_id in enumerate(unique_bins):
                                mask = bin_num == bin_id
                                ratio_map_values[mask] = ratio_values[i] if i < len(ratio_values) else np.nan
                            
                            fig = plot_binned_map(
                                x_2d, y_2d, bin_num, values=ratio_map_values,
                                title=f"{galaxy_name} - {ratio_name} Ratio",
                                cmap='viridis',
                                equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                                savefile=plots_dir / f"{galaxy_name}_RDB_{ratio_name}_ratio_map.png"
                            )
                            plt.close(fig)
            
            # Create spectral indices plots
            if 'indices' in rdb_results:
                indices = rdb_results['indices']
                
                for index_name, values in indices.items():
                    # 确保是一维数组
                    index_values = values.flatten() if hasattr(values, 'flatten') else values
                    
                    # Only plot if we have valid values
                    if np.any(~np.isnan(index_values)):
                        fig = plot_radial_profile(
                            bin_radii, index_values,
                            title=f"{galaxy_name} - {index_name} Index Profile",
                            xlabel="Radius (arcsec)",
                            ylabel=f"{index_name} Index",
                            savefile=plots_dir / f"{galaxy_name}_RDB_{index_name}_index_profile.png"
                        )
                        plt.close(fig)
                        
                        # Create 2D map
                        index_map_values = np.zeros_like(bin_num, dtype=float)
                        for i, bin_id in enumerate(unique_bins):
                            mask = bin_num == bin_id
                            index_map_values[mask] = index_values[i] if i < len(index_values) else np.nan
                        
                        fig = plot_binned_map(
                            x_2d, y_2d, bin_num, values=index_map_values,
                            title=f"{galaxy_name} - {index_name} Index",
                            cmap='viridis',
                            equal_aspect=args.equal_aspect if hasattr(args, 'equal_aspect') else True,
                            savefile=plots_dir / f"{galaxy_name}_RDB_{index_name}_index_map.png"
                        )
                        plt.close(fig)
            
            logger.info("Generated visualization plots")
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
    
    logger.info(f"Radial binning analysis completed in {time.time() - start_time:.1f} seconds")
    return rdb_results