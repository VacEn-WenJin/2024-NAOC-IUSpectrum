"""
Radial binning analysis module for ISAPC
"""
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import spectral_indices
import galaxy_params
import visualization
import binning

from utils.io import save_results_to_npz
from utils.parallel import ParallelTqdm
from joblib import delayed

logger = logging.getLogger(__name__)


def run_rdb_analysis(args, cube, p2p_results=None):
    """
    Run radial binning analysis
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cube : MUSECube
        MUSE data cube object
    p2p_results : dict, optional
        Results from pixel-to-pixel analysis
        
    Returns
    -------
    dict
        Analysis results
    """
    logger.info("Starting radial binning analysis...")
    start_time = time.time()
    
    # Prepare data
    n_y, n_x = cube._cube_data.shape[1:]
    
    # Calculate center if not provided
    center_x = args.center_x if args.center_x is not None else n_x / 2
    center_y = args.center_y if args.center_y is not None else n_y / 2
    
    # Create RadialBinning instance
    rdb = binning.RadialBinning(
        x=cube.x, 
        y=cube.y,
        signal=cube._signal,
        noise=cube._noise,
        wavelength=cube._lambda_gal,
        spectra=cube._spectra,
        shape=(n_y, n_x)
    )
    
    # Execute binning
    try:
        bin_result = rdb.compute_bins(
            n_bins=args.n_rings,
            center_x=center_x,
            center_y=center_y,
            pa=args.pa,
            ellipticity=args.ellipticity,
            log_spacing=args.log_spacing
        )
        
        logger.info(f"Created {args.n_rings} radial bins")
    except Exception as e:
        logger.error(f"Error computing radial bins: {str(e)}")
        return None
    
    # Extract binned spectra
    velocity_field = None
    if p2p_results is not None:
        velocity_field = p2p_results['velocity_field']
    
    try:
        bin_spectra = rdb.extract_binned_spectra(bin_result['bin_map'], velocity_field)
        logger.info(f"Extracted {len(bin_spectra)} bin spectra")
    except Exception as e:
        logger.error(f"Error extracting binned spectra: {str(e)}")
        return None
    
    # Fit binned spectra
    bin_velocity = np.full((n_y, n_x), np.nan)
    bin_dispersion = np.full((n_y, n_x), np.nan)
    bin_results = {}
    
    # Define fitting function - similar to VNB's fit_bin
    def fit_bin(bin_idx):
        """Fit a single radial bin"""
        bin_spectrum = bin_spectra[bin_idx]
        
        try:
            # Get radius for this bin
            radius = (bin_result['bin_edges'][bin_idx] + bin_result['bin_edges'][bin_idx+1]) / 2
            
            # Create temporary data cube for fitting
            temp_cube = cube.__class__(
                filename=args.filename,
                redshift=args.redshift,
                wvl_air_angstrom_range=args.wvl_range
            )
            
            # Replace with bin spectrum
            temp_cube._spectra = bin_spectrum.reshape(-1, 1)
            
            # Fit stellar component
            result = temp_cube.fit_spectra(
                template_filename=args.template,
                ppxf_vel_init=args.vel_init,
                ppxf_vel_disp_init=args.sigma_init,
                ppxf_deg=args.poly_degree if hasattr(args, 'poly_degree') else 3,
                n_jobs=1  # Single bin fitting
            )
            
            # Check if fitting was successful
            if result is None or np.isnan(result[0][0, 0]):
                return bin_idx, None
            
            # Add emission line fitting
            emission_result = None
            if not args.no_emission:
                emission_result = temp_cube.fit_emission_lines(n_jobs=1)
            
            # Calculate spectral indices
            indices = None
            if not args.no_indices:
                indices = temp_cube.calculate_spectral_indices(n_jobs=1)
            
            # Return results
            fit_result = {
                'velocity': result[0][0, 0],  # Take value from first cell
                'dispersion': result[1][0, 0],
                'bestfit': result[2][:, 0, 0],
                'optimal_tmpl': result[3][:, 0, 0],
                'radius': radius  # Include radius for rotation curve
            }
            
            if indices is not None:
                # Extract indices values for this bin
                indices_values = {}
                for name, index_map in indices.items():
                    indices_values[name] = index_map[0, 0]
                fit_result['indices'] = indices_values
                
            if emission_result is not None:
                fit_result['emission'] = {}
                for name, flux_map in emission_result['emission_flux'].items():
                    fit_result['emission'][name] = flux_map[0, 0]
                
                # Save gas spectrum if available
                if 'gas_bestfit_field' in emission_result:
                    fit_result['gas_bestfit'] = emission_result['gas_bestfit_field'][:, 0, 0]
            
            return bin_idx, fit_result
        except Exception as e:
            logger.warning(f"Error fitting radial bin {bin_idx}: {str(e)}")
            return bin_idx, None
    
    # Fit bins in parallel
    bin_idx_list = list(bin_spectra.keys())
    fit_results = ParallelTqdm(
        n_jobs=args.n_jobs, desc='Fitting radial bins', total_tasks=len(bin_idx_list)
    )(delayed(fit_bin)(bin_idx) for bin_idx in bin_idx_list)
    
    # Process results
    for bin_idx, result in fit_results:
        if result is None:
            continue
            
        bin_results[bin_idx] = result
        
        # Update bin mapping
        bin_mask = (bin_result['bin_map'] == bin_idx)
        bin_velocity[bin_mask] = result['velocity']
        bin_dispersion[bin_mask] = result['dispersion']
    
    logger.info(f"Bin fitting completed in {time.time() - start_time:.1f} seconds")
    
    # Build rotation curve data
    rotation_curve = []
    for bin_idx, result in sorted(bin_results.items()):
        rotation_curve.append([result['radius'], result['velocity']])
    
    if rotation_curve:
        rotation_curve = np.array(rotation_curve)
    else:
        rotation_curve = np.array([[0, 0]])
    
    # Calculate galaxy parameters
    try:
        gp = galaxy_params.GalaxyParameters(
            velocity_field=bin_velocity,
            dispersion_field=bin_dispersion,
            pixelsize=cube._pxl_size_x
        )
        
        rotation_result = gp.fit_rotation_curve(
            center_x=center_x,
            center_y=center_y,
            pa_initial=args.pa
        )
        kinematics_result = gp.calculate_kinematics()
    except Exception as e:
        logger.error(f"Error calculating galaxy parameters: {str(e)}")
        rotation_result = {}
        kinematics_result = {}
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    galaxy_name = Path(args.filename).stem
    
    # Create results dictionary
    rdb_results = {
        'bin_map': bin_result['bin_map'],
        'radial_map': bin_result['radial_map'],
        'bin_edges': bin_result['bin_edges'],
        'velocity_field': bin_velocity,
        'dispersion_field': bin_dispersion,
        'bin_results': bin_results,
        'rotation_curve': rotation_curve,
        'kinematics': {**rotation_result, **kinematics_result}
    }
    
    # Save as NPZ file
    save_results_to_npz(
        output_file=output_dir / f"{galaxy_name}_RDB_results.npz",
        data_dict=rdb_results
    )
    
    # Create visualizations
    if not args.no_plots:
        create_rdb_plots(args, cube, rdb_results, bin_spectra, galaxy_name)
    
    logger.info("Radial binning analysis completed")
    return rdb_results


def create_rdb_plots(args, cube, rdb_results, bin_spectra, galaxy_name):
    """
    Create plots for radial binning analysis
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    cube : MUSECube
        MUSE data cube object
    rdb_results : dict
        Analysis results
    bin_spectra : dict
        Dictionary of binned spectra
    galaxy_name : str
        Galaxy name for file naming
    """
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract results
    bin_map = rdb_results['bin_map']
    radial_map = rdb_results['radial_map']
    bin_velocity = rdb_results['velocity_field']
    bin_dispersion = rdb_results['dispersion_field']
    bin_results = rdb_results['bin_results']
    rotation_curve = rdb_results['rotation_curve']
    rotation_result = rdb_results['kinematics']
    
    # Create radial distance plot
    try:
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(radial_map, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, label='Radius (pixels)')
        ax.set_title(f"Radial Distance Map")
        fig.savefig(plots_dir / f"{galaxy_name}_RDB_distance.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating distance map: {str(e)}")
    
    # Create binning plot
    try:
        fig, ax = visualization.plot_binning_map(
            bin_map=bin_map,
            title=f"Radial Binning ({args.n_rings} rings)",
            equal_aspect=args.equal_aspect
        )
        fig.savefig(plots_dir / f"{galaxy_name}_RDB_binning.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating binning map: {str(e)}")
    
    # Create kinematics plot
    try:
        fig = visualization.plot_kinematics_summary(
            velocity_field=bin_velocity,
            dispersion_field=bin_dispersion,
            bin_map=bin_map,
            rotation_curve=rotation_curve,
            params=rotation_result,
            equal_aspect=args.equal_aspect
        )
        fig.savefig(plots_dir / f"{galaxy_name}_RDB_kinematics.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating kinematics plot: {str(e)}")
    
    # Create rotation curve plot
    try:
        fig, ax = visualization.plot_rotation_curve(
            rotation_curve=rotation_curve,
            plot_model=True,
            vmax=rotation_result.get('vmax', None),
            pa=rotation_result.get('pa', None),
            title=f"Rotation Curve"
        )
        fig.savefig(plots_dir / f"{galaxy_name}_RDB_rotation_curve.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating rotation curve plot: {str(e)}")
    
    # Create spectrum fitting plots for sample bins
    sample_bins = list(bin_results.keys())
    if len(sample_bins) > 4:
        # Select evenly spaced samples
        indices = np.linspace(0, len(sample_bins) - 1, 4).astype(int)
        sample_bins = [sample_bins[i] for i in indices if i < len(sample_bins)]
    
    for i, bin_idx in enumerate(sample_bins):
        result = bin_results[bin_idx]
        
        try:
            # Get bin spectrum
            observed = bin_spectra[bin_idx]
            model = result['bestfit']
            
            # Get emission line component
            gas_comp = None
            if 'gas_bestfit' in result:
                gas_comp = result['gas_bestfit']
                # Verify it's a valid array
                if not np.any(np.isfinite(gas_comp)):
                    gas_comp = None
            elif 'emission' in result:
                # Simplified approach - reconstruct from emission fluxes
                gas_flux_sum = sum(result['emission'].values())
                if gas_flux_sum > 0:
                    gas_comp = np.ones_like(model) * 0.01  # Placeholder
            
            # Create stellar component by subtracting gas
            stellar_comp = model.copy()
            if gas_comp is not None:
                stellar_comp -= gas_comp
            
            # Create plot
            fig, axes = visualization.plot_spectrum_fit(
                wavelength=cube._lambda_gal,
                observed_flux=observed,
                model_flux=model,
                stellar_flux=stellar_comp,
                gas_flux=gas_comp,
                title=f"Radial Bin {bin_idx} (r={result['radius']:.1f} px)"
            )
            
            fig.savefig(plots_dir / f"{galaxy_name}_RDB_spectrum_{bin_idx}.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating spectrum plot for bin {bin_idx}: {str(e)}")
    
    # Plot spectral indices for representative bin if available
    if not args.no_indices and sample_bins:
        bin_idx = sample_bins[0]
        result = bin_results[bin_idx]
        
        # Create spectral index plot if indices were calculated
        if 'indices' in result and result['indices']:
            try:
                observed = bin_spectra[bin_idx]
                model = result['bestfit']
                
                # Get gas component if available
                gas_model = result.get('gas_bestfit', None)
                # Verify it's a valid array
                if gas_model is not None and not np.any(np.isfinite(gas_model)):
                    gas_model = None
                
                # Create LineIndexCalculator
                calculator = spectral_indices.LineIndexCalculator(
                    wave=cube._lambda_gal,
                    flux=observed,
                    fit_wave=cube._lambda_gal,
                    fit_flux=model,
                    em_wave=cube._lambda_gal if gas_model is not None else None,
                    em_flux_list=gas_model,
                    velocity_correction=result['velocity'],
                    continuum_mode='auto'
                )
                
                # Plot spectral lines with indices
                fig, axes = calculator.plot_all_lines(
                    mode="RDB", 
                    number=bin_idx,
                    save_path=str(plots_dir),
                    show_index=True
                )
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error creating spectral index plot: {str(e)}")