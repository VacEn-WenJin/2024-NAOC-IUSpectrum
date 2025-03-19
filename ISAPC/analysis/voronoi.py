"""
Voronoi binning analysis module for ISAPC
"""
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import spectral_indices
import galaxy_params
import visualization
from stellar_population import WeightParser

from utils.io import save_results_to_npz
from utils.parallel import ParallelTqdm
from joblib import delayed

logger = logging.getLogger(__name__)


def calculate_distance_to_center(x, y, pxl_size_x, pxl_size_y=None):
    """
    计算给定坐标到中心的物理距离
    
    Parameters
    ----------
    x : numpy.ndarray
        x坐标数组，单位为像素
    y : numpy.ndarray
        y坐标数组，单位为像素
    pxl_size_x : float
        x方向的像素尺寸，单位为角秒
    pxl_size_y : float, optional
        y方向的像素尺寸，默认与x方向相同
        
    Returns
    -------
    numpy.ndarray
        到中心的距离，单位为角秒
    """
    if pxl_size_y is None:
        pxl_size_y = pxl_size_x
        
    # 转换为物理单位（角秒）
    phys_x = x * pxl_size_x
    phys_y = y * pxl_size_y
    
    # 计算欧几里得距离
    distance = np.sqrt(phys_x**2 + phys_y**2)
    
    return distance


def run_vnb_analysis(args, cube, p2p_results=None):
    """
    运行Voronoi分bin的谱分析
    
    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    cube : MUSECube
        MUSE数据立方体对象
    p2p_results : dict, optional
        像素到像素分析的结果，用于初始化速度场
        
    Returns
    -------
    dict
        分析结果
    """
    logger.info("Starting Voronoi binning analysis...")
    start_time = time.time()
    
    # 临时关闭警告
    spectral_indices.set_warnings(False)
    
    # 使用p2p_results的速度场进行光谱校正（如果有）
    velocity_field = None
    if p2p_results is not None and 'stellar_kinematics' in p2p_results:
        velocity_field = p2p_results['stellar_kinematics']['velocity_field']
        logger.info("Using P2P velocity field for spectral extraction")
    
    # 执行Voronoi分bin
    bin_result = cube.voronoi_binning(
        target_snr=args.target_snr,
        min_snr=args.min_snr,
        velocity_field=velocity_field,  # 增加速度场以进行校正
        n_jobs=args.n_jobs
    )
    
    bin_nums, bin_spectra, bin_errors, bin_x, bin_y = bin_result
    
    # 获取有关分bin的信息
    n_bins = len(np.unique(bin_nums[bin_nums >= 0]))
    logger.info(f"Created {n_bins} Voronoi bins")
    
    # 拟合恒星连续谱
    fit_results = cube.bin_fit_stellar_continuum(
        template_filename=args.template,
        ppxf_vel_init=args.vel_init,
        ppxf_vel_disp_init=args.sigma_init,
        ppxf_deg=args.poly_degree if hasattr(args, 'poly_degree') else 3,
        n_jobs=args.n_jobs
    )
    
    bin_vel, bin_disp, bin_bestfit, bin_optimal_tmpls, bin_weights, bin_poly_coeffs = fit_results
    
    logger.info(f"Stellar continuum fitting completed in {time.time() - start_time:.1f} seconds")
    
    # 拟合发射线
    emission_result = None
    if not args.no_emission:
        start_time = time.time()
        emission_result = cube.bin_fit_emission_lines(
            template_filename=args.template,
            ppxf_vel_init=bin_vel,
            ppxf_sig_init=args.sigma_init,
            ppxf_deg=2,
            n_jobs=args.n_jobs
        )
        logger.info(f"Emission line fitting completed in {time.time() - start_time:.1f} seconds")
    
    # 计算谱指数
    indices_result = None
    if not args.no_indices:
        start_time = time.time()
        indices_result = cube.bin_calculate_spectral_indices(
            n_jobs=args.n_jobs
        )
        logger.info(f"Spectral indices calculation completed in {time.time() - start_time:.1f} seconds")
    
    # 准备气体运动学参数
    gas_vel = None
    gas_disp = None
    
    # 从emission_result中提取气体运动学参数，类似于P2P
    if emission_result is not None:
        try:
            # 尝试从emission_result中获取
            if 'bin_vel' in emission_result:
                gas_vel = emission_result['bin_vel']
            if 'bin_disp' in emission_result:
                gas_disp = emission_result['bin_disp']
            
            # 尝试从emission_result中的其他字段获取
            if gas_vel is None and 'emission_vel' in emission_result:
                # 获取第一个可用的发射线
                for line_name, vel_values in emission_result['emission_vel'].items():
                    if not np.all(np.isnan(vel_values)):
                        gas_vel = vel_values
                        logger.info(f"Using velocity from emission line: {line_name}")
                        break
                
                # 获取相应的弥散场
                if gas_vel is not None and 'emission_sig' in emission_result:
                    for line_name, disp_values in emission_result['emission_sig'].items():
                        if not np.all(np.isnan(disp_values)):
                            gas_disp = disp_values
                            logger.info(f"Using dispersion from emission line: {line_name}")
                            break
                            
            # 如果前两种方法都失败，尝试其他可能的键
            if gas_vel is None:
                for key in emission_result:
                    if ('vel' in key.lower() or 'velocity' in key.lower()) and key != 'bin_vel':
                        if isinstance(emission_result[key], np.ndarray) and len(emission_result[key]) == n_bins:
                            gas_vel = emission_result[key]
                            logger.info(f"Using velocity from key: {key}")
                            break
                
                for key in emission_result:
                    if ('disp' in key.lower() or 'sigma' in key.lower()) and key != 'bin_disp':
                        if isinstance(emission_result[key], np.ndarray) and len(emission_result[key]) == n_bins:
                            gas_disp = emission_result[key]
                            logger.info(f"Using dispersion from key: {key}")
                            break
            
            # 验证提取的气体运动学数据是否有效
            if gas_vel is not None:
                valid_bins = np.count_nonzero(~np.isnan(gas_vel))
                if valid_bins < 5:  # 假设少于5个有效bin不足以做有用的分析
                    logger.warning(f"Too few valid bins in gas velocity ({valid_bins}), using stellar velocity instead")
                    gas_vel = None
                    gas_disp = None
                else:
                    logger.info(f"Found {valid_bins} valid bins with gas velocity")
        except Exception as e:
            logger.error(f"Failed to extract gas kinematics: {e}")
    
    # 决定使用哪个速度场
    using_emission = False
    if gas_vel is not None and gas_disp is not None:
        # 检查气体速度场的质量/覆盖率
        valid_gas = np.sum(~np.isnan(gas_vel))
        valid_stellar = np.sum(~np.isnan(bin_vel))
        total_bins = len(bin_vel)
        
        gas_coverage = valid_gas / total_bins
        stellar_coverage = valid_stellar / total_bins
        
        # 如果气体覆盖率合理
        if gas_coverage > 0.3 or gas_coverage > 0.8 * stellar_coverage:
            logger.info(f"Using emission line velocity for kinematics (coverage: {gas_coverage:.2f})")
            velocity = gas_vel
            dispersion = gas_disp
            using_emission = True
        else:
            logger.info(f"Insufficient emission line coverage ({gas_coverage:.2f}), using stellar velocity")
            velocity = bin_vel
            dispersion = bin_disp
    else:
        logger.info("No emission line data available, using stellar velocity")
        velocity = bin_vel
        dispersion = bin_disp
    
    # 计算每个bin到中心的距离
    n_y, n_x = cube._n_y, cube._n_x
    center_y, center_x = n_y // 2, n_x // 2
    
    # 为每个bin计算平均位置和距离
    bin_distances = np.full(n_bins, np.nan)
    
    # 对每个bin，计算其包含的像素的平均位置
    for i in range(n_bins):
        bin_mask = (bin_nums == i)
        if np.any(bin_mask):
            # 获取该bin中的像素位置
            y_pos, x_pos = np.where(bin_mask)
            
            # 计算相对于中心的平均位置
            mean_rel_x = np.mean(x_pos - center_x)
            mean_rel_y = np.mean(y_pos - center_y)
            
            # 计算距离
            bin_distances[i] = calculate_distance_to_center(
                mean_rel_x, mean_rel_y, cube._pxl_size_x, cube._pxl_size_y
            )
    
    # 提取恒星物理参数
    stellar_pop_params = None
    if bin_weights is not None:
        try:
            logger.info("Extracting stellar population parameters...")
            start_time = time.time()
            
            # 初始化权重解析器
            weight_parser = WeightParser(args.template)
            
            # 准备存储物理参数的数组
            stellar_pop_params = {
                'log_age': np.full(n_bins, np.nan),
                'age': np.full(n_bins, np.nan),
                'metallicity': np.full(n_bins, np.nan)
            }
            
            # 处理每个bin的权重
            for i in range(n_bins):
                if np.isfinite(bin_vel[i]):
                    try:
                        bin_weight = bin_weights[i]
                        if np.sum(bin_weight) > 0:
                            params = weight_parser.get_physical_params(bin_weight)
                            for param_name, value in params.items():
                                stellar_pop_params[param_name][i] = value
                    except Exception as e:
                        logger.debug(f"Error calculating stellar params for bin {i}: {e}")
            
            logger.info(f"Stellar population parameters extracted in {time.time() - start_time:.1f} seconds")
        except Exception as e:
            logger.error(f"Failed to extract stellar population parameters: {e}")
    
    # 创建结果字典
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    galaxy_name = Path(args.filename).stem
    
    # 构建结果字典
    vnb_results = {
        # Voronoi分bin信息
        'binning': {
            'bin_nums': bin_nums,
            'n_bins': n_bins,
            'bin_x': bin_x,
            'bin_y': bin_y
        },
        # 恒星运动学
        'stellar_kinematics': {
            'velocity': bin_vel,
            'dispersion': bin_disp
        },
        # 距离信息
        'distance': {
            'bin_distances': bin_distances,
            'pixelsize_x': cube._pxl_size_x,
            'pixelsize_y': cube._pxl_size_y
        },
        # 使用哪种运动学场标志
        'using_emission': using_emission
    }
    
    # 添加恒星物理参数
    if stellar_pop_params is not None:
        vnb_results['stellar_population'] = stellar_pop_params
    
    # 添加气体运动学参数
    if gas_vel is not None and gas_disp is not None:
        vnb_results['gas_kinematics'] = {
            'velocity': gas_vel,
            'dispersion': gas_disp
        }
    
    # 添加发射线参数
    if emission_result is not None:
        emission_params = {}
        
        # 添加发射线通量
        if 'emission_flux' in emission_result and emission_result['emission_flux']:
            for line_name, flux_values in emission_result['emission_flux'].items():
                emission_params[f'flux_{line_name}'] = flux_values
                
        # 如果emission_flux是空的，尝试其他可能的键
        if not emission_params:
            for key in emission_result:
                if 'flux' in key.lower() and isinstance(emission_result[key], np.ndarray):
                    if len(emission_result[key]) == n_bins:
                        emission_params[f'flux_{key}'] = emission_result[key]
                        logger.info(f"Using flux from key: {key}")
        
        # 添加线比
        try:
            # 尝试计算线比
            line_ratios = {}
            
            # 查找Hbeta和OIII的键
            hb_key = None
            oiii_key = None
            
            for key in emission_params.keys():
                if 'Hbeta' in key:
                    hb_key = key
                elif '[OIII]5007' in key or 'OIII_5007' in key:
                    oiii_key = key
            
            # 如果找到两个键，计算线比
            if hb_key is not None and oiii_key is not None:
                hb_flux = emission_params[hb_key]
                oiii_flux = emission_params[oiii_key]
                
                # 计算比率，确保除数不为零
                valid_mask = ~np.isnan(hb_flux) & ~np.isnan(oiii_flux) & (hb_flux > 0)
                
                if np.any(valid_mask):
                    oiii_hb = np.full_like(hb_flux, np.nan)
                    oiii_hb[valid_mask] = oiii_flux[valid_mask] / hb_flux[valid_mask]
                    line_ratios['OIII_Hb'] = oiii_hb
                    logger.info("Calculated OIII/Hb line ratio")
            
            if line_ratios:
                emission_params['line_ratios'] = line_ratios
        except Exception as e:
            logger.warning(f"Error calculating line ratios: {e}")
        
        # 从原始数据中提取气体最佳拟合光谱
        if 'gas_bestfit' in emission_result:
            emission_params['gas_bestfit'] = emission_result['gas_bestfit']
        
        # 只有在有实际数据时才添加
        if emission_params:
            vnb_results['emission'] = emission_params
    
    # 添加谱指数
    if indices_result is not None:
        vnb_results['indices'] = indices_result
    
    # 添加bin光谱信息，用于可视化
    bin_spectra_info = {}
    try:
        if hasattr(cube, '_bin_spectra') and cube._bin_spectra is not None:
            bin_spectra_info['spectra'] = cube._bin_spectra
        if hasattr(cube, '_bin_errors') and cube._bin_errors is not None:
            bin_spectra_info['errors'] = cube._bin_errors
        if bin_bestfit is not None:
            bin_spectra_info['bestfit'] = bin_bestfit
        if bin_optimal_tmpls is not None:
            bin_spectra_info['optimal_tmpls'] = bin_optimal_tmpls
        
        if bin_spectra_info:
            vnb_results['bin_spectra'] = bin_spectra_info
    except Exception as e:
        logger.warning(f"Could not save bin spectra information: {e}")
    
    # 保存结果
    save_results_to_npz(
        output_file=output_dir / f"{galaxy_name}_VNB_results.npz",
        data_dict=vnb_results
    )
    
    # 创建可视化
    if not args.no_plots:
        create_vnb_plots(args, cube, vnb_results, galaxy_name, bin_bestfit, bin_optimal_tmpls)
        # 创建径向剖面图
        create_radial_profile_plots(vnb_results, plots_dir=output_dir / 'plots', galaxy_name=galaxy_name, analysis_type="VNB")
    
    logger.info("Voronoi binning analysis completed")
    return vnb_results


def create_vnb_plots(args, cube, vnb_results, galaxy_name, bin_bestfit, bin_optimal_tmpls):
    """
    为Voronoi分bin结果创建可视化图
    
    Parameters
    ----------
    args : argparse.Namespace
        命令行参数
    cube : MUSECube
        MUSE数据立方体对象
    vnb_results : dict
        分析结果
    galaxy_name : str
        星系名称（用于文件命名）
    bin_bestfit : ndarray
        最佳拟合谱（仅用于绘图）
    bin_optimal_tmpls : ndarray
        最优模板（仅用于绘图）
    """
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # 提取分bin信息
    bin_nums = vnb_results['binning']['bin_nums']
    bin_vel = vnb_results['stellar_kinematics']['velocity']
    bin_disp = vnb_results['stellar_kinematics']['dispersion']
    
    # 创建bin地图
    try:
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(bin_nums, origin='lower', cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Bin Number')
        ax.set_title('Voronoi Binning Map')
        fig.savefig(plots_dir / f"{galaxy_name}_VNB_bin_map.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating bin map: {e}")
        plt.close('all')
    
    # 创建运动学图
    try:
        # 为每个bin着色创建速度图
        vel_map = np.full_like(bin_nums, np.nan, dtype=float)
        disp_map = np.full_like(bin_nums, np.nan, dtype=float)
        
        # 填充每个bin的值
        for i in range(len(bin_vel)):
            if np.isfinite(bin_vel[i]):
                vel_map[bin_nums == i] = bin_vel[i]
            if np.isfinite(bin_disp[i]):
                disp_map[bin_nums == i] = bin_disp[i]
        
        # 创建速度图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 计算速度范围
        valid_vel = vel_map[~np.isnan(vel_map)]
        vmin_vel = np.percentile(valid_vel, 5)
        vmax_vel = np.percentile(valid_vel, 95)
        
        im1 = ax1.imshow(vel_map, origin='lower', cmap='RdBu_r', 
                      vmin=vmin_vel, vmax=vmax_vel,
                      aspect='auto' if not args.equal_aspect else 1)
        plt.colorbar(im1, ax=ax1, label='Velocity (km/s)')
        ax1.set_title('Stellar Velocity Field')
        
        # 计算弥散范围
        valid_disp = disp_map[~np.isnan(disp_map)]
        vmin_disp = np.percentile(valid_disp, 5)
        vmax_disp = np.percentile(valid_disp, 95)
        
        im2 = ax2.imshow(disp_map, origin='lower', cmap='viridis', 
                       vmin=vmin_disp, vmax=vmax_disp,
                       aspect='auto' if not args.equal_aspect else 1)
        plt.colorbar(im2, ax=ax2, label='Velocity Dispersion (km/s)')
        ax2.set_title('Stellar Velocity Dispersion')
        
        plt.tight_layout()
        fig.savefig(plots_dir / f"{galaxy_name}_VNB_kinematics.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error creating kinematics maps: {e}")
        plt.close('all')
    
    # 创建恒星物理参数图
    if 'stellar_population' in vnb_results:
        try:
            stellar_pop = vnb_results['stellar_population']
            
            # 检查哪些参数是可用的
            available_params = []
            for param in ['log_age', 'metallicity']:
                if param in stellar_pop and np.any(np.isfinite(stellar_pop[param])):
                    available_params.append(param)
            
            if available_params:
                n_params = len(available_params)
                fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 5))
                if n_params == 1:
                    axes = [axes]
                
                for i, param in enumerate(available_params):
                    # 创建参数图
                    param_map = np.full_like(bin_nums, np.nan, dtype=float)
                    
                    # 填充每个bin的值
                    for j in range(len(stellar_pop[param])):
                        if np.isfinite(stellar_pop[param][j]):
                            param_map[bin_nums == j] = stellar_pop[param][j]
                    
                    # 计算范围
                    valid_values = param_map[~np.isnan(param_map)]
                    vmin = np.percentile(valid_values, 5)
                    vmax = np.percentile(valid_values, 95)
                    
                    # 对于年龄，转换为Gyr
                    title = param
                    if param == 'log_age':
                        title = 'Log Age (yr)'
                    elif param == 'metallicity':
                        title = 'Metallicity [Z/H]'
                    
                    im = axes[i].imshow(param_map, origin='lower', cmap='plasma', 
                                     vmin=vmin, vmax=vmax,
                                     aspect='auto' if not args.equal_aspect else 1)
                    plt.colorbar(im, ax=axes[i], label=title)
                    axes[i].set_title(f'Stellar {title}')
                
                plt.tight_layout()
                fig.savefig(plots_dir / f"{galaxy_name}_VNB_stellar_pop.png", dpi=150)
                plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating stellar population maps: {e}")
            plt.close('all')
    
    # 创建气体运动学图
    if 'gas_kinematics' in vnb_results:
        try:
            gas_vel = vnb_results['gas_kinematics']['velocity']
            gas_disp = vnb_results['gas_kinematics']['dispersion']
            
            # 为每个bin着色创建气体速度图
            gas_vel_map = np.full_like(bin_nums, np.nan, dtype=float)
            gas_disp_map = np.full_like(bin_nums, np.nan, dtype=float)
            
            # 填充每个bin的值
            for i in range(len(gas_vel)):
                if np.isfinite(gas_vel[i]):
                    gas_vel_map[bin_nums == i] = gas_vel[i]
                if np.isfinite(gas_disp[i]):
                    gas_disp_map[bin_nums == i] = gas_disp[i]
            
            # 创建气体速度图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # 计算速度范围
            valid_vel = gas_vel_map[~np.isnan(gas_vel_map)]
            if len(valid_vel) > 0:
                vmin_vel = np.percentile(valid_vel, 5)
                vmax_vel = np.percentile(valid_vel, 95)
                
                im1 = ax1.imshow(gas_vel_map, origin='lower', cmap='RdBu_r', 
                              vmin=vmin_vel, vmax=vmax_vel,
                              aspect='auto' if not args.equal_aspect else 1)
                plt.colorbar(im1, ax=ax1, label='Velocity (km/s)')
                ax1.set_title('Gas Velocity Field')
            else:
                ax1.text(0.5, 0.5, 'No Valid Data', horizontalalignment='center',
                      verticalalignment='center', transform=ax1.transAxes)
                ax1.set_title('Gas Velocity Field (No Data)')
            
            # 计算弥散范围
            valid_disp = gas_disp_map[~np.isnan(gas_disp_map)]
            if len(valid_disp) > 0:
                vmin_disp = np.percentile(valid_disp, 5)
                vmax_disp = np.percentile(valid_disp, 95)
                
                im2 = ax2.imshow(gas_disp_map, origin='lower', cmap='viridis', 
                               vmin=vmin_disp, vmax=vmax_disp,
                               aspect='auto' if not args.equal_aspect else 1)
                plt.colorbar(im2, ax=ax2, label='Velocity Dispersion (km/s)')
                ax2.set_title('Gas Velocity Dispersion')
            else:
                ax2.text(0.5, 0.5, 'No Valid Data', horizontalalignment='center',
                      verticalalignment='center', transform=ax2.transAxes)
                ax2.set_title('Gas Velocity Dispersion (No Data)')
            
            plt.tight_layout()
            fig.savefig(plots_dir / f"{galaxy_name}_VNB_gas_kinematics.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.error(f"Error creating gas kinematics maps: {e}")
            plt.close('all')
    
    # 创建发射线通量图
    if 'emission' in vnb_results:
        try:
            emission_params = vnb_results['emission']
            
            # 找出所有发射线通量
            flux_keys = [k for k in emission_params.keys() if k.startswith('flux_')]
            
            for flux_key in flux_keys:
                try:
                    line_name = flux_key[5:]  # 去掉'flux_'前缀
                    flux_values = emission_params[flux_key]
                    
                    # 创建通量图
                    flux_map = np.full_like(bin_nums, np.nan, dtype=float)
                    
                    # 填充每个bin的值
                    for i in range(len(flux_values)):
                        if np.isfinite(flux_values[i]) and flux_values[i] > 0:
                            flux_map[bin_nums == i] = flux_values[i]
                    
                    # 计算通量范围（对数尺度）
                    valid_flux = flux_map[~np.isnan(flux_map) & (flux_map > 0)]
                    if len(valid_flux) > 0:
                        log_flux = np.log10(valid_flux)
                        vmin = np.percentile(log_flux, 5)
                        vmax = np.percentile(log_flux, 95)
                        
                        fig, ax = plt.subplots(figsize=(8, 7))
                        im = ax.imshow(np.log10(np.maximum(flux_map, 1e-20)), origin='lower',
                                    cmap='inferno', vmin=vmin, vmax=vmax,
                                    aspect='auto' if not args.equal_aspect else 1)
                        plt.colorbar(im, ax=ax, label='Log Flux')
                        ax.set_title(f'{line_name} Flux')
                        
                        fig.savefig(plots_dir / f"{galaxy_name}_VNB_{line_name}_flux.png", dpi=150)
                        plt.close(fig)
                    else:
                        logger.warning(f"No valid flux values for {line_name}")
                except Exception as e:
                    logger.error(f"Error creating flux map for {flux_key}: {e}")
                    plt.close('all')
            
            # 创建线比图
            if 'line_ratios' in emission_params:
                for ratio_name, ratio_values in emission_params['line_ratios'].items():
                    try:
                        # 创建比值图
                        ratio_map = np.full_like(bin_nums, np.nan, dtype=float)
                        
                        # 填充每个bin的值
                        for i in range(len(ratio_values)):
                            if np.isfinite(ratio_values[i]) and ratio_values[i] > 0:
                                ratio_map[bin_nums == i] = ratio_values[i]
                        
                        # 计算比值范围（对数尺度）
                        valid_ratio = ratio_map[~np.isnan(ratio_map) & (ratio_map > 0)]
                        if len(valid_ratio) > 0:
                            log_ratio = np.log10(valid_ratio)
                            vmin = np.percentile(log_ratio, 5)
                            vmax = np.percentile(log_ratio, 95)
                            
                            fig, ax = plt.subplots(figsize=(8, 7))
                            im = ax.imshow(np.log10(np.maximum(ratio_map, 1e-20)), origin='lower',
                                        cmap='viridis', vmin=vmin, vmax=vmax,
                                        aspect='auto' if not args.equal_aspect else 1)
                            plt.colorbar(im, ax=ax, label='Log Ratio')
                            ax.set_title(f'{ratio_name} Ratio')
                            
                            fig.savefig(plots_dir / f"{galaxy_name}_VNB_{ratio_name}_ratio.png", dpi=150)
                            plt.close(fig)
                        else:
                            logger.warning(f"No valid ratio values for {ratio_name}")
                    except Exception as e:
                        logger.error(f"Error creating ratio map for {ratio_name}: {e}")
                        plt.close('all')
        except Exception as e:
            logger.error(f"Error creating emission maps: {e}")
            plt.close('all')
    
    # 创建谱指数图
    if 'indices' in vnb_results:
        try:
            indices = vnb_results['indices']
            
            for index_name, index_values in indices.items():
                # 创建指数图
                index_map = np.full_like(bin_nums, np.nan, dtype=float)
                
                # 填充每个bin的值
                for i in range(len(index_values)):
                    if np.isfinite(index_values[i]):
                        index_map[bin_nums == i] = index_values[i]
                
                # 计算指数范围
                valid_index = index_map[~np.isnan(index_map)]
                if len(valid_index) > 0:
                    vmin = np.percentile(valid_index, 5)
                    vmax = np.percentile(valid_index, 95)
                    
                    fig, ax = plt.subplots(figsize=(8, 7))
                    im = ax.imshow(index_map, origin='lower', cmap='viridis', 
                                 vmin=vmin, vmax=vmax,
                                 aspect='auto' if not args.equal_aspect else 1)
                    plt.colorbar(im, ax=ax, label='Index Value')
                    ax.set_title(f'{index_name} Index')
                    
                    fig.savefig(plots_dir / f"{galaxy_name}_VNB_{index_name}_index.png", dpi=150)
                    plt.close(fig)
                else:
                    logger.warning(f"No valid index values for {index_name}")
        except Exception as e:
            logger.error(f"Error creating index maps: {e}")
            plt.close('all')
    
    # 创建样本bin的光谱拟合图
    try:
        # 选择几个代表性bin
        bin_dists = vnb_results['distance']['bin_distances']
        valid_bins = np.where(np.isfinite(bin_vel) & np.isfinite(bin_dists))[0]
        
        if len(valid_bins) > 0:
            # 选择3个bin：中心bin，中间距离bin，和外部bin
            sorted_bins = valid_bins[np.argsort(bin_dists[valid_bins])]
            selected_bins = [sorted_bins[0]]  # 中心bin
            
            if len(sorted_bins) >= 3:
                selected_bins.append(sorted_bins[len(sorted_bins) // 2])  # 中间bin
                selected_bins.append(sorted_bins[-1])  # 最外部bin
            else:
                selected_bins.extend(sorted_bins[1:])  # 添加剩余的bin
            
            # 为每个选定的bin创建拟合图
            for i, bin_idx in enumerate(selected_bins):
                try:
                    bin_spec = bin_bestfit[bin_idx] if bin_bestfit is not None else None
                    bin_template = bin_optimal_tmpls[bin_idx] if bin_optimal_tmpls is not None else None
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # 获取原始光谱
                    actual_spectrum = cube._bin_spectra[:, bin_idx] if hasattr(cube, '_bin_spectra') else None
                    
                    if actual_spectrum is not None and bin_spec is not None:
                        # 绘制原始光谱
                        ax.plot(cube._lambda_gal, actual_spectrum, 'k-', label='Observed')
                        
                        # 绘制拟合
                        ax.plot(cube._lambda_gal, bin_spec, 'r-', label='Best Fit')
                        
                        # 绘制残差
                        residual = actual_spectrum - bin_spec
                        offset = np.min(actual_spectrum) - 0.2 * (np.max(actual_spectrum) - np.min(actual_spectrum))
                        ax.plot(cube._lambda_gal, residual + offset, 'b-', label='Residual')
                        
                        ax.set_xlabel('Wavelength (Å)')
                        ax.set_ylabel('Flux')
                        ax.set_title(f'Bin {bin_idx} Fit - Distance: {bin_dists[bin_idx]:.2f} arcsec')
                        ax.legend()
                        
                        fig.savefig(plots_dir / f"{galaxy_name}_VNB_bin{bin_idx}_fit.png", dpi=150)
                    plt.close(fig)
                except Exception as e:
                    logger.error(f"Error creating fit plot for bin {bin_idx}: {e}")
                    plt.close('all')
        else:
            logger.warning("No valid bins found for fit plots")
    except Exception as e:
        logger.error(f"Error creating bin fit plots: {e}")
        plt.close('all')


def create_radial_profile_plots(results, plots_dir, galaxy_name, analysis_type="VNB"):
    """
    创建径向分布图，展示参数随半径的变化
    
    Parameters
    ----------
    results : dict
        分析结果字典
    plots_dir : Path
        保存图表的目录
    galaxy_name : str
        星系名称
    analysis_type : str
        分析类型 ("P2P", "VNB", "RDB")
    """
    # 提取距离信息
    if analysis_type == "P2P":
        # 对于P2P，需要径向平均
        distance_field = results['distance']['field']
        # 把nan值替换为大值，以便后面能够忽略它们
        valid_mask = ~np.isnan(distance_field)
        
        # 创建距离bins
        max_dist = np.nanmax(distance_field)
        r_bins = np.linspace(0, max_dist, 15)
        r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
        
        # 准备存储径向参数的字典
        radial_params = {}
        
        # 处理恒星运动学参数
        if 'stellar_kinematics' in results:
            velocity = results['stellar_kinematics']['velocity_field']
            dispersion = results['stellar_kinematics']['dispersion_field']
            
            # 计算每个径向bin的平均值
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
        
        # 处理恒星物理参数
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
        
        # 处理发射线参数
        if 'emission' in results:
            # 处理线通量
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
            
            # 处理线比
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
        
        # 处理谱指数
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
    
    else:  # 对于VNB和RDB，直接使用bin的距离和参数
        if 'distance' not in results or 'bin_distances' not in results['distance']:
            logger.warning(f"No distance information in {analysis_type} results")
            return
        
        r_centers = results['distance']['bin_distances']
        
        # 准备存储径向参数的字典
        radial_params = {}
        
        # 处理恒星运动学参数
        if 'stellar_kinematics' in results:
            velocity = results['stellar_kinematics']['velocity']
            dispersion = results['stellar_kinematics']['dispersion']
            
            # 对于VNB和RDB，没有直接的误差估计，将误差设为0
            zero_err = np.zeros_like(r_centers)
            
            radial_params['velocity'] = (r_centers, velocity, zero_err)
            radial_params['dispersion'] = (r_centers, dispersion, zero_err)
        
        # 处理恒星物理参数
        if 'stellar_population' in results:
            for param_name, param_values in results['stellar_population'].items():
                zero_err = np.zeros_like(param_values)
                radial_params[param_name] = (r_centers, param_values, zero_err)
        
        # 处理发射线参数
        if 'emission' in results:
            # 处理线通量
            for key, flux_values in results['emission'].items():
                if key.startswith('flux_') and isinstance(flux_values, np.ndarray):
                    zero_err = np.zeros_like(flux_values)
                    radial_params[key] = (r_centers, flux_values, zero_err)
            
            # 处理线比
            if 'line_ratios' in results['emission']:
                for ratio_name, ratio_values in results['emission']['line_ratios'].items():
                    zero_err = np.zeros_like(ratio_values)
                    radial_params[f"ratio_{ratio_name}"] = (r_centers, ratio_values, zero_err)
        
        # 处理谱指数
        if 'indices' in results:
            for index_name, index_values in results['indices'].items():
                zero_err = np.zeros_like(index_values)
                radial_params[f"index_{index_name}"] = (r_centers, index_values, zero_err)
    
    # 创建径向分布图
    # 1. 恒星运动学图
    if 'velocity' in radial_params and 'dispersion' in radial_params:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 速度分布
            r, vel, vel_err = radial_params['velocity']
            ax1.errorbar(r, vel, yerr=vel_err, fmt='o-', capsize=3)
            ax1.set_xlabel('Radius (arcsec)')
            ax1.set_ylabel('Velocity (km/s)')
            ax1.set_title('Stellar Velocity Profile')
            ax1.grid(True, alpha=0.3)
            
            # 弥散分布
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
    
    # 2. 恒星物理参数图
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
                
                # 对于年龄，转换为Gyr
                if param_name == 'age':
                    values = values * 1e-9  # 转换为Gyr
                    errors = errors * 1e-9  # 转换为Gyr
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
    
    # 3. 发射线通量图
    flux_params = [p for p in radial_params if p.startswith('flux_')]
    
    if flux_params:
        try:
            n_plots = min(len(flux_params), 3)  # 最多显示3个线
            fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
            
            for i, param_name in enumerate(flux_params[:n_plots]):
                r, values, errors = radial_params[param_name]
                line_name = param_name[5:]  # 去掉'flux_'前缀
                
                axes[i].errorbar(r, values, yerr=errors, fmt='o-', capsize=3)
                axes[i].set_xlabel('Radius (arcsec)')
                axes[i].set_ylabel('Flux')
                axes[i].set_title(f'{line_name} Flux Profile')
                axes[i].grid(True, alpha=0.3)
                
                # 尝试对数坐标
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
    
    # 4. 线比图
    ratio_params = [p for p in radial_params if p.startswith('ratio_')]
    
    if ratio_params:
        try:
            n_plots = len(ratio_params)
            fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
            
            for i, param_name in enumerate(ratio_params):
                r, values, errors = radial_params[param_name]
                ratio_name = param_name[6:]  # 去掉'ratio_'前缀
                
                axes[i].errorbar(r, values, yerr=errors, fmt='o-', capsize=3)
                axes[i].set_xlabel('Radius (arcsec)')
                axes[i].set_ylabel('Ratio')
                axes[i].set_title(f'{ratio_name} Ratio Profile')
                axes[i].grid(True, alpha=0.3)
                
                # 尝试对数坐标
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
    
    # 5. 谱指数图
    index_params = [p for p in radial_params if p.startswith('index_')]
    
    if index_params:
        try:
            n_plots = min(len(index_params), 3)  # 最多显示3个指数
            fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
            
            for i, param_name in enumerate(index_params[:n_plots]):
                r, values, errors = radial_params[param_name]
                index_name = param_name[6:]  # 去掉'index_'前缀
                
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