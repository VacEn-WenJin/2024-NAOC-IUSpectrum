"""
ISAPC - IFU Spectrum Analysis Pipeline Cluster
主程序和命令行接口
"""
import os
import sys
import argparse
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import warnings

from muse import MUSECube
from utils.parallel import ParallelTqdm
from joblib import delayed
import spectral_indices
import binning
import galaxy_params
import visualization

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='ISAPC - IFU Spectrum Analysis Pipeline Cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本参数
    parser.add_argument('filename', help='MUSE数据立方体文件路径')
    parser.add_argument('--redshift', type=float, required=True, help='星系红移')
    parser.add_argument('--output-dir', default='output', help='输出目录')
    parser.add_argument('--template', default=None, help='恒星模板文件路径')
    
    # 波长设置
    parser.add_argument('--wvl-range', type=float, nargs=2, default=[4822, 5212],
                      help='要分析的波长范围(Å)')
    
    # 分析模式
    parser.add_argument('--mode', choices=['P2P', 'VNB', 'RDB', 'ALL'], default='P2P',
                      help='分析模式: 像素级(P2P)、Voronoi宾化(VNB)、径向宾化(RDB)、全部(ALL)')
    
    # 运行配置
    parser.add_argument('--n-jobs', type=int, default=-1, help='并行任务数(-1表示使用所有CPU)')
    parser.add_argument('--no-plots', action='store_true', help='禁用绘图')
    
    # 拟合参数
    parser.add_argument('--vel-init', type=float, default=0, help='初始速度猜测(km/s)')
    parser.add_argument('--sigma-init', type=float, default=40, help='初始弥散猜测(km/s)')
    parser.add_argument('--no-emission', action='store_true', help='不拟合发射线')
    
    # Voronoi宾化参数
    vnb_group = parser.add_argument_group('Voronoi宾化选项')
    vnb_group.add_argument('--target-snr', type=float, default=20, help='目标信噪比')
    
    # 径向宾化参数
    rdb_group = parser.add_argument_group('径向宾化选项')
    rdb_group.add_argument('--n-rings', type=int, default=10, help='径向环数')
    rdb_group.add_argument('--center-x', type=float, help='中心X坐标')
    rdb_group.add_argument('--center-y', type=float, help='中心Y坐标')
    rdb_group.add_argument('--pa', type=float, default=0.0, help='位置角(度)')
    rdb_group.add_argument('--ellipticity', type=float, default=0.0, help='椭率(0-1)')
    rdb_group.add_argument('--log-spacing', action='store_true', help='使用对数间隔')
    
    return parser


def run_p2p_analysis(args, cube):
    """运行像素级分析"""
    logger.info("开始像素级分析...")
    start_time = time.time()
    
    # 拟合恒星连续谱
    result = cube.fit_spectra(
        template_filename=args.template,
        ppxf_vel_init=args.vel_init,
        ppxf_vel_disp_init=args.sigma_init,
        n_jobs=args.n_jobs
    )
    
    velocity_field, dispersion_field, bestfit_field, optimal_tmpls, apoly = result
    
    logger.info(f"恒星成分拟合完成，用时 {time.time() - start_time:.1f} 秒")
    
    # 拟合发射线
    if not args.no_emission:
        start_time = time.time()
        emission_result = cube.fit_emission_lines(n_jobs=args.n_jobs)
        logger.info(f"发射线拟合完成，用时 {time.time() - start_time:.1f} 秒")
    
    # 计算谱指数
    start_time = time.time()
    indices_result = spectral_indices.calculate_indices_cube(
        wavelength=cube._lambda_gal,
        cube=cube._bestfit_field,
        template_wave=cube._lambda_gal,
        template_cube=cube._bestfit_field,
        velocity_field=velocity_field,
        n_jobs=args.n_jobs
    )
    
    logger.info(f"谱指数计算完成，用时 {time.time() - start_time:.1f} 秒")
    
    # 计算星系参数
    start_time = time.time()
    gp = galaxy_params.GalaxyParameters(
        velocity_field=velocity_field,
        dispersion_field=dispersion_field,
        pixelsize=cube._pxl_size_x
    )
    
    rotation_result = gp.fit_rotation_curve()
    kinematics_result = gp.calculate_kinematics()
    
    logger.info(f"星系参数计算完成，用时 {time.time() - start_time:.1f} 秒")
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    galaxy_name = Path(args.filename).stem
    
    # 创建结果字典
    p2p_results = {
        'velocity_field': velocity_field,
        'dispersion_field': dispersion_field,
        'bestfit_field': bestfit_field,
        'optimal_tmpls': optimal_tmpls,
        'indices': indices_result,
        'kinematics': {**rotation_result, **kinematics_result}
    }
    
    if not args.no_emission:
        p2p_results['emission'] = emission_result
    
    # 保存为NPZ文件
    np.savez(
        output_dir / f"{galaxy_name}_P2P_results.npz",
        **p2p_results
    )
    
    # 创建可视化
    if not args.no_plots:
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # 创建动力学图
        fig = visualization.plot_kinematics_summary(
            velocity_field=velocity_field,
            dispersion_field=dispersion_field,
            rotation_curve=rotation_result['rotation_curve'],
            params={**rotation_result, **kinematics_result}
        )
        
        fig.savefig(plots_dir / f"{galaxy_name}_P2P_kinematics.png", dpi=150)
        plt.close(fig)
        
        # 为几个样本像素创建光谱拟合图
        ny, nx = velocity_field.shape
        center_y, center_x = ny // 2, nx // 2
        
        sample_positions = [
            (center_x, center_y),  # 中心
            (center_x + nx//4, center_y),  # 右侧
            (center_x, center_y + ny//4),  # 上方
            (center_x - nx//4, center_y - ny//4)  # 左下
        ]
        
        for i, (x, y) in enumerate(sample_positions):
            if x < 0 or y < 0 or x >= nx or y >= ny:
                continue
                
            idx = y * nx + x
            
            # 获取数据
            observed = cube._spectra[:, idx]
            model = bestfit_field[:, y, x]
            
            gas_comp = None
            if not args.no_emission and 'gas_bestfit_field' in emission_result:
                gas_comp = emission_result['gas_bestfit_field'][:, y, x]
            
            stellar_comp = model.copy()
            if gas_comp is not None:
                stellar_comp -= gas_comp
            
            # 创建光谱拟合图
            fig, axes = visualization.plot_spectrum_fit(
                wavelength=cube._lambda_gal,
                observed_flux=observed,
                model_flux=model,
                gas_flux=gas_comp,
                stellar_flux=stellar_comp,
                title=f"Pixel ({x}, {y})"
            )
            
            fig.savefig(plots_dir / f"{galaxy_name}_P2P_spectrum_{i}.png", dpi=150)
            plt.close(fig)
        
        # 创建谱指数图
        for name, index_map in indices_result.items():
            fig, ax = plt.subplots(figsize=(8, 7))
            
            vmin = np.nanpercentile(index_map, 5)
            vmax = np.nanpercentile(index_map, 95)
            
            im = ax.imshow(index_map, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{name} Index")
            
            fig.savefig(plots_dir / f"{galaxy_name}_P2P_{name}_index.png", dpi=150)
            plt.close(fig)
    
    logger.info("像素级分析完成")
    return p2p_results


def run_vnb_analysis(args, cube, p2p_results=None):
    """运行Voronoi宾化分析"""
    logger.info("开始Voronoi宾化分析...")
    start_time = time.time()
    
    # 准备数据
    ny, nx = cube._cube_data.shape[1:]
    
    # 创建VoronoiBinning实例
    vnb = binning.VoronoiBinning(
        x=cube.x, 
        y=cube.y,
        signal=cube._signal,
        noise=cube._noise,
        wavelength=cube._lambda_gal,
        spectra=cube._spectra,
        shape=(ny, nx),
        pixelsize=cube._pxl_size_x
    )
    
    # 执行宾化
    bin_result = vnb.compute_bins(target_snr=args.target_snr)
    logger.info(f"创建了{bin_result['n_bins']}个Voronoi宾")
    
    # 提取宾化光谱
    velocity_field = None
    if p2p_results is not None:
        velocity_field = p2p_results['velocity_field']
    
    bin_spectra = vnb.extract_binned_spectra(bin_result['bin_map'], velocity_field)
    logger.info(f"提取了{len(bin_spectra)}个宾光谱")
    
    # 拟合宾光谱
    bin_velocity = np.full((ny, nx), np.nan)
    bin_dispersion = np.full((ny, nx), np.nan)
    bin_results = {}
    
    # 定义拟合函数
    def fit_bin(bin_idx):
        bin_spectrum = bin_spectra[bin_idx]
        
        # 创建临时数据立方体进行拟合
        temp_cube = MUSECube(
            filename=args.filename,
            redshift=args.redshift,
            wvl_air_angstrom_range=args.wvl_range
        )
        
        # 替换为宾光谱
        temp_cube._spectra = bin_spectrum.reshape(-1, 1)
        
        # 拟合恒星成分
        result = temp_cube.fit_spectra(
            template_filename=args.template,
            ppxf_vel_init=args.vel_init,
            ppxf_vel_disp_init=args.sigma_init,
            n_jobs=1  # 单宾拟合
        )
        
        # 添加发射线拟合
        if not args.no_emission:
            emission_result = temp_cube.fit_emission_lines(n_jobs=1)
        else:
            emission_result = None
        
        # 计算谱指数
        indices = spectral_indices.LineIndexCalculator(
            wave=temp_cube._lambda_gal,
            flux=bin_spectrum,
            fit_wave=temp_cube._lambda_gal,
            fit_flux=result[2][:, 0, 0]
        ).calculate_all_indices()
        
        # 返回结果
        fit_result = {
            'velocity': result[0][0, 0],  # 取第一个单元格的值
            'dispersion': result[1][0, 0],
            'bestfit': result[2][:, 0, 0],
            'optimal_tmpl': result[3][:, 0, 0],
            'indices': indices
        }
        
        if emission_result is not None:
            fit_result['emission'] = {}
            for name, flux_map in emission_result['emission_flux'].items():
                fit_result['emission'][name] = flux_map[0, 0]
        
        return bin_idx, fit_result
    
    # 并行拟合宾
    bin_idx_list = list(bin_spectra.keys())
    fit_results = ParallelTqdm(
        n_jobs=args.n_jobs, desc='拟合Voronoi宾', total_tasks=len(bin_idx_list)
    )(delayed(fit_bin)(bin_idx) for bin_idx in bin_idx_list)
    
    # 处理结果
    for bin_idx, result in fit_results:
        bin_results[bin_idx] = result
        
        # 更新宾映射
        bin_mask = (bin_result['bin_map'] == bin_idx)
        bin_velocity[bin_mask] = result['velocity']
        bin_dispersion[bin_mask] = result['dispersion']
    
    logger.info(f"宾拟合完成，用时 {time.time() - start_time:.1f} 秒")
    
    # 计算星系参数
    gp = galaxy_params.GalaxyParameters(
        velocity_field=bin_velocity,
        dispersion_field=bin_dispersion,
        pixelsize=cube._pxl_size_x
    )
    
    rotation_result = gp.fit_rotation_curve()
    kinematics_result = gp.calculate_kinematics()
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    galaxy_name = Path(args.filename).stem
    
    # 创建结果字典
    vnb_results = {
        'bin_map': bin_result['bin_map'],
        'bin_snr': bin_result['bin_snr'],
        'velocity_field': bin_velocity,
        'dispersion_field': bin_dispersion,
        'bin_results': bin_results,
        'kinematics': {**rotation_result, **kinematics_result}
    }
    
    # 保存为NPZ文件
    np.savez(
        output_dir / f"{galaxy_name}_VNB_results.npz",
        **vnb_results
    )
    
    # 创建可视化
    if not args.no_plots:
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # 创建宾图
        fig, ax = visualization.plot_binning_map(
            bin_map=bin_result['bin_map'],
            title=f"Voronoi Binning (SNR={args.target_snr})"
        )
        
        fig.savefig(plots_dir / f"{galaxy_name}_VNB_binning.png", dpi=150)
        plt.close(fig)
        
        # 创建动力学图
        fig = visualization.plot_kinematics_summary(
            velocity_field=bin_velocity,
            dispersion_field=bin_dispersion,
            bin_map=bin_result['bin_map'],
            rotation_curve=rotation_result['rotation_curve'],
            params={**rotation_result, **kinematics_result}
        )
        
        fig.savefig(plots_dir / f"{galaxy_name}_VNB_kinematics.png", dpi=150)
        plt.close(fig)
        
        # 为几个样本宾创建光谱拟合图
        sample_bins = list(bin_results.keys())
        if len(sample_bins) > 4:
            sample_bins = sample_bins[:4]
        
        for i, bin_idx in enumerate(sample_bins):
            result = bin_results[bin_idx]
            
            # 获取宾光谱
            observed = bin_spectra[bin_idx]
            model = result['bestfit']
            
            # 获取发射线成分
            gas_comp = None
            if 'emission' in result:
                # 简化的发射线处理 - 实际应从发射线拟合中恢复原始模板
                gas_flux_sum = sum(result['emission'].values())
                if gas_flux_sum > 0:
                    gas_comp = np.ones_like(model) * gas_flux_sum * 0.01
            
            # 创建光谱拟合图
            fig, axes = visualization.plot_spectrum_fit(
                wavelength=cube._lambda_gal,
                observed_flux=observed,
                model_flux=model,
                gas_flux=gas_comp,
                stellar_flux=model - (gas_comp if gas_comp is not None else 0),
                title=f"Bin {bin_idx}"
            )
            
            fig.savefig(plots_dir / f"{galaxy_name}_VNB_spectrum_{bin_idx}.png", dpi=150)
            plt.close(fig)
    
    logger.info("Voronoi宾化分析完成")
    return vnb_results


def run_rdb_analysis(args, cube, p2p_results=None):
    """运行径向宾化分析"""
    logger.info("开始径向宾化分析...")
    start_time = time.time()
    
    # 准备数据
    ny, nx = cube._cube_data.shape[1:]
    
    # 创建RadialBinning实例
    rdb = binning.RadialBinning(
        x=cube.x, 
        y=cube.y,
        signal=cube._signal,
        noise=cube._noise,
        wavelength=cube._lambda_gal,
        spectra=cube._spectra,
        shape=(ny, nx)
    )
    
    # 执行宾化
    bin_result = rdb.compute_bins(
        n_bins=args.n_rings,
        center_x=args.center_x,
        center_y=args.center_y,
        pa=args.pa,
        ellipticity=args.ellipticity,
        log_spacing=args.log_spacing
    )
    
    logger.info(f"创建了{args.n_rings}个径向宾")
    
    # 提取宾化光谱
    velocity_field = None
    if p2p_results is not None:
        velocity_field = p2p_results['velocity_field']
    
    bin_spectra = rdb.extract_binned_spectra(bin_result['bin_map'], velocity_field)
    logger.info(f"提取了{len(bin_spectra)}个宾光谱")
    
    # 拟合宾光谱
    bin_velocity = np.full((ny, nx), np.nan)
    bin_dispersion = np.full((ny, nx), np.nan)
    bin_results = {}
    
    # 使用与VNB相同的拟合函数
    def fit_bin(bin_idx):
        bin_spectrum = bin_spectra[bin_idx]
        
        # 创建临时数据立方体进行拟合
        temp_cube = MUSECube(
            filename=args.filename,
            redshift=args.redshift,
            wvl_air_angstrom_range=args.wvl_range
        )
        
        # 替换为宾光谱
        temp_cube._spectra = bin_spectrum.reshape(-1, 1)
        
        # 拟合恒星成分
        result = temp_cube.fit_spectra(
            template_filename=args.template,
            ppxf_vel_init=args.vel_init,
            ppxf_vel_disp_init=args.sigma_init,
            n_jobs=1  # 单宾拟合
        )
        
        # 添加发射线拟合
        if not args.no_emission:
            emission_result = temp_cube.fit_emission_lines(n_jobs=1)
        else:
            emission_result = None
        
        # 计算谱指数
        indices = spectral_indices.LineIndexCalculator(
            wave=temp_cube._lambda_gal,
            flux=bin_spectrum,
            fit_wave=temp_cube._lambda_gal,
            fit_flux=result[2][:, 0, 0]
        ).calculate_all_indices()
        
        # 返回结果
        fit_result = {
            'velocity': result[0][0, 0],  # 取第一个单元格的值
            'dispersion': result[1][0, 0],
            'bestfit': result[2][:, 0, 0],
            'optimal_tmpl': result[3][:, 0, 0],
            'indices': indices,
            'radius': (bin_result['bin_edges'][bin_idx] + bin_result['bin_edges'][bin_idx+1]) / 2
        }
        
        if emission_result is not None:
            fit_result['emission'] = {}
            for name, flux_map in emission_result['emission_flux'].items():
                fit_result['emission'][name] = flux_map[0, 0]
        
        return bin_idx, fit_result
    
    # 并行拟合宾
    bin_idx_list = list(bin_spectra.keys())
    fit_results = ParallelTqdm(
        n_jobs=args.n_jobs, desc='拟合径向宾', total_tasks=len(bin_idx_list)
    )(delayed(fit_bin)(bin_idx) for bin_idx in bin_idx_list)
    
    # 处理结果
    for bin_idx, result in fit_results:
        bin_results[bin_idx] = result
        
        # 更新宾映射
        bin_mask = (bin_result['bin_map'] == bin_idx)
        bin_velocity[bin_mask] = result['velocity']
        bin_dispersion[bin_mask] = result['dispersion']
    
    logger.info(f"宾拟合完成，用时 {time.time() - start_time:.1f} 秒")
    
    # 构建旋转曲线数据
    rotation_curve = []
    for bin_idx, result in sorted(bin_results.items()):
        rotation_curve.append([result['radius'], result['velocity']])
    
    rotation_curve = np.array(rotation_curve)
    
    # 计算星系参数
    gp = galaxy_params.GalaxyParameters(
        velocity_field=bin_velocity,
        dispersion_field=bin_dispersion,
        pixelsize=cube._pxl_size_x
    )
    
    rotation_result = gp.fit_rotation_curve(
        center_x=args.center_x if args.center_x is not None else nx/2,
        center_y=args.center_y if args.center_y is not None else ny/2,
        pa_initial=args.pa
    )
    kinematics_result = gp.calculate_kinematics()
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    galaxy_name = Path(args.filename).stem
    
    # 创建结果字典
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
    
    # 保存为NPZ文件
    np.savez(
        output_dir / f"{galaxy_name}_RDB_results.npz",
        **rdb_results
    )
    
    # 创建可视化
    if not args.no_plots:
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # 创建径向宾图
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(bin_result['radial_map'], origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax, label='Radius (pixels)')
        ax.set_title(f"Radial Distance Map")
        fig.savefig(plots_dir / f"{galaxy_name}_RDB_distance.png", dpi=150)
        plt.close(fig)
        
        # 创建宾图
        fig, ax = visualization.plot_binning_map(
            bin_map=bin_result['bin_map'],
            title=f"Radial Binning ({args.n_rings} rings)"
        )
        fig.savefig(plots_dir / f"{galaxy_name}_RDB_binning.png", dpi=150)
        plt.close(fig)
        
        # 创建动力学图
        fig = visualization.plot_kinematics_summary(
            velocity_field=bin_velocity,
            dispersion_field=bin_dispersion,
            bin_map=bin_result['bin_map'],
            rotation_curve=rotation_curve,
            params={**rotation_result, **kinematics_result}
        )
        fig.savefig(plots_dir / f"{galaxy_name}_RDB_kinematics.png", dpi=150)
        plt.close(fig)
        
        # 创建旋转曲线图
        fig, ax = visualization.plot_rotation_curve(
            rotation_curve=rotation_curve,
            plot_model=True,
            vmax=rotation_result['vmax'],
            pa=rotation_result['pa'],
            title=f"Rotation Curve"
        )
        fig.savefig(plots_dir / f"{galaxy_name}_RDB_rotation_curve.png", dpi=150)
        plt.close(fig)
        
        # 为几个样本宾创建光谱拟合图
        sample_bins = list(bin_results.keys())
        if len(sample_bins) > 4:
            # 选择均匀间隔的样本
            indices = np.linspace(0, len(sample_bins) - 1, 4).astype(int)
            sample_bins = [sample_bins[i] for i in indices]
        
        for i, bin_idx in enumerate(sample_bins):
            result = bin_results[bin_idx]
            
            # 获取宾光谱
            observed = bin_spectra[bin_idx]
            model = result['bestfit']
            
            # 获取发射线成分
            gas_comp = None
            if 'emission' in result:
                # 简化的发射线处理
                gas_flux_sum = sum(result['emission'].values())
                if gas_flux_sum > 0:
                    gas_comp = np.ones_like(model) * gas_flux_sum * 0.01
            
            # 创建光谱拟合图
            fig, axes = visualization.plot_spectrum_fit(
                wavelength=cube._lambda_gal,
                observed_flux=observed,
                model_flux=model,
                gas_flux=gas_comp,
                stellar_flux=model - (gas_comp if gas_comp is not None else 0),
                title=f"Radial Bin {bin_idx} (r={result['radius']:.1f} px)"
            )
            
            fig.savefig(plots_dir / f"{galaxy_name}_RDB_spectrum_{bin_idx}.png", dpi=150)
            plt.close(fig)
    
    logger.info("径向宾化分析完成")
    return rdb_results


def main():
    """主程序入口"""
    # 解析命令行参数
    parser = setup_parser()
    args = parser.parse_args()
    
    # 设置文件日志
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    galaxy_name = Path(args.filename).stem
    
    log_file = log_dir / f"{galaxy_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)
    
    logger.info(f"ISAPC 分析开始，目标: {args.filename}")
    logger.info(f"参数: 红移={args.redshift}, 波长范围={args.wvl_range}, 模式={args.mode}")
    
    # 读取数据
    try:
        start_time = time.time()
        cube = MUSECube(
            filename=args.filename,
            redshift=args.redshift,
            wvl_air_angstrom_range=tuple(args.wvl_range)
        )
        logger.info(f"数据加载完成，用时 {time.time() - start_time:.1f} 秒")
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        return 1
    
    # 设置模板文件
    if args.template is None:
        # 尝试查找模板
        template_dir = Path('templates')
        if template_dir.exists():
            templates = list(template_dir.glob("*.npz"))
            if templates:
                args.template = str(templates[0])
                logger.info(f"自动选择模板: {args.template}")
            else:
                logger.error("未找到模板文件，请使用--template参数指定")
                return 1
        else:
            logger.error("未找到模板文件，请使用--template参数指定")
            return 1
    
    # 执行分析
    p2p_results = None
    vnb_results = None
    rdb_results = None
    
    try:
        if args.mode in ['P2P', 'ALL']:
            p2p_results = run_p2p_analysis(args, cube)
        
        if args.mode in ['VNB', 'ALL']:
            vnb_results = run_vnb_analysis(args, cube, p2p_results)
        
        if args.mode in ['RDB', 'ALL']:
            rdb_results = run_rdb_analysis(args, cube, p2p_results)
            
        logger.info("ISAPC 分析完成")
        return 0
        
    except Exception as e:
        logger.error(f"分析过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())