"""
ISAPC可视化工具
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec


def plot_velocity_field(
    velocity_field: np.ndarray,
    mask: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'RdBu_r',
    title: str = 'Velocity Field',
    colorbar: bool = True,
    contours: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制速度场
    
    Parameters:
    -----------
    velocity_field: 速度场数组
    mask: 掩码数组，True表示显示
    ax: 绘图axes对象
    vmin: 颜色映射最小值
    vmax: 颜色映射最大值
    cmap: 颜色映射
    title: 图标题
    colorbar: 是否显示颜色条
    contours: 是否绘制等高线
    
    Returns:
    --------
    fig, ax: 图对象和axes对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure
    
    # 应用掩码
    if mask is not None:
        data = np.copy(velocity_field)
        data[~mask] = np.nan
    else:
        data = velocity_field
    
    # 设置颜色范围
    if vmin is None or vmax is None:
        vabs = np.nanpercentile(np.abs(data), 95)
        vmin = -vabs if vmin is None else vmin
        vmax = vabs if vmax is None else vmax
    
    # 绘制速度场
    im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # 添加等高线
    if contours:
        levels = np.linspace(vmin, vmax, 10)
        cs = ax.contour(data, levels=levels, colors='k', alpha=0.5, linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%d')
    
    # 添加颜色条
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('km/s')
    
    # 添加标题
    ax.set_title(title)
    
    # 设置坐标轴
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Pixels')
    
    return fig, ax


def plot_dispersion_field(
    dispersion_field: np.ndarray,
    mask: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
    title: str = 'Velocity Dispersion',
    colorbar: bool = True,
    contours: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制速度弥散场
    
    Parameters:
    -----------
    dispersion_field: 速度弥散场数组
    mask: 掩码数组，True表示显示
    ax: 绘图axes对象
    vmin: 颜色映射最小值
    vmax: 颜色映射最大值
    cmap: 颜色映射
    title: 图标题
    colorbar: 是否显示颜色条
    contours: 是否绘制等高线
    
    Returns:
    --------
    fig, ax: 图对象和axes对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure
    
    # 应用掩码
    if mask is not None:
        data = np.copy(dispersion_field)
        data[~mask] = np.nan
    else:
        data = dispersion_field
    
    # 设置颜色范围
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = np.nanpercentile(data, 95)
    
    # 绘制弥散场
    im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    
    # 添加等高线
    if contours:
        levels = np.linspace(vmin, vmax, 8)
        cs = ax.contour(data, levels=levels, colors='k', alpha=0.5, linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%d')
    
    # 添加颜色条
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('km/s')
    
    # 添加标题
    ax.set_title(title)
    
    # 设置坐标轴
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Pixels')
    
    return fig, ax


def plot_binning_map(
    bin_map: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = 'Binning Map',
    colorbar: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制宾化映射图
    
    Parameters:
    -----------
    bin_map: 宾化索引映射数组
    ax: 绘图axes对象
    title: 图标题
    colorbar: 是否显示颜色条
    
    Returns:
    --------
    fig, ax: 图对象和axes对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure
    
    # 创建掩码，剔除未分配的像素
    mask = bin_map >= 0
    data = np.copy(bin_map)
    data[~mask] = -1
    
    # 计算宾的数量
    n_bins = int(np.max(bin_map)) + 1
    
    # 使用分立颜色映射
    from matplotlib.colors import ListedColormap
    # 生成随机颜色
    import matplotlib.cm as cm
    cmap = cm.get_cmap('nipy_spectral', n_bins)
    
    # 绘制宾化图
    im = ax.imshow(data, origin='lower', cmap=cmap, 
                  vmin=-0.5, vmax=n_bins-0.5)
    
    # 添加颜色条
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=range(n_bins))
        cbar.set_label('Bin Index')
    
    # 添加标题
    ax.set_title(title)
    
    # 设置坐标轴
    ax.set_xlabel('Pixels')
    ax.set_ylabel('Pixels')
    
    return fig, ax


def plot_spectrum_fit(
    wavelength: np.ndarray,
    observed_flux: np.ndarray,
    model_flux: np.ndarray,
    residual: Optional[np.ndarray] = None,
    gas_flux: Optional[np.ndarray] = None,
    stellar_flux: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 8),
    title: str = 'Spectrum Fit',
    wavelength_range: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制光谱拟合结果
    
    Parameters:
    -----------
    wavelength: 波长数组
    observed_flux: 观测流量
    model_flux: 模型拟合流量
    residual: 残差
    gas_flux: 气体成分流量
    stellar_flux: 恒星成分流量
    ax: 绘图axes对象
    figsize: 图尺寸
    title: 图标题
    wavelength_range: 显示的波长范围
    
    Returns:
    --------
    fig, ax: 图对象和axes对象
    """
    if residual is None:
        residual = observed_flux - model_flux
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1], hspace=0.0)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax3 = plt.subplot(gs[2], sharex=ax1)
        axes = [ax1, ax2, ax3]
    else:
        raise ValueError("Custom axes not supported for spectrum fit plot")
    
    # 限制波长范围
    if wavelength_range is not None:
        wmin, wmax = wavelength_range
        mask = (wavelength >= wmin) & (wavelength <= wmax)
        wavelength = wavelength[mask]
        observed_flux = observed_flux[mask]
        model_flux = model_flux[mask]
        residual = residual[mask]
        if gas_flux is not None:
            gas_flux = gas_flux[mask]
        if stellar_flux is not None:
            stellar_flux = stellar_flux[mask]
    
    # 第一个面板：观测光谱和恒星模型
    ax1.plot(wavelength, observed_flux, 'k-', lw=1, alpha=0.8, label='Observed')
    if stellar_flux is not None:
        ax1.plot(wavelength, stellar_flux, 'r-', lw=1, alpha=0.8, label='Stellar model')
    
    # 第二个面板：观测光谱和完整模型
    ax2.plot(wavelength, observed_flux, 'k-', lw=1, alpha=0.8, label='Observed')
    ax2.plot(wavelength, model_flux, 'r-', lw=1, alpha=0.8, label='Full model')
    
    # 第三个面板：残差和气体成分
    ax3.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax3.plot(wavelength, residual, 'g-', lw=0.8, alpha=0.7, label='Residual')
    if gas_flux is not None:
        ax3.plot(wavelength, gas_flux, 'b-', lw=1, alpha=0.8, label='Gas component')
    
    # 添加常见谱线标记
    spectral_lines = {
        'OII3726': 3726.03,
        'OII3729': 3728.82,
        'Hgamma': 4340.47,
        'Hbeta': 4861.33,
        'OIII4959': 4958.92,
        'OIII5007': 5006.84,
        'Halpha': 6562.80,
        'NII6583': 6583.41
    }
    
    for ax in axes:
        for name, wave in spectral_lines.items():
            if wavelength_range is None or (wave >= wavelength_range[0] and wave <= wavelength_range[1]):
                ax.axvline(wave, ls='--', color='gray', alpha=0.5, lw=0.5)
                if ax == ax3:  # 只在底部面板标注
                    ax.text(wave, ax.get_ylim()[0], name, rotation=90, 
                          fontsize=8, va='bottom', ha='center', alpha=0.7)
    
    # 格式化
    for ax in axes:
        ax.grid(True, alpha=0.3, ls=':')
        ax.legend(loc='upper right', fontsize='small')
    
    # 隐藏中间面板的x标签
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # 设置标签
    ax3.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Flux')
    ax2.set_ylabel('Flux')
    ax3.set_ylabel('Residual')
    
    # 添加标题
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    
    return fig, axes


def plot_rotation_curve(
    rotation_curve: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = 'Rotation Curve',
    plot_model: bool = False,
    vmax: Optional[float] = None,
    pa: Optional[float] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制旋转曲线
    
    Parameters:
    -----------
    rotation_curve: 旋转曲线数组 (r, v)
    ax: 绘图axes对象
    title: 图标题
    plot_model: 是否绘制模型拟合
    vmax: 最大旋转速度
    pa: 位置角 (度)
    
    Returns:
    --------
    fig, ax: 图对象和axes对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # 绘制旋转曲线数据点
    ax.plot(rotation_curve[:, 0], rotation_curve[:, 1], 'o-', color='blue',
           lw=2, ms=8, alpha=0.7, label='Data')
    
    # 绘制模型拟合
    if plot_model and vmax is not None:
        r_model = np.linspace(0, rotation_curve[-1, 0] * 1.1, 100)
        # 简单模型：v(r) = 2/π * vmax * arctan(r/r0)
        r0 = rotation_curve[-1, 0] * 0.2  # 假定拐点在20%最大半径
        v_model = 2/np.pi * vmax * np.arctan(r_model / r0)
        ax.plot(r_model, v_model, '--', color='red', lw=2, alpha=0.7, label='Model')
    
    # 添加标题和位置角信息
    if pa is not None:
        title = f"{title} (PA={pa:.1f}°)"
    ax.set_title(title)
    
    # 设置轴标签
    ax.set_xlabel('Radius (pixels)')
    ax.set_ylabel('Rotation Velocity (km/s)')
    
    # 添加网格和图例
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig, ax


def plot_kinematics_summary(
    velocity_field: np.ndarray,
    dispersion_field: np.ndarray,
    bin_map: Optional[np.ndarray] = None,
    rotation_curve: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    mask: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (16, 12)
) -> plt.Figure:
    """
    绘制动力学分析综合图
    
    Parameters:
    -----------
    velocity_field: 速度场数组
    dispersion_field: 弥散场数组
    bin_map: 宾化映射数组
    rotation_curve: 旋转曲线数组
    params: 动力学参数字典
    mask: 掩码数组
    figsize: 图尺寸
    
    Returns:
    --------
    fig: 图对象
    """
    fig = plt.figure(figsize=figsize)
    
    # 创建网格
    if bin_map is not None and rotation_curve is not None:
        gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1])
    else:
        gs = gridspec.GridSpec(1, 3, figure=fig)
    
    # 速度场
    ax1 = fig.add_subplot(gs[0, 0])
    plot_velocity_field(velocity_field, mask=mask, ax=ax1, title='Velocity Field')
    
    # 弥散场
    ax2 = fig.add_subplot(gs[0, 1])
    plot_dispersion_field(dispersion_field, mask=mask, ax=ax2, title='Velocity Dispersion')
    
    # V/sigma场
    ax3 = fig.add_subplot(gs[0, 2])
    if mask is not None:
        vsigma_map = np.abs(velocity_field) / dispersion_field
        vsigma_map[~mask] = np.nan
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vsigma_map = np.abs(velocity_field) / dispersion_field
    
    im = ax3.imshow(vsigma_map, origin='lower', cmap='viridis', vmin=0, vmax=2)
    plt.colorbar(im, ax=ax3, shrink=0.8)
    ax3.set_title('V/σ Map')
    ax3.set_xlabel('Pixels')
    ax3.set_ylabel('Pixels')
    
    # 如果有宾化和旋转曲线数据，添加到第二行
    if bin_map is not None and rotation_curve is not None:
        # 宾化图
        ax4 = fig.add_subplot(gs[1, 0])
        plot_binning_map(bin_map, ax=ax4, title='Binning Map')
        
        # 旋转曲线
        ax5 = fig.add_subplot(gs[1, 1])
        if params is not None and 'pa' in params:
            pa = params['pa']
        else:
            pa = None
        plot_rotation_curve(rotation_curve, ax=ax5, pa=pa)
        
        # 添加参数信息
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        if params is not None:
            info_text = "Kinematic Parameters:\n\n"
            
            if 'pa' in params:
                info_text += f"Position Angle: {params['pa']:.1f}°\n"
            if 'vsys' in params:
                info_text += f"Systemic Velocity: {params['vsys']:.1f} km/s\n"
            if 'vmax' in params:
                info_text += f"Maximum Rotation: {params['vmax']:.1f} km/s\n"
            if 'sigma_mean' in params:
                info_text += f"Mean σ: {params['sigma_mean']:.1f} km/s\n"
            if 'v_over_sigma' in params:
                info_text += f"V/σ: {params['v_over_sigma']:.2f}\n"
            if 'lambda_r' in params:
                info_text += f"λR: {params['lambda_r']:.2f}\n"
            
            ax6.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=1", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    return fig