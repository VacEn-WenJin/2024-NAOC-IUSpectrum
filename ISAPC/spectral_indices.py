"""
谱指数计算工具
基于原LineIndexCalculator设计 用于计算吸收线指数和可视化结果
"""
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy import interpolate
from typing import Dict, List, Optional, Tuple, Union, Any

from utils.calc import resample_spectrum


class LineIndexCalculator:
    """吸收线指数计算器"""
    
    def __init__(
        self, 
        wave: np.ndarray, 
        flux: np.ndarray, 
        fit_wave: np.ndarray, 
        fit_flux: np.ndarray, 
        em_wave: Optional[np.ndarray] = None, 
        em_flux_list: Optional[np.ndarray] = None, 
        velocity_correction: float = 0, 
        error: Optional[np.ndarray] = None, 
        continuum_mode: str = 'auto'
    ):
        """
        初始化吸收线指数计算器
        
        Parameters:
        -----------
        wave : array-like
            原始光谱的波长数组
        flux : array-like
            原始光谱的流量数组
        fit_wave : array-like
            拟合光谱的波长数组，用于计算连续谱
        fit_flux : array-like
            拟合光谱的流量数组，用于计算连续谱
        em_wave : array-like, optional
            发射线的波长数组
        em_flux_list : array-like, optional
            合并后的发射线光谱
        velocity_correction : float, optional
            速度修正值 单位为km/s 默认为0
        error : array-like, optional
            误差数组
        continuum_mode : str, optional
            连续谱计算模式
            'auto': 仅在原始谱数据不足时使用拟合谱
            'fit': 始终使用拟合谱
            'original': 尽可能使用原始谱（数据不足时报警）
        """
        self.c = 299792.458  # 光速，单位为km/s
        self.velocity = velocity_correction
        self.continuum_mode = continuum_mode
        
        # 进行速度修正
        self.wave = self._apply_velocity_correction(wave)
        self.flux = flux.copy()  # 创建副本以避免修改原始数据

        self.fit_wave = fit_wave
        self.fit_flux = fit_flux
        self.error = error if error is not None else np.ones_like(flux)
        
        # 处理发射线
        if em_wave is not None and em_flux_list is not None:
            self.em_wave = self._apply_velocity_correction(em_wave)
            self.em_flux_list = em_flux_list
            self._subtract_emission_lines()
        else:
            self.em_wave = None
            self.em_flux_list = None
    
    def _subtract_emission_lines(self):
        """
        从原始光谱中减去发射线
        输入的em_flux_list已经是合并后的结果
        """
        # 将发射线光谱重采样到原始光谱的波长网格上
        em_flux_resampled = resample_spectrum(self.wave, self.em_wave, self.em_flux_list)
        
        # 从原始光谱中减去发射线
        self.flux -= em_flux_resampled
    
    def _apply_velocity_correction(self, wave):
        """
        应用速度修正到波长
        
        Parameters:
        -----------
        wave : array-like
            原始波长数组
            
        Returns:
        --------
        array-like : 修正后的波长数组
        """
        return wave / (1 + (self.velocity/self.c))

    def _check_data_coverage(self, wave_range):
        """
        检查原始数据是否完整覆盖给定波长范围
        
        Parameters:
        -----------
        wave_range : tuple
            (min_wave, max_wave)
            
        Returns:
        --------
        bool : 是否完整覆盖
        """
        return (wave_range[0] >= np.min(self.wave)) and (wave_range[1] <= np.max(self.wave))
        
    def define_line_windows(self, line_name):
        """
        定义吸收线和连续谱窗口
        
        Parameters:
        -----------
        line_name : str
            吸收线名称
            
        Returns:
        --------
        dict : 包含蓝端、中心和红端窗口的字典
        """
        windows = {
            'Hbeta': {
                'blue': (4827.875, 4847.875),
                'line': (4847.875, 4876.625),
                'red': (4876.625, 4891.625)
            },
            'Mgb': {
                'blue': (5142.625, 5161.375),
                'line': (5160.125, 5192.625),
                'red': (5191.375, 5206.375)
            },
            'Fe5015': {
                'blue': (4946.500, 4977.750),
                'line': (4977.750, 5054.000),
                'red': (5054.000, 5065.250)
            },
            'Fe5270': {
                'blue': (5233.150, 5248.150),
                'line': (5245.650, 5285.650),
                'red': (5285.650, 5318.150)
            },
            'Fe5335': {
                'blue': (5304.625, 5315.875),
                'line': (5312.125, 5352.125),
                'red': (5353.375, 5363.375)
            },
            'D4000': {
                'blue': (3750.000, 3950.000),
                'line': None,
                'red': (4050.000, 4250.000)
            },
            'Halpha': {
                'blue': (6510.000, 6540.000),
                'line': (6554.000, 6568.000),
                'red': (6575.000, 6585.000)
            },
        }
        return windows.get(line_name)

    def calculate_pseudo_continuum(self, wave_range, flux_range, region_type):
        """
        计算伪连续谱
        
        Parameters:
        -----------
        wave_range : tuple or array-like
            波长范围
        flux_range : array-like or None
            对应的流量值（如果使用拟合谱则不需要）
        region_type : str
            区域类型('blue' 或 'red')
            
        Returns:
        --------
        float : 伪连续谱值
        """
        if self.continuum_mode == 'fit':
            # 使用拟合谱
            mask = (self.fit_wave >= wave_range[0]) & (self.fit_wave <= wave_range[1])
            if not np.any(mask):
                return np.nan
            return np.median(self.fit_flux[mask])
        
        elif self.continuum_mode == 'auto':
            # 检查原始数据覆盖
            if self._check_data_coverage(wave_range):
                mask = (self.wave >= wave_range[0]) & (self.wave <= wave_range[1])
                if not np.any(mask):
                    return np.nan
                return np.median(self.flux[mask])
            else:
                # 数据不足时使用拟合谱
                mask = (self.fit_wave >= wave_range[0]) & (self.fit_wave <= wave_range[1])
                if not np.any(mask):
                    return np.nan
                return np.median(self.fit_flux[mask])
        
        else:  # 'original'
            if not self._check_data_coverage(wave_range):
                raise ValueError(f"原始数据不足以覆盖{region_type}端连续谱区域")
            mask = (self.wave >= wave_range[0]) & (self.wave <= wave_range[1])
            if not np.any(mask):
                return np.nan
            return np.median(self.flux[mask])

    def calculate_index(self, line_name, return_error=False):
        """
        计算吸收线指数
        
        Parameters:
        -----------
        line_name : str
            吸收线名称 ('Hbeta', 'Mgb', 等)
        return_error : bool
            是否返回误差
            
        Returns:
        --------
        float : 吸收线指数值
        float : 误差值（如果return_error=True）
        """
        # D4000特殊处理
        if line_name == 'D4000':
            windows = self.define_line_windows(line_name)
            if windows is None:
                raise ValueError(f"未知的吸收线: {line_name}")
                
            # 计算蓝侧平均流量
            blue_mask = (self.wave >= windows['blue'][0]) & (self.wave <= windows['blue'][1])
            if not np.any(blue_mask):
                return np.nan if not return_error else (np.nan, np.nan)
            blue_flux = np.mean(self.flux[blue_mask])
            
            # 计算红侧平均流量
            red_mask = (self.wave >= windows['red'][0]) & (self.wave <= windows['red'][1])
            if not np.any(red_mask):
                return np.nan if not return_error else (np.nan, np.nan)
            red_flux = np.mean(self.flux[red_mask])
            
            # 计算D4000指数
            d4000 = red_flux / blue_flux
            
            if return_error:
                # 计算误差（简化近似）
                blue_err = np.mean(self.error[blue_mask])
                red_err = np.mean(self.error[red_mask])
                rel_err = np.sqrt((blue_err/blue_flux)**2 + (red_err/red_flux)**2)
                error = d4000 * rel_err
                return d4000, error
            return d4000
        
        # 获取窗口定义
        windows = self.define_line_windows(line_name)
        if windows is None:
            raise ValueError(f"未知的吸收线: {line_name}")

        # 获取线心区域数据
        line_mask = (self.wave >= windows['line'][0]) & (self.wave <= windows['line'][1])
        line_wave = self.wave[line_mask]
        line_flux = self.flux[line_mask]
        line_err = self.error[line_mask]

        # 检查数据点数
        if len(line_flux) < 3:
            return np.nan if not return_error else (np.nan, np.nan)

        # 计算连续谱
        blue_cont = self.calculate_pseudo_continuum(windows['blue'], None, 'blue')
        red_cont = self.calculate_pseudo_continuum(windows['red'], None, 'red')
        
        if np.isnan(blue_cont) or np.isnan(red_cont):
            return np.nan if not return_error else (np.nan, np.nan)
        
        wave_cont = np.array([
            np.mean(windows['blue']), 
            np.mean(windows['red'])
        ])
        flux_cont = np.array([blue_cont, red_cont])
        
        # 线性插值得到连续谱
        f_interp = interpolate.interp1d(wave_cont, flux_cont)
        cont_at_line = f_interp(line_wave)

        # 计算积分
        index = np.trapz((1.0 - line_flux/cont_at_line), line_wave)
        
        if return_error:
            # 计算误差
            error = np.sqrt(np.trapz((line_err/cont_at_line)**2, line_wave))
            return index, error
        
        return index

    def calculate_all_indices(self, return_errors=False):
        """
        计算所有定义的吸收线指数
        
        Parameters:
        -----------
        return_errors : bool
            是否返回误差
            
        Returns:
        --------
        dict : 包含所有吸收线指数的字典
        """
        # 获取所有定义的吸收线
        all_windows = {
            'Hbeta': self.define_line_windows('Hbeta'),
            'Mgb': self.define_line_windows('Mgb'),
            'Fe5015': self.define_line_windows('Fe5015'),
            'Fe5270': self.define_line_windows('Fe5270'),
            'Fe5335': self.define_line_windows('Fe5335'),
            'D4000': self.define_line_windows('D4000'),
            'Halpha': self.define_line_windows('Halpha')
        }
        
        # 计算所有吸收线指数
        indices = {}
        for line_name in all_windows:
            try:
                if return_errors:
                    index, error = self.calculate_index(line_name, return_error=True)
                    indices[line_name] = {'value': index, 'error': error}
                else:
                    index = self.calculate_index(line_name)
                    indices[line_name] = index
            except Exception as e:
                if return_errors:
                    indices[line_name] = {'value': np.nan, 'error': np.nan}
                else:
                    indices[line_name] = np.nan
        
        return indices

    def plot_all_lines(self, mode=None, number=None, save_path=None, show_index=False):
        """
        绘制包含所有谱线的完整图谱
        
        Parameters:
        -----------
        mode : str, optional
            图像的模式，必须是'P2P'、'VNB'或'RDB'之一
        number : int, optional
            图像的编号，必须是整数
        save_path : str, optional
            图像保存的路径。如果提供，图像将被保存到该路径下
        show_index : bool, optional
            是否显示谱指数参数，默认为False
        """
        # 验证mode和number参数
        if mode is not None and number is not None:
            valid_modes = ['P2P', 'VNB', 'RDB']
            if mode not in valid_modes:
                raise ValueError(f"Mode must be one of {valid_modes}")
            if not isinstance(number, int):
                raise ValueError("Number must be an integer")
            mode_title = f"{mode}{number}"
        else:
            mode_title = None

        # 获取所有定义的谱线
        all_windows = {
            'Hbeta': self.define_line_windows('Hbeta'),
            'Mgb': self.define_line_windows('Mgb'),
            'Fe5015': self.define_line_windows('Fe5015')
        }
        
        # 设置固定的X轴范围
        min_wave = 4800
        max_wave = 5250
        
        # 创建图形和整体标题
        fig = plt.figure(figsize=(15, 12))
        if mode_title:
            fig.suptitle(mode_title, fontsize=16, y=0.95)
        
        # 创建子图，调整高度比例以适应整体标题
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.2)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # 设置统一的颜色方案
        colors = {
            'blue': 'tab:blue',
            'line': 'tab:green',
            'red': 'tab:red',
            'orig_cont': 'tab:orange',   # 原始光谱连续谱颜色（橙色）
            'fit_cont': 'tab:green',     # 拟合光谱连续谱颜色（绿色）
            'inactive_cont': 'gray'      # 未使用的连续谱颜色
        }
        
        # 第一个面板：原始数据对比
        wave_mask = (self.wave >= min_wave) & (self.wave <= max_wave)
        fit_mask = (self.fit_wave >= min_wave) & (self.fit_wave <= max_wave)
        
        # 计算y轴范围
        if hasattr(self, 'em_flux_list') and self.em_flux_list is not None:
            em_mask = (self.em_wave >= min_wave) & (self.em_wave <= max_wave)
            flux_range = np.concatenate([self.flux[wave_mask], 
                                      self.flux[wave_mask] + self.em_flux_list[em_mask] 
                                       if np.any(em_mask) else []])
        else:
            flux_range = self.flux[wave_mask]
        
        fit_range = self.fit_flux[fit_mask]
        
        y_min = min(np.nanmin(flux_range), np.nanmin(fit_range)) * 0.9
        y_max = max(np.nanmax(flux_range), np.nanmax(fit_range)) * 1.1
        
        # 绘制光谱
        if hasattr(self, 'em_flux_list') and self.em_flux_list is not None:
            em_mask = (self.em_wave >= min_wave) & (self.em_wave <= max_wave)
            if np.any(em_mask):
                # 计算原始光谱（包含发射线）
                orig_with_em = self.flux.copy()
                em_flux_resampled = resample_spectrum(self.wave, self.em_wave, self.em_flux_list)
                orig_with_em += em_flux_resampled
                
                ax1.plot(self.wave[wave_mask], orig_with_em[wave_mask], color='tab:blue', 
                        label='Original Spectrum', alpha=0.8)
                ax1.plot(self.em_wave[em_mask], self.em_flux_list[em_mask], color='tab:orange', 
                        label='Emission Lines', alpha=0.8)
        else:
            ax1.plot(self.wave[wave_mask], self.flux[wave_mask], color='tab:blue', 
                    label='Original Spectrum', alpha=0.8)
            
        ax1.plot(self.fit_wave[fit_mask], self.fit_flux[fit_mask], color='tab:red', 
                label='Template Fit', alpha=0.8)
        
        # 为第二个面板计算y轴范围
        processed_flux = self.flux[wave_mask]
        fit_flux_range = self.fit_flux[fit_mask]
        y_min_processed = min(np.nanmin(processed_flux), np.nanmin(fit_flux_range)) * 0.9
        y_max_processed = max(np.nanmax(processed_flux), np.nanmax(fit_flux_range)) * 1.1
        
        # 第二个面板：处理后的光谱
        ax2.plot(self.wave[wave_mask], self.flux[wave_mask], color='tab:blue', 
                label='Processed Spectrum', alpha=0.8)
        ax2.plot(self.fit_wave[fit_mask], self.fit_flux[fit_mask], '--', color='tab:red',
                label='Template Fit', alpha=0.8)
        
        # 在两个面板中标记所有谱线区域
        for line_name, windows in all_windows.items():
            for panel in [ax1, ax2]:
                # 标记蓝端、线心和红端区域
                alpha = 0.2  # 恢复原来的透明度
                # 只在图例中显示一次每种区域类型
                if line_name == 'Hbeta':  # 第一个谱线用于图例
                    panel.axvspan(windows['blue'][0], windows['blue'][1], 
                                alpha=alpha, color=colors['blue'], label='Blue window')
                    panel.axvspan(windows['line'][0], windows['line'][1], 
                                alpha=alpha, color=colors['line'], label='Line region')
                    panel.axvspan(windows['red'][0], windows['red'][1], 
                                alpha=alpha, color=colors['red'], label='Red window')
                else:
                    panel.axvspan(windows['blue'][0], windows['blue'][1], 
                                alpha=alpha, color=colors['blue'])
                    panel.axvspan(windows['line'][0], windows['line'][1], 
                                alpha=alpha, color=colors['line'])
                    panel.axvspan(windows['red'][0], windows['red'][1], 
                                alpha=alpha, color=colors['red'])
                
                # 添加文字标注到底部
                if panel == ax1:
                    y_text = y_min + 0.05 * (y_max - y_min)
                else:
                    y_text = y_min_processed + 0.05 * (y_max_processed - y_min_processed)
                
                # 基础标签
                panel.text(np.mean(windows['line']), y_text, line_name,
                        horizontalalignment='center', verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
                # 在第二个面板添加连续谱点
                if panel == ax2:
                    # 计算连续谱点
                    blue_cont_orig = None
                    red_cont_orig = None
                    blue_cont_fit = None
                    red_cont_fit = None
                    
                    # 检查是否可以使用原始光谱
                    if self._check_data_coverage(windows['blue']):
                        mask = (self.wave >= windows['blue'][0]) & (self.wave <= windows['blue'][1])
                        if np.any(mask):
                            blue_cont_orig = np.median(self.flux[mask])
                    if self._check_data_coverage(windows['red']):
                        mask = (self.wave >= windows['red'][0]) & (self.wave <= windows['red'][1])
                        if np.any(mask):
                            red_cont_orig = np.median(self.flux[mask])
                    
                    # 计算拟合光谱的连续谱点
                    mask_blue = (self.fit_wave >= windows['blue'][0]) & (self.fit_wave <= windows['blue'][1])
                    mask_red = (self.fit_wave >= windows['red'][0]) & (self.fit_wave <= windows['red'][1])
                    if np.any(mask_blue):
                        blue_cont_fit = np.median(self.fit_flux[mask_blue])
                    if np.any(mask_red):
                        red_cont_fit = np.median(self.fit_flux[mask_red])
                    
                    if blue_cont_fit is not None and red_cont_fit is not None:
                        wave_cont = np.array([
                            np.mean(windows['blue']), 
                            np.mean(windows['red'])
                        ])

                        # 根据计算模式决定哪个是活动的连续谱
                        is_orig_active = (self.continuum_mode == 'original' or 
                                        (self.continuum_mode == 'auto' and 
                                        blue_cont_orig is not None and 
                                        red_cont_orig is not None))
                        
                        # 绘制原始光谱连续谱点和线（如果存在）
                        if blue_cont_orig is not None and red_cont_orig is not None:
                            flux_cont_orig = np.array([blue_cont_orig, red_cont_orig])
                            if not is_orig_active:
                                # 非活动状态
                                panel.plot(wave_cont, flux_cont_orig, '*', color=colors['inactive_cont'], 
                                        markersize=10, alpha=0.5,
                                        label='Original spectrum continuum (inactive)' if line_name == 'Hbeta' else '')
                                panel.plot(wave_cont, flux_cont_orig, '--', color=colors['inactive_cont'], 
                                        alpha=0.5)
                            else:
                                # 活动状态
                                panel.plot(wave_cont, flux_cont_orig, '*', color=colors['orig_cont'], 
                                        markersize=10, alpha=0.8,
                                        label='Original spectrum continuum (orange)' if line_name == 'Hbeta' else '')
                                panel.plot(wave_cont, flux_cont_orig, '--', color=colors['orig_cont'], 
                                        alpha=0.8)

                        # 绘制拟合光谱连续谱点和线
                        flux_cont_fit = np.array([blue_cont_fit, red_cont_fit])
                        if is_orig_active:
                            # 非活动状态
                            panel.plot(wave_cont, flux_cont_fit, '*', color=colors['inactive_cont'], 
                                    markersize=10, alpha=0.5,
                                    label='Template continuum (inactive)' if line_name == 'Hbeta' else '')
                            panel.plot(wave_cont, flux_cont_fit, '--', color=colors['inactive_cont'], 
                                    alpha=0.5)
                        else:
                            # 活动状态
                            panel.plot(wave_cont, flux_cont_fit, '*', color=colors['fit_cont'], 
                                    markersize=10, alpha=0.8,
                                    label='Template continuum (green)' if line_name == 'Hbeta' else '')
                            panel.plot(wave_cont, flux_cont_fit, '--', color=colors['fit_cont'], 
                                    alpha=0.8)

                        # 添加原始谱计算的连续谱到图例
                        if line_name == 'Hbeta':
                            dummy_line = plt.Line2D([], [], color=colors['orig_cont'], linestyle='--', 
                                                marker='*', markersize=10, alpha=0.8,
                                                label='Original spectrum continuum (orange)')
                            dummy_line2 = plt.Line2D([], [], color=colors['fit_cont'], linestyle='--', 
                                                marker='*', markersize=10, alpha=0.8,
                                                label='Template continuum (green)')
                            panel.legend(handles=panel.get_legend_handles_labels()[0] + [dummy_line, dummy_line2],
                                      labels=panel.get_legend_handles_labels()[1] + ['Original spectrum continuum (orange)', 
                                                                                   'Template continuum (green)'])

                        # 如果需要显示谱指数参数
                        if show_index:
                            try:
                                # 保存当前的continuum_mode
                                original_mode = self.continuum_mode
                                
                                # 计算原始光谱的指数值
                                self.continuum_mode = 'original'
                                try:
                                    orig_index = self.calculate_index(line_name)
                                    if np.isnan(orig_index):
                                        orig_index = None
                                except ValueError:
                                    orig_index = None
                                
                                # 计算拟合光谱的指数值
                                self.continuum_mode = 'fit'
                                fit_index = self.calculate_index(line_name)
                                if np.isnan(fit_index):
                                    fit_index = None
                                
                                # 恢复原始的continuum_mode
                                self.continuum_mode = original_mode
                                
                                # 计算文本位置
                                base_y_text = y_text + 0.05 * (y_max_processed - y_min_processed)
                                
                                # 构建显示文本
                                if orig_index is not None and fit_index is not None:
                                    # 分别显示两个值
                                    y_offset = 0.1 * (y_max_processed - y_min_processed)
                                    
                                    # 显示原始光谱的值（上面）
                                    panel.text(np.mean(windows['line']), 
                                            base_y_text + y_offset,
                                            f"{orig_index:.3f}", 
                                            color=colors['orig_cont'], 
                                            horizontalalignment='center',
                                            verticalalignment='bottom', 
                                            fontsize='x-small',
                                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                                    
                                    # 显示拟合光谱的值（下面）
                                    panel.text(np.mean(windows['line']), 
                                            base_y_text + y_offset/2,
                                            f"{fit_index:.3f}", 
                                            color=colors['fit_cont'], 
                                            horizontalalignment='center',
                                            verticalalignment='bottom', 
                                            fontsize='x-small',
                                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                                    
                                elif fit_index is not None:
                                    # 只显示拟合光谱的值
                                    fit_text = f"{fit_index:.3f}"
                                    panel.text(np.mean(windows['line']), 
                                            base_y_text + 0.02 * (y_max_processed - y_min_processed),
                                            fit_text, 
                                            color=colors['fit_cont'], 
                                            horizontalalignment='center',
                                            verticalalignment='bottom', 
                                            fontsize='x-small',
                                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                            
                            except Exception as e:
                                print(f"Error calculating index for {line_name}: {e}")
        
        # 设置两个面板的属性
        ax1.set_xlim(min_wave, max_wave)
        ax1.set_ylim(y_min, y_max)
        ax2.set_xlim(min_wave, max_wave)
        ax2.set_ylim(y_min_processed, y_max_processed)
        
        # 设置两个面板的共同属性
        for ax in [ax1, ax2]:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.tick_params(axis='both', which='both', labelsize='x-small', 
                        right=True, top=True, direction='in')
            ax.set_xlabel('Rest-frame Wavelength (Å)')
            ax.set_ylabel('Flux')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
        
        ax1.set_title(f'Original Data Comparison (v={self.velocity:.1f} km/s)')
        ax2.set_title('Processed Spectrum with Continuum Fits')
        
        # 调整布局
        if mode_title:
            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.3)
        else:
            plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9, hspace=0.3)
        
        # 如果提供了保存路径，保存图像
        if save_path and mode_title:
            # 确保save_path存在
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            # 构建完整的文件路径
            filepath = os.path.join(save_path, f"{mode_title}.pdf")
            
            # 保存图像
            plt.savefig(filepath, format='pdf', bbox_inches='tight')
            print(f"Figure saved as: {filepath}")
        
        return fig, [ax1, ax2]


class WeightParser:
    """解析拟合权重得到物理参数的类"""
    
    def __init__(self, template_path: Union[str, Path]) -> None:
        """初始化 加载SSP模板
        
        Args:
            template_path: SSP模板.npz文件路径
        """
        # 加载模板数据
        data = np.load(template_path, allow_pickle=True)
        self.ages = data['ages']      # 年龄数组 (25,)
        self.metals = data['metals']  # 金属丰度数组 (6,)
        
        # 验证模板维度
        if len(self.ages) != 25 or len(self.metals) != 6:
            raise ValueError(
                f"Invalid template dimensions: "
                f"ages={len(self.ages)}, metals={len(self.metals)}"
            )
        
        # 构建参数网格
        age_grid, metal_grid = np.meshgrid(self.ages, self.metals, indexing='ij')
        # age_grid shape: (25, 6), 每行相同的age值
        # metal_grid shape: (25, 6), 每列相同的metal值
        
        # 将网格reshape为与模板相同的方式
        self.age_vector = age_grid.reshape(-1)    # (150,)
        self.metal_vector = metal_grid.reshape(-1)  # (150,)
        
        # 计算年龄的对数值
        self.log_age_vector = np.log10(self.age_vector)
    
    def parse_weights(self, weights: Union[List[float], np.ndarray]) -> Tuple[float, float]:
        """解析权重获取平均log(Age)和[M/H]
        
        Args:
            weights: 拟合权重 (150,)
            
        Returns:
            tuple: (mean_log_age, mean_metallicity)
        """
        # 验证权重长度
        weights = np.array(weights)
        if len(weights) != len(self.age_vector):
            raise ValueError(f"Weights must have length {len(self.age_vector)}, got {len(weights)}")
        
        # 计算总权重
        total_weight = np.sum(weights)
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")
        
        # 直接用向量计算加权平均
        mean_log_age = np.sum(self.log_age_vector * weights) / total_weight
        mean_metallicity = np.sum(self.metal_vector * weights) / total_weight
        
        return mean_log_age, mean_metallicity
    
    def get_physical_params(self, weights: Union[List[float], np.ndarray]) -> dict:
        """获取权重对应的所有物理参数"""
        log_age, metal = self.parse_weights(weights)
        
        return {
            'log_age': log_age,
            'age': 10**log_age,
            'metallicity': metal
        }


def calculate_indices_cube(
    wavelength: np.ndarray,
    cube: np.ndarray,
    template_wave: np.ndarray,
    template_cube: np.ndarray,
    em_wave: Optional[np.ndarray] = None,
    em_cube: Optional[np.ndarray] = None,
    velocity_field: Optional[np.ndarray] = None,
    indices: Optional[List[str]] = None,
    continuum_mode: str = 'auto',
    n_jobs: int = -1
) -> Dict[str, np.ndarray]:
    """
    计算数据立方体的谱指数
    
    Parameters:
    -----------
    wavelength: 波长数组
    cube: 3D数据立方体 (波长, y, x)
    template_wave: 模板波长数组
    template_cube: 3D模板数据立方体 (波长, y, x)
    em_wave: 发射线波长数组，可选
    em_cube: 3D发射线数据立方体 (波长, y, x)，可选
    velocity_field: 速度场 (y, x)，可选，用于速度修正
    indices: 要计算的指数列表，默认计算所有
    continuum_mode: 连续谱计算模式，'auto', 'fit', or 'original'
    n_jobs: 并行任务数
    
    Returns:
    --------
    谱指数字典，每个指数为2D数组
    """
    from joblib import delayed
    from utils.parallel import ParallelTqdm
    
    # 创建空索引计算器获取默认指数定义
    dummy_calc = LineIndexCalculator(
        wavelength, np.ones_like(wavelength),
        template_wave, np.ones_like(template_wave)
    )
    
    if indices is None:
        # 获取所有定义的指数
        test_windows = {
            'Hbeta': dummy_calc.define_line_windows('Hbeta'),
            'Mgb': dummy_calc.define_line_windows('Mgb'),
            'Fe5015': dummy_calc.define_line_windows('Fe5015'),
            'Fe5270': dummy_calc.define_line_windows('Fe5270'),
            'Fe5335': dummy_calc.define_line_windows('Fe5335'),
            'D4000': dummy_calc.define_line_windows('D4000'),
            'Halpha': dummy_calc.define_line_windows('Halpha')
        }
        indices = [name for name, window in test_windows.items() if window is not None]
    
    # 获取数据立方体尺寸
    nz, ny, nx = cube.shape
    
    # 初始化结果
    results = {name: np.full((ny, nx), np.nan) for name in indices}
    
    # 定义单像素计算函数
    def calculate_pixel(y, x):
        if np.count_nonzero(~np.isnan(cube[:, y, x])) < 10:
            return y, x, None
        
        try:
            # 获取当前像素的速度修正值
            vel_corr = 0.0
            if velocity_field is not None:
                vel_corr = velocity_field[y, x]
                if np.isnan(vel_corr):
                    vel_corr = 0.0
            
            # 获取当前像素的发射线数据
            em_flux = None
            if em_wave is not None and em_cube is not None:
                em_flux = em_cube[:, y, x]
            
            # 创建指数计算器
            calculator = LineIndexCalculator(
                wavelength, cube[:, y, x],
                template_wave, template_cube[:, y, x],
                em_wave=em_wave, em_flux_list=em_flux,
                velocity_correction=vel_corr,
                continuum_mode=continuum_mode
            )
            
            # 计算所有指数
            indices_values = {}
            for index_name in indices:
                try:
                    indices_values[index_name] = calculator.calculate_index(index_name)
                except Exception:
                    indices_values[index_name] = np.nan
                    
            return y, x, indices_values
        except Exception as e:
            print(f"Error at pixel ({y}, {x}): {e}")
            return y, x, None
    
    # 并行计算所有像素
    parallel_results = ParallelTqdm(
        n_jobs=n_jobs, desc="Computing spectral indices", total_tasks=ny*nx
    )(delayed(calculate_pixel)(y, x) 
      for y in range(ny) for x in range(nx))
    
    # 收集结果
    for res in parallel_results:
        y, x, values = res
        if values is not None:
            for name, value in values.items():
                results[name][y, x] = value
    
    return results