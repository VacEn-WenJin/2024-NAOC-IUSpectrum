"""
光谱宾化工具 - 支持Voronoi宾化和径向宾化
"""
import numpy as np
import warnings
from typing import Tuple, Dict, Optional, Union, List

from vorbin.voronoi_2d_binning import voronoi_2d_binning
from utils.parallel import ParallelTqdm
from joblib import delayed
from utils.calc import resample_spectrum

class VoronoiBinning:
    """基于Voronoi宾化算法的光谱宾化类"""
    
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        signal: np.ndarray,
        noise: np.ndarray,
        wavelength: np.ndarray,
        spectra: np.ndarray,
        shape: Tuple[int, int],
        pixelsize: float = 1.0
    ):
        """
        初始化Voronoi宾化
        
        Parameters:
        -----------
        x: 像素x坐标数组
        y: 像素y坐标数组
        signal: 信号强度数组
        noise: 噪声强度数组
        wavelength: 波长数组
        spectra: 所有像素光谱数组 (波长, n_pixels) 
        shape: 图像原始形状 (ny, nx)
        pixelsize: 像素尺寸 (弧秒)
        """
        self.x = x
        self.y = y
        self.signal = signal
        self.noise = noise
        self.wavelength = wavelength
        self.spectra = spectra
        self.shape = shape
        self.pixelsize = pixelsize
        
        # 计算信噪比
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.snr = np.zeros_like(signal)
            valid_mask = noise > 0
            self.snr[valid_mask] = signal[valid_mask] / noise[valid_mask]
        
        # 初始化结果存储
        self.bin_number = None
        self.bin_signal = None
        self.bin_noise = None
        self.bin_spectrum = None
        self.bin_snr = None
        self.bin_x = None
        self.bin_y = None
        self.bin_npixels = None
        self.binned_spectra = None
    
    def compute_bins(self, target_snr: float, quiet: bool = True) -> Dict:
        """
        计算Voronoi宾化
        
        Parameters:
        -----------
        target_snr: 目标信噪比
        quiet: 是否显示详细信息
        
        Returns:
        --------
        宾化结果字典
        """
        # 只使用有效像素
        valid_mask = (self.snr > 0) & np.isfinite(self.snr)
        x = self.x[valid_mask]
        y = self.y[valid_mask]
        signal = self.signal[valid_mask]
        noise = self.noise[valid_mask]
        
        if len(x) < 10:
            raise ValueError("Not enough valid pixels for binning")
        
        # 执行Voronoi宾化
        bin_number, x_gen, y_gen, bin_x, bin_y, bin_snr, bin_npixels, scale = voronoi_2d_binning(
            x, y, signal, noise, target_snr,
            pixelsize=self.pixelsize, plot=False, quiet=quiet,
            cvt=True, wvt=True
        )
        
        # 保存宾化结果
        self.bin_number = bin_number
        self.bin_x = bin_x
        self.bin_y = bin_y
        self.bin_snr = bin_snr
        self.bin_npixels = bin_npixels
        
        # 创建宾化索引映射
        bin_map = np.full(self.shape, -1)
        valid_idx = np.where(valid_mask)[0]
        
        # 将宾号映射回原始2D网格
        row = self.y.astype(int)
        col = self.x.astype(int)
        
        for i, bin_idx in enumerate(bin_number):
            pixel_idx = valid_idx[i]
            r = row[pixel_idx]
            c = col[pixel_idx]
            
            # 确保索引在有效范围内
            if 0 <= r < self.shape[0] and 0 <= c < self.shape[1]:
                bin_map[r, c] = bin_idx
        
        # 返回宾结果
        return {
            'bin_map': bin_map,
            'n_bins': int(np.max(bin_number)) + 1,
            'bin_x': bin_x,
            'bin_y': bin_y,
            'bin_snr': bin_snr,
            'bin_npixels': bin_npixels
        }
    
    def extract_binned_spectra(
        self, 
        bin_map: np.ndarray,
        velocity_field: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """
        提取宾化后的综合光谱
        
        Parameters:
        -----------
        bin_map: 宾化索引映射
        velocity_field: 速度场，用于校正光谱对齐
        
        Returns:
        --------
        宾化后的光谱字典，键为宾索引
        """
        # 计算宾内像素
        n_bins = int(np.max(bin_map)) + 1
        binned_spectra = {}
        binned_errors = {}
        
        # 重新计算行列索引
        ny, nx = self.shape
        row = (self.y + 0.5).astype(int)  # +0.5以四舍五入
        col = (self.x + 0.5).astype(int)
        
        # 为每个宾收集光谱
        for bin_idx in range(n_bins):
            # 找到宾内所有像素
            pixel_indices = []
            for i, (r, c) in enumerate(zip(row, col)):
                # 确保索引在有效范围内
                if 0 <= r < ny and 0 <= c < nx and bin_map[r, c] == bin_idx:
                    pixel_indices.append(i)
            
            if not pixel_indices:
                continue
                
            # 获取光谱
            bin_spectra = self.spectra[:, pixel_indices]
            
            # 如果提供了速度场，进行光谱对齐
            if velocity_field is not None:
                # 提取宾内速度
                bin_velocities = [velocity_field[r, c] if 0 <= r < ny and 0 <= c < nx else 0
                                 for r, c in zip(row[pixel_indices], col[pixel_indices])]
                
                # 对齐并累积光谱
                aligned_spectra = np.zeros_like(bin_spectra)
                
                for i, vel in enumerate(bin_velocities):
                    if np.isnan(vel):
                        aligned_spectra[:, i] = bin_spectra[:, i]
                    else:
                        # 计算波长偏移
                        z = vel / 299792.458  # c in km/s
                        shifted_wave = self.wavelength * (1 + z)
                        
                        # 插值到原始波长
                        aligned_spectra[:, i] = np.interp(
                            self.wavelength, shifted_wave, bin_spectra[:, i],
                            left=0, right=0
                        )
                
                # 使用对齐后的光谱
                bin_spectra = aligned_spectra
            
            # 计算宾光谱 (简单平均)
            binned_spectra[bin_idx] = np.nanmean(bin_spectra, axis=1)
        
        self.binned_spectra = binned_spectra
        return binned_spectra


class RadialBinning:
    """径向宾化类"""
    
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        signal: np.ndarray,
        noise: np.ndarray,
        wavelength: np.ndarray,
        spectra: np.ndarray,
        shape: Tuple[int, int]
    ):
        """
        初始化径向宾化
        
        Parameters:
        -----------
        x: 像素x坐标数组
        y: 像素y坐标数组
        signal: 信号强度数组
        noise: 噪声强度数组
        wavelength: 波长数组
        spectra: 所有像素光谱数组 (波长, n_pixels)
        shape: 图像原始形状 (ny, nx)
        """
        self.x = x
        self.y = y
        self.signal = signal
        self.noise = noise
        self.wavelength = wavelength
        self.spectra = spectra
        self.shape = shape
        
        # 计算信噪比
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.snr = np.zeros_like(signal)
            valid_mask = noise > 0
            self.snr[valid_mask] = signal[valid_mask] / noise[valid_mask]
        
        # 初始化结果存储
        self.bin_edges = None
        self.binned_spectra = None
        self.bin_map = None
        self.radial_map = None
    
    def compute_bins(
        self,
        n_bins: int = 10,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        pa: float = 0.0,
        ellipticity: float = 0.0,
        log_spacing: bool = True,
        min_radius: float = 0.0,
        max_radius: Optional[float] = None,
        min_snr: float = 0.0
    ) -> Dict:
        """
        计算径向宾化
        
        Parameters:
        -----------
        n_bins: 径向宾数量
        center_x: 中心x坐标，默认为图像中心
        center_y: 中心y坐标，默认为图像中心
        pa: 位置角 (度)
        ellipticity: 椭率 (0-1)
        log_spacing: 是否使用对数间隔
        min_radius: 最小半径
        max_radius: 最大半径，默认使用最大距离
        min_snr: 最小信噪比要求
        
        Returns:
        --------
        宾化结果字典
        """
        ny, nx = self.shape
        
        # 设置默认中心
        if center_x is None:
            center_x = nx / 2
        if center_y is None:
            center_y = ny / 2
        
        # 计算到中心的距离
        x_rel = self.x - center_x
        y_rel = self.y - center_y
        
        # 应用位置角和椭率
        if ellipticity > 0 or pa != 0:
            # 转换为弧度
            pa_rad = np.radians(pa)
            
            # 旋转坐标系
            x_rot = x_rel * np.cos(pa_rad) + y_rel * np.sin(pa_rad)
            y_rot = -x_rel * np.sin(pa_rad) + y_rel * np.cos(pa_rad)
            
            # 应用椭率
            b_to_a = 1 - ellipticity
            radius = np.sqrt(x_rot**2 + (y_rot/b_to_a)**2)
        else:
            # 简单欧几里得距离
            radius = np.sqrt(x_rel**2 + y_rel**2)
        
        # 创建半径图
        radial_map = np.full(self.shape, np.nan)
        
        # 重新计算行列索引
        row = (self.y + 0.5).astype(int)  # +0.5以四舍五入
        col = (self.x + 0.5).astype(int)
        
        for i, r in enumerate(radius):
            if 0 <= row[i] < ny and 0 <= col[i] < nx:
                radial_map[row[i], col[i]] = r
        
        # 保存半径图
        self.radial_map = radial_map
        
        # 设置最大半径
        if max_radius is None:
            max_radius = np.nanmax(radius)
        
        # 创建径向宾边界
        if log_spacing:
            # 对数间隔
            bin_edges = np.logspace(
                np.log10(max(min_radius, 0.5)), 
                np.log10(max_radius), 
                n_bins + 1
            )
        else:
            # 线性间隔
            bin_edges = np.linspace(min_radius, max_radius, n_bins + 1)
        
        self.bin_edges = bin_edges
        
        # 创建宾映射
        bin_map = np.full(self.shape, -1)
        valid_mask = (self.snr >= min_snr)
        
        # 为每个宾分配像素
        for bin_idx in range(n_bins):
            # 获取内外半径
            r_in = bin_edges[bin_idx]
            r_out = bin_edges[bin_idx + 1]
            
            # 分配像素到宾
            for i, r in enumerate(radius):
                if r >= r_in and r < r_out and valid_mask[i]:
                    r_idx = row[i]
                    c_idx = col[i]
                    if 0 <= r_idx < ny and 0 <= c_idx < nx:
                        bin_map[r_idx, c_idx] = bin_idx
        
        # 保存宾映射
        self.bin_map = bin_map
        
        # 计算每个宾的像素数
        bin_counts = []
        for bin_idx in range(n_bins):
            count = np.sum(bin_map == bin_idx)
            bin_counts.append(count)
        
        # 返回宾信息
        return {
            'bin_map': bin_map,
            'radial_map': radial_map,
            'bin_edges': bin_edges,
            'bin_counts': bin_counts,
            'center': (center_x, center_y),
            'pa': pa,
            'ellipticity': ellipticity
        }
    
    def extract_binned_spectra(
        self, 
        bin_map: np.ndarray,
        velocity_field: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """
        提取径向宾综合光谱
        
        Parameters:
        -----------
        bin_map: 宾化索引映射
        velocity_field: 速度场，用于校正光谱对齐
        
        Returns:
        --------
        宾化后的光谱字典，键为宾索引
        """
        # 计算宾内像素
        n_bins = int(np.max(bin_map)) + 1
        binned_spectra = {}
        
        # 重新计算行列索引
        ny, nx = self.shape
        row = (self.y + 0.5).astype(int)  # +0.5以四舍五入
        col = (self.x + 0.5).astype(int)
        
        # 为每个宾收集光谱
        for bin_idx in range(n_bins):
            # 找到宾内所有像素
            pixel_indices = []
            for i, (r, c) in enumerate(zip(row, col)):
                # 确保索引在有效范围内
                if 0 <= r < ny and 0 <= c < nx and bin_map[r, c] == bin_idx:
                    pixel_indices.append(i)
            
            if not pixel_indices:
                continue
                
            # 获取光谱
            bin_spectra = self.spectra[:, pixel_indices]
            
            # 如果提供了速度场，进行光谱对齐
            if velocity_field is not None:
                # 提取宾内速度
                bin_velocities = [velocity_field[r, c] if 0 <= r < ny and 0 <= c < nx else 0
                                 for r, c in zip(row[pixel_indices], col[pixel_indices])]
                
                # 对齐并累积光谱
                aligned_spectra = np.zeros_like(bin_spectra)
                
                for i, vel in enumerate(bin_velocities):
                    if np.isnan(vel):
                        aligned_spectra[:, i] = bin_spectra[:, i]
                    else:
                        # 计算波长偏移
                        z = vel / 299792.458  # c in km/s
                        shifted_wave = self.wavelength * (1 + z)
                        
                        # 插值到原始波长
                        aligned_spectra[:, i] = np.interp(
                            self.wavelength, shifted_wave, bin_spectra[:, i],
                            left=0, right=0
                        )
                
                # 使用对齐后的光谱
                bin_spectra = aligned_spectra
            
            # 计算宾光谱 (简单平均)
            binned_spectra[bin_idx] = np.nanmean(bin_spectra, axis=1)
        
        self.binned_spectra = binned_spectra
        return binned_spectra