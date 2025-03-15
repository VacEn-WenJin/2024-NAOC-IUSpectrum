"""
星系参数计算工具
"""
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Union


class GalaxyParameters:
    """星系参数计算类"""
    
    def __init__(
        self,
        velocity_field: np.ndarray,
        dispersion_field: np.ndarray,
        x: np.ndarray = None,
        y: np.ndarray = None,
        pixelsize: float = 1.0,
        distance: float = None
    ):
        """
        初始化星系参数计算
        
        Parameters:
        -----------
        velocity_field: 速度场 (2D数组)
        dispersion_field: 弥散场 (2D数组)
        x: x坐标数组，默认使用像素坐标
        y: y坐标数组，默认使用像素坐标
        pixelsize: 像素大小 (弧秒)
        distance: 星系距离 (兆秒差距)
        """
        self.velocity_field = velocity_field
        self.dispersion_field = dispersion_field
        self.pixelsize = pixelsize
        self.distance = distance
        
        ny, nx = velocity_field.shape
        if x is None or y is None:
            # 使用像素坐标
            y_grid, x_grid = np.indices((ny, nx))
            self.x = x_grid.ravel()
            self.y = y_grid.ravel()
        else:
            self.x = x
            self.y = y
            
        # 初始化结果存储
        self.kinematic_pa = None
        self.vsys = None
        self.vmax = None
        self.sigma_mean = None
        self.v_over_sigma = None
        self.lambda_r = None
    
    def fit_rotation_curve(
        self,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        pa_initial: float = 0.0,
        r_max: Optional[float] = None
    ) -> Dict:
        """
        拟合旋转曲线和运动学参数
        
        Parameters:
        -----------
        center_x: 中心x坐标
        center_y: 中心y坐标
        pa_initial: 初始位置角猜测 (度)
        r_max: 最大拟合半径
        
        Returns:
        --------
        拟合参数字典
        """
        ny, nx = self.velocity_field.shape
        
        # 设置默认中心
        if center_x is None:
            center_x = nx / 2
        if center_y is None:
            center_y = ny / 2
            
        # 提取有效速度值
        vel_data = self.velocity_field.ravel()
        valid_mask = np.isfinite(vel_data)
        x = self.x[valid_mask]
        y = self.y[valid_mask]
        vel = vel_data[valid_mask]
        
        # 搜索位置角
        best_pa = pa_initial
        best_amplitude = 0
        
        for test_pa in np.linspace(0, 180, 19):  # 10度步长搜索
            pa_rad = np.radians(test_pa)
            
            # 投影到位置角上
            proj_dist = (x - center_x) * np.cos(pa_rad) + (y - center_y) * np.sin(pa_rad)
            
            # 计算速度振幅
            vel_pos = vel[proj_dist > 0]
            vel_neg = vel[proj_dist < 0]
            
            if len(vel_pos) > 0 and len(vel_neg) > 0:
                amplitude = np.median(vel_pos) - np.median(vel_neg)
                
                if abs(amplitude) > abs(best_amplitude):
                    best_amplitude = amplitude
                    best_pa = test_pa
        
        # 精细搜索
        search_range = 20  # 在找到的最佳PA值附近±10度范围内搜索
        for test_pa in np.linspace(best_pa - search_range/2, best_pa + search_range/2, 21):
            pa_rad = np.radians(test_pa % 180)
            
            # 投影到位置角上
            proj_dist = (x - center_x) * np.cos(pa_rad) + (y - center_y) * np.sin(pa_rad)
            
            # 计算速度振幅
            vel_pos = vel[proj_dist > 0]
            vel_neg = vel[proj_dist < 0]
            
            if len(vel_pos) > 0 and len(vel_neg) > 0:
                amplitude = np.median(vel_pos) - np.median(vel_neg)
                
                if abs(amplitude) > abs(best_amplitude):
                    best_amplitude = amplitude
                    best_pa = test_pa % 180
        
        # 使用最佳位置角计算径向分布
        best_pa_rad = np.radians(best_pa)
        
        # 计算到中心的距离和投影距离
        dx = x - center_x
        dy = y - center_y
        radius = np.sqrt(dx**2 + dy**2)
        proj_dist = dx * np.cos(best_pa_rad) + dy * np.sin(best_pa_rad)
        
        # 应用最大半径限制
        if r_max is not None:
            radius_mask = radius <= r_max
            radius = radius[radius_mask]
            proj_dist = proj_dist[radius_mask]
            vel = vel[radius_mask]
        
        # 将位置角调整为约定区间 [0, 180)
        if best_amplitude < 0:
            best_pa = (best_pa + 180) % 180
            best_amplitude = -best_amplitude
        
        # 计算系统速度
        self.vsys = np.median(vel)
        self.vmax = best_amplitude / 2
        self.kinematic_pa = best_pa
        
        # 构建旋转曲线数据
        # 为多个径向仓计算平均速度
        radial_bins = 10
        if r_max is None:
            r_max = np.max(radius)
        
        r_bins = np.linspace(0, r_max, radial_bins + 1)
        rotation_curve = []
        
        for i in range(radial_bins):
            r_min = r_bins[i]
            r_max_bin = r_bins[i + 1]
            
            # 选择该径向仓内的数据
            bin_mask = (radius >= r_min) & (radius < r_max_bin)
            
            if np.sum(bin_mask) > 5:  # 至少需要5个点
                # 分别计算正投影和负投影区域
                pos_mask = bin_mask & (proj_dist > 0)
                neg_mask = bin_mask & (proj_dist < 0)
                
                if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
                    v_pos = np.median(vel[pos_mask])
                    v_neg = np.median(vel[neg_mask])
                    v_rot = (v_pos - v_neg) / 2
                    r_mean = np.mean(radius[bin_mask])
                    
                    rotation_curve.append((r_mean, v_rot))
        
        # 转换为数组
        if rotation_curve:
            rotation_curve = np.array(rotation_curve)
        else:
            rotation_curve = np.array([(0, 0)])
            
        # 返回拟合结果
        result = {
            'pa': best_pa,
            'vsys': self.vsys,
            'vmax': self.vmax,
            'center': (center_x, center_y),
            'rotation_curve': rotation_curve
        }
        
        return result
    
    def calculate_kinematics(self) -> Dict:
        """
        计算动力学统计量
        
        Returns:
        --------
        动力学参数字典
        """
        # 获取有效的速度和弥散值
        vel = self.velocity_field.ravel()
        disp = self.dispersion_field.ravel()
        
        # 提取有效数据
        mask = np.isfinite(vel) & np.isfinite(disp) & (disp > 0)
        vel_valid = vel[mask]
        disp_valid = disp[mask]
        x_valid = self.x[mask]
        y_valid = self.y[mask]
        
        if len(vel_valid) == 0:
            warnings.warn("No valid velocity/dispersion values found")
            return {
                'sigma_mean': np.nan,
                'v_over_sigma': np.nan,
                'lambda_r': np.nan
            }
        
        # 计算平均弥散
        self.sigma_mean = np.mean(disp_valid)
        
        # 计算V/σ
        v_rms = np.sqrt(np.mean(vel_valid**2))
        self.v_over_sigma = v_rms / self.sigma_mean if self.sigma_mean > 0 else np.nan
        
        # 计算λR参数 (需要流量加权)
        # 这里简化处理，假设所有像素具有相同权重
        flux = np.ones_like(vel_valid)
        
        # 计算中心
        ny, nx = self.velocity_field.shape
        center_x = nx / 2
        center_y = ny / 2
        
        # 计算每个像素的半径
        r = np.sqrt((x_valid - center_x)**2 + (y_valid - center_y)**2)
        
        # λR = Σ(Fi*Ri*|Vi|) / Σ(Fi*Ri*sqrt(Vi^2 + σi^2))
        numerator = np.sum(flux * r * np.abs(vel_valid))
        denominator = np.sum(flux * r * np.sqrt(vel_valid**2 + disp_valid**2))
        
        self.lambda_r = numerator / denominator if denominator > 0 else np.nan
        
        # 返回结果
        result = {
            'sigma_mean': self.sigma_mean,
            'v_over_sigma': self.v_over_sigma,
            'lambda_r': self.lambda_r
        }
        
        return result
    
    def calculate_physical_scales(self) -> Dict:
        """
        计算物理尺度参数
        
        Returns:
        --------
        物理尺度参数字典
        """
        if self.distance is None:
            warnings.warn("Galaxy distance not provided, physical scales unavailable")
            return {'scale': np.nan, 'r_eff_kpc': np.nan}
        
        # 计算线性比例尺 (kpc/arcsec)
        scale = self.distance * 1000 * np.pi / (180 * 3600)  # kpc/arcsec
        
        # 计算像素物理尺寸
        pixel_kpc = scale * self.pixelsize
        
        # 计算有效半径
        ny, nx = self.velocity_field.shape
        
        # 简单估计：假设r_eff为图像尺寸的1/4
        r_eff_pix = min(ny, nx) / 4
        r_eff_kpc = r_eff_pix * pixel_kpc
        
        return {
            'scale': pixel_kpc,
            'r_eff_kpc': r_eff_kpc
        }