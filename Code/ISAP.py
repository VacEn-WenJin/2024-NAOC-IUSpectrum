'''
IFU Spectral Analysis Pipeline
'''

#!/usr/bin/env python
# coding: utf-8

"""
综合恒星+气体光谱拟合程序。
支持像素级拟合、沃罗诺伊分箱和径向分箱。
"""

import os
import sys
import time
import logging
import argparse
import warnings
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 第三方库
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from astropy.io import fits

# 尝试导入需要的模块
try:
    from ppxf.ppxf import ppxf, robust_sigma
    import ppxf.miles_util as miles
    import ppxf.ppxf_util as util
except ImportError as e:
    print(f"警告: 无法导入pPXF模块: {e}")
    print("请确保已安装pPXF库及其依赖项")
    print("可以使用: pip install ppxf")
    sys.exit(1)

try:
    from vorbin.voronoi_2d_binning import voronoi_2d_binning
    from vorbin.display_bins import display_bins
except ImportError as e:
    print(f"警告: 无法导入vorbin模块: {e}")
    print("请确保已安装vorbin库及其依赖项")
    print("可以使用: pip install vorbin")
    sys.exit(1)

# 忽略特定警告
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 配置日志记录
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)

# 工具函数
def to_scalar(array):
    """
    将numpy数组、列表或可迭代对象转换为标量。
    如果不可转换，原样返回。
    
    Parameters
    ----------
    array : array_like
        输入数组或标量
        
    Returns
    -------
    scalar or array
        转换后的标量或原始数组
    """
    try:
        if hasattr(array, '__iter__') and not isinstance(array, str):
            if len(array) == 1:
                return array[0]
            else:
                return array
        return array
    except (TypeError, ValueError):
        return array

def Apply_velocity_correction(Wavelength,z=0):
    """
    应用红移速度修正到波长
    
    Parameters
    ----------
    Wavelength : float or array_like
        输入波长
    z : float, optional
        红移
        
    Returns
    -------
    float or array_like
        修正后的波长
    """
    return Wavelength * (1+z)


### ------------------------------------------------- ###
# Configuration Class
### ------------------------------------------------- ###

class P2PConfig:
    """
    配置类：包含所有P2P分析所需的参数
    """
    
    def __init__(self, args=None):
        # 光速 (km/s)
        self.c = 299792.458
        
        # 默认路径
        self.base_dir = Path(".")
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "output"
        self.plot_dir = self.base_dir / "plots"
        
        # 默认文件名
        self.galaxy_name = "galaxy"
        self.data_file = None
        
        # 线程和进程数
        self.n_threads = max(1, os.cpu_count() // 2)
        
        # 图形设置
        self.make_plots = True
        self.no_plots = False
        self.plot_count = 0
        self.max_plots = 50
        self.dpi = 120
        self.LICplot = True  # 是否创建谱线指数图
        
        # 恒星和气体模板设置
        self.template_file = None
        self.template_dir = self.base_dir / "templates"
        self.use_miles = True
        self.library_dir = None
        self.ssp_template = "MILES_BASTI_CH_baseFe_BaSTI_T00-10.0"
        self.z_metal = [0.0]
        self.t_age = [8.0, 9.0, 10.0]
        
        # 波长范围设置
        self.lam_range_gal = None
        self.lam_range_temp = [3540, 7410]
        self.good_wavelength_range = [3700, 6800]
        
        # 拟合设置
        self.compute_errors = False
        self.compute_emission_lines = True
        self.compute_spectral_indices = True
        self.use_two_stage_fit = True
        self.retry_with_degree_zero = True
        self.global_search = False
        self.fallback_to_simple_fit = True
        
        # 并行模式
        self.parallel_mode = 'grouped'  # 'grouped' 或 'global'
        self.batch_size = 50  # 分组并行处理时每批的分箱数
        
        # 运动学设置
        self.vel_s = 0.0            # 初始恒星速度
        self.vel_dis_s = 100.0      # 初始恒星速度弥散度
        self.vel_gas = 0.0          # 初始气体速度
        self.vel_dis_gas = 100.0    # 初始气体速度弥散度
        self.redshift = 0.0         # 星系红移
        self.helio_vel = 0.0        # 日心修正速度
        self.moments = [2, 2]       # pPXF moments参数
        
        # 连续谱模式
        self.continuum_mode = "Cubic"
        
        # 气体发射线
        self.el_wave = None
        self.gas_names = ['OII3726', 'OII3729', 'Hgamma', 'OIII4363', 'HeII4686', 'Hbeta', 
                          'OIII5007', 'HeI5876', 'OI6300', 'Halpha', 'NII6583', 'SII6716', 'SII6731']
        self.line_indices = ['Fe5015', 'Fe5270', 'Fe5335', 'Mgb', 'Hbeta', 'Halpha']
        
        # 进度条设置
        self.progress_bar = True
        
        # 如果提供了参数，使用它们更新配置
        if args is not None:
            self.update_from_args(args)
    
    def update_from_args(self, args):
        """
        从命令行参数更新配置
        
        Parameters
        ----------
        args : argparse.Namespace
            命令行参数
        """
        # 基本参数
        if hasattr(args, 'data_dir') and args.data_dir:
            self.data_dir = Path(args.data_dir)
        
        if hasattr(args, 'output_dir') and args.output_dir:
            self.output_dir = Path(args.output_dir)
        
        if hasattr(args, 'galaxy_name') and args.galaxy_name:
            self.galaxy_name = args.galaxy_name
        
        if hasattr(args, 'data_file') and args.data_file:
            self.data_file = args.data_file
            
        # 线程数
        if hasattr(args, 'threads') and args.threads:
            self.n_threads = args.threads
            
        # 图形设置
        if hasattr(args, 'no_plots'):
            self.no_plots = args.no_plots
            if self.no_plots:
                self.make_plots = False
                
        if hasattr(args, 'max_plots') and args.max_plots is not None:
            self.max_plots = args.max_plots
            
        if hasattr(args, 'dpi') and args.dpi:
            self.dpi = args.dpi
        
        # 并行模式设置
        if hasattr(args, 'parallel_mode') and args.parallel_mode:
            self.parallel_mode = args.parallel_mode
            
        if hasattr(args, 'batch_size') and args.batch_size:
            self.batch_size = args.batch_size
            
        # 模板设置
        if hasattr(args, 'template_dir') and args.template_dir:
            self.template_dir = Path(args.template_dir)
            
        if hasattr(args, 'use_miles') and args.use_miles is not None:
            self.use_miles = args.use_miles
            
        if hasattr(args, 'template_file') and args.template_file:
            self.template_file = args.template_file
            
        # 红移和速度设置
        if hasattr(args, 'redshift') and args.redshift is not None:
            self.redshift = args.redshift
            
        # 处理其他可能的参数
        if hasattr(args, 'compute_emission_lines') and args.compute_emission_lines is not None:
            self.compute_emission_lines = args.compute_emission_lines
            
        if hasattr(args, 'compute_spectral_indices') and args.compute_spectral_indices is not None:
            self.compute_spectral_indices = args.compute_spectral_indices
            
        if hasattr(args, 'global_search') and args.global_search is not None:
            self.global_search = args.global_search
            
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
    
    def get_data_path(self):
        """
        获取数据文件路径
        
        Returns
        -------
        Path
            数据文件路径
        """
        if self.data_file:
            return self.data_dir / self.data_file
        else:
            # 默认查找与星系名相关的文件
            wildcard = f"{self.galaxy_name}*.fits"
            matching_files = list(self.data_dir.glob(wildcard))
            
            if not matching_files:
                # 尝试更宽松的搜索
                wildcard = "*.fits"
                matching_files = list(self.data_dir.glob(wildcard))
                
            if matching_files:
                return matching_files[0]
            else:
                raise FileNotFoundError(f"无法找到与 {self.galaxy_name} 匹配的FITS文件")
                
    def get_template_path(self):
        """
        获取模板文件路径
        
        Returns
        -------
        Path
            模板文件路径
        """
        if self.template_file:
            return self.template_dir / self.template_file
        else:
            # 查找可能的模板文件
            wildcard = "*.fits"
            matching_files = list(self.template_dir.glob(wildcard))
            
            if matching_files:
                return matching_files[0]
            else:
                if self.use_miles:
                    return None  # 使用MILES内置库
                else:
                    raise FileNotFoundError("无法找到模板文件")
                    
    def __str__(self):
        """
        输出配置信息
        
        Returns
        -------
        str
            配置信息字符串
        """
        info = [
            f"Galaxy: {self.galaxy_name} (z={self.redshift})",
            f"Data Directory: {self.data_dir}",
            f"Output Directory: {self.output_dir}",
            f"Template: {'MILES (internal)' if self.use_miles else self.template_file}",
            f"Parallel Mode: {self.parallel_mode}",
            f"Threads: {self.n_threads}",
            f"Compute Emission Lines: {self.compute_emission_lines}",
            f"Compute Spectral Indices: {self.compute_spectral_indices}",
            f"Use Two-Stage Fit: {self.use_two_stage_fit}"
        ]
        return "\n".join(info)


### ------------------------------------------------- ###
# Data Loading Class
### ------------------------------------------------- ###

class IFUDataCube:
    """
    IFU数据立方体处理类。
    
    此类负责读取FITS格式的数据立方体，并进行初步处理。
    """
    
    def __init__(self, filename, lam_range_temp, redshift, config):
        """
        初始化IFU数据立方体。
        
        Parameters
        ----------
        filename : str or Path
            FITS文件路径
        lam_range_temp : list
            模板波长范围
        redshift : float
            星系红移
        config : P2PConfig
            配置对象
        """
        self.filename = str(filename)
        self.lam_range_temp = lam_range_temp
        self.redshift = redshift
        self.config = config
        
        # 初始化属性
        self.cube = None
        self.header = None
        self.lam_gal = None
        self.spectra = None
        self.variance = None
        self.signal = None
        self.noise = None
        self.x = None
        self.y = None
        self.row = None
        self.col = None
        self.velscale = None
        self.pixsize_x = None  # 像素大小（X方向）
        self.pixsize_y = None  # 像素大小（Y方向）
        self.CD1_1 = 1.0
        self.CD1_2 = 0.0
        self.CD2_1 = 0.0
        self.CD2_2 = 1.0
        self.CRVAL1 = 0.0
        self.CRVAL2 = 0.0
        
        # 初始化结果映射
        self.velfield = None
        self.sigfield = None
        self.bin_map = None
        self.el_flux_maps = {}
        self.el_snr_maps = {}
        self.index_maps = {}
        
        for name in config.gas_names:
            self.el_flux_maps[name] = None
            self.el_snr_maps[name] = None
            
        for name in config.line_indices:
            self.index_maps[name] = None
        
        # 读取和处理数据
        self._read_data()
        self._preprocess_data()
    
    def _read_data(self):
        """
        读取FITS文件数据。
        """
        try:
            logging.info(f"读取数据文件: {self.filename}")
            with fits.open(self.filename) as hdul:
                # 读取主HDU
                primary_hdu = hdul[0]
                self.header = primary_hdu.header
                
                # 判断数据类型
                if primary_hdu.data is not None and len(primary_hdu.data.shape) == 3:
                    # 直接是数据立方体
                    self.cube = primary_hdu.data
                elif len(hdul) > 1:
                    # 数据在扩展HDU中
                    for hdu in hdul[1:]:
                        if hdu.data is not None and len(hdu.data.shape) == 3:
                            self.cube = hdu.data
                            # 合并头文件
                            for key in hdu.header:
                                if key not in self.header and key not in ('XTENSION', 'BITPIX', 'NAXIS', 'PCOUNT', 'GCOUNT'):
                                    self.header[key] = hdu.header[key]
                            break
                
                # 查找方差扩展
                for i, hdu in enumerate(hdul):
                    if 'VARIANCE' in hdu.name.upper() or 'STAT' in hdu.name.upper() or 'ERROR' in hdu.name.upper():
                        if hdu.data is not None and len(hdu.data.shape) == 3:
                            self.variance = hdu.data
                            logging.info(f"发现方差扩展: {hdu.name}")
                            break
                
                # 如果没有找到立方体数据
                if self.cube is None:
                    raise ValueError("无法在FITS文件中找到数据立方体")
                
                # 提取头文件中的基本信息
                # 提取头文件中的基本信息
                if 'CD1_1' in self.header: self.CD1_1 = self.header['CD1_1']
                if 'CD1_2' in self.header: self.CD1_2 = self.header['CD1_2']
                if 'CD2_1' in self.header: self.CD2_1 = self.header['CD2_1']
                if 'CD2_2' in self.header: self.CD2_2 = self.header['CD2_2']
                if 'CRVAL1' in self.header: self.CRVAL1 = self.header['CRVAL1']
                if 'CRVAL2' in self.header: self.CRVAL2 = self.header['CRVAL2']
                
                # 提取像素大小
                if 'CDELT1' in self.header:
                    self.pixsize_x = abs(self.header['CDELT1'])
                elif 'CD1_1' in self.header:
                    self.pixsize_x = abs(self.header['CD1_1'])
                else:
                    self.pixsize_x = 1.0

                if 'CDELT2' in self.header:
                    self.pixsize_y = abs(self.header['CDELT2'])
                elif 'CD2_2' in self.header:
                    self.pixsize_y = abs(self.header['CD2_2'])
                else:
                    self.pixsize_y = 1.0
                
                logging.info(f"数据立方体形状: {self.cube.shape}")
                
                # 计算波长轴
                nw = self.cube.shape[0]
                if 'CRVAL3' in self.header and 'CDELT3' in self.header and 'CRPIX3' in self.header:
                    crval3 = self.header['CRVAL3']
                    cdelt3 = self.header['CDELT3']
                    crpix3 = self.header['CRPIX3']
                    self.lam_gal = np.arange(nw) * cdelt3 + (crval3 - crpix3 * cdelt3)
                else:
                    # 使用默认波长范围
                    self.lam_gal = np.linspace(4800, 5500, nw)
                    logging.warning("未在头文件中找到波长信息，使用默认值")
                
                # 计算速度比例
                lam_gal_log = np.log(self.lam_gal)
                self.velscale = (lam_gal_log[1] - lam_gal_log[0]) * self.config.c
                
                logging.info(f"速度比例: {self.velscale:.2f} km/s/pix")
                
        except Exception as e:
            logging.error(f"读取FITS文件时出错: {str(e)}")
            if "无法在FITS文件中找到数据立方体" in str(e):
                logging.error("文件结构不符合预期格式。请检查FITS文件的结构。")
            raise
    
    def _preprocess_data(self):
        """
        预处理数据立方体，提取光谱和计算信噪比。
        """
        try:
            # 获取数据维度
            nw, ny, nx = self.cube.shape
            
            # 整理光谱数据
            self.spectra = np.reshape(self.cube, (nw, ny * nx))
            
            # 整理方差数据
            if self.variance is not None:
                self.variance = np.reshape(self.variance, (nw, ny * nx))
            
            # 创建坐标数组
            y_coords, x_coords = np.indices((ny, nx))
            self.x = (x_coords + 1).flatten()  # 从1开始计数
            self.y = (y_coords + 1).flatten()
            self.row = self.y  # 行索引等于y
            self.col = self.x  # 列索引等于x
            
            # 计算信噪比
            self._calculate_snr()
            
            logging.info(f"波长范围: {self.lam_gal[0]:.1f} - {self.lam_gal[-1]:.1f} Å")
            logging.info(f"像素总数: {ny * nx}")
            
        except Exception as e:
            logging.error(f"预处理数据时出错: {str(e)}")
            raise
    
    def _calculate_snr(self):
        """
        计算每个像素的信噪比。
        """
        try:
            nw, npix = self.spectra.shape
            
            # 波长范围限制
            w1, w2 = self.lam_range_temp
            mask = (self.lam_gal > w1) & (self.lam_gal < w2)
            
            # 初始化信号和噪声数组
            self.signal = np.zeros(npix)
            self.noise = np.zeros(npix)
            
            for i in range(npix):
                # 获取光谱中心区域
                spectrum = self.spectra[mask, i]
                
                # 如果方差可用，直接使用
                if self.variance is not None:
                    pixel_var = self.variance[mask, i]
                    pixel_noise = np.sqrt(np.median(pixel_var[pixel_var > 0]))
                else:
                    # 否则从光谱估计
                    pixel_noise = robust_sigma(spectrum)
                
                # 计算信号（光谱中值）
                pixel_signal = np.median(spectrum)
                
                # 保存结果
                self.signal[i] = pixel_signal
                self.noise[i] = pixel_noise if pixel_noise > 0 else 1.0
                
        except Exception as e:
            logging.error(f"计算信噪比时出错: {str(e)}")
            # 创建默认值
            npix = self.spectra.shape[1]
            self.signal = np.ones(npix)
            self.noise = np.ones(npix)


### ------------------------------------------------- ###
# Template Preparation Functions
### ------------------------------------------------- ###

def prepare_templates(config, velscale):
    """
    准备恒星和气体模板。
    
    Parameters
    ----------
    config : P2PConfig
        配置对象
    velscale : float
        速度比例（km/s/像素）
        
    Returns
    -------
    tuple
        (sps, gas_templates, gas_names, line_wave)
    """
    # 1. 准备恒星模板
    if config.use_miles:
        sps = prepare_miles_templates(config, velscale)
    else:
        sps = prepare_custom_templates(config, velscale)
    
    # 2. 准备气体模板
    if config.compute_emission_lines:
        gas_templates, gas_names, line_wave = prepare_gas_templates(config, sps)
    else:
        gas_templates = np.zeros((sps.templates.shape[0], 0))
        gas_names = []
        line_wave = []
    
    return sps, gas_templates, gas_names, line_wave

def prepare_miles_templates(config, velscale):
    """
    准备MILES模板库。
    
    Parameters
    ----------
    config : P2PConfig
        配置对象
    velscale : float
        速度比例（km/s/像素）
        
    Returns
    -------
    object
        具有templates、ln_lam_temp和lam_temp属性的对象
    """
    logging.info("使用MILES恒星模板库...")
    
    # 选择MILES模型
    miles_model = config.ssp_template
    
    # 金属量和年龄网格
    z_metal = config.z_metal
    t_age = config.t_age
    
    # 创建MILES实例
    try:
        miles_sps = miles.MilesSsp(miles_model, velscale, z_metal, t_age)
        logging.info(f"MILES模板形状: {miles_sps.templates.shape}")
        return miles_sps
    except Exception as e:
        logging.error(f"无法创建MILES模板: {str(e)}")
        raise

def prepare_custom_templates(config, velscale):
    """
    准备自定义模板库。
    
    Parameters
    ----------
    config : P2PConfig
        配置对象
    velscale : float
        速度比例（km/s/像素）
        
    Returns
    -------
    object
        具有templates、ln_lam_temp和lam_temp属性的对象
    """
    logging.info("使用自定义恒星模板...")
    
    try:
        # 获取模板文件路径
        template_path = config.get_template_path()
        
        # 读取模板文件
        with fits.open(template_path) as hdul:
            template_data = hdul[0].data
            template_header = hdul[0].header
        
        # 提取波长信息
        nw = template_data.shape[0]
        if 'CRVAL1' in template_header and 'CDELT1' in template_header and 'CRPIX1' in template_header:
            crval1 = template_header['CRVAL1']
            cdelt1 = template_header['CDELT1']
            crpix1 = template_header['CRPIX1']
            lam_temp = np.arange(nw) * cdelt1 + (crval1 - crpix1 * cdelt1)
        else:
            # 使用默认波长范围
            lam_temp = np.linspace(3500, 7500, nw)
            logging.warning("未在模板头文件中找到波长信息，使用默认值")
        
        # 将模板重采样到恒定的对数间隔（与PPXF要求一致）
        ln_lam_temp = np.log(lam_temp)
        new_ln_lam_temp = np.arange(ln_lam_temp[0], ln_lam_temp[-1], velscale/config.c)
        templates = util.log_rebin(lam_temp, template_data, velscale=velscale)[0]
        
        # 创建与MILES类似的对象
        class CustomSps:
            pass
        
        custom_sps = CustomSps()
        custom_sps.templates = templates
        custom_sps.ln_lam_temp = new_ln_lam_temp
        custom_sps.lam_temp = np.exp(new_ln_lam_temp)
        
        logging.info(f"自定义模板形状: {templates.shape}")
        return custom_sps
        
    except Exception as e:
        logging.error(f"无法创建自定义模板: {str(e)}")
        raise

def prepare_gas_templates(config, sps):
    """
    准备气体发射线模板。
    
    Parameters
    ----------
    config : P2PConfig
        配置对象
    sps : object
        恒星模板对象，包含ln_lam_temp和lam_temp
        
    Returns
    -------
    tuple
        (gas_templates, gas_names, line_wave)
    """
    logging.info("准备气体发射线模板...")
    
    # 准备气体发射线
    try:
        # 发射线列表
        line_names = config.gas_names
        
        # 如果指定了自定义发射线波长
        if config.el_wave is not None:
            line_wave = config.el_wave
        else:
            # 使用默认发射线波长
            line_wave = {
                'OII3726': 3726.03,
                'OII3729': 3728.82,
                'Hgamma': 4340.47,
                'OIII4363': 4363.21,
                'HeII4686': 4685.7,
                'Hbeta': 4861.33,
                'OIII4959': 4958.92,
                'OIII5007': 5006.84,
                'HeI5876': 5875.67,
                'OI6300': 6300.30,
                'Halpha': 6562.80,
                'NII6548': 6548.03,
                'NII6583': 6583.41,
                'SII6716': 6716.47,
                'SII6731': 6730.85
            }
            
            # 转换为数组
            line_wave = np.array([line_wave[name] for name in line_names if name in line_wave])
            
            # 如果一些名字不在默认字典中，报警告
            missing_lines = [name for name in line_names if name not in line_wave]
            if missing_lines:
                logging.warning(f"未知发射线: {missing_lines}")
        
        # 使用ppxf_util的emission_lines函数创建气体模板
        gas_templates, gas_names, line_wave = util.emission_lines(
            sps.ln_lam_temp, line_wave, FWHM=config.vel_dis_gas)
        
        logging.info(f"气体模板形状: {gas_templates.shape}, 发射线数量: {len(gas_names)}")
        return gas_templates, gas_names, line_wave
    
    except Exception as e:
        logging.error(f"无法创建气体模板: {str(e)}")
        # 返回空模板
        gas_templates = np.zeros((sps.templates.shape[0], 0))
        return gas_templates, [], []


### ------------------------------------------------- ###
# Line Index Calculator
### ------------------------------------------------- ###

class LineIndexCalculator:
    """
    谱线指数计算器。
    
    计算常见光谱特征的谱线指数，如Lick指数。
    """
    
    def __init__(self, wavelength, flux, template_wave=None, template_flux=None, 
                 em_wave=None, em_flux_list=None, velocity_correction=0,
                 continuum_mode="Cubic"):
        """
        初始化谱线指数计算器。
        
        Parameters
        ----------
        wavelength : array
            观测波长
        flux : array
            观测光通量
        template_wave : array, optional
            模板波长
        template_flux : array, optional
            模板光通量
        em_wave : array, optional
            发射线波长
        em_flux_list : array, optional
            发射线光通量
        velocity_correction : float, optional
            速度修正（应用于定义带通滤波器的波长）
        continuum_mode : str, optional
            连续谱计算模式，"Cubic"（默认）或"Linear"
        """
        self.wavelength = wavelength
        self.flux = flux
        self.template_wave = template_wave
        self.template_flux = template_flux
        self.velocity_correction = velocity_correction
        self.continuum_mode = continuum_mode
        
        # 发射线信息
        self.em_wave = em_wave
        self.em_flux = em_flux_list
        
        # 谱线指数定义
        self.define_indices()
    
    def define_indices(self):
        """
        定义常见谱线指数的波长范围。
        """
        # 定义一些常见的谱线指数定义（蓝，中心，红连续谱范围）
        self.indices = {
            # Lick指数
            'Hbeta': {'blue': (4827.875, 4847.875), 'band': (4847.875, 4876.625), 'red': (4876.625, 4891.625)},
            'Mgb': {'blue': (5142.625, 5161.375), 'band': (5160.125, 5192.625), 'red': (5191.375, 5206.375)},
            'Fe5015': {'blue': (4946.500, 4977.750), 'band': (4977.750, 5054.000), 'red': (5054.000, 5065.250)},
            'Fe5270': {'blue': (5233.150, 5248.150), 'band': (5245.650, 5285.650), 'red': (5285.650, 5318.150)},
            'Fe5335': {'blue': (5304.625, 5315.875), 'band': (5312.125, 5352.125), 'red': (5353.375, 5363.375)},
            
            # 其他常见指数
            'Halpha': {'blue': (6510.0, 6540.0), 'band': (6554.0, 6568.0), 'red': (6575.0, 6585.0)},
            'Na D': {'blue': (5860.625, 5875.625), 'band': (5876.875, 5909.375), 'red': (5922.125, 5948.125)},
            'TiO 1': {'blue': (5936.625, 5994.125), 'band': (5937.875, 5994.875), 'red': (6038.625, 6103.625)},
            'TiO 2': {'blue': (6066.625, 6141.625), 'band': (6189.625, 6272.125), 'red': (6372.625, 6415.125)},
            'Ca H&K': {'blue': (3806.5, 3833.8), 'band': (3899.5, 4003.5), 'red': (4019.8, 4051.2)}
        }
    
    def calculate_index(self, index_name):
        """
        计算指定谱线指数的值。
        
        Parameters
        ----------
        index_name : str
            谱线指数名称
            
        Returns
        -------
        float
            谱线指数值
        """
        if index_name not in self.indices:
            raise ValueError(f"未知谱线指数: {index_name}")
        
        # 获取指数定义
        index_def = self.indices[index_name]
        
        # 应用速度修正
        vel_corr_factor = 1 + self.velocity_correction / 299792.458
        blue_range = (index_def['blue'][0] * vel_corr_factor, index_def['blue'][1] * vel_corr_factor)
        band_range = (index_def['band'][0] * vel_corr_factor, index_def['band'][1] * vel_corr_factor)
        red_range = (index_def['red'][0] * vel_corr_factor, index_def['red'][1] * vel_corr_factor)
        
        # 计算连续谱
        if self.continuum_mode.lower() == "cubic":
            continuum = self._calculate_cubic_continuum(blue_range, band_range, red_range)
        else:  # linear
            continuum = self._calculate_linear_continuum(blue_range, band_range, red_range)
        
        # 计算带通区域的光通量和连续谱
        band_mask = (self.wavelength >= band_range[0]) & (self.wavelength <= band_range[1])
        
        if not np.any(band_mask):
            logging.warning(f"带通区域 {band_range} 没有数据点")
            return np.nan
        
        band_flux = self.flux[band_mask]
        band_wave = self.wavelength[band_mask]
        band_continuum = continuum[band_mask]
        
        # 计算指数值
        if index_name in ['Fe5015', 'Fe5270', 'Fe5335', 'Mgb', 'Na D', 'TiO 1', 'TiO 2']:
            # 等值宽度指数 (EW)
            dwave = np.abs(np.diff(np.append(band_wave, 2*band_wave[-1]-band_wave[-2])))
            index_value = np.sum((1 - band_flux / band_continuum) * dwave)
        else:
            # 磁带指数 (如Hbeta)
            index_value = -2.5 * np.log10(np.mean(band_flux) / np.mean(band_continuum))
        
        return index_value
    
    def _calculate_linear_continuum(self, blue_range, band_range, red_range):
        """
        计算线性连续谱。
        
        Parameters
        ----------
        blue_range : tuple
            蓝侧连续谱范围
        band_range : tuple
            带通滤波器范围
        red_range : tuple
            红侧连续谱范围
            
        Returns
        -------
        array
            全波长范围的连续谱估计
        """
        # 蓝侧连续谱
        blue_mask = (self.wavelength >= blue_range[0]) & (self.wavelength <= blue_range[1])
        if not np.any(blue_mask):
            logging.warning(f"蓝侧连续谱区域 {blue_range} 没有数据点")
            return np.ones_like(self.wavelength) * np.nanmean(self.flux)
            
        blue_flux = np.nanmean(self.flux[blue_mask])
        blue_wave = np.nanmean(self.wavelength[blue_mask])
        
        # 红侧连续谱
        red_mask = (self.wavelength >= red_range[0]) & (self.wavelength <= red_range[1])
        if not np.any(red_mask):
            logging.warning(f"红侧连续谱区域 {red_range} 没有数据点")
            return np.ones_like(self.wavelength) * np.nanmean(self.flux)
            
        red_flux = np.nanmean(self.flux[red_mask])
        red_wave = np.nanmean(self.wavelength[red_mask])
        
        # 线性插值
        slope = (red_flux - blue_flux) / (red_wave - blue_wave)
        continuum = blue_flux + slope * (self.wavelength - blue_wave)
        
        return continuum
    
    def _calculate_cubic_continuum(self, blue_range, band_range, red_range):
        """
        计算三次样条连续谱。
        
        Parameters
        ----------
        blue_range : tuple
            蓝侧连续谱范围
        band_range : tuple
            带通滤波器范围
        red_range : tuple
            红侧连续谱范围
            
        Returns
        -------
        array
            全波长范围的连续谱估计
        """
        # 蓝侧连续谱区域
        blue_mask = (self.wavelength >= blue_range[0]) & (self.wavelength <= blue_range[1])
        
        # 红侧连续谱区域
        red_mask = (self.wavelength >= red_range[0]) & (self.wavelength <= red_range[1])
        
        # 检查是否有足够的数据点
        if not np.any(blue_mask) or not np.any(red_mask):
            logging.warning(f"连续谱区域缺少数据点：蓝侧={np.sum(blue_mask)}, 红侧={np.sum(red_mask)}")
            # 回退到线性连续谱
            return self._calculate_linear_continuum(blue_range, band_range, red_range)
        
        # 连接蓝侧和红侧连续谱区域
        cont_mask = np.logical_or(blue_mask, red_mask)
        
        # 连续谱波长和光通量
        cont_wave = self.wavelength[cont_mask]
        cont_flux = self.flux[cont_mask]
        
        # 使用三次样条插值
        try:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(cont_wave, cont_flux)
            continuum = cs(self.wavelength)
        except:
            # 回退到scipy.interpolate的使用
            try:
                from scipy.interpolate import interp1d
                f = interp1d(cont_wave, cont_flux, kind='cubic', bounds_error=False,
                            fill_value=(cont_flux[0], cont_flux[-1]))
                continuum = f(self.wavelength)
            except:
                # 最终回退到线性插值
                logging.warning("立方插值失败，回退到线性连续谱")
                continuum = self._calculate_linear_continuum(blue_range, band_range, red_range)
        
        return continuum
    
    def plot_all_lines(self, mode='P2P', number=None):
        """
        绘制所有定义的谱线指数的图形。
        
        Parameters
        ----------
        mode : str, optional
            模式（"P2P"，"VNB"或"RDB"）
        number : int, optional
            像素或分箱编号
            
        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # 创建图形
        n_indices = len(self.indices)
        n_cols = min(3, n_indices)
        n_rows = (n_indices + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(4*n_cols, 3*n_rows), dpi=100)
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        # 绘制每个指数
        for i, (name, index_def) in enumerate(self.indices.items()):
            # 计算行列位置
            row = i // n_cols
            col = i % n_cols
            
            # 创建子图
            ax = fig.add_subplot(gs[row, col])
            
            # 应用速度修正
            vel_corr_factor = 1 + self.velocity_correction / 299792.458
            blue_range = (index_def['blue'][0] * vel_corr_factor, index_def['blue'][1] * vel_corr_factor)
            band_range = (index_def['band'][0] * vel_corr_factor, index_def['band'][1] * vel_corr_factor)
            red_range = (index_def['red'][0] * vel_corr_factor, index_def['red'][1] * vel_corr_factor)
            
            # 计算连续谱
            if self.continuum_mode.lower() == "cubic":
                continuum = self._calculate_cubic_continuum(blue_range, band_range, red_range)
            else:  # linear
                continuum = self._calculate_linear_continuum(blue_range, band_range, red_range)
            
            # 确定波长范围
            min_wave = min(blue_range[0], band_range[0], red_range[0])
            max_wave = max(blue_range[1], band_range[1], red_range[1])
            
            # 稍微扩大范围
            range_width = max_wave - min_wave
            display_min = min_wave - 0.05 * range_width
            display_max = max_wave + 0.05 * range_width
            
            # 仅显示所需波长范围
            plot_mask = (self.wavelength >= display_min) & (self.wavelength <= display_max)
            wave_plot = self.wavelength[plot_mask]
            flux_plot = self.flux[plot_mask]
            continuum_plot = continuum[plot_mask]
            
            # 检查是否有有效数据
            if len(wave_plot) == 0:
                ax.set_title(f"{name} (no data)")
                continue
            
            # 绘制原始光谱
            ax.plot(wave_plot, flux_plot, color='black', lw=1, alpha=0.7, label='Observed')
            
            # 绘制连续谱
            ax.plot(wave_plot, continuum_plot, color='red', lw=1, ls='--', alpha=0.8, label='Continuum')
            
            # 绘制模板光谱（如果有）
            if self.template_wave is not None and self.template_flux is not None:
                mask_temp = (self.template_wave >= display_min) & (self.template_wave <= display_max)
                if np.any(mask_temp):
                    ax.plot(self.template_wave[mask_temp], self.template_flux[mask_temp], 
                            color='green', lw=1, alpha=0.5, label='Model')
            
            # 绘制连续谱范围
            ax.axvspan(blue_range[0], blue_range[1], color='blue', alpha=0.1)
            ax.axvspan(red_range[0], red_range[1], color='red', alpha=0.1)
            
            # 绘制带通滤波器范围
            ax.axvspan(band_range[0], band_range[1], color='green', alpha=0.1)
            
            # 计算指数值
            try:
                index_value = self.calculate_index(name)
                index_text = f"{name} = {index_value:.2f} Å"
            except:
                index_text = f"{name} = N/A"
            
            # 标题和标签
            ax.set_title(index_text)
            ax.set_xlabel('Wavelength [Å]')
            ax.set_ylabel('Flux')
            
            if i == 0:  # 只在第一个子图添加图例
                ax.legend(loc='upper right', fontsize='small')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        if mode == 'P2P':
            save_path = f"plots/LIC_{mode}_pixel{number}.png"
        elif mode == 'VNB':
            save_path = f"plots/LIC_{mode}_bin{number}.png"
        elif mode == 'RDB':
            save_path = f"plots/LIC_{mode}_ring{number}.png"
        else:
            save_path = f"plots/LIC_indices.png"
            
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        
        logging.info(f"谱线指数图保存到: {save_path}")


### ------------------------------------------------- ###
# Voronoi Binning Implementation
### ------------------------------------------------- ###

class VoronoiBinning:
    """
    基于Voronoi分箱的光谱分析类。
    
    使用Voronoi tessellation将像素分组为具有目标信噪比的区域，
    然后对每个分箱进行单一光谱拟合。
    """
    
    def __init__(self, galaxy_data, config):
        """
        初始化Voronoi分箱。
        
        Parameters
        ----------
        galaxy_data : IFUDataCube
            包含星系数据的对象
        config : P2PConfig
            配置对象
        """
        self.galaxy_data = galaxy_data
        self.config = config
        self.bin_data = None
        self.bin_results = {}
        
        # 存储分箱映射和结果的数组
        ny, nx = galaxy_data.cube.shape[1:3]
        self.bin_map = np.full((ny, nx), -1)  # -1表示未分箱的像素
        self.bin_signal = np.full((ny, nx), np.nan)
        self.bin_noise = np.full((ny, nx), np.nan)
        self.bin_snr = np.full((ny, nx), np.nan)
        
        # 创建结果映射
        self.velfield = np.full((ny, nx), np.nan)
        self.sigfield = np.full((ny, nx), np.nan)
        
        # 保存信噪比图
        galaxy_data.bin_map = self.bin_map.copy()
        
        # 发射线和指数映射
        self.el_flux_maps = {}
        self.el_snr_maps = {}
        self.index_maps = {}
        
        for name in config.gas_names:
            self.el_flux_maps[name] = np.full((ny, nx), np.nan)
            self.el_snr_maps[name] = np.full((ny, nx), np.nan)
            
        for name in config.line_indices:
            self.index_maps[name] = np.full((ny, nx), np.nan)
    
    def create_bins(self, target_snr=20, pixsize=None, cores=None):
        """
        创建Voronoi分箱。
        
        Parameters
        ----------
        target_snr : float
            目标信噪比
        pixsize : float, optional
            像素大小
        cores : int, optional
            用于并行计算的核心数
            
        Returns
        -------
        int
            分箱的数量
        """
        logging.info(f"===== 创建Voronoi分箱 (目标SNR: {target_snr}) =====")
        
        # 获取数据维度
        ny, nx = self.galaxy_data.cube.shape[1:3]
        
        # 提取坐标和信噪比数据
        x = self.galaxy_data.x
        y = self.galaxy_data.y
        signal = self.galaxy_data.signal
        noise = self.galaxy_data.noise
        
        # 计算信噪比
        snr = np.divide(signal, noise, out=np.zeros_like(signal), where=noise>0)
        
        # 创建有效像素的掩码（SNR>0）
        mask = (snr > 0) & np.isfinite(snr)
        x_good = x[mask]
        y_good = y[mask]
        signal_good = signal[mask]
        noise_good = noise[mask]
        
        # 确保我们有足够的像素进行分箱
        if len(x_good) < 10:
            logging.error("不足够的有效像素用于Voronoi分箱")
            return 0
            
        logging.info(f"使用 {len(x_good)}/{len(x)} 个有效像素进行Voronoi分箱")
        
        try:
            # 执行Voronoi分箱
            start_time = time.time()
            logging.info(f"开始Voronoi分箱计算...")
            
            # 设置像素大小
            if pixsize is None:
                pixsize = 0.5 * (self.galaxy_data.pixsize_x + self.galaxy_data.pixsize_y)
                
            # 执行Voronoi分箱
            bin_num, x_gen, y_gen, x_bar, y_bar, sn, n_pixels, scale = voronoi_2d_binning(
                x_good, y_good, signal_good, noise_good, 
                target_snr, pixsize=pixsize, plot=False, quiet=True,
                cvt=True, wvt=True, cores=cores)
            
            # 计算分箱耗时
            end_time = time.time()
            logging.info(f"Voronoi分箱完成: {np.max(bin_num)+1} 个分箱创建，用时 {end_time - start_time:.1f} 秒")
            
            # 创建完整的分箱映射（包括未使用的像素）
            full_bin_map = np.full(len(x), -1)  # 默认为-1（未使用）
            full_bin_map[mask] = bin_num
            
            # 重新构建2D分箱映射
            bin_map_2d = np.full((ny, nx), -1)
            for i, (r, c) in enumerate(zip(self.galaxy_data.row, self.galaxy_data.col)):
                if full_bin_map[i] >= 0:  # 只映射有效分箱
                    # 注意：坐标从1开始，需要-1转换为数组索引
                    bin_map_2d[r-1, c-1] = full_bin_map[i]
            
            # 保存分箱映射
            self.bin_map = bin_map_2d
            self.galaxy_data.bin_map = bin_map_2d.copy()
            
            # 保存分箱信息
            self.n_bins = np.max(bin_num) + 1
            self.x_bar = x_bar
            self.y_bar = y_bar
            self.bin_snr = sn
            self.n_pixels = n_pixels
            
            # 返回分箱数量
            return self.n_bins
            
        except Exception as e:
            logging.error(f"Voronoi分箱过程中出错: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return 0
    
    def extract_bin_spectra(self, p2p_velfield=None):
        """
        从分箱中提取合并的光谱。
        
        Parameters
        ----------
        p2p_velfield : ndarray, optional
            P2P分析得到的速度场，用于修正光谱。如果为None，则从galaxy_data中获取
            
        Returns
        -------
        dict
            包含合并光谱的字典
        """
        logging.info(f"===== 提取 {self.n_bins} 个分箱的合并光谱 =====")
        
        # 获取数据维度
        ny, nx = self.galaxy_data.cube.shape[1:3]
        npix = self.galaxy_data.spectra.shape[0]
        
        # 获取P2P速度场，优先使用传入参数，否则从galaxy_data获取
        if p2p_velfield is None:
            if hasattr(self.galaxy_data, 'velfield') and self.galaxy_data.velfield is not None:
                velfield = self.galaxy_data.velfield
                logging.info("使用galaxy_data中的速度场进行光谱修正")
            else:
                velfield = None
                logging.info("未找到速度场，不进行速度修正")
        else:
            velfield = p2p_velfield
            logging.info("使用传入的P2P速度场进行光谱修正")
        
        # 检查velfield是否有效
        if velfield is not None and not np.all(np.isnan(velfield)):
            apply_vel_correction = True
            logging.info("启用速度修正")
        else:
            apply_vel_correction = False
            logging.info("不进行速度修正 - 未找到有效的速度场")
        
        # 创建共同的波长网格（用于重采样）
        lam_gal = self.galaxy_data.lam_gal
        
        # 初始化合并光谱字典
        bin_spectra = {}
        bin_variances = {}
        bin_positions = {}
        
        # 为每个分箱创建合并光谱
        for bin_id in range(self.n_bins):
            # 找到属于这个分箱的所有像素
            bin_mask = (self.bin_map == bin_id)
            
            if not np.any(bin_mask):
                logging.warning(f"分箱 {bin_id} 没有包含像素")
                continue
                
            # 获取这个分箱中所有像素的行列索引
            rows, cols = np.where(bin_mask)
            
            # 初始化累积光谱和权重
            coadded_spectrum = np.zeros(npix)
            coadded_variance = np.zeros(npix)
            total_weight = 0
            
            # 处理每个像素
            for r, c in zip(rows, cols):
                k_index = r * nx + c
                
                # 获取原始光谱
                pixel_spectrum = self.galaxy_data.spectra[:, k_index]
                
                # 如果可以获取方差数据，则使用它，否则创建统一方差
                if hasattr(self.galaxy_data, 'variance'):
                    pixel_variance = self.galaxy_data.variance[:, k_index]
                else:
                    pixel_variance = np.ones_like(pixel_spectrum)
                
                # 计算当前像素的权重（使用信噪比）
                if hasattr(self.galaxy_data, 'signal') and hasattr(self.galaxy_data, 'noise'):
                    if k_index < len(self.galaxy_data.signal):
                        signal = self.galaxy_data.signal[k_index]
                        noise = self.galaxy_data.noise[k_index]
                        weight = (signal / noise)**2 if noise > 0 else 0
                    else:
                        weight = 1.0
                else:
                    weight = 1.0
                
                # 应用速度修正（如果可用）
                if apply_vel_correction and not np.isnan(velfield[r, c]):
                    vel = velfield[r, c]
                    
                    # 修正后的波长
                    lam_shifted = lam_gal * (1 + vel/self.config.c)
                    
                    # 重采样到原始波长网格
                    corrected_spectrum = np.interp(lam_gal, lam_shifted, pixel_spectrum,
                                                 left=0, right=0)
                    corrected_variance = np.interp(lam_gal, lam_shifted, pixel_variance,
                                                 left=np.inf, right=np.inf)
                    
                    # 累积修正后的光谱（加权）
                    coadded_spectrum += corrected_spectrum * weight
                    coadded_variance += corrected_variance * weight**2
                else:
                    # 不修正，直接累积
                    coadded_spectrum += pixel_spectrum * weight
                    coadded_variance += pixel_variance * weight**2
                
                total_weight += weight
            
            # 归一化累积光谱
            if total_weight > 0:
                merged_spectrum = coadded_spectrum / total_weight
                merged_variance = coadded_variance / (total_weight**2)
            else:
                logging.warning(f"分箱 {bin_id} 的总权重为零，使用简单平均")
                merged_spectrum = coadded_spectrum / len(rows) if len(rows) > 0 else coadded_spectrum
                merged_variance = coadded_variance / (len(rows)**2) if len(rows) > 0 else coadded_variance
            
            # 存储合并的数据
            bin_spectra[bin_id] = merged_spectrum
            bin_variances[bin_id] = merged_variance
            
            # 保存分箱的位置信息
            bin_positions[bin_id] = {
                'x': np.mean(cols),  # 列对应x
                'y': np.mean(rows),  # 行对应y
                'n_pixels': len(rows)
            }
            
            # 添加SNR信息
            snr = np.median(merged_spectrum / np.sqrt(merged_variance))
            bin_positions[bin_id]['snr'] = snr
            
            # 记录信息
            if bin_id % 50 == 0 or bin_id == self.n_bins - 1:
                logging.info(f"已提取 {bin_id+1}/{self.n_bins} 个分箱的光谱，SNR={snr:.1f}")
        
        # 保存提取的数据
        self.bin_data = {
            'spectra': bin_spectra,
            'variances': bin_variances,
            'positions': bin_positions
        }
        
        logging.info(f"成功提取 {len(bin_spectra)}/{self.n_bins} 个分箱的光谱")
        
        return self.bin_data
    
    def fit_bins(self, sps, gas_templates, gas_names, line_wave):
        """
        对每个分箱的合并光谱进行拟合。
        
        Parameters
        ----------
        sps : object
            恒星合成种群库
        gas_templates : ndarray
            气体发射线模板
        gas_names : array
            气体发射线名称
        line_wave : array
            发射线波长
            
        Returns
        -------
        dict
            拟合结果字典
        """
        logging.info(f"===== 开始拟合 {self.n_bins} 个分箱的光谱 (并行模式={self.config.parallel_mode}) =====")
        
        if self.bin_data is None:
            logging.error("没有可用的分箱数据")
            return {}
        
        # 准备拟合参数
        bin_ids = list(self.bin_data['spectra'].keys())
        
        # 使用多进程进行并行拟合
        start_time = time.time()
        results = {}
        
        # 根据并行模式选择处理方式
        if self.config.parallel_mode == 'grouped':
            # 内存优化：分批处理分箱
            batch_size = self.config.batch_size
            
            for batch_start in range(0, len(bin_ids), batch_size):
                batch_end = min(batch_start + batch_size, len(bin_ids))
                batch_bins = bin_ids[batch_start:batch_end]
                
                logging.info(f"处理批次 {batch_start//batch_size + 1}/{(len(bin_ids)-1)//batch_size + 1} "
                            f"(分箱 {batch_start+1}-{batch_end})")
                
                with ProcessPoolExecutor(max_workers=self.config.n_threads) as executor:
                    # 提交批次任务
                    future_to_bin = {}
                    for bin_id in batch_bins:
                        # 准备参数
                        spectrum = self.bin_data['spectra'][bin_id]
                        position = self.bin_data['positions'][bin_id]
                        
                        # 创建模拟单像素输入
                        args = (bin_id, -1, self.galaxy_data, sps, gas_templates, gas_names, line_wave, self.config)
                        args[2].spectra = np.column_stack([spectrum])  # 替换为分箱光谱
                        
                        # 提交任务
                        future = executor.submit(fit_bin, args)
                        future_to_bin[future] = bin_id
                    
                    # 处理结果
                    with tqdm(total=len(batch_bins), desc=f"批次 {batch_start//batch_size + 1}") as pbar:
                        for future in as_completed(future_to_bin):
                            bin_id, result = future.result()
                            if result is not None:
                                results[bin_id] = result
                            pbar.update(1)
                
                # 强制垃圾回收
                import gc
                gc.collect()
        
        else:  # global模式
            logging.info(f"使用全局并行模式处理所有 {len(bin_ids)} 个分箱")
            
            with ProcessPoolExecutor(max_workers=self.config.n_threads) as executor:
                # 提交所有任务
                future_to_bin = {}
                for bin_id in bin_ids:
                    # 准备参数
                    spectrum = self.bin_data['spectra'][bin_id]
                    position = self.bin_data['positions'][bin_id]
                    
                    # 创建模拟单像素输入
                    args = (bin_id, -1, self.galaxy_data, sps, gas_templates, gas_names, line_wave, self.config)
                    args[2].spectra = np.column_stack([spectrum])  # 替换为分箱光谱
                    
                    # 提交任务
                    future = executor.submit(fit_bin, args)
                    future_to_bin[future] = bin_id
                
                # 处理结果
                with tqdm(total=len(bin_ids), desc="处理分箱") as pbar:
                    for future in as_completed(future_to_bin):
                        bin_id, result = future.result()
                        if result is not None:
                            results[bin_id] = result
                        pbar.update(1)
        
        # 计算完成时间
        end_time = time.time()
        successful = len(results)
        logging.info(f"完成 {successful}/{self.n_bins} 个分箱的拟合，用时 {end_time - start_time:.1f} 秒")
        
        # 保存结果
        self.bin_results = results
        
        return results
    
    def process_results(self):
        """
        处理拟合结果并填充映射。
        
        Returns
        -------
        dict
            处理后的结果字典
        """
        logging.info(f"===== 处理 {len(self.bin_results)} 个分箱的拟合结果 =====")
        
        if not self.bin_results:
            logging.error("没有可用的拟合结果")
            return {}
        
        # 获取数据维度
        ny, nx = self.galaxy_data.cube.shape[1:3]
        
        # 初始化结果数组
        velfield = np.full((ny, nx), np.nan)
        sigfield = np.full((ny, nx), np.nan)
        
        # 初始化发射线和指数映射
        el_flux_maps = {}
        el_snr_maps = {}
        index_maps = {}
        
        for name in self.config.gas_names:
            el_flux_maps[name] = np.full((ny, nx), np.nan)
            el_snr_maps[name] = np.full((ny, nx), np.nan)
            
        for name in self.config.line_indices:
            index_maps[name] = np.full((ny, nx), np.nan)
        
        # 处理每个分箱的结果
        for bin_id, result in self.bin_results.items():
            if not result.get('success', False):
                continue
                
            # 提取结果数据
            velocity = result['velocity']
            sigma = result['sigma']
            
            # 提取发射线数据
            el_results = result.get('el_results', {})
            
            # 提取指数数据
            indices = result.get('indices', {})
            
            # 找到属于这个分箱的所有像素
            bin_mask = (self.bin_map == bin_id)
            
            # 填充速度和弥散度映射
            velfield[bin_mask] = velocity
            sigfield[bin_mask] = sigma
            
            # 填充发射线映射
            for name, data in el_results.items():
                if name in el_flux_maps:
                    el_flux_maps[name][bin_mask] = data['flux']
                    el_snr_maps[name][bin_mask] = data['an']
            
            # 填充指数映射
            for name, value in indices.items():
                if name in index_maps:
                    index_maps[name][bin_mask] = value
        
        # 保存处理后的映射
        self.velfield = velfield
        self.sigfield = sigfield
        self.el_flux_maps = el_flux_maps
        self.el_snr_maps = el_snr_maps
        self.index_maps = index_maps
        
        # 同时更新galaxy_data中的映射
        self.galaxy_data.velfield = velfield.copy()
        self.galaxy_data.sigfield = sigfield.copy()
        
        for name in self.config.gas_names:
            self.galaxy_data.el_flux_maps[name] = el_flux_maps[name].copy()
            self.galaxy_data.el_snr_maps[name] = el_snr_maps[name].copy()
            
        for name in self.config.line_indices:
            self.galaxy_data.index_maps[name] = index_maps[name].copy()
        
        # 创建CSV摘要
        self.create_bin_summary()
        
        return {
            'velfield': velfield,
            'sigfield': sigfield,
            'el_flux_maps': el_flux_maps,
            'el_snr_maps': el_snr_maps,
            'index_maps': index_maps
        }
    
    def create_bin_summary(self):
        """
        创建分箱结果摘要。
        
        Returns
        -------
        DataFrame
            结果摘要
        """
        # 创建数据记录列表
        data = []
        
        for bin_id, result in self.bin_results.items():
            if not result.get('success', False):
                continue
                
            # 获取分箱位置
            position = self.bin_data['positions'][bin_id]
            n_pixels = position['n_pixels']
            
            # 创建基本记录
            record = {
                'bin_id': bin_id,
                'x': position['x'],
                'y': position['y'],
                'n_pixels': n_pixels,
                'velocity': result['velocity'],
                'sigma': result['sigma'],
                'snr': result['snr']
            }
            
            # 添加发射线数据
            for name, data_dict in result.get('el_results', {}).items():
                record[f'{name}_flux'] = data_dict['flux']
                record[f'{name}_snr'] = data_dict['an']
            
            # 添加指数数据
            for name, value in result.get('indices', {}).items():
                record[f'{name}_index'] = value
            
            data.append(record)
        
        # 创建DataFrame
        if data:
            import pandas as pd
            df = pd.DataFrame(data)
            
            # 保存CSV文件
            csv_path = self.config.output_dir / f"{self.config.galaxy_name}_VNB_bins.csv"
            df.to_csv(csv_path, index=False)
            logging.info(f"保存分箱摘要到 {csv_path}")
            
            return df
        else:
            logging.warning("没有可用的分箱结果来创建摘要")
            return None
    
    def plot_binning(self):
        """
        绘制Voronoi分箱结果。
        
        Returns
        -------
        None
        """
        if self.config.no_plots:
            return
            
        # 如果没有成功创建分箱，返回
        if not hasattr(self, 'n_bins') or self.n_bins == 0:
            logging.warning("没有可用的分箱来绘制")
            return
        
        try:
            with plt.rc_context({'figure.max_open_warning': False}):
                # 创建图形
                fig, ax = plt.subplots(figsize=(10, 8), dpi=self.config.dpi)
                
                # 使用vorbin的display_bins函数绘制分箱
                rnd_colors = display_bins(self.bin_map, ax=ax)
                
                # 标签和标题
                ax.set_xlabel('X [pixels]')
                ax.set_ylabel('Y [pixels]')
                ax.set_title(f"{self.config.galaxy_name} - Voronoi Binning: {self.n_bins} bins")
                
                # 紧凑布局
                plt.tight_layout()
                
                # 保存
                plot_path = self.config.plot_dir / f"{self.config.galaxy_name}_voronoi_bins.png"
                plt.savefig(plot_path, dpi=self.config.dpi)
                plt.close(fig)
                
                logging.info(f"Voronoi分箱图保存到 {plot_path}")
        except Exception as e:
            logging.error(f"绘制分箱图时出错: {str(e)}")
            plt.close('all')  # 确保关闭所有图形
    
    def plot_bin_results(self, bin_id):
        """
        绘制单个分箱的拟合结果。
        
        Parameters
        ----------
        bin_id : int
            分箱ID
            
        Returns
        -------
        None
        """
        if self.config.no_plots or self.config.plot_count >= self.config.max_plots:
            return
            
        if bin_id not in self.bin_results:
            logging.warning(f"分箱 {bin_id} 没有可用的拟合结果")
            return
            
        result = self.bin_results[bin_id]
        if not result.get('success', False):
            logging.warning(f"分箱 {bin_id} 的拟合不成功")
            return
            
        try:
            # 获取分箱位置信息
            position = self.bin_data['positions'][bin_id]
            i, j = int(position['y']), int(position['x'])
            
            # 获取pp对象
            pp = result.get('pp_obj')
            if pp is None:
                logging.warning(f"分箱 {bin_id} 没有可用的pp对象")
                return
                
            # 设置pp的附加属性，以便plot_bin_fit函数正常工作
            if hasattr(result, 'stage1_bestfit'):
                pp.stage1_bestfit = result['stage1_bestfit'] 
            else:
                pp.stage1_bestfit = pp.bestfit
                
            pp.optimal_stellar_template = result['optimal_template']
            pp.full_bestfit = result['bestfit']
            pp.full_gas_bestfit = result['gas_bestfit']
            
            # 调用绘图函数
            plot_bin_fit(bin_id, self.galaxy_data, pp, position, self.config)
            
            # 增加计数器
            self.config.plot_count += 1
        except Exception as e:
            logging.error(f"绘制分箱 {bin_id} 结果时出错: {str(e)}")
            plt.close('all')
    
    def create_summary_plots(self):
        """
        创建VNB结果的汇总图。
        
        Returns
        -------
        None
        """
        if self.config.no_plots:
            return
            
        try:
            # 创建plots目录
            os.makedirs(self.config.plot_dir, exist_ok=True)
            
            # 1. 分箱图
            self.plot_binning()
            
            # 2. 运动学图
            with plt.rc_context({'figure.max_open_warning': False}):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.config.dpi)
                
                # 速度图
                vmax = np.nanpercentile(np.abs(self.velfield), 90)
                im0 = axes[0].imshow(self.velfield, origin='lower', cmap='RdBu_r', 
                                  vmin=-vmax, vmax=vmax)
                axes[0].set_title('Velocity [km/s]')
                plt.colorbar(im0, ax=axes[0])
                
                # 速度弥散度图
                sigma_max = np.nanpercentile(self.sigfield, 95)
                im1 = axes[1].imshow(self.sigfield, origin='lower', cmap='viridis', 
                                  vmin=0, vmax=sigma_max)
                axes[1].set_title('Velocity Dispersion [km/s]')
                plt.colorbar(im1, ax=axes[1])
                
                plt.suptitle(f"{self.config.galaxy_name} - VNB Stellar Kinematics")
                plt.tight_layout()
                plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_VNB_kinematics.png", dpi=self.config.dpi)
                plt.close(fig)
            
            # 3. 发射线图
            if self.config.compute_emission_lines and len(self.config.gas_names) > 0:
                n_lines = len(self.config.gas_names)
                
                with plt.rc_context({'figure.max_open_warning': False}):
                    fig, axes = plt.subplots(2, n_lines, figsize=(4*n_lines, 8), dpi=self.config.dpi)
                    
                    if n_lines == 1:  # 处理单个发射线的情况
                        axes = np.array([[axes[0]], [axes[1]]])
                    
                    for i, name in enumerate(self.config.gas_names):
                        # 流量图
                        flux_map = self.el_flux_maps[name]
                        vmax = np.nanpercentile(flux_map, 95)
                        im = axes[0, i].imshow(flux_map, origin='lower', cmap='inferno', vmin=0, vmax=vmax)
                        axes[0, i].set_title(f"{name} Flux")
                        plt.colorbar(im, ax=axes[0, i])
                        
                        # 信噪比图
                        snr_map = self.el_snr_maps[name]
                        im = axes[1, i].imshow(snr_map, origin='lower', cmap='viridis', vmin=0, vmax=5)
                        axes[1, i].set_title(f"{name} S/N")
                        plt.colorbar(im, ax=axes[1, i])
                    
                    plt.suptitle(f"{self.config.galaxy_name} - VNB Emission Lines")
                    plt.tight_layout()
                    plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_VNB_emission_lines.png", dpi=self.config.dpi)
                    plt.close(fig)
            
            # 4. 谱指数图
            if self.config.compute_spectral_indices and len(self.config.line_indices) > 0:
                n_indices = len(self.config.line_indices)
                n_cols = min(3, n_indices)
                n_rows = (n_indices + n_cols - 1) // n_cols
                
                with plt.rc_context({'figure.max_open_warning': False}):
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), dpi=self.config.dpi)
                    axes = np.atleast_2d(axes)
                    
                    for i, name in enumerate(self.config.line_indices):
                        row = i // n_cols
                        col = i % n_cols
                        
                        index_map = self.index_maps[name]
                        vmin = np.nanpercentile(index_map, 5)
                        vmax = np.nanpercentile(index_map, 95)
                        
                        im = axes[row, col].imshow(index_map, origin='lower', cmap='viridis', 
                                                vmin=vmin, vmax=vmax)
                        axes[row, col].set_title(f"{name} Index")
                        plt.colorbar(im, ax=axes[row, col])
                    
                    # 隐藏空的子图
                    for i in range(n_indices, n_rows * n_cols):
                        row = i // n_cols
                        col = i % n_cols
                        axes[row, col].axis('off')
                    
                    plt.suptitle(f"{self.config.galaxy_name} - VNB Spectral Indices")
                    plt.tight_layout()
                    plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_VNB_indices.png", dpi=self.config.dpi)
                    plt.close(fig)
            
            # 强制清理
            plt.close('all')
            import gc
            gc.collect()
        
        except Exception as e:
            logging.error(f"创建VNB汇总图时出错: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
            plt.close('all')
    
    def save_results_to_fits(self):
        """
        保存VNB结果到FITS文件。
        
        Returns
        -------
        None
        """
        try:
            # 创建头文件
            hdr = fits.Header()
            hdr['OBJECT'] = self.config.galaxy_name
            hdr['REDSHIFT'] = self.config.redshift
            hdr['CD1_1'] = self.galaxy_data.CD1_1
            hdr['CD1_2'] = self.galaxy_data.CD1_2
            hdr['CD2_1'] = self.galaxy_data.CD2_1
            hdr['CD2_2'] = self.galaxy_data.CD2_2
            hdr['CRVAL1'] = self.galaxy_data.CRVAL1
            hdr['CRVAL2'] = self.galaxy_data.CRVAL2
            
            # 添加VNB信息
            hdr['BINTYPE'] = 'VNB'
            hdr['NBINS'] = self.n_bins
            hdr['PARMODE'] = self.config.parallel_mode
            
            # 保存分箱映射
            hdu_binmap = fits.PrimaryHDU(self.bin_map, header=hdr)
            hdu_binmap.header['CONTENT'] = 'Voronoi bin map'
            hdu_binmap.writeto(self.config.output_dir / f"{self.config.galaxy_name}_VNB_binmap.fits", overwrite=True)
            
            # 保存速度场
            hdu_vel = fits.PrimaryHDU(self.velfield, header=hdr)
            hdu_vel.header['CONTENT'] = 'Stellar velocity field (VNB)'
            hdu_vel.header['BUNIT'] = 'km/s'
            hdu_vel.writeto(self.config.output_dir / f"{self.config.galaxy_name}_VNB_velfield.fits", overwrite=True)
            
            # 保存速度弥散度场
            hdu_sig = fits.PrimaryHDU(self.sigfield, header=hdr)
            hdu_sig.header['CONTENT'] = 'Stellar velocity dispersion (VNB)'
            hdu_sig.header['BUNIT'] = 'km/s'
            hdu_sig.writeto(self.config.output_dir / f"{self.config.galaxy_name}_VNB_sigfield.fits", overwrite=True)
            
            # 保存发射线图
            for name in self.config.gas_names:
                if name in self.el_flux_maps:
                    hdu = fits.PrimaryHDU(self.el_flux_maps[name], header=hdr)
                    hdu.header['CONTENT'] = f'{name} emission line flux (VNB)'
                    hdu.header['BUNIT'] = 'flux units'
                    hdu.writeto(self.config.output_dir / f"{self.config.galaxy_name}_VNB_{name}_flux.fits", overwrite=True)
                    
                    hdu = fits.PrimaryHDU(self.el_snr_maps[name], header=hdr)
                    hdu.header['CONTENT'] = f'{name} emission line S/N (VNB)'
                    hdu.header['BUNIT'] = 'ratio'
                    hdu.writeto(self.config.output_dir / f"{self.config.galaxy_name}_VNB_{name}_snr.fits", overwrite=True)
            
            # 保存谱指数图
            for name in self.config.line_indices:
                if name in self.index_maps:
                    hdu = fits.PrimaryHDU(self.index_maps[name], header=hdr)
                    hdu.header['CONTENT'] = f'{name} spectral index (VNB)'
                    hdu.header['BUNIT'] = 'Angstrom'
                    hdu.writeto(self.config.output_dir / f"{self.config.galaxy_name}_VNB_{name}_index.fits", overwrite=True)
            
            logging.info(f"VNB结果保存到FITS文件")
            
        except Exception as e:
            logging.error(f"保存VNB结果到FITS文件时出错: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())


def fit_bin(args):
    """
    拟合单个Voronoi分箱的合并光谱。
    
    Parameters
    ----------
    args : tuple
        (bin_id, _, galaxy_data, sps, gas_templates, gas_names, line_wave, config)
        
    Returns
    -------
    tuple
        (bin_id, results_dict or None)
    """
    bin_id, _, galaxy_data, sps, gas_templates, gas_names, line_wave, config = args
    
    logging.debug(f"===== FITTING BIN {bin_id} =====")
    
    try:
        # 获取分箱光谱
        spectrum = galaxy_data.spectra[:, 0]  # 在提交任务时已替换为分箱光谱
        
        wave_range = [(Apply_velocity_correction(config.good_wavelength_range[0], config.redshift)),
                      (Apply_velocity_correction(config.good_wavelength_range[1], config.redshift))]
        
        # 波长范围
        lam_gal = galaxy_data.lam_gal
        lam_range_temp = np.exp(sps.ln_lam_temp[[0, -1]])
        
        # 应用波长范围筛选
        spectrum = spectrum[np.where((lam_gal > wave_range[0]) & (lam_gal < wave_range[1]))]
        lam_gal = lam_gal[np.where((lam_gal > wave_range[0]) & (lam_gal < wave_range[1]))]
        
        # 使用统一噪声
        noise = np.ones_like(spectrum)
        
        # 自动计算掩码
        mask = util.determine_mask(np.log(lam_gal), lam_range_temp, width=1000)
        
        if not np.any(mask):
            logging.warning(f"分箱 {bin_id} 的掩码为空。波长范围可能不重叠。")
            return bin_id, None
        
        # 第一阶段：仅拟合恒星成分
        logging.debug(f"STEP: FIRST STAGE - Stellar component only fit")
        
        try:
            pp_stars = ppxf(sps.templates, spectrum, noise, galaxy_data.velscale, 
                           [config.vel_s, config.vel_dis_s],
                           degree=3,
                           plot=False, mask=mask, lam=lam_gal, 
                           lam_temp=sps.lam_temp, quiet=True)
            
            logging.debug(f"  - First stage fit successful: v={pp_stars.sol[0]:.1f}, σ={pp_stars.sol[1]:.1f}")
        except Exception as e:
            if config.retry_with_degree_zero:
                logging.warning(f"Initial stellar fit failed for bin {bin_id}: {str(e)}")
                logging.debug(f"  - Retrying with simplified parameters: degree=0")
                # Try with simpler polynomial
                pp_stars = ppxf(sps.templates, spectrum, noise, galaxy_data.velscale, 
                               [config.vel_s, config.vel_dis_s],
                               degree=0, 
                               plot=False, mask=mask, lam=lam_gal, 
                               lam_temp=sps.lam_temp, quiet=True)
                logging.debug(f"  - Retry successful: v={pp_stars.sol[0]:.1f}, σ={pp_stars.sol[1]:.1f}")
            else:
                raise  # Re-raise the exception if we're not retrying
        
        # 创建最优恒星模板
        if pp_stars.weights is None or not np.any(np.isfinite(pp_stars.weights)):
            logging.warning(f"Invalid weights in stellar fit for bin {bin_id}")
            return bin_id, None
        
        # 计算最优恒星模板
        optimal_stellar_template = sps.templates @ pp_stars.weights
        
        # 记录apoly
        apoly = pp_stars.apoly if hasattr(pp_stars, 'apoly') and pp_stars.apoly is not None else None
        
        # 保存第一阶段结果
        vel_stars = to_scalar(pp_stars.sol[0])
        sigma_stars = to_scalar(pp_stars.sol[1]) 
        bestfit_stars = pp_stars.bestfit
        
        # 确保sigma值合理
        if sigma_stars < 0:
            logging.warning(f"Negative velocity dispersion detected: {sigma_stars:.1f} km/s. Setting to 10 km/s.")
            sigma_stars = 10.0
        
        # 第二阶段：使用恒星模板和气体模板一起拟合
        if config.use_two_stage_fit and config.compute_emission_lines:
            
            logging.debug(f"STEP: SECOND STAGE - Combined fit with optimal stellar template")
            
            # 定义波长范围
            wave_range = [(Apply_velocity_correction(config.good_wavelength_range[0], config.redshift)),
                          (Apply_velocity_correction(config.good_wavelength_range[1], config.redshift))]
            
            # 只截取观测数据的波长范围
            wave_mask = (lam_gal >= wave_range[0]) & (lam_gal <= wave_range[1])
            galaxy_subset = spectrum[wave_mask]
            noise_subset = np.ones_like(galaxy_subset)
            
            # 确保恒星模板是正确的形状
            if optimal_stellar_template.ndim > 1 and optimal_stellar_template.shape[1] == 1:
                optimal_stellar_template = optimal_stellar_template.flatten()
            
            # 合并恒星和气体模板
            stars_gas_templates = np.column_stack([optimal_stellar_template, gas_templates])
            
            # 设置成分数组
            component = [0] + [1]*gas_templates.shape[1]
            gas_component = np.array(component) > 0
            
            # 设置moments参数
            moments = config.moments
            
            # 设置起始值
            start = [
                [vel_stars, sigma_stars],  # 恒星成分
                [vel_stars, 50]            # 气体成分
            ]
            
            # 设置边界
            vlim = lambda x: vel_stars + x*np.array([-100, 100])
            bounds = [
                [vlim(2), [20, 300]],  # 恒星成分
                [vlim(2), [20, 100]]   # 气体成分
            ]
            
            # 设置tied参数
            ncomp = len(moments)
            tied = [['', ''] for _ in range(ncomp)]
            
            try:
                # 执行第二阶段拟合
                pp = ppxf(stars_gas_templates, galaxy_subset, noise_subset, galaxy_data.velscale, start,
                         plot=False, moments=moments, degree=3, mdegree=-1, 
                         component=component, gas_component=gas_component, gas_names=gas_names,
                         lam=lam_gal[wave_mask], lam_temp=sps.lam_temp, 
                         tied=tied, bounds=bounds, quiet=True,
                         global_search=config.global_search)
                
                logging.debug(f"  - Combined fit successful: v={to_scalar(pp.sol[0]):.1f}, "
                             f"σ={to_scalar(pp.sol[1]):.1f}, χ²={to_scalar(pp.chi2):.3f}")
                
                # 检查是否成功拟合到发射线
                has_emission = False
                if hasattr(pp, 'gas_bestfit') and pp.gas_bestfit is not None:
                    has_emission = np.any(np.abs(pp.gas_bestfit) > 1e-10)
                
                # 创建完整的bestfit
                full_bestfit = np.copy(bestfit_stars)
                
                # 计算模板
                Apoly_Params = np.polyfit(lam_gal[wave_mask], pp.apoly, 3)
                Temp_Calu = (stars_gas_templates[:,0] * pp.weights[0]) + np.poly1d(Apoly_Params)(sps.lam_temp)
                
                # 添加完整的气体模板
                if hasattr(pp, 'gas_bestfit') and pp.gas_bestfit is not None:
                    # 在子集范围内替换为第二阶段的拟合结果
                    full_bestfit[wave_mask] = pp.bestfit
                    
                    # 创建完整范围的gas_bestfit
                    full_gas_bestfit = np.zeros_like(spectrum)
                    if has_emission:
                        full_gas_bestfit[wave_mask] = pp.gas_bestfit
                    
                    pp.full_gas_bestfit = full_gas_bestfit
                else:
                    pp.full_gas_bestfit = np.zeros_like(spectrum)
                
                pp.full_bestfit = full_bestfit
                
            except Exception as e:
                logging.warning(f"Combined fit failed for bin {bin_id}: {str(e)}")
                
                if config.fallback_to_simple_fit:
                    logging.debug(f"  - Using fallback to stellar-only fit")
                    # Fallback: just use stellar fit
                    pp = pp_stars
                    pp.full_bestfit = bestfit_stars
                    pp.full_gas_bestfit = np.zeros_like(spectrum)
                    
                    # No gas template results
                    pp.gas_bestfit = np.zeros_like(spectrum[wave_mask]) if wave_mask.any() else np.zeros_like(spectrum)
                    if not hasattr(pp, 'gas_flux'):
                        pp.gas_flux = np.zeros(len(gas_names))
                    pp.gas_bestfit_templates = np.zeros((pp.gas_bestfit.shape[0], len(gas_names)))
                    
                    logging.info(f"Used fallback stellar-only fit for bin {bin_id}")
                else:
                    raise  # Re-raise the exception if we're not using fallback
        else:
            # 不使用两阶段拟合，只使用第一阶段结果
            logging.debug(f"STEP: Using single-stage fit (two-stage disabled)")
            pp = pp_stars
            pp.full_bestfit = bestfit_stars
            pp.full_gas_bestfit = np.zeros_like(spectrum)
            
            # Add gas attributes manually 
            pp.gas_bestfit = np.zeros_like(spectrum)
            pp.gas_flux = np.zeros(len(gas_names)) if gas_names is not None else np.zeros(1)
            pp.gas_bestfit_templates = np.zeros((spectrum.shape[0], 
                                               len(gas_names) if gas_names is not None else 1))
        
        # 安全检查
        if pp is None or not hasattr(pp, 'full_bestfit') or pp.full_bestfit is None:
            logging.warning(f"Missing valid fit results for bin {bin_id}")
            return bin_id, None
            
        # 计算信噪比
        residuals = spectrum - pp.full_bestfit
        rms = robust_sigma(residuals[mask], zero=1)
        signal = np.median(spectrum[mask])
        snr = signal / rms if rms > 0 else 0
        
        # 提取发射线信息
        el_results = {}
        
        if config.compute_emission_lines:
            # 检查是否有气体发射线结果
            has_emission = False
            if hasattr(pp, 'full_gas_bestfit') and pp.full_gas_bestfit is not None:
                has_emission = np.any(np.abs(pp.full_gas_bestfit) > 1e-10)
                
            if has_emission:
                for name in config.gas_names:
                    # Find matching gas names
                    matches = [idx for idx, gname in enumerate(gas_names) if name in gname]
                    if matches:
                        idx = matches[0]
                        
                        # Extract the flux
                        dlam = line_wave[idx] * galaxy_data.velscale / config.c
                        
                        # Safety check for gas_flux
                        if hasattr(pp, 'gas_flux') and pp.gas_flux is not None and idx < len(pp.gas_flux):
                            flux = pp.gas_flux[idx] * dlam
                        else:
                            flux = 0.0
                        
                        # Calculate A/N
                        an = 0
                        if (hasattr(pp, 'gas_bestfit_templates') and 
                            pp.gas_bestfit_templates is not None and 
                            idx < pp.gas_bestfit_templates.shape[1]):
                            
                            peak = np.max(pp.gas_bestfit_templates[:, idx])
                            an = peak / rms if rms > 0 else 0
                        
                        el_results[name] = {'flux': flux, 'an': an}
            else:
                # 填充空结果
                for name in config.gas_names:
                    el_results[name] = {'flux': 0.0, 'an': 0.0}
        
        # 保存最优模板
        optimal_template = optimal_stellar_template
        
        # 计算光谱指数
        indices = {}
        
        if config.compute_spectral_indices:
            try:
                # 创建指数计算器
                calculator = LineIndexCalculator(
                    lam_gal, spectrum,
                    sps.lam_temp, Temp_Calu,
                    em_wave=lam_gal,
                    em_flux_list=pp.full_gas_bestfit if hasattr(pp, 'full_gas_bestfit') else None,
                    velocity_correction=to_scalar(pp.sol[0]),
                    continuum_mode=config.continuum_mode)
                
                # 只在需要时生成LIC图
                if config.LICplot and not config.no_plots and config.plot_count < config.max_plots:
                    calculator.plot_all_lines(mode='VNB', number=bin_id)
                    config.plot_count += 1
                
                # 计算请求的光谱指数
                for index_name in config.line_indices:
                    try:
                        indices[index_name] = calculator.calculate_index(index_name)
                    except Exception as e:
                        logging.warning(f"Failed to calculate index {index_name}: {str(e)}")
                        indices[index_name] = np.nan
            except Exception as e:
                logging.warning(f"Failed to initialize LineIndexCalculator: {str(e)}")
                for index_name in config.line_indices:
                    indices[index_name] = np.nan
        
        # 汇总结果
        sol_0 = 0.0
        sol_1 = 0.0
        if hasattr(pp, 'sol') and pp.sol is not None:
            if len(pp.sol) > 0:
                sol_0 = to_scalar(pp.sol[0])
            if len(pp.sol) > 1:
                sol_1 = to_scalar(pp.sol[1])
        
        # 确保所有需要的数组都存在
        bestfit = pp.full_bestfit if hasattr(pp, 'full_bestfit') else pp.bestfit
        gas_bestfit = pp.full_gas_bestfit if hasattr(pp, 'full_gas_bestfit') else np.zeros_like(spectrum)
        
        results = {
            'success': True,
            'velocity': sol_0,
            'sigma': sol_1,
            'bestfit': bestfit,
            'weights': pp.weights if hasattr(pp, 'weights') and pp.weights is not None else np.zeros(1),
            'gas_bestfit': gas_bestfit,
            'optimal_template': optimal_template,
            'apoly': apoly,
            'rms': rms,
            'snr': snr,
            'el_results': el_results,
            'indices': indices,
            'stage1_bestfit': bestfit_stars,
            'pp_obj': pp
        }
        
        logging.debug(f"===== BIN {bin_id} FIT COMPLETED SUCCESSFULLY =====")
        return bin_id, results
    
    except Exception as e:
        logging.error(f"Error fitting bin {bin_id}: {str(e)}")
        import traceback
        logging.debug(traceback.format_exc())
        return bin_id, None


def plot_bin_fit(bin_id, galaxy_data, pp, position, config):
    """
    创建分箱拟合的诊断图 - 内存优化版本
    
    Parameters
    ----------
    bin_id : int
        分箱ID
    galaxy_data : IFUDataCube
        包含星系数据的对象
    pp : ppxf object
        pPXF拟合结果
    position : dict
        包含分箱位置信息的字典
    config : P2PConfig
        配置对象
    """
    # 如果禁用了所有图形，直接返回
    if config.no_plots or config.plot_count >= config.max_plots:
        return
    
    try:
        # 创建绘图目录
        plot_dir = config.plot_dir / 'VNB_res'
        os.makedirs(plot_dir, exist_ok=True)
        
        # 准备文件名和路径
        plot_path_png = plot_dir / f"{config.galaxy_name}_bin_{bin_id}.png"
        
        # 获取数据
        lam_gal = galaxy_data.lam_gal
        
        # 在这里，我们假设pp的spectra已经被替换为分箱光谱
        # 因此直接使用光谱数据的第一列
        spectrum = galaxy_data.spectra[:, 0]
        
        # 获取拟合结果
        bestfit = pp.full_bestfit if hasattr(pp, 'full_bestfit') else pp.bestfit
        stage1_bestfit = pp.stage1_bestfit if hasattr(pp, 'stage1_bestfit') else bestfit
        gas_bestfit = pp.full_gas_bestfit if hasattr(pp, 'full_gas_bestfit') else np.zeros_like(spectrum)
        
        # 提取需要的属性值
        velocity = to_scalar(pp.sol[0]) if hasattr(pp, 'sol') and pp.sol is not None and len(pp.sol) > 0 else 0.0
        sigma = to_scalar(pp.sol[1]) if hasattr(pp, 'sol') and pp.sol is not None and len(pp.sol) > 1 else 0.0
        chi2 = to_scalar(pp.chi2) if hasattr(pp, 'chi2') and pp.chi2 is not None else 0.0
        
        # 使用with语句创建图形，确保资源正确释放
        with plt.rc_context({'figure.max_open_warning': False}):
            # 创建图形，指定较低的DPI以减少内存使用
            fig = plt.figure(figsize=(12, 8), dpi=config.dpi)
            
            # 创建子图
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax3 = plt.subplot(gs[2])
            
            # 第一个面板：原始数据和第一阶段拟合
            ax1.plot(lam_gal, spectrum, c='k', lw=1, alpha=.8, 
                    label=f"{config.galaxy_name} bin:{bin_id} - Original")
            ax1.plot(lam_gal, stage1_bestfit, '-', c='r', alpha=.8, 
                    label='Stage 1: Stellar fit only')
            
            # 第二个面板：最终拟合结果
            ax2.plot(lam_gal, spectrum, c='k', lw=1, alpha=.8, 
                    label='Original spectrum')
            ax2.plot(lam_gal, bestfit, '-', c='r', alpha=.8, 
                    label='Stage 2: Full fit')
            
            # 绘制恒星成分（总拟合减去气体）
            stellar_comp = bestfit - gas_bestfit
            ax2.plot(lam_gal, stellar_comp, '-', c='g', alpha=.7, lw=0.7, 
                    label='Stellar component')
            
            # 第三个面板：发射线和残差
            residuals = spectrum - bestfit
            
            # 绘制零线
            ax3.axhline(0, color='k', lw=0.7, alpha=.5)
            
            # 绘制残差
            ax3.plot(lam_gal, residuals, 'g-', lw=0.8, alpha=.7, 
                    label='Residuals (data - full fit)')
            
            # 绘制发射线
            if np.any(gas_bestfit != 0):
                ax3.plot(lam_gal, gas_bestfit, 'r-', lw=1.2, alpha=0.8,
                      label='Gas component')
            
            # 定义并绘制感兴趣的光谱区域
            spectral_regions = {
                'Hbeta': (4847.875, 4876.625),
                'Fe5015': (4977.750, 5054.000),
                'Mgb': (5160.125, 5192.625),
                '[OIII]': (4997, 5017)
            }
            
            # 在所有面板上标记光谱区域
            for name, (start, end) in spectral_regions.items():
                color = 'orange' if 'OIII' in name else 'lightgray'
                alpha = 0.3 if 'OIII' in name else 0.2
                for ax in [ax1, ax2, ax3]:
                    ax.axvspan(start, end, alpha=alpha, color=color)
                    # 在底部添加标签
                    if ax == ax3:
                        y_pos = ax3.get_ylim()[0] + 0.1 * (ax3.get_ylim()[1] - ax3.get_ylim()[0])
                        ax.text((start + end)/2, y_pos, name, ha='center', va='bottom',
                              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # 设置所有面板的属性
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(4800, 5250)
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(axis='both', which='both', labelsize='x-small', 
                            right=True, top=True, direction='in')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize='small')
            
            # 设置Y轴范围
            y_min = np.min(spectrum) * 0.9
            y_max = np.max(spectrum) * 1.1
            ax1.set_ylim(y_min, y_max)
            ax2.set_ylim(y_min, y_max)
            
            # 为第三个面板设置不同的Y轴范围
            if np.any(gas_bestfit != 0):
                gas_max = np.max(np.abs(gas_bestfit)) * 3
                res_max = max(np.max(np.abs(residuals)), gas_max)
            else:
                res_max = np.max(np.abs(residuals)) * 3
            
            ax3.set_ylim(-res_max, res_max)
            
            # 设置标签
            ax3.set_xlabel(r'Rest-frame Wavelength [$\AA$]', size=11)
            ax1.set_ylabel('Flux', size=11)
            ax2.set_ylabel('Flux', size=11)
            ax3.set_ylabel('Emission & Residuals', size=11)
            
            # 获取分箱位置信息
            x_pos = position.get('x', 0)
            y_pos = position.get('y', 0)
            n_pixels = position.get('n_pixels', 0)
            
            # 添加标题
            fig.suptitle(
                f"Bin {bin_id} - Two-stage Spectral Fit\n"
                f"Position: ({x_pos:.1f}, {y_pos:.1f}), {n_pixels} pixels\n"
                f"v={velocity:.1f} km/s, σ={sigma:.1f} km/s, χ²={chi2:.3f}", 
                fontsize=13
            )
            
            # 紧凑布局
            plt.tight_layout(rect=[0, 0, 1, 0.9])
            
            # 保存图像
            plt.savefig(plot_path_png, format='png', dpi=config.dpi, bbox_inches='tight')
            
            # 立即关闭图形并释放资源
            plt.close(fig)
            
            # 增加计数器
            config.plot_count += 1
        
    except Exception as e:
        logging.error(f"绘制分箱 {bin_id} 图像时出错: {str(e)}")
        # 确保任何失败的图形也会被关闭
        plt.close('all')


def run_vnb_analysis(config, target_snr=20):
    """
    运行Voronoi分箱分析的完整流程。
    
    Parameters
    ----------
    config : P2PConfig
        配置对象
    target_snr : float, optional
        目标信噪比，默认为20
        
    Returns
    -------
    tuple
        (galaxy_data, vnb)
    """
    logging.info(f"===== 开始Voronoi分箱分析 (SNR={target_snr}, 并行模式={config.parallel_mode}) =====")
    
    # 开始计时
    start_time = time.time()
    
    try:
        # 1. 加载数据
        logging.info("加载数据...")
        galaxy_data = IFUDataCube(config.get_data_path(), config.lam_range_temp, config.redshift, config)
        
        # 2. 准备模板
        logging.info("准备恒星和气体模板...")
        sps, gas_templates, gas_names, line_wave = prepare_templates(config, galaxy_data.velscale)
        
        # 3. 初始化Voronoi分箱
        vnb = VoronoiBinning(galaxy_data, config)
        
        # 4. 创建分箱
        n_bins = vnb.create_bins(target_snr=target_snr)
        if n_bins == 0:
            logging.error("无法创建Voronoi分箱")
            return galaxy_data, vnb
        
        # 5. 尝试加载P2P速度场用于光谱修正
        p2p_velfield = None
        p2p_path = config.output_dir / f"{config.galaxy_name}_velfield.fits"
        try:
            if os.path.exists(p2p_path):
                logging.info(f"加载P2P速度场: {p2p_path}")
                p2p_velfield = fits.getdata(p2p_path)
                logging.info(f"成功加载P2P速度场，形状: {p2p_velfield.shape}")
        except Exception as e:
            logging.warning(f"无法加载P2P速度场: {str(e)}")
            p2p_velfield = None
            
        # 提取分箱光谱
        bin_data = vnb.extract_bin_spectra(p2p_velfield)
        
        # 6. 拟合分箱
        bin_results = vnb.fit_bins(sps, gas_templates, gas_names, line_wave)
        
        # 7. 处理结果
        vnb.process_results()
        
        # 8. 创建汇总图
        if config.make_plots and not config.no_plots:
            logging.info("创建汇总图...")
            vnb.create_summary_plots()
            
            # 为前几个分箱创建诊断图
            logging.info("为样本分箱创建诊断图...")
            for bin_id in range(min(5, n_bins)):
                if bin_id in vnb.bin_results:
                    vnb.plot_bin_results(bin_id)
        
        # 9. 保存结果到FITS文件
        logging.info("保存结果到FITS文件...")
        vnb.save_results_to_fits()
        
        # 计算完成时间
        end_time = time.time()
        logging.info(f"VNB分析在 {end_time - start_time:.1f} 秒内完成")
        
        return galaxy_data, vnb
        
    except Exception as e:
        logging.error(f"VNB分析中出错: {str(e)}")
        logging.exception("堆栈跟踪:")
        raise


### ------------------------------------------------- ###
# Radial Binning Implementation
### ------------------------------------------------- ###

class RadialBinning:
    """
    基于径向分箱的光谱分析类。
    
    将像素按与星系中心的距离划分为多个径向环，
    然后对每个环进行单一光谱拟合。
    """
    
    def __init__(self, galaxy_data, config):
        """
        初始化径向分箱。
        
        Parameters
        ----------
        galaxy_data : IFUDataCube
            包含星系数据的对象
        config : P2PConfig
            配置对象
        """
        self.galaxy_data = galaxy_data
        self.config = config
        self.bin_data = None
        self.bin_results = {}
        
        # 存储分箱映射和结果的数组
        ny, nx = galaxy_data.cube.shape[1:3]
        self.bin_map = np.full((ny, nx), -1)  # -1表示未分箱的像素
        self.rmap = np.full((ny, nx), np.nan)  # 径向距离图
        
        # 创建结果映射
        self.velfield = np.full((ny, nx), np.nan)
        self.sigfield = np.full((ny, nx), np.nan)
        
        # 保存分箱图
        galaxy_data.bin_map = self.bin_map.copy()
        
        # 发射线和指数映射
        self.el_flux_maps = {}
        self.el_snr_maps = {}
        self.index_maps = {}
        
        for name in config.gas_names:
            self.el_flux_maps[name] = np.full((ny, nx), np.nan)
            self.el_snr_maps[name] = np.full((ny, nx), np.nan)
            
        for name in config.line_indices:
            self.index_maps[name] = np.full((ny, nx), np.nan)
        
        # 径向分箱特定参数
        self.n_bins = 0
        self.bin_edges = None
        self.center_x = None
        self.center_y = None
        self.pa = 0.0  # 位置角（度）
        self.ellipticity = 0.0  # 椭率
    
    def create_bins(self, n_bins=10, min_radius=None, max_radius=None, 
                   center_x=None, center_y=None, pa=0.0, ellipticity=0.0,
                   log_spacing=True, snr_min=3.0, target_snr=None,
                   adaptive_bins=False):
        """
        创建径向分箱。
        
        Parameters
        ----------
        n_bins : int
            分箱数量（如果adaptive_bins=True，这是最大分箱数）
        min_radius : float, optional
            最小半径（像素）
        max_radius : float, optional
            最大半径（像素）
        center_x : float, optional
            中心x坐标
        center_y : float, optional
            中心y坐标
        pa : float, optional
            位置角（度）
        ellipticity : float, optional
            椭率 (0-1)
        log_spacing : bool, optional
            是否使用对数间隔（仅当adaptive_bins=False时有效）
        snr_min : float, optional
            最小信噪比要求
        target_snr : float, optional
            目标信噪比（仅当adaptive_bins=True时有效）
        adaptive_bins : bool, optional
            是否使用自适应分箱来平衡SNR
            
        Returns
        -------
        int
            分箱的数量
        """
        if adaptive_bins:
            logging.info(f"===== 创建自适应径向分箱 (目标SNR={target_snr}) =====")
        else:
            logging.info(f"===== 创建均匀径向分箱 ({n_bins} 环) =====")
        
        # 获取数据维度
        ny, nx = self.galaxy_data.cube.shape[1:3]
        
        # 设置中心坐标（如果未提供）
        if center_x is None:
            center_x = nx / 2
        if center_y is None:
            center_y = ny / 2
            
        # 保存参数
        self.center_x = center_x
        self.center_y = center_y
        self.pa = pa
        self.ellipticity = ellipticity
        
        logging.info(f"星系中心: ({center_x:.1f}, {center_y:.1f}), PA: {pa:.1f}°, e: {ellipticity:.2f}")
        
        # 计算径向距离图
        y_coords, x_coords = np.indices((ny, nx))
        x_diff = x_coords - center_x
        y_diff = y_coords - center_y
        
        # 应用位置角和椭率
        if ellipticity > 0 or pa != 0:
            # 转换位置角为弧度
            pa_rad = np.radians(pa)
            
            # 旋转坐标系，使主轴与PA对齐
            x_rot = x_diff * np.cos(pa_rad) + y_diff * np.sin(pa_rad)
            y_rot = -x_diff * np.sin(pa_rad) + y_diff * np.cos(pa_rad)
            
            # 应用椭率
            b_to_a = 1 - ellipticity  # 短轴与长轴比例
            r_ell = np.sqrt((x_rot)**2 + (y_rot/b_to_a)**2)
        else:
            # 计算普通欧几里得距离
            r_ell = np.sqrt(x_diff**2 + y_diff**2)
        
        # 保存径向距离图
        self.rmap = r_ell
        
        # 确定径向范围
        if min_radius is None:
            min_radius = 0.0
        if max_radius is None:
            max_radius = np.nanmax(r_ell)
        
        # 计算信噪比图
        if hasattr(self.galaxy_data, 'signal') and hasattr(self.galaxy_data, 'noise'):
            snr_map = np.zeros((ny, nx))
            for i in range(ny):
                for j in range(nx):
                    k_index = i * nx + j
                    if k_index < len(self.galaxy_data.signal):
                        signal = self.galaxy_data.signal[k_index]
                        noise = self.galaxy_data.noise[k_index]
                        snr_map[i, j] = signal / noise if noise > 0 else 0
        else:
            # 如果没有SNR数据，假设所有像素都符合要求
            snr_map = np.ones((ny, nx)) * snr_min * 2
        
        # 创建有效像素掩码（应用SNR和半径限制）
        valid_mask = (r_ell >= min_radius) & (r_ell <= max_radius) & (snr_map >= snr_min)
        
        if not np.any(valid_mask):
            logging.error("没有符合条件的像素用于径向分箱")
            return 0
        
        # 根据不同模式创建分箱
        if adaptive_bins and target_snr is not None:
            # 自适应模式：根据SNR自动调整环的边界
            return self._create_adaptive_bins(r_ell, valid_mask, snr_map, target_snr, 
                                             min_radius, max_radius, n_bins)
        else:
            # 等距或对数间隔模式
            if log_spacing:
                # 对数间隔
                self.bin_edges = np.logspace(np.log10(max(min_radius, 0.5)), np.log10(max_radius), n_bins+1)
            else:
                # 线性间隔
                self.bin_edges = np.linspace(min_radius, max_radius, n_bins+1)
                
            return self._create_uniform_bins(r_ell, valid_mask, snr_map, n_bins)

    def _create_uniform_bins(self, r_ell, valid_mask, snr_map, n_bins):
        """
        使用均匀间隔创建径向环。
        
        Parameters
        ----------
        r_ell : ndarray
            径向距离图
        valid_mask : ndarray
            有效像素掩码
        snr_map : ndarray
            信噪比图
        n_bins : int
            分箱数量
            
        Returns
        -------
        int
            分箱的数量
        """
        logging.info(f"径向范围: {self.bin_edges[0]:.1f} - {self.bin_edges[-1]:.1f} 像素")
        logging.info(f"分箱边界: {self.bin_edges}")
        
        # 创建分箱映射
        ny, nx = r_ell.shape
        bin_map = np.full((ny, nx), -1)
        
        # 分配像素到分箱
        for bin_id in range(n_bins):
            # 获取当前环的内外半径
            r_in = self.bin_edges[bin_id]
            r_out = self.bin_edges[bin_id+1]
            
            # 找到落在这个环中的所有像素
            bin_mask = (r_ell >= r_in) & (r_ell < r_out) & valid_mask
            bin_map[bin_mask] = bin_id
            
            # 计算这个环的平均SNR
            bin_snr = np.mean(snr_map[bin_mask]) if np.any(bin_mask) else 0
            
            n_pixels = np.sum(bin_mask)
            logging.info(f"环 {bin_id}: 半径 {r_in:.1f}-{r_out:.1f} 像素, 包含 {n_pixels} 个像素, SNR~{bin_snr:.1f}")
            
            if n_pixels == 0:
                logging.warning(f"环 {bin_id} 没有符合条件的像素")
        
        # 保存分箱映射
        self.bin_map = bin_map
        self.galaxy_data.bin_map = bin_map.copy()
        self.n_bins = n_bins
        
        # 统计分箱像素数量
        n_binned = np.sum(bin_map >= 0)
        n_total = ny * nx
        logging.info(f"分箱完成: {n_binned}/{n_total} 个像素 ({n_binned/n_total*100:.1f}%) 被分配到 {n_bins} 个环")
        
        return n_bins

    def _create_adaptive_bins(self, r_ell, valid_mask, snr_map, target_snr, min_radius, max_radius, max_bins):
        """
        创建自适应径向环，使每个环的SNR大致相等。
        
        Parameters
        ----------
        r_ell : ndarray
            径向距离图
        valid_mask : ndarray
            有效像素掩码
        snr_map : ndarray
            信噪比图
        target_snr : float
            目标信噪比
        min_radius : float
            最小半径
        max_radius : float
            最大半径
        max_bins : int
            最大分箱数量
            
        Returns
        -------
        int
            分箱的数量
        """
        logging.info(f"自适应分箱: 目标SNR={target_snr}, 最大分箱数={max_bins}")
        
        # 获取维度
        ny, nx = r_ell.shape
        
        # 创建有效像素的排序索引（按径向距离）
        valid_indices = np.where(valid_mask)
        radii = r_ell[valid_indices]
        snrs = snr_map[valid_indices]
        
        # 创建包含坐标、半径和SNR的数组
        pixel_data = np.array([(y, x, r, s) for y, x, r, s in 
                               zip(valid_indices[0], valid_indices[1], radii, snrs)],
                              dtype=[('y', int), ('x', int), ('r', float), ('snr', float)])
        
        # 按半径排序
        pixel_data.sort(order='r')
        
        # 自适应创建分箱
        bin_map = np.full((ny, nx), -1)
        bin_edges = [min_radius]
        current_bin = 0
        start_idx = 0
        
        while start_idx < len(pixel_data) and current_bin < max_bins:
            # 开始累积SNR
            cum_snr = 0
            cum_pixels = 0
            target_cum_snr = target_snr**2  # 需要累积SNR^2
            
            # 添加像素直到达到目标SNR或用完像素
            end_idx = start_idx
            while end_idx < len(pixel_data) and cum_snr < target_cum_snr:
                y, x = pixel_data[end_idx]['y'], pixel_data[end_idx]['x']
                pixel_snr = pixel_data[end_idx]['snr']
                cum_snr += pixel_snr**2
                bin_map[y, x] = current_bin
                cum_pixels += 1
                end_idx += 1
                
                # 如果添加太多像素但仍未达到目标，则强制终止
                if cum_pixels > len(pixel_data) // max_bins * 2:
                    break
            
            # 计算这个bin的实际SNR
            actual_snr = np.sqrt(cum_snr) if cum_pixels > 0 else 0
            
            # 记录这个bin的外半径
            if end_idx < len(pixel_data):
                bin_edges.append(pixel_data[end_idx]['r'])
            else:
                bin_edges.append(max_radius)
            
            logging.info(f"环 {current_bin}: 半径 {bin_edges[current_bin]:.1f}-{bin_edges[current_bin+1]:.1f} 像素, "
                        f"包含 {cum_pixels} 个像素, SNR={actual_snr:.1f}")
            
            # 前进到下一个bin
            start_idx = end_idx
            current_bin += 1
            
            # 如果没有更多像素，退出循环
            if end_idx >= len(pixel_data):
                break
        
        # 保存bin信息
        self.bin_edges = np.array(bin_edges)
        self.bin_map = bin_map
        self.galaxy_data.bin_map = bin_map.copy()
        self.n_bins = current_bin
        
        # 统计分箱像素数量
        n_binned = np.sum(bin_map >= 0)
        n_total = ny * nx
        logging.info(f"自适应分箱完成: {n_binned}/{n_total} 个像素 ({n_binned/n_total*100:.1f}%) "
                   f"被分配到 {current_bin} 个环，目标SNR={target_snr}")
        
        return current_bin
    
    def extract_bin_spectra(self, p2p_velfield=None):
        """
        提取每个径向环的合并光谱。
        
        Parameters
        ----------
        p2p_velfield : ndarray, optional
            P2P分析得到的速度场，用于修正光谱。如果为None，则从galaxy_data中获取
            
        Returns
        -------
        dict
            包含合并光谱的字典
        """
        logging.info(f"===== 提取 {self.n_bins} 个径向环的合并光谱 =====")
        
        # 获取数据维度
        ny, nx = self.galaxy_data.cube.shape[1:3]
        npix = self.galaxy_data.spectra.shape[0]
        
        # 获取P2P速度场，优先使用传入参数，否则从galaxy_data获取
        if p2p_velfield is None:
            if hasattr(self.galaxy_data, 'velfield') and self.galaxy_data.velfield is not None:
                velfield = self.galaxy_data.velfield
                logging.info("使用galaxy_data中的速度场进行光谱修正")
            else:
                velfield = None
                logging.info("未找到速度场，不进行速度修正")
        else:
            velfield = p2p_velfield
            logging.info("使用传入的P2P速度场进行光谱修正")
        
        # 检查velfield是否有效
        if velfield is not None and not np.all(np.isnan(velfield)):
            apply_vel_correction = True
            logging.info("启用速度修正")
        else:
            apply_vel_correction = False
            logging.info("不进行速度修正 - 未找到有效的速度场")
        
        # 创建共同的波长网格（用于重采样）
        lam_gal = self.galaxy_data.lam_gal
        
        # 初始化合并光谱字典
        bin_spectra = {}
        bin_variances = {}
        bin_positions = {}
        
        # 为每个分箱创建合并光谱
        for bin_id in range(self.n_bins):
            # 找到属于这个分箱的所有像素
            bin_mask = (self.bin_map == bin_id)
            
            if not np.any(bin_mask):
                logging.warning(f"径向环 {bin_id} 没有包含像素")
                continue
                
            # 获取这个分箱中所有像素的行列索引
            rows, cols = np.where(bin_mask)
            
            # 初始化累积光谱和权重
            coadded_spectrum = np.zeros(npix)
            coadded_variance = np.zeros(npix)
            total_weight = 0
            
            # 处理每个像素
            for r, c in zip(rows, cols):
                k_index = r * nx + c
                
                # 获取原始光谱
                pixel_spectrum = self.galaxy_data.spectra[:, k_index]
                
                # 如果可以获取方差数据，则使用它，否则创建统一方差
                if hasattr(self.galaxy_data, 'variance'):
                    pixel_variance = self.galaxy_data.variance[:, k_index]
                else:
                    pixel_variance = np.ones_like(pixel_spectrum)
                
                # 计算当前像素的权重（使用信噪比）
                if hasattr(self.galaxy_data, 'signal') and hasattr(self.galaxy_data, 'noise'):
                    if k_index < len(self.galaxy_data.signal):
                        signal = self.galaxy_data.signal[k_index]
                        noise = self.galaxy_data.noise[k_index]
                        weight = (signal / noise)**2 if noise > 0 else 0
                    else:
                        weight = 1.0
                else:
                    weight = 1.0
                
                # 应用速度修正（如果可用）
                if apply_vel_correction and not np.isnan(velfield[r, c]):
                    vel = velfield[r, c]
                    
                    # 修正后的波长
                    lam_shifted = lam_gal * (1 + vel/self.config.c)
                    
                    # 重采样到原始波长网格
                    corrected_spectrum = np.interp(lam_gal, lam_shifted, pixel_spectrum,
                                                 left=0, right=0)
                    corrected_variance = np.interp(lam_gal, lam_shifted, pixel_variance,
                                                 left=np.inf, right=np.inf)
                    
                    # 累积修正后的光谱（加权）
                    coadded_spectrum += corrected_spectrum * weight
                    coadded_variance += corrected_variance * weight**2
                else:
                    # 不修正，直接累积
                    coadded_spectrum += pixel_spectrum * weight
                    coadded_variance += pixel_variance * weight**2
                
                total_weight += weight
            
            # 归一化累积光谱
            if total_weight > 0:
                merged_spectrum = coadded_spectrum / total_weight
                merged_variance = coadded_variance / (total_weight**2)
            else:
                logging.warning(f"分箱 {bin_id} 的总权重为零，使用简单平均")
                merged_spectrum = coadded_spectrum / len(rows) if len(rows) > 0 else coadded_spectrum
                merged_variance = coadded_variance / (len(rows)**2) if len(rows) > 0 else coadded_variance
            
            # 存储合并的数据
            bin_spectra[bin_id] = merged_spectrum
            bin_variances[bin_id] = merged_variance
            
            # 计算环的平均半径
            r_in = self.bin_edges[bin_id]
            r_out = self.bin_edges[bin_id+1]
            avg_radius = (r_in + r_out) / 2
            
            # 保存分箱的位置信息
            bin_positions[bin_id] = {
                'radius': avg_radius,
                'r_in': r_in,
                'r_out': r_out,
                'n_pixels': len(rows)
            }
            
            # 添加SNR信息
            snr = np.median(merged_spectrum / np.sqrt(merged_variance))
            bin_positions[bin_id]['snr'] = snr
            
            # 记录信息
            if bin_id % 5 == 0 or bin_id == self.n_bins - 1:
                logging.info(f"已提取 {bin_id+1}/{self.n_bins} 个径向环的光谱，"
                           f"半径={avg_radius:.1f}，SNR={snr:.1f}")
        
        # 保存提取的数据
        self.bin_data = {
            'spectra': bin_spectra,
            'variances': bin_variances,
            'positions': bin_positions
        }
        
        logging.info(f"成功提取 {len(bin_spectra)}/{self.n_bins} 个径向环的光谱")
        
        return self.bin_data
    
    def fit_bins(self, sps, gas_templates, gas_names, line_wave):
        """
        对每个径向环的合并光谱进行拟合。
        
        Parameters
        ----------
        sps : object
            恒星合成种群库
        gas_templates : ndarray
            气体发射线模板
        gas_names : array
            气体发射线名称
        line_wave : array
            发射线波长
            
        Returns
        -------
        dict
            拟合结果字典
        """
        logging.info(f"===== 开始拟合 {self.n_bins} 个径向环的光谱 (并行模式={self.config.parallel_mode}) =====")
        
        if self.bin_data is None:
            logging.error("没有可用的分箱数据")
            return {}
        
        # 准备拟合参数
        bin_ids = list(self.bin_data['spectra'].keys())
        
        # 使用多进程进行并行拟合
        start_time = time.time()
        results = {}
        
        # 径向环通常数量较少，使用更小的批次大小
        rdb_batch_size = min(5, self.config.batch_size)
        
        # 根据并行模式选择处理方式
        if self.config.parallel_mode == 'grouped':
            # 内存优化：分批处理分箱
            for batch_start in range(0, len(bin_ids), rdb_batch_size):
                batch_end = min(batch_start + rdb_batch_size, len(bin_ids))
                batch_bins = bin_ids[batch_start:batch_end]
                
                logging.info(f"处理批次 {batch_start//rdb_batch_size + 1}/{(len(bin_ids)-1)//rdb_batch_size + 1} "
                            f"(环 {batch_start+1}-{batch_end})")
                
                with ProcessPoolExecutor(max_workers=self.config.n_threads) as executor:
                    # 提交批次任务
                    future_to_bin = {}
                    for bin_id in batch_bins:
                        # 准备参数
                        spectrum = self.bin_data['spectra'][bin_id]
                        position = self.bin_data['positions'][bin_id]
                        
                        # 创建模拟单像素输入
                        args = (bin_id, -1, self.galaxy_data, sps, gas_templates, gas_names, line_wave, self.config)
                        args[2].spectra = np.column_stack([spectrum])  # 替换为分箱光谱
                        
                        # 提交任务
                        future = executor.submit(fit_radial_bin, args)
                        future_to_bin[future] = bin_id
                    
                    # 处理结果
                    with tqdm(total=len(batch_bins), desc=f"批次 {batch_start//rdb_batch_size + 1}") as pbar:
                        for future in as_completed(future_to_bin):
                            bin_id, result = future.result()
                            if result is not None:
                                results[bin_id] = result
                            pbar.update(1)
                
                # 强制垃圾回收
                import gc
                gc.collect()
        
        else:  # global模式
            logging.info(f"使用全局并行模式处理所有 {len(bin_ids)} 个径向环")
            
            with ProcessPoolExecutor(max_workers=self.config.n_threads) as executor:
                # 提交所有任务
                future_to_bin = {}
                for bin_id in bin_ids:
                    # 准备参数
                    spectrum = self.bin_data['spectra'][bin_id]
                    position = self.bin_data['positions'][bin_id]
                    
                    # 创建模拟单像素输入
                    args = (bin_id, -1, self.galaxy_data, sps, gas_templates, gas_names, line_wave, self.config)
                    args[2].spectra = np.column_stack([spectrum])  # 替换为分箱光谱
                    
                    # 提交任务
                    future = executor.submit(fit_radial_bin, args)
                    future_to_bin[future] = bin_id
                
                # 处理结果
                with tqdm(total=len(bin_ids), desc="处理径向环") as pbar:
                    for future in as_completed(future_to_bin):
                        bin_id, result = future.result()
                        if result is not None:
                            results[bin_id] = result
                        pbar.update(1)
        
        # 计算完成时间
        end_time = time.time()
        successful = len(results)
        logging.info(f"完成 {successful}/{self.n_bins} 个径向环的拟合，用时 {end_time - start_time:.1f} 秒")
        
        # 保存结果
        self.bin_results = results
        
        return results
    
    def process_results(self):
        """
        处理拟合结果并填充映射。
        
        Returns
        -------
        dict
            处理后的结果字典
        """
        logging.info(f"===== 处理 {len(self.bin_results)} 个径向环的拟合结果 =====")
        
        if not self.bin_results:
            logging.error("没有可用的拟合结果")
            return {}
        
        # 获取数据维度
        ny, nx = self.galaxy_data.cube.shape[1:3]
        
        # 初始化结果数组
        velfield = np.full((ny, nx), np.nan)
        sigfield = np.full((ny, nx), np.nan)
        
        # 初始化发射线和指数映射
        el_flux_maps = {}
        el_snr_maps = {}
        index_maps = {}
        
        for name in self.config.gas_names:
            el_flux_maps[name] = np.full((ny, nx), np.nan)
            el_snr_maps[name] = np.full((ny, nx), np.nan)
            
        for name in self.config.line_indices:
            index_maps[name] = np.full((ny, nx), np.nan)
        
        # 处理每个分箱的结果
        for bin_id, result in self.bin_results.items():
            if not result.get('success', False):
                continue
                
            # 提取结果数据
            velocity = result['velocity']
            sigma = result['sigma']
            
            # 提取发射线数据
            el_results = result.get('el_results', {})
            
            # 提取指数数据
            indices = result.get('indices', {})
            
            # 找到属于这个分箱的所有像素
            bin_mask = (self.bin_map == bin_id)
            
            # 填充速度和弥散度映射
            velfield[bin_mask] = velocity
            sigfield[bin_mask] = sigma
            
            # 填充发射线映射
            for name, data in el_results.items():
                if name in el_flux_maps:
                    el_flux_maps[name][bin_mask] = data['flux']
                    el_snr_maps[name][bin_mask] = data['an']
            
            # 填充指数映射
            for name, value in indices.items():
                if name in index_maps:
                    index_maps[name][bin_mask] = value
        
        # 保存处理后的映射
        self.velfield = velfield
        self.sigfield = sigfield
        self.el_flux_maps = el_flux_maps
        self.el_snr_maps = el_snr_maps
        self.index_maps = index_maps
        
        # 同时更新galaxy_data中的映射
        self.galaxy_data.velfield = velfield.copy()
        self.galaxy_data.sigfield = sigfield.copy()
        
        for name in self.config.gas_names:
            self.galaxy_data.el_flux_maps[name] = el_flux_maps[name].copy()
            self.galaxy_data.el_snr_maps[name] = el_snr_maps[name].copy()
            
        for name in self.config.line_indices:
            self.galaxy_data.index_maps[name] = index_maps[name].copy()
        
        # 创建CSV摘要
        self.create_bin_summary()
        
        return {
            'velfield': velfield,
            'sigfield': sigfield,
            'el_flux_maps': el_flux_maps,
            'el_snr_maps': el_snr_maps,
            'index_maps': index_maps
        }
    
    def create_bin_summary(self):
        """
        创建分箱结果摘要。
        
        Returns
        -------
        DataFrame
            结果摘要
        """
        # 创建数据记录列表
        data = []
        
        for bin_id, result in self.bin_results.items():
            if not result.get('success', False):
                continue
                
            # 获取分箱位置
            position = self.bin_data['positions'][bin_id]
            n_pixels = position['n_pixels']
            radius = position['radius']
            r_in = position['r_in']
            r_out = position['r_out']
            
            # 创建基本记录
            record = {
                'bin_id': bin_id,
                'radius': radius,
                'r_in': r_in,
                'r_out': r_out,
                'n_pixels': n_pixels,
                'velocity': result['velocity'],
                'sigma': result['sigma'],
                'snr': result['snr']
            }
            
            # 添加发射线数据
            for name, data_dict in result.get('el_results', {}).items():
                record[f'{name}_flux'] = data_dict['flux']
                record[f'{name}_snr'] = data_dict['an']
            
            # 添加指数数据
            for name, value in result.get('indices', {}).items():
                record[f'{name}_index'] = value
            
            data.append(record)
        
        # 创建DataFrame
        if data:
            import pandas as pd
            df = pd.DataFrame(data)
            
            # 保存CSV文件
            csv_path = self.config.output_dir / f"{self.config.galaxy_name}_RDB_bins.csv"
            df.to_csv(csv_path, index=False)
            logging.info(f"保存径向环摘要到 {csv_path}")
            
            return df
        else:
            logging.warning("没有可用的径向环结果来创建摘要")
            return None
    
    def plot_binning(self):
        """
        绘制径向分箱结果。
        
        Returns
        -------
        None
        """
        if self.config.no_plots:
            return
            
        # 如果没有成功创建分箱，返回
        if not hasattr(self, 'n_bins') or self.n_bins == 0:
            logging.warning("没有可用的分箱来绘制")
            return
        
        try:
            with plt.rc_context({'figure.max_open_warning': False}):
                # 创建图形
                fig, ax = plt.subplots(figsize=(10, 8), dpi=self.config.dpi)
                
                # 使用不同的颜色显示不同的径向环
                cmap = plt.cm.get_cmap('viridis', self.n_bins)
                im = ax.imshow(self.bin_map, origin='lower', cmap=cmap, 
                              vmin=-0.5, vmax=self.n_bins-0.5)
                
                # 添加颜色条
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Radial Bin')
                
                # 绘制圆形标记边界
                if self.ellipticity < 0.05:  # 近似圆形
                    # 绘制圆形边界
                    for r in self.bin_edges:
                        circle = plt.Circle((self.center_x, self.center_y), r, 
                                          fill=False, edgecolor='red', linestyle='--', alpha=0.7)
                        ax.add_patch(circle)
                else:
                    # 绘制椭圆边界
                    for r in self.bin_edges:
                        # 椭圆的长轴和短轴
                        a = r
                        b = r * (1 - self.ellipticity)
                        
                        # 转换位置角为弧度
                        pa_rad = np.radians(self.pa)
                        
                        ellipse = plt.matplotlib.patches.Ellipse(
                            (self.center_x, self.center_y), 2*a, 2*b,
                            angle=self.pa, fill=False, edgecolor='red', linestyle='--', alpha=0.7
                        )
                        ax.add_patch(ellipse)
                
                # 标记中心
                ax.plot(self.center_x, self.center_y, 'r+', markersize=10)
                
                # 标签和标题
                ax.set_xlabel('X [pixels]')
                ax.set_ylabel('Y [pixels]')
                ax.set_title(f"{self.config.galaxy_name} - Radial Binning: {self.n_bins} rings")
                
                # 紧凑布局
                plt.tight_layout()
                
                # 保存
                plot_path = self.config.plot_dir / f"{self.config.galaxy_name}_radial_bins.png"
                plt.savefig(plot_path, dpi=self.config.dpi)
                plt.close(fig)
                
                logging.info(f"径向分箱图保存到 {plot_path}")
        except Exception as e:
            logging.error(f"绘制分箱图时出错: {str(e)}")
            plt.close('all')  # 确保关闭所有图形
    
    def plot_bin_results(self, bin_id):
        """
        绘制单个径向环的拟合结果。
        
        Parameters
        ----------
        bin_id : int
            分箱ID
            
        Returns
        -------
        None
        """
        if self.config.no_plots or self.config.plot_count >= self.config.max_plots:
            return
            
        if bin_id not in self.bin_results:
            logging.warning(f"径向环 {bin_id} 没有可用的拟合结果")
            return
            
        result = self.bin_results[bin_id]
        if not result.get('success', False):
            logging.warning(f"径向环 {bin_id} 的拟合不成功")
            return
            
        try:
            # 获取分箱位置信息
            position = self.bin_data['positions'][bin_id]
            
            # 获取pp对象
            pp = result.get('pp_obj')
            if pp is None:
                logging.warning(f"径向环 {bin_id} 没有可用的pp对象")
                return
                
            # 设置pp的附加属性，以便plot_bin_fit函数正常工作
            if hasattr(result, 'stage1_bestfit'):
                pp.stage1_bestfit = result['stage1_bestfit'] 
            else:
                pp.stage1_bestfit = pp.bestfit
                
            pp.optimal_stellar_template = result['optimal_template']
            pp.full_bestfit = result['bestfit']
            pp.full_gas_bestfit = result['gas_bestfit']
            
            # 调用绘图函数 - 使用与VNB相同的函数，但改为RDB模式
            plot_radial_bin_fit(bin_id, self.galaxy_data, pp, position, self.config)
            
            # 增加计数器
            self.config.plot_count += 1
        except Exception as e:
            logging.error(f"绘制径向环 {bin_id} 结果时出错: {str(e)}")
            plt.close('all')
    
    def plot_radial_profiles(self):
        """
        绘制径向剖面图。
        
        Returns
        -------
        None
        """
        if self.config.no_plots:
            return
            
        # 如果没有成功创建分箱，返回
        if not self.bin_results:
            logging.warning("没有可用的拟合结果来绘制径向剖面")
            return
        
        try:
            # 准备数据
            radii = []
            velocities = []
            velocity_errs = []
            sigmas = []
            sigma_errs = []
            
            # 指数数据
            index_data = {name: [] for name in self.config.line_indices}
            
            # 发射线数据
            flux_data = {name: [] for name in self.config.gas_names}
            
            # 提取数据
            for bin_id, result in sorted(self.bin_results.items()):
                if not result.get('success', False):
                    continue
                    
                # 获取分箱半径
                radius = self.bin_data['positions'][bin_id]['radius']
                radii.append(radius)
                
                # 速度和弥散度
                velocities.append(result['velocity'])
                velocity_errs.append(10.0)  # 假设误差
                sigmas.append(result['sigma'])
                sigma_errs.append(10.0)  # 假设误差
                
                # 指数
                for name in self.config.line_indices:
                    if name in result.get('indices', {}):
                        index_data[name].append(result['indices'][name])
                    else:
                        index_data[name].append(np.nan)
                
                # 发射线流量
                for name in self.config.gas_names:
                    if name in result.get('el_results', {}):
                        flux_data[name].append(result['el_results'][name]['flux'])
                    else:
                        flux_data[name].append(np.nan)
            
            with plt.rc_context({'figure.max_open_warning': False}):
                # 创建速度和弥散度图
                fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=self.config.dpi, sharex=True)
                
                # 绘制速度剖面
                axes[0].errorbar(radii, velocities, yerr=velocity_errs, fmt='o-', capsize=3)
                axes[0].set_ylabel('Velocity [km/s]')
                axes[0].set_title('Radial Velocity Profile')
                axes[0].grid(True, alpha=0.3)
                
                # 绘制弥散度剖面
                axes[1].errorbar(radii, sigmas, yerr=sigma_errs, fmt='o-', capsize=3)
                axes[1].set_ylabel('Velocity Dispersion [km/s]')
                axes[1].set_xlabel('Radius [pixels]')
                axes[1].set_title('Radial Velocity Dispersion Profile')
                axes[1].grid(True, alpha=0.3)
                
                plt.suptitle(f"{self.config.galaxy_name} - Kinematic Radial Profiles")
                plt.tight_layout()
                plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_RDB_kinematic_profiles.png", dpi=self.config.dpi)
                plt.close(fig)
                
                # 创建指数剖面图
                if self.config.compute_spectral_indices and len(self.config.line_indices) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6), dpi=self.config.dpi)
                    
                    # 绘制每个指数的径向剖面
                    for name in self.config.line_indices:
                        ax.plot(radii, index_data[name], 'o-', label=name)
                    
                    ax.set_xlabel('Radius [pixels]')
                    ax.set_ylabel('Index Value [Å]')
                    ax.set_title(f"{self.config.galaxy_name} - Spectral Indices Radial Profiles")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_RDB_indices_profiles.png", dpi=self.config.dpi)
                    plt.close(fig)
                
                # 创建发射线剖面图
                if self.config.compute_emission_lines and len(self.config.gas_names) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6), dpi=self.config.dpi)
                    
                    # 绘制每个发射线的径向剖面
                    for name in self.config.gas_names:
                        ax.plot(radii, flux_data[name], 'o-', label=name)
                    
                    ax.set_xlabel('Radius [pixels]')
                    ax.set_ylabel('Flux')
                    ax.set_title(f"{self.config.galaxy_name} - Emission Line Flux Radial Profiles")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_RDB_emission_profiles.png", dpi=self.config.dpi)
                    plt.close(fig)
                
                logging.info(f"径向剖面图保存到 {self.config.plot_dir}")
        except Exception as e:
            logging.error(f"绘制径向剖面图时出错: {str(e)}")
            plt.close('all')
    
    def create_summary_plots(self):
        """
        创建RDB结果的汇总图。
        
        Returns
        -------
        None
        """
        if self.config.no_plots:
            return
            
        try:
            # 创建plots目录
            os.makedirs(self.config.plot_dir, exist_ok=True)
            
            # 1. 分箱图
            self.plot_binning()
            
            # 2. 径向剖面图
            self.plot_radial_profiles()
            
            # 3. 运动学图
            with plt.rc_context({'figure.max_open_warning': False}):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.config.dpi)
                
                # 速度图
                vmax = np.nanpercentile(np.abs(self.velfield), 90)
                im0 = axes[0].imshow(self.velfield, origin='lower', cmap='RdBu_r', 
                                  vmin=-vmax, vmax=vmax)
                axes[0].set_title('Velocity [km/s]')
                plt.colorbar(im0, ax=axes[0])
                
                # 速度弥散度图
                sigma_max = np.nanpercentile(self.sigfield, 95)
                im1 = axes[1].imshow(self.sigfield, origin='lower', cmap='viridis', 
                                  vmin=0, vmax=sigma_max)
                axes[1].set_title('Velocity Dispersion [km/s]')
                plt.colorbar(im1, ax=axes[1])
                
                plt.suptitle(f"{self.config.galaxy_name} - RDB Stellar Kinematics")
                plt.tight_layout()
                plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_RDB_kinematics.png", dpi=self.config.dpi)
                plt.close(fig)
            
            # 4. 发射线图
            if self.config.compute_emission_lines and len(self.config.gas_names) > 0:
                n_lines = len(self.config.gas_names)
                
                with plt.rc_context({'figure.max_open_warning': False}):
                    fig, axes = plt.subplots(2, n_lines, figsize=(4*n_lines, 8), dpi=self.config.dpi)
                    
                    if n_lines == 1:  # 处理单个发射线的情况
                        axes = np.array([[axes[0]], [axes[1]]])
                    
                    for i, name in enumerate(self.config.gas_names):
                        # 流量图
                        flux_map = self.el_flux_maps[name]
                        vmax = np.nanpercentile(flux_map, 95)
                        im = axes[0, i].imshow(flux_map, origin='lower', cmap='inferno', vmin=0, vmax=vmax)
                        axes[0, i].set_title(f"{name} Flux")
                        plt.colorbar(im, ax=axes[0, i])
                        
                        # 信噪比图
                        snr_map = self.el_snr_maps[name]
                        im = axes[1, i].imshow(snr_map, origin='lower', cmap='viridis', vmin=0, vmax=5)
                        axes[1, i].set_title(f"{name} S/N")
                        plt.colorbar(im, ax=axes[1, i])
                    
                    plt.suptitle(f"{self.config.galaxy_name} - RDB Emission Lines")
                    plt.tight_layout()
                    plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_RDB_emission_lines.png", dpi=self.config.dpi)
                    plt.close(fig)
            
            # 5. 谱指数图
            if self.config.compute_spectral_indices and len(self.config.line_indices) > 0:
                n_indices = len(self.config.line_indices)
                n_cols = min(3, n_indices)
                n_rows = (n_indices + n_cols - 1) // n_cols
                
                with plt.rc_context({'figure.max_open_warning': False}):
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), dpi=self.config.dpi)
                    axes = np.atleast_2d(axes)
                    
                    for i, name in enumerate(self.config.line_indices):
                        row = i // n_cols
                        col = i % n_cols
                        
                        index_map = self.index_maps[name]
                        vmin = np.nanpercentile(index_map, 5)
                        vmax = np.nanpercentile(index_map, 95)
                        
                        im = axes[row, col].imshow(index_map, origin='lower', cmap='viridis', 
                                                vmin=vmin, vmax=vmax)
                        axes[row, col].set_title(f"{name} Index")
                        plt.colorbar(im, ax=axes[row, col])
                    
                    # 隐藏空的子图
                    for i in range(n_indices, n_rows * n_cols):
                        row = i // n_cols
                        col = i % n_cols
                        axes[row, col].axis('off')
                    
                    plt.suptitle(f"{self.config.galaxy_name} - RDB Spectral Indices")
                    plt.tight_layout()
                    plt.savefig(self.config.plot_dir / f"{self.config.galaxy_name}_RDB_indices.png", dpi=self.config.dpi)
                    plt.close(fig)
            
            # 强制清理
            plt.close('all')
            import gc
            gc.collect()
        
        except Exception as e:
            logging.error(f"创建RDB汇总图时出错: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
            plt.close('all')
    
    def save_results_to_fits(self):
        """
        保存RDB结果到FITS文件。
        
        Returns
        -------
        None
        """
        try:
            # 创建头文件
            hdr = fits.Header()
            hdr['OBJECT'] = self.config.galaxy_name
            hdr['REDSHIFT'] = self.config.redshift
            hdr['CD1_1'] = self.galaxy_data.CD1_1
            hdr['CD1_2'] = self.galaxy_data.CD1_2
            hdr['CD2_1'] = self.galaxy_data.CD2_1
            hdr['CD2_2'] = self.galaxy_data.CD2_2
            hdr['CRVAL1'] = self.galaxy_data.CRVAL1
            hdr['CRVAL2'] = self.galaxy_data.CRVAL2
            
            # 添加RDB信息
            hdr['BINTYPE'] = 'RDB'
            hdr['NBINS'] = self.n_bins
            hdr['CENTERX'] = self.center_x
            hdr['CENTERY'] = self.center_y
            hdr['PA'] = self.pa
            hdr['ELLIP'] = self.ellipticity
            hdr['PARMODE'] = self.config.parallel_mode
            
            # 保存分箱映射和径向距离图
            hdu_binmap = fits.PrimaryHDU(self.bin_map, header=hdr)
            hdu_binmap.header['CONTENT'] = 'Radial bin map'
            hdu_binmap.writeto(self.config.output_dir / f"{self.config.galaxy_name}_RDB_binmap.fits", overwrite=True)
            
            hdu_rmap = fits.PrimaryHDU(self.rmap, header=hdr)
            hdu_rmap.header['CONTENT'] = 'Radial distance map'
            hdu_rmap.writeto(self.config.output_dir / f"{self.config.galaxy_name}_RDB_radiusmap.fits", overwrite=True)
            
            # 保存速度场
            hdu_vel = fits.PrimaryHDU(self.velfield, header=hdr)
            hdu_vel.header['CONTENT'] = 'Stellar velocity field (RDB)'
            hdu_vel.header['BUNIT'] = 'km/s'
            hdu_vel.writeto(self.config.output_dir / f"{self.config.galaxy_name}_RDB_velfield.fits", overwrite=True)
            
            # 保存速度弥散度场
            hdu_sig = fits.PrimaryHDU(self.sigfield, header=hdr)
            hdu_sig.header['CONTENT'] = 'Stellar velocity dispersion (RDB)'
            hdu_sig.header['BUNIT'] = 'km/s'
            hdu_sig.writeto(self.config.output_dir / f"{self.config.galaxy_name}_RDB_sigfield.fits", overwrite=True)
            
            # 保存发射线图
            for name in self.config.gas_names:
                if name in self.el_flux_maps:
                    hdu = fits.PrimaryHDU(self.el_flux_maps[name], header=hdr)
                    hdu.header['CONTENT'] = f'{name} emission line flux (RDB)'
                    hdu.header['BUNIT'] = 'flux units'
                    hdu.writeto(self.config.output_dir / f"{self.config.galaxy_name}_RDB_{name}_flux.fits", overwrite=True)
                    
                    hdu = fits.PrimaryHDU(self.el_snr_maps[name], header=hdr)
                    hdu.header['CONTENT'] = f'{name} emission line S/N (RDB)'
                    hdu.header['BUNIT'] = 'ratio'
                    hdu.writeto(self.config.output_dir / f"{self.config.galaxy_name}_RDB_{name}_snr.fits", overwrite=True)
            
            # 保存谱指数图
            for name in self.config.line_indices:
                if name in self.index_maps:
                    hdu = fits.PrimaryHDU(self.index_maps[name], header=hdr)
                    hdu.header['CONTENT'] = f'{name} spectral index (RDB)'
                    hdu.header['BUNIT'] = 'Angstrom'
                    hdu.writeto(self.config.output_dir / f"{self.config.galaxy_name}_RDB_{name}_index.fits", overwrite=True)
            
            logging.info(f"RDB结果保存到FITS文件")
            
        except Exception as e:
            logging.error(f"保存RDB结果到FITS文件时出错: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())


def fit_radial_bin(args):
    """
    拟合单个径向环的合并光谱。使用与fit_bin相同的逻辑，但针对径向环优化。
    
    Parameters
    ----------
    args : tuple
        (bin_id, _, galaxy_data, sps, gas_templates, gas_names, line_wave, config)
        
    Returns
    -------
    tuple
        (bin_id, results_dict or None)
    """
    # 直接调用fit_bin函数，但使用不同的日志标识
    bin_id, _, galaxy_data, sps, gas_templates, gas_names, line_wave, config = args
    
    logging.debug(f"===== FITTING RADIAL BIN {bin_id} =====")
    
    # 调用通用拟合函数
    result = fit_bin(args)
    
    if result[1] is not None:
        logging.debug(f"===== RADIAL BIN {bin_id} FIT COMPLETED SUCCESSFULLY =====")
    
    return result


def plot_radial_bin_fit(bin_id, galaxy_data, pp, position, config):
    """
    创建径向环拟合的诊断图 - 内存优化版本
    
    Parameters
    ----------
    bin_id : int
        分箱ID
    galaxy_data : IFUDataCube
        包含星系数据的对象
    pp : ppxf object
        pPXF拟合结果
    position : dict
        包含分箱位置信息的字典
    config : P2PConfig
        配置对象
    """
    # 如果禁用了所有图形，直接返回
    if config.no_plots or config.plot_count >= config.max_plots:
        return
    
    try:
        # 创建绘图目录
        plot_dir = config.plot_dir / 'RDB_res'
        os.makedirs(plot_dir, exist_ok=True)
        
        # 准备文件名和路径
        plot_path_png = plot_dir / f"{config.galaxy_name}_ring_{bin_id}.png"
        
        # 获取数据
        lam_gal = galaxy_data.lam_gal
        
        # 在这里，我们假设pp的spectra已经被替换为分箱光谱
        # 因此直接使用光谱数据的第一列
        spectrum = galaxy_data.spectra[:, 0]
        
        # 获取拟合结果
        bestfit = pp.full_bestfit if hasattr(pp, 'full_bestfit') else pp.bestfit
        stage1_bestfit = pp.stage1_bestfit if hasattr(pp, 'stage1_bestfit') else bestfit
        gas_bestfit = pp.full_gas_bestfit if hasattr(pp, 'full_gas_bestfit') else np.zeros_like(spectrum)
        
        # 提取需要的属性值
        velocity = to_scalar(pp.sol[0]) if hasattr(pp, 'sol') and pp.sol is not None and len(pp.sol) > 0 else 0.0
        sigma = to_scalar(pp.sol[1]) if hasattr(pp, 'sol') and pp.sol is not None and len(pp.sol) > 1 else 0.0
        chi2 = to_scalar(pp.chi2) if hasattr(pp, 'chi2') and pp.chi2 is not None else 0.0
        
        # 使用with语句创建图形，确保资源正确释放
        with plt.rc_context({'figure.max_open_warning': False}):
            # 创建图形，指定较低的DPI以减少内存使用
            fig = plt.figure(figsize=(12, 8), dpi=config.dpi)
            
            # 创建子图
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax3 = plt.subplot(gs[2])
            
            # 第一个面板：原始数据和第一阶段拟合
            ax1.plot(lam_gal, spectrum, c='k', lw=1, alpha=.8, 
                    label=f"{config.galaxy_name} ring:{bin_id} - Original")
            ax1.plot(lam_gal, stage1_bestfit, '-', c='r', alpha=.8, 
                    label='Stellar component fit')
            
            # 第二个面板：最终拟合结果
            ax2.plot(lam_gal, spectrum, c='k', lw=1, alpha=.8, 
                    label='Original spectrum')
            ax2.plot(lam_gal, bestfit, '-', c='r', alpha=.8, 
                    label='Full fit')
            
            # 绘制恒星成分（总拟合减去气体）
            stellar_comp = bestfit - gas_bestfit
            ax2.plot(lam_gal, stellar_comp, '-', c='g', alpha=.7, lw=0.7, 
                    label='Stellar component')
            
            # 第三个面板：发射线和残差
            residuals = spectrum - bestfit
            
            # 绘制零线
            ax3.axhline(0, color='k', lw=0.7, alpha=.5)
            
            # 绘制残差
            ax3.plot(lam_gal, residuals, 'g-', lw=0.8, alpha=.7, 
                    label='Residuals (data - full fit)')
            
            # 绘制发射线
            if np.any(gas_bestfit != 0):
                ax3.plot(lam_gal, gas_bestfit, 'r-', lw=1.2, alpha=0.8,
                      label='Gas component')
            
            # 定义并绘制感兴趣的光谱区域
            spectral_regions = {
                'Hbeta': (4847.875, 4876.625),
                'Fe5015': (4977.750, 5054.000),
                'Mgb': (5160.125, 5192.625),
                '[OIII]': (4997, 5017)
            }
            
            # 在所有面板上标记光谱区域
            for name, (start, end) in spectral_regions.items():
                color = 'orange' if 'OIII' in name else 'lightgray'
                alpha = 0.3 if 'OIII' in name else 0.2
                for ax in [ax1, ax2, ax3]:
                    ax.axvspan(start, end, alpha=alpha, color=color)
                    # 在底部添加标签
                    if ax == ax3:
                        y_pos = ax3.get_ylim()[0] + 0.1 * (ax3.get_ylim()[1] - ax3.get_ylim()[0])
                        ax.text((start + end)/2, y_pos, name, ha='center', va='bottom',
                              bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # 设置所有面板的属性
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim(4800, 5250)
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(axis='both', which='both', labelsize='x-small', 
                            right=True, top=True, direction='in')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize='small')
            
            # 设置Y轴范围
            y_min = np.min(spectrum) * 0.9
            y_max = np.max(spectrum) * 1.1
            ax1.set_ylim(y_min, y_max)
            ax2.set_ylim(y_min, y_max)
            
            # 为第三个面板设置不同的Y轴范围
            if np.any(gas_bestfit != 0):
                gas_max = np.max(np.abs(gas_bestfit)) * 3
                res_max = max(np.max(np.abs(residuals)), gas_max)
            else:
                res_max = np.max(np.abs(residuals)) * 3
            
            ax3.set_ylim(-res_max, res_max)
            
            # 设置标签
            ax3.set_xlabel(r'Rest-frame Wavelength [$\AA$]', size=11)
            ax1.set_ylabel('Flux', size=11)
            ax2.set_ylabel('Flux', size=11)
            ax3.set_ylabel('Emission & Residuals', size=11)
            
            # 获取径向环信息
            radius = position.get('radius', 0)
            r_in = position.get('r_in', 0)
            r_out = position.get('r_out', 0)
            n_pixels = position.get('n_pixels', 0)
            
            # 添加标题
            fig.suptitle(
                f"Radial Ring {bin_id} - Two-stage Spectral Fit\n"
                f"Radius: {radius:.1f} pixels ({r_in:.1f}-{r_out:.1f}), {n_pixels} pixels\n"
                f"v={velocity:.1f} km/s, σ={sigma:.1f} km/s, χ²={chi2:.3f}", 
                fontsize=13
            )
            
            # 紧凑布局
            plt.tight_layout(rect=[0, 0, 1, 0.9])
            
            # 保存图像
            plt.savefig(plot_path_png, format='png', dpi=config.dpi, bbox_inches='tight')
            
            # 立即关闭图形并释放资源
            plt.close(fig)
            
            # 增加计数器
            config.plot_count += 1
        
    except Exception as e:
        logging.error(f"绘制径向环 {bin_id} 图像时出错: {str(e)}")
        # 确保任何失败的图形也会被关闭
        plt.close('all')


def run_rdb_analysis(config, n_bins=10, center_x=None, center_y=None, 
                    pa=0.0, ellipticity=0.0, log_spacing=True,
                    adaptive_bins=False, target_snr=None):
    """
    运行径向分箱分析的完整流程。
    
    Parameters
    ----------
    config : P2PConfig
        配置对象
    n_bins : int, optional
        径向环数量，默认为10
    center_x : float, optional
        中心x坐标
    center_y : float, optional
        中心y坐标
    pa : float, optional
        位置角（度）
    ellipticity : float, optional
        椭率 (0-1)
    log_spacing : bool, optional
        是否使用对数间隔
    adaptive_bins : bool, optional
        是否使用自适应分箱以平衡SNR
    target_snr : float, optional
        目标信噪比（仅当adaptive_bins=True时有效）
        
    Returns
    -------
    tuple
        (galaxy_data, rdb)
    """
    if adaptive_bins:
        logging.info(f"===== 开始自适应径向分箱分析 (目标SNR={target_snr}, 并行模式={config.parallel_mode}) =====")
    else:
        logging.info(f"===== 开始均匀径向分箱分析 (环数={n_bins}, 并行模式={config.parallel_mode}) =====")
    
    # 开始计时
    start_time = time.time()
    
    try:
        # 1. 加载数据
        logging.info("加载数据...")
        galaxy_data = IFUDataCube(config.get_data_path(), config.lam_range_temp, config.redshift, config)
        
        # 2. 准备模板
        logging.info("准备恒星和气体模板...")
        sps, gas_templates, gas_names, line_wave = prepare_templates(config, galaxy_data.velscale)
        
        # 3. 初始化径向分箱
        rdb = RadialBinning(galaxy_data, config)
        
        # 4. 创建分箱
        ny, nx = galaxy_data.cube.shape[1:3]
        if center_x is None:
            center_x = nx / 2
        if center_y is None:
            center_y = ny / 2
        
        # 根据模式创建分箱
        if adaptive_bins:
            if target_snr is None:
                target_snr = 20.0  # 默认目标SNR
            n_bins = rdb.create_bins(n_bins=n_bins, center_x=center_x, center_y=center_y, 
                                    pa=pa, ellipticity=ellipticity, 
                                    adaptive_bins=True, target_snr=target_snr)
        else:    
            n_bins = rdb.create_bins(n_bins=n_bins, center_x=center_x, center_y=center_y, 
                                    pa=pa, ellipticity=ellipticity, log_spacing=log_spacing)
                                    
        if n_bins == 0:
            logging.error("无法创建径向分箱")
            return galaxy_data, rdb
        
        # 5. 提取分箱光谱
        # 尝试使用先前的P2P结果进行速度修正
        p2p_velfield = None
        p2p_path = config.output_dir / f"{config.galaxy_name}_velfield.fits"
        try:
            if os.path.exists(p2p_path):
                logging.info(f"加载P2P速度场: {p2p_path}")
                p2p_velfield = fits.getdata(p2p_path)
                logging.info(f"成功加载P2P速度场，形状: {p2p_velfield.shape}")
        except Exception as e:
            logging.warning(f"无法加载P2P速度场: {str(e)}")
            p2p_velfield = None
        
        # 提取光谱（带速度修正）
        bin_data = rdb.extract_bin_spectra(p2p_velfield)
        
        # 6. 拟合分箱
        bin_results = rdb.fit_bins(sps, gas_templates, gas_names, line_wave)
        
        # 7. 处理结果
        rdb.process_results()
        
        # 8. 创建汇总图
        if config.make_plots and not config.no_plots:
            logging.info("创建汇总图...")
            rdb.create_summary_plots()
            
            # 为前几个分箱创建诊断图
            logging.info("为样本径向环创建诊断图...")
            for bin_id in range(min(5, n_bins)):
                if bin_id in rdb.bin_results:
                    rdb.plot_bin_results(bin_id)
        
        # 9. 保存结果到FITS文件
        logging.info("保存结果到FITS文件...")
        rdb.save_results_to_fits()
        
        # 计算完成时间
        end_time = time.time()
        logging.info(f"RDB分析在 {end_time - start_time:.1f} 秒内完成")
        
        return galaxy_data, rdb
        
    except Exception as e:
        logging.error(f"RDB分析中出错: {str(e)}")
        logging.exception("堆栈跟踪:")
        raise


### ------------------------------------------------- ###
# Main Function
### ------------------------------------------------- ###

def main():
    """
    主函数 - 解析命令行参数并运行程序
    """
    # 创建解析器
    parser = argparse.ArgumentParser(description="P2P - 光谱拟合程序")
    
    # 基本参数
    parser.add_argument("--data-dir", type=str, default="data",
                       help="数据目录路径")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="输出目录路径")
    parser.add_argument("--galaxy-name", type=str, default=None,
                       help="星系名称")
    parser.add_argument("--data-file", type=str, default=None,
                       help="数据文件名")
    parser.add_argument("--mode", type=str, choices=['VNB', 'RDB', 'ALL'], default='ALL',
                       help="分析模式: VNB (Voronoi分箱), RDB (径向分箱), ALL (两种都做)")
    
    # 并行设置
    parser.add_argument("--threads", type=int, default=None,
                       help="使用的线程数 (默认: CPU核心数的一半)")
    parser.add_argument("--parallel-mode", type=str, default='grouped', choices=['grouped', 'global'],
                       help="并行处理模式: 'grouped'为分批处理, 'global'为一次提交所有任务 (默认: grouped)")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="分组处理模式下每批的分箱数量 (默认: 50)")
    
    # 图形设置
    parser.add_argument("--no-plots", action="store_true",
                       help="不创建图形")
    parser.add_argument("--max-plots", type=int, default=50,
                       help="每种类型的最大图形数量")
    parser.add_argument("--dpi", type=int, default=120,
                       help="图形DPI")
    
    # 拟合设置
    parser.add_argument("--no-emission-lines", dest="compute_emission_lines", action="store_false",
                       help="不拟合发射线")
    parser.add_argument("--no-spectral-indices", dest="compute_spectral_indices", action="store_false",
                       help="不计算谱线指数")
    parser.add_argument("--global-search", action="store_true",
                       help="在pPXF拟合中使用全局搜索")
    
    # 模板设置
    parser.add_argument("--template-dir", type=str, default="templates",
                       help="模板目录路径")
    parser.add_argument("--use-miles", action="store_true", default=True,
                       help="使用MILES模板库")
    parser.add_argument("--no-miles", dest="use_miles", action="store_false",
                       help="不使用MILES模板库")
    parser.add_argument("--template-file", type=str, default=None,
                       help="自定义模板文件名")
    
    # Voronoi分箱参数
    vnb_group = parser.add_argument_group('Voronoi Binning Options')
    vnb_group.add_argument("--target-snr", type=float, default=20.0,
                          help="Voronoi分箱的目标信噪比 (默认: 20)")
    
    # 径向分箱参数
    rdb_group = parser.add_argument_group('Radial Binning Options')
    rdb_group.add_argument("--n-rings", type=int, default=10,
                          help="Number of radial rings (default: 10)")
    rdb_group.add_argument("--center-x", type=float,
                          help="X coordinate of galaxy center (default: image center)")
    rdb_group.add_argument("--center-y", type=float,
                          help="Y coordinate of galaxy center (default: image center)")
    rdb_group.add_argument("--pa", type=float, default=0.0,
                          help="Position angle in degrees (default: 0)")
    rdb_group.add_argument("--ellipticity", type=float, default=0.0,
                          help="Ellipticity (e=1-b/a) for radial binning (default: 0)")
    rdb_group.add_argument("--linear-spacing", action="store_true",
                          help="Use linear spacing instead of logarithmic (for uniform bins mode)")
    rdb_group.add_argument("--adaptive-rdb", action="store_true",
                          help="Use adaptive radial binning to balance SNR")
    rdb_group.add_argument("--rdb-target-snr", type=float,
                          help="Target SNR for adaptive radial binning (default: same as VNB)")
    
    # 红移参数
    parser.add_argument("--redshift", type=float, default=0.0,
                       help="星系红移 (默认: 0)")
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建配置对象
    config = P2PConfig(args)
    
    # 创建文件日志处理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = config.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{config.galaxy_name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    
    logging.info(f"开始分析 - 日志保存到: {log_path}")
    logging.info(f"配置参数:\n{config}")
    
    # 根据模式运行不同的分析
    try:
        if args.mode == "VNB" or args.mode == "ALL":
            # VNB模式
            print(f"Running Voronoi binning with target SNR={args.target_snr}")
            run_vnb_analysis(config, target_snr=args.target_snr)
            
        if args.mode == "RDB" or args.mode == "ALL":
            # RDB模式
            log_spacing = not args.linear_spacing
            spacing_type = "logarithmic" if log_spacing else "linear"
            
            # 设置自适应分箱参数
            if args.adaptive_rdb:
                # 使用与VNB相同的目标SNR，除非明确指定
                rdb_target_snr = args.rdb_target_snr if args.rdb_target_snr else args.target_snr
                print(f"Running Adaptive Radial binning with target SNR={rdb_target_snr}, max bins={args.n_rings}")
            else:
                print(f"Running Uniform Radial binning with {args.n_rings} rings, {spacing_type} spacing")
            
            if args.center_x is not None and args.center_y is not None:
                print(f"Using specified center: ({args.center_x}, {args.center_y})")
            if args.pa != 0 or args.ellipticity > 0:
                print(f"Using PA={args.pa}°, ellipticity={args.ellipticity}")
                
            run_rdb_analysis(config, n_bins=args.n_rings, 
                           center_x=args.center_x, center_y=args.center_y,
                           pa=args.pa, ellipticity=args.ellipticity,
                           log_spacing=log_spacing,
                           adaptive_bins=args.adaptive_rdb, 
                           target_snr=rdb_target_snr if args.adaptive_rdb else None)
        
        logging.info("分析完成")
        print(f"Analysis completed. Results in {config.output_dir}, logs in {log_path}")
        
    except Exception as e:
        logging.error(f"程序执行过程中出错: {str(e)}")
        logging.exception("堆栈跟踪:")
        print(f"Error during execution: {str(e)}")
        print(f"See log for details: {log_path}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())