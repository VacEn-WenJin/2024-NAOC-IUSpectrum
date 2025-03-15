import warnings
from typing import Union, Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
from astropy import constants
from astropy.io import fits
from joblib import delayed
from ppxf import ppxf_util
from ppxf.ppxf import ppxf
from ppxf.sps_util import sps_lib

from utils.parallel import ParallelTqdm
from utils.calc import apply_velocity_shift, resample_spectrum

# get speed of light from astropy in km/s
SPEED_OF_LIGHT = constants.c.to('km/s').value


class MUSECube:
    def __init__(
            self,
            filename: str,
            redshift: float,
            wvl_air_angstrom_range: tuple[float, float] = (4822, 5212),
    ) -> None:
        """
        Read MUSE data cube, extract relevant information and preprocess it
        (de-redshift, ppxf log-rebin, and form coordinate grid).
        @param filename: filename of the MUSE data cube
        @param redshift: redshift of the galaxy
        @param wvl_air_angstrom_range: wavelength range to consider (Angstrom in air wavelength)
        """
        self._filename = filename
        self._wvl_air_angstrom_range = wvl_air_angstrom_range
        self._redshift = redshift

        self._read_fits_file()
        self._preprocess_cube()

        # initialize fields for storing results
        self._optimal_tmpls = None
        self._emission_flux = {}
        self._emission_sig = {}
        self._emission_vel = {}
        self._gas_bestfit_field = None

    def _read_fits_file(self) -> None:
        """
        Read MUSE cube data, create a dummy noise cube, extract the wavelength axis,
        and obtain instrumental information from the FITS header.
        """
        cut_lhs, cut_rhs = 1, 1

        with fits.open(self._filename) as fits_hdu:
            # load fits header info
            self._fits_hdu_header = fits_hdu[0].header
            
            # 增强处理多类型FITS文件的能力
            # 首先尝试主HDU
            if fits_hdu[0].data is not None and len(fits_hdu[0].data.shape) == 3:
                self._raw_cube_data = fits_hdu[0].data[cut_lhs:-cut_rhs, :, :] * 1e18
            
            # 如果主HDU没有数据立方体，检查扩展HDU
            elif len(fits_hdu) > 1:
                data_found = False
                for ext in range(1, len(fits_hdu)):
                    if fits_hdu[ext].data is not None and len(fits_hdu[ext].data.shape) == 3:
                        self._raw_cube_data = fits_hdu[ext].data[cut_lhs:-cut_rhs, :, :] * 1e18
                        # 合并头信息
                        for key in fits_hdu[ext].header:
                            if key not in ('XTENSION', 'BITPIX', 'NAXIS', 'PCOUNT', 'GCOUNT'):
                                self._fits_hdu_header[key] = fits_hdu[ext].header[key]
                        data_found = True
                        break
                
                if not data_found:
                    raise ValueError("No valid 3D data found in the FITS file")
            else:
                raise ValueError("Invalid FITS file structure: no data cube found")
            
            # create a variance cube (dummy, as the file contains no errors)
            self._raw_cube_var = np.ones_like(self._raw_cube_data)

            # calculate wavelength axis
            if 'CRVAL3' in self._fits_hdu_header and 'CD3_3' in self._fits_hdu_header:
                self._obs_wvl_air_angstrom = (
                        self._fits_hdu_header['CRVAL3'] + self._fits_hdu_header['CD3_3'] *
                        (np.arange(self._raw_cube_data.shape[0]) + cut_lhs)
                )
            else:
                # 如果找不到波长信息，生成一个线性波长轴并发出警告
                warnings.warn("Wavelength information not found in header. Using a linear scale.")
                self._obs_wvl_air_angstrom = np.linspace(4000, 7000, self._raw_cube_data.shape[0])

            self._FWHM_gal = 1
            # instrument specific parameters from ESO
            if 'CD1_1' in self._fits_hdu_header and 'CD2_1' in self._fits_hdu_header:
                self._pxl_size_x = abs(np.sqrt(
                    self._fits_hdu_header['CD1_1'] ** 2 + self._fits_hdu_header['CD2_1'] ** 2
                )) * 3600
                self._pxl_size_y = abs(np.sqrt(
                    self._fits_hdu_header['CD1_2'] ** 2 + self._fits_hdu_header['CD2_2'] ** 2
                )) * 3600
            else:
                # 默认像素大小
                self._pxl_size_x = 0.2  # 弧秒
                self._pxl_size_y = 0.2  # 弧秒
                warnings.warn("Pixel size information not found in header. Using default value of 0.2 arcsec.")

    def _preprocess_cube(self):
        # 应用红移修正
        wvl_air_angstrom = self._obs_wvl_air_angstrom / (1 + self._redshift)

        # 选择有效波长范围
        valid_mask = (
                (wvl_air_angstrom > self._wvl_air_angstrom_range[0]) &
                (wvl_air_angstrom < self._wvl_air_angstrom_range[1])
        )
        self._wvl_air_angstrom = wvl_air_angstrom[valid_mask]
        self._cube_data = self._raw_cube_data[valid_mask, :, :]
        self._cube_var = self._raw_cube_var[valid_mask, :, :]

        # derive signal and noise
        signal_2d = np.nanmedian(self._cube_data, axis=0)
        noise_2d = np.sqrt(np.nanmedian(self._cube_var, axis=0))
        self._signal = signal_2d.ravel()
        self._noise = noise_2d.ravel()

        # create spatial coordinates for each spaxel using the image indices
        ny, nx = signal_2d.shape  # note: cube shape is (n_wave, ny, nx)
        rows, cols = np.indices((ny, nx))
        
        # 增强：查找最亮像素，基于整个波长范围的积分流量
        flux_sum = np.nansum(self._cube_data, axis=0)
        brightest_idx = np.unravel_index(np.nanargmax(flux_sum), flux_sum.shape)
        brightest_y, brightest_x = brightest_idx
        
        # centering coordinates on the brightest spaxel and scale to arcseconds
        self.x = (cols.ravel() - brightest_x) * self._pxl_size_x
        self.y = (rows.ravel() - brightest_y) * self._pxl_size_y

        # reshape cube to 2D: each column corresponds to one spaxel spectrum
        n_wvl = self._cube_data.shape[0]
        self._spectra_2d = self._cube_data.reshape(n_wvl, -1)
        self._variance_2d = self._cube_var.reshape(n_wvl, -1)

        # log-rebin of the spectra
        self._vel_scale = np.min(
            SPEED_OF_LIGHT * np.diff(np.log(self._wvl_air_angstrom))
        )

        self._spectra, self._ln_lambda_gal, _ = ppxf_util.log_rebin(
            lam=[np.min(self._wvl_air_angstrom), np.max(self._wvl_air_angstrom)],
            spec=self._spectra_2d,
            velscale=self._vel_scale
        )
        self._log_variance, _, _ = ppxf_util.log_rebin(
            lam=[np.min(self._wvl_air_angstrom), np.max(self._wvl_air_angstrom)],
            spec=self._variance_2d,
            velscale=self._vel_scale
        )
        self._lambda_gal = np.exp(self._ln_lambda_gal)
        self._FWHM_gal = self._FWHM_gal / (1 + self._redshift)

        self._row = rows.ravel() + 1
        self._col = cols.ravel() + 1

        # initialize fields for storing results
        self._ny, self._nx = ny, nx
        self._velocity_field = np.full((ny, nx), np.nan)
        self._dispersion_field = np.full((ny, nx), np.nan)
        self._bestfit_field = np.full((self._spectra.shape[0], ny, nx), np.nan)
        self._apoly = []

    # first time fit (FTF)
    def fit_spectra(
            self,
            template_filename: str,
            ppxf_vel_init: int = 0,
            ppxf_vel_disp_init: int = 40,
            ppxf_deg: int = 3,
            n_jobs: int = -1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
        """
        Fit the stellar continuum in each spaxel using pPXF.
        @param template_filename: Filename of the stellar template
        @param ppxf_vel_init: Initial guess for the velocity in pPXF
        @param ppxf_vel_disp_init: Initial guess for the velocity dispersion in pPXF
        @param ppxf_deg: Degree of the additive polynomial for pPXF
        @param n_jobs: Number of parallel jobs to run (-1 means using all processors).
        @return: Tuple of velocity field, dispersion field, bestfit field, optimal templates, and additive polynomials.
        """

        # load template
        sps = sps_lib(
            filename=template_filename,
            velscale=self._vel_scale,
            fwhm_gal=None,
            norm_range=self._wvl_air_angstrom_range
        )
        sps.templates = sps.templates.reshape(sps.templates.shape[0], -1)
        # normalize stellar template
        sps.templates /= np.median(sps.templates)
        tmpl_mask = ppxf_util.determine_mask(
            ln_lam=self._ln_lambda_gal,
            lam_range_temp=np.exp(sps.ln_lam_temp[[0, -1]]),
            width=1000
        )
        # update optimal templates shape
        self._optimal_tmpls = np.empty((sps.templates.shape[0], self._ny, self._nx))

        n_wvl, n_spaxel = self._spectra.shape

        def fit_spaxel(idx):
            i, j = np.unravel_index(idx, (self._ny, self._nx))
            galaxy_data = self._spectra[:, idx]
            # Use the square root of the variance as the noise estimate
            galaxy_noise = np.sqrt(self._log_variance[:, idx])

            # 跳过低信噪比或无效像素
            if np.count_nonzero(galaxy_data) < 50 or np.count_nonzero(np.isfinite(galaxy_data)) < 50:
                return i, j, None

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', category=RuntimeWarning,
                    message='invalid value encountered in scalar divide'
                )
                try:
                    pp = ppxf(
                        sps.templates, galaxy_data, galaxy_noise,
                        self._vel_scale, mask=tmpl_mask,
                        start=[ppxf_vel_init, ppxf_vel_disp_init], degree=ppxf_deg,
                        lam=self._lambda_gal, lam_temp=sps.lam_temp,
                        quiet=True
                    )
                    
                    # 确保弥散值合理
                    if pp.sol[1] < 0:
                        pp.sol[1] = 10.0  # 设置为合理的最小值
                        
                    return i, j, (
                        pp.sol[0], pp.sol[1], pp.bestfit,
                        sps.templates @ pp.weights,
                        pp.apoly
                    )
                except Exception as e:
                    # 如果拟合失败，尝试使用更简单的配置再试一次
                    try:
                        pp = ppxf(
                            sps.templates, galaxy_data, galaxy_noise,
                            self._vel_scale, mask=tmpl_mask,
                            start=[ppxf_vel_init, ppxf_vel_disp_init], degree=0,  # 简化为常数多项式
                            lam=self._lambda_gal, lam_temp=sps.lam_temp,
                            quiet=True
                        )
                        
                        # 确保弥散值合理
                        if pp.sol[1] < 0:
                            pp.sol[1] = 10.0
                            
                        return i, j, (
                            pp.sol[0], pp.sol[1], pp.bestfit,
                            sps.templates @ pp.weights,
                            pp.apoly
                        )
                    except:
                        # 两次尝试都失败，返回None
                        return i, j, None

        fit_results = ParallelTqdm(
            n_jobs=n_jobs, desc='Fitting spectra', total_tasks=n_spaxel
        )(delayed(fit_spaxel)(idx) for idx in range(n_spaxel))
        
        for fit_result in fit_results:
            if fit_result[2] is None:
                continue
            row, col, (vel, disp, bestfit, optimal_tmpl, apoly_val) = fit_result
            self._velocity_field[row, col] = vel
            self._dispersion_field[row, col] = disp
            self._bestfit_field[:, row, col] = bestfit
            self._optimal_tmpls[:, row, col] = optimal_tmpl
            self._apoly.append(apoly_val)

        return (self._velocity_field, self._dispersion_field,
                self._bestfit_field, self._optimal_tmpls, self._apoly)

    # second time fit (STF)
    def fit_emission_lines(
        self,
        line_names: Optional[List[str]] = None,
        ppxf_vel_init: Optional[np.ndarray] = None,
        ppxf_sig_init: float = 50.0,
        ppxf_deg: int = 4,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        在恒星模板基础上拟合发射线成分
        @param line_names: 发射线名称列表，默认None使用标准集合
        @param ppxf_vel_init: 发射线速度初始值，默认为恒星速度场
        @param ppxf_sig_init: 发射线弥散初始值，默认为50
        @param ppxf_deg: 多项式阶数
        @param n_jobs: 并行任务数
        @return: 发射线拟合结果字典
        """
        if line_names is None:
            # 默认发射线集合
            line_names = ['OII3726', 'OII3729', 'Hgamma', 'Hbeta', 'OIII4959', 'OIII5007', 
                          'HeI5876', 'OI6300', 'Halpha', 'NII6548', 'NII6583', 'SII6716', 'SII6731']
        
        # 检查是否已经进行过恒星拟合
        if self._optimal_tmpls is None:
            raise ValueError("Must run fit_spectra() before fit_emission_lines()")
        
        if ppxf_vel_init is None:
            # 使用恒星速度场作为初始值
            ppxf_vel_init = self._velocity_field
        
        # 初始化结果存储
        self._emission_flux = {name: np.full((self._ny, self._nx), np.nan) for name in line_names}
        self._emission_vel = {name: np.full((self._ny, self._nx), np.nan) for name in line_names}
        self._emission_sig = {name: np.full((self._ny, self._nx), np.nan) for name in line_names}
        self._gas_bestfit_field = np.full_like(self._bestfit_field, np.nan)
        
        # 构建波长对应表
        line_wave = {
            'OII3726': 3726.03, 'OII3729': 3728.82, 'Hgamma': 4340.47,
            'OIII4363': 4363.21, 'HeII4686': 4685.7, 'Hbeta': 4861.33,
            'OIII4959': 4958.92, 'OIII5007': 5006.84, 'HeI5876': 5875.67,
            'OI6300': 6300.30, 'Halpha': 6562.80, 'NII6548': 6548.03,
            'NII6583': 6583.41, 'SII6716': 6716.47, 'SII6731': 6730.85
        }
        
        # 提取波长值
        emission_lines = np.array([line_wave[name] for name in line_names if name in line_wave])
        
        n_wvl, n_spaxel = self._spectra.shape
        
        # 定义拟合函数
        def fit_spaxel_emission(idx):
            """拟合单个像素的发射线"""
            i, j = np.unravel_index(idx, (self._ny, self._nx))
            galaxy_data = self._spectra[:, idx]
            galaxy_noise = np.sqrt(self._log_variance[:, idx])
            
            # 如果没有足够的数据或第一次拟合失败，跳过
            if (np.count_nonzero(galaxy_data) < 50 or 
                np.count_nonzero(np.isfinite(galaxy_data)) < 50 or 
                np.isnan(self._velocity_field[i, j])):
                return i, j, None
            
            # 获取最佳恒星模板和恒星速度参数
            optimal_template = self._optimal_tmpls[:, i, j]
            
            # 获取初始速度值
            if isinstance(ppxf_vel_init, np.ndarray):
                vel_init = ppxf_vel_init[i, j]
                if np.isnan(vel_init):
                    vel_init = 0
            else:
                vel_init = ppxf_vel_init if ppxf_vel_init is not None else 0
            
            # 使用ppxf_util.emission_lines创建气体模板
            gas_templates, gas_names, emission_wave = ppxf_util.emission_lines(
                self._ln_lambda_gal, emission_lines, FWHM=ppxf_sig_init
            )
            
            # 合并恒星和气体模板
            all_templates = np.column_stack([optimal_template.reshape(-1, 1), gas_templates])
            
            # 定义成分标记
            component = [0] + [1] * gas_templates.shape[1]
            gas_component = np.array(component) > 0
            
            # 设置拟合参数
            start = [
                [vel_init, self._dispersion_field[i, j]],  # 恒星成分
                [vel_init, ppxf_sig_init]                 # 气体成分
            ]
            
            # 设置约束范围
            vlim = lambda x: vel_init + x*np.array([-100, 100])
            bounds = [
                [vlim(2), [20, 300]],  # 恒星成分
                [vlim(2), [20, 100]]   # 气体成分
            ]
            
            # 设置tied参数
            moments = [2, 2]  # 恒星和气体的moments
            ncomp = len(moments)
            tied = [['', ''] for _ in range(ncomp)]
            
            # 执行pPXF拟合
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                try:
                    pp = ppxf(
                        all_templates, galaxy_data, galaxy_noise, 
                        self._vel_scale, start, mask=None,
                        moments=moments, degree=ppxf_deg, mdegree=-1,
                        component=component, gas_component=gas_component,
                        gas_names=gas_names, lam=self._lambda_gal,
                        lam_temp=np.exp(self._ln_lambda_gal),
                        bounds=bounds, tied=tied,
                        quiet=True
                    )
                    
                    # 提取结果
                    result = {
                        'flux': pp.gas_flux if hasattr(pp, 'gas_flux') else None,
                        'gas_bestfit': pp.gas_bestfit if hasattr(pp, 'gas_bestfit') else None,
                        'gas_bestfit_templates': pp.gas_bestfit_templates if hasattr(pp, 'gas_bestfit_templates') else None,
                        'sol': pp.sol,
                        'gas_sol': pp.gas_sol if hasattr(pp, 'gas_sol') else None
                    }
                    return i, j, result
                except Exception as e:
                    # 拟合失败时，尝试更简单的配置
                    try:
                        pp = ppxf(
                            all_templates, galaxy_data, galaxy_noise, 
                            self._vel_scale, start, mask=None,
                            moments=moments, degree=0, mdegree=-1,  # 使用常数多项式
                            component=component, gas_component=gas_component,
                            gas_names=gas_names, lam=self._lambda_gal,
                            lam_temp=np.exp(self._ln_lambda_gal),
                            quiet=True
                        )
                        
                        # 提取结果
                        result = {
                            'flux': pp.gas_flux if hasattr(pp, 'gas_flux') else None,
                            'gas_bestfit': pp.gas_bestfit if hasattr(pp, 'gas_bestfit') else None,
                            'gas_bestfit_templates': pp.gas_bestfit_templates if hasattr(pp, 'gas_bestfit_templates') else None,
                            'sol': pp.sol,
                            'gas_sol': pp.gas_sol if hasattr(pp, 'gas_sol') else None
                        }
                        return i, j, result
                    except:
                        return i, j, None
        
        # 并行执行拟合
        fit_results = ParallelTqdm(
            n_jobs=n_jobs, desc='Fitting emission lines', total_tasks=n_spaxel
        )(delayed(fit_spaxel_emission)(idx) for idx in range(n_spaxel))
        
        # 处理结果
        for fit_result in fit_results:
            if fit_result[2] is None:
                continue
            
            row, col, result = fit_result
            
            # 保存发射线流量和速度信息
            if result['flux'] is not None:
                for k, name in enumerate(line_names):
                    if k < len(result['flux']):
                        self._emission_flux[name][row, col] = result['flux'][k]
                        
                        if result['gas_sol'] is not None:
                            self._emission_vel[name][row, col] = result['gas_sol'][0]
                            self._emission_sig[name][row, col] = result['gas_sol'][1]
            
            # 保存气体拟合结果
            if result['gas_bestfit'] is not None:
                self._gas_bestfit_field[:, row, col] = result['gas_bestfit']
        
        # 返回结果字典
        return {
            'emission_flux': self._emission_flux,
            'emission_vel': self._emission_vel,
            'emission_sig': self._emission_sig,
            'gas_bestfit_field': self._gas_bestfit_field
        }

    @property
    def redshift(self):
        return self._redshift

    @property
    def raw_data(self):
        return {
            'obs_wvl_air_angstrom': self._obs_wvl_air_angstrom,
            'raw_cube_data': self._raw_cube_data,
            'raw_cube_var': self._raw_cube_var,
        }

    @property
    def instrument_info(self):
        return {
            'CD1_1': self._fits_hdu_header.get('CD1_1', 0),
            'CD1_2': self._fits_hdu_header.get('CD1_2', 0),
            'CD2_1': self._fits_hdu_header.get('CD2_1', 0),
            'CD2_2': self._fits_hdu_header.get('CD2_2', 0),
            'CRVAL1': self._fits_hdu_header.get('CRVAL1', 0),
            'CRVAL2': self._fits_hdu_header.get('CRVAL2', 0),
            'pxl_size_x': self._pxl_size_x,
            'pxl_size_y': self._pxl_size_y
        }

    @property
    def fits_hdu_header(self):
        return self._fits_hdu_header

    @property
    def fit_spectra_result(self):
        """获取恒星拟合结果"""
        return {
            'velocity_field': self._velocity_field,
            'dispersion_field': self._dispersion_field,
            'bestfit_field': self._bestfit_field,
            'optimal_tmpls': self._optimal_tmpls,
            'apoly': self._apoly
        }
        
    @property
    def fit_emission_result(self):
        """获取发射线拟合结果"""
        # 检查是否已经进行过发射线拟合
        if self._gas_bestfit_field is None:
            warnings.warn("Emission line fitting has not been performed")
            return None
            
        return {
            'emission_flux': self._emission_flux,
            'emission_vel': self._emission_vel,
            'emission_sig': self._emission_sig,
            'gas_bestfit_field': self._gas_bestfit_field
        }

    def get_grid_stat(
            self, x_idx: Union[int, list[int]], y_idx: Union[int, list[int]]
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get the velocity, dispersion, and bestfit for the given spaxel indices.
        @param x_idx: x index of the spaxel
        @param y_idx: y index of the spaxel
        @return: Dictionary of velocity, dispersion, and bestfit for the given spaxel indices.
        """
        if isinstance(x_idx, int):
            x_idx = [x_idx]
        if isinstance(y_idx, int):
            y_idx = [y_idx]

        return {
            (x, y): (
                self._velocity_field[y, x],
                self._dispersion_field[y, x],
                self._bestfit_field[:, y, x]
            )
            for x in x_idx for y in y_idx
        }
        
    def save_results(self, output_dir: str, prefix: Optional[str] = None):
        """
        保存拟合结果到FITS文件
        
        Parameters:
        -----------
        output_dir: 输出目录
        prefix: 文件名前缀，默认使用文件名
        """
        import os
        from astropy.io import fits
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取文件名
        if prefix is None:
            prefix = os.path.splitext(os.path.basename(self._filename))[0]
        
        # 创建头信息
        hdr = fits.Header()
        for key, value in self.instrument_info.items():
            hdr[key] = value
        hdr['REDSHIFT'] = self._redshift
        
        # 保存速度场
        hdu = fits.PrimaryHDU(self._velocity_field, header=hdr)
        hdu.writeto(os.path.join(output_dir, f"{prefix}_velocity.fits"), overwrite=True)
        
        # 保存弥散场
        hdu = fits.PrimaryHDU(self._dispersion_field, header=hdr)
        hdu.writeto(os.path.join(output_dir, f"{prefix}_dispersion.fits"), overwrite=True)
        
        # 保存最佳拟合结果
        hdu = fits.PrimaryHDU(self._bestfit_field, header=hdr)
        hdu.writeto(os.path.join(output_dir, f"{prefix}_bestfit.fits"), overwrite=True)
        
        # 如果有发射线结果，也保存
        if self._gas_bestfit_field is not None:
            # 保存气体拟合结果
            hdu = fits.PrimaryHDU(self._gas_bestfit_field, header=hdr)
            hdu.writeto(os.path.join(output_dir, f"{prefix}_gas_bestfit.fits"), overwrite=True)
            
            # 保存发射线流量
            for name, flux in self._emission_flux.items():
                hdu = fits.PrimaryHDU(flux, header=hdr)
                hdu.writeto(os.path.join(output_dir, f"{prefix}_{name}_flux.fits"), overwrite=True)

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(filename={self._filename!r}, '
                f'redshift={self._redshift!r}, '
                f'wvl_air_angstrom_range={self._wvl_air_angstrom_range!r})')


if __name__ == '__main__':
    example_muse_cube = MUSECube(
        filename='../data/MUSE/VCC1588_stack.fits',
        redshift=.0042
    )

    example_muse_cube.fit_spectra(
        template_filename='../data/templates/spectra_emiles_9.0.npz'
    )

    print(
        example_muse_cube,
        example_muse_cube.instrument_info,
        example_muse_cube.raw_data,
        example_muse_cube.fit_spectra_result,
        sep='\n'
    )