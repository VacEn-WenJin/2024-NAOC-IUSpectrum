"""
Galaxy Parameters Calculation Tools
"""
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Union


class GalaxyParameters:
    """Galaxy parameters calculation class"""
    
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
        Initialize galaxy parameters calculation
        
        Parameters
        ----------
        velocity_field : ndarray
            Velocity field (2D array)
        dispersion_field : ndarray
            Dispersion field (2D array)
        x : ndarray, optional
            x coordinate array, default uses pixel coordinates
        y : ndarray, optional
            y coordinate array, default uses pixel coordinates
        pixelsize : float, default=1.0
            Pixel size (arcsec)
        distance : float, optional
            Galaxy distance (Mpc)
        """
        self.velocity_field = velocity_field
        self.dispersion_field = dispersion_field
        self.pixelsize = pixelsize
        self.distance = distance
        
        n_y, n_x = velocity_field.shape
        if x is None or y is None:
            # Use pixel coordinates
            y_grid, x_grid = np.indices((n_y, n_x))
            self.x = x_grid.ravel()
            self.y = y_grid.ravel()
        else:
            self.x = x
            self.y = y
            
        # Initialize result storage
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
        Fit rotation curve and kinematic parameters
        
        Parameters
        ----------
        center_x : float, optional
            Center x coordinate
        center_y : float, optional
            Center y coordinate
        pa_initial : float, default=0.0
            Initial position angle guess (degrees)
        r_max : float, optional
            Maximum fitting radius
            
        Returns
        -------
        dict
            Dictionary of fitting parameters
        """
        n_y, n_x = self.velocity_field.shape
        
        # Set default center
        if center_x is None:
            center_x = n_x / 2
        if center_y is None:
            center_y = n_y / 2
        
        try:
            # Extract valid velocity values
            vel_data = self.velocity_field.ravel()
            valid_mask = np.isfinite(vel_data)
            x = self.x[valid_mask]
            y = self.y[valid_mask]
            vel = vel_data[valid_mask]
            
            # Check if enough valid data points
            if len(vel) < 10:
                warnings.warn("Not enough valid velocity points for fitting")
                return {
                    'pa': pa_initial,
                    'vsys': 0.0,
                    'vmax': 0.0,
                    'center': (center_x, center_y),
                    'rotation_curve': np.array([[0, 0]])
                }
            
            # Search for position angle
            best_pa = pa_initial
            best_amplitude = 0
            
            for test_pa in np.linspace(0, 180, 19):  # 10 degree step search
                pa_rad = np.radians(test_pa)
                
                # Project onto position angle
                proj_dist = (x - center_x) * np.cos(pa_rad) + (y - center_y) * np.sin(pa_rad)
                
                # Calculate velocity amplitude
                vel_pos = vel[proj_dist > 0]
                vel_neg = vel[proj_dist < 0]
                
                if len(vel_pos) > 0 and len(vel_neg) > 0:
                    amplitude = np.median(vel_pos) - np.median(vel_neg)
                    
                    if abs(amplitude) > abs(best_amplitude):
                        best_amplitude = amplitude
                        best_pa = test_pa
            
            # Fine search
            search_range = 20  # Search within ±10 degrees of best PA
            for test_pa in np.linspace(best_pa - search_range/2, best_pa + search_range/2, 21):
                pa_rad = np.radians(test_pa % 180)
                
                # Project onto position angle
                proj_dist = (x - center_x) * np.cos(pa_rad) + (y - center_y) * np.sin(pa_rad)
                
                # Calculate velocity amplitude
                vel_pos = vel[proj_dist > 0]
                vel_neg = vel[proj_dist < 0]
                
                if len(vel_pos) > 0 and len(vel_neg) > 0:
                    amplitude = np.median(vel_pos) - np.median(vel_neg)
                    
                    if abs(amplitude) > abs(best_amplitude):
                        best_amplitude = amplitude
                        best_pa = test_pa % 180
            
            # Use best position angle to calculate radial distribution
            best_pa_rad = np.radians(best_pa)
            
            # Calculate distance to center and projected distance
            dx = x - center_x
            dy = y - center_y
            radius = np.sqrt(dx**2 + dy**2)
            proj_dist = dx * np.cos(best_pa_rad) + dy * np.sin(best_pa_rad)
            
            # Apply maximum radius limit
            if r_max is not None:
                radius_mask = radius <= r_max
                radius = radius[radius_mask]
                proj_dist = proj_dist[radius_mask]
                vel = vel[radius_mask]
            
            # Adjust position angle to conventional range [0, 180)
            if best_amplitude < 0:
                best_pa = (best_pa + 180) % 180
                best_amplitude = -best_amplitude
            
            # Calculate systemic velocity
            self.vsys = np.median(vel)
            self.vmax = best_amplitude / 2
            self.kinematic_pa = best_pa
            
            # Build rotation curve data
            # Calculate average velocity for multiple radial bins
            radial_bins = 10
            if r_max is None:
                r_max = np.max(radius)
            
            r_bins = np.linspace(0, r_max, radial_bins + 1)
            rotation_curve = []
            
            for i in range(radial_bins):
                r_min = r_bins[i]
                r_max_bin = r_bins[i + 1]
                
                # Select data in this radial bin
                bin_mask = (radius >= r_min) & (radius < r_max_bin)
                
                if np.sum(bin_mask) > 5:  # Need at least 5 points
                    # Calculate for positive and negative projection regions
                    pos_mask = bin_mask & (proj_dist > 0)
                    neg_mask = bin_mask & (proj_dist < 0)
                    
                    if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
                        v_pos = np.median(vel[pos_mask])
                        v_neg = np.median(vel[neg_mask])
                        v_rot = (v_pos - v_neg) / 2
                        r_mean = np.mean(radius[bin_mask])
                        
                        rotation_curve.append((r_mean, v_rot))
            
            # Convert to array
            if rotation_curve:
                rotation_curve = np.array(rotation_curve)
            else:
                rotation_curve = np.array([(0, 0)])
        
        except Exception as e:
            warnings.warn(f"Error in rotation curve fitting: {str(e)}")
            # Return default values
            return {
                'pa': pa_initial,
                'vsys': 0.0,
                'vmax': 0.0,
                'center': (center_x, center_y),
                'rotation_curve': np.array([[0, 0]])
            }
            
        # Return fitting result
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
        Calculate kinematic statistics
        
        Returns
        -------
        dict
            Dictionary of kinematic parameters
        """
        try:
            # Get valid velocity and dispersion values
            vel = self.velocity_field.ravel()
            disp = self.dispersion_field.ravel()
            
            # Extract valid data
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
            
            # Calculate mean dispersion
            self.sigma_mean = np.mean(disp_valid)
            
            # Calculate V/σ
            v_rms = np.sqrt(np.mean(vel_valid**2))
            self.v_over_sigma = v_rms / self.sigma_mean if self.sigma_mean > 0 else np.nan
            
            # Calculate λR parameter (needs flux weighting)
            # Simplified approach, assume all pixels have same weight
            flux = np.ones_like(vel_valid)
            
            # Calculate center
            n_y, n_x = self.velocity_field.shape
            center_x = n_x / 2
            center_y = n_y / 2
            
            # Calculate radius for each pixel
            r = np.sqrt((x_valid - center_x)**2 + (y_valid - center_y)**2)
            
            # λR = Σ(Fi*Ri*|Vi|) / Σ(Fi*Ri*sqrt(Vi^2 + σi^2))
            numerator = np.sum(flux * r * np.abs(vel_valid))
            denominator = np.sum(flux * r * np.sqrt(vel_valid**2 + disp_valid**2))
            
            self.lambda_r = numerator / denominator if denominator > 0 else np.nan
            
            # Return results
            result = {
                'sigma_mean': self.sigma_mean,
                'v_over_sigma': self.v_over_sigma,
                'lambda_r': self.lambda_r
            }
            
            return result
        except Exception as e:
            warnings.warn(f"Error calculating kinematics: {str(e)}")
            return {
                'sigma_mean': np.nan,
                'v_over_sigma': np.nan,
                'lambda_r': np.nan
            }
    
    def calculate_physical_scales(self) -> Dict:
        """
        Calculate physical scale parameters
        
        Returns
        -------
        dict
            Dictionary of physical scale parameters
        """
        if self.distance is None:
            warnings.warn("Galaxy distance not provided, physical scales unavailable")
            return {'scale': np.nan, 'r_eff_kpc': np.nan}
        
        try:
            # Calculate linear scale (kpc/arcsec)
            scale = self.distance * 1000 * np.pi / (180 * 3600)  # kpc/arcsec
            
            # Calculate pixel physical size
            pixel_kpc = scale * self.pixelsize
            
            # Calculate effective radius
            n_y, n_x = self.velocity_field.shape
            
            # Simple estimate: assume r_eff is 1/4 of image size
            r_eff_pix = min(n_y, n_x) / 4
            r_eff_kpc = r_eff_pix * pixel_kpc
            
            return {
                'scale': pixel_kpc,
                'r_eff_kpc': r_eff_kpc
            }
        except Exception as e:
            warnings.warn(f"Error calculating physical scales: {str(e)}")
            return {'scale': np.nan, 'r_eff_kpc': np.nan}