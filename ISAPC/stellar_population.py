"""
Stellar population analysis utilities
"""
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple


class WeightParser:
    """Class for parsing fitting weights to physical parameters."""
    
    def __init__(self, template_path: Union[str, Path]) -> None:
        """Initialize with SSP template.
        
        Args:
            template_path: Path to the SSP template .npz file
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
        """Parse weights to get mean log(Age) and [M/H].
        
        Args:
            weights: Fitting weights (150,)
            
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
        """Get all physical parameters from weights."""
        log_age, metal = self.parse_weights(weights)
        
        return {
            'log_age': log_age,
            'age': 10**log_age,
            'metallicity': metal
        }