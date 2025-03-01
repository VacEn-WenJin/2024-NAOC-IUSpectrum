import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class SSPTemplateViewer:
    def __init__(self, npz_path):
        """
        初始化模板查看器
        
        Parameters:
        -----------
        npz_path : str or Path
            npz文件的路径
        """
        self.data = np.load(npz_path)
        # 打印所有可用的键
        print("Available keys in the file:")
        print(self.data.files)
        
        # 常见的键名
        self.templates = self.data.get('templates', None)  # SSP模板
        self.wave = self.data.get('wave', None)           # 波长
        self.ages = self.data.get('ages', None)           # 年龄网格
        self.metals = self.data.get('metals', None)       # 金属丰度网格
        
        # 打印基本信息
        self._print_basic_info()
    
    def _print_basic_info(self):
        """打印模板文件的基本信息"""
        print("\nBasic Information:")
        print("-" * 50)
        
        if self.templates is not None:
            print(f"Template shape: {self.templates.shape}")
        
        if self.wave is not None:
            print(f"Wavelength range: {self.wave[0]:.2f} - {self.wave[-1]:.2f} Å")
            print(f"Number of wavelength points: {len(self.wave)}")
        
        if self.ages is not None:
            print(f"\nAges (Gyr):")
            print(self.ages)
        
        if self.metals is not None:
            print(f"\nMetallicities [M/H]:")
            print(self.metals)
    
    def plot_template(self, age_idx=None, metal_idx=None, normalize=True):
        """
        绘制特定的模板光谱
        
        Parameters:
        -----------
        age_idx : int, optional
            年龄索引
        metal_idx : int, optional
            金属丰度索引
        normalize : bool, default=True
            是否归一化光谱
        """
        if self.templates is None or self.wave is None:
            print("No template or wavelength data available")
            return
            
        if age_idx is None:
            age_idx = len(self.ages) // 2
        if metal_idx is None:
            metal_idx = len(self.metals) // 2
            
        # 获取光谱
        spectrum = self.templates[age_idx, metal_idx, :]
        if normalize:
            spectrum = spectrum / np.median(spectrum)
            
        plt.figure(figsize=(12, 6))
        plt.plot(self.wave, spectrum, 'k-', lw=1)
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Flux (normalized)' if normalize else 'Flux')
        plt.title(f'SSP Template: Age={self.ages[age_idx]:.2f} Gyr, [M/H]={self.metals[metal_idx]:.2f}')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_age_sequence(self, metal_idx=None, n_ages=5):
        """
        绘制不同年龄的模板序列
        
        Parameters:
        -----------
        metal_idx : int, optional
            金属丰度索引
        n_ages : int, default=5
            要显示的年龄数量
        """
        if metal_idx is None:
            metal_idx = len(self.metals) // 2
            
        age_indices = np.linspace(0, len(self.ages)-1, n_ages, dtype=int)
        
        plt.figure(figsize=(12, 6))
        for idx in age_indices:
            spectrum = self.templates[idx, metal_idx, :]
            spectrum = spectrum / np.median(spectrum)
            plt.plot(self.wave, spectrum + idx*0.5, 
                    label=f'Age={self.ages[idx]:.2f} Gyr')
            
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Normalized Flux (offset for clarity)')
        plt.title(f'Age Sequence at [M/H]={self.metals[metal_idx]:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_template_stats(self):
        """返回模板的基本统计信息"""
        stats = {
            'flux_min': np.min(self.templates),
            'flux_max': np.max(self.templates),
            'flux_mean': np.mean(self.templates),
            'flux_std': np.std(self.templates)
        }
        return stats

# 使用示例
if __name__ == "__main__":
    # 替换为你的npz文件路径
    npz_path = "path_to_your_template.npz"
    viewer = SSPTemplateViewer(npz_path)
    
    # 查看单个模板
    viewer.plot_template()
    
    # 查看年龄序列
    viewer.plot_age_sequence()
    
    # 获取统计信息
    stats = viewer.get_template_stats()
    print("\nTemplate Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")

        