"""
File: HistogramAnalyzer.py
Project: MPF-Toolbox
Created Date: October 2022
Author: Qiuyi Shen
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Tuple
from skimage import measure
import os
import pydicom


class HistogramAnalyzer:
    """用于分析和可视化RMPFSL数据的直方图工具类"""

    def __init__(self, data_type: str = "RMPFSL"):
        """
        初始化HistogramAnalyzer类

        Parameters:
        -----------
        vmax : float, optional
            图像显示的最大值，如果为None则使用图像实际最大值
        """
        self.default_figsize = (15, 6)
        self.default_bins = 30
        self.default_color = '#3498DB' 
        self.default_edge_color = '#2C3E50' 
        self.default_mask_color = "red"
        self.colormap = "jet"
        self.data_type = data_type
        self.vmax = 4 if data_type == "RMPFSL" else 14

    def calculate_statistics(
        self, data: np.ndarray, mask: np.ndarray
    ) -> Dict[str, float]:
        """
        计算ROI区域的统计信息

        Parameters:
        -----------
        data : np.ndarray
            RMPFSL数据数组
        mask : np.ndarray
            掩码数组

        Returns:
        --------
        Dict[str, float]
            包含均值和标准差的字典
        """
        roi_values = data[mask != 0]
        if len(roi_values) == 0:
            return {"mean": 0.0, "std": 0.0}

        return {"mean": np.mean(roi_values), "std": np.std(roi_values)}

    def plot_histogram(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        slice_num: int,
        ax: Optional[plt.Axes] = None,
        bins: int = None,
        title: Optional[str] = None,
    ) -> plt.Axes:
        """
        绘制ROI区域的直方图

        Parameters:
        -----------
        data : np.ndarray
            RMPFSL数据数组
        mask : np.ndarray
            掩码数组
        slice_num : int
            切片编号
        ax : plt.Axes, optional
            matplotlib轴对象
        bins : int, optional
            直方图的柱数
        title : str, optional
            图表标题

        Returns:
        --------
        plt.Axes
            matplotlib轴对象
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        bins = bins or self.default_bins
        roi_values = data[mask != 0]

        if len(roi_values) > 0:
            data_min = np.min(roi_values)
            data_max = np.max(roi_values)

            n_bins = 30  
            bin_edges = np.linspace(data_min, data_max, n_bins + 1)

            ax.hist(
                roi_values,
                bins=bin_edges,
                color=self.default_color,
                edgecolor=self.default_edge_color,
            )

            title = title or f"Slice {slice_num} {self.data_type} ROI Histogram"
            ax.set_title(title)
            ax.set_xlabel(f"{self.data_type} Value")
            ax.set_ylabel("Frequency")
            ax.grid(True)

            # 添加统计信息
            stats = self.calculate_statistics(data, mask)
            ax.text(
                0.98,
                0.95,
                f"Mean: {stats['mean']:.3f}\nStd: {stats['std']:.3f}",
                transform=ax.transAxes,
                horizontalalignment="right",
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8),
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No ROI values available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        return ax

    def plot_masked_image(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        slice_num: int,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
    ) -> plt.Axes:
        """
        绘制带掩码轮廓的图像

        Parameters:
        -----------
        data : np.ndarray
            图像数据数组
        mask : np.ndarray
            掩码数组
        slice_num : int
            切片编号
        ax : plt.Axes, optional
            matplotlib轴对象
        title : str, optional
            图表标题

        Returns:
        --------
        plt.Axes
            matplotlib轴对象
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        vmax = self.vmax if self.vmax is not None else np.max(data)

        im = ax.imshow(data, cmap=self.colormap, vmax=vmax)
        plt.colorbar(im, ax=ax)

        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            ax.plot(
                contour[:, 1], contour[:, 0], linewidth=2, color=self.default_mask_color
            )

        title = title or f"Slice {slice_num} Image with Mask"
        ax.set_title(title)

        return ax

    def plot_combined_analysis(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        slice_num: int,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        生成组合分析视图

        Parameters:
        -----------
        data : np.ndarray
            RMPFSL数据数组
        mask : np.ndarray
            掩码数组
        slice_num : int
            切片编号
        save_path : str, optional
            保存路径

        Returns:
        --------
        Tuple[plt.Figure, plt.Axes]
            matplotlib图形和轴对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.default_figsize)

        self.plot_masked_image(data, mask, slice_num, ax1)
        self.plot_histogram(data, mask, slice_num, ax2)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)

        return fig, (ax1, ax2)


if __name__ == "__main__":
    base_path = r"MPF_Pipeline\Sample"

    mask_path = os.path.join(base_path, "MASK\Mask.npy")
    mask_data = np.load(mask_path)

    rmpfsl_path = r"MPF_Pipeline\Sample\Rmpfsl\RMPFSL.dcm"
    rmpfsl = pydicom.dcmread(rmpfsl_path)
    rmpfsl_data = rmpfsl.pixel_array.astype(np.float32)/100

    rmpfsl_analyzer = HistogramAnalyzer(data_type="RMPFSL")
    rmpfsl_analyzer.plot_combined_analysis(rmpfsl_data, mask_data, slice_num=1)
    plt.show()

    mpf_path = r"MPF_Pipeline\Sample\MPF\MPF.dcm"
    mpf = pydicom.dcmread(mpf_path)
    mpf_data = mpf.pixel_array.astype(np.float32)/10

    mpf_analyzer = HistogramAnalyzer(data_type="MPF")
    mpf_analyzer.plot_combined_analysis(mpf_data, mask_data, slice_num=1)
    plt.show()
