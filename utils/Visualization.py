"""
File: Visualization.py
Project: MPF-Toolbox
Created Date: October 2022
Author: Qiuyi Shen
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import os


class Visualization:
    """MPF和RMPFSL结果的可视化类"""

    def __init__(self):
        """初始化可视化参数"""
        # 默认参数设置
        self.default_figsize = (10, 15)
        self.default_cmap = "jet"

        # MPF和RMPFSL的特定参数
        self.MPF_VMAX = 14
        self.RMPFSL_VMAX = 4

        # 颜色设置
        self.mask_color = "red"
        self.mask_alpha = 0.3

    def plot_single_mpf(
        self,
        mpf_image: np.ndarray,
        title: Optional[str] = None,
        figsize: tuple = (8, 8),
        show_colorbar: bool = True,
    ) -> None:
        """
        绘制单张MPF图像

        Parameters:
        -----------
        mpf_image : np.ndarray
            MPF图像数组
        title : str, optional
            图像标题
        figsize : tuple
            图像大小
        show_colorbar : bool
            是否显示颜色条
        """
        plt.figure(figsize=figsize)
        mpf_image = mpf_image*100
        im = plt.imshow(mpf_image, cmap=self.default_cmap, vmax=self.MPF_VMAX)
        if show_colorbar:
            plt.colorbar(im)
        if title:
            plt.title(title)
        plt.axis("on")
        plt.tight_layout()

    def plot_single_rmpfsl(
        self,
        rmpfsl_image: np.ndarray,
        title: Optional[str] = None,
        figsize: tuple = (8, 8),
        show_colorbar: bool = True,
    ) -> None:
        """
        绘制单张RMPFSL图像

        Parameters:
        -----------
        rmpfsl_image : np.ndarray
            RMPFSL图像数组
        title : str, optional
            图像标题
        figsize : tuple
            图像大小
        show_colorbar : bool
            是否显示颜色条
        """
        plt.figure(figsize=figsize)
        im = plt.imshow(rmpfsl_image, cmap=self.default_cmap, vmax=self.RMPFSL_VMAX)
        if show_colorbar:
            plt.colorbar(im)
        if title:
            plt.title(title)
        plt.axis("on")
        plt.tight_layout()

    def plot_multiple_mpf(
        self,
        mpf_images: List[np.ndarray],
        patient_name: str,
        ncols: int = 3,
        figsize: Optional[tuple] = None,
    ) -> None:
        """
        绘制多张MPF图像

        Parameters:
        -----------
        mpf_images : List[np.ndarray]
            MPF图像列表
        patient_name : str
            病人名称
        ncols : int
            每行显示的图像数量
        figsize : tuple, optional
            图像大小
        """
        n_images = len(mpf_images)
        nrows = (n_images + ncols - 1) // ncols
        figsize = figsize or (5 * ncols, 5 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        plt.suptitle(f"Patient: {patient_name} MPF", fontsize=16, fontweight="bold")

        axes = axes.flatten()
        for idx, (ax, image) in enumerate(zip(axes, mpf_images)):
            im = ax.imshow(image, cmap=self.default_cmap, vmax=self.MPF_VMAX)
            fig.colorbar(im, ax=ax)
            ax.set_title(f"Slice {idx + 1}")

        for idx in range(len(mpf_images), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

    def plot_multiple_rmpfsl(
        self,
        rmpfsl_images: List[np.ndarray],
        patient_name: str,
        ncols: int = 3,
        figsize: Optional[tuple] = None,
    ) -> None:
        """
        绘制多张RMPFSL图像

        Parameters:
        -----------
        rmpfsl_images : List[np.ndarray]
            RMPFSL图像列表
        patient_name : str
            病人名称
        ncols : int
            每行显示的图像数量
        figsize : tuple, optional
            图像大小
        """
        n_images = len(rmpfsl_images)
        nrows = (n_images + ncols - 1) // ncols
        figsize = figsize or (5 * ncols, 5 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        plt.suptitle(f"Patient: {patient_name} RMPFSL", fontsize=16, fontweight="bold")

        axes = axes.flatten()
        for idx, (ax, image) in enumerate(zip(axes, rmpfsl_images)):
            im = ax.imshow(image, cmap=self.default_cmap, vmax=self.RMPFSL_VMAX)
            fig.colorbar(im, ax=ax)
            ax.set_title(f"Slice {idx + 1}")

        # 隐藏空白子图
        for idx in range(len(rmpfsl_images), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

    def plot_with_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        title: str,
        is_mpf: bool = False,
        figsize: tuple = (8, 8),
    ) -> None:
        """
        绘制带有掩码的图像，可选显示直方图

        Parameters:
        -----------
        image : np.ndarray
            原始图像
        mask : np.ndarray
            掩码图像
        title : str
            图像标题
        is_mpf : bool
            是否为MPF图像（决定vmax值）
        figsize : tuple
            图像大小
        """
        plt.close("all")
        plt.figure(figsize=figsize)

        if title == "MPF":
            image = image*100

        vmax = self.MPF_VMAX if is_mpf else self.RMPFSL_VMAX

        im = plt.imshow(image, cmap=self.default_cmap, vmax=vmax)

        plt.imshow(
            np.ones_like(image), cmap="gray", alpha=np.where(mask, self.mask_alpha, 0)
        )

        plt.colorbar(im)
        plt.title(f"{title} with Mask")

    def save_figure(self, save_path: str, filename: str) -> None:
        """
        保存当前图像

        Parameters:
        -----------
        save_path : str
            保存路径
        filename : str
            文件名
        """
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")
