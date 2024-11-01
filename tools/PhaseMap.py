"""
File: PhaseMap.py
Project: MPF-Toolbox
Created Date: October 2022
Author: Qiuyi Shen
"""

import matplotlib.pyplot as plt
import numpy as np
import cmath
import pydicom
from unwrap import unwrap
import os
from typing import List, Tuple, Optional


class PhaseMap:
    """用于处理和分析相位图的工具类"""

    def __init__(self):
        """初始化PhaseMap类"""
        self.default_figsize = (12, 8)
        self.default_cmap = "jet"
        self.default_vmax = 10

    @staticmethod
    def loadDICOM(path: str) -> np.ndarray:
        """
        加载DICOM文件并应用缩放

        Parameters:
        -----------
        path : str
            DICOM文件路径

        Returns:
        --------
        np.ndarray
            处理后的像素数组
        """
        dcm = pydicom.dcmread(path)
        slope = dcm.RescaleSlope if hasattr(dcm, "RescaleSlope") else 1
        intercept = dcm.RescaleIntercept if hasattr(dcm, "RescaleSlope") else 0
        return dcm.pixel_array * slope + intercept

    @staticmethod
    def phase_to_0_180(phase: float) -> float:
        """
        将相位转换到0-180度范围

        Parameters:
        -----------
        phase : float
            输入相位值

        Returns:
        --------
        float
            转换后的相位值
        """
        phase_deg = np.degrees(phase)
        phase_0_180 = (phase_deg + 360) % 360
        return phase_0_180

    def check_file(self, folder_path: str) -> Tuple[List[str], List[str]]:
        """
        检查文件夹中的DICOM文件

        Parameters:
        -----------
        folder_path : str
            文件夹路径

        Returns:
        --------
        Tuple[List[str], List[str]]
            实部和虚部文件路径列表
        """
        print("检查目录:", folder_path)
        realctr = []
        ictr = []

        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                ds = pydicom.dcmread(file_path)
                if ds.ImageType[3] in ["R", "r"]:
                    realctr.append(file_path)
                elif ds.ImageType[3] in ["I", "i"]:
                    ictr.append(file_path)

        if len(realctr) != len(ictr):
            print("Error: Unmatched real and imaginary parts")

        return realctr, ictr

    def calculate_phase_maps(
        self, realctr: List[str], ictr: List[str]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        计算相位图和相位差异图

        Parameters:
        -----------
        realctr : List[str]
            实部文件路径列表
        ictr : List[str]
            虚部文件路径列表

        Returns:
        --------
        Tuple[List[np.ndarray], List[np.ndarray]]
            原始相位图列表和相位差异图列表
        """
        dyn_real = [self.loadDICOM(realctr[i]) for i in [0, 2, 1, 3]]
        dyn_img = [self.loadDICOM(ictr[i]) for i in [0, 2, 1, 3]]

        complex_arrays = []
        for real, imag in zip(dyn_real, dyn_img):
            complex_img = real + 1j * imag
            phase = np.angle(complex_img)
            phase_deg = np.degrees(phase)
            complex_arrays.append(phase_deg)

        # 计算相位差
        phase_diffs = [
            np.abs(complex_arrays[0] - complex_arrays[1]),
            np.abs(complex_arrays[0] - complex_arrays[2]),
            np.abs(complex_arrays[1] - complex_arrays[3]),
            np.abs(complex_arrays[2] - complex_arrays[3]),
        ]

        return complex_arrays, phase_diffs

    def plot_phase_maps(
        self, phase_maps: List[np.ndarray], titles: Optional[List[str]] = None
    ) -> None:
        """
        绘制相位图

        Parameters:
        -----------
        phase_maps : List[np.ndarray]
            相位图列表
        titles : Optional[List[str]]
            标题列表
        """
        if titles is None:
            titles = [f"Phase Map {i+1}" for i in range(len(phase_maps))]

        plt.figure(figsize=self.default_figsize)
        for idx, (phase_map, title) in enumerate(zip(phase_maps, titles), 1):
            plt.subplot(2, 2, idx)
            plt.imshow(phase_map, cmap="gray",vmax=360)
            plt.colorbar()
            plt.title(title)
        plt.tight_layout()

    def plot_phase_differences(
        self, phase_differences: List[np.ndarray], vmax: Optional[float] = None
    ) -> None:
        """
        绘制相位差异图

        Parameters:
        -----------
        phase_differences : List[np.ndarray]
            相位差异图列表
        vmax : Optional[float]
            最大显示值
        """
        vmax = vmax or self.default_vmax
        titles = [
            "Phase Difference 1-2",
            "Phase Difference 1-3",
            "Phase Difference 2-4",
            "Phase Difference 3-4",
        ]

        plt.figure(figsize=self.default_figsize)
        for idx, (diff, title) in enumerate(zip(phase_differences, titles), 1):
            plt.subplot(2, 2, idx)
            plt.imshow(diff, cmap=self.default_cmap, vmax=vmax)
            plt.colorbar()
            plt.title(title)
        plt.tight_layout()

    def process_folder(self, folder_path: str) -> None:
        """
        处理整个文件夹

        Parameters:
        -----------
        folder_path : str
            文件夹路径
        """
        realctr, ictr = self.check_file(folder_path)
        if len(realctr) == len(ictr):
            phase_maps, phase_diffs = self.calculate_phase_maps(realctr, ictr)
            self.plot_phase_maps(phase_maps)
            self.plot_phase_differences(phase_diffs)
            plt.show()

    def save_phase_diff_results(
        self,
        phase_maps: List[np.ndarray],
        save_path: str,
        prefix: str = "phase_diff"
    ) -> None:
        """
        保存相位差异图为npy格式

        Parameters:
        -----------
        phase_maps : List[np.ndarray]
            相位差异图列表
        save_path : str
            保存目录路径
        prefix : str, optional
            文件名前缀，默认为'phase_diff'
        """
        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)

        # 定义相位差异的名称
        diff_names = [
            "1-2",
            "1-3",
            "2-4",
            "3-4"
        ]

        # 保存每个相位差异图
        for idx, (phase_map, diff_name) in enumerate(zip(phase_maps, diff_names)):
            file_name = f"{prefix}_{diff_name}.npy"
            file_path = os.path.join(save_path, file_name)
            np.save(file_path, phase_map)
            print(f"Saved phase difference {diff_name} to: {file_path}")

    def save_phase_maps(
        self,
        complex_arrays: List[np.ndarray],
        save_path: str,
        prefix: str = "phase_map"
    ) -> None:
        """
        保存原始相位图为npy格式

        Parameters:
        -----------
        complex_arrays : List[np.ndarray]
            原始相位图列表
        save_path : str
            保存目录路径
        prefix : str, optional
            文件名前缀，默认为'phase_map'
        """
        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)

        # 定义相位图的名称
        map_names = [
            "dyn1",
            "dyn2",
            "dyn3",
            "dyn4"
        ]

        # 保存每个相位图
        for idx, (phase_map, map_name) in enumerate(zip(complex_arrays, map_names)):
            file_name = f"{prefix}_{map_name}.npy"
            file_path = os.path.join(save_path, file_name)
            np.save(file_path, phase_map)
            print(f"Saved phase map {map_name} to: {file_path}")


if __name__ == "__main__":
    base_path = r"MPF_Pipeline\Sample"
    slice_folder_path = os.path.join(base_path, "Dicom")
    phase_output = os.path.join(base_path, "Phase")
    os.makedirs(phase_output, exist_ok=True)
    
    phase_processor = PhaseMap()
    
    realctr, ictr = phase_processor.check_file(slice_folder_path)
    if len(realctr) == len(ictr):
        phase_maps, phase_diffs = phase_processor.calculate_phase_maps(realctr, ictr)
        
        phase_processor.plot_phase_maps(phase_maps[:4])
        phase_processor.plot_phase_differences(phase_diffs)
        
        # phase_processor.save_phase_maps(
        #     complex_arrays=phase_maps, save_path=phase_output, prefix="phase_map"
        # )

        # phase_processor.save_phase_diff_results(
        #     phase_maps=phase_diffs, save_path=phase_output, prefix="phase_diff"
        # )
    
        plt.show()
