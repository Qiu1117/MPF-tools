"""
File: Json2Npy.py
Project: MPF-Toolbox
Created Date: October 2022
Author: Qiuyi Shen
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from typing import Optional, List


class Json2Npy:
    """用于处理JSON文件转换为NPY文件的工具类"""

    def __init__(self, folder_path: str):
        """
        初始化Json2Npy类

        Parameters:
        -----------
        folder_path : str
            要处理的文件夹路径
        """
        self.folder_path = folder_path
        self.visualization = None

    def json2npy(self, prefix: str, file_path: str, output_folder_name: str) -> str:
        """
        将单个JSON文件转换为NPY文件

        Parameters:
        -----------
        prefix : str
            输出文件的前缀
        file_path : str
            JSON文件路径
        output_folder_name : str
            输出文件夹路径

        Returns:
        --------
        str
            生成的NPY文件路径
        """
        with open(file_path, "r") as file:
            json_data = json.load(file)

        array = np.array(json_data)

        basename = os.path.basename(file_path)
        number = "".join(filter(str.isdigit, basename))

        if not number:
            number = 1
        else:
            number = int(number) + 1

        output_name = os.path.join(output_folder_name, f"{prefix}_{number}.npy")
        np.save(output_name, array)
        return output_name

    def process_files(self, prefix: str) -> List[str]:
        """
        处理文件夹中的所有JSON文件

        Parameters:
        -----------
        prefix : str
            输出文件的前缀

        Returns:
        --------
        List[str]
            生成的NPY文件路径列表
        """
        processed_files = []
        files_to_process = [
            f
            for f in os.listdir(self.folder_path)
            if os.path.isfile(os.path.join(self.folder_path, f))
        ]

        for filename in files_to_process:
            if filename.lower().endswith(".json"):
                file_path = os.path.join(self.folder_path, filename)
                print(f"Processing: {file_path}")
                output_path = self.json2npy(prefix, file_path, self.folder_path)
                processed_files.append(output_path)

        return processed_files

    def delete_json_files(self) -> None:
        """删除文件夹中的所有JSON文件"""
        pattern = os.path.join(self.folder_path, "*.json")
        json_files = glob.glob(pattern)

        for file_path in json_files:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")

    def plot_npy_files(self, show: bool = True) -> None:
        """
        绘制文件夹中的所有NPY文件

        Parameters:
        -----------
        show : bool, optional
            是否显示图像，默认为True
        """
        if not os.path.exists(self.folder_path):
            print("Directory does not exist")
            return

        for filename in os.listdir(self.folder_path):
            if filename.endswith(".npy"):
                file_path = os.path.join(self.folder_path, filename)
                data = np.load(file_path)
                plt.figure()
                plt.title(f"Plot of {filename}")
                plt.imshow(data)
                plt.colorbar()
                if show:
                    plt.show()

    def save_plots(self, output_dir: Optional[str] = None) -> None:
        """
        保存所有NPY文件的可视化结果

        Parameters:
        -----------
        output_dir : str, optional
            输出目录，默认为None（使用当前文件夹）
        """
        if output_dir is None:
            output_dir = self.folder_path

        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(self.folder_path):
            if filename.endswith(".npy"):
                file_path = os.path.join(self.folder_path, filename)
                data = np.load(file_path)
                plt.figure()
                plt.title(f"Plot of {filename}")
                plt.imshow(data)
                plt.colorbar()

                output_path = os.path.join(output_dir, f"{filename[:-4]}.png")
                plt.savefig(output_path)
                plt.close()

    def process_and_visualize(self, prefix: str, save_plots: bool = False) -> None:
        """
        处理JSON文件并可视化结果

        Parameters:
        -----------
        prefix : str
            输出文件的前缀
        save_plots : bool, optional
            是否保存图像，默认为False
        """
        self.process_files(prefix)
        if save_plots:
            self.save_plots()
        else:
            self.plot_npy_files()
        self.delete_json_files()
