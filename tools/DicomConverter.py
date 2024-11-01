"""
File: DicomConverter.py
Project: MPF-Toolbox
Created Date: October 2022
Author: Qiuyi Shen
"""

import os
import numpy as np
import SimpleITK as sitk
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt


class DicomConverter:
    """用于将DICOM序列转换为NIfTI格式的工具类"""

    def __init__(self, folder_path: str):
        """
        初始化DicomConverter类

        Parameters:
        -----------
        folder_path : str
            DICOM文件所在文件夹路径
        """
        self.folder_path = folder_path
        self.output_filename = "magnitude.nii.gz"

    def count_dicom_files(self) -> int:
        """
        计算文件夹中DICOM文件的数量

        Returns:
        --------
        int
            DICOM文件数量
        """
        return len(
            [f for f in os.listdir(self.folder_path) if f.lower().endswith(".dcm")]
        )

    def get_dicom_files(
        self, start_idx: int = 0, num_files: Optional[int] = None
    ) -> List[str]:
        """
        获取DICOM文件路径列表

        Parameters:
        -----------
        start_idx : int
            起始文件索引
        num_files : int, optional
            需要读取的文件数量，如果为None则读取所有文件

        Returns:
        --------
        List[str]
            DICOM文件路径列表
        """
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(".dcm")]
        files.sort()  # 确保文件顺序一致

        if num_files is None:
            num_files = len(files)

        selected_files = files[start_idx : start_idx + num_files]
        return [os.path.join(self.folder_path, f) for f in selected_files]

    def read_dicom_series(self, file_paths: List[str]) -> Tuple[np.ndarray, sitk.Image]:
        """
        读取DICOM序列

        Parameters:
        -----------
        file_paths : List[str]
            DICOM文件路径列表

        Returns:
        --------
        Tuple[np.ndarray, sitk.Image]
            numpy数组和SimpleITK图像对象
        """
        images = []
        for file_path in file_paths:
            img = sitk.ReadImage(file_path)
            img_array = sitk.GetArrayFromImage(img)
            images.append(img_array)

        combined_array = np.concatenate(images, axis=0)
        combined_image = sitk.GetImageFromArray(combined_array)

        # 复制原始DICOM的元数据
        for key in img.GetMetaDataKeys():
            combined_image.SetMetaData(key, img.GetMetaData(key))

        return combined_array, combined_image

    def save_nifti(self, image: sitk.Image, output_path: Optional[str] = None) -> str:
        """
        保存为NIfTI格式

        Parameters:
        -----------
        image : sitk.Image
            SimpleITK图像对象
        output_path : str, optional
            输出文件路径，如果为None则使用默认路径

        Returns:
        --------
        str
            保存的文件路径
        """
        if output_path is None:
            output_path = os.path.join(self.folder_path, self.output_filename)

        sitk.WriteImage(image, output_path)
        return output_path

    def convert(
        self,
        start_idx: int = 0,
        num_files: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        执行转换过程

        Parameters:
        -----------
        start_idx : int
            起始文件索引
        num_files : int, optional
            需要读取的文件数量，如果为None则读取所有文件
        output_path : str, optional
            输出文件路径

        Returns:
        --------
        str
            保存的文件路径
        """
        # 获取文件路径
        total_files = self.count_dicom_files()
        if num_files is None:
            num_files = total_files

        if start_idx >= total_files:
            raise ValueError(
                f"Start index {start_idx} exceeds total number of files {total_files}"
            )

        if start_idx + num_files > total_files:
            print(
                f"Warning: Requested {num_files} files but only {total_files - start_idx} available"
            )
            num_files = total_files - start_idx

        file_paths = self.get_dicom_files(start_idx, num_files)
        if not file_paths:
            raise ValueError(f"No DICOM files found in {self.folder_path}")

        print(f"Converting {len(file_paths)} files starting from index {start_idx}")

        # 读取和合并图像
        array, image = self.read_dicom_series(file_paths)
        print(f"Converted array shape: {array.shape}, dtype: {array.dtype}")

        # 保存NIfTI文件
        saved_path = self.save_nifti(image, output_path)
        print(f"Saved NIfTI file to: {saved_path}")

        return saved_path

    def preview(self, slice_idx: Optional[int] = None) -> None:
        """
        预览转换结果

        Parameters:
        -----------
        slice_idx : int, optional
            要显示的切片索引，如果为None则显示中间切片
        """
        nifti_path = os.path.join(self.folder_path, self.output_filename)
        if not os.path.exists(nifti_path):
            print("No NIfTI file found. Converting first...")
            nifti_path = self.convert()

        img = sitk.ReadImage(nifti_path)
        array = sitk.GetArrayFromImage(img)

        if slice_idx is None:
            slice_idx = array.shape[0] // 2

        plt.figure(figsize=(8, 8))
        plt.imshow(array[slice_idx], cmap="gray")
        plt.colorbar()
        plt.title(f"Slice {slice_idx}")
        plt.axis("on")
        plt.show()


if __name__ == "__main__":
    # 使用示例
    folder_path = r"D:\Code\Playgroud\sample\slice1"
    output_path = os.path.join(folder_path, "custom_output.nii.gz")

    converter = DicomConverter(folder_path)

    # 示例1：转换所有文件
    converter.convert(output_path=output_path)

    # 示例2：只转换前4个文件
    converter.convert(num_files=4)

    # 示例3：从第2个文件开始转换3个文件
    converter.convert(start_idx=1, num_files=3)

    # 示例4：自定义输出路径
    converter.convert(num_files=4, output_path=output_path)

    # 示例5：预览结果
    converter.preview()
