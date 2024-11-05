"""
File: DicomConverter.py
Project: MPF-Toolbox
Created Date: October 2022
Author: Qiuyi Shen
"""

import os
import shutil
import pydicom
import numpy as np
from collections import defaultdict
import SimpleITK as sitk
from typing import List, Optional, Tuple, Dict
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

    @staticmethod
    def get_series_description(dicom_path: str) -> Optional[str]:
        """
        获取DICOM文件的SeriesDescription

        Parameters:
        -----------
        dicom_path : str
            DICOM文件路径

        Returns:
        --------
        Optional[str]
            SeriesDescription，如果不存在则返回None
        """
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            if hasattr(dicom_data, "SeriesDescription"):
                return dicom_data.SeriesDescription
            return None
        except Exception as e:
            print(f"读取文件 {dicom_path} 时出错: {str(e)}")
            return None

    def standardize_dicom_extensions(self, recursive: bool = False) -> None:
        """
        将文件夹中的DICOM文件统一添加.dcm扩展名

        Parameters:
        -----------
        recursive : bool
            是否递归处理子文件夹
        """
        def process_folder(folder_path: str) -> None:
            files = os.listdir(folder_path)
            for file_name in files:
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    try:
                        pydicom.dcmread(file_path)
                        if not file_name.lower().endswith('.dcm'):
                            new_name = file_name.split('.')[0] + '.dcm'
                            new_path = os.path.join(folder_path, new_name)
                            os.rename(file_path, new_path)
                            print(f"Renamed: {file_name} -> {new_name}")
                    except:
                        continue

        process_folder(self.folder_path)
        if recursive:
            for root, dirs, _ in os.walk(self.folder_path):
                for dir_name in dirs:
                    process_folder(os.path.join(root, dir_name))

    def organize_by_series(self, output_base_path: str) -> Dict[str, List[str]]:
        """
        根据SeriesDescription组织DICOM文件

        Parameters:
        -----------
        output_base_path : str
            输出基础路径

        Returns:
        --------
        Dict[str, List[str]]
            各系列的文件路径字典
        """
        os.makedirs(output_base_path, exist_ok=True)

        series_dict = defaultdict(list)

        total_files = [f for f in os.listdir(self.folder_path)]
        print(f"开始处理 {len(total_files)} 个文件...")

        for i, file_name in enumerate(total_files, 1):
            if i % 100 == 0:
                print(f"已处理 {i}/{len(total_files)} 个文件...")

            file_path = os.path.join(self.folder_path, file_name)
            series_desc = self.get_series_description(file_path)

            if series_desc:
                series_desc = "".join(
                    x for x in series_desc if x.isalnum() or x in [" ", "_", "-"]
                )
                series_dict[series_desc].append((file_name, file_path))

        for series_name, files in series_dict.items():
            print(f"\n处理 {series_name}, 共 {len(files)} 个文件")
            series_path = os.path.join(output_base_path, series_name)

            if len(files) >= 12 and len(files) % 12 == 0:
                slice_count = len(files) // 12
                print(f"分为 {slice_count} 个slice")

                for slice_idx in range(slice_count):
                    slice_path = os.path.join(series_path, f"Slice{slice_idx + 1}")
                    os.makedirs(slice_path, exist_ok=True)

                    for file_idx in range(12):
                        file_index = slice_idx * 12 + file_idx
                        file_name, src_path = files[file_index]
                        dst_path = os.path.join(slice_path, file_name)
                        shutil.copyfile(src_path, dst_path)
            else:
                os.makedirs(series_path, exist_ok=True)
                for file_name, src_path in files:
                    dst_path = os.path.join(series_path, file_name)
                    shutil.copyfile(src_path, dst_path)

        return series_dict

    def process_and_organize(self, output_path: str) -> None:
        """
        完整的处理流程：标准化扩展名并按series组织

        Parameters:
        -----------
        output_path : str
            输出路径
        """
        print("1. 标准化DICOM文件扩展名...")
        self.standardize_dicom_extensions()

        print("\n2. 按Series组织文件...")
        self.organize_by_series(output_path)

        print("\n处理完成！")


if __name__ == "__main__":
    # folder_path = r"D:\Code\Playgroud\sample\slice1"
    # output_folder = r"D:\Data\OrganizedDICOM"

    # converter = DicomConverter(folder_path)

    # converter.standardize_dicom_extensions()
    # converter.organize_by_series(output_folder)
    # converter.process_and_organize(output_folder)

    # 示例1：转换所有文件
    folder_path = r"D:\Code\Playgroud\sample\slice1"
    output_path = os.path.join(folder_path, "custom_output.nii.gz")

    converter = DicomConverter(folder_path)

    converter.convert(output_path=output_path)

    # 示例2：只转换前4个文件
    converter.convert(num_files=4)

    # 示例3：从第2个文件开始转换3个文件
    converter.convert(start_idx=1, num_files=3)

    # 示例4：自定义输出路径
    converter.convert(num_files=4, output_path=output_path)

    # 示例5：预览结果
    converter.preview()
