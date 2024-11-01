"""
File: DicomGenerator.py
Project: MPF-Toolbox
Created Date: October 2022
Author: Qiuyi Shen
"""

import pydicom
import numpy as np
import time

class DicomGenerator:
    def __init__(self, template_dicom_path):
        """
        初始化DicomGenerator
        Args:
            template_dicom_path: 模板DICOM文件路径
        """
        self.ds = pydicom.dcmread(template_dicom_path)

    def generate_checksum(self, uid):
        """生成UID校验和"""
        uid_parts = uid.split(".")
        sum_digits = [sum(int(digit) for digit in part) for part in uid_parts]
        count_digits = [len(part) if len(part) % 9 != 0 else 9 for part in uid_parts]
        checksum_digits = [
            (int(sum_digit) % count_digit)
            for sum_digit, count_digit in zip(sum_digits, count_digits)
        ]
        checksum = "".join(str(digit) for digit in checksum_digits)
        return checksum[:10] if len(checksum) > 10 else checksum

    def get_processing_type(self, image_type):
        """获取处理类型"""
        processing_dict = {
            "QUANT": "1",
            "MASK": "2",
            "NORMAL": "3",
            "OTHER": "4",
        }

        if len(image_type) >= 3:
            part_three = image_type[2]
            processing_type = "1" if part_three == "DEIDENT" else "0"
            processing_type += processing_dict.get(image_type[3], "")
            return processing_type
        return ""

    def generate_uid(self, uid, fileNum, index, image_type):
        """生成新的UID"""
        root_unique = "1.2.826.0.1.3680043.10.1338"
        uid_version = ".001"
        check_code = "." + self.generate_checksum(uid)
        timestamp = "." + str(int(round(time.time() * 100)))
        inputFileNum = "." + str(fileNum).zfill(2)
        index = "." + str(index).zfill(2)
        type = "." + self.get_processing_type(image_type)

        return (
            root_unique
            + uid_version
            + check_code
            + timestamp
            + inputFileNum
            + index
            + type
        )

    def update_pixel_data(self, image_array):
        """更新像素数据和相关属性"""
        image_result = (image_array).astype(np.uint16)

        self.ds.PixelData = image_result.tobytes()
        self.ds.Rows, self.ds.Columns = image_result.shape

        min_val = np.min(image_result)
        max_val = np.max(image_result)
        self.ds.WindowWidth = max_val - min_val
        self.ds.WindowCenter = (max_val + min_val) / 2

        self.ds.SamplesPerPixel = 1
        self.ds.BitsAllocated = 16
        self.ds.BitsStored = 16
        self.ds.HighBit = 15
        self.ds.RescaleIntercept = 0
        self.ds.RescaleSlope = 1

    def update_uids(self, fileNum=1, index=1):
        """更新DICOM的UIDs"""
        for tag in [(0x08, 0x18), (0x20, 0x0E), (0x20, 0x0D)]:
            original_uid = self.ds[tag].value
            image_type = self.ds[0x08, 0x08].value
            new_uid = self.generate_uid(original_uid, fileNum, index, image_type)
            self.ds[tag].value = new_uid

    def generate_dicom(self, image_array, output_path):
        """生成完整的DICOM文件"""
        self.update_pixel_data(image_array)
        self.update_uids()
        self.ds.save_as(output_path)
