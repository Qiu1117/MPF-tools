"""
File: MPFCal.py
Project: MPF-Pipeline
Created Date: October 2022
Author: Qiuyi Shen
"""

import os
import numpy as np
from scipy import io
import pydicom
import cv2
from typing import List, Dict, Union, Optional
import matplotlib.pyplot as plt
from skimage import transform


class MPFSL:
    def __init__(
        self,
        dyn_real1: str,
        dyn_img1: str,
        dyn_real2: str,
        dyn_img2: str,
        dyn_real3: str,
        dyn_img3: str,
        dyn_real4: str,
        dyn_img4: str,
        dic: str,
        B1_map: str,
        tsl: float = 0.1,
    ):
        self.dyn_cpx1 = self.to_complex(dyn_real1, dyn_img1)
        self.dyn_cpx2 = self.to_complex(dyn_real2, dyn_img2)
        self.dyn_cpx3 = self.to_complex(dyn_real3, dyn_img3)
        self.dyn_cpx4 = self.to_complex(dyn_real4, dyn_img4)
        self.tsl = tsl
        assert (
            self.dyn_cpx1.shape
            == self.dyn_cpx2.shape
            == self.dyn_cpx3.shape
            == self.dyn_cpx4.shape
        )

        self.dict_path = dic
        self.load_dict()
        self.rmpfsl = self.cal_rmpfsl(
            self.dyn_cpx1, self.dyn_cpx2, self.dyn_cpx3, self.dyn_cpx4
        )
        self.B1_map = self._load_dicom(B1_map)
        self.B1_map = transform.resize(self.B1_map, self.dyn_cpx1.shape)

    def load_dict(self):
        mat_dic = io.loadmat(self.dict_path)
        self.var_MPF = mat_dic["var_MPF"][0]
        if "jtmt" in mat_dic:
            self.jtmt = mat_dic["jtmt"]
            self.inhom_b1 = mat_dic["inhom_b1"][0]
        elif "all_jtmt" in mat_dic:
            self.jtmt = mat_dic["all_jtmt"]
            self.inhom_b1 = mat_dic["all_inhom_b1"][0]

    def cal_rmpfsl(self, dyn1, dyn2, dyn3, dyn4):
        nonzero_divide = np.divide(
            np.abs(dyn1 - dyn2),
            np.abs(dyn3 - dyn4),
            out=np.zeros(dyn1.shape),
            where=(dyn3 - dyn4 != 0),
        )
        nonzero_log = np.log(
            nonzero_divide, out=np.zeros_like(nonzero_divide), where=nonzero_divide != 0
        )
        cal_rmpfsl = np.abs(nonzero_log / -self.tsl)
        return cal_rmpfsl

    def to_complex(self, real_path, imginary_path):
        real_dcm_data = self._load_dicom(real_path)
        imginary_dcm_data = self._load_dicom(imginary_path)
        return real_dcm_data + 1j * imginary_dcm_data

    @staticmethod
    def _load_dicom(path: str) -> np.ndarray:
        """加载DICOM文件"""
        dcm = pydicom.dcmread(path)
        slope = getattr(dcm, "RescaleSlope", 1)
        intercept = getattr(dcm, "RescaleIntercept", 0)
        data = dcm.pixel_array * slope + intercept

        # 应用高斯模糊
        kernel = cv2.getGaussianKernel(5, 1.0)
        kernel = kernel * kernel.T
        return cv2.filter2D(data, -1, kernel)

    def cal_mpf(self):
        h, w = self.rmpfsl.shape
        mpf = np.zeros((w, h))
        for y in range(h):
            for x in range(w):
                b1_check = np.abs(self.B1_map[y, x] - self.inhom_b1)
                num_b1 = np.argmin(b1_check)
                if self.rmpfsl[y, x] > 0 and self.rmpfsl[y, x] < 100:
                    y_check = np.abs(self.rmpfsl[y, x] - self.jtmt[num_b1, :])
                    num_fc = np.argmin(y_check)
                    mpf[y, x] = self.var_MPF[num_fc]
        return mpf


if __name__ == "__main__":
    dict_path = r"D:\Code\MPF_Data_process\MPF_N10Tp10Td50_dictionary.mat"
    B1_path = r"D:\Code\Playgroud\sample\IMG-0002-00001.dcm"
    tsl = 0.1

    base_path = r"D:\Code\Playgroud\sample\slice1"

    dyn_real1 = os.path.join(base_path, "IM_0005.dcm")
    dyn_real3 = os.path.join(base_path, "IM_0006.dcm")
    dyn_real2 = os.path.join(base_path, "IM_0007.dcm")
    dyn_real4 = os.path.join(base_path, "IM_0008.dcm")

    dyn_img1 = os.path.join(base_path, "IM_0009.dcm")
    dyn_img3 = os.path.join(base_path, "IM_0010.dcm")
    dyn_img2 = os.path.join(base_path, "IM_0011.dcm")
    dyn_img4 = os.path.join(base_path, "IM_0012.dcm")

    mpfsl = MPFSL(
        dyn_real1,
        dyn_img1,
        dyn_real2,
        dyn_img2,
        dyn_real3,
        dyn_img3,
        dyn_real4,
        dyn_img4,
        dict_path,
        B1_path,
        tsl
    )
    mpf = mpfsl.cal_mpf()

    plt.imshow(mpf, cmap="jet")
    plt.colorbar()
    plt.show()

    rmpf = mpfsl.rmpfsl
    rmpf = np.clip(rmpf, np.min(rmpf), 4, out=None)
    plt.imshow(rmpf, cmap="jet")
    plt.colorbar()
    plt.show()
