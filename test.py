"""
File: test.py
Project: MPF-Pipeline
Created Date: October 2022
Author: Qiuyi Shen
"""

import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from core.MPFCal import MPFSL
from tools.HistogramAnalyzer import HistogramAnalyzer
from tools.DicomConverter import DicomConverter
from tools.Json2Npy import Json2Npy
from tools.PhaseMap import PhaseMap
from utils.DicomGenerator import DicomGenerator
from utils.Visualization import Visualization


class PathConfig:
    """管理所有输入输出路径的配置类"""

    def __init__(self, base_path: str):
        """
        初始化路径配置

        Parameters:
        -----------
        base_path : str
            基础路径
        """
        self.base_path = base_path

        self.dicom_path = os.path.join(base_path, "Dicom")
        self.dict_path = os.path.join(base_path, "MPF_N10Tp10Td50_dictionary.mat")
        self.b1_path = os.path.join(base_path, "B1.dcm")
        self.mask_path = os.path.join(base_path, "MASK")

        self.nifti_output = os.path.join(base_path, "Nifti")
        self.phase_output = os.path.join(base_path, "Phase")
        self.rmpfsl_output = os.path.join(base_path, "Rmpfsl")
        self.mpf_output = os.path.join(base_path, "MPF")
        self.results_output = os.path.join(base_path, "Results")

        self._create_directories()

    def _create_directories(self):
        """创建所有必要的输出目录"""
        directories = [
            self.nifti_output,
            self.phase_output,
            self.rmpfsl_output,
            self.mpf_output,
            self.results_output,
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


def test_dicom_to_nifti(config: PathConfig):
    """Test DICOM to NIFTI conversion"""
    print("\n1. Testing DICOM to NIFTI conversion...")

    try:
        converter = DicomConverter(config.dicom_path)
        output_path = os.path.join(config.nifti_output, "MPF-Slice.nii.gz")
        nifti_file = converter.convert(output_path=output_path)
        print(f"Successfully converted folder to NIFTI format")
        return nifti_file
    except Exception as e:
        print(f"DICOM to NIFTI conversion failed: {str(e)}")
        return None


def test_phase_calculation(config: PathConfig):
    """Test phase calculation and difference"""
    print("\n2. Testing phase calculation...")

    try:
        phase_processor = PhaseMap()
        realctr, ictr = phase_processor.check_file(config.dicom_path)

        if len(realctr) == len(ictr):
            phase_maps, phase_diffs = phase_processor.calculate_phase_maps(
                realctr, ictr
            )

            phase_processor.plot_phase_maps(phase_maps[:4])
            phase_processor.plot_phase_differences(phase_diffs)

            phase_processor.save_phase_maps(
                complex_arrays=phase_maps,
                save_path=config.phase_output,
                prefix="phase_map",
            )

            phase_processor.save_phase_diff_results(
                phase_maps=phase_diffs,
                save_path=config.phase_output,
                prefix="phase_diff",
            )

            plt.show()
            return realctr, ictr, phase_maps
    except Exception as e:
        print(f"Phase calculation failed: {str(e)}")
        return None, None, None


def test_mpf_calculation(config: PathConfig, realctr, ictr):
    """Test MPF calculation"""
    print("\n3. Testing MPF calculation...")

    try:
        mpfsl = MPFSL(
            realctr[0],
            ictr[0],
            realctr[2],
            ictr[2],
            realctr[1],
            ictr[1],
            realctr[3],
            ictr[3],
            config.dict_path,
            config.b1_path,
        )

        mpf = mpfsl.cal_mpf()
        rmpfsl = mpfsl.rmpfsl
        rmpfsl = np.clip(rmpfsl, np.min(rmpfsl), 4, out=None)

        dicom_gen = DicomGenerator(realctr[0])

        rmpfsl_path = os.path.join(config.rmpfsl_output, "RMPFSL.dcm")
        dicom_gen.generate_dicom(rmpfsl * 100, rmpfsl_path)

        mpf_path = os.path.join(config.mpf_output, "MPF.dcm")
        dicom_gen.generate_dicom(mpf * 1000, mpf_path)

        viz = Visualization()
        viz.plot_single_rmpfsl(
            rmpfsl_image=rmpfsl, title="RMPFSL Result", show_colorbar=True
        )
        plt.show()

        viz.plot_single_mpf(mpf_image=mpf, title="MPF Result", show_colorbar=True)
        plt.show()

        return rmpfsl, mpf
    except Exception as e:
        print(f"MPF calculation failed: {str(e)}")
        return None, None


def test_mask_analysis(config: PathConfig, rmpfsl_data, mpf_data):
    """Test mask analysis"""
    print("\n4. Testing Mask analysis...")

    try:
        json_processor = Json2Npy(config.mask_path)
        processed_files = json_processor.process_files(prefix="mask")
        mask_data = np.load(processed_files[0])

        viz = Visualization()

        viz.plot_with_mask(
            image=rmpfsl_data,
            mask=mask_data,
            title="RMPFSL",
            is_mpf=False,
        )
        plt.savefig(os.path.join(config.results_output, "rmpfsl_with_mask.png"))
        plt.show()

        viz.plot_with_mask(
            image=mpf_data,
            mask=mask_data,
            title="MPF",
            is_mpf=True,
        )
        plt.savefig(os.path.join(config.results_output, "mpf_with_mask.png"))
        plt.show()

        rmpfsl_analyzer = HistogramAnalyzer(data_type="RMPFSL")
        rmpfsl_analyzer.plot_combined_analysis(
            rmpfsl_data,
            mask_data,
            slice_num=1,
            save_path=os.path.join(config.results_output, "rmpfsl_histogram.png"),
        )
        plt.show()

        mpf_analyzer = HistogramAnalyzer(data_type="MPF")
        mpf_analyzer.plot_combined_analysis(
            mpf_data * 100,
            mask_data,
            slice_num=1,
            save_path=os.path.join(config.results_output, "mpf_histogram.png"),
        )
        plt.show()

        return mask_data
    except Exception as e:
        print(f"Mask analysis failed: {str(e)}")
        return None


def run_full_pipeline(base_path: str):
    """Run complete processing pipeline"""
    print("Starting full processing pipeline...")

    config = PathConfig(base_path)

    # 1. DICOM to NIFTI conversion
    nifti_files = test_dicom_to_nifti(config)
    if nifti_files is None:
        return

    # 2. Phase calculation
    realctr, ictr, phase_maps = test_phase_calculation(config)
    if phase_maps is None:
        return

    # 3. MPF calculation
    rmpfsl, mpf = test_mpf_calculation(config, realctr, ictr)
    if mpf is None:
        return

    # 4. Mask analysis
    mask = test_mask_analysis(config, rmpfsl, mpf)
    if mask is None:
        return

    print("\nProcessing pipeline completed")


if __name__ == "__main__":
    base_path = r"MPF_Pipeline\Sample"
    run_full_pipeline(base_path)
