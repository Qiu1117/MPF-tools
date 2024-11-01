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


def test_dicom_to_nifti():
    """Test DICOM to NIFTI conversion"""
    print("\n1. Testing DICOM to NIFTI conversion...")

    base_path = r"MPF_Pipeline\Sample"
    nifti_output = os.path.join(base_path, "Nifti")

    os.makedirs(nifti_output, exist_ok=True)

    try:

        slice_folder = os.path.join(base_path, "Dicom")

        converter = DicomConverter(slice_folder)

        output_path = os.path.join(nifti_output, f"MPF-Slice.nii.gz")

        nifti_file = converter.convert(output_path=output_path)

        print(f"Successfully converted  folder to NIFTI format")
        return nifti_file

    except Exception as e:
        print(f"DICOM to NIFTI conversion failed: {str(e)}")
        return None


def test_phase_calculation():
    """Test phase calculation and difference"""
    print("\n2. Testing phase calculation...")

    base_path = r"MPF_Pipeline\Sample"
    slice_folder_path = os.path.join(base_path, "Dicom")
    phase_output = os.path.join(base_path, "Phase")
    os.makedirs(phase_output, exist_ok=True)
    
    try:
        phase_processor = PhaseMap()
        
        realctr, ictr = phase_processor.check_file(slice_folder_path)
        if len(realctr) == len(ictr):
            phase_maps, phase_diffs = phase_processor.calculate_phase_maps(realctr, ictr)
            
            phase_processor.plot_phase_maps(phase_maps[:4])
            phase_processor.plot_phase_differences(phase_diffs)
            
            phase_processor.save_phase_maps(
                complex_arrays=phase_maps, save_path=phase_output, prefix="phase_map"
            )

            phase_processor.save_phase_diff_results(
                phase_maps=phase_diffs, save_path=phase_output, prefix="phase_diff"
            )
        
            plt.show()

        return realctr, ictr, phase_maps

    except Exception as e:
        print(f"Phase calculation failed: {str(e)}")
        return None, None, None


def test_mpf_calculation(realctr, ictr):
    """Test MPF calculation"""
    print("\n3. Testing MPF calculation...")

    base_path = r"MPF_Pipeline\Sample"
    dict_path = os.path.join(base_path, "MPF_N10Tp10Td50_dictionary.mat")
    B1_path = os.path.join(base_path, "B1.dcm")

    # 创建输出目录
    rmpfsl_output = os.path.join(base_path, "Rmpfsl")
    mpf_output = os.path.join(base_path, "MPF")
    os.makedirs(rmpfsl_output, exist_ok=True)
    os.makedirs(mpf_output, exist_ok=True)

    try:
        # 计算RMPFSL和MPF
        mpfsl = MPFSL(
            realctr[0],
            ictr[0],  
            realctr[2],
            ictr[2],  
            realctr[1],
            ictr[1],  
            realctr[3],
            ictr[3],  
            dict_path,
            B1_path,
        )

        mpf = mpfsl.cal_mpf()

        rmpfsl = mpfsl.rmpfsl
        rmpfsl = np.clip(rmpfsl, np.min(rmpfsl), 4, out=None)

        dicom_gen = DicomGenerator(realctr[0])

        rmpfsl_path = os.path.join(rmpfsl_output, "RMPFSL.dcm")
        dicom_gen.generate_dicom(rmpfsl*100, rmpfsl_path)
        print(f"RMPFSL saved to: {rmpfsl_path}")

        mpf_path = os.path.join(mpf_output, "MPF.dcm")
        dicom_gen.generate_dicom(mpf*1000, mpf_path)

        print(f"MPF saved to: {mpf_path}")

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


def test_mask_analysis(rmpfsl_data, mpf_data):
    """Test mask analysis"""
    print("\n4. Testing Mask analysis...")

    base_path = r"MPF_Pipeline\Sample"
    mask_path = os.path.join(base_path, "MASK")

    try:
        json_processor = Json2Npy(mask_path)
        processed_files = json_processor.process_files(prefix="mask")

        mask_data = np.load(processed_files[0])

        viz = Visualization()

        viz.plot_with_mask(
            image=rmpfsl_data,
            mask=mask_data,
            title="RMPFSL",
            is_mpf=False,
        )
        plt.show()

        viz.plot_with_mask(
            image=mpf_data,
            mask=mask_data,
            title="MPF",
            is_mpf=True,
        )
        plt.show()

        # viz.save_figure(output_path, "mpf_with_mask.png")
        # viz.save_figure(output_path, "rmpfsl_with_mask.png")

        rmpfsl_analyzer = HistogramAnalyzer(data_type="RMPFSL")
        rmpfsl_analyzer.plot_combined_analysis(rmpfsl_data, mask_data, slice_num=1)
        plt.show()

        mpf_analyzer = HistogramAnalyzer(data_type="MPF")
        mpf_analyzer.plot_combined_analysis(mpf_data*100, mask_data, slice_num=1)
        plt.show()

        return mask_data
    except Exception as e:
        print(f"Mask analysis failed: {str(e)}")
        return None


def run_full_pipeline():
    """Run complete processing pipeline"""
    print("Starting full processing pipeline...")

    # 1. DICOM to NIFTI conversion
    nifti_files = test_dicom_to_nifti()
    if nifti_files is None:
        return

    # 2. Phase calculation - 首先计算相位
    realctr, ictr, phase_maps = test_phase_calculation()
    if phase_maps is None:
        return

    # 3. MPF calculation - 使用phase计算得到的realctr和ictr
    rmpfsl, mpf = test_mpf_calculation(realctr, ictr)
    if mpf is None:
        return

    # 4. Mask analysis
    mask = test_mask_analysis(rmpfsl, mpf)
    if mask is None:
        return

    print("\nProcessing pipeline completed")


if __name__ == "__main__":
    run_full_pipeline()
