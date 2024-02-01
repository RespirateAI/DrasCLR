# Read a nifti image and print size

import nibabel as nib
import numpy as np
import os
import sys


def read_nifti(file):
    img = nib.load(file)
    return img.get_fdata()


filepath = "./preprocess/misc/COPD_Atlas_INSP_BSpline_Iso1mm.nii.gz"
arr = read_nifti(filepath)
print(arr.shape)
