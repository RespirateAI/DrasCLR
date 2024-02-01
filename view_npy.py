import numpy as np


def read_npy(file):
    return np.load(file)


if __name__ == "__main__":
    # filename = "./preprocess/patch_data_32_6_reg/atlas_patch_loc_case_0094.npy"
    # filename = "./preprocess/misc/atlas_patch_loc.npy"
    # filename = "./preprocess/patch_data_32_6_reg/atlas_patch_loc_COPD_Atlas_INSP_BSpline_Iso1mm.npy"
    # filename = "./preprocess/patch_data_chaimeleon/patch/case_0109_patch.npy"
    filename = "DrasCLR_pretrained_ckpt/patch_rep/pred_arr_patch_full.npy"
    arr = read_npy(filename)
    print(arr.shape)
    print(arr[0][350])
