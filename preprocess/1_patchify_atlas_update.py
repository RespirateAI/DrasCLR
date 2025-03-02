"""
Pre-process the data to extract patches
Input: A csv file containing path to input files

"""
import argparse
import os
import sys
import math
import numpy as np
import SimpleITK as sitk
import pandas as pd

lowerThreshold = -1024
upperThreshold = 240


def Image2Patch(inputImg, step_size, patch_size, acceptRate):
    """This function converts image to patches.
    Here is the input of the function:
      inputImg : input image. This should be simpleITK object
      labelMaskImg : label image containing mask of the lobes (values greater than 0)
      patchSize : size of the patch. It should be array of three scalar
      acceptRate : If portion of the patch inside of the mask exceeds value, it would be accepted
    Here is the output of the function:
      patchImgData : It is a list containing the patches of the image
      patchLblData : Is is a list containing the patches of the label image

    """
    patch_vol = patch_size[0] * patch_size[1] * patch_size[2]
    patch_img_data = []
    patch_loc = []
    img_size = inputImg.GetSize()

    # Open a csv file
    csv = open("patch_loc.csv", "w")

    for idx_x, x in enumerate(range(0, img_size[0] - 0 - patch_size[0], step_size[0])):
        for idx_y, y in enumerate(
            range(0, img_size[1] - 0 - patch_size[1], step_size[1])
        ):
            for idx_z, z in enumerate(
                range(0, img_size[2] - 0 - patch_size[2], step_size[2])
            ):
                # patchLblImg = sitk.RegionOfInterest(labelMaskImg, size=patch_size, index=[x,y,z])
                # npPatchLblImg = sitk.GetArrayFromImage(patchLblImg)
                patchImg = sitk.RegionOfInterest(
                    inputImg, size=patch_size, index=[x, y, z]
                )
                npLargePatchImg = sitk.GetArrayFromImage(patchImg)
                if (npLargePatchImg > 0).sum() > 0:
                    print(x, y, z)
                    csv.write(str(x) + "," + str(y) + "," + str(z) + "\n")
                    patch_img_data.append(npLargePatchImg.copy())
                    patch_loc.append([x, y, z])

    patch_img_data = np.asarray(patch_img_data)
    patch_loc = np.asarray(patch_loc)
    return patch_img_data, patch_loc


def extract_patch(isoRawImage_file):
    # Read the input isotropic image volume
    isoRawImage = sitk.ReadImage(isoRawImage_file)
    npIsoRawImage = sitk.GetArrayFromImage(isoRawImage)
    # print(npIsoRawImage.shape)

    # Thresholding the isotropic raw image
    npIsoRawImage[npIsoRawImage > upperThreshold] = upperThreshold
    npIsoRawImage[npIsoRawImage < lowerThreshold] = lowerThreshold

    thresholdIsoRawImage = sitk.GetImageFromArray(npIsoRawImage)
    thresholdIsoRawImage.SetOrigin(isoRawImage.GetOrigin())
    thresholdIsoRawImage.SetSpacing(isoRawImage.GetSpacing())
    thresholdIsoRawImage.SetDirection(isoRawImage.GetDirection())

    # Read the input isotropic label image
    # isoLabelImage = sitk.ReadImage(isoLabelImage_file)
    # npIsoLabelImage = sitk.GetArrayFromImage(isoLabelImage)

    # Generate binary label map
    # binaryLabelImage = sitk.GetArrayFromImage(isoLabelImage)
    # binaryLabelImage[binaryLabelImage > 0] = 1
    # binaryLabelImage = sitk.GetImageFromArray(binaryLabelImage)
    # binaryLabelImage.SetOrigin(isoLabelImage.GetOrigin())
    # binaryLabelImage.SetSpacing(isoLabelImage.GetSpacing())
    # binaryLabelImage.SetDirection(isoLabelImage.GetDirection())
    # assert thresholdIsoRawImage.GetSize() == binaryLabelImage.GetSize()

    # Extract Patches
    # Generate Patches of the masked Image

    print(thresholdIsoRawImage.GetSize())
    threshold = 0.01
    while True:
        patchImgData, patch_loc = Image2Patch(
            thresholdIsoRawImage, [step_size] * 3, [patch_size] * 3, threshold
        )
        return patchImgData, patch_loc
        # if patchImgData.shape[0] < 1000:
        #     return patchImgData, patch_loc
        # if threshold < 1:
        #     threshold += 0.1
        #     print("Too many patches, trying again with threshold:", threshold)
        # else:
        #     return np.empty([0]), np.empty([0])


def prep_adjacency_matrix(patch_loc):
    adj = []
    for i in range(patch_loc.shape[0]):
        adj_row = np.zeros((patch_loc.shape[0],))
        dist = np.abs(patch_loc - patch_loc[i])
        max_side_dist = dist.max(1)
        dist = dist[max_side_dist < patch_size, :]
        volume = np.abs(dist - patch_size)
        volume = volume[:, 0] * volume[:, 1] * volume[:, 2]
        # print(volume.shape)
        # print(adj_row[max_side_dist<patch_size].shape)
        adj_row[max_side_dist < patch_size] = volume / (patch_size**3)
        adj.append(adj_row.transpose())
    adj = np.asarray(adj)
    # adj = (adj / np.sum(adj, 0)).transpose()
    return adj


def run():
    isotropicFileName = atlas_image
    # partialLungLabelMapFileName = atlas_roi_mask

    output_basename = isotropicFileName.split("/")[-1].split(".")[0]
    print(output_basename)
    patchFileName = os.path.join(output_dir, "patch", output_basename + "_patch.npy")

    # if os.path.exists(patchFileName):
    #     print("Patch file already exists, skipping...")
    #     return

    patchImgData, patch_loc = extract_patch(isotropicFileName)

    # save the anatomical location for each landmark in atlas space
    np.save(patchFileName, patchImgData)
    np.save(
        os.path.join(output_dir, "atlas_patch_loc_" + output_basename + ".npy"),
        patch_loc,
    )

    print("Finished. Total number of patches:", patchImgData.shape[0])


def main(argv):
    global output_dir
    global patch_size
    global step_size
    global atlas_image, atlas_roi_mask
    global image_dir

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Data Preprocessing")
    # parser.add_argument("--image_dir", type=str)
    parser.add_argument("--atlas_image", type=str)
    # parser.add_argument('--atlas_roi_mask', type=str)
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./patch_data_chaimeleon_new",
        help="Directory where intermediate and final files are saved.",
    )
    parser.add_argument(
        "-s", "--patch_size", type=int, default=32, help="The size of the 3D patch."
    )
    parser.add_argument(
        "-l",
        "--step_size",
        type=int,
        default=26,
        help="The overlap between consecutive patches.",
    )

    args = parser.parse_args()
    output_dir = args.output_dir
    patch_size = args.patch_size
    step_size = args.step_size
    atlas_image = args.atlas_image
    # atlas_roi_mask = args.atlas_roi_mask
    # img_dir = args.image_dir

    # file_list = os.listdir(img_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        # Create a folder called Patch in the output directory
        os.makedirs(os.path.join(output_dir, "patch"))

    # for file in file_list:
    #     atlas_image = os.path.join(img_dir, file)
    #     run()

    run()


if __name__ == "__main__":
    main(sys.argv[1:])
