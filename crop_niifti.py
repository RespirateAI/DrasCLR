import nibabel as nib
import numpy as np


def crop_nifti(input_path, output_path, crop_value_xy, crop_value_z):
    # Load NIfTI image
    img = nib.load(input_path)
    data = img.get_fdata()

    # Print original size
    print(f"Original size: {data.shape}")

    # Calculate the cropping range
    crop_range_xy = slice(crop_value_xy, data.shape[0] - crop_value_xy)
    crop_range_z = slice(crop_value_z, data.shape[2] - crop_value_z)

    # Crop along each dimension
    cropped_data = data[
        crop_range_xy, crop_range_xy, crop_range_z
    ]  # Add more dimensions as needed

    # Create a new NIfTI image with cropped data
    cropped_img = nib.Nifti1Image(cropped_data, img.affine)

    # Print cropped size
    print(f"Cropped size: {cropped_data.shape}")

    # Save the cropped NIfTI image
    nib.save(cropped_img, output_path)


# Example usage
input_nifti_path = "/media/ravindu/SSD-PLU3/Lung_CT_Dataset/Lung-PET-CT-Dx-NBIA-Manifest-122220/DrasNew/DrasCLR/test/Lung_Dx-A0200&10-02-2009-PET03WholebodyFirstHead_Adult-17944&10.000000-Thorax_1.0_B70f-28020.nii.gz"
output_nifti_path = "/media/ravindu/SSD-PLU3/Lung_CT_Dataset/Lung-PET-CT-Dx-NBIA-Manifest-122220/DrasNew/DrasCLR/test/Lung_Dx-A0200&10-02-2009-PET03WholebodyFirstHead_Adult-17944&10.000000-Thorax_1.0_B70f-28020_cropped.nii.gz"
crop_value_xy = 78  # You can adjust this value based on your requirement
crop_value_z = 0

crop_nifti(input_nifti_path, output_nifti_path, crop_value_xy, crop_value_z)
