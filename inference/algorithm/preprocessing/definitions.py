import SimpleITK as sitk

SPACING = [1.] * 3  # isotropic in mm
MIN_HU = -1000  # air
MAX_HU = 100
CROP_MARGIN = 5
MIN_NUM_VOXEL_PER_COMP = 100000

# LABELS
LABELS_BINARY = {
    'lesion': 1,
    'lung': 2,  # only non lesion
    'background': 0,
}
LABELS_MULTI = {
    'ggo': 1,
    'consolidation': 2,
    'lung': 3,
    'background': 0,
    'any_lesion': 4,
}


def convert_to_sitk(img_np, ref_img_sitk):
    img_sitk = sitk.GetImageFromArray(img_np)
    img_sitk.SetOrigin(ref_img_sitk.GetOrigin())
    img_sitk.SetSpacing(ref_img_sitk.GetSpacing())
    img_sitk.SetDirection(ref_img_sitk.GetDirection())
    return img_sitk