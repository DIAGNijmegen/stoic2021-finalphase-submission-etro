import numpy as np
import SimpleITK as sitk
from algorithm.preprocessing.definitions import *


def resample(image_sitk, out_spacing=SPACING, is_seg=False):
    # Compute the volume output size
    original_spacing = image_sitk.GetSpacing()
    original_size = image_sitk.GetSize()
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    # Create the resampling operator
    resample = sitk.ResampleImageFilter()
    if is_seg:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(image_sitk.GetDirection())
    resample.SetOutputOrigin(image_sitk.GetOrigin())
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    # resample.SetTransform(sitk.Transform())
    # resample.SetDefaultPixelValue(image_sitk.GetPixelIDValue())

    return resample.Execute(image_sitk)


def crop_around_mask(img, mask, crop_margin=CROP_MARGIN, return_coordinates=False):
    """
    Crop img around mask with a margin.
    All axis dimension of the cropped image will not be smaller
    than the one of the patch size.
    """
    assert img.shape == mask.shape, "image and mask do not have the same shape"

    # Get the number of foreground pixels
    num_fg = np.sum(mask)
    x_dim, y_dim, z_dim = tuple(img.shape)

    assert num_fg > 0, "The segmentation contains only background."
    x_fg, y_fg, z_fg = np.where(mask >= 1)

    # Get the extremal acceptable coordinates for cropping
    x_min = max(int(np.min(x_fg)) - crop_margin, 0)
    x_max = min(int(np.max(x_fg)) + crop_margin, x_dim)
    y_min = max(int(np.min(y_fg)) - crop_margin, 0)
    y_max = min(int(np.max(y_fg)) + crop_margin, y_dim)
    z_min = max(int(np.min(z_fg)) - crop_margin, 0)
    z_max = min(int(np.max(z_fg)) + crop_margin, z_dim)

    # Crop the image
    crop_img = img[x_min:x_max, y_min:y_max, z_min:z_max]

    if return_coordinates:
        coords = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
        return crop_img, coords
    else:
        return crop_img
