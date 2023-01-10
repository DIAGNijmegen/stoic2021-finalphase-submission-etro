import numpy as np
import SimpleITK as sitk
from typing import Iterable
from scipy.ndimage.measurements import label
from algorithm.preprocessing.segmentation import compute_lung_seg, add_lung_label
from algorithm.preprocessing.spatial import crop_around_mask, resample
from algorithm.preprocessing.intensity import normalize_intensity
from algorithm.preprocessing.definitions import *
from algorithm.preprocessing.lungmask import mask


def compute_lung_seg(img_sitk):
    lung_mask_np = mask.apply(img_sitk)
    # binarize the mask and keep only the two largest connected components
    lung_mask_np = postprocess_lung_seg(lung_mask_np)
    return lung_mask_np


def postprocess_lung_seg(lung_seg_np):
    # Binarize the lung segmentation
    lung_seg_np[lung_seg_np > 1] = 1

    # Keep only the two largest connected components
    structure = np.ones((3, 3, 3), dtype=np.int)
    labeled, ncomp = label(lung_seg_np, structure)
    size_comp = [
        np.sum(labeled == l) for l in range(1, ncomp + 1)
    ]
    first_largest_comp = np.argmax(size_comp)
    label_first = first_largest_comp + 1
    size_comp[first_largest_comp] = -1
    second_largest_comp = np.argmax(size_comp)
    label_second = second_largest_comp + 1

    # To avoid cases where the two lungs are in the same component
    # and the second largest component is outside the lungs
    # we set a minimum size for the second largest component
    if size_comp[second_largest_comp] < MIN_NUM_VOXEL_PER_COMP:
        label_second = -1
    for i in range(1, ncomp + 1):
        if i in [label_first, label_second]:
            labeled[labeled == i] = 1
        else:
            labeled[labeled == i] = 0
    return labeled


def preprocess(input_image: sitk.Image) -> np.ndarray:

    # Compute the lung mask automatically
    lung_seg_np = compute_lung_seg(input_image)

    # Crop the volumes
    img_np = crop_around_mask(sitk.GetArrayFromImage(input_image), lung_seg_np, crop_margin=CROP_MARGIN)

    # Normalize the image intensity
    img_np = normalize_intensity(img_np)

    img_sitk = convert_to_sitk(img_np, input_image)

    # Resample the image
    img_sitk = resample(img_sitk)

    return img_sitk
