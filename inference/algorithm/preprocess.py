import numpy as np
import SimpleITK as sitk
from typing import Iterable
from algorithm.preprocessing.segmentation import compute_lung_seg, add_lung_label
from algorithm.preprocessing.spatial import crop_around_mask, resample, crop_around_mask_convnext
from algorithm.preprocessing.intensity import normalize_intensity
from algorithm.preprocessing.definitions import *
from monai.transforms import AddChannel, CastToType, ScaleIntensityRange, ToTensor, ResizeWithPadOrCrop, Orientation, MaskIntensity


def preprocess(input_image: sitk.Image):

    # Compute the lung mask automatically
    lung_seg_np = compute_lung_seg(input_image)

    # Crop the volumes
    img_np = crop_around_mask(sitk.GetArrayFromImage(input_image), lung_seg_np, crop_margin=CROP_MARGIN)

    # Normalize the image intensity
    img_np = normalize_intensity(img_np)

    img_sitk = convert_to_sitk(img_np, input_image)

    # Resample the image
    img_sitk = resample(img_sitk)

    # Part for Convnext
    image_ct = AddChannel()(img=sitk.GetArrayFromImage(input_image))
    image_lung_mask = AddChannel()(img=lung_seg_np)
    image_ct = Orientation(axcodes="RAS")(image_ct)[0]
    image_lung_mask = Orientation(axcodes="RAS")(image_lung_mask)[0]
    image_ct = MaskIntensity(mask_data=image_lung_mask)(image_ct)
    image_ct = ScaleIntensityRange(a_min=-1150, a_max=350, b_min=0.0, b_max=1.0, clip=True)(image_ct)
    image_ct = crop_around_mask_convnext(image_ct, image_lung_mask, crop_margin=CROP_MARGIN, return_coordinates=False)
    image_ct = ResizeWithPadOrCrop(spatial_size=(112, 224, 224), mode='constant')(image_ct)
    image_ct = np.expand_dims(image_ct, axis=0)
    image_ct = CastToType(dtype=np.float32)(image_ct)
    image_ct = ToTensor()(image_ct)

    return img_sitk, image_ct
