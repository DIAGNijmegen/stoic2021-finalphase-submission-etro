from algorithm.factory import get_network
import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    AddChannel,
    CropForeground,
    NormalizeIntensity,
    CastToType,
    ToTensor,
)
import SimpleITK as sitk

SUPPORTED_DATASET = [
    'COVID19',
]

SUPPORTED_DATA_AUGMENTATION_PIPELINES = [
    'nnUNet',
]


def convert_to_sitk(img_np, ref_img_sitk):
    img_sitk = sitk.GetImageFromArray(img_np)
    img_sitk.SetOrigin(ref_img_sitk.GetOrigin())
    img_sitk.SetSpacing(ref_img_sitk.GetSpacing())
    img_sitk.SetDirection(ref_img_sitk.GetDirection())
    return img_sitk


def segment(input_image, config, data_config, model_path):
    def pad_if_needed(img, patch_size):
        # Define my own dummy padding function because the one from MONAI
        # does not retain the padding values, and as a result
        # we cannot unpad after inference...
        img_np = img.cpu().numpy()
        shape = img.shape[2:]
        need_padding = np.any(shape < np.array(patch_size))
        if not need_padding:
            pad_list = [(0, 0)] * 3
            return img, np.array(pad_list)
        else:
            pad_list = []
            for dim in range(3):
                diff = patch_size[dim] - shape[dim]
                if diff > 0:
                    margin = diff // 2
                    pad_dim = (margin, diff - margin)
                    pad_list.append(pad_dim)
                else:
                    pad_list.append((0, 0))
            padded_array = np.pad(
                img_np,
                [(0, 0), (0, 0)] + pad_list,  # pad only the spatial dimensions
                'constant',
                constant_values=[(0, 0)] * 5,
                )
            padded_img = torch.tensor(padded_array).float()
            return padded_img, np.array(pad_list)

    device = torch.device("cuda:0")

    # Create the network and load the checkpoint
    net = get_network(
        config=config,
        in_channels=1,
        n_class=4,
        device=device,
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint['net'])

    # The inferer is in charge of taking a full volumetric input
    # and run the window-based prediction using the network.
    inferer = SlidingWindowInferer(
        roi_size=config['data']['patch_size'],      # patch size to use for inference
        sw_batch_size=1,                            # max number of windows per network inference iteration
        overlap=0.5,                                # amount of overlap between windows (in [0, 1])
        mode="gaussian",                            # how to blend output of overlapping windows
        sigma_scale=0.125,                          # sigma for the Gaussian blending. MONAI default=0.125
        padding_mode="constant",                    # for when ``roi_size`` is larger than inputs
        cval=0.,                                    # fill value to use for padding
    )

    torch.cuda.empty_cache()

    net.eval()  # Put the CNN in evaluation mode

    image_ct = sitk.GetArrayFromImage(input_image)
    image_ct = AddChannel()(img=image_ct)
    image_ct, fg_start_ine, fg_end_ine = CropForeground(return_coords=True)(img=image_ct)
    image_ct = NormalizeIntensity(nonzero=False, channel_wise=True)(img=image_ct)
    image_ct = CastToType(dtype=(np.float32))(img=image_ct)
    image_ct = np.expand_dims(image_ct, axis=0)
    input = ToTensor()(image_ct)

    with torch.no_grad():  # we do not need to compute the gradient during inference
        # Load and prepare the full image
        data = {'ct': input}
        input = torch.cat(tuple([data[key] for key in data_config['info']['image_keys']]), 1)
        input, pad_values = pad_if_needed(input, config['data']['patch_size'])
        input = input.to(device)
        pred = inferer(inputs=input, network=net)
        n_pred = 1
        # Perform test-time flipping augmentation
        flip_dims = [(2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)]
        for dims in flip_dims:
            flip_input = torch.flip(input, dims=dims)
            pred += torch.flip(
                inferer(inputs=flip_input, network=net),
                dims=dims,
            )
            n_pred += 1
        pred /= n_pred
    seg = pred.argmax(dim=1, keepdims=True).float()

    # Unpad the prediction
    seg = seg[:, :, pad_values[0,0]:seg.size(2)-pad_values[0,1], pad_values[1,0]:seg.size(3)-pad_values[1,1], pad_values[2,0]:seg.size(4)-pad_values[2,1]]

    # Insert the segmentation in the original image size
    dim = input_image.GetSize()
    full_dim = [1, 1, dim[2], dim[1], dim[0]]
    full_seg = torch.zeros(full_dim)

    full_seg[:, :, fg_start_ine[0]:fg_end_ine[0], fg_start_ine[1]:fg_end_ine[1], fg_start_ine[2]:fg_end_ine[2]] = seg
    full_seg = full_seg.cpu().numpy()
    full_seg = np.squeeze(full_seg, axis=(0, 1))

    return convert_to_sitk(full_seg, input_image)
