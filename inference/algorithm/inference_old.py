from algorithm.factory import get_network
import numpy as np
import torch
from monai.data import Dataset, DataLoader, NiftiSaver
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    CropForegroundd,
    NormalizeIntensityd,
    CastToTyped,
    ToTensord,
)

SUPPORTED_DATASET = [
    'COVID19',
]

SUPPORTED_DATA_AUGMENTATION_PIPELINES = [
    'nnUNet',
]


def covid19_inference_transform(config, image_keys):
    inference_transform = Compose([
        LoadImaged(keys=image_keys),
        AddChanneld(keys=image_keys),
        CropForegroundd(keys=image_keys, source_key=image_keys[0]),
        NormalizeIntensityd(keys=image_keys, nonzero=False, channel_wise=True),
        CastToTyped(keys=image_keys, dtype=(np.float32,)),
        ToTensord(keys=image_keys),
    ])
    return inference_transform


def single_case_dataloader(inference_transform, input_path_dict):
    """
    :param inference_transform
    :param input_path_dict: dict; keys=image_keys, values=paths
    :return:
    """
    data_dicts = [input_path_dict]
    ds = Dataset(
        data=data_dicts,
        transform=inference_transform,
    )
    loader = DataLoader(
        ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=0, # you can set it to a value higher than 0 to activate parallel preprocessing; for me it leads to an error...
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def get_single_case_dataloader(config, data_config, input_path_dict):
    """
    Typically used for inference.
    :param config:
    :param data_config:
    :param input_path_dict:
    :return:
    """
    dataset_name = data_config['name']

    # Check the dataset name
    assert dataset_name in SUPPORTED_DATASET, \
        'Found dataset %s. But only %s are supported for inference.' % \
        (dataset_name, str(SUPPORTED_DATASET))

    # Get the inference transform pipeline for the dataset
    if dataset_name == 'COVID19':
        inference_pipeline = covid19_inference_transform(
            config=config,
            image_keys=data_config['info']['image_keys'],
        )
    else:
        inference_pipeline = None
        NotImplementedError('Unknown dataset for inference transform: %s' % dataset_name)

    # Create the dataloader
    dataloader = single_case_dataloader(
        inference_transform=inference_pipeline,
        input_path_dict=input_path_dict,
    )

    return dataloader


def segment(config, data_config, model_path, input_path_dict):
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

    # Create the dataloader for the single case to segment
    dataloader = get_single_case_dataloader(
        config=config,
        data_config=data_config,
        input_path_dict=input_path_dict,
    )

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
        roi_size=config['data']['patch_size'],  # patch size to use for inference
        sw_batch_size=1,  # max number of windows per network inference iteration
        overlap=0.5,  # amount of overlap between windows (in [0, 1])
        mode="gaussian",  # how to blend output of overlapping windows
        sigma_scale=0.125,  # sigma for the Gaussian blending. MONAI default=0.125
        padding_mode="constant",  # for when ``roi_size`` is larger than inputs
        cval=0.,  # fill value to use for padding
    )

    torch.cuda.empty_cache()

    net.eval()  # Put the CNN in evaluation mode
    with torch.no_grad():  # we do not need to compute the gradient during inference
        # Load and prepare the full image
        data = [d for d in dataloader][0]  # load the full image
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
    meta_data = data['%s_meta_dict' % data_config['info']['image_keys'][0]]
    dim = meta_data['spatial_shape'].cpu().numpy()
    full_dim = [1, 1, dim[0,0], dim[0,1], dim[0,2]]
    fg_start = data['foreground_start_coord'][0]
    fg_end = data['foreground_end_coord'][0]
    full_seg = torch.zeros(full_dim)
    full_seg[:, :, fg_start[0]:fg_end[0], fg_start[1]:fg_end[1], fg_start[2]:fg_end[2]] = seg

    saver = NiftiSaver(output_dir='./', mode="nearest", output_postfix="")
    saver.save_batch(full_seg, meta_data=meta_data)
