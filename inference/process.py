
from algorithm.config.loader import load_config
from algorithm.dataset_config.loader import load_covid19_data_config
from algorithm.inference import segment
from algorithm.models.convnext3D_MT_FE import convnext_base
from algorithm.preprocess import preprocess
from einops import repeat
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from features import *
import json
import numpy as np
from pathlib import Path
import pickle
import SimpleITK as sitk
import sys
from timm.models import create_model
import torch
from typing import Dict
from utils import MultiClassAlgorithm, device


COVID_OUTPUT_NAME = Path("probability-covid-19")
SEVERE_OUTPUT_NAME = Path("probability-severe-covid-19")


class StoicAlgorithm(MultiClassAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path("/input/images/ct/"),
            output_path=Path("/output/")
        )

    def predict(self, *, input_image: sitk.Image) -> Dict:
        # Get metadata
        try:
            age_from_meta_data = float(input_image.GetMetaData('PatientAge').replace('Y', ''))
            sex_from_meta_data = input_image.GetMetaData('PatientSex')
            age = (age_from_meta_data - 55)/20
            if sex_from_meta_data == 'F':
                gender = 0
            else:
                gender = 1
        except Exception as e:
            sys.exit("The following error occurred: {} \n".format(str(e)))

        # Preprocess image
        preprocessed_image, convnext_image = preprocess(input_image)

        # Perform lesion segmentation
        config = load_config(None)
        data_config = load_covid19_data_config()
        segmentation = segment(
            input_image=preprocessed_image,
            config=config,
            data_config=data_config,
            model_path='./algorithm/models/covidAug21_multiclass_v0_split112/checkpoint_epoch=1400.pt',
        )
        segmentation = sitk.Cast(segmentation, sitk.sitkUInt8)

        # Get lesion fractions from segmentations
        fraction_ggo, fraction_cons = lesion_fractions(segmentation)

        # Get intensity, kurtosis and skewness
        mean_healthy, mean_ggo, mean_cons, kurtosis_healthy, kurtosis_ggo, kurtosis_cons, \
        skewness_healthy, skewness_ggo, skewness_cons = intensity_and_texture(preprocessed_image, segmentation)

        # Rescale kurtosis and skewness to [-1, 1]. Min, max values are taken from the training data
        with open('./artifact/rescale_values.json') as json_file:
            rescale_values = json.load(json_file)
        rescale_values = json.loads(rescale_values)
        kurtosis_healthy = (2 * (kurtosis_healthy - (rescale_values['Kurtosis_healthy_min'])) / (rescale_values['Kurtosis_healthy_max'] - (rescale_values['Kurtosis_healthy_min']))) - 1
        kurtosis_ggo = (2 * (kurtosis_ggo - (rescale_values['Kurtosis_ggo_min'])) / (rescale_values['Kurtosis_ggo_max'] - (rescale_values['Kurtosis_ggo_min']))) - 1
        kurtosis_cons = (2 * (kurtosis_cons - (rescale_values['Kurtosis_cons_min'])) / (rescale_values['Kurtosis_cons_max'] - (rescale_values['Kurtosis_cons_min']))) - 1

        skewness_healthy = (2 * (skewness_healthy - (rescale_values['Skewness_healthy_min'])) / (rescale_values['Skewness_healthy_max'] - (rescale_values['Skewness_healthy_min']))) - 1
        skewness_ggo = (2 * (skewness_ggo - (rescale_values['Skewness_ggo_min'])) / (rescale_values['Skewness_ggo_max'] - (rescale_values['Skewness_ggo_min']))) - 1
        skewness_cons = (2 * (skewness_cons - (rescale_values['Skewness_cons_min'])) / (rescale_values['Skewness_cons_max'] - (rescale_values['Skewness_cons_min']))) - 1

        # Create testing array
        X_test = np.array([[age, gender, fraction_ggo, fraction_cons,
                            mean_ggo, mean_cons, mean_healthy,
                            kurtosis_ggo, kurtosis_cons, kurtosis_healthy,
                            skewness_ggo, skewness_cons, skewness_healthy]])

        # Predict severity using the saved regression model
        probabilities_severe = []
        for bootstrap in range(20):
            model_severity = pickle.load(open(f'./artifact/regression_models/probSevere_{bootstrap}', 'rb'))
            lr_prob_severe = model_severity.predict_proba(X_test)
            prob_severe = lr_prob_severe[:, 1][0]
            probabilities_severe.append(prob_severe)

        # prob_covid = np.mean(probabilities_covid)
        prob_severe = np.mean(probabilities_severe)

        # Covid classification
        model_covid_cls = create_model(
            'convnext_base',
            pretrained=False,
            in_22k=False,
            num_classes=2,
            drop_path_rate=0.0,
            layer_scale_init_value=1e-6,
            head_init_scale=1.0,
        )
        checkpoint_cls = torch.load('./algorithm/models/model_conv_net_covid.pt', map_location=device)
        model_covid_cls = model_covid_cls.to(device)
        model_covid_cls.load_state_dict(checkpoint_cls['state'], strict=False)
        model_covid_cls = model_covid_cls.eval()
        input = convnext_image.to(device)
        input = repeat(input, 'b c d h w ->b (repeat c) d h w', repeat=3)
        with torch.no_grad():
            covid_output = model_covid_cls(input, do_fe=False)
            probas = torch.nn.functional.softmax(covid_output, dim=1)
            prob_covid = probas[:, 1].item()

        return {
            COVID_OUTPUT_NAME: prob_covid,
            SEVERE_OUTPUT_NAME: prob_severe
        }


if __name__ == "__main__":
    StoicAlgorithm().process()
