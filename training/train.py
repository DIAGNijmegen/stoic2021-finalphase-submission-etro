
from algorithm.preprocess import preprocess
from algorithm.config.loader import load_config
from algorithm.dataset_config.loader import load_covid19_data_config
from algorithm.inference import segment
from features import *
import gc
import json
import os
import pandas as pd
import pickle
import SimpleITK as sitk
from sklearn.linear_model import LogisticRegression
import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
import time
import numpy as np


def get_features(data_dir):
    image_dir = os.path.join(data_dir, 'data/mha/')
    reference_path = os.path.join(data_dir, 'metadata/reference.csv')
    df_full = pd.read_csv(reference_path)

    df_full['x'] = df_full.apply(lambda row: os.path.join(image_dir, str(row['PatientID']) + '.mha'), axis=1)
    df_full['y'] = df_full.apply(lambda row: [row['probCOVID'], row['probSevere']], axis=1)

    df = df_full[df_full['probCOVID'] == 1].copy(deep=True)
    df.reset_index(drop=True, inplace=True)

    all_times = []
    for file_idx, filename in enumerate(df['x'].values):
        start = time.time()
        print('Processing {} of {}'.format(file_idx+1, len(df['x'].values)))
        patient_id = filename.split("/")[-1].replace('.mha', '')
        input_image = sitk.ReadImage(filename)

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
            print("The following error occurred: {} \n".format(str(e)))
            continue
        df.loc[df['PatientID'] == int(patient_id), 'Age'] = age
        df.loc[df['PatientID'] == int(patient_id), 'Gender'] = gender

        # Preprocess image
        preprocessed_image = preprocess(input_image)

        # Perform lesion segmentation
        config_seg = load_config(None)
        data_config = load_covid19_data_config()
        segmentation = segment(
            input_image=preprocessed_image,
            config=config_seg,
            data_config=data_config,
            model_path='./algorithm/models/covidAug21_multiclass_v0_split112/checkpoint_epoch=1400.pt',
        )
        segmentation = sitk.Cast(segmentation, sitk.sitkUInt8)

        # Get lesion fractions from segmentations
        fraction_ggo, fraction_cons = lesion_fractions(segmentation)
        df.loc[df['PatientID'] == int(patient_id), 'Fraction_ggo'] = fraction_ggo
        df.loc[df['PatientID'] == int(patient_id), 'Fraction_cons'] = fraction_cons

        # Get intensity, kurtosis and skewness
        mean_healthy, mean_ggo, mean_cons, kurtosis_healthy, kurtosis_ggo, kurtosis_cons, \
        skewness_healthy, skewness_ggo, skewness_cons = intensity_and_texture(preprocessed_image, segmentation)
        df.loc[df['PatientID'] == int(patient_id), 'Mean_intensity_healthy'] = mean_healthy
        df.loc[df['PatientID'] == int(patient_id), 'Mean_intensity_ggo'] = mean_ggo
        df.loc[df['PatientID'] == int(patient_id), 'Mean_intensity_cons'] = mean_cons

        df.loc[df['PatientID'] == int(patient_id), 'Kurtosis_healthy'] = kurtosis_healthy
        df.loc[df['PatientID'] == int(patient_id), 'Kurtosis_ggo'] = kurtosis_ggo
        df.loc[df['PatientID'] == int(patient_id), 'Kurtosis_cons'] = kurtosis_cons

        df.loc[df['PatientID'] == int(patient_id), 'Skewness_healthy'] = skewness_healthy
        df.loc[df['PatientID'] == int(patient_id), 'Skewness_ggo'] = skewness_ggo
        df.loc[df['PatientID'] == int(patient_id), 'Skewness_cons'] = skewness_cons

        gc.collect()

        end = time.time()
        print(f'Elapsed timee: {end - start}')

        all_times.append(end - start)
    print(f'Average time per instance: {np.mean(all_times)}')

    return df


def train(df, artifact_dir):
    os.makedirs(f'{artifact_dir}regression_models', exist_ok=True)

    # Rescale kurtosis and skewness to [-1, 1]. Min, max values are taken from the training data
    rescale_values = {}
    for col_name in ['Kurtosis_healthy', 'Kurtosis_ggo', 'Kurtosis_cons',
                     'Skewness_healthy', 'Skewness_ggo', 'Skewness_cons']:
        rescale_values[f'{col_name}_min'] = df[col_name].min()
        rescale_values[f'{col_name}_max'] = df[col_name].max()
        df[col_name] = (2 * (df[col_name] - df[col_name].min()) / (df[col_name].max() - df[col_name].min())) - 1
    json_string = json.dumps(rescale_values)
    with open(f'{artifact_dir}rescale_values.json', 'w') as json_file:
        json.dump(json_string, json_file)

    covariates = ['Age',
                  'Gender',
                  'Fraction_ggo',
                  'Fraction_cons',
                  'Mean_intensity_ggo',
                  'Mean_intensity_cons',
                  'Mean_intensity_healthy',
                  'Kurtosis_ggo',
                  'Kurtosis_cons',
                  'Kurtosis_healthy',
                  'Skewness_ggo',
                  'Skewness_cons',
                  'Skewness_healthy']

    # for metric in ['probSevere', 'probCOVID']:
    random_state = 0
    n_models = 20
    model = LogisticRegression(random_state=0,
                               solver='liblinear',
                               penalty='l2',
                               C=1.0,
                               l1_ratio=None)
    for bootstrap in range(n_models):
        df_train = df.sample(frac=1.0, replace=True, weights=None, random_state=random_state, axis=0)

        X_train = df_train[covariates].to_numpy()
        y_train = df_train[['probSevere']].to_numpy()

        model.fit(X_train, y_train.ravel())
        pickle.dump(model, open(f'{artifact_dir}regression_models/probSevere_{bootstrap}', 'wb'))

        random_state += 1


def do_learning(data_dir, artifact_dir):
    """
    You can implement your own solution to the STOIC2021 challenge by editing this function.
    :param data_dir: Input directory that the training Docker container has read access to. This directory has the same
        structure as the stoic2021-training S3 bucket (see https://registry.opendata.aws/stoic2021-training/)
    :param artifact_dir: Output directory that, after training has completed, should contain all artifacts (e.g. model
        weights) that the inference Docker container needs. It is recommended to continuously update the contents of
        this directory during training.
    :returns: A list of filenames that are needed for the inference Docker container. These are copied into artifact_dir
        in main.py. If your model already produces all necessary artifacts into artifact_dir, an empty list can be
        returned. Note: To limit the size of your inference Docker container, please make sure to only place files that 
        are necessary for inference into artifact_dir.
    """

    df = get_features(data_dir)
    train(df, artifact_dir)
    artifacts = []  # empty list because train() already writes all artifacts to artifact_dir

    return artifacts

