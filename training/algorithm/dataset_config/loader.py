# Copyright 2021 Lucas Fidon

import os
import yaml


def load_dataset_config(config_file=None, verbose=False):
    if not os.path.isfile(config_file):
        raise FileNotFoundError('Dataset config file %s not found' % config_file)

    if verbose:
        print('\n*** Dataset config file')
        print(config_file)

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Set the number of input channels
    config['info']['in_channels'] = len(config['info']['image_keys'])

    # Set the number of classes
    config['info']['n_class'] = 1 + len(config['info']['labels'].keys())

    if verbose:
        print(config['log']['message'])

    return config


def load_covid19_data_config():
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "covid19.yml"
    )
    config = load_dataset_config(config_file=config_file)
    return config
