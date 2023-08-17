import yaml
import json

import os
import sys

import cfbd


def load_configs(config_path:str="./configs/configs.yml"):
    """
    loads a yaml file using a specified path
    """
    with open(config_path) as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    return configs

def authenticate_api(api_key:str):
    """
    authenticates the user using a specified api key.

    Returns
    -------
    configuration: cfbd.Configuration
        config object that allows user to access different endpoints
    """
    configuration = cfbd.Configuration()
    configuration.api_key['Authorization'] = api_key
    configuration.api_key_prefix['Authorization'] = 'Bearer'
    return configuration
