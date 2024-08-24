import yaml
import json

import os
import sys

import cfbd

import pickle


def load_configs(config_path: str = "./configs/configs.yml"):
    """
    loads a yaml file using a specified path
    """
    with open(config_path) as f:
        configs = yaml.load(f, Loader=yaml.Loader)
    return configs


def authenticate_api(api_key: str):
    """
    authenticates the user using a specified api key.

    Returns
    -------
    configuration: cfbd.Configuration
        config object that allows user to access different endpoints
    """
    configuration = cfbd.Configuration()
    configuration.api_key["Authorization"] = api_key
    configuration.api_key_prefix["Authorization"] = "Bearer"
    return configuration


def save_model(model, metadata):
    name = metadata.get("model_name")
    if not os.path.isdir(f"src/models/{name}"):
        os.mkdir(f"src/models/{name}")
    model_version = get_latest_version_number(f"src/models/{name}", ext=".pkl") + 1
    metadata["model_version"] = str(model_version)
    with open(f"src/models/{name}/v{model_version}.pkl", "wb") as file:
        pickle.dump(model, file)
    with open(f"src/models/{name}/v{model_version}_metadata.json", "w") as json_file:
        json.dump(metadata, json_file, indent=4)


def load_model():
    inference_configs = load_configs("configs/inference_config.yml")
    model_name = inference_configs.get("model_name")
    model_version = inference_configs.get("model_version")
    model_version = (
        model_version
        if model_version
        else get_latest_version_number(f"src/models/{model_name}", ".pkl")
    )
    with open(f"src/models/{model_name}/v{model_version}.pkl", "rb") as f:
        model = pickle.load(f)
    with open(
        f"src/models/{model_name}/v{model_version}_metadata.json", "r"
    ) as json_file:
        metadata = json.load(json_file)
    return model, metadata


def get_latest_version_number(path, ext=".csv"):
    """
    grabs the latest version number based on a specific folder.

    parameters
    ----------
    path: str
        path to directory with data
    ext: str
        type of file. default is .csv

    returns
    -------
    v_num: int
        latest version number
    """
    v_nums = []
    for file in os.listdir(path):
        if file.endswith(ext):
            if file.startswith("v"):
                v_nums.append(int(file.split(ext)[0].strip("v")))
    if v_nums:
        v_num = sorted(v_nums)[-1]
    else:
        v_num = 0
    return v_num
