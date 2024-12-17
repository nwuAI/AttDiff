import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

class DictConfig(object):
    """Creates a Config object from a dict
       such that object attributes correspond to dict keys.
    """

    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def __str__(self):
        return '\n'.join(f"{key}: {val}" for key, val in self.__dict__.items())

    def __repr__(self):
        return self.__str__()

def get_config(fname):
    with open(fname, 'r', encoding='utf-8') as stream:
        config_dict = yaml.load(stream, Loader)
    config = DictConfig(config_dict)
    return config