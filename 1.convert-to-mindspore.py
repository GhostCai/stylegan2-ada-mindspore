import mindspore


import upfirdn2d
import networks
import bias_act
import dnnlib
import legacy

# import torch
import numpy as np
import PIL.Image
import pickle
import yaml

from tqdm import tqdm
import pretty_errors
import rich
from rich.console import Console

console = Console()

from func import convert_torch_to_mindspore

if __name__ == "__main__":

    convert_torch_to_mindspore("ckpt/stylegan2-ffhq-config-f.pkl")
