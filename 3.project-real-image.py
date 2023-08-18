import mindspore
import mindspore.nn as nn
import pickle
import yaml
from rich.progress import track
from rich.console import Console
import numpy as np
from PIL import Image
import argparse

from networks import Generator
import PIL

console = Console()
import func
import mindspore.context as context
import face_alignment

context.set_context(device_target="CPU")  # GPU or CPU

if __name__ == "__main__":

    net = func.get_model("ckpt/stylegan2-ffhq-config-f.mindspore.ckpt")
    input_image_aligned = func.open_and_align("obama.png")
    print("Visualizing the cropped and aligned image at test_aligned.png")
    input_image_aligned.save("test_aligned.png")
    ws = func.img_2_ws(net, input_image_aligned, step=500)  # slow!
    img = func.ws_2_img(net, ws)
    img.save("test.png")
