import mindspore
import mindspore.nn as nn
import pickle
import yaml
from rich.progress import track
from rich.console import Console
import numpy as np
from PIL import Image


from networks import Generator
import func

console = Console()

mindspore.context.set_context(device_target="CPU")

if __name__ == "__main__":
    # 创建mindspore模型
    net = func.get_model("ckpt/stylegan2-ffhq-config-f.mindspore.ckpt")

    # 生成图片
    for seed in track(range(100), description="Generating"):
        print(f"Seed {seed}...")
        ws = func.seed_2_ws(net, seed)
        img = func.ws_2_img(net, ws)
        img.save(f"out/test_{seed}.png")
