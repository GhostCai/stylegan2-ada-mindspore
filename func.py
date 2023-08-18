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
from dnnlib.util import EasyDict
import json
from PIL import Image
from mindspore.train import Model

from tqdm import tqdm
import pretty_errors
import rich
from rich.console import Console
from easydict import EasyDict

import mslpips

lpips = mslpips.get_lpips("ckpt/vgg16.ckpt")


def convert_torch_to_mindspore(torch_path):
    """
    将pytorch模型转换为mindspore模型。
    输入：torch_path，pytorch模型的路径，pkl格式。
    """
    input_path = torch_path
    output_weight_path = torch_path.replace(".pkl", ".mindspore.ckpt")
    output_config_path = torch_path.replace(".pkl", ".mindspore.json")
    with dnnlib.util.open_url(input_path) as f:
        torch_G = legacy.load_network_pkl(f)["G_ema"]
    weight_dict = torch_G.state_dict()
    print("Torch Model Loaded.")

    for k, v in weight_dict.items():
        weight_dict[k] = mindspore.Parameter(v.cpu().numpy(), name=k)
    print("Weights converted to mindspore.")

    # yaml.dump(torch_G.init_kwargs, open(output_config_path, 'w'))
    # print(f'Config saved to {output_config_path}')
    json.dump(torch_G.init_kwargs, open(output_config_path, "w"))
    print(f"Config saved to {output_config_path}")

    net = networks.Generator(**torch_G.init_kwargs)
    net.set_train(False)
    not_loaded, _ = mindspore.load_param_into_net(net, weight_dict, strict_load=False)

    print("All weights loaded.")

    net.compute_mean()

    mindspore.save_checkpoint(net, "ckpt/stylegan2-ffhq-config-f.mindspore.ckpt")
    print(f"Weight saved to {output_weight_path}")


def get_model(model_path):
    """
    从model_path加载模型，返回mindspore模型。
    """
    config_path = model_path.replace(".ckpt", ".json")
    print(f'Loading model config from {model_path.replace(".ckpt", ".json")}...')
    config = json.load(open(config_path))
    net = networks.Generator(**config)
    net.set_train(False)
    print(f"Loading model weights from {model_path}...")
    parms = mindspore.load_checkpoint(model_path, net=net)
    mindspore.load_param_into_net(net, parameter_dict=parms)
    print("Model loaded.")
    return net


def seed_2_ws(net: networks.Generator, seed: int) -> mindspore.Tensor:
    """
    用种子seed生成ws code
    """
    z = mindspore.Tensor(np.random.RandomState(seed).randn(1, 512).astype(np.float32))
    ws = net.mapping(z, None, truncation_psi=1, truncation_cutoff=None)
    return ws


def ws_2_img(net: networks.Generator, ws: np.ndarray) -> PIL.Image.Image:
    """
    用ws code生成真实图片
    """
    img = net(ws, truncation_psi=0.5, noise_mode="const")
    img = (img - 0.5) / 0.5
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(mindspore.uint8)
    img = Image.fromarray(img[0].numpy(), "RGB")
    return img


def psnr_from_mse(v):
    """Convert MSE to PSNR."""
    return -10.0 * (
        mindspore.numpy.log(v) / mindspore.numpy.log(mindspore.Tensor([10.0]))
    )


def img_2_ws(
    net: networks.Generator, img: PIL.Image.Image, step=501
) -> mindspore.Tensor:

    """
    用真实照片反投影得到ws code
    """
    step = step + 1
    import mindspore as ms
    import mindspore.ops as ops
    import numpy as np

    ws_opt = ms.Parameter(ms.Tensor(net.mean_latent, dtype=ms.float32), name="ws_opt")
    optimizer = ms.nn.Adam(
        [ws_opt], learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8
    )
    optimizer_ft = ms.nn.Adam(
        net.trainable_params() + [ws_opt],
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    )
    # pil to numpy
    img = np.array(img)
    img = img.transpose(2, 0, 1)
    img = img / 255
    img = ms.Tensor(img, dtype=ms.float32)

    def forward_fn(data, label):
        logits = net(data)
        label = label.unsqueeze(0)
        loss_l1 = ((logits - label) ** 2).mean()
        loss_lpips = lpips(logits, label)
        loss = loss_l1 + loss_lpips
        print(
            f"loss:{loss.asnumpy()},psnr:{psnr_from_mse(mindspore.numpy.array(loss_l1))},lpips:{loss_lpips.asnumpy()}"
        )
        return loss, logits

    grad_fn = mindspore.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True
    )
    grad_fn_ft = mindspore.value_and_grad(
        forward_fn, None, optimizer_ft.parameters, has_aux=True
    )
    for i in range(step):
        tmp = net(ws_opt)
        if i < step // 2:
            (loss, _), grads = grad_fn(ws_opt, img)
            optimizer(grads)
        else:
            (loss, _), grads = grad_fn_ft(ws_opt, img)
            optimizer_ft(grads)

        if i % 10 == 0:
            img_peek = ws_2_img(net, ws_opt)
            print(i)
            if i < step // 2:
                img_peek.save(f"debug/{i:04d}.png")
            else:
                img_peek.save(f"debug/{i:04d}_ft.png")

    return ws_opt


import numpy as np
import PIL
import PIL.Image
import scipy
import scipy.ndimage
import dlib
from pathlib import Path


"""
brief: face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
author: lzhbrian (https://lzhbrian.me)
date: 2020.1.5
note: code is heavily borrowed from
    https://github.com/NVlabs/ffhq-dataset
    http://dlib.net/face_landmark_detection.py.html

requirements:
    apt install cmake
    conda install Pillow numpy scipy
    pip install dlib
    # download face landmark model from:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""


def get_landmark(filepath, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()
    if isinstance(filepath, str):
        img = dlib.load_rgb_image(filepath)
    elif isinstance(filepath, np.ndarray):
        img = filepath
    elif isinstance(filepath, PIL.Image.Image):
        img = np.array(filepath)
    else:
        raise TypeError("filepath should be str or np.ndarray or PIL.Image.Image")
    dets = detector(img, 1)
    filepath = Path(filepath)
    print(f"{filepath.name}: Number of faces detected: {len(dets)}")
    shapes = [predictor(img, d) for k, d in enumerate(dets)]

    lms = [np.array([[tt.x, tt.y] for tt in shape.parts()]) for shape in shapes]

    return lms


def open_and_align(filepath):
    """
    :param filepath: str
    :return: list of PIL Images
    """

    predictor = dlib.shape_predictor("ckpt/shape_predictor_68_face_landmarks.dat")
    lms = get_landmark(filepath, predictor)
    imgs = []
    for lm in lms:
        lm_chin = lm[0:17]  # left-right
        lm_eyebrow_left = lm[17:22]  # left-right
        lm_eyebrow_right = lm[22:27]  # left-right
        lm_nose = lm[27:31]  # top-down
        lm_nostrils = lm[31:36]  # top-down
        lm_eye_left = lm[36:42]  # left-clockwise
        lm_eye_right = lm[42:48]  # left-clockwise
        lm_mouth_outer = lm[48:60]  # left-clockwise
        lm_mouth_inner = lm[60:68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # read image
        img = PIL.Image.open(filepath)

        output_size = 1024
        transform_size = 4096
        enable_padding = True

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (
                int(np.rint(float(img.size[0]) / shrink)),
                int(np.rint(float(img.size[1]) / shrink)),
            )
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))),
        )
        crop = (
            max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]),
        )
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))),
        )
        pad = (
            max(-pad[0] + border, 0),
            max(-pad[1] + border, 0),
            max(pad[2] - img.size[0] + border, 0),
            max(pad[3] - img.size[1] + border, 0),
        )
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(
                np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect"
            )
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(
                1.0
                - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                1.0
                - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
            )
            blur = qsize * 0.02
            img += (
                scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img
            ) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), "RGB")
            quad += pad[:2]

        # Transform.
        img = img.transform(
            (transform_size, transform_size),
            PIL.Image.QUAD,
            (quad + 0.5).flatten(),
            PIL.Image.BILINEAR,
        )
        if output_size < transform_size:
            img = img.resize((output_size, output_size))

        # Save aligned image.
        imgs.append(img)
    if len(imgs) == 0:
        print("no face detected")
        return None
    elif len(imgs) > 1:
        print("more than one face detected, choose the first one")
    imgs = imgs[0]
    # if has alpha channel, remove
    if imgs.mode == "RGBA":
        imgs = imgs.convert("RGB")

    return imgs
