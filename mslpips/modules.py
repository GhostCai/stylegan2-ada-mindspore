import numpy as np
from mindspore import Tensor, load_checkpoint, load_param_into_net, nn
from mindspore.train import Model
import mindspore
from .vgg16.fine_tune import import_data
from .vgg16.model_utils.moxing_adapter import config
from .vgg16.src.vgg import Vgg


def get_vgg16(vgg16_path:str):

    cfg = {
        '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    net = Vgg(cfg['16'], num_classes=1000, args=config, batch_norm=True,include_top=False,phase='test')

    param_dict = load_checkpoint(vgg16_path)
    load_param_into_net(net, param_dict)
    return net

def get_lpips(vgg16_path:str):
    net = LPIPS(vgg16_path)
    return net

class LPIPS(mindspore.nn.Cell):
    def __init__(self,vgg16_path:str):
        super().__init__()
        self.vgg = get_vgg16(vgg16_path)
        
    def construct(self,im1,im2):
        im1 = self.vgg(im1)
        im2 = self.vgg(im2)
        ret = ((im1-im2)**2).mean()
        # print('ret:',ret)
        return ret
        

    