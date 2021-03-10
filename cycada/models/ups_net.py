import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from torch.utils import model_zoo
from torchvision.models import vgg

from .models import register_model
from upsnet.models.resnet_upsnet import resnet_upsnet
from upsnet.config.config import config
from upsnet.config.parse_args import parse_args

args = parse_args()

class UPSNET(resnet_upsnet):
    
    def __init__(self, backbone_depth):
        super().__init__(backbone_depth)
        print("Initialized super class")

@register_model('upsnet')
def upsnet(num_cls, weights_init, pretrained=True, output_last_ft=False):
    print("Number of classes to pass forward are: ", num_cls)
    model = UPSNET([3, 4, 6, 3]) # resnet_50_upsnet
    return model

if __name__ == '__main__':
    model = UPSNET([3, 4, 6, 3])