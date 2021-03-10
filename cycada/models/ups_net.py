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

class UPSNET(resnet_upsnet):
    
    def __init__(self):
        super().__init__()
        # print("Hello! I am the initial guy")
        # pass
    # pass

@register_model('upsnet')
def upsnet():
    model = UPSNET([3, 4, 23, 3])
    return model

if __name__ == '__main__':
    model = UPSNET([3, 4, 23, 3])