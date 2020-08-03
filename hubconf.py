# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

"""File for accessing HRNet via PyTorch Hub https://pytorch.org/hub/

Usage:
    import torch
    model = torch.hub.load('AlexeyAB/HRNet-Image-Classification', 'hrnetv2_w18', pretrained=True)
"""

dependencies = ['torch', 'yaml']


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import tools._init_paths
import models
from config import config
from config import update_config
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger

from models.yolo import Model
from utils import google_utils
import urllib.request


def parse_args(yaml_file, pt_file):
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        default=yaml_file)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default=pt_file)

    args = parser.parse_args()
    update_config(config, args)

    return args

def create(yaml_name, pt_file, pretrained):
    """Creates a specified HRNet model

    Arguments:
        yaml_name (str): name of model, i.e. 'experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        pt_file (str): name of model, i.e. 'hrnetv2_w18_imagenet_pretrained.pth'
        pretrained (bool): load pretrained weights into the model

    Returns:
        pytorch model
    """
    #config = os.path.join(os.path.dirname(__file__), 'models', '%s.yaml' % name)  # model.yaml path
    try:
        args = parse_args(yaml_name, pt_file)
        model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(config)
        if pretrained:
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
        return model

    except Exception as e:
        s = 'Cache maybe be out of date, deleting cache and retrying may solve this. See %s for help.' % help_url
        raise Exception(s) from e


def hrnetv2_w18(pretrained=False):
    """ hrnetv2_w18-small model from 

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False

    Returns:
        pytorch model
    """
    return create('experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml', 'hrnetv2_w18_imagenet_pretrained.pth', pretrained)


def hrnetv2_w32(pretrained=False):
    """ hrnetv2_w32-middle model from 

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False

    Returns:
        pytorch model
    """
    return create('experiments/cls_hrnet_w32_sgd_lr5e-2_wd1e-4_bs32_x100.yaml', 'hrnetv2_w32_imagenet_pretrained.pth', pretrained)


def hrnetv2_w64(pretrained=False):
    """ hrnetv2_w64-large model from 

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False

    Returns:
        pytorch model
    """
    return create('experiments/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml', 'hrnetv2_w64_imagenet_pretrained.pth', pretrained)
