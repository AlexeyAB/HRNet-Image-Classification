"""File for accessing HRNet via PyTorch Hub https://pytorch.org/hub/

Usage:
    import torch
    model = torch.hub.load('AlexeyAB/HRNet-Image-Classification', 'hrnetv2_w18', pretrained=True)
"""

dependencies = ['torch', 'yaml']


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

import urllib.request


_hubconf_path = os.path.dirname(__file__)

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

def create(yaml_file, pt_file, pretrained):
    """Creates a specified HRNet model

    Arguments:
        yaml_file (str): name of model, i.e. 'experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        pt_file (str): name of model, i.e. 'hrnetv2_w18_imagenet_pretrained.pth'
        pretrained (bool): load pretrained weights into the model

    Returns:
        pytorch model
    """
    try:
        print('yaml_file: ', yaml_file)
        print(" parse_args...")
        args = parse_args(yaml_file, pt_file)
        print(" eval...")
        model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(config)
        print(" load_state_dict...")
        if pretrained:
            url_name = 'https://github.com/AlexeyAB/HRNet-Image-Classification/releases/download/26_05_2020_pyhub/' + pt_file
            print('pt_file: ', pt_file)
            if not os.path.isfile(pt_file):
                print('Download file: ', url_name)
                urllib.request.urlretrieve(url_name, pt_file)
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
        return model

    except Exception as e:
        s = 'Cache maybe be out of date, deleting cache and retrying may solve this. See %s for help.' % help_url
        raise Exception(s) from e

def hrnetv2_w18_small1(pretrained=False):
    """ hrnetv2_w18-small model from 

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False

    Returns:
        pytorch model
    """
    return create(_hubconf_path + '/experiments/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100.yaml', 'hrnet_w18_small_model_v1.pth', pretrained)


def hrnetv2_w18_small2(pretrained=False):
    """ hrnetv2_w18-small model from 

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False

    Returns:
        pytorch model
    """
    return create(_hubconf_path + '/experiments/cls_hrnet_w18_small_v2_sgd_lr5e-2_wd1e-4_bs32_x100.yaml', 'hrnet_w18_small_model_v2.pth', pretrained)


def hrnetv2_w18(pretrained=False):
    """ hrnetv2_w18-small model from 

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False

    Returns:
        pytorch model
    """
    return create(_hubconf_path + '/experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml', 'hrnetv2_w18_imagenet_pretrained.pth', pretrained)


def hrnetv2_w32(pretrained=False):
    """ hrnetv2_w32-middle model from 

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False

    Returns:
        pytorch model
    """
    return create(_hubconf_path + '/experiments/cls_hrnet_w32_sgd_lr5e-2_wd1e-4_bs32_x100.yaml', 'hrnetv2_w32_imagenet_pretrained.pth', pretrained)


def hrnetv2_w64(pretrained=False):
    """ hrnetv2_w64-large model from 

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False

    Returns:
        pytorch model
    """
    return create(_hubconf_path + '/experiments/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml', 'hrnetv2_w64_imagenet_pretrained.pth', pretrained)
