import os
os_path=os.getcwd()
import sys
sys.path.append(os_path)

import logging
import torch
from os import path as osp

import archs
import data
import models

from archs.test_pipeline import test_pipeline

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    opt = '../options/test/test_realsrresnet_withdropout.yml'
    test_pipeline(root_path,opt)
