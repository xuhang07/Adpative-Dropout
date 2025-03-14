# flake8: noqa
import os
import os.path as osp
os_path=os.getcwd()
import sys
sys.path.append(os_path)
from archs.train_pipeline import train_pipeline

import archs
import data
import models



if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    opt = '../options/train/train_realsrresnet_withdropout.yml'
    train_pipeline(root_path, opt)
