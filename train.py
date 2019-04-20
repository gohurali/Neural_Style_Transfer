__author__='Gohur Ali'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import time
import yaml
import argparse
"""
Neural Style Transfer in PyTorch

This code is based off of CVPR 2016 Paper:
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

Using VGG19 Architecture for Feature Extraction
"""
class Trainer:
    def __init__(self):
        self.cfg = yaml.safe_load(open('config.yaml'))
        pass


def main():
    pass

if __name__=='__main__':
    main()