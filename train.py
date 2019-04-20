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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self):
        self.cfg = yaml.safe_load(open('config.yaml'))
        self.content_im_loc = self.cfg['content_image_location']
        self.style_im_loc = self.cfg['style_image_location']
        self.img_size = self.cfg['image_size']
        
    
    def preprocess_images(self, image, IMAGE_SIZE):
        """Pre-processing images to the specifications
        of the VGG19 Pre-Trained Model.
        """
        gen_transformer = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        return gen_transformer(image).to(device)
    
    def get_mode(self):
        pass
    


def main():
    t = Trainer()

if __name__=='__main__':
    main()