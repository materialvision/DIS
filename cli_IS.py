# Set up
import os
import shutil
import cv2
import matplotlib.pyplot as plt

import time
import numpy as np
from skimage import io
import time
from PIL import Image

import glob

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from data_loader_cache import get_im_gt_name_dict, create_dataloaders, GOSRandomHFlip, GOSResize, GOSRandomCrop, GOSNormalize #GOSDatasetCache,
from models import *

import argparse

def imread(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def save_output(image_name,pred,d_dir):

    predict_np = pred
    #predict = predict.squeeze()
    #predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGBA')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+'/'+imidx+'.png')
    print(d_dir+'/'+imidx+'.png')

    # mask = Image.open('data/src/horse.png').convert('L').resize(im1.size)
    #mask = imo
    mode = 'RGBA'  # for color image “L” (luminance) for greyscale images, “RGB” for true color images, and “CMYK” for pre-press images.
    width, height = imo.size
    size = (width, height)
    color = (0,0,0,0)

    bkim = Image.new(mode, size, color)
    im1 = Image.open(image_name)
    imomask = imo.convert('L')
    frontimo = Image.composite(im1, bkim, imomask)
    frontimo.save(d_dir+'/front_'+imidx+'.png')
    print(d_dir+'/front_'+imidx+'.png')

net = ISNetDIS() 
net.load_state_dict(torch.load("isnet.pth",map_location="cpu"))
net.eval()
net.cuda()

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(description="Process input and output folder paths")
parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder")
parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder")
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder

# Get the list of image files from the input folder
input_list = glob.glob(os.path.join(input_folder, '*'))

for image_file in input_list:
    im = io.imread(image_file)/255
    w,h,_ = im.shape
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    im_tensor = torch.tensor(im, dtype=torch.float32).cuda()
    im_tensor = torch.transpose(torch.transpose(im_tensor,1,2),0,1)
    im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1,1,1])
    im_tensor = torch.unsqueeze(im_tensor,0)
    im_tensor = F.interpolate(im_tensor, size=(1024,1024))

    ds_val = net(im_tensor)[0]
    im_pred = F.interpolate(ds_val[0], size=(w,h))

    im_pred = torch.squeeze(im_pred)
    ma = torch.max(im_pred)
    mi = torch.min(im_pred)
    im_pred = (im_pred-mi)/(ma-mi)
    im_result = im_pred.to('cpu').detach().numpy().copy()
    #io.imsave(os.path.join(result_folder,os.path.basename(image_file)), im_result)
    save_output(os.path.join(input_folder,os.path.basename(image_file)),im_result,output_folder)
