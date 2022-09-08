from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.transforms.functional as TF

from tqdm import tqdm
import numpy as np
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import skimage
import skimage.io
import skimage.transform
from PIL import Image
import time
import os
from os.path import join, exists
import copy
import random
from collections import OrderedDict
from sklearn.metrics import r2_score

from torch.nn import functional as F
from torchvision.models import Inception3, resnet18, resnet34, resnet50

from utils.image_dataset import *

subgroup = '27counties'
data_dir = '/home/ubuntu/projects/data/DeepSolar_images_' + subgroup
old_ckpt_path = 'checkpoint/type_5classes/resnet50_multilabels/FiveClasses_lr_0.0001_decay_4_wd_0_2_last.tar'
result_path = 'results/type_5classes/type_prob_dict_' + subgroup + '.pickle'
error_list_path = 'results/type_5classes/type_error_list_' + subgroup + '.pickle'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_arch = 'resnet50'
nclasses = 4
batch_size = 128
input_size = 299


transform_test = transforms.Compose([
        transforms.Resize((input_size, input_size)),
#         MyCrop(17, 0, 240, 299),
        # transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


class OrdinaryImageDataset(Dataset):
    def __init__(self, data_dir, transform, latest_prob_dict):
        self.path_list = []
        self.transform = transform

        for f in os.listdir(data_dir):
            if not f[-4:] == '.png':
                continue
            idx = int(f[:-4])
            if idx in latest_prob_dict:
                continue
            self.path_list.append((join(data_dir, f), idx))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        image_path, idx = self.path_list[index]
        img = Image.open(image_path)
        if not img.mode == 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, idx

if __name__ == '__main__':
    # load existing prob dict or initialize a new one
    if exists(result_path):
        with open(result_path, 'rb') as f:
            prob_dict = pickle.load(f)
    else:
        prob_dict = {}

    # load existing error list or initialize a new one
    if exists(error_list_path):
        with open(error_list_path, 'rb') as f:
            error_list = pickle.load(f)
    else:
        error_list = []

    # dataloader
    dataset_pred = OrdinaryImageDataset(data_dir, transform=transform_test, latest_prob_dict=prob_dict)
    print('Dataset size: ' + str(len(dataset_pred)))
    dataloader_pred = DataLoader(dataset_pred, batch_size=batch_size, shuffle=False, num_workers=4)

    if model_arch == 'resnet18':
        model = resnet18(num_classes=nclasses)
    elif model_arch == 'resnet34':
        model = resnet34(num_classes=nclasses)
    elif model_arch == 'resnet50':
        model = resnet50(num_classes=nclasses)
    elif model_arch == 'inception':
        model = Inception3(num_classes=nclasses, aux_logits=True, transform_input=False)
    else:
        raise

    model = model.to(device)

    # load old parameters
    checkpoint = torch.load(old_ckpt_path, map_location=device)
    if old_ckpt_path[-4:] == '.tar':  # it is a checkpoint dictionary rather than just model parameters
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print('Old checkpoint loaded: ' + old_ckpt_path)

    model.eval()
    # run
    count = 0
    for inputs, idx_list in tqdm(dataloader_pred):
        try:
            inputs = inputs.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                prob = torch.sigmoid(outputs)
            prob_list = prob.cpu().numpy()
            for i in range(len(idx_list)):
                idx = idx_list[i]
                idx = idx.item()
                prob_sample = prob_list[i]
                prob_dict[idx] = prob_sample

        except:  # take a note on the batch that causes error
            error_list.append(idx_list)

        if count % 400 == 0:
            with open(join(result_path), 'wb') as f:
                pickle.dump(prob_dict, f)
            with open(join(error_list_path), 'wb') as f:
                pickle.dump(error_list, f)
        count += 1

    with open(join(result_path), 'wb') as f:
        pickle.dump(prob_dict, f)
    with open(join(error_list_path), 'wb') as f:
        pickle.dump(error_list, f)

    print('Done!')
