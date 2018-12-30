import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision
from easydict import EasyDict
import model_level_attention
from anchors import Anchors
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import csv_eval
import cv2

print('CUDA available: {}'.format(torch.cuda.is_available()))

ckpt =  False
def main(args=None):
    parser = EasyDict()
    parser.pretrained='resnet50-19c8e357.pth'
    parser.depth=50
    parser.class_list=['face']
    parser.train_file='face'
    parser.num_classes=1


    # dataset_val = CSVDataset(train_file=parser.train_file, class_list=parser.class_list, transform=transforms.Compose([Resizer(), Normalizer()]))

    # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=2, drop_last=False)
    # dataloader_val = DataLoader(dataset_val, num_workers=16, collate_fn=collater, batch_sampler=sampler_val)
        #dataloader_val = DataLoader(dataset_train, num_workers=16, collate_fn=collater, batch_size=8, shuffle=True)

    # Create the model_pose_level_attention
    if parser.depth == 18:
        retinanet = model_level_attention.resnet18(num_classes=parser.num_classes)
    elif parser.depth == 34:
        retinanet = model_level_attention.resnet34(num_classes=parser.num_classes)
    elif parser.depth == 50:
        retinanet = model_level_attention.resnet50(num_classes=parser.num_classes)
    elif parser.depth == 101:
        retinanet = model_level_attention.resnet101(num_classes=parser.num_classes)
    elif parser.depth == 152:
        retinanet = model_level_attention.resnet152(num_classes=parser.num_classes)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')


    retinanet_dict = retinanet.state_dict()
    pretrained_dict = torch.load('./weight/' + parser.pretrained)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in retinanet_dict}
    retinanet_dict.update(pretrained_dict)
    retinanet.load_state_dict(retinanet_dict)
    print('load pretrained backbone')

    print(retinanet)
    retinanet = torch.nn.DataParallel(retinanet, device_ids=[0])
    retinanet.cuda()

    retinanet.training = False

    retinanet.eval()

    iters = 0

    path=r'D:\project\face\facenet\data\images/'

    for i in range(10):
        for img in os.listdir(path):

            image=cv2.imread(path+img)
            image = np.transpose(image, (2, 0, 1))
            image=np.array([image])

            image=torch.from_numpy(image)
            start=time.time()
            pres = retinanet(image.cuda().float())
            print(pres,time.time()-start)
            # [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]] = retinanet([data['img'].cuda().float(), data['annot']])
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

if __name__ == '__main__':
    main()
