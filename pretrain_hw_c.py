#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
# MEhreen comment change warpctc_pytorch to torch.nn.CTCLoss
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss

import hw
from hw import hw_dataset
from hw import cnn_lstm
from hw.hw_dataset import HwDataset
import pickle

from utils.dataset_wrapper import DatasetWrapper
from utils import safe_load

import numpy as np
import cv2
import sys
import json
import os
from utils import string_utils, error_rates
import time
import random
import yaml

from utils.dataset_parse import load_file_list
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys

print(sys.executable)
import torch

print(torch.__file__)
print("CUDA Available", torch.cuda.is_available())
from torch.utils import collect_env

print(collect_env.main())
print("------")
train_loss = []
valid_loss = []
ctc_loss = []

LOAD_HW = False

# os.environ["CUDA_VISIBLE_DEVICES"]= "1"


with open("sample_config.yaml") as f:
    config = yaml.load(f, Loader=yaml.loader.SafeLoader)

print(config["network"]["hw"]['char_set_path'])

hw_network_config = config['network']['hw']
pretrain_config = config['pretraining']

char_set_path = hw_network_config['char_set_path']

with open(char_set_path) as f:
    char_set = json.load(f)

idx_to_char = {}
for k, v in char_set['idx_to_char'].items():
    idx_to_char[int(k)] = v
print(len(idx_to_char))

config["network"]["hw"]["num_of_outputs"] = len(idx_to_char) + 1
print(config['network']['hw'])

training_set_list = load_file_list(pretrain_config['training_set'])
train_dataset = HwDataset(training_set_list,
                          char_set['char_to_idx'], augmentation=True,
                          img_height=hw_network_config['input_height'])

train_dataloader = DataLoader(train_dataset,
                              batch_size=pretrain_config['hw']['batch_size'],
                              shuffle=True, num_workers=0, drop_last=True,
                              collate_fn=hw_dataset.collate)

batches_per_epoch = int(pretrain_config['hw']['images_per_epoch'] / pretrain_config['hw']['batch_size'])
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

test_set_list = load_file_list(pretrain_config['validation_set'])
test_dataset = HwDataset(test_set_list,
                         char_set['char_to_idx'],
                         img_height=hw_network_config['input_height'])

test_dataloader = DataLoader(test_dataset,
                             batch_size=pretrain_config['hw']['batch_size'],
                             shuffle=False, num_workers=0,
                             collate_fn=hw_dataset.collate)

criterion = CTCLoss(blank=0, zero_infinity=True)

hw = cnn_lstm.create_model(hw_network_config)
hw.cuda()

if LOAD_HW:
    hw_path = os.path.join(pretrain_config['snapshot_path'], "hw.pt")
    hw_state = safe_load.torch_state(hw_path)
    hw.load_state_dict(hw_state)

# torch.autograd.set_detect_anomaly(True)

optimizer = torch.optim.Adam(hw.parameters(), lr=pretrain_config['hw']['learning_rate'])
dtype = torch.cuda.FloatTensor

lowest_loss = np.inf
cnt_since_last_improvement = 0
for epoch in range(1000):
    first = True
    print("Epoch", epoch)
    sum_loss = 0.0
    steps = 0.0
    hw.train()
    total_ctc_loss = 0.0
    count = 0
    for i, x in enumerate(train_dataloader):

        line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
        labels = Variable(x['labels'], requires_grad=False)
        ###MEhreen
        # print('....before', labels)
        # labels = torch.flip(labels, [0])
        # print('....aftr', labels)
        ###End mehreen
        label_lengths = Variable(x['label_lengths'], requires_grad=False)

        preds = hw(line_imgs).cpu()
        if torch.any(torch.isnan(line_imgs)):
            bd_line_imgs = line_imgs
            print('...NAN line_imgs: i, count', i, count)
        if torch.any(torch.isnan(preds)):
            bd_line_imgs = line_imgs
            print('...NAN preds: i, count', i, count)
            break
        output_batch = preds.permute(1, 0, 2)
        out = output_batch.data.cpu().numpy()

        for i, gt_line in enumerate(x['gt']):
            logits = out[i, ...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred, idx_to_char, False)
            cer = error_rates.cer(gt_line, pred_str)
            sum_loss += cer
            steps += 1

        batch_size = preds.size(1)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

        # print "before"
        loss = criterion(preds, labels, preds_size, label_lengths)
        # print('...count', count, 'loss', loss)
        total_ctc_loss += loss
        # print('preds', preds)
        # print('labels', labels)
        # print('ctc loss, len(preds), len(labels)', loss, preds.size(), labels.size())
        if torch.isnan(loss):
            print('...NAN Loss, count', count)
            # iimg = line_imgs.cpu()[0].permute(1, 2, 0)
            # iimg = (iimg+1)*128
            # print('....iimg', iimg.size())
            # plt.imshow(iimg)
            # plt.show()
            # print('NAN LOSS i', i)
            # print('preds.size', preds.size())
            # print('labels length', label_lengths)
            # print('preds', preds)
            # print('labels', labels)
            # print('line imgs', line_imgs)
            break
        # print "after"

        optimizer.zero_grad()
        if torch.any(torch.isnan(hw.cnn[0].weight)):
            print("NAN WEIGHT BEFORE BACKWARD")
        # if first:
        #    hw2 = pickle.loads(pickle.dumps(hw))
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(hw.parameters(), 1.0)
        optimizer.step()
        count = count + 1

    print('....ctc loss', total_ctc_loss)
    ### MEhreen add break for one ex
    # break
    print("Train Loss", sum_loss / steps)
    print("Real Epoch", train_dataloader.epoch)
    train_loss.append(sum_loss / steps)
    ctc_loss.append(total_ctc_loss.item())

    sum_loss = 0.0
    steps = 0.0
    hw.eval()

    for x in test_dataloader:
        with torch.no_grad():
            line_imgs = Variable(x['line_imgs'].type(dtype), requires_grad=False)
            labels = Variable(x['labels'], requires_grad=False)
            label_lengths = Variable(x['label_lengths'], requires_grad=False)

        preds = hw(line_imgs).cpu()

        output_batch = preds.permute(1, 0, 2)
        out = output_batch.data.cpu().numpy()

        for i, gt_line in enumerate(x['gt']):
            logits = out[i, ...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred, idx_to_char, False)
            cer = error_rates.cer(gt_line, pred_str)
            sum_loss += cer
            steps += 1

    cnt_since_last_improvement += 1
    if lowest_loss > sum_loss / steps:
        cnt_since_last_improvement = 0
        lowest_loss = sum_loss / steps
        print("Saving Best")

        if not os.path.exists(pretrain_config['snapshot_path']):
            os.makedirs(pretrain_config['snapshot_path'])

        torch.save(hw.state_dict(), os.path.join(pretrain_config['snapshot_path'], 'hw.pt'))

    print("Test Loss", sum_loss / steps, lowest_loss)
    valid_loss.append(sum_loss / steps)

    if cnt_since_last_improvement >= pretrain_config['hw']['stop_after_no_improvement'] and lowest_loss < 0.9:
        break
print('Done ...')

# In[ ]:


# In[ ]:


import csv

train_stat = {'train_loss': train_loss, 'ctc_loss': ctc_loss, 'valid_loss': valid_loss}

strm = csv.writer(open("pretrain_hw_log.csv", "a"))
for key, val in train_stat.items():
    strm.writerow([key, val])
