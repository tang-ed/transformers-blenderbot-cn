#! /usr/bin/env python
# coding=utf-8

from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

__C.data_file = "./data/cn_train.txt" # 原文
__C.data_file_v2 = "./data/cn_train_v2.txt" # 原文
__C.batch_size = 2 # 用预训练模型进行训练时建议是8或者4，不用预训练模型训练建议是16或者32
__C.epoch = 100
__C.lr_rate = 2e-5
