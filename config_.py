#! /usr/bin/env python
# coding=utf-8

from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

__C.data_file = "./data/cn_train.txt" # 原文
__C.data_file_v2 = "./data/cn_train_v2.txt" # 原文
__C.batch_size = 4
__C.epoch = 10
__C.lr_rate = 1e-3
