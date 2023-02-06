#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by Collin Song on 2023/1/31.
# 参数类
import os

import paddle

# ------------------------- 可调参数 -------------------------
CURR_TRAIN_DIR_NAME = "students"  # 上传的 zip 训练数据集(名称不包括后缀)
CURR_EVAL_DIR_NAME = "students"  # 上传的 zip 验证数据集(名称不包括后缀)

PADDLE_PLACE = paddle.CPUPlace()  # 设备配置: paddle.CUDAPinnedPlace() 使用 GPU(推荐), paddle.CPUPlace() 使用 CPU

# 训练参数
INPUT_SIZE = [3, 64, 64]  # 输入图片的shape，三通道，64*64的图片
CLASS_DIM = -1  # 分类数
NUM_EPOCHS = 20  # 训练轮数
TRAIN_BATCH_SIZE = 64  # 训练时每个批次的大小
LEARNING_STRATEGY = {  # 优化函数相关的配置
    "lr": 0.0001  # 超参数学习率
}

# ------------------------- 默认参数 -------------------------
# 路径
PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))  # 项目路径
LIBRARY_PATH = PROJECT_PATH + "/external-libraries"  # 外来包的安装路径

# 数据集
DATASET_DIR = PROJECT_PATH + "/data"  # 数据集
TRAIN_DATA_DIR = DATASET_DIR + "/train/" + CURR_TRAIN_DIR_NAME + "/"  # 存放训练数据集的文件夹
EVAL_DATA_DIR = DATASET_DIR + "/eval/" + CURR_EVAL_DIR_NAME + "/"  # 存放验证数据集的文件夹
ZIPPED_TRAIN_DATA_DIR = DATASET_DIR + "/zipped-train/"  # 存放 zip 训练数据集的文件夹
ZIPPED_EVAL_DATA_DIR = DATASET_DIR + "/zipped-eval/"  # 存放 zip 验证数据集的文件夹

# 输出
MODEL_INFO_PATH = DATASET_DIR + "/model_info.json"  # 模型训练集信息
TRAIN_TEXT_PATH = DATASET_DIR + "/train.txt"  # 训练过程
EVAL_TEXT_PATH = DATASET_DIR + "/eval.txt"  # 预测过程
RESULT_TEXT_PATH = DATASET_DIR + "/result.txt"  # 预测结果
