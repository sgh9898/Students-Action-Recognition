#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by Collin Song on 2023/1/31.
# 训练模型
import json
from collections import Counter

import numpy
import paddle
import paddle.nn.functional as fun
from matplotlib import pyplot as plt

import constants as const

class CNN(paddle.nn.Layer):
    """
    模型配置
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=128 * 15 * 15, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=80)
        self.linear3 = paddle.nn.Linear(in_features=80, out_features=5)

    # 定义网络前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        x = fun.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = fun.relu(x)
        x = self.pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = fun.relu(x)
        x = self.linear2(x)
        x = fun.relu(x)
        x = self.linear3(x)
        return x

# 展示训练过程
def draw_train_process(title, iters, costs, accs, label_cost, lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("loss/acc", fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.plot(iters, accs, color='green', label=lable_acc)
    plt.legend()
    plt.grid()
    plt.show()

# 展示过程
def draw_process(title, color, iters, data, label):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(iters, data, color=color, label=label)
    plt.legend()
    plt.grid()
    plt.show()

# 使用模型进行分类
def predict(model, eval_reader, people_number, result_txt_path):
    """
    对于每个人的所有图片数据, 对 CNN 模型改进为：
        1. 判断该人的所有图片, 对该人文件夹内的所有图片进行类别划分, 并将对应的数字化类别存放于列表中
        2. 对列表进行处理, 列表中种类最多的类别便是当前人所对应的行为类别
        3. 新建文本文档, 将当前人的序号与行为类别一一对应存入.txt中
    """
    model.eval()

    with open(const.MODEL_INFO_PATH, 'r') as f_model_info:
        model_info = json.load(f_model_info)
        label_dict = model_info["class_detail"]

    for batch_id, data in enumerate(eval_reader()):
        x_data = numpy.array([item[0] for item in data], dtype='float32').reshape(-1, 3, 64, 64)

        x_data = paddle.to_tensor(x_data, dtype='float32', place=const.PADDLE_PLACE)  # 将numpy.array数据转换成tensor张量

        # 此时获得的预测值是一个n * 5的张量
        predicts = model(x_data)

        # 用于存放每张图片的预测类别对应的数字标签
        # 预测值的每行5个值, 代表一张照片分别属于5个类别的概率值
        # 找到每一行最大的数所对应的索引值, 也就是识别类别对应的数字标签
        # result_list 表示当前人的每一张照片的预测类别
        result_list = numpy.argmax(predicts.numpy(), axis=1)

        # result 为预测类别列表中出现次数最多的类别
        result = Counter(result_list).most_common(1)[0][0]
        with open(result_txt_path, 'a') as f_result:
            f_result.write(str(people_number) + "\t" + label_dict[str(result)]["class_name"] + "\n")
