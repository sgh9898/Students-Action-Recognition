#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by Collin Song on 2023/2/1.
# 使用模型进行预测
import os

import paddle

import init as init
import constants as const
import cnn_model

# 将 zip 训练数据集解压到目标路径
init.unzip_data(const.CURR_EVAL_DIR_NAME, const.ZIPPED_EVAL_DATA_DIR, const.DATASET_DIR + "/eval/")
# 每次生成数据列表之前, 首先清空 eval.txt
with open(const.EVAL_TEXT_PATH, 'w') as f_eval:
    f_eval.seek(0)
    f_eval.truncate()

# 构造验证集数据提供器
result_txt_path = const.RESULT_TEXT_PATH
# 获取所有类别保存的文件夹名称
class_dirs = os.listdir(const.EVAL_DATA_DIR)  # eval 文件夹下一级所有文件信息
people_number = 0  # eval 文件夹下第几个人

# 加载模型
# 载入模型参数、优化器参数和最后一个epoch保存的检查点
layer_state_dict = paddle.load(const.PROJECT_PATH + "/model/cnn_layer.pdparams")
opt_state_dict = paddle.load(const.PROJECT_PATH + "/model/adam.pdopt")

# 将load后的参数与模型关联起来
model = cnn_model.CNN()
optimizer = paddle.optimizer.Adam(learning_rate=const.LEARNING_STRATEGY['lr'], parameters=model.parameters())

model.set_state_dict(layer_state_dict)
optimizer.set_state_dict(opt_state_dict)

# 读取所有的类别
for class_dir in class_dirs:
    if class_dir != '.DS.Store':
        # 每次构造一个新的数据提供器, 只提供当前人所有照片信息
        people_number += 1
        init.generate_eval_list(const.EVAL_DATA_DIR, const.EVAL_TEXT_PATH)  # 调用子函数得到当前人的所有照片路径数据
        eval_reader = paddle.batch(init.generate_eval_reader(const.EVAL_TEXT_PATH),
                                   batch_size=60,
                                   drop_last=False)  # 每个人的数据照片大概在35~40张, 但是最大的达到51张, 设定为每个人取60张照片, 设定为False 便于让不是batch_size尺寸的不会被丢弃, 能够一次性输入当前人物的所有照片
        # 当前人的照片输入网络模型进行预测
        cnn_model.predict(model, eval_reader, people_number, const.RESULT_TEXT_PATH)
