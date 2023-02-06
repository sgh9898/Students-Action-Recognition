#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by Collin Song on 2023/1/31.
# 模型训练与存储
import json
import sys

import numpy
import paddle
import paddle.nn.functional as func

import init
import constants as const
import cnn_model as cnn_model

# 存放导入的包
sys.path.append(const.LIBRARY_PATH)

# ------------------------- 初始化 -------------------------
# 将 zip 训练数据集解压到目标路径
init.unzip_data(const.CURR_TRAIN_DIR_NAME, const.ZIPPED_TRAIN_DATA_DIR, const.DATASET_DIR + "/train/")

# 每次生成数据列表之前, 首先清空 train.txt
with open(const.TRAIN_TEXT_PATH, 'w') as f_train:
    f_train.seek(0)
    f_train.truncate()

# 生成训练集数据列表
init.generate_train_list(const.TRAIN_DATA_DIR, const.TRAIN_TEXT_PATH)

# 构造训练集数据提供器
train_reader = paddle.batch(init.custom_reader(const.TRAIN_TEXT_PATH),
                            batch_size=const.TRAIN_BATCH_SIZE,
                            drop_last=True)

# 模型配置部分
all_train_iter = 0
all_train_iters = []
all_train_costs = []
all_train_accs = []

# ------------------------- 模型训练与评估 -------------------------
with open(const.MODEL_INFO_PATH, 'r') as f_model_info:
    model_info = json.load(f_model_info)
print("分类总数: " + str(model_info["all_class_dim"]))
print("分类明细: " + str(model_info["class_detail"].values))

model = cnn_model.CNN()
optimizer = paddle.optimizer.Adam(learning_rate=const.LEARNING_STRATEGY['lr'], parameters=model.parameters())
model.train()
print("模型训练开始! ")
for epoch_num in range(const.NUM_EPOCHS):
    for batch_id, data in enumerate(train_reader()):

        # 将元组转换成 numpy.array
        x_data = numpy.array([item[0] for item in data], dtype='float32').reshape(-1, 3, 64, 64)
        # 将 numpy.array 数据转换成 tensor 张量
        x_data = paddle.to_tensor(x_data, dtype='float32', place=const.PADDLE_PLACE)

        y_data = numpy.array([item[1] for item in data], dtype='int64').reshape(-1, 1)
        y_data = paddle.to_tensor(y_data, dtype='int64', place=const.PADDLE_PLACE)

        # print("y的数据类型： ", type(y_data))
        # print("x_data的数据类型：", type(x_data))
        predicts = model(x_data)
        # print("预测值的数据类型： ", type(predicts), "\n", "predicts内容： ", predicts)

        loss = func.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        loss.backward()

        all_train_iter = all_train_iter + const.TRAIN_BATCH_SIZE
        all_train_iters.append(all_train_iter)
        all_train_costs.append(loss.numpy()[0])
        all_train_accs.append(acc.numpy()[0])

        if batch_id % 1 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}, accuracy is: {}".format(epoch_num, batch_id, loss.numpy(), acc.numpy()))
        optimizer.step()
        optimizer.clear_grad()

cnn_model.draw_train_process("training", all_train_iters, all_train_costs, all_train_accs, "trainning loss", "trainning acc")
cnn_model.draw_process("trainning loss", "red", all_train_iters, all_train_costs, "trainning loss")
cnn_model.draw_process("trainning acc", "green", all_train_iters, all_train_accs, "trainning acc")

# ------------------------- 保存训练好的模型 -------------------------
# 保存Layer参数
paddle.save(model.state_dict(), const.PROJECT_PATH + "/model/cnn_layer.pdparams")
# 保存优化器参数
paddle.save(optimizer.state_dict(), const.PROJECT_PATH + "/model/adam.pdopt")
# 保存检查点checkpoint信息
# paddle.save(final_checkpoint, const.PROJECT_PATH + "/model/final_checkpoint.pkl")
