#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by Collin Song on 2023/1/31.
# 初始化及一般方法
import json
import os
import zipfile

import numpy
from PIL import Image

import constants as const

# 解压训练数据集, filename 不含后缀
def unzip_data(file_name, src_dir, dest_dir):
    # 不存在同名文件夹时进行解压
    if not os.path.isdir(dest_dir + file_name):
        curr_zip = zipfile.ZipFile(src_dir + file_name + ".zip", 'r')
        curr_zip.extractall(path=dest_dir)
        curr_zip.close()

# 生成训练集数据列表
def generate_train_list(training_dataset, train_list_path):
    # 用于存放训练集中所有类别的信息
    class_detail = {}
    # 获取所有类别保存的文件夹名称
    class_dirs = os.listdir(training_dataset)
    # 总的图像数量
    all_class_images = 0
    # 存放类别标签
    class_label = 0
    # 存放类别数目
    class_dim = 0
    # 存放要写进 train.txt 的内容
    trainer_list = []
    # 读取每个类别
    for class_dir in class_dirs:
        if class_dir != '.DS_Store':
            class_dim += 1
            # 每个类别的信息
            class_detail_list = {}
            trainer_sum = 0
            # 统计每个类别有多少照片
            class_sum = 0
            # 获取类别路径
            path_for_single_class = training_dataset + class_dir  # 当前是获取的是当前类别的路径
            # 获取当前类别下所有个人路径
            path_for_single_students = os.listdir(path_for_single_class)
            # 对当前路径下的单个人的路径进行照片读取
            for path_for_single_student in path_for_single_students:
                if path_for_single_student != '.DS_Store':
                    # 获取当前个人路径下的所有照片
                    path = path_for_single_class + '/' + path_for_single_student
                    img_paths = os.listdir(path)
                    for img_path in img_paths:
                        name_path = path + '/' + img_path  # 每张照片的路径
                        trainer_list.append(name_path + "\t%d" % class_label + "\n")  # trainer_sum 测试数据的数目
                        trainer_sum += 1
                        class_sum += 1  # 每类图片的数目
                        all_class_images += 1  # 所有类图片的数目

            # 用于说明的 json 文件的 class_detail 数据
            class_detail_list['class_name'] = class_dir  # 类别名称
            class_detail_list['class_label'] = class_label  # 类别标签
            class_detail_list['class_trainer_images'] = trainer_sum  # 该类数据的训练集项目

            # 记录标签
            class_detail[class_label] = class_detail_list

            class_label += 1

    with open(train_list_path, 'a') as f_train:
        for train_image in trainer_list:
            f_train.write(train_image)

    # 训练数据集信息
    read_json = {'all_class_name': training_dataset, 'all_class_images': all_class_images, 'all_class_dim': class_dim,
                 'class_detail': class_detail}
    jsons = json.dumps(read_json, sort_keys=True, indent=4, separators=(',', ': '))
    with open(const.MODEL_INFO_PATH, 'w') as f_json:
        f_json.write(jsons)
    print("生成训练集数据列表完成!")

# 此函数用于自定义 reader
def custom_reader(file_list):
    def reader():
        with open(file_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((64, 64), Image.BILINEAR)
                img = numpy.array(img).astype('float32')
                img = img.transpose((2, 0, 1))  # HWC to CHW
                img = img / 255  # 像素值归一化
                yield img, int(lab)

    return reader

# 生成验证数据列表
def generate_eval_list(target_path, eval_list_path):
    # 存放要写入eval.txt的内容
    eval_list = []
    # 获取所有类别保存的文件夹名称
    class_dirs = os.listdir(target_path)  # val文件夹下一级所有文件信息
    # 读取所有类别
    for class_dir in class_dirs:
        if class_dir != '.DS.Store':
            path = target_path + class_dir  # 获取val文件夹下每个人的路径信息
            # 对当前人的所有照片进行遍历
            img_paths = os.listdir(path)
            for img_path in img_paths:
                name_path = path + '/' + img_path
                eval_list.append(name_path + "\n")
            # 每次验证集的数据列表只会存储当前这个人的所有照片信息
            # 因此不采用'a'追加的方式打开文件, 而是以'w'写的方式打开, 这样每次都会覆盖上一个人的路径
            with open(eval_list_path, 'w') as f_eval:
                for eval_image in eval_list:
                    f_eval.write(eval_image)

# 此函数用于定义 eval_reader
def generate_eval_reader(file_list):
    def reader():
        with open(file_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path = line.strip()
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((64, 64), Image.BILINEAR)
                img = numpy.array(img).astype('float32')
                img = img.transpose((2, 0, 1))  # HWC to CHW
                img = img / 255  # 像素值归一化
                yield img

    return reader
