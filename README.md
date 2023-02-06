# 课堂行为识别

> * 源码来自飞桨开源项目:[课堂行为识别检测](https://aistudio.baidu.com/aistudio/projectdetail/3180141?channelType=0&channel=0), 并在此基础上修复了bug
> * 项目推荐使用 GPU 版本 Paddle, 运行速度远高于 CPU 版本, 尤其是模型训练过程
> * 存在多版本 python 与 pip 时, 命令中可能需要使用 python3 及 pip3 作为替代
---

## 项目配置说明

### 1. 根据操作系统与训练设备安装 Paddle

* 安装 GPU 版本 Paddle:  
  * 前置需求: 安装对应版本的 CUDA, cuDNN, TensorRT(如需使用 PaddleTensorRT 推理)
    .详情参见 [GPU Paddle 前置需求](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html#old-version-anchor-4-首先请您选择您的版本)
  * Linux 下安装 GPU 版本 Paddle

    ```
    python -m pip install paddlepaddle-gpu==2.4.1.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
    ```
  * Windows 下安装 GPU 版本 Paddle

      ```
      python -m pip install paddlepaddle-gpu==2.4.1.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
      ```
* 无法使用 GPU 版本时可以选择 CPU 版本 Paddle, 但图像处理速度会明显降低
  * Linux 下安装 CPU 版本 Paddle
    ```
    python -m pip install paddlepaddle==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```
  * Windows 下安装 CPU 版本 Paddle

    ```
    python -m pip install paddlepaddle==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

* 更多信息请参照 [官方说明](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)

### 2. 数据及参数配置
1. 上传 zip 数据集: 训练数据集上传至 /data/zipped-train, 预测分析数据集上传至 /data/zipped-eval. 上传后需修改 constants.py 中对应数据集名称
2. 参数配置均位于 constants.py 中, 可根据需求自行调整

### 3. 使用模型进行数据分析
1. 当前项目下, 导入 requirements.txt (仅初次部署需要)

    ```
    pip install -r requirements.txt
    ```
    如速度慢可以使用阿里云镜像

    ```
    pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
    ```
2. 运行 main.py, 分析结果为 result.txt, 路径可在 constants.py 中进行配置
