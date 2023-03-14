# 目录

<!-- TOC -->

- [目录](#目录)
- [vit_base描述](#vit_base描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [ImageNet-1k上的vit_base](#ImageNet-1k上的vit_base)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# vit_base描述

Transformer架构已广泛应用于自然语言处理领域。本模型的作者发现，Vision Transformer（ViT）模型在计算机视觉领域中对CNN的依赖不是必需的，直接将其应用于图像块序列来进行图像分类时，也能得到和目前卷积网络相媲美的准确率。

[论文](https://arxiv.org/pdf/2010.11929v2.pdf) ：Dosovitskiy, A. , Beyer, L. , Kolesnikov, A. , Weissenborn, D. , & Houlsby, N.. (2020). An image is worth 16x16 words: transformers for image recognition at scale.

# 模型架构

Vision Transformer（ViT）的总体网络架构如下： [链接](https://arxiv.org/pdf/2010.11929v2.pdf)

# 数据集

使用的数据集：ImageNet-1k

- 数据集大小：125G，共1000个类、125万张彩色图像
    - 训练集：120G，共120万张图像
    - 测试集：5G，共5万张图像
- 数据格式：RGB
    - 注：数据将在src/dataset.py中处理。

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/enable_mixed_precision.html) 的训练方法，使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

# 环境要求

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```python
  # 运行训练示例
  python train.py --device_id=0 > train.log 2>&1 &

  # 运行分布式训练示例
  bash ./scripts/run_train.sh [RANK_TABLE_FILE] imagenet

  # 运行评估示例
  python eval.py --checkpoint_path ./ckpt_0 > ./eval.log 2>&1 &
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.>

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                    // 所有模型相关说明
    ├── vit_base
        ├── README_CN.md             // vit_base相关说明
        ├── scripts
        │   ├──run_eval.sh           // Ascend评估的shell脚本
        │   ├──run_train.sh          // 分布式到Ascend的shell脚本
        ├── src
        │   ├──config.py             // 参数配置
        │   ├──configs.py            // 不同架构的ViT
        │   ├──dataset.py            // 创建数据集
        │   ├──modeling_ms.py        // vit_base架构
        ├── eval.py                  // 评估脚本
        ├── train.py                 // 训练脚本
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置vit_base和ImageNet-1k数据集。

  ```python
  'name':'imagenet'        # 数据集
  'pre_trained':'False'    # 是否基于预训练模型训练
  'num_classes':1000       # 数据集类数
  'lr_init':0.02           # 初始学习率，单卡训练时设置为0.02，八卡并行训练时设置为0.18
  'batch_size':128         # 训练批次大小
  'epoch_size':160         # 总计训练epoch数
  'momentum':0.9           # 动量
  'weight_decay':1e-4      # 权重衰减值
  'image_height':224       # 输入到模型的图像高度
  'image_width':224        # 输入到模型的图像宽度
  'data_path':'/data/ILSVRC2012_train/'  # 训练数据集的绝对全路径
  'val_data_path':'/data/ILSVRC2012_val/'  # 评估数据集的绝对全路径
  'device_target':'Ascend' # 运行设备
  'device_id':0            # 用于训练或评估数据集的设备ID使用run_train.sh进行分布式训练时可以忽略。
  'keep_checkpoint_max':30 # 最多保存40个ckpt模型文件
  'checkpoint_path':'./ckpt_0/train_vit_imagenet-156_10009.ckpt'  # checkpoint文件保存的绝对全路径
  ```

更多配置细节请参考脚本`config.py`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py --device_id=0 > train.log 2>&1 &
  ```

  上述python命令将在后台运行，可以通过生成的train.log文件查看结果。

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash ./scripts/run_train.sh [RANK_TABLE_FILE] imagenet
  ```

  上述shell脚本将在后台运行分布训练。

## 评估过程

### 评估

- 在Ascend环境运行时评估ImageNet-1k数据集

  “./ckpt_0”是保存了训练好的.ckpt模型文件的目录。

  ```bash
  python eval.py --checkpoint_path ./ckpt_0 > ./eval.log 2>&1 &
  OR
  bash ./scripts/run_eval.sh
  ```

# 模型描述

## 性能

### 评估性能

#### ImageNet-1k上的vit_base

| 参数                 | Ascend                                                       |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | vit_base(ViT-B/16)                                                |
| 资源                   | Ascend 910               |
| 上传日期              | 2021-08-01                                 |
| MindSpore版本          | 1.2.0                                                 |
| 数据集                    | ImageNet-1k，5万张图像                                                |
| 训练参数        | epoch=160, batch_size=128, lr_init=0.02（单卡为0.02,八卡为0.18）               |
| 优化器                  | Momentum                                                    |
| 损失函数              | Softmax交叉熵                                       |
| 输出                    | 概率                                                 |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。