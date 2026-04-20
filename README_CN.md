# IterDiff-PL

IterDiff-PL 是一个面向视网膜血管分割任务的半监督学习框架，核心思想是利用扩散模型对粗伪标签进行结构精修，并通过不确定性加权蒸馏将精修能力固化到轻量化分割网络中。该框架适用于标注数据有限、细小血管难以恢复、血管结构连续性容易断裂等场景。

框架首先由分割主干网络生成粗血管概率图，再利用条件扩散模型对粗结果进行局部结构修复；随后通过多次采样估计软伪标签和像素级不确定性，并在不确定性加权监督下训练 Student 网络。推理阶段仅保留 Student 分割网络，无需扩散模型多次采样，从而兼顾分割性能与推理效率。

## 主要特点

- **可插拔的分割主干网络**  
  粗分割网络可以根据任务需要替换为不同的血管分割模型。当前代码中包含基于 TransUNet 的分割主干实现。

- **扩散模型伪标签精修**  
  扩散模块不是从纯噪声直接生成分割掩码，而是以粗分割结果为起点进行局部反向去噪，重点修复断裂血管、模糊边界和细小分支。

- **不确定性加权蒸馏**  
  通过多轮扩散采样得到软伪标签均值和像素级不确定性图，在 Student 训练时降低低置信区域的监督权重，减少错误伪标签传播。

- **高效推理**  
  扩散模型主要用于训练阶段的伪标签精修与知识增强。最终部署时，只需要 Student 网络单次前向传播即可完成血管分割。

- **灵活的数据适配能力**  
  代码可适配普通眼底图、超广角眼底图、切块训练、多分辨率混合训练等不同数据处理流程。

## 项目结构

```text
IterDiff-PL/
├─ model/
│  ├─ transunet.py              # 基于 TransUNet 的分割主干网络
│  ├─ diffusion_model.py        # 条件扩散精修模型
│  └─ diffusion_model_v2.py     # 扩展版扩散模型实现
├─ dataset/
│  ├─ image_process.py          # 图像切块与预处理
│  └─ prepare_diffusion_data.py # 构建扩散模型训练数据
├─ predict/
│  ├─ generate_transunet_predictions.py # 生成粗分割预测结果
│  └─ test_diffusion_patch.py           # 基于 patch 的扩散精修推理
├─ train_transunet.py           # 监督式分割主干训练脚本
├─ train_diffusion.py           # 扩散精修模型训练与测试入口
├─ semi-supervised.py           # Teacher-Diffusion-Student 半监督训练框架
├─ requirements.txt
├─ README_CN.md                 # 中文说明文档
└─ README.md
```

## 环境安装

建议使用 Python 3.10 环境：

```bash
conda create -n iterdiff-pl python=3.10
conda activate iterdiff-pl
pip install -r requirements.txt
```

当前 `requirements.txt` 默认使用 CUDA 11.8 对应的 PyTorch 版本。如果本机 CUDA 或 PyTorch 版本不同，请根据实际环境安装匹配版本的 `torch` 与 `torchvision`。

## 运行时数据组织方式

数据集目录、生成的伪标签、模型权重和输出结果默认不包含在仓库中，并已在 `.gitignore` 中忽略。运行实验时需要在本地自行准备或生成这些目录。

训练脚本默认要求图像和标注掩码按相同文件名成对存放。推荐的本地运行时目录结构如下：

```text
IterDiff-PL/
├─ blood-vessel/
│  ├─ train/
│  │  ├─ image/
│  │  └─ label/
│  └─ test/
│     ├─ image/
│     └─ label/
├─ blood-vessel-patch/
│  ├─ train/
│  │  ├─ image/
│  │  └─ label/
│  └─ test/
│     ├─ image/
│     └─ label/
├─ transunet_predictions/
│  ├─ train/
│  └─ test/
└─ weights/
```

图像可使用 `.png`、`.jpg` 或 `.jpeg` 格式。血管标注应为与原图同名的二值掩码。

使用前请根据实际数据集和权重路径，在本地准备对应目录。

## 使用流程

### 1. 训练粗分割主干网络

```bash
python train_transunet.py
```

该脚本会训练基于 TransUNet 的血管分割模型，并将模型权重保存到 `weights/` 或 `output/` 目录下。

### 2. 生成粗伪标签

训练完成分割主干网络后，或将已有预训练权重放入 `weights/` 后，运行：

```bash
python predict/generate_transunet_predictions.py
```

生成的粗分割概率图会以 `.npy` 格式保存到：

```text
transunet_predictions/train/
transunet_predictions/test/
```

### 3. 构建扩散模型训练数据

如需进行 patch 训练或混合分辨率训练，可以运行：

```bash
python dataset/prepare_diffusion_data.py
```

该脚本会将原图、真实标签和粗分割预测整理为扩散精修模型所需的数据格式。

### 4. 训练扩散精修模型

```bash
python train_diffusion.py
```

脚本会提供交互式菜单，可选择：

```text
1. 从头训练扩散模型
2. 从 checkpoint 恢复扩散模型训练
3. 使用 patch 推理方式测试扩散模型
4. 生成分割主干预测结果
5. 构建混合扩散训练数据
```

扩散模型权重默认保存到 `weights/` 目录。

### 5. 半监督 Teacher-Diffusion-Student 训练

半监督训练框架位于：

```bash
python semi-supervised.py
```

整体训练流程包括：

1. 使用有标签数据预训练 Student 分割网络。
2. 通过 EMA Teacher 生成稳定粗预测。
3. 使用扩散模型对粗预测进行多次采样精修，得到软伪标签和不确定性图。
4. 使用不确定性加权蒸馏损失训练 Student 网络。

推理阶段只需要 Student 网络，不依赖扩散模型。

## 方法概览

整体流程如下：

```text
输入眼底图像
   │
   ▼
粗分割主干网络
   │
   ▼
粗血管概率图
   │
   ▼
条件扩散精修模型
   │
   ├─ 多次随机采样
   ├─ 软伪标签估计
   └─ 不确定性估计
   │
   ▼
不确定性加权蒸馏
   │
   ▼
Student 分割网络
```

扩散模型在本框架中主要承担“结构精修器”的角色，而不是最终部署时的独立分割模型。这样的设计既利用了扩散模型在局部结构修复方面的能力，也保留了判别式分割网络的高效推理优势。

## 评价指标

代码中包含多种常用血管分割评价指标：

- Dice coefficient
- Intersection over Union, IoU
- Sensitivity / Recall
- Specificity
- Accuracy
- Precision
- AUC-ROC
- AUC-PR

这些指标可用于评估粗分割结果、扩散精修结果以及最终 Student 模型结果。

## 自定义使用

### 替换分割主干网络

框架不限制具体的粗分割模型。若需要接入其他血管分割网络，只需保证其输出为单通道血管概率图，并与扩散精修模块的数据接口保持一致。

### 适配新的数据集

使用新数据集时，可以优先按照默认目录结构组织图像和标签；也可以根据实际数据格式修改对应脚本中的 Dataset 类和路径配置。

### 调整图像尺寸

当前主要脚本默认使用 `512 x 512` 图像或 patch。对于高分辨率眼底图或超广角眼底图，建议使用 patch 预处理与拼接推理，以降低显存占用。

## 注意事项

- 扩散精修模型训练前需要先生成粗分割预测结果。
- 大尺寸图像建议使用 patch-based 训练和推理。
- 最终 Student 模型推理时不需要扩散采样，适合高效部署。
- 部分脚本中包含默认数据路径和权重路径，运行前请根据实际数据位置进行检查或修改。

## License

本项目主要用于学术研究与实验验证。使用公开数据集和第三方依赖时，请遵循对应数据集和软件包的许可协议。
