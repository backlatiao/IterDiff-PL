# IterDiff-PL

血管分割项目（TransUNet + Diffusion refinement）。


```text
IterDiff-PL/
├─ model/                       # 模型定义
├─ dataset/                     # 数据处理
│  ├─ image_process.py
│  └─ prepare_diffusion_data.py
└─ predict/                     # 推理预测
│  ├─ generate_transunet_predictions.py
│  └─ test_diffusion_patch.py
├─ semi-supervised.py           #半监督训练
├─ train_diffusion.py           #扩散模型训
├─ train_transunet.py           #transunet模型训练
├─ .gitignore
└─ README.md
```
