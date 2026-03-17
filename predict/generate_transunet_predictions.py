"""
预先生成TransUNet预测结果并保存
这样在训练Diffusion模型时就不需要每次都重复预测
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.transunet import TransUNet


class BloodVesselDataset(Dataset):
    """血管分割数据集"""

    def __init__(self, image_dir, mask_dir, img_size=(512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.mask_preprocess = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.preprocess(image)
        mask = self.mask_preprocess(mask)
        mask = (mask > 0.5).float()

        return image, mask, img_name


def load_transunet_model(model_path, device):
    """加载预训练的TransUNet模型"""
    model = TransUNet(
        img_size=512,
        patch_size=16,
        in_channels=3,
        out_channels=1,
        embed_dim=512,
        num_heads=8,
        mlp_dim=2048,
        num_layers=6,
        features=[64, 128, 256, 512],
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def generate_predictions(model, dataloader, output_dir, device):
    """生成并保存TransUNet预测结果"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"开始生成预测结果，保存到: {output_dir}")

    with torch.no_grad():
        for images, _, img_names in tqdm(dataloader, desc="生成预测"):
            images = images.to(device)

            # TransUNet预测（已包含sigmoid）
            predictions = model(images)

            # 保存预测结果
            for i, img_name in enumerate(img_names):
                pred = predictions[i].cpu().numpy()

                # 保存为numpy格式（保留连续值，不二值化）
                save_path = os.path.join(output_dir, img_name.rsplit(".", 1)[0] + ".npy")
                np.save(save_path, pred)

    print(f"预测结果生成完成！共 {len(dataloader.dataset)} 张图像")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载TransUNet模型
    transunet_path = os.path.join(project_root, "weights", "best_model_0307.pth")
    if not os.path.exists(transunet_path):
        print(f"错误：找不到TransUNet权重文件: {transunet_path}")
        return

    print("加载TransUNet模型...")
    transunet_model = load_transunet_model(transunet_path, device)

    # 数据路径配置
    base_data_path = os.path.join(project_root, "blood-vessel-patch")
    train_image_dir = os.path.join(base_data_path, "train", "image")
    train_mask_dir = os.path.join(base_data_path, "train", "label")
    test_image_dir = os.path.join(base_data_path, "test", "image")
    test_mask_dir = os.path.join(base_data_path, "test", "label")

    # 创建数据集
    train_dataset = BloodVesselDataset(train_image_dir, train_mask_dir, img_size=(512, 512))
    test_dataset = BloodVesselDataset(test_image_dir, test_mask_dir, img_size=(512, 512))

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    # 生成训练集预测
    train_output_dir = os.path.join(project_root, "transunet_predictions", "train")
    generate_predictions(transunet_model, train_loader, train_output_dir, device)

    # 生成测试集预测
    test_output_dir = os.path.join(project_root, "transunet_predictions", "test")
    generate_predictions(transunet_model, test_loader, test_output_dir, device)

    print("\n所有预测结果已生成完成！")
    print(f"训练集预测: {train_output_dir}")
    print(f"测试集预测: {test_output_dir}")


if __name__ == "__main__":
    main()
