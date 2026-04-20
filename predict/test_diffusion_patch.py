"""
测试扩散模型：patch切块预测 + 拼接成整图
"""

import os
import sys
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.diffusion_model_v2 import DiffusionNet, DiffusionPipeline


def split_image_to_patches(image, cols=4, rows=4):
    """
    将大图均匀切分为4x4的patch块（与TransUNet预测逻辑一致，不使用重叠）
    返回patches和位置信息
    """
    width, height = image.size
    patch_width = width // cols
    patch_height = height // rows

    patches = []
    positions = []

    for i in range(rows):
        for j in range(cols):
            left = j * patch_width
            top = i * patch_height
            right = left + patch_width
            bottom = top + patch_height

            # 裁剪patch
            patch = image.crop((left, top, right, bottom))
            patches.append(patch)
            positions.append((left, top, right, bottom, patch_width, patch_height))

    return patches, positions


def merge_patches_to_image(patches, positions, original_size):
    """
    将4x4的patch块拼接回完整图像（直接拼接，不使用加权平均）
    """
    width, height = original_size
    merged = np.zeros((height, width), dtype=np.float32)

    for patch, (left, top, right, bottom, patch_width, patch_height) in zip(patches, positions):
        # 确保patch尺寸与目标区域一致
        if patch.shape != (patch_height, patch_width):
            patch_img = Image.fromarray((patch * 255).astype(np.uint8), mode="L")
            patch_img = patch_img.resize((patch_width, patch_height), Image.BILINEAR)
            patch = np.array(patch_img, dtype=np.float32) / 255.0

        merged[top:bottom, left:right] = patch

    return merged


def test_diffusion_model_with_patches(
    model_path,
    test_image_dir,
    test_label_dir,
    test_pred_dir,
    output_dir,
    cols=4,
    rows=4,
    device="cuda",
):
    """
    测试扩散模型：patch切块预测 + 拼接成整图
    """
    print(f"使用设备: {device}")

    # 初始化模型
    diffusion_model = DiffusionNet(img_size=512, in_channels=5, out_channels=1).to(device)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_keys = set(diffusion_model.state_dict().keys())
        state_dict_filtered = {k: v for k, v in checkpoint["model_state_dict"].items() if k in model_keys}
        diffusion_model.load_state_dict(state_dict_filtered, strict=False)
        print(f"成功加载模型权重: {model_path}")
    else:
        model_keys = set(diffusion_model.state_dict().keys())
        state_dict_filtered = {k: v for k, v in checkpoint.items() if k in model_keys}
        diffusion_model.load_state_dict(state_dict_filtered, strict=False)
        print(f"成功加载模型权重（旧格式）: {model_path}")

    diffusion_model.eval()
    diffusion_pipeline = DiffusionPipeline(diffusion_model, device=device)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 数据预处理（与训练时一致：resize到512x512）
    image_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    mask_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    # 获取测试图像列表
    image_files = [f for f in os.listdir(test_image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    print(f"测试集样本数: {len(image_files)}")
    print(f"切块参数: {cols}x{rows}（均匀切分，不重叠）")

    print("开始patch推理测试...")
    with torch.no_grad():
        for img_file in tqdm(image_files, desc="Testing"):
            # 加载原图、标签和预测
            img_path = os.path.join(test_image_dir, img_file)
            label_path = os.path.join(test_label_dir, img_file)
            pred_path = os.path.join(test_pred_dir, img_file)

            if not os.path.exists(label_path) or not os.path.exists(pred_path):
                print(f"跳过 {img_file}：缺少标签或预测文件")
                continue

            original_image = Image.open(img_path).convert("RGB")
            original_label = Image.open(label_path).convert("L")
            original_pred = Image.open(pred_path).convert("L")
            original_size = original_image.size

            # 切patch
            image_patches, positions = split_image_to_patches(original_image, cols, rows)
            pred_patches, _ = split_image_to_patches(original_pred, cols, rows)

            # 对每个patch进行推理
            refined_patches = []
            for img_patch, pred_patch, (_, _, _, _, patch_width, patch_height) in zip(
                image_patches, pred_patches, positions
            ):
                # Resize到512x512并预处理
                img_tensor = image_transform(img_patch).unsqueeze(0).to(device)
                pred_tensor = mask_transform(pred_patch).unsqueeze(0).to(device)

                # Diffusion推理
                refined = diffusion_pipeline.sample(
                    pred_tensor,
                    img_tensor,
                    num_inference_steps=20,
                    refine_step=100,
                )

                # 将512x512结果resize回原patch尺寸
                refined_np = refined.squeeze().cpu().numpy()
                refined_patch_img = Image.fromarray((refined_np * 255).astype(np.uint8), mode="L")
                refined_patch_img = refined_patch_img.resize((patch_width, patch_height), Image.BILINEAR)
                refined_patch_np = np.array(refined_patch_img, dtype=np.float32) / 255.0
                refined_patches.append(refined_patch_np)

            # 拼接回整图
            refined_full = merge_patches_to_image(refined_patches, positions, original_size)

            # 二值化后保存
            refined_full_bin = (refined_full > 0.5).astype(np.uint8) * 255
            refined_image = Image.fromarray(refined_full_bin, mode="L")

            # 确保输出尺寸与原label一致
            if refined_image.size != original_label.size:
                refined_image = refined_image.resize(original_label.size, Image.NEAREST)

            refined_image.save(os.path.join(output_dir, img_file))

    print(f"\n预测结果已保存至: {output_dir}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型路径
    model_path = os.path.join(project_root, "weights", "diffusion_model_best_last.pth")
    if not os.path.exists(model_path):
        print("错误：找不到训练好的模型")
        sys.exit(1)

    # 测试集数据路径
    test_image_dir = os.path.join(project_root, "blood-vessel", "test", "image")
    test_label_dir = os.path.join(project_root, "blood-vessel", "test", "label")
    test_pred_dir = os.path.join(project_root, "transunet_predictions_last", "test")

    # 输出目录
    test_output_dir = os.path.join(project_root, "output", "diffusion_test_result_999")

    # 执行测试
    test_diffusion_model_with_patches(
        model_path,
        test_image_dir,
        test_label_dir,
        test_pred_dir,
        output_dir=test_output_dir,
        cols=4,
        rows=4,
        device=device,
    )
