"""
数据预处理脚本：将原图、标签和预测切成patch块或resize到512x512
用于Diffusion模型训练
"""

from PIL import Image
import os
from tqdm import tqdm

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def split_image_with_prediction(
    image,
    label,
    prediction,
    cols,
    rows,
    overlap_percent=0.2,
    imageout='',
    labelout='',
    predout='',
    name='',
):
    """
    将大图像、标签和预测分割成有重叠的小图像块
    """
    img_width, img_height = image.size

    # 计算每个小图的宽度和高度（考虑重叠）
    overlap_width = int((img_width / cols) * overlap_percent)
    overlap_height = int((img_height / rows) * overlap_percent)

    small_width = img_width // cols + overlap_width
    small_height = img_height // rows + overlap_height

    # 创建输出文件夹
    os.makedirs(imageout, exist_ok=True)
    os.makedirs(labelout, exist_ok=True)
    os.makedirs(predout, exist_ok=True)

    # 切分图像
    for i in range(rows):
        for j in range(cols):
            # 计算起始坐标，考虑重叠
            start_x = max(0, j * (img_width // cols) - overlap_width // 2)
            start_y = max(0, i * (img_height // rows) - overlap_height // 2)

            # 确保不超出边界
            end_x = min(img_width, start_x + small_width)
            end_y = min(img_height, start_y + small_height)

            # 边界处尺寸不足时回退
            if end_x - start_x < small_width:
                start_x = max(0, end_x - small_width)
            if end_y - start_y < small_height:
                start_y = max(0, end_y - small_height)

            # 裁剪小图
            small_img = image.crop((start_x, start_y, end_x, end_y))
            small_label = label.crop((start_x, start_y, end_x, end_y))
            small_pred = prediction.crop((start_x, start_y, end_x, end_y))

            # 保存小图
            patch_name = f'crop_{name}_{i}_{j}.png'
            small_img.save(os.path.join(imageout, patch_name))
            small_label.save(os.path.join(labelout, patch_name))
            small_pred.save(os.path.join(predout, patch_name))

    return small_width, small_height


def resize_image_with_prediction(
    image,
    label,
    prediction,
    target_size=(512, 512),
    imageout='',
    labelout='',
    predout='',
    name='',
):
    """
    将图像、标签和预测resize到目标尺寸
    """
    # 创建输出文件夹
    os.makedirs(imageout, exist_ok=True)
    os.makedirs(labelout, exist_ok=True)
    os.makedirs(predout, exist_ok=True)

    # Resize
    resized_img = image.resize(target_size, Image.BILINEAR)
    resized_label = label.resize(target_size, Image.NEAREST)
    resized_pred = prediction.resize(target_size, Image.BILINEAR)

    # 保存
    resized_img.save(os.path.join(imageout, name))
    resized_label.save(os.path.join(labelout, name))
    resized_pred.save(os.path.join(predout, name))


def process_dataset_patch(
    image_dir,
    label_dir,
    pred_dir,
    output_image_dir,
    output_label_dir,
    output_pred_dir,
    cols=4,
    rows=4,
    overlap_percent=0.2,
):
    """
    处理数据集：切成patch块
    """
    print(f"开始处理数据集（切块模式）: {image_dir}")
    print(f"切块参数: {cols}x{rows}, 重叠率: {overlap_percent}")

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in tqdm(image_files, desc="切块处理"):
        # 加载图像、标签和预测
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file)
        pred_path = os.path.join(pred_dir, img_file)

        if not os.path.exists(label_path):
            print(f"警告: 找不到标签文件 {label_path}")
            continue
        if not os.path.exists(pred_path):
            print(f"警告: 找不到预测文件 {pred_path}")
            continue

        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        pred = Image.open(pred_path).convert('L')

        # 切块
        name = os.path.splitext(img_file)[0]
        split_image_with_prediction(
            img,
            label,
            pred,
            cols,
            rows,
            overlap_percent,
            output_image_dir,
            output_label_dir,
            output_pred_dir,
            name,
        )

    print(f"切块处理完成！输出目录: {output_image_dir}")


def process_dataset_resize(
    image_dir,
    label_dir,
    pred_dir,
    output_image_dir,
    output_label_dir,
    output_pred_dir,
    target_size=(512, 512),
):
    """
    处理数据集：resize到目标尺寸
    """
    print(f"开始处理数据集（resize模式）: {image_dir}")
    print(f"目标尺寸: {target_size}")

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in tqdm(image_files, desc="Resize处理"):
        # 加载图像、标签和预测
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file)
        pred_path = os.path.join(pred_dir, img_file)

        if not os.path.exists(label_path):
            print(f"警告: 找不到标签文件 {label_path}")
            continue
        if not os.path.exists(pred_path):
            print(f"警告: 找不到预测文件 {pred_path}")
            continue

        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        pred = Image.open(pred_path).convert('L')

        # Resize
        resize_image_with_prediction(
            img,
            label,
            pred,
            target_size,
            output_image_dir,
            output_label_dir,
            output_pred_dir,
            img_file,
        )

    print(f"Resize处理完成！输出目录: {output_image_dir}")


if __name__ == '__main__':
    print("=" * 60)
    print("数据预处理脚本 - Diffusion训练数据准备")
    print("=" * 60)

    # 原始数据路径
    base_image_dir = os.path.join(project_root, "blood-vessel")

    # 预测目录（优先使用默认目录）
    default_pred_dir = os.path.join(project_root, "transunet_predictions")
    legacy_pred_dir = os.path.join(project_root, "transunet_predictions_0124")
    base_pred_dir = default_pred_dir if os.path.exists(default_pred_dir) else legacy_pred_dir

    print(f"使用预测目录: {base_pred_dir}")

    # 处理训练集：patch + resize 合并
    train_image_dir = os.path.join(base_image_dir, "train", "image")
    train_label_dir = os.path.join(base_image_dir, "train", "label")
    train_pred_dir = os.path.join(base_pred_dir, "train")

    # 输出到统一训练目录
    train_output_base = os.path.join(project_root, "blood-vessel-diffusion-train")
    train_output_image = os.path.join(train_output_base, "image")
    train_output_label = os.path.join(train_output_base, "label")
    train_output_pred = os.path.join(train_output_base, "prediction")

    # 先处理patch数据
    print("\n--- 生成Patch数据 ---")
    process_dataset_patch(
        train_image_dir,
        train_label_dir,
        train_pred_dir,
        train_output_image,
        train_output_label,
        train_output_pred,
        cols=4,
        rows=4,
        overlap_percent=0.2,
    )

    # 再处理resize数据（追加到同一目录）
    print("\n--- 生成Resize数据 ---")
    process_dataset_resize(
        train_image_dir,
        train_label_dir,
        train_pred_dir,
        train_output_image,
        train_output_label,
        train_output_pred,
        target_size=(512, 512),
    )

    print("\n" + "=" * 60)
    print("训练数据处理完成")
    print("=" * 60)
    print(f"生成的训练数据集: {train_output_base}")
