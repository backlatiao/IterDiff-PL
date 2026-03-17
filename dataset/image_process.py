from PIL import Image
import os

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def split_image(image, label, cols, rows, overlap_percent=0.2, imageout='', labelout='', name=''):
    """
    将大图像分割成有重叠的小图像块
    :param image: 输入图像
    :param label: 对应标签图像
    :param cols: 列数
    :param rows: 行数
    :param overlap_percent: 重叠百分比，默认0.2 (20%)
    :param imageout: 输出图像路径
    :param labelout: 输出标签路径
    :param name: 图像名称
    :return: 小图像的尺寸
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

    # 切分图像
    for i in range(rows):
        for j in range(cols):
            # 计算起始坐标，考虑重叠
            start_x = max(0, j * (img_width // cols) - overlap_width // 2)
            start_y = max(0, i * (img_height // rows) - overlap_height // 2)

            # 确保不超出边界
            end_x = min(img_width, start_x + small_width)
            end_y = min(img_height, start_y + small_height)

            # 若边界处尺寸不足，回退起点
            if end_x - start_x < small_width:
                start_x = max(0, end_x - small_width)
            if end_y - start_y < small_height:
                start_y = max(0, end_y - small_height)

            # 裁剪小图
            small_img = image.crop((start_x, start_y, end_x, end_y))
            small_label = label.crop((start_x, start_y, end_x, end_y))

            # 保存小图
            small_img.save(os.path.join(imageout, f'crop_{name}_{i}_{j}.png'))
            small_label.save(os.path.join(labelout, f'crop_{name}_{i}_{j}.png'))

    return small_width, small_height


def process(image_orig, label_orig, image_out, label_out, cols=2, rows=2, overlap_percent=0.2):
    i = 1
    for pathi in os.listdir(image_orig):
        print(f"处理第 {i} 张图像")
        img = Image.open(os.path.join(image_orig, pathi))
        label = Image.open(os.path.join(label_orig, pathi))
        small_width, small_height = split_image(
            img,
            label,
            cols,
            rows,
            overlap_percent,
            image_out,
            label_out,
            pathi[:-4],
        )
        i = i + 1
    print("处理后的图像大小", small_height, small_width)


if __name__ == '__main__':
    # 本地路径设置
    image_orig = os.path.join(project_root, "blood-vessel", "train", "image")
    label_orig = os.path.join(project_root, "blood-vessel", "train", "label")
    image_out = os.path.join(project_root, "blood-vessel-patch", "train", "image")
    label_out = os.path.join(project_root, "blood-vessel-patch", "train", "label")

    # 可以调整重叠比例，例如0.2表示20%重叠
    process(image_orig, label_orig, image_out, label_out, cols=4, rows=4, overlap_percent=0.2)
    # cols和rows是切块网格
    # overlap_percent是重叠比例
