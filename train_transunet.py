import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import random
from model.transunet import TransUNet
import sys

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据路径配置
train_data_path = os.path.join(project_root, "blood-vessel-patch")
test_data_path = os.path.join(project_root, "blood-vessel")

train_image_dir = os.path.join(train_data_path, "train/image")
train_mask_dir = os.path.join(train_data_path, "train/label")
test_image_dir = os.path.join(test_data_path, "test/image")
test_mask_dir = os.path.join(test_data_path, "test/label")

# 自定义数据集类
class FIVESDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, img_size=(512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.preprocess = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),  # 新增中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.mask_preprocess = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),  # 新增中心裁剪
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        image = self.preprocess(image)
        mask = self.mask_preprocess(mask)
        mask = (mask > 0.5).float()
        
        return image, mask, img_name

# 早停机制类
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.005, path=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_iou_max = -np.inf  # IoU越大越好，初始化为负无穷
        self.delta = delta
        self.path = path if path is not None else os.path.join(project_root, 'output', 'pic', 'checkpoint_1231.pt')
    
    def __call__(self, val_iou, model):
        score = val_iou  # IoU直接作为分数，越大越好
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_iou, model)
        elif score < self.best_score + self.delta:
            # IoU没有提升超过delta
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # IoU提升，保存模型
            self.best_score = score
            self.save_checkpoint(val_iou, model)
            self.counter = 0
    
    def save_checkpoint(self, val_iou, model):
        if self.verbose:
            print(f'Validation IoU increased ({self.val_iou_max:.6f} --> {val_iou:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_iou_max = val_iou

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    train_losses = []
    val_losses = []
    val_iou_scores = []
    
    # 早停机制现在基于IoU
    best_model_path = os.path.join(project_root, 'weights', 'best_model_1231.pth')
    early_stopping = EarlyStopping(patience=15, verbose=True, path=best_model_path)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, masks, _ in train_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=loss.item())
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        val_loss = 0.0
        iou_scores = []
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for images, masks, _ in val_bar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
                preds = (outputs > 0.5).float()
                iou = calculate_iou(preds, masks)
                iou_scores.append(iou)
                
                val_bar.set_postfix(loss=loss.item(), iou=iou)
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_iou = np.mean(iou_scores)
        val_losses.append(epoch_val_loss)
        val_iou_scores.append(epoch_val_iou)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val IoU: {epoch_val_iou:.4f}')
        
        # 使用IoU进行早停判断
        early_stopping(epoch_val_iou, model)
        if early_stopping.early_stop:
            print("早停触发!")
            break
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print(f"警告: 未找到最佳权重文件 {best_model_path}，将使用当前权重进行后续评估")
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_iou_scores': val_iou_scores
    }

# 计算IoU
def calculate_iou(preds, targets):
    smooth = 1e-6
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

# 计算Dice系数
def calculate_dice(preds, targets):
    smooth = 1e-6
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()


def calculate_bacc(preds, targets):
    smooth = 1e-6
    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    fp = (preds * (1 - targets)).sum()

    sensitivity = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    return (0.5 * (sensitivity + specificity)).item()


def _to_4d_mask(mask):
    if mask.dim() == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    if mask.dim() == 3:
        return mask.unsqueeze(1)
    return mask


def _soft_erode(mask):
    eroded_h = -F.max_pool2d(-mask, kernel_size=(3, 1), stride=1, padding=(1, 0))
    eroded_w = -F.max_pool2d(-mask, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(eroded_h, eroded_w)


def _soft_dilate(mask):
    return F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)


def _soft_open(mask):
    return _soft_dilate(_soft_erode(mask))


def _soft_skeleton(mask, num_iter=10):
    mask = mask.float()
    opened = _soft_open(mask)
    skeleton = F.relu(mask - opened)

    for _ in range(num_iter):
        mask = _soft_erode(mask)
        opened = _soft_open(mask)
        delta = F.relu(mask - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)

    return skeleton


def calculate_cldice(preds, targets, num_iter=10):
    smooth = 1e-6
    preds = _to_4d_mask(preds.float())
    targets = _to_4d_mask(targets.float())

    pred_skeleton = _soft_skeleton(preds, num_iter=num_iter)
    target_skeleton = _soft_skeleton(targets, num_iter=num_iter)

    topology_precision = (pred_skeleton * targets).sum() / (pred_skeleton.sum() + smooth)
    topology_sensitivity = (target_skeleton * preds).sum() / (target_skeleton.sum() + smooth)
    cldice = (2 * topology_precision * topology_sensitivity) / (
        topology_precision + topology_sensitivity + smooth
    )
    return cldice.item()

# 可视化训练历史
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['val_iou_scores'], label='Validation IoU', color='green')
    ax2.set_title('Validation IoU Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'output', 'pic', 'training_history_1231.png'), dpi=300, bbox_inches='tight')
    plt.show()

# 可视化预测结果
def visualize_predictions(model, test_loader, num_samples=5):
    model.eval()
    samples = []
    
    all_indices = list(range(len(test_loader.dataset)))
    # 修复：当测试集样本数小于num_samples时，使用实际样本数
    actual_num_samples = min(num_samples, len(all_indices))
    if actual_num_samples == 0:
        print("警告：测试集为空，无法生成可视化")
        return []
    selected_indices = random.sample(all_indices, actual_num_samples)
    
    with torch.no_grad():
        for idx in selected_indices:
            image, mask, img_name = test_loader.dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            
            prediction = model(image_tensor)
            prediction_binary = (prediction > 0.5).float()
            
            samples.append({
                'name': img_name,
                'image': image,
                'true_mask': mask.squeeze(),
                'pred_mask': prediction_binary.squeeze().cpu()
            })
    
    fig, axes = plt.subplots(actual_num_samples, 4, figsize=(20, 5*actual_num_samples))
    if actual_num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        image = sample['image'].permute(1, 2, 0).numpy()
        image = (image * 0.5 + 0.5)
        true_mask = sample['true_mask'].numpy()
        pred_mask = sample['pred_mask'].numpy()
        
        difference = np.abs(true_mask - pred_mask)
        
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"原始图像: {sample['name']}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 1].set_title("True")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title("Predict")
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(image, alpha=0.6)
        axes[i, 3].imshow(difference, cmap='Reds', alpha=0.5)
        axes[i, 3].set_title("Difference(Red)")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'output', 'pic', 'prediction_comparison_1231.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return samples

# 在测试集上评估模型
def evaluate_model(model, test_loader):
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    total_bacc = 0.0
    total_cldice = 0.0
    total_samples = 0
    
    test_bar = tqdm(test_loader, desc='Testing')
    
    with torch.no_grad():
        for images, masks, _ in test_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = (outputs > 0.5).float()
            
            batch_iou = calculate_iou(preds, masks)
            total_iou += batch_iou * images.size(0)
            
            batch_dice = calculate_dice(preds, masks)
            total_dice += batch_dice * images.size(0)

            batch_bacc = calculate_bacc(preds, masks)
            total_bacc += batch_bacc * images.size(0)

            batch_cldice = calculate_cldice(preds, masks)
            total_cldice += batch_cldice * images.size(0)
            
            total_samples += images.size(0)
            
            test_bar.set_postfix(iou=batch_iou, dice=batch_dice, bacc=batch_bacc, cldice=batch_cldice)
    
    mean_iou = total_iou / total_samples
    mean_dice = total_dice / total_samples
    mean_bacc = total_bacc / total_samples
    mean_cldice = total_cldice / total_samples
    
    print(f"测试集评估结果:")
    print(f"平均IoU: {mean_iou:.4f}")
    print(f"平均Dice系数: {mean_dice:.4f}")
    print(f"平均BACC: {mean_bacc:.4f}")
    print(f"平均clDice: {mean_cldice:.4f}")
    
    return mean_iou, mean_dice, mean_bacc, mean_cldice

# 主函数
def main():
    # 超参数设置
    batch_size = 4  # 根据GPU内存调整
    learning_rate = 1e-4
    num_epochs = 100
    img_size = 512
    
    # 创建数据集和数据加载器
    train_dataset = FIVESDataset(train_image_dir, train_mask_dir, img_size=img_size)
    test_dataset = FIVESDataset(test_image_dir, test_mask_dir, img_size=img_size)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"训练样本数: {len(train_subset)}")
    print(f"验证样本数: {len(val_subset)}")
    print(f"测试样本数: {len(test_dataset)}")
    
    # 初始化TransUNet模型 - 使用完全匹配的通道参数
    model = TransUNet(
        img_size=img_size,
        patch_size=16,
        in_channels=3,
        out_channels=1,
        embed_dim=512,  # 与最高级特征通道数匹配
        num_heads=8,
        mlp_dim=2048,
        num_layers=6
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 训练模型
    print("开始训练TransUNet模型...")
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 在测试集上评估模型
    print("在测试集上评估模型...")
    mean_iou, mean_dice, mean_bacc, mean_cldice = evaluate_model(model, test_loader)
    
    # 可视化预测结果
    print("可视化预测结果...")
    visualize_predictions(model, test_loader, num_samples=5)
    
    # 保存最终模型
    os.makedirs(os.path.join(project_root, 'output'), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(project_root, 'output', 'unet_vessel_segmentation.pth'))
    print("模型已保存为 './output/unet_vessel_segmentation.pth'")

if __name__ == "__main__":
    main()
