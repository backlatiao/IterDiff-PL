"""血管分割扩散模型训练脚本
使用预训练的TransUNet生成预测掩码，然后用扩散模型进行精细化处理
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from model.diffusion_model_v2 import DiffusionNet, DiffusionPipeline

from train_transunet import calculate_iou, calculate_dice

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def calculate_sensitivity(preds, targets, smooth=1e-6):
    """计算敏感度/召回率(Sensitivity/Recall)"""
    preds = preds.view(-1)
    targets = targets.view(-1)
    tp = (preds * targets).sum()
    fn = ((1 - preds) * targets).sum()
    return (tp + smooth) / (tp + fn + smooth)

def calculate_specificity(preds, targets, smooth=1e-6):
    """计算特异度(Specificity)"""
    preds = preds.view(-1)
    targets = targets.view(-1)
    tn = ((1 - preds) * (1 - targets)).sum()
    fp = (preds * (1 - targets)).sum()
    return (tn + smooth) / (tn + fp + smooth)

def calculate_accuracy(preds, targets, smooth=1e-6):
    """计算准确率(Accuracy)"""
    preds = preds.view(-1)
    targets = targets.view(-1)
    correct = (preds == targets).sum()
    return (correct + smooth) / (targets.numel() + smooth)

def calculate_precision(preds, targets, smooth=1e-6):
    """计算精确率(Precision)"""
    preds = preds.view(-1)
    targets = targets.view(-1)
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    return (tp + smooth) / (tp + fp + smooth)

def calculate_auc_roc(preds_prob, targets):
    """
    计算AUC-ROC
    preds_prob: 预测概率值(0-1之间的连续值)
    targets: 真实标签 (0或1)
    """
    try:
        from sklearn.metrics import roc_auc_score
        preds_prob_np = preds_prob.view(-1).cpu().numpy()
        targets_np = targets.view(-1).cpu().numpy()
        
        # 检查是否只有一个类别
        if len(np.unique(targets_np)) < 2:
            return 0.5  # 如果只有一个类别，返回0.5
        
        auc = roc_auc_score(targets_np, preds_prob_np)
        return auc
    except Exception as e:
        print(f"计算AUC-ROC时出错: {e}")
        return 0.0

def calculate_auc_pr(preds_prob, targets):
    """
    计算AUC-PR (平均精度 Average Precision)
    preds_prob: 预测概率值(0-1之间的连续值)
    targets: 真实标签 (0或1)
    """
    try:
        from sklearn.metrics import average_precision_score
        preds_prob_np = preds_prob.view(-1).cpu().numpy()
        targets_np = targets.view(-1).cpu().numpy()
        
        # 检查是否只有一个类别
        if len(np.unique(targets_np)) < 2:
            return 0.0  # 如果只有一个类别，返回0
        
        ap = average_precision_score(targets_np, preds_prob_np)
        return ap
    except Exception as e:
        print(f"计算AUC-PR时出错: {e}")
        return 0.0

# --- 早停机制 (参考 train.py) ---
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_iou_max = -np.inf
        self.delta = delta
        self.path = path if path is not None else os.path.join(".", "weights", "diffusion_model_best.pth")
    
    def __call__(self, val_iou, model, optimizer=None, epoch=None, scheduler=None):
        score = val_iou
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_iou, model, optimizer, epoch, scheduler)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_iou, model, optimizer, epoch, scheduler)
            self.counter = 0
    
    def save_checkpoint(self, val_iou, model, optimizer=None, epoch=None, scheduler=None):
        if self.verbose:
            print(f'Validation IoU increased ({self.val_iou_max:.6f} --> {val_iou:.6f}). Saving model...')
        
        # 保存完整的checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'best_iou': val_iou
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, self.path)
        self.val_iou_max = val_iou


class BloodVesselDataset(Dataset):
    """
    血管分割数据集（支持加载预先生成的TransUNet预测）
    """
    def __init__(self, image_dir, mask_dir, transunet_pred_dir=None, transform=None, img_size=(512, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transunet_pred_dir = transunet_pred_dir  # TransUNet预测结果目录
        self.transform = transform
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.preprocess = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.mask_preprocess = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
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
        
        # 加载预先生成的TransUNet预测（如果提供了路径）
        transunet_pred = None
        if self.transunet_pred_dir is not None:
            # 优先尝试加载.npy文件
            pred_path_npy = os.path.join(self.transunet_pred_dir, img_name.replace('.png', '.npy').replace('.jpg', '.npy').replace('.jpeg', '.npy'))
            # 如果没有.npy，尝试加载图片文件
            pred_path_png = os.path.join(self.transunet_pred_dir, img_name)
            
            if os.path.exists(pred_path_npy):
                # 加载.npy文件
                transunet_pred = np.load(pred_path_npy)
                transunet_pred = torch.from_numpy(transunet_pred).float()
            elif os.path.exists(pred_path_png):
                # 加载图片文件
                pred_image = Image.open(pred_path_png).convert('L')
                transunet_pred = self.mask_preprocess(pred_image)
                # 归一化到[0,1]，保留连续值
                transunet_pred = transunet_pred.float()
            else:
                print(f"警告：找不到预测文件 {pred_path_npy} 或 {pred_path_png}")
        
        return image, mask, transunet_pred, img_name


def train_diffusion_model(resume_from_checkpoint=None, use_mixed_data=False):
    """
    训练扩散模型（优化版：使用预先生成的模型预测）
    
    Args:
        resume_from_checkpoint: 可选，从指定的checkpoint继续训练
        use_mixed_data: 是否使用混合数据（patch+resize）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径配置：使用相对路径
    if use_mixed_data:
        train_data_path = "./blood-vessel-diffusion-train"
        print(f"使用混合训练数据（patch+resize）: {train_data_path}")
    else:
        train_data_path = "./blood-vessel"
        print(f"使用原始数据: {train_data_path}")
    
    train_image_dir = os.path.join(train_data_path, "image")
    train_mask_dir = os.path.join(train_data_path, "label")
    
    # 预测结果路径
    if use_mixed_data:
        transunet_pred_dir = os.path.join(train_data_path, "prediction")
        print(f"使用对应的预测目录: {transunet_pred_dir}")
    else:
        transunet_pred_dir = os.path.join(".", "transunet_predictions_last", "train")
        print(f"使用预测目录: {transunet_pred_dir}")
    
    # 检查是否已生成预测
    if not os.path.exists(transunet_pred_dir):
        print(f"错误：找不到预测结果目录: {transunet_pred_dir}")
        if use_mixed_data:
            print("请先运行 scripts/preprocess/prepare_diffusion_data.py 生成混合训练数据")
        return
    
    # 超参数设置
    batch_size = 2
    learning_rate = 4e-5  # 降低学习率，精细化任务需要更温和的优化
    num_epochs = 500
    img_size = 512
    max_refine_step = 100  # 降低到100，从更小的噪声开始，保留更多TransUNet结构
    patience = 50  # 增加早停耐心
    start_epoch = 0  # 起始epoch
    
    # 创建完整数据集（包含TransUNet预测）
    full_dataset = BloodVesselDataset(
        train_image_dir, 
        train_mask_dir, 
        transunet_pred_dir=transunet_pred_dir,
        img_size=(img_size, img_size)
    )
    
    # 划分训练集和验证集(80% 训练, 20% 验证)，使用固定种子确保每次验证相同样本
    torch.manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 固定验证集索引，确保每个epoch验证相同的样本
    val_sample_indices = list(range(min(20, len(val_dataset))))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print(f"使用预测目录: {transunet_pred_dir}")
    if use_mixed_data:
        print("训练数据包含: patch切块 + resize数据")
    print("使用预先生成的模型预测，无需加载模型")
    
    # 初始化扩散模型
    print("初始化扩散模型...")
    diffusion_model = DiffusionNet(
        img_size=img_size,
        in_channels=5, 
        out_channels=1
    ).to(device)
    
    # 定义损失函数和优化器
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()  # 用于分割损失
    optimizer = optim.Adam(diffusion_model.parameters(), lr=learning_rate)
    
    # 定义学习率调度器
    # 当验证集 IoU 在 'patience' 个 epoch 内没有提升时，学习率将被 'factor' 缩放
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    
    # 初始化扩散管道
    diffusion_pipeline = DiffusionPipeline(diffusion_model, device=device)
    
    # 生成带时间戳的模型保存路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(".", "weights", f"diffusion_model_best_{timestamp}.pth")
    print(f"最佳模型将保存到: {best_model_path}")
    
    # 初始化早停机制（使用相对路径）
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=best_model_path)
    
    # 从checkpoint恢复训练
    if resume_from_checkpoint is not None:
        if os.path.exists(resume_from_checkpoint):
            print(f"\n从checkpoint加载权重: {resume_from_checkpoint}")
            checkpoint = torch.load(resume_from_checkpoint, map_location=device, weights_only=False)
            
            # 加载模型权重
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                diffusion_model.load_state_dict(checkpoint['model_state_dict'])
                
                # 询问是否继承训练进度
                print(f"\nDetected checkpoint with training state (epoch {checkpoint.get('epoch', 0)})")
                continue_training = input("是否继承训练进度？(y/n，默认y): ").strip().lower()
                
                if continue_training != 'n':
                    # 继承完整训练状态
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    early_stopping.best_score = checkpoint.get('best_iou', None)
                    early_stopping.val_iou_max = checkpoint.get('best_iou', -np.inf)
                    print(f"继承训练进度：从epoch {start_epoch} 继续训练，最佳IoU: {early_stopping.val_iou_max:.4f}")
                else:
                    # 只加载模型权重，从epoch 0开始
                    print("Loaded model weights only. Restart training from epoch 0.")
            else:
                # 兼容旧格式（只有模型权重）
                diffusion_model.load_state_dict(checkpoint)
                print("Loaded model weights successfully. Start from epoch 0.")
        else:
            print(f"Warning: checkpoint not found: {resume_from_checkpoint}. Start from scratch.")
    else:
        print("\nStart training from scratch")
    
    print("开始训练扩散模型...")
    for epoch in range(start_epoch, num_epochs):
        diffusion_model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Train')
        
        for images, true_masks, transunet_preds, _ in train_bar:
            images = images.to(device)
            true_masks = true_masks.to(device)
            transunet_preds = transunet_preds.to(device)
            
            # 关键改进：使用TransUNet预测作为"干净起点"，而不是GT
            # 这样训练和推理的分布就一致了，消除domain gap
            clean_start = transunet_preds
            
            # 让模型专注学习[0, max_refine_step] 范围内的去噪，与测试时的 refine_step 对齐
            timesteps = torch.randint(0, max_refine_step, (images.size(0),)).to(device)
            
            # 为TransUNet预测添加噪声（而不是为GT添加噪声）
            noisy_target_masks, noise = diffusion_pipeline.forward_process(
                clean_start, transunet_preds, timesteps
            )
            
            # 预测噪声
            predicted_noise = diffusion_model(noisy_target_masks, images, transunet_preds, timesteps)
            
            # 计算噪声 MSE 损失
            mse_loss = mse_criterion(predicted_noise, noise)
            
            # 使用预测的噪声进行一步去噪，然后计算与真实掩码的分割损失
            alpha_t = diffusion_pipeline.model.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            
            # 使用DDPM公式反推干净图像
            predicted_clean = (noisy_target_masks - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            predicted_clean = torch.clamp(predicted_clean, 0, 1)
            
            # 计算分割损失（IoU损失 + BCE损失）
            # 注意：这里仍然使用GT作为监督信号，确保模型学习正确的分割
            # IoU损失
            intersection = (predicted_clean * true_masks).sum(dim=[1,2,3])
            union = predicted_clean.sum(dim=[1,2,3]) + true_masks.sum(dim=[1,2,3]) - intersection
            iou_loss = 1 - (intersection + 1e-6) / (union + 1e-6)
            iou_loss = iou_loss.mean()
            
            # BCE损失（需要logits，所以反sigmoid）
            predicted_clean_logits = torch.log(predicted_clean + 1e-7) - torch.log(1 - predicted_clean + 1e-7)
            bce_loss = bce_criterion(predicted_clean_logits, true_masks)

            # 混合损失：噪声损失 + 分割损失
            # 噪声损失帮助学习扩散过程，分割损失确保去噪后的质量
            loss = mse_loss + 0.5 * iou_loss + 0.5 * bce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- 验证环节 ---
        diffusion_model.eval()
        val_iou_list = []
        transunet_iou_list = []
        print("验证中...")
        # 使用固定的验证样本确保可比性
        with torch.no_grad():
            for idx in val_sample_indices:
                image, true_mask, transunet_pred, _ = val_dataset[idx]
                image = image.unsqueeze(0).to(device)
                true_mask = true_mask.unsqueeze(0).to(device)
                transunet_pred = transunet_pred.unsqueeze(0).to(device)
                
                # TransUNet预测（直接使用预先生成的）
                # 二值化后计算IoU（与train.py完全一致）
                initial_pred_bin = (transunet_pred > 0.5).float()
                transunet_iou = calculate_iou(initial_pred_bin, true_mask)  # train.py的函数已经返回item()
                transunet_iou_list.append(transunet_iou)
                
                # refine_step 必须在训练范围内，且与max_refine_step一致
                refined_mask = diffusion_pipeline.sample(transunet_pred, image, num_inference_steps=10, refine_step=max_refine_step)
                # 二值化后计算IoU
                refined_mask_bin = (refined_mask > 0.5).float()
                iou = calculate_iou(refined_mask_bin, true_mask)  # train.py的函数已经返回item()
                val_iou_list.append(iou)
        
        avg_val_iou = np.mean(val_iou_list)
        avg_transunet_iou = np.mean(transunet_iou_list)
        improvement = avg_val_iou - avg_transunet_iou
        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f}, TransUNet IoU: {avg_transunet_iou:.4f}, Refined IoU: {avg_val_iou:.4f}, 提升: {improvement:+.4f}')
        
        # 更新学习率调度器
        scheduler.step(avg_val_iou)
        
        # 早停判断与保存最佳模型
        early_stopping(avg_val_iou, diffusion_model, optimizer, epoch, scheduler)
        if early_stopping.early_stop:
            print("触发早停机制!")
            break
        
        # 定期保存（包含完整训练状态）
        if (epoch + 1) % 10 == 0:
            model_save_path = os.path.join(".", "weights", f"diffusion_model_epoch_{epoch+1}_{timestamp}.pth")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': diffusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': early_stopping.val_iou_max,
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, model_save_path)
            print(f"周期性保存: {model_save_path}")
    
    # 加载最佳模型参数
    if os.path.exists(early_stopping.path):
        checkpoint = torch.load(early_stopping.path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            diffusion_model.load_state_dict(checkpoint)
    
    final_model_path = os.path.join(".", "weights", f"diffusion_model_final_{timestamp}.pth")
    final_checkpoint = {
        'epoch': num_epochs - 1,
        'model_state_dict': diffusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': early_stopping.val_iou_max,
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(final_checkpoint, final_model_path)
    print(f"训练结束，最终模型已保存: {final_model_path}")
    print(f"最佳模型: {best_model_path}")


def test_diffusion_model(custom_pred_dir=None, custom_data_dir=None):
    """
    测试扩散模型并输出评估指标（优化版：使用预先生成的TransUNet预测）
    
    Args:
        custom_pred_dir: 可选，自定义预测目录
        custom_data_dir: 可选，自定义数据目录
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化扩散模型
    diffusion_model = DiffusionNet(
        img_size=512,
        in_channels=5,
        out_channels=1
    ).to(device)
    
    # 检查是否存在训练好的模型（使用相对路径）
    model_path = os.path.join(".", "weights", "diffusion_model_final.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(".", "weights", "diffusion_model_best.pth")
        
    if os.path.exists(model_path):
        diffusion_pipeline = DiffusionPipeline(diffusion_model, device=device)
        checkpoint = torch.load(model_path, map_location=device)
        
        # 兼容新旧格式
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model_keys = set(diffusion_model.state_dict().keys())
            state_dict_filtered = {k: v for k, v in checkpoint.items() if k in model_keys}
            diffusion_model.load_state_dict(state_dict_filtered, strict=False)
        
        print(f"成功加载模型权重: {model_path}")
    else:
        print("未找到训练好的模型，无法进行测试评估")
        return
    
    diffusion_model.eval()
    
    # 加载测试集（使用相对路径）
    if custom_data_dir is not None:
        test_data_path = custom_data_dir
        print(f"使用自定义数据目录: {test_data_path}")
    else:
        test_data_path = "./blood-vessel"
        print(f"使用默认数据目录: {test_data_path}")
    
    test_image_path = os.path.join(test_data_path, "test", "image")
    test_mask_path = os.path.join(test_data_path, "test", "label")
    
    # TransUNet预测结果路径（使用相对路径）
    if custom_pred_dir is not None:
        transunet_pred_dir = custom_pred_dir
        print(f"使用自定义预测目录: {transunet_pred_dir}")
    else:
        # 如果使用了自定义数据目录，尝试使用对应的预测目录
        if custom_data_dir is not None and "diffusion" in custom_data_dir:
            transunet_pred_dir = os.path.join(custom_data_dir, "test", "prediction")
            print(f"使用数据集对应的预测目录: {transunet_pred_dir}")
        else:
            transunet_pred_dir = os.path.join(".", "transunet_predictions", "test")
            print(f"使用默认预测目录: {transunet_pred_dir}")
    
    # 检查是否已生成TransUNet预测
    if not os.path.exists(transunet_pred_dir):
        print(f"错误：找不到TransUNet预测结果目录: {transunet_pred_dir}")
        print("请先运行 scripts/inference/generate_transunet_predictions.py 生成预测结果")
        return
    
    test_dataset = BloodVesselDataset(
        test_image_path, 
        test_mask_path, 
        transunet_pred_dir=transunet_pred_dir,
        img_size=(512, 512)
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"测试集样本数: {len(test_dataset)}")
    
    metrics = {
        'iou': [], 'dice': [], 'precision': [], 'sensitivity': [], 
        'specificity': [], 'accuracy': [], 'auc_roc': [], 'auc_pr': []
    }
    
    print("开始模型评估...")
    with torch.no_grad():
        for images, true_masks, transunet_preds, name in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            true_masks = true_masks.to(device)
            transunet_preds = transunet_preds.to(device)
            
            # 扩散精细化：使用改进后的采样逻辑
            # refine_step 设置为100，与训练时的max_refine_step对齐
            refined_result = diffusion_pipeline.sample(transunet_preds, images, num_inference_steps=20, refine_step=100)
            
            # 保留连续值用于计算AUC
            refined_result_prob = refined_result.clone()
            
            # 二值化后计算指标（与train.py完全一致）
            refined_result_bin = (refined_result > 0.5).float()
            metrics['iou'].append(calculate_iou(refined_result_bin, true_masks))  # 已经是item()
            metrics['dice'].append(calculate_dice(refined_result_bin, true_masks))  # 已经是item()
            metrics['precision'].append(calculate_precision(refined_result_bin, true_masks).item())
            metrics['sensitivity'].append(calculate_sensitivity(refined_result_bin, true_masks).item())
            metrics['specificity'].append(calculate_specificity(refined_result_bin, true_masks).item())
            metrics['accuracy'].append(calculate_accuracy(refined_result_bin, true_masks).item())
            
            # 使用连续概率值计算AUC
            metrics['auc_roc'].append(calculate_auc_roc(refined_result_prob, true_masks))
            metrics['auc_pr'].append(calculate_auc_pr(refined_result_prob, true_masks))
    
    # 打印平均指标
    print("\n" + "="*60)
    print("扩散模型评估结果".center(60))
    print("="*60)
    print(f"IoU (Intersection over Union):     {np.mean(metrics['iou']):.4f}")
    print(f"Dice 系数 (Dice Coefficient):      {np.mean(metrics['dice']):.4f}")
    print(f"精确率(Precision):                {np.mean(metrics['precision']):.4f}")
    print(f"敏感度/召回率(Sensitivity/Recall): {np.mean(metrics['sensitivity']):.4f}")
    print(f"特异度(Specificity):              {np.mean(metrics['specificity']):.4f}")
    print(f"准确率(Accuracy):                 {np.mean(metrics['accuracy']):.4f}")
    print(f"AUC-ROC:                           {np.mean(metrics['auc_roc']):.4f}")
    print(f"AUC-PR (Average Precision):        {np.mean(metrics['auc_pr']):.4f}")
    print("="*60)
    
    # 生成可视化结果
    print("\n生成预测可视化结果...")
    visualize_diffusion_results(diffusion_pipeline, test_loader, device, num_samples=len(test_dataset))


def visualize_diffusion_results(diffusion_pipeline, test_loader, device, num_samples=5, save_dir=None):
    """
    可视化扩散模型的预测结果（优化版：使用预先生成的TransUNet预测）
    显示：原始图像、真实掩码、TransUNet预测、扩散模型精细化后的结果
    """
    if save_dir is None:
        save_dir = os.path.join(".", "output", "diffusion_visualization")
    os.makedirs(save_dir, exist_ok=True)
    
    diffusion_pipeline.model.eval()
    
    # 适配小样本测试集
    actual_num_samples = min(num_samples, len(test_loader.dataset))
    if actual_num_samples == 0:
        print("警告：测试集为空，无法生成可视化")
        return
    
    # 随机选择样本（但不超过实际数量）
    if actual_num_samples < len(test_loader.dataset):
        indices = np.random.choice(len(test_loader.dataset), actual_num_samples, replace=False)
    else:
        indices = list(range(actual_num_samples))
    
    with torch.no_grad():
        fig, axes = plt.subplots(actual_num_samples, 4, figsize=(20, 5*actual_num_samples))
        if actual_num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            image, mask, transunet_pred, img_name = test_loader.dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            mask_tensor = mask.unsqueeze(0).to(device)
            transunet_pred_tensor = transunet_pred.unsqueeze(0).to(device)
            
            # TransUNet预测（直接使用预先生成的）
            transunet_pred_bin = (transunet_pred_tensor > 0.5).float().squeeze().cpu().numpy()
            
            # 扩散模型精细化
            refined_pred = diffusion_pipeline.sample(transunet_pred_tensor, image_tensor, num_inference_steps=20, refine_step=100)
            refined_pred_bin = (refined_pred > 0.5).float().squeeze().cpu().numpy()
            
            # 处理图像用于显示
            image_np = image.permute(1, 2, 0).numpy()
            image_np = (image_np * 0.5 + 0.5).clip(0, 1)  # 反归一化
            mask_np = mask.squeeze().numpy()
            
            # 绘制四列：原图、真实掩码、TransUNet预测、扩散模型结果
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title(f"原始图像: {img_name}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title("真实掩码")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(transunet_pred_bin, cmap='gray')
            axes[i, 2].set_title("TransUNet预测")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(refined_pred_bin, cmap='gray')
            axes[i, 3].set_title("Diffusion refined")
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'diffusion_prediction_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
        plt.close()


if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 训练扩散模型（从头开始）")
    print("2. Train diffusion model (resume from checkpoint)")
    print("3. Test diffusion model (patch inference + stitching)")
    print("4. 生成TransUNet预测")
    print("5. Prepare mixed training data (patch + resize)")
    
    choice = input("请输入选择 (1, 2, 3, 4 或 5): ")
    
    if choice == "1":
        # 询问是否使用混合数据
        print("\n选择训练数据:")
        print("1. Original data (blood-vessel/train)")
        print("2. Mixed data (patch+resize, generate with option 5 first)")
        data_choice = input("请输入选择 (1 或 2，默认为1): ").strip()
        
        use_mixed = (data_choice == "2")
        train_diffusion_model(use_mixed_data=use_mixed)
        
    elif choice == "2":
        print("\n可用的checkpoint:")
        weights_dir = "./weights"
        if os.path.exists(weights_dir):
            checkpoints = [f for f in os.listdir(weights_dir) if f.startswith('diffusion_model')]
            for i, ckpt in enumerate(checkpoints):
                print(f"  {i+1}. {ckpt}")
        
        ckpt_path = input("\n请输入checkpoint路径（或直接输入编号）: ").strip()
        
        # 如果输入的是编号
        if ckpt_path.isdigit():
            idx = int(ckpt_path) - 1
            if 0 <= idx < len(checkpoints):
                ckpt_path = os.path.join(weights_dir, checkpoints[idx])
            else:
                print("Invalid index")
                exit()
        # 如果输入的是相对路径，补全
        elif not ckpt_path.startswith('.'):
            ckpt_path = os.path.join(weights_dir, ckpt_path)
        
        # 询问是否使用混合数据
        print("\n选择训练数据:")
        print("1. Original data (blood-vessel/train)")
        print("2. Mixed data (patch+resize)")
        data_choice = input("请输入选择 (1 或 2，默认为1): ").strip()
        
        use_mixed = (data_choice == "2")
        train_diffusion_model(resume_from_checkpoint=ckpt_path, use_mixed_data=use_mixed)
        
    elif choice == "3":
        print("\n使用patch推理模式测试")
        print("将对原图切patch预测，然后拼接回整图")
        print("请运行: python scripts/inference/test_diffusion_patch.py")
        
    elif choice == "4":
        print("Run scripts/inference/generate_transunet_predictions.py to generate predictions")
        print("命令: python scripts/inference/generate_transunet_predictions.py")
        
    elif choice == "5":
        print("Run scripts/preprocess/prepare_diffusion_data.py to generate mixed training data")
        print("命令: python scripts/preprocess/prepare_diffusion_data.py")
        print("This will generate a mixed dataset with patch and resize samples")
        
    else:
        print("无效选择")
