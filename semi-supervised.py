import time
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from model.diffusion_model import DiffusionNet, DiffusionPipeline
from model.transunet import TransUNet
from train_transunet import calculate_dice, calculate_iou


class DiffusionGuidedMeanTeacher:
    """融合方案：扩散引导的Mean Teacher半监督框架"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        img_size = getattr(config, 'img_size', 512)
        transunet_kwargs = dict(
            img_size=img_size,
            patch_size=getattr(config, 'patch_size', 16),
            in_channels=3,
            out_channels=1,
            embed_dim=getattr(config, 'embed_dim', 512),
            num_heads=getattr(config, 'num_heads', 8),
            mlp_dim=getattr(config, 'mlp_dim', 2048),
            num_layers=getattr(config, 'num_layers', 6),
            features=getattr(config, 'features', [64, 128, 256, 512]),
        )

        # Student 模型
        self.student = TransUNet(**transunet_kwargs).to(self.device)

        # Teacher 模型 (EMA)
        self.teacher = TransUNet(**transunet_kwargs).to(self.device)
        self.teacher.load_state_dict(self.student.state_dict())
        for param in self.teacher.parameters():
            param.requires_grad = False  # Teacher不参与梯度计算

        # 扩散精修器
        self.diffusion = DiffusionNet(
            img_size=img_size,
            in_channels=5,
            out_channels=1
        ).to(self.device)
        self.diffusion_pipeline = DiffusionPipeline(self.diffusion, self.device)

        # EMA 系数
        self.ema_decay = getattr(config, 'ema_decay', 0.999)

    def train(self, labeled_loader, unlabeled_loader=None, val_loader=None):
        """完整训练入口：监督预训练 -> 扩散精修器训练 -> 半监督自训练。"""
        self._pretrain_on_labeled(labeled_loader, val_loader)
        self._train_diffusion(labeled_loader, val_loader)

        if unlabeled_loader is not None:
            self.train_iteration(
                labeled_loader,
                unlabeled_loader,
                num_epochs=getattr(self.config, 'semi_epochs', getattr(self.config, 'epochs_per_iteration', 20))
            )

        return self.student

    def _split_labeled_batch(self, batch):
        if len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("labeled_loader需要返回(images, masks, ...)格式")

    def _split_unlabeled_batch(self, batch):
        return batch[0] if isinstance(batch, (list, tuple)) else batch

    def _pretrain_on_labeled(self, labeled_loader, val_loader=None):
        """只用有标签数据预训练Student，并同步Teacher。"""
        epochs = getattr(self.config, 'pretrain_epochs', 0)
        if epochs <= 0:
            return

        optimizer = optim.AdamW(
            self.student.parameters(),
            lr=getattr(self.config, 'lr', 1e-4),
            weight_decay=getattr(self.config, 'weight_decay', 1e-5)
        )

        for epoch in range(epochs):
            self.student.train()
            total_loss = 0.0
            for batch in labeled_loader:
                images, masks = self._split_labeled_batch(batch)
                images, masks = images.to(self.device), masks.to(self.device)

                pred = self.student(images)
                loss = hierarchical_weighted_loss(pred, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            self.teacher.load_state_dict(self.student.state_dict())
            val_iou = self._validate(val_loader or labeled_loader)
            print(f"Pretrain [{epoch + 1}/{epochs}] loss={total_loss / len(labeled_loader):.4f}, IoU={val_iou:.4f}")

    def _train_diffusion(self, labeled_loader, val_loader=None):
        """用Teacher粗预测和GT训练扩散精修器。"""
        epochs = getattr(self.config, 'diffusion_epochs', 0)
        if epochs <= 0:
            return

        optimizer = optim.AdamW(
            self.diffusion.parameters(),
            lr=getattr(self.config, 'diffusion_lr', getattr(self.config, 'lr', 1e-4)),
            weight_decay=getattr(self.config, 'weight_decay', 1e-5)
        )
        mse_loss = torch.nn.MSELoss()
        max_refine_step = getattr(self.config, 'max_refine_step', 100)

        self.teacher.eval()
        for epoch in range(epochs):
            self.diffusion.train()
            total_loss = 0.0
            for batch in labeled_loader:
                images, masks = self._split_labeled_batch(batch)
                images, masks = images.to(self.device), masks.to(self.device)

                with torch.no_grad():
                    coarse_pred = self.teacher(images)

                timesteps = torch.randint(0, max_refine_step, (images.size(0),), device=self.device)
                noisy_masks, noise = self.diffusion_pipeline.forward_process(coarse_pred, coarse_pred, timesteps)
                pred_noise = self.diffusion(noisy_masks, images, coarse_pred, timesteps)

                alpha_t = self.diffusion.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                pred_clean = (noisy_masks - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
                pred_clean = pred_clean.clamp(0, 1)

                loss = mse_loss(pred_noise, noise) + 0.5 * hierarchical_weighted_loss(pred_clean, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            val_iou = self._validate(val_loader or labeled_loader, use_diffusion=True)
            print(f"Diffusion [{epoch + 1}/{epochs}] loss={total_loss / len(labeled_loader):.4f}, refined IoU={val_iou:.4f}")

    @torch.no_grad()
    def _validate(self, data_loader, use_diffusion=False):
        """返回IoU；同时打印Dice，方便观察训练质量。"""
        if data_loader is None:
            return 0.0

        self.teacher.eval()
        self.diffusion.eval()
        iou_list, dice_list = [], []

        for batch in data_loader:
            images, masks = self._split_labeled_batch(batch)
            images, masks = images.to(self.device), masks.to(self.device)
            pred = self.teacher(images)
            if use_diffusion:
                pred = self.diffusion_pipeline.sample(
                    pred,
                    images,
                    num_inference_steps=getattr(self.config, 'num_inference_steps', 20),
                    refine_step=getattr(self.config, 'refine_step', 80)
                )
            pred_bin = (pred > 0.5).float()
            iou_list.append(calculate_iou(pred_bin, masks))
            dice_list.append(calculate_dice(pred_bin, masks))

        mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
        mean_dice = float(np.mean(dice_list)) if dice_list else 0.0
        print(f"Validate IoU={mean_iou:.4f}, Dice={mean_dice:.4f}")
        return mean_iou

    def train_iteration(self, labeled_loader, unlabeled_loader, num_epochs):
        """一轮迭代训练"""

        optimizer = optim.AdamW(
            self.student.parameters(),
            lr=getattr(self.config, 'lr', 1e-4),
            weight_decay=getattr(self.config, 'weight_decay', 1e-5)
        )

        # 伪标签权重从小到大（课程学习）
        lambda_schedule = np.linspace(0.1, 1.0, num_epochs)

        for epoch in range(num_epochs):
            self.student.train()

            # 同时遍历有标签和无标签数据
            labeled_iter = iter(labeled_loader)
            unlabeled_iter = iter(unlabeled_loader)

            for step in range(max(len(labeled_loader), len(unlabeled_loader))):

                # ========== 有标签数据：监督损失 ==========
                try:
                    batch_l = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    batch_l = next(labeled_iter)
                images_l, masks_l = self._split_labeled_batch(batch_l)

                images_l, masks_l = images_l.to(self.device), masks_l.to(self.device)

                pred_l = self.student(images_l)
                # loss_sup = 0.5 * F.binary_cross_entropy(pred_l, masks_l) + 0.5 * dice_loss(pred_l, masks_l)
                loss_sup = hierarchical_weighted_loss(pred_l, masks_l)

                # ========== 无标签数据：伪标签损失 ==========
                try:
                    batch_u = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    batch_u = next(unlabeled_iter)
                images_u = self._split_unlabeled_batch(batch_u)

                images_u = images_u.to(self.device)

                # Teacher 生成粗分割
                with torch.no_grad():
                    coarse_pred = self.teacher(images_u)

                # 扩散模型 K 次采样
                soft_label, uncertainty = self._diffusion_ensemble(
                    images_u, coarse_pred, K=getattr(self.config, 'K', 4),
                    refine_step=getattr(self.config, 'refine_step', 80),
                    num_inference_steps=getattr(self.config, 'num_inference_steps', 20)
                )

                # 不确定性加权
                confidence_weight = torch.exp(-getattr(self.config, 'gamma', 4.0) * uncertainty)

                # Student 预测
                pred_u = self.student(images_u)

                # 软伪标签损失（加权）
                loss_pl = weighted_bce_loss(pred_u, soft_label, confidence_weight)

                # # ========== 一致性正则 (FixMatch风格) ==========
                # # 弱增强
                # images_u_weak = weak_augment(images_u)
                # # 强增强
                # images_u_strong = strong_augment(images_u)

                # with torch.no_grad():
                #     pred_weak = self.teacher(images_u_weak)
                #     pseudo_mask = (pred_weak > 0.5).float()
                #
                # pred_strong = self.student(images_u_strong)

                # # 只在高置信度区域计算一致性损失
                # high_conf_mask = (pred_weak > 0.9) | (pred_weak < 0.1)
                # loss_cons = masked_bce_loss(pred_strong, pseudo_mask, high_conf_mask)

                # ========== 总损失 ==========
                lambda_pl = lambda_schedule[epoch]
                loss = loss_sup + lambda_pl * loss_pl # + self.config.mu * loss_cons

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ========== EMA 更新 Teacher ==========
                self._update_ema_teacher()

            print(f"Epoch [{epoch + 1}/{num_epochs}] "
                  f"L_sup: {loss_sup:.4f}, L_pl: {loss_pl:.4f}")

    @staticmethod
    def _bernoulli_entropy(p, eps=1e-6):
        p = p.clamp(eps, 1 - eps)
        return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

    @torch.no_grad()
    def _diffusion_ensemble(self, images, coarse_pred, K=4, refine_step=80, num_inference_steps=20, seed_base=12345):
        """扩散模型 K 次采样，返回软标签和不确定性"""
        self.teacher.eval()
        self.diffusion.eval()

        samples = []
        for k in range(K):
            torch.manual_seed(seed_base + k)   # 固定 refine_step，用 seed 控制多样性

            sample = self.diffusion_pipeline.sample(
            initial_prediction=coarse_pred,
            original_image=images,
            num_inference_steps=num_inference_steps,
            refine_step=refine_step,
            )
            samples.append(sample)

        samples = torch.stack(samples)  # (K, B, 1, H, W)

        # 软标签：均值
        soft_label = samples.mean(dim=0)  # (B, 1, H, W)

        # 不确定性熵
        uncertainty = self._bernoulli_entropy(soft_label)  # (B, 1, H, W)

        return soft_label, uncertainty

    @torch.no_grad()
    def diffusion_ensemble(self, images_u, coarse_pred, K=4, refine_step=80, num_inference_steps=20, seed_base=12345):
        """
        images_u: (B,3,H,W)
        coarse_pred: (B,1,H,W)  # teacher 输出概率mask
        return:
          p: (B,1,H,W) soft label
          u: (B,1,H,W) uncertainty (entropy)
        """
        self.teacher.eval()
        self.diffusion.eval()

        samples = []
        for k in range(K):
            torch.manual_seed(seed_base + k)  # 固定 refine_step，用 seed 控制多样性

            mk = self.diffusion_pipeline.sample(
                initial_prediction=coarse_pred,
                original_image=images_u,
                num_inference_steps=num_inference_steps,
                refine_step=refine_step,
            )
            samples.append(mk)

        samples = torch.stack(samples, dim=0)  # (K,B,1,H,W)
        p = samples.mean(dim=0)  # (B,1,H,W)

        # Bernoulli entropy
        eps = 1e-6
        p_clamp = p.clamp(eps, 1 - eps)
        u = -(p_clamp * torch.log(p_clamp) + (1 - p_clamp) * torch.log(1 - p_clamp))  # (B,1,H,W)

        return p, u

    def _update_ema_teacher(self):
        """EMA 更新 Teacher 参数"""
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(),
                                        self.student.parameters()):
                t_param.data = (self.ema_decay * t_param.data +
                                (1 - self.ema_decay) * s_param.data)

    def run_iterative_training(self, labeled_loader, unlabeled_loader, num_iterations=3):
        """迭代式训练（2-3轮）"""

        # 先在有标签数据上预训练
        print("=" * 60)
        print("预训练阶段：在有标签数据上训练基础模型")
        print("=" * 60)
        self._pretrain_on_labeled(labeled_loader)

        # 训练扩散模型
        print("=" * 60)
        print("训练扩散精修器")
        print("=" * 60)
        self._train_diffusion(labeled_loader)

        # 迭代式自训练
        for iteration in range(num_iterations):
            print("=" * 60)
            print(f"迭代 {iteration + 1}/{num_iterations}")
            print("=" * 60)

            self.train_iteration(
                labeled_loader,
                unlabeled_loader,
                num_epochs=self.config.epochs_per_iteration
            )

            # 评估
            val_iou = self._validate(labeled_loader)
            print(f"迭代 {iteration + 1} 完成，验证 IoU: {val_iou:.4f}")

        return self.student


def weak_augment(images):
    """弱增强：简单的翻转、小角度旋转"""
    # 随机水平翻转
    if random.random() > 0.5:
        images = torch.flip(images, dims=[3])
    return images


def strong_augment(images):
    """强增强：颜色抖动、随机裁剪、更大的几何变换"""
    # 颜色抖动
    images = images + torch.randn_like(images) * 0.1
    # 随机遮挡 (Cutout)
    h, w = images.shape[2], images.shape[3]
    mask_size = random.randint(50, 100)
    x = random.randint(0, w - mask_size)
    y = random.randint(0, h - mask_size)
    images[:, :, y:y + mask_size, x:x + mask_size] = 0
    return images


def weighted_bce_loss(pred, target, weight):
    """加权BCE损失"""
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    weighted_bce = bce * weight
    return weighted_bce.mean()


def masked_bce_loss(pred, target, mask):
    """只在mask区域计算BCE"""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    return (bce * mask).sum() / mask.sum()

def hierarchical_weighted_loss(pred, target):
    """ 分级加权的混合损失  """
    # Step1: 血管分级掩码生成（基于像素值/面积）
    target_erode = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)  # 主干血管
    target_dilate = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)  # 毛细血管
    trunk_mask = (target_erode == 1).float()
    branch_mask = (target - trunk_mask - target_dilate).clamp(0,1)
    cap_mask = target_dilate

    # Step2: Dice-BCE损失（基础损失）
    bce = F.binary_cross_entropy(pred, target)
    smooth = 1e-6
    intersection = (pred * target).sum()
    dice = 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    dice_bce = 0.5 * bce + 0.5 * dice

    # Step3: 分级IoU损失（毛细血管权重最高）
    def iou_loss(pred, mask, weight):
        inter = (pred * mask).sum()
        union = (pred + mask).sum() - inter
        return weight * (1 - (inter + smooth) / (union + smooth))

    trunk_iou = iou_loss(pred, trunk_mask, 0.2)
    branch_iou = iou_loss(pred, branch_mask, 0.3)
    cap_iou = iou_loss(pred, cap_mask, 0.5)  # 毛细血管权重最高
    hierarchical_iou = trunk_iou + branch_iou + cap_iou

    # Step4: 边缘损失（优化血管边缘）
    # 生成边缘掩码（Canny）
    target_np = target.cpu().detach().numpy()
    if target_np.ndim == 4:
        target_np = target_np[:, 0]
    elif target_np.ndim == 2:
        target_np = target_np[None, ...]
    edge_mask = np.zeros_like(target_np)
    for i in range(target_np.shape[0]):
        edge_mask[i] = cv2.Canny((target_np[i]*255).astype(np.uint8), 50, 150) / 255
    edge_mask = torch.tensor(edge_mask, dtype=torch.float32).to(pred.device).unsqueeze(1)
    edge_loss = F.binary_cross_entropy(pred, edge_mask)

    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    recall = (tp + smooth) / (tp + fn + smooth)
    recall_loss = 1 - recall

    # Step6: 新增 Soft-BACC loss，直接优化 BACC
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()

    specificity = (tn + smooth) / (tn + fp + smooth)
    bacc = 0.5 * (recall + specificity)
    bacc_loss = 1 - bacc

    # 总损失：加权融合
    # total_loss = 0.6 * dice_bce + 0.3 * hierarchical_iou + 0.1 * edge_loss
    # Step7: 总损失
    total_loss = (
        0.45 * dice_bce +
        0.20 * hierarchical_iou +
        0.10 * edge_loss +
        0.15 * recall_loss +
        0.10 * bacc_loss
    )
    return total_loss
