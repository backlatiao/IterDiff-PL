import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.transunet import TransUNet
import os
import sys
import cv2

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    target_np = target.cpu().detach().numpy().squeeze()
    edge_mask = np.zeros_like(target_np)
    for i in range(target_np.shape[0]):
        edge_mask[i] = cv2.Canny((target_np[i]*255).astype(np.uint8), 50, 150) / 255
    edge_mask = torch.tensor(edge_mask, dtype=torch.float32).to(pred.device).unsqueeze(1)
    edge_loss = F.binary_cross_entropy(pred, edge_mask)

    # 总损失：加权融合
    total_loss = 0.6 * dice_bce + 0.3 * hierarchical_iou + 0.1 * edge_loss
    return total_loss


class ResidualBlock(nn.Module):
    """残差块，集成时间嵌入"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 添加时间嵌入
        time_emb = self.time_emb_proj(F.silu(time_emb))[:, :, None, None]
        h = h + time_emb
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """自注意力块"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 重塑为注意力形式
        q = q.view(B, C, H * W).transpose(-1, -2)  # (B, H*W, C)
        k = k.view(B, C, H * W)  # (B, C, H*W)
        v = v.view(B, C, H * W)  # (B, C, H*W)
        
        # 计算注意力
        attn = torch.softmax(torch.bmm(q, k) / np.sqrt(C), dim=-1)
        out = torch.bmm(attn, v.transpose(-1, -2))  # (B, H*W, C)
        out = out.transpose(-1, -2).view(B, C, H, W)  # (B, C, H, W)
        
        out = self.proj(out)
        return x + out


class DownBlock(nn.Module):
    """下采样块"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.block1 = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.block2 = ResidualBlock(out_channels, out_channels, time_emb_dim)
        self.downsample = nn.MaxPool2d(2)
    
    def forward(self, x, time_emb):
        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        # in_channels = skip_channels + prev_channels
        mid_channels = in_channels // 2
        self.block1 = ResidualBlock(in_channels, mid_channels, time_emb_dim)
        self.block2 = ResidualBlock(mid_channels, out_channels, time_emb_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x, skip, time_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  # 拼接跳跃连接
        x = self.block1(x, time_emb)
        x = self.block2(x, time_emb)
        return x


class DiffusionNet(nn.Module):
    """
    扩散模型网络，用于血管分割细化
    输入：原始图像 + 预测掩码 + 带噪声的目标掩码（拼接）
    输出：预测的噪声
    """
    
    def __init__(self, 
                 img_size=512,
                 in_channels=5,  # 3个原始图像通道 + 1个预测掩码通道 + 1个带噪声目标掩码通道
                 out_channels=1,  # 预测噪声（针对目标掩码的噪声）
                 base_channels=64,
                 time_emb_dim=128,
                 condition_emb_dim=256):
        super(DiffusionNet, self).__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.time_emb_dim = time_emb_dim
        
        # 时间嵌入层
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # 主干网络 - 类似UNet结构
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3)
        
        # 下采样路径
        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_emb_dim)
        
        # 中间瓶颈层
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim),
            AttentionBlock(base_channels * 8),
            ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        ])
        
        # 上采样路径
        self.up3 = UpBlock(base_channels * 8 * 2, base_channels * 4, time_emb_dim)  # *2 因为有跳跃连接
        self.up2 = UpBlock(base_channels * 4 * 2, base_channels * 2, time_emb_dim)
        self.up1 = UpBlock(base_channels * 2 * 2, base_channels, time_emb_dim)
        
        # 输出层
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=1)
        )
    
    def _time_embedding(self, timesteps, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 != 0:
            emb = F.pad(emb, (0, 1))
        return emb
    
    def forward(self, noisy_target, original_image, predicted_mask, timesteps):
        # noisy_target: 带噪声的目标掩码 (B, 1, H, W)
        # original_image: 原始图像 (B, 3, H, W)
        # predicted_mask: 预测掩码 (B, 1, H, W)
        # timesteps: 时间步 (B,)
        
        # 拼接所有输入
        input_combined = torch.cat([original_image, predicted_mask, noisy_target], dim=1)  # (B, 5, H, W)
        
        # 获取时间嵌入
        time_emb = self._time_embedding(timesteps, self.time_emb_dim)
        time_emb = self.time_mlp(time_emb)  # (B, time_emb_dim)
        
        # 初始卷积
        h = self.init_conv(input_combined)
        
        # 下采样路径，收集跳跃连接
        h, skip1 = self.down1(h, time_emb)
        h, skip2 = self.down2(h, time_emb)
        h, skip3 = self.down3(h, time_emb)
        
        # 中间瓶颈
        for layer in self.mid_blocks:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)
        
        # 上采样路径
        h = self.up3(h, skip3, time_emb)
        h = self.up2(h, skip2, time_emb)
        h = self.up1(h, skip1, time_emb)
        
        # 输出预测的噪声（只针对目标掩码）
        output = self.out_conv(h)
        # 我们只关心预测掩码通道的噪声，所以只返回第一个输出通道
        return output[:, :1, :, :]  # (B, 1, H, W)


class DiffusionPipeline:
    """
    扩散模型管道，包括加噪、去噪过程
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.device = device
        self.num_timesteps = num_timesteps
        
        # 定义噪声调度
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
    
    def register_buffer(self, name, tensor):
        """注册缓冲区，将其移到适当的设备"""
        # 检查模型是否已经有该缓冲区，避免覆盖
        if hasattr(self.model, name):
            # 如果缓冲区已存在，则更新其值而不是重新注册
            getattr(self.model, name).copy_(tensor.to(self.device))
        else:
            # 否则，注册新的缓冲区
            self.model.register_buffer(name, tensor.to(self.device))
    
    def add_noise_based_on_vessels(self, clean_target, predicted_mask, timesteps):
        """
        根据血管位置添加不同程度的噪声
        在血管区域添加更多噪声，在背景区域添加较少噪声
        """
        noise = torch.randn_like(clean_target)
        
        # 根据预测掩码调整噪声强度
        # 血管区域(预测掩码为1)添加更多噪声，背景区域(预测掩码为0)添加较少噪声
        vessel_noise_factor = 1.0 + predicted_mask * 0.5  # 血管区域噪声强度增加50%
        adjusted_noise = noise * vessel_noise_factor
        
        sqrt_alpha_cumprod = self.model.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.model.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        noisy_target = sqrt_alpha_cumprod * clean_target + sqrt_one_minus_alpha_cumprod * adjusted_noise
        
        return noisy_target, adjusted_noise
    
    def forward_process(self, clean_target, predicted_mask, timesteps):
        """前向加噪过程"""
        return self.add_noise_based_on_vessels(clean_target, predicted_mask, timesteps)
    
    def reverse_process(self, noisy_target, original_image, predicted_mask, timesteps):
        """反向去噪过程"""
        self.model.eval()
        with torch.no_grad():
            predicted_noise = self.model(noisy_target, original_image, predicted_mask, timesteps)
        return predicted_noise
    
    def sample(self, initial_prediction, original_image, num_inference_steps=50, refine_step=100, eta=0.0):
        """
        从初始预测开始进行局部细化采样
        refine_step: 细化开始的时间步，越小保留的原始结构越多，建议 50-200
        """
        self.model.eval()
        
        # 1. 为初始预测添加对应 refine_step 强度的噪声
        t_start = torch.full((initial_prediction.shape[0],), refine_step, device=self.device, dtype=torch.long)
        x_t, _ = self.forward_process(initial_prediction, initial_prediction, t_start)
        
        # 2. 从 refine_step 逐步去噪到 0
        inference_timesteps = torch.linspace(
            refine_step, 0, num_inference_steps, dtype=torch.long, device=self.device
        )
        
        for i, t in enumerate(inference_timesteps):
            t_batch = torch.full((initial_prediction.shape[0],), t, device=self.device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.model(x_t, original_image, initial_prediction, t_batch)
            
            # 计算下一步的图像
            alpha_t = self.model.alphas[t]
            alpha_cumprod_t = self.model.alphas_cumprod[t]
            
            # 计算均值和方差
            if t > 0:
                # 获取上一个时间步
                prev_t = inference_timesteps[i+1] if i < len(inference_timesteps) - 1 else torch.tensor(0, device=self.device)
                alpha_cumprod_prev_t = self.model.alphas_cumprod[prev_t]
                
                # DDIM/DDPM 采样逻辑改进
                noise_scale = torch.sqrt((1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * (1 - alpha_t))
                mean = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_t)
                
                noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
                x_t = mean + noise_scale * noise
            else:
                x_t = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_t)
        
        return torch.clamp(x_t, 0, 1)


def get_pretrained_transunet_weights(model_path, model_class=TransUNet):
    """
    加载预训练的TransUNet权重
    """
    # 初始化模型
    pretrained_model = model_class(
        img_size=512,
        patch_size=16,
        in_channels=3,
        out_channels=1,
        embed_dim=512,
        num_heads=8,
        mlp_dim=2048,
        num_layers=6,
        features=[64, 128, 256, 512]
    )
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    pretrained_model.load_state_dict(checkpoint)
    return pretrained_model


def predict_with_transunet(transunet_model, image):
    """
    使用预训练的TransUNet生成预测掩码
    注意：TransUNet的OutConv已经包含sigmoid，不需要再次sigmoid
    """
    transunet_model.eval()
    with torch.no_grad():
        pred_mask = transunet_model(image)  # TransUNet已经包含sigmoid，直接返回[0,1]概率
        return pred_mask


if __name__ == "__main__":
    # 示例：如何使用扩散模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载预训练的TransUNet
    transunet_path = os.path.join(project_root, 'weights', 'best_model.pth')
    transunet_model = get_pretrained_transunet_weights(transunet_path)
    transunet_model.to(device)
    
    # 初始化扩散模型
    diffusion_model = DiffusionNet(
        img_size=512,
        in_channels=5,  # 3个原始图像通道 + 1个预测掩码通道 + 1个带噪声目标掩码通道
        out_channels=1  # 预测噪声
    ).to(device)
    
    print("扩散模型结构:")
    print(diffusion_model)
    
    # 测试前向传播
    batch_size = 1
    noisy_target = torch.randn(batch_size, 1, 512, 512).to(device)  # 带噪声的目标掩码
    original_image = torch.randn(batch_size, 3, 512, 512).to(device)  # 原始图像
    predicted_mask = torch.rand(batch_size, 1, 512, 512).to(device)  # 预测掩码
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)  # 时间步
    
    print(f"\n输入形状: noisy_target={noisy_target.shape}, original_image={original_image.shape}, predicted_mask={predicted_mask.shape}, timesteps={timesteps.shape}")
    
    # 测试模型前向传播
    output = diffusion_model(noisy_target, original_image, predicted_mask, timesteps)
    print(f"输出形状: {output.shape}")
    
    print("\n扩散模型创建成功！")
