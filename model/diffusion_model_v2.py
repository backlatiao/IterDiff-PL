import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.transunet import TransUNet
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import sys
import cv2

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

class VesselPriorExtractor(nn.Module):
    """
    可解释血管先验模块
    输出 3 通道：
      - vesselness: 管状性响应（粗略）
      - dir_x, dir_y: 方向场（由 Sobel 梯度归一化）
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        laplace = torch.tensor([[0,  1, 0],
                                [1, -4, 1],
                                [0,  1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplace", laplace)

        # 可学习强度参数（仍保留结构可解释）
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))

    def forward(self, img_rgb: torch.Tensor) -> torch.Tensor:
        # 转灰度
        gray = 0.2989 * img_rgb[:, 0:1] + 0.5870 * img_rgb[:, 1:2] + 0.1140 * img_rgb[:, 2:3]

        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        gmag = torch.sqrt(gx * gx + gy * gy + self.eps)

        dir_x = gx / (gmag + self.eps)
        dir_y = gy / (gmag + self.eps)

        lap = F.conv2d(gray, self.laplace, padding=1).abs()
        vesselness = torch.sigmoid(self.a * lap - self.b * gmag)

        return torch.cat([vesselness, dir_x, dir_y], dim=1)  # (B,3,H,W)


class CondToFiLM(nn.Module):
    """
    条件 -> FiLM 参数 (gamma, beta)
    cond_full_res: (B, cond_ch, H, W)
    输出 gamma,beta: (B, bottleneck_ch, H/8, W/8)  (与之前的 3 次 MaxPool 对齐)
    """
    def __init__(self, cond_in_ch: int, bottleneck_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(cond_in_ch, bottleneck_ch // 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, bottleneck_ch // 4),
            nn.SiLU(),
            nn.Conv2d(bottleneck_ch // 4, bottleneck_ch // 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, bottleneck_ch // 2),
            nn.SiLU(),
            nn.Conv2d(bottleneck_ch // 2, bottleneck_ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, bottleneck_ch),
            nn.SiLU(),
        )
        self.to_gamma = nn.Conv2d(bottleneck_ch, bottleneck_ch, 1)
        self.to_beta  = nn.Conv2d(bottleneck_ch, bottleneck_ch, 1)

    def forward(self, cond_full_res: torch.Tensor):
        z = self.down(cond_full_res)
        gamma = torch.tanh(self.to_gamma(z))  # 有界更稳定
        beta  = self.to_beta(z)
        return gamma, beta

class DiffusionNet(nn.Module):
    """
    扩散模型网络，用于血管分割细化（改进版：缝合先验 / grafting / FiLM）
    输出仍然只预测噪声 ε̂（不改变扩散范式与接口）
    """
    def __init__(self,
                 img_size=512,
                 in_channels=5,
                 out_channels=1,
                 base_channels=64,
                 time_emb_dim=128,
                 condition_emb_dim=256,

                 # three graft switches =====
                 use_vessel_prior: bool = True,           # 缝合血管先验分支
                 use_trans_graft: bool = True,            # 缝合 TransUNet 多尺度特征
                 use_film_fusion: bool = True,            # FiLM 条件调制（显式融合块）
                 trans_feat_channels: Tuple[int, int, int, int] = (64, 128, 256, 512)  # TransUNet taps 通道
                 ):
        super(DiffusionNet, self).__init__()

        self.img_size = img_size
        self.time_emb_dim = time_emb_dim

        self.use_vessel_prior = use_vessel_prior
        self.use_trans_graft = use_trans_graft
        self.use_film_fusion = use_film_fusion

        # prior module =====
        self.prior = VesselPriorExtractor() if self.use_vessel_prior else None
        prior_ch = 3 if self.use_vessel_prior else 0

        # ===== 输入通道：baseline 5ch + prior 3ch =====
        expected_in = 5 + prior_ch
        # 兼容原来传 in_channels=5 的写法：如果打开 prior，就自动扩展输入通道
        self.in_channels = expected_in

        # 时间嵌入层（保持原写法）
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )

        # init conv（输入通道改为 expected_in）
        self.init_conv = nn.Conv2d(expected_in, base_channels, kernel_size=7, padding=3)

        # 下采样路径（保持）
        self.down1 = DownBlock(base_channels, base_channels * 2, time_emb_dim)  # skip1: 2C, H, W
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)  # skip2: 4C, H/2, W/2
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, time_emb_dim)  # skip3: 8C, H/4, W/4
        # h after down3: 8C, H/8, W/8

        # TransUNet multi-scale graft adapters =====
        # 将 TransUNet 的 F1..F4 注入到 skip1/skip2/skip3/h (bottleneck input)
        if self.use_trans_graft:
            self.trans_adapt_1 = nn.Conv2d(trans_feat_channels[0], base_channels * 2, 1)  # -> skip1
            self.trans_adapt_2 = nn.Conv2d(trans_feat_channels[1], base_channels * 4, 1)  # -> skip2
            self.trans_adapt_3 = nn.Conv2d(trans_feat_channels[2], base_channels * 8, 1)  # -> skip3
            self.trans_adapt_4 = nn.Conv2d(trans_feat_channels[3], base_channels * 8, 1)  # -> h (H/8)

        # FiLM fusion block at bottleneck =====
        if self.use_film_fusion:
            # cond: P(1) + x_t(1) + prior(3 optional)
            cond_in_ch = 2 + prior_ch
            self.cond_film = CondToFiLM(cond_in_ch=cond_in_ch, bottleneck_ch=base_channels * 8)

        # 中间瓶颈层（保持：Residual + Attention + Residual）
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim),
            AttentionBlock(base_channels * 8),
            ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        ])

        # 上采样路径（保持）
        self.up3 = UpBlock(base_channels * 8 * 2, base_channels * 4, time_emb_dim)
        self.up2 = UpBlock(base_channels * 4 * 2, base_channels * 2, time_emb_dim)
        self.up1 = UpBlock(base_channels * 2 * 2, base_channels, time_emb_dim)

        # 输出层（保持：输出 ε̂）
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

    @staticmethod
    def _maybe_resize(feat: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """把 graft 特征插值到 ref 的空间分辨率（避免尺寸不一致）"""
        if feat.shape[2:] != ref.shape[2:]:
            feat = F.interpolate(feat, size=ref.shape[2:], mode='bilinear', align_corners=False)
        return feat

    def forward(self,
                noisy_target,
                original_image,
                predicted_mask,
                timesteps,
                trans_feats: Optional[List[torch.Tensor]] = None):
        # noisy_target: 带噪声的目标掩码 (B, 1, H, W)
        # original_image: 原始图像 (B, 3, H, W)
        # predicted_mask: 预测掩码 (B, 1, H, W)
        # timesteps: 时间步 (B,)

        # 拼接所有输入
        feats = [original_image, predicted_mask, noisy_target]  # baseline 5ch
        if self.use_vessel_prior:
            prior = self.prior(original_image)  # (B,3,H,W)
            feats.append(prior)

        input_combined = torch.cat(feats, dim=1)  # (B, 5(+3), H, W)

        # 获取时间嵌入
        time_emb = self._time_embedding(timesteps, self.time_emb_dim)
        time_emb = self.time_mlp(time_emb)

        # 初始卷积
        h = self.init_conv(input_combined)

        # 下采样路径，收集跳跃连接
        h, skip1 = self.down1(h, time_emb)
        h, skip2 = self.down2(h, time_emb)
        h, skip3 = self.down3(h, time_emb)

        # TransUNet 多尺度特征缝合 (inject into skips + bottleneck input) =====
        if self.use_trans_graft and trans_feats is not None and len(trans_feats) == 4:
            f1, f2, f3, f4 = trans_feats

            if f1 is not None:
                f1 = self._maybe_resize(f1, skip1)
                skip1 = skip1 + self.trans_adapt_1(f1)

            if f2 is not None:
                f2 = self._maybe_resize(f2, skip2)
                skip2 = skip2 + self.trans_adapt_2(f2)

            if f3 is not None:
                f3 = self._maybe_resize(f3, skip3)
                skip3 = skip3 + self.trans_adapt_3(f3)

            if f4 is not None:
                f4 = self._maybe_resize(f4, h)
                h = h + self.trans_adapt_4(f4)

        # FILM条件调制
        if self.use_film_fusion:
            cond_parts = [predicted_mask, noisy_target]
            if self.use_vessel_prior:
                cond_parts.append(prior)
            cond_full = torch.cat(cond_parts, dim=1)  # (B, 2(+3), H, W)
            gamma, beta = self.cond_film(cond_full)   # (B, 8C, H/8, W/8)
            h = (1.0 + gamma) * h + beta

        # mid blocks
        for layer in self.mid_blocks:
            if isinstance(layer, ResidualBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)

        # up path (use (possibly grafted) skips)
        h = self.up3(h, skip3, time_emb)
        h = self.up2(h, skip2, time_emb)
        h = self.up1(h, skip1, time_emb)

        # output eps
        output = self.out_conv(h)
        return output[:, :1, :, :]

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
    
    def reverse_process(self, noisy_target, original_image, predicted_mask, timesteps, trans_feats=None):
        """反向去噪过程"""
        self.model.eval()
        with torch.no_grad():
            predicted_noise = self.model(noisy_target, original_image, predicted_mask, timesteps, trans_feats=trans_feats)
        return predicted_noise
    
    def sample(self, initial_prediction, original_image, num_inference_steps=50, refine_step=100, trans_feats=None):
        """
        从初始预测开始进行局部细化采样
        trans_feats: 可选，来自 TransUNet 的多尺度特征，用于 grafting
        """
        self.model.eval()

        # 1. 为初始预测添加对应 refine_step 强度的噪声
        t_start = torch.full((initial_prediction.shape[0],), refine_step, device=self.device, dtype=torch.long)
        x_t, _ = self.forward_process(initial_prediction, initial_prediction, t_start)

        # 2. 从 refine_step 逐步去噪到 0
        inference_timesteps = torch.linspace(refine_step, 0, num_inference_steps, dtype=torch.long, device=self.device)

        for i, t in enumerate(inference_timesteps):
            t_batch = torch.full((initial_prediction.shape[0],), t, device=self.device, dtype=torch.long)

            # 预测噪声
            predicted_noise = self.model(
                x_t, original_image, initial_prediction, t_batch, trans_feats=trans_feats
            )

            # 计算下一步的图像
            alpha_t = self.model.alphas[t]
            alpha_cumprod_t = self.model.alphas_cumprod[t]

            if t > 0:
                # 获取上一个时间步
                prev_t = inference_timesteps[i+1] if i < len(inference_timesteps) - 1 else torch.tensor(0, device=self.device)
                alpha_cumprod_prev_t = self.model.alphas_cumprod[prev_t]

                # DDIM/DDPM 采样逻辑
                noise_scale = torch.sqrt((1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * (1 - alpha_t))
                mean = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_t)

                noise = torch.randn_like(x_t)
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


def predict_with_transunet(transunet_model, image, return_features=False,
                           feat_channels=(64, 128, 256, 512),
                           feat_sizes=((512, 512), (256, 256), (128, 128), (64, 64))):
    """
    使用预训练的TransUNet生成预测掩码
    return_features=True 时，返回 (pred_mask, [F1,F2,F3,F4])
    说明：
      - 如果 TransUNet 内部模块命名不确定，此方法用 forward hooks 自动抓取
      - 抓取规则：输出是 4D Tensor，且 (C, H, W) 匹配 feat_channels & feat_sizes
    """
    transunet_model.eval()

    if not return_features:
        with torch.no_grad():
            pred_mask = transunet_model(image)
        return pred_mask

    # --- 自动抓取多尺度特征 ---
    collected = {sz: None for sz in feat_sizes}
    handles = []

    def make_hook():
        def hook_fn(module, inp, out):
            if not isinstance(out, torch.Tensor):
                return
            if out.dim() != 4:
                return
            B, C, H, W = out.shape
            for (ch, sz) in zip(feat_channels, feat_sizes):
                if (C == ch) and ((H, W) == sz) and (collected[sz] is None):
                    collected[sz] = out
        return hook_fn

    # 只 hook Conv2d，够用且开销较小
    for m in transunet_model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(make_hook()))

    with torch.no_grad():
        pred_mask = transunet_model(image)

    for h in handles:
        h.remove()

    feats = [collected[feat_sizes[0]], collected[feat_sizes[1]],
             collected[feat_sizes[2]], collected[feat_sizes[3]]]

    return pred_mask, feats
