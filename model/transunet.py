import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

import cv2
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


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class MBlock(nn.Module):
    def __init__(self, c, out, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self._conv = nn.Sequential(

            nn.Conv2d(c, out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return self._conv(y + x * self.gamma)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class PatchEmbedding(nn.Module):
    """将图像分割为补丁并进行嵌入"""
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        # 可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.proj(x)  # (b, num_patches, embed_dim)

        # 添加分类令牌
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)  # (b, num_patches + 1, embed_dim)

        # 添加位置编码
        x += self.pos_embedding

        return x

# =====================   分层自适应补丁嵌入模块 =====================
class HierarchicalAdaptivePatchEmbedding(nn.Module):
    """分层自适应补丁嵌入模块（HierarchicalAdaptivePatchEmbedding）"""
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 分层多尺度补丁投影（ 自适应逻辑）
        patch_scales = [patch_size, patch_size//2, patch_size*2]  # 基础/细/粗尺度patch
        self.patch_projs = nn.ModuleList([
            nn.Conv2d(in_channels, embed_dim//len(patch_scales), kernel_size=s, stride=s)
            for s in patch_scales
        ])

        # 分层特征融合卷积
        self.fusion_conv = nn.Conv2d(510, embed_dim, kernel_size=1, padding=0)

        # 保持与原PatchEmbedding一致的输出结构（cls_token + pos_embedding）
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. 多尺度补丁投影 + 统一尺寸
        patch_feats = []
        target_h, target_w = h // self.patch_size, w // self.patch_size  # 基础尺度空间尺寸
        for proj in self.patch_projs:
            feat = proj(x)
            # 上采样/下采样到基础尺度尺寸，保证特征维度一致
            feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=True)
            patch_feats.append(feat)

        # 2. 拼接并融合分层特征
        x = torch.cat(patch_feats, dim=1)
        x = self.fusion_conv(x)

        # 3. 重塑为序列（与原PatchEmbedding输出格式一致）
        x = rearrange(x, 'b c h w -> b (h w) c')

        # 4. 添加cls_token和位置编码（完全复用原逻辑）
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding

        return x
# ======================================================================

class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, dim, num_heads, dropout=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == dim, "嵌入维度必须能被头数整除"

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape

        # 计算Q、K、V
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 应用注意力
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)

        return out

class TransformerEncoder(nn.Module):
    """Transformer编码器块"""
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# =====================  分层特征校准跳跃连接模块 =====================
class HierarchicalFeatureCalibratedSkip(nn.Module):
    """分层特征校准跳跃连接模块 """
    def __init__(self, in_channels):
        super().__init__()
        # 通道注意力校准：对拼接后的跳跃特征做通道加权
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力校准：对拼接后的跳跃特征做空间加权
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        # 分层权重融合参数（可学习）
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 通道注意力加权
        ca_weight = self.channel_att(x)
        x_ca = x * ca_weight
        # 空间注意力加权
        sa_weight = self.spatial_att(x_ca)
        x_sa = x_ca * sa_weight
        # 分层加权融合（校准后特征 + 原始特征）
        out = self.gamma * x_sa + (1 - self.gamma) * x
        return out
# ======================================================================

class Up(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 确保中间通道数与输入通道数匹配
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
        self.feature_calibrator = HierarchicalFeatureCalibratedSkip(in_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 输入大小可能不匹配，进行调整
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.feature_calibrator(x)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))

# TransUNet 模型
class TransUNet(nn.Module):
    def __init__(self,
                 img_size=512,
                 patch_size=16,
                 in_channels=3,
                 out_channels=1,
                 # 关键调整：确保embed_dim与最高级特征通道数匹配
                 embed_dim=512,
                 num_heads=8,
                 mlp_dim=2048,
                 num_layers=6,   # 适度减少层数，提高稳定性
                 # 特征通道设计：保持逐级翻倍的规律
                 features=[64, 128, 256, 512]):
        super(TransUNet, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # 卷积编码器（用于生成跳跃连接特征）
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            MBlock(features[0], features[1])
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            MBlock(features[1], features[2])
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            MBlock(features[2], features[3])
        )

        # =====================  分层自适应补丁嵌入 =====================
        self.patch_embed = HierarchicalAdaptivePatchEmbedding(
            image_size=img_size//8,  # 经过3次下采样后: 512 -> 64
            patch_size=patch_size//8,  # 16 -> 2
            in_channels=features[3],  # 输入通道是最高级特征通道数
            embed_dim=embed_dim  # 确保与最高级特征通道数匹配
        )

        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])

        # 卷积解码器 - 精确计算每个阶段的输入通道数
        # up1输入: features[3](transformer输出) + features[2](编码器特征)
        self.up1 = Up(features[3] + features[2], features[2])
        # up2输入: features[2] + features[1]
        self.up2 = Up(features[2] + features[1], features[1])
        # up3输入: features[1] + features[0]
        self.up3 = Up(features[1] + features[0], features[0])
        self.outc = OutConv(features[0], out_channels)

        # 确保Transformer输出通道与最高级特征通道匹配
        self.transformer_conv = nn.Conv2d(
            embed_dim,
            features[3],
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        # 卷积编码器前向传播（跟踪通道数变化）
        x1 = self.inc(x)          # (b, 64, 512, 512)
        x2 = self.down1(x1)       # (b, 128, 256, 256)
        x3 = self.down2(x2)       # (b, 256, 128, 128)
        x4 = self.down3(x3)       # (b, 512, 64, 64)

        # Transformer 编码器处理
        transformer_input = self.patch_embed(x4)  # (b, num_patches + 1, 512)

        for encoder in self.transformer_encoders:
            transformer_input = encoder(transformer_input)

        # 移除分类令牌并重塑为特征图
        transformer_output = transformer_input[:, 1:, :]  # (b, num_patches, 512)
        h = w = self.img_size // 8 // (self.patch_size // 8)  # 512//8=64, 16//8=2, 64//2=32
        transformer_output = rearrange(transformer_output, 'b (h w) c -> b c h w', h=h, w=w)

        # 转换为与x4相同通道数的特征图 (512)
        transformer_output = self.transformer_conv(transformer_output)  # (b, 512, 32, 32)

        # 解码器前向传播（每个步骤都检查通道数）
        # x3(256) + transformer_output(512) = 768 输入到up1
        x = self.up1(transformer_output, x3)  # (b, 256, 64, 64)
        # x2(128) + 上一步输出(256) = 384 输入到up2
        x = self.up2(x, x2)                   # (b, 128, 128, 128)
        # x1(64) + 上一步输出(128) = 192 输入到up3
        x = self.up3(x, x1)                   # (b, 64, 256, 256)

        # 输出分割结果
        logits = self.outc(x)  # (b, 1, 512, 512)

        return logits