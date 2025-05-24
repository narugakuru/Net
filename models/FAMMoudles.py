import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数式API模块
import numpy as np  # 导入NumPy库

# from .encoder import Res101Encoder  # 从当前包导入Res50Encoder类（被注释掉）
# from utils import set_logger

# logger = set_logger()


class FAM_Optimized(nn.Module):  # 优化后的FAM模块

    def __init__(self, feature_dim=512):
        """
        优化后的FAM模块，直接处理2D图像特征
        参数:
            feature_dim: 特征维度，默认512
        """
        super(FAM_Optimized, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def forward(self, image_features):
        """
        直接对2D图像特征进行频率分离

        参数:
            image_features: torch.Size([1, 1, 512, 64, 64]) 或 torch.Size([1, 512, 64, 64])

        返回:
            tuple: (low_freq, mid_freq, high_freq) 每个都是 [B, C, H, W] 格式
        """
        # 处理输入维度 - 支持4D和5D输入
        if image_features.dim() == 5:  # [1, 1, 512, 64, 64]
            B, _, C, H, W = image_features.shape
            image_features = image_features.squeeze(1)  # -> [1, 512, 64, 64]
        elif image_features.dim() == 4:  # [1, 512, 64, 64]
            B, C, H, W = image_features.shape
        else:
            raise ValueError(f"Expected 4D or 5D input, got {image_features.dim()}D")

        # 直接对2D特征进行频率分离
        low_freq, mid_freq, high_freq = self.filter_frequency_bands_2d(
            image_features, cutoff=0.30
        )

        return low_freq, mid_freq, high_freq

    def filter_frequency_bands_2d(self, tensor, cutoff=0.3):
        """
        直接对2D图像特征进行频率带分离

        参数:
            tensor: [B, C, H, W] 格式的2D图像特征
            cutoff: 频率分离的截止比例

        返回:
            low_freq, mid_freq, high_freq: 分离后的频率特征
        """
        tensor = tensor.float()
        B, C, H, W = tensor.shape

        # 计算频率域的最大半径
        max_radius = np.sqrt((H // 2) ** 2 + (W // 2) ** 2)
        low_cutoff = max_radius * cutoff  # 低频截止
        high_cutoff = max_radius * (1 - cutoff)  # 高频截止

        # 2D FFT变换
        fft_tensor = torch.fft.fftshift(
            torch.fft.fft2(tensor, dim=(-2, -1)), dim=(-2, -1)
        )

        # 创建频率滤波器
        low_pass_filter = self.create_frequency_filter(
            (H, W), low_cutoff, None, mode="low"
        )[
            None, None, :, :
        ]  # [1, 1, H, W]

        high_pass_filter = self.create_frequency_filter(
            (H, W), None, high_cutoff, mode="high"
        )[None, None, :, :]

        mid_pass_filter = self.create_frequency_filter(
            (H, W), low_cutoff, high_cutoff, mode="band"
        )[None, None, :, :]

        # 应用滤波器
        low_freq_fft = fft_tensor * low_pass_filter
        high_freq_fft = fft_tensor * high_pass_filter
        mid_freq_fft = fft_tensor * mid_pass_filter

        # 反FFT变换，保持2D格式
        low_freq_tensor = torch.fft.ifft2(
            torch.fft.ifftshift(low_freq_fft, dim=(-2, -1)), dim=(-2, -1)
        ).real

        high_freq_tensor = torch.fft.ifft2(
            torch.fft.ifftshift(high_freq_fft, dim=(-2, -1)), dim=(-2, -1)
        ).real

        mid_freq_tensor = torch.fft.ifft2(
            torch.fft.ifftshift(mid_freq_fft, dim=(-2, -1)), dim=(-2, -1)
        ).real

        return low_freq_tensor, mid_freq_tensor, high_freq_tensor

    def create_frequency_filter(self, shape, low_cutoff, high_cutoff, mode="band"):
        """
        创建频率域滤波器

        参数:
            shape: (H, W) 图像尺寸
            low_cutoff: 低频截止值
            high_cutoff: 高频截止值
            mode: 滤波器类型 ("low", "high", "band")

        返回:
            mask: 频率滤波器掩码
        """
        rows, cols = shape
        center_row, center_col = rows // 2, cols // 2

        # 创建距离矩阵
        y, x = torch.meshgrid(
            torch.arange(rows, device=self.device),
            torch.arange(cols, device=self.device),
            indexing="ij",
        )
        distance = torch.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)

        # 创建滤波器掩码
        mask = torch.zeros((rows, cols), dtype=torch.float32, device=self.device)

        if mode == "low":
            mask[distance <= low_cutoff] = 1
        elif mode == "high":
            mask[distance >= high_cutoff] = 1
        elif mode == "band":
            mask[(distance > low_cutoff) & (distance < high_cutoff)] = 1

        return mask


class CrossAttentionFusion_2D(nn.Module):
    """
    适配2D特征的交叉注意力融合模块
    """

    def __init__(self, embed_dim):
        super(CrossAttentionFusion_2D, self).__init__()
        self.query = nn.Conv2d(embed_dim, embed_dim, 1)  # 1x1卷积替代线性层
        self.key = nn.Conv2d(embed_dim, embed_dim, 1)
        self.value = nn.Conv2d(embed_dim, embed_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q_feature, K_feature):
        """
        参数:
            Q_feature, K_feature: [B, C, H, W]
        返回:
            attended_features: [B, C, H, W]
        """
        B, C, H, W = Q_feature.shape

        # 生成查询、键、值
        Q = self.query(Q_feature).view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        K = self.key(K_feature).view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        V = self.value(K_feature).view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

        # 计算注意力
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(C)
        attention_weights = self.softmax(attention_scores)  # [B, HW, HW]

        # 应用注意力
        attended_features = torch.matmul(attention_weights, V)  # [B, HW, C]
        attended_features = attended_features.permute(0, 2, 1).view(B, C, H, W)

        return attended_features


class EfficientCrossAttentionFusion_2D(nn.Module):
    """
    高效的交叉注意力融合模块
    """

    def __init__(self, embed_dim, reduction_ratio=4):
        super(EfficientCrossAttentionFusion_2D, self).__init__()
        reduced_dim = embed_dim // reduction_ratio

        self.query = nn.Sequential(nn.Conv2d(embed_dim, reduced_dim, 1), nn.ReLU())
        self.key = nn.Sequential(nn.Conv2d(embed_dim, reduced_dim, 1), nn.ReLU())
        self.value = nn.Conv2d(embed_dim, embed_dim, 1)
        self.output_proj = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, Q_feature, K_feature):
        """
        使用降维的高效注意力计算
        Q=mid
        K,V=low/high
        """
        B, C, H, W = Q_feature.shape

        # 降维的查询和键
        Q = self.query(Q_feature).view(B, -1, H * W)  # [B, C//4, HW]
        K = self.key(K_feature).view(B, -1, H * W)  # [B, C//4, HW]
        V = self.value(K_feature).view(B, C, H * W)  # [B, C, HW]

        # 计算注意力权重
        attention_scores = torch.bmm(Q.transpose(1, 2), K) / np.sqrt(
            Q.size(1)
        )  # [B, HW, HW]
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 应用注意力
        attended_features = torch.bmm(
            V, attention_weights.transpose(1, 2)
        )  # [B, C, HW]
        attended_features = attended_features.view(B, C, H, W)

        return self.output_proj(attended_features)


class MSFM_2D(nn.Module):
    """
    适配2D特征的多尺度特征融合模块
    """

    def __init__(self, feature_dim):
        super(MSFM_2D, self).__init__()
        self.CA1 = EfficientCrossAttentionFusion_2D(feature_dim)
        self.CA2 = EfficientCrossAttentionFusion_2D(feature_dim)
        self.relu = nn.ReLU()

    def forward(self, low, mid, high):
        """
        参数:
            low, mid, high: [B, C, H, W] 格式的特征
        返回:
            fused_features: [B, C, H, W]
        """
        low_new = self.CA1(mid, low)
        high_new = self.CA2(mid, high)
        fused_features = self.relu(low_new + mid + high_new)

        return fused_features


class FADAM_2D_Optimized(nn.Module):
    """
    显存优化版本的FADAM模块
    """

    def __init__(self, feature_dim=512, target_size=32):
        super(FADAM_2D_Optimized, self).__init__()
        self.FAM = FAM_Optimized(feature_dim=feature_dim)
        self.MSFM = MSFM_2D(feature_dim=feature_dim)
        self.target_size = target_size

        # 添加降维和升维模块
        self.downsample = nn.AdaptiveAvgPool2d((target_size, target_size))
        self.upsample = nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False)

    def forward(self, image_features):
        """
        参数:
            image_features: [1, 1, 512, 64, 64] 或 [1, 512, 64, 64]
        """
        # 处理输入维度
        if image_features.dim() == 5:
            _, B, C, H, W = image_features.shape
            image_features = image_features.squeeze(1)
        else:
            B, C, H, W = image_features.shape

        # 降维到32x32进行处理
        downsampled_features = self.downsample(image_features)  # [B, C, 32, 32]

        # FAM频率分离
        low_freq, mid_freq, high_freq = self.FAM(downsampled_features)

        # MSFM特征融合（在32x32尺寸下）
        fused_features = self.MSFM(low_freq, mid_freq, high_freq)

        # 上采样回64x64
        output_features = self.upsample(fused_features)  # [B, C, 64, 64]

        # 调整输出格式
        output = output_features.unsqueeze(0).unsqueeze(0)  # [1, 1, B, C, H, W]

        return output
