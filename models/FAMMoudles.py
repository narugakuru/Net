import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数式API模块
import numpy as np  # 导入NumPy库

# from .encoder import Res101Encoder  # 从当前包导入Res50Encoder类（被注释掉）
from models.encoder import Res101Encoder  # 从models.encoder模块导入Res101Encoder类
from utils import set_logger

logger = set_logger()

class AttentionMacthcing(nn.Module):  # 定义AttentionMacthcing类，继承自nn.Module
    def __init__(self, feature_dim=512, seq_len=5000):  # 初始化方法
        super(AttentionMacthcing, self).__init__()  # 调用父类的初始化方法
        self.fc_spt = nn.Sequential(  # 定义用于空间特征的全连接层序列
            nn.Linear(seq_len, seq_len // 10),  # 第一个全连接层（序列长度到长度的1/10）
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(seq_len // 10, seq_len),  # 第二个全连接层（1/10到原序列长度）
        )
        self.fc_qry = nn.Sequential(  # 定义用于查询特征的全连接层序列
            nn.Linear(seq_len, seq_len // 10),  # 第一个全连接层
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(seq_len // 10, seq_len),  # 第二个全连接层
        )
        self.fc_fusion = nn.Sequential(  # 定义融合特征的全连接层序列
            nn.Linear(
                seq_len * 2, seq_len // 5
            ),  # 第一个全连接层，输入为两个序列的长度
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(seq_len // 5, 2 * seq_len),  # 第二个全连接层
        )
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def correlation_matrix(self, spt_fg_fts, qry_fg_fts):  # 计算相关矩阵的方法
        """
        计算空间前景特征与查询前景特征之间的相关矩阵。

        参数:
            spt_fg_fts (torch.Tensor): 空间前景特征。
            qry_fg_fts (torch.Tensor): 查询前景特征。

        返回:
            torch.Tensor: 余弦相似度矩阵。形状: [1, 1, N]。
        """

        spt_fg_fts = F.normalize(
            spt_fg_fts, p=2, dim=1
        )  # 对空间前景特征进行L2归一化，形状 [1, 512, 900]
        qry_fg_fts = F.normalize(
            qry_fg_fts, p=2, dim=1
        )  # 对查询前景特征进行L2归一化，形状 [1, 512, 900]

        cosine_similarity = torch.sum(  # 计算余弦相似度
            spt_fg_fts * qry_fg_fts, dim=1, keepdim=True
        )  # 形状: [1, 1, N]

        return cosine_similarity  # 返回余弦相似度矩阵

    def forward(self, spt_fg_fts, qry_fg_fts, band):  # 前向传播方法
        """
        参数:
            spt_fg_fts (torch.Tensor): 空间前景特征。
            qry_fg_fts (torch.Tensor): 查询前景特征。
            band (str): 频带类型，'low'，'high'或其他。

        返回:
            torch.Tensor: 融合后的张量。形状: [1, 512, 5000]。
        """

        spt_proj = F.relu(
            self.fc_spt(spt_fg_fts)
        )  # 使用全连接层对空间特征进行投影，形状: [1, 512, 900]
        qry_proj = F.relu(
            self.fc_qry(qry_fg_fts)
        )  # 使用全连接层对查询特征进行投影，形状: [1, 512, 900]

        similarity_matrix = self.sigmoid(  # 计算余弦相似度并应用sigmoid激活
            self.correlation_matrix(spt_fg_fts, qry_fg_fts)
        )

        if band == "low" or band == "high":  # 判断频带类型
            weighted_spt = (1 - similarity_matrix) * spt_proj  # 根据相似度加权空间特征
            weighted_qry = (1 - similarity_matrix) * qry_proj  # 根据相似度加权查询特征
        else:  # 其他频带类型
            weighted_spt = similarity_matrix * spt_proj  # 根据相似度加权空间特征
            weighted_qry = similarity_matrix * qry_proj  # 根据相似度加权查询特征

        combined = torch.cat(  # 将加权后的空间特征与查询特征拼接
            (weighted_spt, weighted_qry), dim=2
        )  # 形状: [1, 1024, 900]
        fused_tensor = F.relu(
            self.fc_fusion(combined)
        )  # 使用全连接层对拼接结果进行融合，形状: [1, 512, 900]

        return fused_tensor  # 返回融合后的张量


class FAM(nn.Module):  # 定义FAM类（特征注意力匹配模块）

    def __init__(self, feature_dim=512, N=1024):  # 初始化方法
        """
        参数:
            feature_dim 在这里没有使用到，只是为了保持一致
            N 池化层，AttentionMacthcing的序列长度控制参数
        返回:

        """
        super(FAM, self).__init__()  # 调用父类的初始化方法
        if torch.cuda.is_available():  # 检查CUDA是否可用
            self.device = torch.device("cuda")  # 如果可用，设置为CUDA设备
        else:
            self.device = torch.device("cpu")  # 否则，设置为CPU设备

        self.attention_matching = AttentionMacthcing(
            feature_dim, N
        )  # 实例化特征注意力匹配模块
        self.adapt_pooling = nn.AdaptiveAvgPool1d(N)  # 定义自适应平均池化层

    def forward(self, spt_fg_fts, qry_fg_fts):  # 前向传播方法
        """
        前向传播FAM模块。

        参数:
            # spt_fg_fts torch.Size([1, 512, 5650]) qry_fg_fts torch.Size([1, 512, 65536])
            spt_fg_fts (list): 空间前景特征的列表。
            qry_fg_fts (list): 查询前景特征的列表。
            torch.Size([1, 1, 1, 512, 900]) --> torch.Size([1, 512, 900])

        返回:
            tuple: 由融合后的低频、中频和高频特征组成的元组。
        """
        if qry_fg_fts[0].shape[2] == 0:  # 检查查询前景特征的形状
            qry_fg_fts[0] = F.pad(qry_fg_fts[0], (0, 1))  # 填充特征以避免尺寸为零

        spt_fg_fts = [
            [self.adapt_pooling(fts) for fts in way] for way in spt_fg_fts
        ]  # 自适应池化空间前景特征
        qry_fg_fts = [
            self.adapt_pooling(fts) for fts in qry_fg_fts
        ]  # 自适应池化查询前景特征

        # 获取高中低频 全部都是torch.Size([1, 512, 900])
        spt_fg_fts_low, spt_fg_fts_mid, spt_fg_fts_high = (
            self.filter_frequency_bands(  # 过滤空间前景特征的频带
                spt_fg_fts[0][0], cutoff=0.30
            )
        )
        qry_fg_fts_low, qry_fg_fts_mid, qry_fg_fts_high = (
            self.filter_frequency_bands(  # 过滤查询前景特征的频带
                qry_fg_fts[0], cutoff=0.30
            )
        )

        # torch.Size([1, 512, 1800])
        fused_fts_low = self.attention_matching(
            spt_fg_fts_low, qry_fg_fts_low, "low"
        )  # 低频特征的融合
        fused_fts_mid = self.attention_matching(
            spt_fg_fts_mid, qry_fg_fts_mid, "mid"
        )  # 中频特征的融合
        fused_fts_high = self.attention_matching(  # 高频特征的融合
            spt_fg_fts_high, qry_fg_fts_high, "high"
        )

        # 和clip文本特征做对齐
        # spt_fg_fts torch.Size([1, 512, 900])
        # fused_fts torch.Size([1, 512, 1800])

        return (
            fused_fts_low,
            fused_fts_mid,
            fused_fts_high,
        )  # 返回融合后的低、中、高频特征

    def reshape_to_square(self, tensor):  # 将张量重塑为方形的方法
        """
        将张量重塑为方形形状。

        参数:
            tensor (torch.Tensor): 输入张量，形状为(B, C, N)，其中B是批量大小，
                C是通道数，N是元素数量。

        返回:
            tuple: 返回一个元组：
                - square_tensor (torch.Tensor): 重塑后的方形张量，形状为(B, C, side_length, side_length)，
                  其中side_length是方形张量每一边的长度。
                - side_length (int): 方形张量每一边的长度。
                - side_length (int): 方形张量每一边的长度。
                - N (int): 输入张量中的原始元素数量。
        """
        B, C, N = tensor.shape  # 解包输入张量的形状
        side_length = int(np.ceil(np.sqrt(N)))  # 计算方形张量边长
        padded_length = side_length**2  # 计算填充的长度

        padded_tensor = torch.zeros(
            (B, C, padded_length), device=tensor.device
        )  # 创建填充张量
        padded_tensor[:, :, :N] = tensor  # 将原始张量填充到填充张量中

        square_tensor = padded_tensor.view(
            B, C, side_length, side_length
        )  # 变形为方形张量

        return square_tensor, side_length, side_length, N  # 返回重塑后的张量及相关信息

    def filter_frequency_bands(self, tensor, cutoff=0.2):  # 过滤频带的方法
        """
        将输入张量过滤为低、中和高频带。

        参数:
            tensor (torch.Tensor): 要过滤的输入张量。
            cutoff (float, optional): 频带过滤的截止值。

        返回:
            torch.Tensor: 输入张量的低频带。
            torch.Tensor: 输入张量的中频带。
            torch.Tensor: 输入张量的高频带。
        """

        tensor = tensor.float()  # 转换为浮点型
        tensor, H, W, N = self.reshape_to_square(tensor)  # 将张量重塑为方形
        B, C, _, _ = tensor.shape  # 解包张量的形状

        max_radius = np.sqrt((H // 2) ** 2 + (W // 2) ** 2)  # 计算最大半径
        low_cutoff = max_radius * cutoff  # 低频截止值
        high_cutoff = max_radius * (1 - cutoff)  # 高频截止值

        fft_tensor = torch.fft.fftshift(  # 计算傅里叶变换并进行频移
            torch.fft.fft2(tensor, dim=(-2, -1)), dim=(-2, -1)
        )

        def create_filter(  # 创建滤波器的内部方法
            shape, low_cutoff, high_cutoff, mode="band", device=self.device
        ):
            rows, cols = shape  # 获取形状
            center_row, center_col = rows // 2, cols // 2  # 计算中心位置

            y, x = torch.meshgrid(  # 创建网格坐标
                torch.arange(rows, device=device),
                torch.arange(cols, device=device),
                indexing="ij",
            )
            distance = torch.sqrt(
                (y - center_row) ** 2 + (x - center_col) ** 2
            )  # 计算距离

            mask = torch.zeros(
                (rows, cols), dtype=torch.float32, device=device
            )  # 创建掩码

            if mode == "low":  # 低频模式
                mask[distance <= low_cutoff] = 1  # 在低频截止值内的mask设为1
            elif mode == "high":  # 高频模式
                mask[distance >= high_cutoff] = 1  # 在高频截止值外的mask设为1
            elif mode == "band":  # 带通模式
                mask[(distance > low_cutoff) & (distance < high_cutoff)] = (
                    1  # 在频带内的mask设为1
                )

            return mask  # 返回掩码

        low_pass_filter = create_filter(
            (H, W), low_cutoff, None, mode="low"
        )[  # 创建低通滤波器
            None, None, :, :
        ]
        high_pass_filter = create_filter(
            (H, W), None, high_cutoff, mode="high"
        )[  # 创建高通滤波器
            None, None, :, :
        ]
        mid_pass_filter = create_filter(
            (H, W), low_cutoff, high_cutoff, mode="band"
        )[  # 创建带通滤波器
            None, None, :, :
        ]

        low_freq_fft = fft_tensor * low_pass_filter  # 通过低通滤波器获得低频傅里叶变换
        high_freq_fft = (
            fft_tensor * high_pass_filter
        )  # 通过高通滤波器获得高频傅里叶变换
        mid_freq_fft = fft_tensor * mid_pass_filter  # 通过带通滤波器获得中频傅里叶变换

        low_freq_tensor = torch.fft.ifft2(  # 反傅里叶变换获得低频张量
            torch.fft.ifftshift(low_freq_fft, dim=(-2, -1)), dim=(-2, -1)
        ).real
        high_freq_tensor = torch.fft.ifft2(  # 反傅里叶变换获得高频张量
            torch.fft.ifftshift(high_freq_fft, dim=(-2, -1)), dim=(-2, -1)
        ).real
        mid_freq_tensor = torch.fft.ifft2(  # 反傅里叶变换获得中频张量
            torch.fft.ifftshift(mid_freq_fft, dim=(-2, -1)), dim=(-2, -1)
        ).real

        low_freq_tensor = low_freq_tensor.view(B, C, H * W)[
            :, :, :N
        ]  # 调整低频张量的形状
        high_freq_tensor = high_freq_tensor.view(B, C, H * W)[
            :, :, :N
        ]  # 调整高频张量的形状
        mid_freq_tensor = mid_freq_tensor.view(B, C, H * W)[
            :, :, :N
        ]  # 调整中频张量的形状

        return (
            low_freq_tensor,
            mid_freq_tensor,
            high_freq_tensor,
        )  # 返回低、中、高频张量


class CrossAttentionFusion(nn.Module):  # 定义CrossAttentionFusion类
    def __init__(self, embed_dim):  # 初始化方法
        super(CrossAttentionFusion, self).__init__()  # 调用父类的初始化方法
        self.query = nn.Linear(embed_dim, embed_dim)  # 定义查询的线性层
        self.key = nn.Linear(embed_dim, embed_dim)  # 定义键的线性层
        self.value = nn.Linear(embed_dim, embed_dim)  # 定义值的线性层
        self.softmax = nn.Softmax(dim=-1)  # 定义softmax层

    def forward(self, Q_feature, K_feature):  # 前向传播方法
        B, C, N = Q_feature.shape  # 解包查询特征的形状

        Q_feature = Q_feature.permute(0, 2, 1)  # 转置查询特征
        K_feature = K_feature.permute(0, 2, 1)  # 转置键特征

        Q = self.query(Q_feature)  # 形状: [B, N, C]
        K = self.key(K_feature)  # 形状: [B, N, C]
        V = self.value(K_feature)  # 形状: [B, N, C]

        attention_scores = torch.matmul(
            Q, K.transpose(-2, -1)
        ) / torch.sqrt(  # 计算注意力分数
            torch.tensor(C, dtype=torch.float32)
        )
        attention_weights = self.softmax(
            attention_scores
        )  # 应用softmax获得注意力权重，形状: [B, N, N]

        attended_features = torch.matmul(
            attention_weights, V
        )  # 加权特征，形状: [B, N, C]
        attended_features = attended_features.permute(0, 2, 1)  # 转置

        return attended_features  # 返回加权后的特征


class MSFM(nn.Module):  # 定义MSFM类（多尺度特征融合模块）
    def __init__(self, feature_dim):  # 初始化方法
        super(MSFM, self).__init__()  # 调用父类的初始化方法
        self.CA1 = CrossAttentionFusion(feature_dim)  # 实例化第一个交叉注意力融合模块
        self.CA2 = CrossAttentionFusion(feature_dim)  # 实例化第二个交叉注意力融合模块
        self.relu = nn.ReLU()  # ReLU激活函数

    def forward(self, low, mid, high):  # 前向传播方法
        low_new = self.CA1(mid, low)  # 低频特征的注意力融合
        high_new = self.CA2(mid, high)  # 高频特征的注意力融合
        fused_features = self.relu(
            low_new + mid + high_new
        )  # 融合低、中、高频特征并激活

        return fused_features  # 返回融合后的特征


"""
class FADAM(nn.Module):
    def __init__(self, feature_dim=512, N=900):
        # Frequency-Aware Domain Adaptation Module (FADAM)
        super(FADAM, self).__init__()
        self.FAM = FAM(feature_dim=512, N=N)  # 实例化特征注意力匹配模块
        self.MSFM = MSFM(feature_dim=512)  # 实例化多尺度特征融合模块

    def forward(self, sp_fts, qry_fts):
        # n,512,900
        fused_fts_low, fused_fts_mid, fused_fts_high = self.FAM(  # 融合特征
            sp_fts, qry_fts
        )
        # n,512,1800
        fused_fts = self.MSFM(fused_fts_low, fused_fts_mid, fused_fts_high)
        return fused_fts 
"""


class FADAM(nn.Module):

    def __init__(self, feature_dim=512, N=1024):
        super(FADAM, self).__init__()
        self.FAM = FAM(feature_dim=512, N=N)  # 实例化特征注意力匹配模块
        self.MSFM = MSFM(feature_dim=512)  # 实例化多尺度特征融合模块
        # 额外的卷积层用于特征形状转换
        self.reshape_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, sp_fts, qry_fts):
        """
        用于清洗域相关信息

        输入特征要求 b,512,n
        n可以为任意值?
        为了保留信息, 这里可能不使用掩码, 直接把所有特征塞进来比较好?

        """
        # n, 512, 900 -> n, 512, 1800
        # n, 512, 1024 -> n, 512, 2048
        fused_fts_low, fused_fts_mid, fused_fts_high = self.FAM(sp_fts, qry_fts)
        logger.debug("FAM: ", fused_fts_low.shape)
        fused_fts = self.MSFM(fused_fts_low, fused_fts_mid, fused_fts_high)
        logger.debug("MSFM: ", fused_fts.shape)
        # ([1, 512, 2304])
        # 将1D特征转换为2D形状
        # torch.Size([1, 512, 2048]) -> [1, 512, 32, 32]
        # 动态重塑特征为二维
        B, C, N = fused_fts.shape  # B: Batch, C: Channels, N: 第三维
        side_length = int(np.ceil(np.sqrt(N)))  # 计算正方形边长
        if side_length**2 != N:
            # 补齐到正方形
            padded_fts = torch.zeros((B, C, side_length**2), device=fused_fts.device)
            padded_fts[:, :, :N] = fused_fts
            fused_fts = padded_fts
        fused_fts_square = fused_fts.view(B, C, side_length, side_length)

        logger.debug("1D to 2D: ", fused_fts_square.shape)
        # fused_fts_square = fused_fts.view(
        #     fused_fts.shape[0], fused_fts.shape[1], int(2048**0.5), int(2048**0.5)
        # )
        # 使用卷积处理维度
        fused_fts_reshaped = F.interpolate(
            fused_fts_square, size=(64, 64), mode="bilinear", align_corners=True
        )
        output = self.reshape_conv(fused_fts_reshaped)

        # 在这里调整 output 的维度到 [1, 1, 1, 512, 64, 64]
        output = output.unsqueeze(0).unsqueeze(0)  # 在第1和第2维度插入两层
        logger.debug("FADAM output reshaped: ", output.shape)

        return output
