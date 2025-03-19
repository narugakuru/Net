from turtle import forward
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数式API模块
import numpy as np  # 导入NumPy库

# from .encoder import Res101Encoder  # 从当前包导入Res50Encoder类（被注释掉）
from models.encoder import Res101Encoder  # 从models.encoder模块导入Res101Encoder类


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
    def __init__(self, feature_dim=512, N=900):  # 初始化方法
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


class FewShotSeg(nn.Module):  # 定义FewShotSeg类（少量样本分割模型）

    def __init__(self, args):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法

        # 编码器
        self.encoder = Res101Encoder(  # 实例化Res50编码器
            replace_stride_with_dilation=[True, True, False], pretrained_weights="COCO"
        )  # 或者使用"ImageNet"预训练权重
        if torch.cuda.is_available():  # 检查CUDA是否可用
            self.device = torch.device("cuda")  # 如果可用，设置为CUDA设备
        else:
            self.device = torch.device("cpu")  # 否则，设置为CPU设备

        self.args = args  # 存储输入参数
        self.scaler = 20.0  # 定义缩放因子
        self.criterion = nn.NLLLoss(  # 定义负对数似然损失
            ignore_index=255,
            weight=torch.FloatTensor([0.1, 1.0]).cuda(),  # 设置分类权重
        )

        self.N = 900  # 定义N
        self.FAM = FAM(feature_dim=512, N=self.N)  # 实例化特征注意力匹配模块
        self.MSFM = MSFM(feature_dim=512)  # 实例化多尺度特征融合模块

    def forward(
        self, supp_imgs, supp_mask, qry_imgs, qry_mask, opt, train=False
    ):  # 前向传播方法
        """
        参数:
            supp_imgs: 支持图像
                way x shot x [B x 3 x H x W], tensor列表的列表
            fore_mask: 支持图像的前景掩码
                way x shot x [B x H x W], tensor列表的列表
            back_mask: 支持图像的背景掩码
                way x shot x [B x H x W], tensor列表的列表
            qry_imgs: 查询图像
                N x [B x 3 x H x W], tensor列表 (1, 3, 257, 257)
            qry_mask: 标签
                N x 2 x H x W, tensor
            text_fts: 支持图像类别对应的文本特征
                标量（1）
        """

        self.n_ways = len(supp_imgs)  # 获取方式数量
        self.n_shots = len(supp_imgs[0])  # 获取每种方式的样本数量
        self.n_queries = len(qry_imgs)  # 获取查询图像数量
        assert self.n_ways == 1  # 确保只有一种方式（暂时只考虑单一方式）
        assert self.n_queries == 1  # 确保只有一个查询图像
        # qry_imgs[0] torch.Size([1, 3, 70, 74])  # 获取查询图像的形状
        qry_bs = qry_imgs[0].shape[0]  # 获取查询图像的批量大小
        # supp_imgs[0][0] torch.Size([1, 3, 53, 74])  # 获取支持图像的形状
        supp_bs = supp_imgs[0][0].shape[0]  # 获取支持图像的批量大小
        # # img_size ([53, 74])  # 获取图像大小
        img_size = supp_imgs[0][0].shape[-2:]  # 解包支持图像的最后两个维度，得到H和W

        supp_mask = torch.stack(  # 堆叠支持掩码
            [torch.stack(way, dim=0) for way in supp_mask], dim=0
        ).view(  # 重塑形状
            supp_bs, self.n_ways, self.n_shots, *img_size
        )  # 形状: B x Wa x Sh x H x W

        ## 使用ResNet骨干网络提取特征
        # 提取特征 #
        imgs_concat = torch.cat(  # 将所有支持图像和查询图像连接起来
            [torch.cat(way, dim=0) for way in supp_imgs]
            + [
                torch.cat(qry_imgs, dim=0),
            ],
            dim=0,
        )
        # 假设输入 1,3,256,256
        # feature 的最终输出形状为 (1, 512, 16, 16)。 tao是(1,1)
        # img_fts(2,512,64,64) tao(1)
        img_fts, tao = self.encoder(imgs_concat)  # 使用编码器提取特征和阈值

        # supp_fts torch.Size([1, 1, 1, 512, 64, 64])
        supp_fts = img_fts[  # 解析支持特征，形状为 B x Wa x Sh x C x H' x W'
            : self.n_ways * self.n_shots * supp_bs
        ].view(supp_bs, self.n_ways, self.n_shots, -1, *img_fts.shape[-2:])
        # qry_fts torch.Size([1, 1, 512, 64, 64])
        qry_fts = img_fts[  # 解析查询特征，形状为 B x N x C x H' x W'
            self.n_ways * self.n_shots * supp_bs :
        ].view(qry_bs, self.n_queries, -1, *img_fts.shape[-2:])

        # 获取阈值 #
        self.t = tao[self.n_ways * self.n_shots * supp_bs :]  # 获取查询特征的阈值
        self.thresh_pred = [self.t for _ in range(self.n_ways)]  # 为每种方式设置阈值

        self.t_ = tao[: self.n_ways * self.n_shots * supp_bs]  # 获取支持特征的阈值
        self.thresh_pred_ = [self.t_ for _ in range(self.n_ways)]  # 为每种方式设置阈值

        outputs_qry = []  # 初始化查询输出的列表
        coarse_loss = torch.zeros(1).to(self.device)  # 初始化粗损失
        for epi in range(supp_bs):  # 对于每个支持图像

            """
            supp_fts[[epi], way, shot]: (B, C, H, W)
            """

            if supp_mask[[0], 0, 0].max() > 0.0:  # 检查支持掩码中是否有前景
                spt_fts_ = [  # 通过掩码获取支持特征
                    [
                        self.getFeatures(  # 获取每个样本的特征
                            supp_fts[[epi], way, shot], supp_mask[[epi], way, shot]
                        )
                        for shot in range(self.n_shots)
                    ]
                    for way in range(self.n_ways)
                ]
                spt_fg_proto = self.getPrototype(spt_fts_)  # 获取前景原型

                # CPG模块 *******************
                qry_pred = torch.stack(  # 计算查询预测
                    [
                        self.getPred(  # 获取每个方式的预测
                            qry_fts[way], spt_fg_proto[way], self.thresh_pred[way]
                        )
                        for way in range(self.n_ways)
                    ],
                    dim=1,
                )  # N x Wa x H' x W'

                qry_pred_coarse = F.interpolate(  # 上采样查询预测
                    qry_pred, size=img_size, mode="bilinear", align_corners=True
                )

                if train:  # 如果在训练模式
                    log_qry_pred_coarse = torch.cat(  # 计算对数预测
                        [1 - qry_pred_coarse, qry_pred_coarse], dim=1
                    ).log()

                    coarse_loss = self.criterion(
                        log_qry_pred_coarse, qry_mask
                    )  # 计算损失

                # ************************************************

                spt_fg_fts = [  # 获取支持前景特征
                    [
                        self.get_fg(
                            supp_fts[way][shot], supp_mask[[0], way, shot]
                        )  # 获取每个样本的前景特征
                        for shot in range(self.n_shots)
                    ]
                    for way in range(self.n_ways)
                ]  # (1, 512, N)

                qry_fg_fts = [  # 获取查询前景特征
                    self.get_fg(qry_fts[way], qry_pred_coarse[epi])
                    for way in range(self.n_ways)
                ]  # (1, 512, N)
                # spt_fg_fts torch.Size([1, 512, 5650]) qry_fg_fts torch.Size([1, 512, 65536])
                # fused_fts_* torch.Size([1, 512, 1800])
                fused_fts_low, fused_fts_mid, fused_fts_high = self.FAM(  # 融合特征
                    spt_fg_fts, qry_fg_fts
                )

                fused_fg_fts = self.MSFM(
                    fused_fts_low, fused_fts_mid, fused_fts_high
                )  # 多尺度融合彩色特征

                fg_proto = [self.get_proto_new(fused_fg_fts)]  # 获取新前景原型

                pred = torch.stack(  # 计算最终预测
                    [
                        self.getPred(qry_fts[way], fg_proto[way], self.thresh_pred[way])
                        for way in range(self.n_ways)
                    ],
                    dim=1,
                )  # N x Wa x H' x W'

                pred_up = F.interpolate(  # 上采样最终预测
                    pred, size=img_size, mode="bilinear", align_corners=True
                )
                pred = torch.cat(
                    (1.0 - pred_up, pred_up), dim=1
                )  # 拼接前景和背景的预测
                outputs_qry.append(pred)  # 将预测添加到输出列表

            else:  # 如果没有前景
                ######################## 默认原型网络 ################
                supp_fts_ = [  # 获取支持特征
                    [
                        self.getFeatures(  # 获取每个样本的特征
                            supp_fts[[epi], way, shot], supp_mask[[epi], way, shot]
                        )
                        for shot in range(self.n_shots)
                    ]
                    for way in range(self.n_ways)
                ]
                fg_prototypes = self.getPrototype(supp_fts_)  # 获取前景原型

                qry_pred = torch.stack(  # 计算查询预测
                    [
                        self.getPred(  # 获取每个方式的预测
                            qry_fts[epi], fg_prototypes[way], self.thresh_pred[way]
                        )
                        for way in range(self.n_ways)
                    ],
                    dim=1,
                )  # N x Wa x H' x W'
                ########################################################################

                # Combine predictions of different feature maps #
                qry_pred_up = F.interpolate(  # 上采样最终预测
                    qry_pred, size=img_size, mode="bilinear", align_corners=True
                )
                preds = torch.cat(
                    (1.0 - qry_pred_up, qry_pred_up), dim=1
                )  # 拼接前景和背景的预测

                outputs_qry.append(preds)  # 将预测添加到输出列表

        output_qry = torch.stack(outputs_qry, dim=1)  # 堆叠查询输出
        output_qry = output_qry.view(-1, *output_qry.shape[2:])  # 重塑输出形状

        return output_qry, coarse_loss  # 返回输出和粗损失

    def getPred(self, fts, prototype, thresh):  # 计算预测的方法
        """
        计算特征和原型之间的距离

        参数:
            fts: 输入特征
                期望形状: N x C x H x W
            prototype: 一个语义类别的原型
                期望形状: 1 x C
        """

        sim = (
            -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        )  # 计算余弦相似度
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))  # 计算预测概率

        return pred  # 返回预测结果

    def getFeatures(self, fts, mask):  # 通过掩码平均池化提取特征
        """
        通过掩码平均池化提取前景和背景特征

        参数:
            fts: 输入特征，期望形状: 1 x C x H' x W'
            mask: 二进制掩码，期望形状: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode="bilinear")  # 进行上采样

        # 掩码前景特征
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) / (  # 计算加权平均
            mask[None, ...].sum(dim=(-2, -1)) + 1e-5
        )  # 1 x C

        return masked_fts  # 返回掩码后的特征

    def getPrototype(self, fg_fts):  # 计算原型的方法
        """
        平均特征以获得原型

        参数:
            fg_fts: 每种方式/样本的前景特征的列表
                期望形状: Wa x Sh x [1 x C]
            bg_fts: 每种方式/样本的背景特征的列表
                期望形状: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])  # 获取方式和样本的数量
        fg_prototypes = [  # 计算前景原型
            torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True)
            / n_shots
            for way in fg_fts
        ]  ## 将所有前景特征连接在一起

        return fg_prototypes  # 返回前景原型

    def get_fg(self, fts, mask):  # 获取前景特征的方法
        """
        :param fts: (1, C, H', W')
        :param mask: (1, H, W)
        :return:
        """

        mask = torch.round(mask)  # 对掩码进行四舍五入
        fts = F.interpolate(fts, size=mask.shape[-2:], mode="bilinear")  # 进行上采样

        mask = mask.unsqueeze(1).bool()  # 扩展掩码维度
        result_list = []  # 初始化结果列表

        for batch_id in range(fts.shape[0]):  # 遍历批次
            tmp_tensor = fts[batch_id]  # 获取当前批次的特征
            tmp_mask = mask[batch_id]  # 获取当前批次的掩码

            foreground_features = tmp_tensor[:, tmp_mask[0]]  # 提取前景特征

            if foreground_features.shape[1] == 1:  # 如果前景特征的通道数为1
                foreground_features = torch.cat(  # 将前景特征复制一次
                    (foreground_features, foreground_features), dim=1
                )

            result_list.append(foreground_features)  # 将前景特征添加到结果列表

        foreground_features = torch.stack(result_list)  # 将结果列表堆叠为张量

        return foreground_features  # 返回前景特征

    def get_proto_new(self, fts):  # 获取新原型的方法
        """
        :param fts:  (1, 512, N)
        :return: 1, 512, 1
        """
        N = fts.size(2)  # 获取特征的数量
        proto = torch.sum(fts, dim=2) / (N + 1e-5)  # 计算新的原型

        return proto  # 返回新原型


class FADAM(nn.Module):
    def __init__(self, featrue_dim=512, N=900):
        # Frequency-Aware Domain Adaptation Module (FADAM)
        self.FAM = FAM(feature_dim=512, N=N)  # 实例化特征注意力匹配模块
        self.MSFM = MSFM(feature_dim=512)  # 实例化多尺度特征融合模块

    def forward(self, sp_fts, qry_fts):
        fused_fts_low, fused_fts_mid, fused_fts_high = self.FAM(  # 融合特征
            sp_fts, qry_fts
        )
        fused_fts = self.MSFM(fused_fts_low, fused_fts_mid, fused_fts_high)
        return fused_fts
