"""
FSS via GMRD
Extended from ADNet code by Hansen et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.encoder import Res101Encoder
import numpy as np
import random
import cv2
from models.moudles import MLP, Decoder
from models.FAMMoudles import FADAM

class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # 编码器
        self.encoder = Res101Encoder(
            replace_stride_with_dilation=[True, True, False],
            pretrained_weights=pretrained_weights,
        )  # 或 "resnet101"
        self.device = torch.device("cuda")  # 设置设备为GPU
        self.scaler = 20.0  # 缩放因子
        self.criterion = nn.NLLLoss()  # 负对数似然损失
        self.criterion_MSE = nn.MSELoss()  # 均方误差损失
        self.fg_num = 100  # 前景原型数量
        self.bg_num = 600  # 背景原型数量
        self.FADAM = FADAM(feature_dim=512, N=900)
        self.mlp1 = MLP(256, self.fg_num)  # 多层感知机，用于前景
        self.mlp2 = MLP(256, self.bg_num)  # 多层感知机，用于背景
        self.decoder1 = Decoder(self.fg_num)  # 前景解码器
        self.decoder2 = Decoder(self.bg_num)  # 背景解码器

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, train=False):
        """
        前向传播函数，用于执行模型的前向推理

        Args:
            supp_imgs: 支持图像
                形状: way x shot x [B x 3 x H x W], tensor的列表的列表
            supp_mask: 支持图像的前景和背景掩码
                形状: way x shot x [B x H x W], tensor的列表的列表
            qry_imgs: 查询图像
                形状: N x [B x 3 x H x W], tensor的列表
            train: 布尔值，指示模型是否在培训模式中

        Returns:
            output: 模型的输出
            align_loss: 对齐损失
            aux_loss: 辅助损失
        """

        self.n_ways = len(supp_imgs)  # 1
        self.n_shots = len(supp_imgs[0])  # 1
        self.n_queries = len(qry_imgs)  # 1
        assert self.n_ways == 1  # 目前只支持单一类别，因为不是每个shot都有多个子图
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]  # 查询批量大小
        supp_bs = supp_imgs[0][0].shape[0]  # 支持批量大小
        img_size = supp_imgs[0][0].shape[-2:]  # 图像尺寸
        supp_mask = torch.stack(
            [torch.stack(way, dim=0) for way in supp_mask], dim=0
        ).view(
            supp_bs, self.n_ways, self.n_shots, *img_size
        )  # 形状: Batch x way x shot x 3 x H x W 实际：1*1*1*256*256

        ###### 特征提取 ######
        # 支持图像集维度 Batch x way x shot x 3 x H x W
        # - >(shot * B) x 3 x H x W
        # -> (way * shot * B) x 3 x H x W
        # 查询图像集 N x B x 3 x H x W
        # -> (N * B) x 3 x H x W
        # 两者拼接 (way * shot * B + N * B) x 3 x H x W
        imgs_concat = torch.cat(
            [torch.cat(way, dim=0) for way in supp_imgs]
            + [
                torch.cat(qry_imgs, dim=0),
            ],
            dim=0,
        )  # (2, 3, 256,256)
        # 编码器输出
        # 将 (way * shot * B) x C x H' x W' 的张量重新调整为 B x Wa x Sh x C x H' x W'。
        # img_fts包含layer2，layer3层的特征
        img_fts, tao = self.encoder(imgs_concat)

        supp_fts = [
            img_fts[dic][
                : self.n_ways * self.n_shots * supp_bs
            ].view(  # B x Wa x Sh x C x H' x W'
                supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]
            )  # -1：自动推断特征通道数（C）
            for _, dic in enumerate(img_fts)
        ]  # torch.Size([1, 1, 1, 512, 64, 64])
        supp_fts = supp_fts[0]

        qry_fts = [
            img_fts[dic][
                self.n_ways * self.n_shots * supp_bs :
            ].view(  # B x N x C x H' x W'
                qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]
            )
            for _, dic in enumerate(img_fts)
        ]  # [1, 1, 512, 64, 64] 和 [1, 1, 512, 32, 32]
        qry_fts = qry_fts[0]

        ##### 获取阈值 #######
        self.t = tao[self.n_ways * self.n_shots * supp_bs :]  # 获取查询图像的阈值
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        ###### 计算损失 ######
        # 初始化损失值为0
        align_loss = torch.zeros(1).to(self.device)
        aux_loss = torch.zeros(1).to(self.device)
        outputs = []

        # supp_bs支持集的批量大小。
        # 对支持集批量的每个Support单独和Query 进行计算相似度。获取累加损失和预测结果。
        for epi in range(supp_bs):
            ###### 提取原型 ######
            if supp_mask[epi][0].sum() == 0:  # 如果没有前景
                supp_fts_ = [
                    [
                        self.getFeatures(
                            supp_fts[[epi], way, shot], supp_mask[[epi], way, shot]
                        )
                        for shot in range(self.n_shots)
                    ]
                    for way in range(self.n_ways)
                ]
                fg_prototypes = self.getPrototype(supp_fts_)

                ###### 获取查询预测 ######
                qry_pred = torch.stack(
                    [
                        self.getPred(
                            qry_fts[epi], fg_prototypes[way], self.thresh_pred[way]
                        )
                        for way in range(self.n_ways)
                    ],
                    dim=1,
                )  # 2 x N x Wa x H' x W'
                preds = F.interpolate(
                    qry_pred, size=img_size, mode="bilinear", align_corners=True
                )
                preds = torch.cat((1.0 - preds, preds), dim=1)  # 拼接前景和背景预测
                outputs.append(preds)  # 追加输出
                if train:
                    align_loss_epi = self.alignLoss(
                        [supp_fts[epi]], [qry_fts[epi]], preds, supp_mask[epi]
                    )
                    align_loss += align_loss_epi  # 计算对齐损失
            else:
                # supp_fts torch.Size([1, 1, 1, 512, 64, 64]) batch*way*shot*512*H*W
                # qry_fts  torch.Size([1, 1, 512, 64, 64]) batch*shot*512*H*W

                # CPG模块 ############################################################

                spt_fts_ = [  # 通过掩码获取支持特征 torch.Size([1, 512, 64, 64])
                    [
                        self.getFeatures(  # 获取每个样本的特征
                            # supp_fts torch.Size([1, 1, 1, 512, 64, 64])
                            supp_fts[[epi], way, shot],
                            supp_mask[[epi], way, shot],
                        )
                        for shot in range(self.n_shots)
                    ]
                    for way in range(self.n_ways)
                ]

                spt_fg_proto = self.getPrototype(
                    spt_fts_
                )  # 获取前景原型 torch.Size([1, 512])

                qry_pred = torch.stack(  # 计算查询预测 torch.Size([1, 1, 64, 64])
                    [
                        # qry_fts torch.Size([1, 1, 512, 64, 64])
                        self.getPred(  # 获取每个方式的预测
                            qry_fts[way], spt_fg_proto[way], self.thresh_pred[way]
                        )
                        for way in range(self.n_ways)
                    ],
                    dim=1,
                )  # N x Wa x H' x W'

                qry_pred_coarse = (
                    F.interpolate(  # 上采样查询预测 torch.Size([1, 1, 256, 256])
                        qry_pred, size=img_size, mode="bilinear", align_corners=True
                    )
                )
                if train:  # 如果在训练模式
                    log_qry_pred_coarse = torch.cat(  # 计算对数预测
                        [1 - qry_pred_coarse, qry_pred_coarse], dim=1
                    ).log()

                    coarse_loss = self.criterion(
                        log_qry_pred_coarse, qry_mask
                    )  # 计算损失

                ####################################################################

                spt_fg_fts = [  # 获取支持前景特征 torch.Size([1, 512, 44])
                    [
                        self.get_fg(
                            supp_fts[way][shot], supp_mask[[0], way, shot]
                        )  # 获取每个样本的前景特征
                        for shot in range(self.n_shots)
                    ]
                    for way in range(self.n_ways)
                ]  # (1, 512, N)

                qry_fg_fts = [  # 获取查询前景特征 torch.Size([1, 512, 65536])
                    self.get_fg(qry_fts[way], qry_pred_coarse[epi])
                    for way in range(self.n_ways)
                ]  # (1, 512, N)

                # 使用FADAM清洗域信息
                # FAM要求输入是b,512,n，FAM转为b,512,900。
                # 最终MSFM输出是torch.Size([1, 512, 1800])
                supp_fts = self.FADAM(spt_fg_fts, qry_fg_fts)

                # GMRD 生成多个原型
                ####################################################################

                # supp_fts[[epi], way, shot] -> torch.Size([1, 512, 64, 64])
                # supp_mask[[epi], way, shot] -> torch.Size([1, 256, 256])
                fg_pts = [
                    [
                        self.get_fg_pts(
                            supp_fts[[epi], way, shot], supp_mask[[epi], way, shot]
                        )
                        for shot in range(self.n_shots)
                    ]
                    for way in range(self.n_ways)
                ]
                fg_pts = self.get_all_prototypes(fg_pts)  # 所有前景原型 （100+2）*512

                bg_pts = [
                    [
                        self.get_bg_pts(
                            supp_fts[[epi], way, shot], supp_mask[[epi], way, shot]
                        )
                        for shot in range(self.n_shots)
                    ]
                    for way in range(self.n_ways)
                ]
                bg_pts = self.get_all_prototypes(bg_pts)  # 所有背景原型 （600+2）* 512

                ###### 获取查询预测 ######
                fg_sim = torch.stack(
                    [
                        self.get_fg_sim(qry_fts[epi], fg_pts[way])
                        for way in range(self.n_ways)
                    ],
                    dim=1,
                ).squeeze(0)
                bg_sim = torch.stack(
                    [
                        self.get_bg_sim(qry_fts[epi], bg_pts[way])
                        for way in range(self.n_ways)
                    ],
                    dim=1,
                ).squeeze(0)

                preds = F.interpolate(
                    fg_sim, size=img_size, mode="bilinear", align_corners=True
                )
                bg_preds = F.interpolate(
                    bg_sim, size=img_size, mode="bilinear", align_corners=True
                )

                preds = torch.cat([bg_preds, preds], dim=1)  # 拼接背景和前景的预测
                preds = torch.softmax(preds, dim=1)  # 应用softmax

                outputs.append(preds)  # 追加输出
                if train:
                    align_loss_epi, aux_loss_epi = self.align_aux_Loss(
                        [supp_fts[epi]],
                        [qry_fts[epi]],
                        preds,
                        supp_mask[epi],
                        fg_pts,
                        bg_pts,
                    )  # fg_pts, bg_pts
                    align_loss += align_loss_epi  # 计算对齐损失
                    aux_loss += aux_loss_epi  # 计算辅助损失

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])  # 重塑输出形状

        return output, align_loss / supp_bs, aux_loss / supp_bs  # 返回输出和损失

    def getPred(self, fts, prototype, thresh):
        """
        计算特征与原型之间的距离

        Args:
            fts: 输入特征
                期望形状: N x C x H x W
            prototype: 一个语义类的原型
                期望形状: 1 x C

        Returns:
            pred: 预测值
        """

        sim = (
            -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        )  # 计算余弦相似度并缩放
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))  # 应用sigmoid激活

        return pred

    def getFeatures(self, fts, mask):
        """
        通过掩模平均池化提取前景和背景特征

        Args:
            fts: 输入特征
                期望形状: 1 x C x H' x W'
            mask: 二进制掩模
                期望形状: 1 x H x W

        Returns:
            masked_fts: 掩模后的特征
        """

        fts = F.interpolate(
            fts, size=mask.shape[-2:], mode="bilinear", align_corners=True
        )  # 调整尺寸

        # 掩模前景特征
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) / (
            mask[None, ...].sum(dim=(-2, -1)) + 1e-5
        )  # 1 x C

        return masked_fts  # 返回前景特征

    def getPrototype(self, fg_fts):
        """
        通过平均特征获得原型

        Args:
            fg_fts: 每个方式/shot的前景特征列表
                期望形状: Wa x Sh x [1 x C]

        Returns:
            fg_prototypes: 前景原型列表
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])  # 获取way和shot的数量
        fg_prototypes = [
            torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True)
            / n_shots
            for way in fg_fts
        ]  # 连接所有前景特征并计算平均值

        return fg_prototypes  # 返回前景原型

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        """
        计算对齐损失

        Args:
            supp_fts: 支持特征
            qry_fts: 查询特征
            pred: 预测结果
            fore_mask: 前景掩模

        Returns:
            loss: 对齐损失值
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])  # 获取way和shot的数量

        # 获取查询掩模
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]  # 创建二进制掩模
        skip_ways = [
            i for i in range(n_ways) if binary_masks[i + 1].sum() == 0
        ]  # 跳过没有前景的way
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # 计算支持损失
        loss = torch.zeros(1).to(self.device)  # 初始化损失
        for way in range(n_ways):
            if way in skip_ways:
                continue  # 跳过没有前景的way
            # 获取查询原型
            for shot in range(n_shots):
                # 获取原型
                qry_fts_ = [self.getFeatures(qry_fts[0], pred_mask[way + 1])]
                fg_prototypes = self.getPrototype([qry_fts_])  # 计算查询前景原型

                # 获取预测
                supp_pred = self.getPred(
                    supp_fts[0][way, [shot]], fg_prototypes[way], self.thresh_pred[way]
                )  # N x Wa x H' x W'
                supp_pred = F.interpolate(
                    supp_pred[None, ...],
                    size=fore_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )

                # 合并不同特征图的预测
                pred_ups = torch.cat(
                    (1.0 - supp_pred, supp_pred), dim=1
                )  # (1, 2, 256, 256)

                # 构建支持的Ground-Truth分割
                supp_label = torch.full_like(
                    fore_mask[way, shot], 255, device=fore_mask.device
                )
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # 计算损失
                eps = torch.finfo(torch.float32).eps  # 防止取log时出现无穷大
                log_prob = torch.log(
                    torch.clamp(pred_ups, eps, 1 - eps)
                )  # 计算对数概率
                loss += (
                    self.criterion(log_prob, supp_label[None, ...].long())
                    / n_shots
                    / n_ways
                )  # 累加损失

        return loss  # 返回损失

    def align_aux_Loss(
        self, supp_fts, qry_fts, pred, fore_mask, sup_fg_pts, sup_bg_pts
    ):
        """
        计算对齐辅助损失

        Args:
            supp_fts: 支持特征
                期望形状: [1, 512, 64, 64]
            qry_fts: 查询特征
                期望形状: (1, 512, 64, 64)
            pred: 预测值
                期望形状: [1, 2, 256, 256]
            fore_mask: 前景掩模
                期望形状: [Way, Shot, 256, 256]

        Returns:
            loss: 对齐损失
            loss_aux: 辅助损失
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])  # 获取way和shot的数量

        # 获取查询掩模
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]  # 创建二进制掩模
        skip_ways = [
            i for i in range(n_ways) if binary_masks[i + 1].sum() == 0
        ]  # 跳过没有前景的way
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # 计算支持损失
        loss = torch.zeros(1).to(self.device)  # 初始化损失
        loss_aux = torch.zeros(1).to(self.device)  # 初始化辅助损失
        for way in range(n_ways):
            if way in skip_ways:
                continue  # 跳过没有前景的way
            # 获取查询原型
            for shot in range(n_shots):
                # 获取原型
                qry_fts_ = [self.get_fg_pts(qry_fts[0], pred_mask[way + 1])]
                fg_prototypes = self.get_all_prototypes([qry_fts_])  # 获取查询前景原型
                bg_pts_ = [self.get_bg_pts(qry_fts[0], pred_mask[way + 1])]
                bg_pts_ = self.get_all_prototypes([bg_pts_])  # 获取查询背景原型

                loss_aux += self.get_aux_loss(
                    sup_fg_pts[way], fg_prototypes[way], sup_bg_pts[way], bg_pts_[way]
                )  # 计算辅助损失

                # 获取预测
                supp_pred = self.get_fg_sim(
                    supp_fts[0][way, [shot]], fg_prototypes[way]
                )  # N x Wa x H' x W'
                bg_pred_ = self.get_bg_sim(
                    supp_fts[0][way, [shot]], bg_pts_[way]
                )  # N x Wa x H' x W'
                supp_pred = F.interpolate(
                    supp_pred,
                    size=fore_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                bg_pred_ = F.interpolate(
                    bg_pred_,
                    size=fore_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )

                # 合并不同特征图的预测
                preds = torch.cat([bg_pred_, supp_pred], dim=1)  # 合并前景后景预测
                preds = torch.softmax(preds, dim=1)  # 应用softmax

                # 构建支持的Ground-Truth分割
                supp_label = torch.full_like(
                    fore_mask[way, shot], 255, device=fore_mask.device
                )
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # 计算损失
                eps = torch.finfo(torch.float32).eps  # 防止取log时出现无穷大
                log_prob = torch.log(torch.clamp(preds, eps, 1 - eps))  # 计算对数概率
                loss += (
                    self.criterion(log_prob, supp_label[None, ...].long())
                    / n_shots
                    / n_ways
                )  # 累加损失

        return loss, loss_aux  # 返回损失和辅助损失

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

    def get_fg_pts(self, features, mask):
        """
        获取生成前景原型
        该方法通过调整特征大小、形态学腐蚀、特征加权平均和 MLP 计算，从输入特征和前景掩模中提取前景原型。
        最终返回的前景原型 fg_prototypes 包含了多个不同区域的原型信息，形状为 k x C。

        Args:
            features: 输入特征，形状为 1 x C x H x W
            mask: 前景掩模，形状为 1 x H x W

        Returns:
            fg_prototypes: 前景原型，形状为 k x C
        """
        features_trans = F.interpolate(
            features, size=mask.shape[-2:], mode="bilinear", align_corners=True
        )  # 调整大小
        ie_mask = mask.squeeze(0) - torch.tensor(
            cv2.erode(
                mask.squeeze(0).cpu().numpy(),
                np.ones((3, 3), dtype=np.uint8),
                iterations=2,
            )
        ).to(
            self.device
        )  # 形态学腐蚀
        ie_mask = ie_mask.unsqueeze(0)
        ie_prototype = torch.sum(features_trans * ie_mask[None, ...], dim=(-2, -1)) / (
            ie_mask[None, ...].sum(dim=(-2, -1)) + 1e-5
        )  # 1 x C
        origin_prototype = torch.sum(features_trans * mask[None, ...], dim=(-2, -1)) / (
            mask[None, ...].sum(dim=(-2, -1)) + 1e-5
        )  # 1 x C

        fg_fts = self.get_fg_fts(features_trans, mask)  # 获取前景特征
        fg_prototypes = self.mlp1(fg_fts.view(512, 256 * 256)).permute(
            1, 0
        )  # 通过MLP计算前景原型
        fg_prototypes = torch.cat(
            [fg_prototypes, origin_prototype, ie_prototype], dim=0
        )  # 合并原型

        return fg_prototypes  # 返回前景原型

    def get_bg_pts(self, features, mask):
        """
        获取背景原型

        Args:
            features: 输入特征，形状为 1 x C x H x W
            mask: 背景掩模，形状为 H x W

        Returns:
            bg_prototypes: 背景原型，形状为 k x C
        """
        bg_mask = 1 - mask  # 反转掩模
        features_trans = F.interpolate(
            features, size=bg_mask.shape[-2:], mode="bilinear", align_corners=True
        )  # 调整大小
        oe_mask = torch.tensor(
            cv2.dilate(
                bg_mask.squeeze(0).cpu().numpy(),
                np.ones((3, 3), dtype=np.uint8),
                iterations=2,
            )
        ).to(self.device) - mask.squeeze(
            0
        )  # 形态学膨胀
        oe_mask = oe_mask.unsqueeze(0)
        oe_prototype = torch.sum(features_trans * oe_mask[None, ...], dim=(-2, -1)) / (
            oe_mask[None, ...].sum(dim=(-2, -1)) + 1e-5
        )  # 1 x C
        origin_prototype = torch.sum(
            features_trans * bg_mask[None, ...], dim=(-2, -1)
        ) / (
            bg_mask[None, ...].sum(dim=(-2, -1)) + 1e-5
        )  # 1 x C

        bg_fts = self.get_fg_fts(features_trans, bg_mask)  # 获取背景特征
        bg_prototypes = self.mlp2(bg_fts.view(512, 256 * 256)).permute(
            1, 0
        )  # 通过MLP计算背景原型
        bg_prototypes = torch.cat(
            [bg_prototypes, origin_prototype, oe_prototype], dim=0
        )  # 合并原型

        return bg_prototypes  # 返回背景原型

    def get_random_pts(self, features_trans, mask, n_protptype):
        """
        随机选择特征中的点作为原型

        Args:
            features_trans: 特征，形状为 [C, H, W]
            mask: 掩模，形状为 [H, W]
            n_protptype: 原型数量

        Returns:
            prototypes: 随机选择的原型，形状为 (n_protptype, 512)
        """

        features_trans = features_trans.squeeze(
            0
        )  # 将特征从 (1, C, H, W) 变为 (C, H, W)
        features_trans = features_trans.permute(1, 2, 0)  # 调整维度为 (H, W, C)
        features_trans = features_trans.view(
            features_trans.shape[-2] * features_trans.shape[-3],
            features_trans.shape[-1],
        )  # 重塑为 (H*W, C)
        mask = mask.squeeze(0).view(-1)  # 将掩模重塑
        indx = mask == 1  # 找到前景点
        features_trans = features_trans[indx]  # 选择前景特征
        if len(features_trans) >= n_protptype:  # 如果前景特征足够多
            k = random.sample(range(len(features_trans)), n_protptype)  # 随机选择
            prototypes = features_trans[k]  # 选择原型
        else:
            if len(features_trans) == 0:
                prototypes = torch.zeros(n_protptype, 512).to(
                    self.device
                )  # 如果没有可用特征，返回零
            else:
                r = (n_protptype) // len(features_trans)  # 计算重复次数
                k = random.sample(
                    range(len(features_trans)),
                    (n_protptype - len(features_trans)) % len(features_trans),
                )  # 选择额外的特征
                prototypes = torch.cat(
                    [features_trans for _ in range(r)], dim=0
                )  # 重复特征以填充
                prototypes = torch.cat([features_trans[k], prototypes], dim=0)  # 合并

        return prototypes  # 返回 (n_prototype, 512)

    def get_fg_fts(self, fts, mask):
        """
        通过掩模选择前景特征

        Args:
            fts: 输入特征，形状为 1 x C x H x W
            mask: 前景掩模，形状为 H x W

        Returns:
            fg_fts: 掩模后的前景特征
        """
        _, c, h, w = fts.shape
        # 选择掩模后的前景特征
        fg_fts = fts * mask[None, ...]
        bg_fts = torch.ones_like(fts) * mask[None, ...]
        mask_ = mask.view(-1)
        n_pts = len(mask_) - len(mask_[mask_ == 1])  # 计算背景点数量
        select_pts = self.get_random_pts(fts, mask, n_pts)  # 获取随机点作为特征
        index = bg_fts == 0  # 背景掩模
        fg_fts[index] = select_pts.permute(1, 0).reshape(512 * n_pts)  # 替换

        return fg_fts  # 返回掩模后的前景特征

    def get_all_prototypes(self, fg_fts):
        """
        获取所有原型的平均值，每way独立计算n个shot的平均值作为这个way的原型

        Args:
            fg_fts: 原型特征列表
                期望形状: Wa x Sh x [all x C]

        Returns:
            fg_prototypes: 原型列表
                形状: [(all, 512) * way]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])  # 获取way和shot的数量
        prototypes = [
            sum([shot for shot in way]) / n_shots for way in fg_fts
        ]  # 计算平均原型
        return prototypes  # 返回平均原型

    def get_fg_sim(self, fts, prototypes):
        """
        计算特征与前景原型之间的相似度

        Args:
            fts: 输入特征
                期望形状: N x C x H x W
            prototypes: 前景原型
                期望形状: 1 x C

        Returns:
            fg_sim: 前景相似度，形状为 [1, 1, 64, 64]
        """
        fts_ = fts.permute(0, 2, 3, 1)  # 调整维度
        fts_ = F.normalize(fts_, dim=-1)  # 归一化特征
        pts_ = F.normalize(prototypes, dim=-1)  # 归一化原型
        fg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(
            0, 3, 1, 2
        )  # 计算相似度
        fg_sim = self.decoder1(fg_sim)  # 解码

        return fg_sim  # 返回前景相似度

    def get_bg_sim(self, fts, prototypes):
        """
        计算特征与背景原型之间的相似度

        Args:
            fts: 输入特征
                期望形状: N x C x H x W
            prototypes: 背景原型
                期望形状: 1 x C

        Returns:
            bg_sim: 背景相似度，形状为 [1, 1, 64, 64]
        """
        fts_ = fts.permute(0, 2, 3, 1)  # 调整维度
        fts_ = F.normalize(fts_, dim=-1)  # 归一化特征
        pts_ = F.normalize(prototypes, dim=-1)  # 归一化原型
        bg_sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(
            0, 3, 1, 2
        )  # 计算相似度
        bg_sim = self.decoder2(bg_sim)  # 解码

        return bg_sim  # 返回背景相似度

    def get_aux_loss(self, sup_fg_pts, qry_fg_pts, sup_bg_pts, qry_bg_pts):
        """
        计算辅助损失

        Args:
            sup_fg_pts: 支持前景点
            qry_fg_pts: 查询前景点
            sup_bg_pts: 支持背景点
            qry_bg_pts: 查询背景点

        Returns:
            综合损失值
        """
        d1 = F.normalize(sup_fg_pts, dim=-1)  # 归一化支持前景点
        d2 = F.normalize(qry_fg_pts, dim=-1)  # 归一化查询前景点
        b1 = F.normalize(sup_bg_pts, dim=-1)  # 归一化支持背景点
        b2 = F.normalize(qry_bg_pts, dim=-1)  # 归一化查询背景点

        fg_intra0 = torch.matmul(d1[:-2], d2[:-2].transpose(0, 1))  # 计算前景点间的内积
        fg_intra1 = torch.matmul(d1[[-2]], d2[-2])  # 计算最后两个点的相似度
        fg_intra2 = torch.matmul(d1[[-1]], d2[-1])
        fg_intra0, _ = torch.max(fg_intra0, dim=1)  # 获取最大值
        fg_intra0 = torch.cat([fg_intra0, fg_intra1, fg_intra2])  # 合并
        fg_intra0 = torch.mean(fg_intra0)  # 计算均值

        bg_intra0 = torch.matmul(b1[:-2], b2[:-2].transpose(0, 1))  # 计算背景点间的内积
        bg_intra1 = torch.matmul(b1[[-2]], b2[-2])
        bg_intra2 = torch.matmul(b1[[-1]], b2[-1])
        bg_intra0, _ = torch.max(bg_intra0, dim=1)
        bg_intra0 = torch.cat([bg_intra0, bg_intra1, bg_intra2])
        bg_intra0 = torch.mean(bg_intra0)  # 计算均值
        intra_loss = 2 - fg_intra0 - bg_intra0  # 计算内在损失

        sup_inter = torch.matmul(d1, b1.transpose(0, 1))  # 计算支持前景与背景的相似度
        qry_inter = torch.matmul(d2, b2.transpose(0, 1))  # 计算查询前景与背景的相似度
        inter_loss = max((0, torch.mean(sup_inter))) + max(
            (0, torch.mean(qry_inter))
        )  # 计算外部损失

        return intra_loss + inter_loss  # 返回总损失
