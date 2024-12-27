import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskMultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """

    def __init__(self, in_features, head_num, bias=True, activation=F.relu):
        """
        初始化函数

        参数:
        in_features (int): 输入特征的维度
        head_num (int): 注意力头的数量
        bias (bool, optional): 是否使用偏置，默认为True
        activation (function, optional): 激活函数，默认为ReLU
        """
        super(MaskMultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError(
                "`in_features`({}) 应该可以被 `head_num`({}) 整除".format(
                    in_features, head_num
                )
            )
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, ori_k=None, mask=None, cross=False):
        """
        前向传播函数

        参数:
        q (Tensor): 查询矩阵
        k (Tensor): 键矩阵
        v (Tensor): 值矩阵
        ori_k (Tensor, optional): 原始键矩阵，默认为None
        mask (Tensor, optional): 掩膜矩阵，默认为None
        cross (bool, optional): 是否为交叉注意力，默认为False

        返回:
        Tuple: 输出的张量和注意力权重
        """
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat_interleave(self.head_num, 0)
        y, weights = self.scaled_dotproduct(
            q, k, v, mask, cross
        )  # [2,225,64]->#[2,225,2,32]  #[2,225,2,32]->#[2,2,225,32]  #[2,2,225,32]->#[4,225,32]
        y = self._reshape_from_batches(y)
        y = self.linear_o(y)
        return y, weights

    @staticmethod
    def gen_history_mask(x):
        """
        生成历史掩膜

        参数:
        x (Tensor): 输入张量

        返回:
        Tensor: 生成的历史掩膜
        """
        batch_size, seq_len, _ = x.size()
        return (
            torch.tril(torch.ones(seq_len, seq_len))
            .view(1, seq_len, seq_len)
            .repeat(batch_size, 1, 1)
        )

    def _reshape_to_batches(self, x):
        """
        将输入张量重塑为批次形式

        参数:
        x (Tensor): 输入张量

        返回:
        Tensor: 重塑后的张量
        """
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return (
            x.reshape(batch_size, seq_len, self.head_num, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.head_num, seq_len, sub_dim)
        )

    # [2,225,64]->#[2,225,2,32]  #[2,225,2,32]->#[2,2,225,32]  #[2,2,225,32]->#[4,225,32]

    def _reshape_from_batches(self, x):
        """
        将批次形式的张量重塑为原始形状

        参数:
        x (Tensor): 输入张量

        返回:
        Tensor: 重塑后的张量
        """
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return (
            x.reshape(batch_size, self.head_num, seq_len, in_feature)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, out_dim)
        )

    def extra_repr(self):
        """
        额外信息的字符串表示

        返回:
        str: 额外信息
        """
        return "in_features={}, head_num={}, bias={}, activation={}".format(
            self.in_features,
            self.head_num,
            self.bias,
            self.activation,
        )

    def scaled_dotproduct(self, query, key, value, mask=None, cross_att=False, tmp=0.1):
        """
        计算缩放点积注意力

        参数:
        query (Tensor): 查询张量
        key (Tensor): 键张量
        value (Tensor): 值张量
        mask (Tensor, optional): 掩膜张量，默认为None
        cross_att (bool, optional): 是否为交叉注意力，默认为False
        tmp (float, optional): 缩放因子，默认为0.1

        返回:
        Tuple: 注意力输出和权重
        """
        assert (cross_att and mask is not None) or (not cross_att and mask is None)
        dk = query.shape[-1]
        if not cross_att:
            scores = torch.einsum("bmc,bnc->bmn", query, key) / (math.sqrt(dk) + 1e-9)
        else:
            query, key = F.normalize(query, dim=2), F.normalize(key, dim=2)
            scores = torch.einsum("bmc,bnc->bmn", query, key) / tmp
        weight = scores

        if cross_att:
            attention = F.softmax(scores, dim=-2)
            attention = attention.masked_fill(mask == 0, 0)
            weight = (weight * mask).sum(2) / (mask.sum(2) + 1e-9)
        else:
            attention = F.softmax(scores, dim=-1)
            weight = weight.mean(2)

        if self.head_num > 1:
            weight = weight.reshape(
                weight.size(0) // self.head_num, self.head_num, weight.size(1)
            )
            weight = weight.mean(1)

        return attention.matmul(value), weight
