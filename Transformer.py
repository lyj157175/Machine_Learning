import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy  # 深拷贝，修改互不影响
    attn = MultiHeadedAttention(h, d_model)      #  attn实例化  返回(512, 512)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # FF实例化
    position = PositionalEncoding(d_model, dropout)    # position实例化   返回 input_dim + position_dim

    # EncoderDecoder模型的五大组件
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),   # encoder模型实例化
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),   # decoder实例化
        nn.Sequential(Embeddings(d_model, src_vocab),   # 将src_vocab转化为d_model, 返回[d_model, d_model]
                      c(position)),    # encoder输入:embedding+position_embedding
        nn.Sequential(Embeddings(d_model, tgt_vocab),    # 将tgt_vocab转化为d_model, 返回[d_model, d_model]
                      c(position)),   # decoder输入：embedding+position_embedding
        Generator(d_model, tgt_vocab)   # linear+softmax来输出: d_model -> vocab
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# 模型五大组件
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder       # encoder部分
        self.decoder = decoder       # decoder部分
        self.src_embed = src_embed   # encoder中将输入src转化为embedding + position
        self.tgt_embed = tgt_embed   # decoder中将输入tgt转化为embedding + position
        self.generator = generator   # linear+softmax来模型输出：d_model -> vocab

    def encode(self, src, src_mask):
        # 输入: encode(src, src_mask)
        return self.encoder(self.src_embed(src), src_mask)   # 输出表示为memory传入decoder

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # deocder的输入： (memory, src_mask, tgt, tgt_mask)
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    # 模型输入(src, tgt, src_mask, tgt_mask)
    def forward(self, src, tgt, src_mask, tgt_mask):
        # src=tgt:[batch, max_len]
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)  # 模型输出为decoder的输出

# 输出 [512, 512] -> [512, vocab]
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
class Encoder(nn.Module):
    def __init__(self, encoderlayer, N):
        super(Encoder, self).__init__()
        self.layers = clones(encoderlayer, N)
        self.norm = LayerNorm(encoderlayer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)   # 最后做一个LayerNorm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # 残差第一层
        return self.sublayer[1](x, self.feed_forward)    # 第二层


# 残差结构: 包括attn和FF层
class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# encoder后的norm
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
class Decoder(nn.Module):
    def __init__(self, decoderlayer, N):
        super(Decoder, self).__init__()
        self.layers = clones(decoderlayer, N)
        self.norm = LayerNorm(decoderlayer.size)

    def forward(self, memory, x, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(memory, x, src_mask, tgt_mask)
        return self.norm(x)     # 最后做一个LayerNorm  输出[d_model, d_model]


class DecoderLayer(nn.Module):
    # self_attn和src_attn都是拷贝的attn函数，一个东西
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, memory, x, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # 第一层
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))   # 第二层
        return self.sublayer[2](x, self.feed_forward)   #第三层

# 防止decoder看到未来信息
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  # 上三角矩阵
#     print(subsequent_mask)
    return torch.from_numpy(subsequent_mask) == 0


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)  # [d_model, d_model]
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0,)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)   # input_dim + position_dim


# 前馈网络层中包括： 两个线性转换和一个relu激活
# FF(𝑥)=max(0,𝑥𝑊1+𝑏1)𝑊2+𝑏2
class PositionwiseFeedForward(nn.Module):
    # d_model=512, d_ff=2048
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_w1(x))))


def attention(query, key, value, mask=None, dropout=None):
    # query=key=value---->[batch_size,8,max_length,64]

    d_k = query.size(-1)

    # k的纬度交换后为：[batch_size,8,64,max_length]
    # scores的纬度为:[batch_size,8,max_length,max_length]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # padding mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    # mask维三维
    def forward(self, query, key, value, mask=None):
        # query=key=value--->:[batch_size, max_legnth, embedding_dim=512]

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 第一步：将q,k,v分别与Wq，Wk，Wv矩阵进行相乘
        # shape:Wq=Wk=Wv----->[512,512]
        # 第二步：将获得的Q、K、V在第三个纬度上进行切分
        # shape:[batch_size,max_length,8,64]
        # 第三部：填充到第一个纬度
        # shape:[batch_size,8,max_length,64]
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 进入到attention之后纬度不变，shape:[batch_size,8,max_length,64]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 将纬度进行还原
        # 交换纬度：[batch_size,max_length,8,64]
        # 纬度还原：[batch_size,max_length,512]
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 最后与WO大矩阵相乘 shape:[512,512]
        return self.linears[-1](x)


