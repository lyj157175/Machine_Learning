import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np



'''
整体的x维度: [batch, max_len, d_model], 一直保持不变
'''

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy  # 深拷贝，修改互不影响
    attn = MultiHeadedAttention(h, d_model)      #  attn实例化  返回(512, 512)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # FF实例化
    position = PositionalEncoding(d_model, dropout)    # position实例化   返回 input_dim + position_dim

    # EncoderDecoder模型的五大组件
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),   # encoder模型实例化
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),   # decoder实例化
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),   # encoder输入: input_embed + position_embed
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),   # decoder输入：input_embed + position_embed
        Generator(d_model, tgt_vocab)   # linear+softmax来输出: d_model -> vocab
    )

    # 参数初始化
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
        self.src_embed = src_embed   # encoder中将输入src转化为input_embed + position_embed
        self.tgt_embed = tgt_embed   # decoder中将输入tgt转化为input_embed + position_embed
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
        self.norm = LayerNorm(encoderlayer.d_model)

    def forward(self, x, mask):
        # [batch, max_len, d_model]
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


# 小trick，每经过一层encoder和decoder后都会有一定的偏差，不容易收敛，因此需要加入layernorm
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
        self.norm = LayerNorm(decoderlayer.d_model)

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
        div_term = torch.exp(torch.arange(0., d_model, 2) *-(math.log(10000.0,)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)   # 序列化

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


# scale dot-product
# attention(Q, K, V) = softmax(Q*K.T/sqrt(d_k))*V
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
        self.linears = clones(nn.Linear(d_model, d_model), 4)   # 4层linears
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
        # 第三部：填充到第一个纬度, 维度交换
        # shape:[batch_size,8,max_length,64]
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
                                            # q*wq, k*wk, v*wv  self.linears调用了3层

        # 进入到attention之后纬度不变，shape:[batch_size,8,max_length,64]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 将纬度进行还原
        # 交换纬度：[batch_size,max_length,8,64]
        # 纬度还原：[batch_size,max_length,512]
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 最后与WO大矩阵相乘 shape:[512,512]
        return self.linears[-1](x)   #调用第四层linear


# Mask机制
# Transformer 模型里面涉及两种mask，padding mask 和 sequence mask。
# padding mask 在所有的 scaled dot-product attention 里面都需要用到，
# 而 sequence mask 只有在 decoder 的 self-attention 里面用到。
#
# Padding Mask
# 因为每个批次输入序列长度是不一样的也就是说，我们要对输入序列进行对齐。
# 具体来说，就是给在较短的序列后面填充 0。但是如果输入的序列太长，则是截取左边的内容，把多余的直接舍弃。
# 因为这些填充的位置，其实是没什么意义的，所以我们的attention机制不应该把注意力放在这些位置上，所以我们需要进行一些处理。
# 具体的做法是，把这些位置的值加上一个非常大的负数(负无穷)，这样的话，经过 softmax，这些位置的概率就会接近0！
# 而我们的 padding mask 实际上是一个张量，每个值都是一个Boolean，值为 false 的地方就是我们要进行处理的地方。
#
# Sequence mask
# 文章前面也提到，sequence mask 是为了使得 decoder 不能看见未来的信息。也就是对于一个序列，
# 在 time_step 为 t 的时刻，我们的解码输出应该只能依赖于 t 时刻之前的输出，而不能依赖 t 之后的输出。
# 因此我们需要想一个办法，把 t 之后的信息给隐藏起来。 那么具体怎么做呢？也很简单：产生一个上三角矩阵，上三角的值全为0。
# 把这个矩阵作用在每一个序列上，就可以达到我们的目的。
#
# 对于 decoder 的 self-attention，里面使用到的 scaled dot-product attention，同时需要padding mask 和
# sequence mask 作为 attn_mask，具体实现就是两个mask相加作为attn_mask。
# 其他情况，attn_mask 一律等于 padding mask。



### 解码过程
# Decoder的最后一个部分是过一个linear layer将decoder的输出扩展到与vocabulary size一样的维度上。
# 经过softmax 后，选择概率最高的一个word作为预测结果。在做预测时，步骤如下：
# （1）给 decoder 输入 encoder 对整个句子 embedding 的结果 和一个特殊的开始符号 。
#  decoder 将产生预测，在我们的例子中应该是 ”I”。 　　
# （2）给 decoder 输入 encoder 的 embedding 结果和 “I”，在这一步 decoder预测 “am”。
# （3）给 decoder 输入 encoder 的 embedding 结果和 “I am”，在这一步 decoder预测 “a”。
# （4）给 decoder 输入 encoder 的 embedding 结果和 “I am a”，在这一步 decoder预测 “student”。
# （5）给 decoder 输入 encoder 的 embedding 结果和 “I am a student”, decoder应该输出 ”。”
# （6）然后 decoder 生成了 ，翻译完成。