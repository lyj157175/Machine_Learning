from Transformer import *
import time

# model = make_model(src_vocab=10, tgt_vocab=10, N=2)
# batch_size = 16
# seq_len = 50  # 序列长度
# d_model = 512  # 词向量维度
# h = 8  # 头数量
# x = torch.randn(batch_size, seq_len, d_model)
#
#
# # 测试MultiHeadedattention的过程
# attn = MultiHeadedAttention(h=8, d_model=512)
# q = torch.randn(2, 10, 512)
# line_net = clones(nn.Linear(512, 512), 4)
# # shape:query=key=value---->[batch_size,8,max_length,64]
# q, k, v = [l(x).view(2, -1, 8, 64).transpose(1, 2) for l,x in zip(line_net, (q, q, q))]


# training
def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens  # 总tokens
        tokens += batch.ntokens   # 每个batch的tokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print('Epoch_step: {} Loss: {} Tokens per sec: {}'.format(i, loss/batch.ntokens, tokens/elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        # x====>[batch_size*max_length-1,vocab_size]
        # target====>[batch_size*max_length-1]
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # fill_就是填充
        true_dist.fill_(self.smoothing / (self.size - 2))
        # scatter_修改元素
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# (v, 30, 20)
def data_gen(v, batch, nbatches):
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, v, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)  # [batch, max_len]
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


class Batch:
    def __init__(self, src, trg=None, pad=0):
        # src：[batch_size,max_legth]
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            # self.trg表示去掉每行的最后一个单词=====》相当于t-1时刻
            self.trg = trg[:, :-1]

            # self.trg_y表示去掉每行的第一个单词=====》相当于t时刻
            # decode 就是使用encoder和t-1时刻去预测t时刻
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # eq: tgt=[
        #         [2,1,3],
        #         [2,3,1]
        #        ]
        #     shape=(2,3)

        #    tgt_mask=[
        #       [
        #            [1,1,1],
        #            [1,1,1]
        #        ]
        #       ]
        #      shape=[2,1,3]

        #   tgt.size(-1)=3
        #    subsequent_mask(3) 生成下三角矩阵
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion  # criterion 为LabelSmoothing方法
        self.opt = opt

    def __call__(self, x, y, norm):
        # x对对应于out，也就是预测的时刻[batch_size,max_length-1,vocab_size]
        # y对应于trg_y,也就是t时刻 [batch_size,max_length-1]
        x = self.generator(x)

        # x.contiguous().view(-1, x.size(-1)) ====>[batch_size*max_length-1,vocab_size]
        # y.contiguous().view(-1)=========>[batch_size*max_length-1]
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm


# Greedy Decodeing贪心解码
v = 11   # vocab
criterion = LabelSmoothing(size=v, padding_idx=0, smothing=0.0)
model = make_model(v, v, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

data_iter = data_gen(v, 30, 20)
loss_compute = SimpleLossCompute(model.generator, criterion, model_opt)
for epoch in range(10):
    model.train()
    run_epoch(data_iter, model, SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(v, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))




