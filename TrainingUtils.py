from Modules import *


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                           Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
                           nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                           nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
                           Generator(d_model, len(tgt_vocab)))
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1 :
            nn.init.xavier_uniform_(p)
#     for name, p in model.named_parameters():
#         if p.dim() > 1 and name != "src_embed.0.lut.weight":
#             nn.init.xavier_uniform_(p)
#         elif name == "src_embed.0.lut.weight":
#             print("[Using pretrain embedding]")
#             weight_matirx = src_vocab.vectors
#             p.data.copy_(weight_matirx)
    return model


class Batch:
    """
    src_mask[batch_size,1,seq_len]
    make是类里面的一个静态方法，跟普通函数没什么区别，
    与类和实例都没有所谓的绑定关系，它只不过是碰巧存在类中的一个函数而已。
    不论是通过类还是实例都可以引用该方法。
    """
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        #sentences mask [batch_size,1,seq_len]
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]#remove eos
            self.trg_y = trg[:, 1:]#remove bos
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        print(tgt.size())[batch_size,seq_len-1]
        两种mask相加作为attention mask subsequent_mask[1,seq_len-1,seq_len-1]
        这样做padding前的句子长度小于seq_len时候，mask矩阵的下一行1的个数将不会再递增
        也就不是完全的下三角矩阵了
        tgt_mask.size()[bach_size,seq_len-1,seq_len-1]
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute,epoch,batch_size):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        if len(batch.src) != batch_size:
            break
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        # torch.cuda.empty_cache()
        if i % 2000 == 0:
            elapsed = time.time() - start
            print("Epoch:%d  Step: %d Loss: %f Tokens per Sec: %f" % (
                epoch,i, loss / batch.ntokens.cpu().numpy().astype("float32"), tokens.cpu().numpy().astype("float32") / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens.cpu().numpy().astype("float32")


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    """
    l_rate = d_model^(−0.5) * min(step_num^(−0.5), step_num * warmup_steps^(−1.5))
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        :param x:
        :param target:
        non_zero:返回tensor中非零元素下标
        :return:
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 1:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm.float()
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm.float().item()
