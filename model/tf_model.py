import config
import math
import copy
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
将输入的离散词索引转换为连续的向量表示
例如，将词汇表中的第5个词映射为一个512维的向量
"""
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        # 初始化方法，传入模型的维度（d_model）和词汇表的大小（vocab）
        super(Embeddings, self).__init__()
        # Embedding层，将词汇表的大小映射为d_model维的向量
        self.lut = nn.Embedding(vocab, d_model)
        # 存储模型的维度 d_model
        self.d_model = d_model

    def forward(self, x):
        # 返回x对应的embedding矩阵（需要乘以math.sqrt(d_model)）
        # 这是为了保持词向量的方差，使其适应后续层的训练。
        return self.lut(x) * math.sqrt(self.d_model)

"""
为每个输入位置添加一个唯一的位置编码
这个编码会被添加到词向量中，使模型能够理解位置信息
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个size为 max_len(设定的最大长度)×embedding维度 的全零矩阵
        # 来存放所有小于这个长度位置对应的positional embedding
        pe = torch.zeros(max_len, d_model)
        # 生成一个位置下标的tensor矩阵(每一行都是一个位置下标)
        position = torch.arange(0., max_len).unsqueeze(1)
        # 这里幂运算太多，我们使用exp和log来转换实现公式中pos下面要除以的分母（由于是分母，要注意带负号）
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))

        # 根据公式，计算各个位置在各embedding维度上的位置纹理值，存放到pe矩阵中
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 加1个维度，使得pe维度变为：1×max_len×embedding维度
        # (方便后续与一个batch的句子所有词的embedding批量相加)
        pe = pe.unsqueeze(0)
        # 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)
        self.register_buffer('pe', pe)

    def forward(self, x, start_pos=0):
        # 将一个batch的句子所有词的embedding与已构建好的positional embeding相加
        # (这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值)
        end_pos = start_pos + x.size(1)
        if end_pos > self.pe.size(1):
            raise ValueError(
                f"Sequence end position {end_pos} exceeds positional encoding limit "
                f"{self.pe.size(1)}."
            )
        x = x + Variable(self.pe[:, start_pos:end_pos], requires_grad=False)
        return self.dropout(x)

"""
这个函数是Transformer模型的基础，它实现了模型的核心注意力机制
"""
def attention(query, key, value, mask=None, dropout=None):
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)

    # 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    """
        Encoder：主要使用padding mask处理不等长序列
        Decoder：同时使用padding mask和sequence mask
        padding mask处理填充部分
        sequence mask防止看到未来信息
    """
    # 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 将mask后的attention矩阵按照最后一个维度进行softmax
    p_attn = F.softmax(scores, dim=-1)

    # 如果dropout参数设置为非空，则进行dropout操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
    return torch.matmul(p_attn, value), p_attn


"""
这个MultiHeadedAttention类,实现多头注意力机制
"""
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0
        # 得到一个head的attention表示维度
        self.d_k = d_model // h
        # head数量
        self.h = h
        # 定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, past_key_value=None, use_cache=False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        # query的第一个维度值为batch size
        nbatches = query.size(0)
        # 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
        # 并将结果拆成h块，然后将第二个和第三个维度值互换(具体过程见上述解析)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        if past_key_value is not None:
            past_key, past_value = past_key_value
            if past_key.size(0) != nbatches:
                raise ValueError("KV cache batch size does not match the query batch size.")
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        present_key_value = (key, value) if use_cache else None
        # 调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        output = self.linears[-1](x)
        return (output, present_key_value) if use_cache else output

"""
定义一个层归一化的类
"""
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 初始化α为全1, 而β为全0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        # 这是一个前向传播函数，用于执行Layer Normalization操作
        # 输入x是神经网络层的输出
        # 按最后一个维度计算均值和方差
        # keepdim=True确保输出的维度与输入相同
        mean = x.mean(-1, keepdim=True)  # 计算最后一个维度的均值
        std = x.std(-1, keepdim=True)   # 计算最后一个维度的标准差

        # 返回Layer Norm的结果
        # Layer Norm公式: y = a * (x - mean) / sqrt(std^2 + eps) + b
        # 其中a和b是可学习的参数，eps是为了防止除以0的小常数
        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2

"""
Transformer模型中前馈神经网络的实现。
"""
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        位置前馈神经网络初始化函数
        参数:
            d_model: 模型的输入维度
            d_ff: 前馈神经网络中间层的维度
            dropout: dropout概率，默认为0.1
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 第一个线性层，将维度从d_model扩展到d_ff
        self.w_2 = nn.Linear(d_ff, d_model)  # 第二个线性层，将维度从d_ff压缩回d_model
        self.dropout = nn.Dropout(dropout)    # dropout层，用于防止过拟合

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))  # 先通过第一个线性层，然后应用ReLU激活函数，

"""
SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层连在一起
"""
class SublayerConnection(nn.Module):
    """
    SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层
    连在一起只不过每一层输出之后都要先做Layer Norm再残差连接
    sublayer是lambda函数
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    """
    sublayer参数是SublayerConnection类中的核心组件，
    它代表了Transformer中的具体处理层，通过这种设计实现了代码的模块化和灵活性。
    """
    def forward(self, x, sublayer):
        # 返回Layer Norm和残差连接后结果
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    """克隆模型块，克隆的模型块参数不共享"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # SublayerConnection的作用就是把multi和ffn连在一起
        # 只不过每一层输出之后都要先做Layer Norm再残差连接
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # d_model
        self.size = size

    def forward(self, x, mask):
        # 将embedding层进行Multi head Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 注意到attn得到的结果x直接作为了下一层的输入
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        使用循环连续eecode N次(这里为6次)
        这里的Eecoderlayer会接收一个对于输入的attention mask处理
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        # Self-Attention
        self.self_attn = self_attn
        # 与Encoder传入的Context进行Attention
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 用m来存放encoder的最终hidden表示结果
        m = memory

        # Self-Attention：注意self-attention的q，k和v均为decoder hidden
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Context-Attention：注意context-attention的q为decoder hidden，而k和v为encoder hidden
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)



class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        使用循环连续decode N次(这里为6次)
        这里的Decoderlayer会接收一个对于输入的attention mask处理
        和一个对输出的attention mask + subsequent mask处理
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


class CausalDecoderLayer(nn.Module):
    """A decoder block with causal self-attention and no cross-attention."""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(CausalDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask, past_key_value=None, use_cache=False):
        if use_cache:
            normalized = self.sublayer[0].norm(x)
            attention_output, present_key_value = self.self_attn(
                normalized,
                normalized,
                normalized,
                mask,
                past_key_value=past_key_value,
                use_cache=True,
            )
            x = x + self.sublayer[0].dropout(attention_output)
            return self.sublayer[1](x, self.feed_forward), present_key_value
        x = self.sublayer[0](x, lambda value: self.self_attn(value, value, value, mask))
        return self.sublayer[1](x, self.feed_forward)


class CausalDecoder(nn.Module):
    """A stack of decoder-only Transformer blocks."""

    def __init__(self, layer, N):
        super(CausalDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, past_key_values=None, use_cache=False):
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("KV cache must contain one entry for each decoder layer.")
        present_key_values = []
        for index, layer in enumerate(self.layers):
            past_key_value = None if past_key_values is None else past_key_values[index]
            if use_cache:
                x, present_key_value = layer(x, mask, past_key_value, use_cache=True)
                present_key_values.append(present_key_value)
            else:
                x = layer(x, mask)
        output = self.norm(x)
        return (output, tuple(present_key_values)) if use_cache else output


class DecoderOnlyTransformer(nn.Module):
    """Autoregressive Transformer language model with a causal attention mask."""

    def __init__(
        self,
        decoder,
        token_embed,
        generator,
        d_model,
        padding_idx=0,
        max_positions=5000,
    ):
        super(DecoderOnlyTransformer, self).__init__()
        self.decoder = decoder
        self.token_embed = token_embed
        self.generator = generator
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.max_positions = max_positions

    def make_causal_mask(self, tokens, past_length=0, past_key_padding_mask=None):
        sequence_length = tokens.size(1)
        key_length = past_length + sequence_length
        query_positions = past_length + torch.arange(sequence_length, device=tokens.device)
        key_positions = torch.arange(key_length, device=tokens.device)
        causal = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)).unsqueeze(0)
        current_padding = tokens != self.padding_idx
        if past_length:
            if past_key_padding_mask is None:
                past_key_padding_mask = torch.ones(
                    (tokens.size(0), past_length), dtype=torch.bool, device=tokens.device
                )
            key_padding = torch.cat([past_key_padding_mask, current_padding], dim=1)
        else:
            key_padding = current_padding
        return key_padding.unsqueeze(1) & causal

    def _embed(self, tokens, start_pos):
        embeddings = self.token_embed[0](tokens)
        return self.token_embed[1](embeddings, start_pos=start_pos)

    def forward(
        self,
        tokens,
        attention_mask=None,
        past_key_values=None,
        past_key_padding_mask=None,
        use_cache=False,
    ):
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].size(-2)
        if past_length + tokens.size(1) > self.max_positions:
            raise ValueError(
                f"Requested context length {past_length + tokens.size(1)} exceeds "
                f"decoder_only_context_length={self.max_positions}."
            )
        if attention_mask is None:
            attention_mask = self.make_causal_mask(
                tokens,
                past_length=past_length,
                past_key_padding_mask=past_key_padding_mask,
            )
        return self.decoder(
            self._embed(tokens, past_length),
            attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    # 实例化Attention对象
    attn = MultiHeadedAttention(h, d_model)
    # 实例化FeedForward对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 实例化PositionalEncoding对象
    position = PositionalEncoding(d_model, dropout)
    # 实例化Transformer模型对象

    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # 初始化模型参数
    # 遍历模型中的所有参数
    for p in model.parameters():
        # 判断参数是否为二维或更高维（例如权重矩阵，而不是偏置向量）
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(config.device)


def make_decoder_only_model(
    vocab_size,
    N=6,
    d_model=512,
    d_ff=2048,
    h=8,
    dropout=0.1,
    max_positions=None,
):
    """Build a decoder-only causal language model."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    max_positions = config.decoder_only_context_length if max_positions is None else max_positions
    position = PositionalEncoding(d_model, dropout, max_len=max_positions)
    model = DecoderOnlyTransformer(
        CausalDecoder(CausalDecoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, vocab_size), c(position)),
        Generator(d_model, vocab_size),
        d_model=d_model,
        padding_idx=config.padding_idx,
        max_positions=max_positions,
    )
    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
    return model.to(config.device)
