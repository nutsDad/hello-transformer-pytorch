# Hello Transformer PyTorch

一个面向大模型训练与推理初学者的 PyTorch Transformer 项目。项目保留英文到中文的 Encoder-Decoder 翻译模型，并提供完整的 Decoder-only 自回归语言模型：支持数据准备、训练、评估、断点续训、KV Cache 批量生成和 HTTP 推理服务。

默认运行环境是 Windows CPU，不需要显卡；也可通过配置切换到有 CUDA 的 GPU 服务器。本项目适合理解从语料到 token、从 loss 到 checkpoint、从 prompt 到生成结果的完整链路。

## 目录

1. 项目功能
2. 核心概念
3. 五分钟跑通
4. 本地启动
5. 安装环境
6. 配置系统
7. 数据准备
8. Decoder-only 训练与评估
9. Decoder-only 生成与服务
10. Encoder-Decoder 翻译
11. 实验产物、测试与排错
12. 项目结构与限制

## 项目功能

| 功能 | 说明 | 主要入口 |
| --- | --- | --- |
| Decoder-only 训练 | 学习根据历史 token 预测下一个 token | main.py --action train |
| Decoder-only 评估 | 输出 token loss 与 perplexity | main.py --action evaluate |
| 单条续写 | 贪心解码或 top-k 采样 | translate.py |
| 批量续写 | 多条 prompt 一次推理，生成阶段复用 KV Cache | translate.generate_texts |
| HTTP 推理 | FastAPI 健康检查和批量生成接口 | serve.py |
| 数据治理 | 文本规范化、空文本和短文本过滤、精确去重、统计 | tools/data_loader.py |
| 数据切分 | 固定随机种子生成 train、dev、test | tools/prepare_lm_data.py |
| 可复现实验 | 固定随机源、配置快照、JSONL 指标、TensorBoard | tools/experiment.py |
| 断点续训 | 恢复模型、Adam、Noam 学习率状态和 epoch | TRANSFORMER_RESUME_CHECKPOINT |
| 英译中 | Encoder-Decoder 训练、BLEU 评估、Beam Search | main.py、translate.py |

支持两种架构：

| 架构 | 配置值 | 适用任务 | 训练目标 | 推理方式 |
| --- | --- | --- | --- | --- |
| Encoder-Decoder | encoder_decoder | 英文到中文翻译 | 根据源句预测目标句 | Beam Search |
| Decoder-only | decoder_only | 续写和语言模型实验 | 根据历史预测下一个 token | 贪心、top-k、KV Cache |

早期的 encoder_only 分类和句向量模式已移除。历史翻译权重只能用于 encoder_decoder，不能用于 decoder_only。

## 核心概念

### Token、词表和特殊符号

模型不直接理解字符串。SentencePiece 会将文本切分为子词，并映射成整数 token ID。项目使用以下特殊 token：

| 名称 | ID | 作用 |
| --- | ---: | --- |
| PAD | 0 | 对齐 batch 中较短序列，不参与 loss |
| UNK | 1 | 词表外 token |
| BOS | 2 | 一段文本的开始 |
| EOS | 3 | 一段文本的结束和默认生成停止标志 |

### Decoder-only 为什么可以生成

Decoder-only 使用因果注意力。位置 i 只能读取位置 0 到 i，不能读取未来 token。

~~~text
完整序列: [BOS, w1, w2, EOS]
模型输入: [BOS, w1, w2]
训练标签: [w1,  w2, EOS]
~~~

模型结构：

~~~text
token IDs
  -> Embedding + Positional Encoding
  -> Causal Self-Attention + Feed Forward Network，重复 N 层
  -> LayerNorm
  -> 词表投影
  -> 下一个 token 的概率
~~~

推理时，模型预测一个 token 后将它追加到输入，再继续预测，直到输出 EOS 或达到 max_new_tokens。

### Loss 与 Perplexity

训练使用忽略 PAD 的 token 级交叉熵 loss。loss 越低，通常代表模型给正确下一个 token 的概率越高。

perplexity，简称 PPL，是 loss 的指数形式。它应与生成样例一起分析：

- train loss 下降说明模型在拟合训练集。
- dev loss 或 dev PPL 反弹，可能表示过拟合。
- 小样例、随机初始化或训练不足时，生成无意义文本是正常现象。

### KV Cache

生成阶段如果每一步都重新计算全部历史 token，会越来越慢。KV Cache 会缓存每层 Attention 已计算的 Key 和 Value：

~~~text
第一次: prompt -> hidden states + KV Cache
后续步: 新 token + KV Cache -> 新结果 + 更新后的 KV Cache
~~~

本项目的批量生成已实现 KV Cache。它适合学习与小规模服务，但不是完整的大规模推理引擎。

## 五分钟跑通

以下步骤使用 CPU、仓库小样例和 smoke_test 小模型，只用于确认环境和流程。

### 1. 进入目录

~~~powershell
cd D:\00_tranformer\Transformer-pytorch
~~~

### 2. 安装依赖

~~~powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
~~~

### 3. 选择运行模式

~~~powershell
$env:TRANSFORMER_PROFILE = "windows_cpu"
$env:TRANSFORMER_MODEL_ARCHITECTURE = "decoder_only"
$env:TRANSFORMER_MODEL_PRESET = "smoke_test"
~~~

### 4. 训练

~~~powershell
python main.py --action train
~~~

smoke_test 会读取 data/json/decoder_only_train.json。训练输出位于 run/train/exp*/weights。

### 5. 评估

请把路径改为本机实际产生的检查点路径：

~~~powershell
$env:TRANSFORMER_INFERENCE_CHECKPOINT = "run/train/exp/weights/best_loss.pth"
python main.py --action evaluate
~~~

### 6. 生成

~~~powershell
python translate.py
~~~

输入一句 prompt 后回车生成；直接输入空行退出。小样例模型不会生成高质量内容，这是预期行为。

## 本地启动

本节给出 Windows 本地电脑的完整启动流程。新手建议先使用 CPU 与 smoke_test，确认所有链路正常后再替换真实语料和更大的模型。

### 第一次启动

在 PowerShell 中执行：

~~~powershell
cd D:\00_tranformer\Transformer-pytorch
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

$env:TRANSFORMER_PROFILE = "windows_cpu"
$env:TRANSFORMER_MODEL_ARCHITECTURE = "decoder_only"
$env:TRANSFORMER_MODEL_PRESET = "smoke_test"
python main.py --action train
~~~

训练完成后，项目会在 run/train/exp、run/train/exp1 等目录下写入 decoder-only 检查点。

### 本地推理启动

下面的命令自动选择最近生成的 best_loss.pth。若没有找到文件，请先完成一次训练。

~~~powershell
$env:TRANSFORMER_INFERENCE_CHECKPOINT = (
  Get-ChildItem .\run\train -Recurse -Filter best_loss.pth |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName

python translate.py
~~~

输入一条英文 prompt 后按回车生成；直接输入空行退出。当前小样例主要用于验证流程，生成结果不代表正式模型质量。

这一步就是本地推理启动：translate.py 会读取 TRANSFORMER_INFERENCE_CHECKPOINT 指向的 decoder-only 权重并进入交互模式。启动前必须先设置 decoder_only 架构、与训练时一致的模型预设和有效 checkpoint 路径。

也可以在 Python 中调用单条或批量推理：

~~~python
from translate import build_inference_model, one_sentence_generate, generate_texts

model = build_inference_model()
print(one_sentence_generate("transformers", model, max_new_tokens=20))
print(generate_texts(["transformers", "pytorch"], model, max_new_tokens=20))
~~~

### 本地评估

~~~powershell
python main.py --action evaluate
~~~

程序会输出 test_loss 和 test_perplexity。评估依赖上一步设置的 TRANSFORMER_INFERENCE_CHECKPOINT。

### 本地 HTTP 服务

在保持上述环境变量的 PowerShell 窗口中运行：

~~~powershell
python serve.py --checkpoint $env:TRANSFORMER_INFERENCE_CHECKPOINT --host 127.0.0.1 --port 8000
~~~

另开一个 PowerShell 窗口验证：

~~~powershell
Invoke-RestMethod http://127.0.0.1:8000/health
~~~

服务运行时按 Ctrl+C 停止。默认仅监听本机地址 127.0.0.1，不会暴露到局域网或公网。

### 本地查看训练曲线

~~~powershell
python -m tensorboard.main --logdir run\train
~~~

打开命令输出的本地浏览器地址，可查看 loss、perplexity、学习率和 epoch 耗时。

## 安装环境

### 运行要求

- Python 3.10 或更高版本。
- Windows CPU 可以直接运行。
- GPU 服务器需要 NVIDIA 驱动、CUDA 兼容的 PyTorch 和可见 GPU。
- 推荐使用虚拟环境，避免污染系统 Python。

### Windows CPU

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
~~~

### GPU 服务器

先从 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择与驱动和 CUDA 匹配的 PyTorch 安装命令，再安装项目依赖。

~~~powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
~~~

只有 torch.cuda.is_available 输出 True 时才应使用 gpu_server。

### 依赖用途

| 依赖 | 用途 |
| --- | --- |
| torch | 模型、训练和 CPU/GPU 计算 |
| sentencepiece | 中英文子词分词 |
| sacrebleu | 翻译 BLEU 指标 |
| tqdm | 进度条 |
| tensorboard | 曲线可视化 |
| fastapi、uvicorn | HTTP 推理服务 |

## 配置系统

默认配置位于 config.py。短期实验优先使用环境变量，长期固定实验可直接修改 config.py。

### 设备配置

| TRANSFORMER_PROFILE | 行为 | 适用环境 |
| --- | --- | --- |
| windows_cpu | 强制 CPU | 本地电脑、功能验证 |
| gpu_server | 强制 CUDA；没有 CUDA 时立即报错 | GPU 服务器 |
| auto | 有 CUDA 用 GPU，否则 CPU | 同一代码跨机器 |

GPU 示例：

~~~powershell
$env:TRANSFORMER_PROFILE = "gpu_server"
$env:TRANSFORMER_MODEL_ARCHITECTURE = "decoder_only"
python main.py --action train
~~~

多 GPU 时可在 config.py 将 gpu_server 的 use_data_parallel 改为 True。该项目使用 DataParallel，适合学习；大规模训练推荐使用 DistributedDataParallel。

### 模型预设

| TRANSFORMER_MODEL_PRESET | d_model | 层数 | FFN 维度 | 建议 |
| --- | ---: | ---: | ---: | --- |
| smoke_test | 64 | 2 | 256 | 只验证流程 |
| small | 256 | 4 | 1024 | 资源有限实验 |
| base | 512 | 6 | 2048 | 默认教学配置 |

模型越大，CPU 时间、GPU 显存和训练成本越高。新手应先用 smoke_test 跑通，再逐步使用 small 或 base。

### 高频配置项

| 配置项 | 含义 | 初学建议 |
| --- | --- | --- |
| batch_size | 每次梯度更新的样本数 | CPU 从 2 或 4 开始 |
| epoch_num | 训练目标总轮数 | 先设为 1 |
| decoder_only_max_sequence_length | 训练文本最大 token 数 | 从 128 开始 |
| decoder_only_context_length | 推理上下文最大 token 数 | 不应小于训练长度 |
| decoder_only_max_new_tokens | 单次最多生成新 token 数 | 16 到 64 |
| decoder_only_temperature | 采样温度 | 1.0 是常用起点 |
| decoder_only_top_k | 每步候选数 | 20 到 50 常用 |
| decoder_only_do_sample | 是否随机采样 | False 为稳定贪心结果 |
| seed | 随机种子 | 默认 42 |
| deterministic | 优先确定性算法 | 调试时保持 True |

### 环境变量参考

~~~powershell
$env:TRANSFORMER_PROFILE = "windows_cpu"
$env:TRANSFORMER_MODEL_ARCHITECTURE = "decoder_only"
$env:TRANSFORMER_MODEL_PRESET = "smoke_test"
$env:TRANSFORMER_ACTION = "train"
$env:TRANSFORMER_SEED = "42"
$env:TRANSFORMER_DETERMINISTIC = "true"
$env:TRANSFORMER_TENSORBOARD = "true"
$env:TRANSFORMER_INFERENCE_CHECKPOINT = "run/train/exp/weights/best_loss.pth"
$env:TRANSFORMER_RESUME_CHECKPOINT = "run/train/exp/weights/last.pth"
$env:TRANSFORMER_STOP_TOKEN_IDS = "3,10,11"
~~~

这些变量只对当前 PowerShell 窗口有效。

## 数据准备

### Decoder-only 语料格式

支持 UTF-8 纯文本和 JSON。

纯文本中每个非空行是一条独立文档：

~~~text
transformers predict the next token
causal masks hide future information
~~~

JSON 必须是数组，元素可以是字符串或包含 text 字段的对象：

~~~json
[
  "transformers predict the next token",
  {"text": "causal masks hide future information"}
]
~~~

不要把翻译的二维数组直接交给 decoder-only。翻译数据应使用 encoder_decoder 流程。

### 数据清洗与统计

DecoderOnlyDataset 会：

1. 折叠多余空白。
2. 丢弃空文本。
3. 丢弃长度小于 decoder_only_min_characters 的文本。
4. 当 decoder_only_deduplicate 为 True 时移除完全重复文本。
5. 记录原始数、保留数、丢弃原因与字符长度分布。

这不是完整的数据合规工具。它不会识别近似重复、隐私信息、版权状态、数据泄漏或语义质量。请只使用有合法授权的语料。

### 固定切分真实语料

建议从原始语料一次性生成固定的 train、dev、test，避免同一文本同时进入训练和评估。

~~~powershell
python tools/prepare_lm_data.py --input data/my_corpus.txt --output-dir data/my_lm_split --dev-ratio 0.05 --test-ratio 0.05 --seed 42 --min-characters 2
~~~

输出：

~~~text
data/my_lm_split/
  train.json
  dev.json
  test.json
  dataset_report.json
~~~

dataset_report.json 包含清洗统计、随机种子和每个切分的文档数。相同输入和相同 seed 会得到相同的切分。

然后在 config.py 中设置三个 decoder_only 数据路径，并选择匹配的分词器：

~~~python
decoder_only_train_data_path = data_dir / "my_lm_split" / "train.json"
decoder_only_dev_data_path = data_dir / "my_lm_split" / "dev.json"
decoder_only_test_data_path = data_dir / "my_lm_split" / "test.json"
decoder_only_tokenizer = "english"
~~~

中文语料请使用 chinese 分词器。词表和语料语言不匹配会严重影响结果。

## Decoder-only 训练与评估

### 训练流程

~~~powershell
$env:TRANSFORMER_PROFILE = "windows_cpu"
$env:TRANSFORMER_MODEL_ARCHITECTURE = "decoder_only"
$env:TRANSFORMER_MODEL_PRESET = "smoke_test"
python main.py --action train
~~~

内部流程：

~~~text
加载和清洗文本
  -> SentencePiece 编码
  -> 组成 batch、补齐 PAD
  -> 构造因果 attention mask
  -> 预测下一个 token
  -> 忽略 PAD 计算交叉熵
  -> Adam 和 Noam 学习率更新
  -> 在 dev 集计算 loss 和 PPL
  -> 保存检查点和实验指标
~~~

### 评估

评估不会更新参数：

~~~powershell
$env:TRANSFORMER_MODEL_ARCHITECTURE = "decoder_only"
$env:TRANSFORMER_MODEL_PRESET = "smoke_test"
$env:TRANSFORMER_INFERENCE_CHECKPOINT = "run/train/exp/weights/best_loss.pth"
python main.py --action evaluate
~~~

输出含有 test_loss 和 test_perplexity。test 集不应参与训练。

### 实验目录和 TensorBoard

每次训练会创建 run/train/exp、run/train/exp1 等目录：

| 文件或目录 | 用途 |
| --- | --- |
| weights/best_loss.pth | dev loss 最低的模型 |
| weights/last.pth | 最后一个 epoch，适合续训 |
| config.json | 本次实验配置快照 |
| metrics.jsonl | 每个 epoch 一行 JSON 指标 |
| tensorboard/ | TensorBoard event 文件 |
| run/train/history.jsonl | 所有 exp 实验合并后的机器可读历史 |
| run/train/history.log | 所有 exp 实验合并后的可读训练日志 |
| run/train/tensorboard_history/ | 跨实验连续 TensorBoard 曲线 |

每次新的训练都会保留旧实验的记录，不会覆盖历史。查看全部历史训练日志：

~~~powershell
Get-Content .\run\train\history.log
Get-Content .\run\train\history.log -Tail 50
~~~

history.jsonl 适合脚本、Notebook 或数据分析工具读取；history.log 适合在 PowerShell 直接查看。

查看跨实验的连续曲线：

~~~powershell
python -m tensorboard.main --logdir run\train\tensorboard_history
~~~

可查看包含历史实验的 train loss、dev loss、perplexity、学习率和 epoch 耗时。若要按单个实验分别比较，也可使用 python -m tensorboard.main --logdir run\train。

### 可复现训练

训练会设置 Python、NumPy、PyTorch 和 DataLoader 的随机源：

~~~powershell
$env:TRANSFORMER_SEED = "123"
$env:TRANSFORMER_DETERMINISTIC = "true"
python main.py --action train
~~~

不同 GPU、驱动和 PyTorch 版本仍可能造成细微数值差异。确定性模式主要用于调试和公平对比实验。

### 断点续训

last.pth 会保存模型、架构、epoch、指标、Adam 状态和 Noam 调度状态。epoch_num 表示目标总 epoch，不是额外训练 epoch。

~~~powershell
$env:TRANSFORMER_MODEL_ARCHITECTURE = "decoder_only"
$env:TRANSFORMER_MODEL_PRESET = "smoke_test"
$env:TRANSFORMER_RESUME_CHECKPOINT = "run/train/exp/weights/last.pth"
python main.py --action train
~~~

例如 checkpoint 已完成 epoch 1，epoch_num 为 3，程序会从 epoch 2 继续至 epoch 3。模型架构、尺寸或学习率计划不匹配时会拒绝恢复。

## Decoder-only 生成与服务

### 命令行生成

~~~powershell
$env:TRANSFORMER_MODEL_ARCHITECTURE = "decoder_only"
$env:TRANSFORMER_MODEL_PRESET = "smoke_test"
$env:TRANSFORMER_INFERENCE_CHECKPOINT = "run/train/exp/weights/best_loss.pth"
python translate.py
~~~

### Python 单条与批量生成

~~~python
from translate import build_inference_model, one_sentence_generate, generate_texts

model = build_inference_model()
print(one_sentence_generate(
    "transformers",
    model,
    max_new_tokens=40,
    temperature=0.8,
    top_k=20,
    do_sample=True,
    seed=123,
))

print(generate_texts(
    ["transformers", "pytorch"],
    model,
    max_new_tokens=32,
    top_k=20,
    do_sample=True,
    seed=123,
))
~~~

| 参数 | 含义 |
| --- | --- |
| max_new_tokens | 最多生成多少新 token |
| temperature | 采样温度，必须大于 0 |
| top_k | 只保留概率最高的 k 个候选；0 或 None 表示不截断 |
| do_sample | False 为贪心；True 为随机采样 |
| seed | 固定采样随机性 |
| stop_token_ids | 额外停止 token ID；EOS 始终停止 |

生成会检查 prompt token 数加 max_new_tokens 是否超过 decoder_only_context_length。超过时请缩短 prompt、降低生成长度或提高上下文配置。

### HTTP 服务

服务只支持 decoder_only。启动时加载一次 checkpoint，后续请求复用内存中的模型：

~~~powershell
$env:TRANSFORMER_PROFILE = "windows_cpu"
$env:TRANSFORMER_MODEL_ARCHITECTURE = "decoder_only"
$env:TRANSFORMER_MODEL_PRESET = "smoke_test"
python serve.py --checkpoint run/train/exp/weights/best_loss.pth --host 127.0.0.1 --port 8000
~~~

健康检查：

~~~powershell
Invoke-RestMethod http://127.0.0.1:8000/health
~~~

批量生成：

~~~powershell
$body = @{ prompts = @("transformers", "pytorch"); max_new_tokens = 32; temperature = 0.8; top_k = 20; do_sample = $true; seed = 123 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/generate -Method Post -ContentType "application/json" -Body $body
~~~

请求 JSON：

~~~json
{
  "prompts": ["transformers", "pytorch"],
  "max_new_tokens": 32,
  "temperature": 0.8,
  "top_k": 20,
  "do_sample": true,
  "seed": 123,
  "stop_token_ids": [3]
}
~~~

成功响应：

~~~json
{
  "texts": ["generated text one", "generated text two"],
  "count": 2
}
~~~

单次请求上限由 decoder_only_server_max_batch_size 控制，默认 16。服务没有认证、限流、TLS 或生产级调度，不应直接暴露到公网。

## Encoder-Decoder 翻译

翻译功能保留，用于理解 Encoder、Decoder、交叉注意力和 Beam Search。

翻译 JSON 是二维数组，每项为英文和中文：

~~~json
[
  ["Hello world", "你好，世界"],
  ["The model runs on a CPU.", "模型运行在 CPU 上。"]
]
~~~

训练：

~~~powershell
$env:TRANSFORMER_PROFILE = "windows_cpu"
$env:TRANSFORMER_MODEL_ARCHITECTURE = "encoder_decoder"
$env:TRANSFORMER_MODEL_PRESET = "small"
python main.py --action train
~~~

翻译任务输出 BLEU，通常越高越好。Python 调用：

~~~python
from translate import one_sentence_translate
print(one_sentence_translate("The model runs on a local CPU."))
~~~

翻译检查点不能加载为 decoder-only，反之亦然。

## 检查点、测试与验证

新检查点包含 architecture 元数据。加载时会验证架构，并优先使用 PyTorch 的 weights_only 模式。仍然只应加载可信来源的权重。

运行回归测试：

~~~powershell
$env:TRANSFORMER_PROFILE = "windows_cpu"
$env:TRANSFORMER_MODEL_ARCHITECTURE = "decoder_only"
$env:TRANSFORMER_MODEL_PRESET = "smoke_test"
python -m unittest discover -s tests -v
~~~

当前测试覆盖数据清洗与固定切分、KV Cache 与完整前向一致性、带 seed 的批量采样、优化器和 Noam 状态恢复。

语法检查：

~~~powershell
python -m compileall -q .
~~~

## 项目结构

~~~text
config.py                 运行模式、路径和超参数
main.py                   训练与评估入口
translate.py              翻译、单条生成、批量 KV Cache 生成
serve.py                  FastAPI 批量生成服务
beam_decoder.py           翻译 Beam Search

model/
  tf_model.py             Transformer 基础组件、两种模型、KV Cache
  train_utils.py          loss、Noam 优化器和恢复状态
  checkpoint_utils.py     保存、安全加载和断点恢复

tools/
  data_loader.py          数据加载、清洗和统计
  experiment.py           随机种子、配置快照、JSONL、TensorBoard
  prepare_lm_data.py      decoder-only 语料固定切分
  tokenizer_utils.py      SentencePiece 分词器加载

tokenizer/                中英文 SentencePiece 模型和词表
data/                     公开语料和小样例
weights/                  历史翻译权重，使用 Git LFS
run/                      训练实验产物
tests/                    回归测试
~~~

## 常见问题

### 没有显卡或 CUDA 报错

使用 windows_cpu，不要使用 gpu_server：

~~~powershell
$env:TRANSFORMER_PROFILE = "windows_cpu"
~~~

auto 会在有 CUDA 时使用 GPU，否则使用 CPU。

### 评估或生成时提示架构不匹配

检查 TRANSFORMER_MODEL_ARCHITECTURE 与 checkpoint 是否同属一种架构。翻译权重只用于 encoder_decoder，语言模型权重只用于 decoder_only。

### 找不到检查点

训练目录可能是 exp、exp1、exp2。先查看 run/train 下实际目录，再把完整路径传给 TRANSFORMER_INFERENCE_CHECKPOINT 或 serve.py 的 checkpoint 参数。

### loss 高或生成无意义

优先检查：

1. 是否只训练了 smoke_test 小样例。
2. 分词器是否与语料语言匹配。
3. 语料是否过少、重复或噪声过大。
4. 模型是否太小或训练不足。
5. temperature 是否设置过高。

先验证流程，再逐步提高高质量语料规模、训练轮数和模型大小。

### 上下文超长

缩短 prompt、降低 max_new_tokens，或提高 decoder_only_context_length。后者会提高 Attention 计算量和内存需求。

### TensorBoard 命令找不到

使用：

~~~powershell
python -m tensorboard.main --logdir run\train\tensorboard_history
~~~

## 限制、安全与发布

- 本项目面向教学、小型实验和流程理解，不是完整的大模型训练或高性能推理框架。
- 未实现混合精度、梯度累积、DistributedDataParallel、量化、流式响应、连续批处理和生产级调度。
- 清洗逻辑只处理空白、短文本和完全重复，不保证语料合法、无偏见、无隐私或无版权风险。
- 不要提交 API Key、密码、个人信息、未授权语料或私有权重。
- HTTP 服务应只在可信内网或受保护环境中运行。
- 仓库的大 .pth 权重使用 Git LFS。克隆前执行 git lfs install。
- 发布权重前应评估数据授权、隐私泄漏和模型记忆风险。

推荐学习顺序：先运行 smoke_test，阅读 logs 和 TensorBoard，再准备自己的小型合法语料，最后逐步增加模型和数据规模。
