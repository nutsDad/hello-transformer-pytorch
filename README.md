# Hello Transformer PyTorch

一个面向学习、实验和小规模训练的 PyTorch Transformer 项目。它同时保留了英译中的 Encoder-Decoder Transformer，并提供完整的 **Decoder-only 自回归语言模型**：可在 Windows CPU 上运行，也可通过配置切换到 CUDA GPU 服务器训练和推理。

项目默认安全地运行在 CPU。训练、评估、检查点、实验日志和推理服务均由同一份配置驱动，便于从本地验证迁移到 GPU 环境。

## 功能概览

| 架构 | 配置值 | 用途 | 训练目标 | 推理方式 |
| --- | --- | --- | --- | --- |
| Encoder-Decoder | `encoder_decoder` | 英文到中文翻译 | 目标句 token 预测 | Beam Search |
| Decoder-only | `decoder_only` | 续写与语言模型实验 | 下一个 token 预测 | 贪心解码或 top-k 采样 |

`encoder_only` 分类/句向量模式已被移除并替换为 `decoder_only`。以前保存的 `encoder_only` 检查点与新架构不兼容；翻译的历史无元数据权重仍保持兼容。

## 架构说明

### Decoder-only 语言模型

Decoder-only 模型由 token embedding、位置编码、N 层因果自注意力块、前馈网络和词表投影层组成。每个位置只能关注自身及其左侧 token，因此适合自回归生成。

```text
tokens -> Embedding + Positional Encoding
       -> [Causal Self-Attention -> FFN] x N
       -> LayerNorm -> Vocabulary Projection
       -> next-token probabilities
```

训练时，序列 `[BOS, w1, w2, EOS]` 被切分为：

```text
输入: [BOS, w1, w2]
目标: [w1,  w2, EOS]
```

损失函数为忽略 `<pad>` 的 token 级交叉熵。推理时，每一步将新 token 拼接回输入，直到生成 `<eos>` 或达到 `decoder_only_max_new_tokens`。

## 环境要求

- Python 3.10 或更高版本
- Windows CPU：安装 CPU 版 PyTorch 即可
- GPU 服务器：NVIDIA 驱动、CUDA 兼容的 PyTorch 和可见 GPU

CPU 安装示例：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

GPU 服务器请从 [PyTorch 安装页](https://pytorch.org/get-started/locally/) 选择与驱动/CUDA 匹配的安装命令，然后安装其余依赖：

```powershell
pip install -r requirements.txt
```

验证运行时：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 配置与运行模式

所有开关都在 `config.py`，也可使用环境变量覆盖。最常用的变量如下：

```python
runtime_profile = "windows_cpu"       # windows_cpu | gpu_server | auto
model_architecture = "decoder_only"   # decoder_only | encoder_decoder
model_preset = "base"                 # base | small | smoke_test
```

| `runtime_profile` | 行为 | 适用场景 |
| --- | --- | --- |
| `windows_cpu` | 强制使用 CPU | Windows 本地电脑、功能验证 |
| `gpu_server` | 强制使用 CUDA；无 GPU 时明确报错 | 单机 GPU 服务器 |
| `auto` | 检测到 CUDA 则用 GPU，否则用 CPU | 同一份代码跨环境运行 |

| `model_preset` | `d_model` / 层数 / FFN | 建议 |
| --- | --- | --- |
| `base` | 512 / 6 / 2048 | 默认教学配置 |
| `small` | 256 / 4 / 1024 | 资源有限的实验 |
| `smoke_test` | 64 / 2 / 256 | CPU 快速功能验证，不用于效果评估 |

PowerShell 中临时切换配置：

```powershell
$env:TRANSFORMER_PROFILE = "windows_cpu"
$env:TRANSFORMER_MODEL_ARCHITECTURE = "decoder_only"
$env:TRANSFORMER_MODEL_PRESET = "smoke_test"
python main.py --action train
```

`smoke_test` 会自动使用仓库内的小型 `decoder_only_train.json` 样例；`base` 和 `small` 默认读取 `data/corpus.en`。可在 `config.py` 将 `decoder_only_*_data_path` 指向自己的数据集。

## Decoder-only：训练、评估与生成

### 数据格式

支持两种 UTF-8 数据格式，每一行或每一项是一条独立训练文本。

纯文本：

```text
transformers predict the next token
causal masks hide future tokens
```

JSON：

```json
[
  "transformers predict the next token",
  {"text": "causal masks hide future tokens"}
]
```

选择与语料一致的 SentencePiece 分词器：

```python
decoder_only_tokenizer = "english"  # 或 "chinese"
decoder_only_max_sequence_length = 128
```

读取数据时会折叠多余空白、过滤过短文本并按配置去重。训练启动时会记录原始/保留文档数、重复和空文本丢弃数、字符长度分布，便于在训练前发现语料问题。

### 可复现数据切分

对自己的原始语料，先生成固定且互不重叠的训练、验证和测试集：

~~~powershell
python tools/prepare_lm_data.py --input data/my_corpus.txt --output-dir data/my_lm_split --dev-ratio 0.05 --test-ratio 0.05 --seed 42
~~~

该命令支持纯文本和本项目的 decoder-only JSON，输出 train.json、dev.json、test.json 及 dataset_report.json。报告记录清洗结果、随机种子和每个切分的样本数。随后在 config.py 设置三个 decoder_only 数据路径。

### 训练和评估

```powershell
$env:TRANSFORMER_MODEL_ARCHITECTURE = "decoder_only"
python main.py --action train
python main.py --action evaluate
```

每次训练会在 `run/train/exp*/weights/` 写入：

- `best_loss.pth`：开发集 token loss 最低的检查点
- `last.pth`：当前训练最后一个 epoch 的检查点

评估或生成前，把 `inference_model_path` 设置为需要加载的 `decoder_only` 检查点。检查点保存架构元数据，加载时会拒绝错误架构，避免将翻译权重误用为语言模型权重。

每个实验目录还包含：

- config.json：完整配置快照，可复现模型结构、数据路径、随机种子和运行环境。
- metrics.jsonl：逐 epoch 的结构化指标，易于脚本分析。
- tensorboard/：TensorBoard event 文件，记录 loss、perplexity、学习率和单 epoch 耗时。

启动 TensorBoard：

~~~powershell
tensorboard --logdir run/train
~~~

Decoder-only 的评估会同时输出 token loss 和 **perplexity**。perplexity 越低，代表模型对验证/测试文本的平均预测不确定性越低；它应与生成样例一起观察，不能单独代表生成质量。

训练默认使用 seed=42 和确定性模式。通过环境变量调整：

~~~powershell
$env:TRANSFORMER_SEED = "123"
$env:TRANSFORMER_DETERMINISTIC = "true"
~~~

断点续训会恢复模型参数、Adam 状态、Noam 学习率步数和已完成 epoch。epoch_num 表示目标总 epoch，必须大于断点的已完成 epoch：

~~~powershell
$env:TRANSFORMER_RESUME_CHECKPOINT = "run/train/exp/weights/last.pth"
$env:TRANSFORMER_MODEL_ARCHITECTURE = "decoder_only"
python main.py --action train
~~~

评估或生成前，将 inference_model_path 或 TRANSFORMER_INFERENCE_CHECKPOINT 设置为需要加载的 decoder_only 检查点。检查点保存架构元数据，加载时会拒绝错误架构，避免将翻译权重误用为语言模型权重。

### Python 生成接口

```python
from translate import build_inference_model, one_sentence_generate

model = build_inference_model()
text = one_sentence_generate(
    "transformers",
    model,
    max_new_tokens=40,
    temperature=0.8,
    top_k=20,
    do_sample=True,
)
print(text)
```

- `do_sample=False`：贪心解码，结果可复现。
- `do_sample=True`：从 top-k 分布采样，输出更多样。
- `temperature` 必须大于 0；较低值更保守，较高值更发散。

也可运行交互式推理：

```powershell
python translate.py
```

### 批量生成与 KV Cache

批量推理接口会为同一批 prompt 复用一次前向预填充，并在后续每一步复用每层 Attention 的 KV Cache，避免反复计算完整历史上下文：

~~~python
from translate import build_inference_model, generate_texts

model = build_inference_model()
print(generate_texts(
    ["transformers", "pytorch"],
    model,
    max_new_tokens=32,
    do_sample=True,
    top_k=20,
    seed=123,
))
~~~

seed 可使采样结果可复现；stop_token_ids 可传入额外停止 token ID，EOS 始终会停止生成。模型会检查 prompt 长度加 max_new_tokens 是否超过 decoder_only_context_length，超过上下文窗口时直接报错，而不会静默截断生成。

### HTTP 推理服务

服务一次加载模型，并提供批量 POST /generate 接口：

~~~powershell
$env:TRANSFORMER_MODEL_ARCHITECTURE = "decoder_only"
$env:TRANSFORMER_MODEL_PRESET = "smoke_test"
python serve.py --checkpoint run/train/exp/weights/best_loss.pth --host 127.0.0.1 --port 8000
~~~

健康检查：

~~~powershell
Invoke-RestMethod http://127.0.0.1:8000/health
~~~

生成请求：

~~~powershell
$body = @{ prompts = @("transformers", "pytorch"); max_new_tokens = 32; do_sample = $true; top_k = 20; seed = 123 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/generate -Method Post -ContentType "application/json" -Body $body
~~~

单次请求最多接收 decoder_only_server_max_batch_size 条 prompt。服务仅面向可信内网或开发环境；生产部署还应在反向代理、认证、限流、请求大小限制和进程管理之后运行。

## Encoder-Decoder：英译中

原翻译功能继续保留。切换配置后运行相同入口：

```powershell
$env:TRANSFORMER_MODEL_ARCHITECTURE = "encoder_decoder"
python main.py --action train
python translate.py
```

翻译 JSON 是二维数组，每项为 `[英文, 中文]`：

```json
[["Hello world", "你好，世界"]]
```

单句翻译接口：

```python
from translate import one_sentence_translate

print(one_sentence_translate("The model runs on a local CPU."))
```

## GPU 训练建议

1. 设定 `runtime_profile = "gpu_server"` 或 `TRANSFORMER_PROFILE=gpu_server`。
2. 确认 `torch.cuda.is_available()` 为 `True`。
3. 根据显存调整 `batch_size`、`decoder_only_max_sequence_length` 与 `model_preset`。
4. 多 GPU 时将 `RUNTIME_PROFILES["gpu_server"]["use_data_parallel"]` 设为 `True`。

本项目使用标准全精度训练，适合学习和中小规模实验。更大模型训练建议进一步加入混合精度、梯度累积、断点续训中的优化器状态、分布式数据并行和分片检查点。

## 项目结构

```text
config.py                 运行模式、架构、路径和超参数
main.py                   训练与评估统一入口
translate.py              翻译与 decoder-only 生成入口
beam_decoder.py           Encoder-Decoder 的 Beam Search
model/tf_model.py         Transformer 基础组件和两种模型架构
model/train_utils.py      损失计算、Noam 优化器和可恢复调度状态
model/checkpoint_utils.py 检查点保存、加载和架构校验
tools/data_loader.py      翻译与自回归语言模型数据集、清洗和统计
tools/experiment.py       随机种子、配置快照、JSONL/TensorBoard 实验追踪
tools/prepare_lm_data.py  可复现的语料清洗与 train/dev/test 切分
serve.py                  FastAPI 批量推理服务
tokenizer/                SentencePiece 分词器及训练脚本
data/                     已发布语料和小型 decoder-only 样例
weights/                  已发布的历史翻译权重（Git LFS）
```

## 公开数据与安全

- 语料和现有 `.pth` 权重已随仓库公开；大权重使用 Git LFS 管理。克隆前请先执行 `git lfs install`。
- 现有公开权重属于翻译模型，不能作为 `decoder_only` 语言模型检查点加载。请先训练新的 decoder-only 模型。
- 请勿提交 API Key、密码、私有语料、未获授权的数据或权重。
- 仅加载可信来源的检查点。新版本优先使用 PyTorch 的 `weights_only` 安全加载选项。
