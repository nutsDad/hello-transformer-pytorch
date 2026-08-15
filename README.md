# Transformer PyTorch：本地 CPU、阿里云与 Encoder-only

这是一个基于 PyTorch 实现的教学型 Transformer 项目，支持通过配置切换两类模型：

- `encoder_decoder`：英译中的 Encoder-Decoder Transformer，含训练、验证、BLEU 评估和 Beam Search 推理。
- `encoder_only`：Encoder-only Transformer，支持文本分类训练、分类推理和句向量编码。

默认运行在 Windows CPU，无需 NVIDIA 显卡；也支持阿里云 NVIDIA GPU 实例。

## 快速开始

建议使用 Python 3.10 或更高版本。先安装与机器相符的 PyTorch，再安装其余依赖。

Windows CPU：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

阿里云 GPU：请根据实例驱动与 CUDA 版本，从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应安装命令，再执行：

```powershell
pip install -r requirements.txt
```

在 `config.py` 设置运行环境和模型结构：

```python
runtime_profile = "windows_cpu"          # windows_cpu | aliyun_gpu | auto
model_architecture = "encoder_only"      # encoder_only | encoder_decoder
```

CPU 首次验证建议缩小模型：

```python
d_model = 128
n_layers = 2
d_ff = 512
batch_size = 4
epoch_num = 1
```

启动训练、评估和交互式推理：

```powershell
python main.py --action train
python main.py --action evaluate
python translate.py
```

## 配置切换

### 运行环境

| `runtime_profile` | 适用环境 | 行为 |
| --- | --- | --- |
| `windows_cpu` | Windows 本地，无 NVIDIA 显卡 | 强制 CPU，默认值 |
| `aliyun_gpu` | 阿里云 NVIDIA GPU 实例 | 强制 CUDA；无可用 CUDA 时直接报错 |
| `auto` | 同一份代码在不同机器运行 | 有 CUDA 时使用 GPU，否则使用 CPU |

多 GPU 训练时，将 `config.py` 中 `aliyun_gpu` 的 `use_data_parallel` 改为 `True`。单 GPU 与 CPU 保持 `False`。

### 模型结构

| `model_architecture` | 功能 | 训练指标 |
| --- | --- | --- |
| `encoder_decoder` | 英文到中文翻译 | 开发集 BLEU |
| `encoder_only` | 文本分类和句向量 | 开发集准确率 |

## 翻译：Encoder-Decoder

```python
model_architecture = "encoder_decoder"
runtime_profile = "windows_cpu"
```

翻译数据为 JSON 数组，每项是 `[英文, 中文]`：

```json
[["Hello world", "你好，世界"]]
```

单句翻译：

```python
from translate import one_sentence_translate

print(one_sentence_translate("The model runs on a local CPU."))
```

在推理前，将 `inference_model_path` 设置为相应检查点。原项目纯 `state_dict` 翻译权重仍可加载。

## 分类与向量：Encoder-only

```python
model_architecture = "encoder_only"
encoder_only_num_labels = 2
encoder_only_label_names = ["negative", "positive"]
encoder_only_tokenizer = "english"
```

数据为 JSON 数组，每项包含文本和整数标签：

```json
[
  {"text": "this product is excellent", "label": 1},
  {"text": "this product is broken", "label": 0}
]
```

仓库保留 `data/json/encoder_only_{train,dev,test}.json` 三份极小样例。实际任务请使用合法数据，并确保标签范围为 `[0, encoder_only_num_labels)`。

```python
from translate import build_inference_model, one_sentence_encode, one_sentence_predict

model = build_inference_model()
print(one_sentence_predict("the service was excellent", model))
vector = one_sentence_encode("the service was excellent", model)
print(len(vector))
```

`one_sentence_predict` 返回类别 ID、类别名称和概率；`one_sentence_encode` 返回句向量。默认 `mean` 池化会忽略填充 token，也可将 `encoder_only_pooling` 设为 `cls`。

## 项目结构

```text
config.py                 运行环境、模型结构、路径和超参数
main.py                   统一训练与评估入口
translate.py              翻译、分类与向量推理入口
beam_decoder.py           Encoder-Decoder 的 Beam Search
model/tf_model.py         Transformer 与 Encoder-only 模型
model/train_utils.py      CPU/GPU 通用损失计算和 Noam 优化器
model/checkpoint_utils.py 检查点保存、加载与架构校验
tools/data_loader.py      翻译与分类数据集
tokenizer/                SentencePiece 分词器及训练脚本
```

## 安全与发布说明

- `.gitignore` 排除了完整语料、模型权重、训练输出、IDE 文件和 Python 缓存；这些文件未公开发布。
- 不要提交 API Key、账号密码、私有数据或未获授权的语料和权重。
- 仅加载可信来源的检查点。新版本优先启用 PyTorch 的 `weights_only` 安全加载选项；旧版 PyTorch 回退到兼容模式。
- 本项目面向学习与中小规模实验。大参数量训练还需要混合精度、梯度累积、分布式并行和分片检查点等工程能力。
