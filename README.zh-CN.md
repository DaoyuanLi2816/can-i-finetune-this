<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/can-i-finetune-this/v0.4.1/docs/assets/logo.png" alt="canifinetune GPU 微调前检查标识" width="104">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/can-i-finetune-this/v0.4.1/docs/banner.svg" alt="canifinetune：先估算显存，再实测训练，生成可运行的微调配方" width="880">
</p>

<p align="center">
  <a href="https://daoyuanli2816.github.io/can-i-finetune-this/zh-CN/">中文入门</a> ·
  <a href="https://daoyuanli2816.github.io/can-i-finetune-this/">Documentation</a> ·
  <a href="https://pypi.org/project/canifinetune/">PyPI</a> ·
  <a href="https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.1/README.md">English</a>
</p>

[![CI](https://github.com/DaoyuanLi2816/can-i-finetune-this/actions/workflows/ci.yml/badge.svg)](https://github.com/DaoyuanLi2816/can-i-finetune-this/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/canifinetune.svg)](https://pypi.org/project/canifinetune/)
[![Docs](https://img.shields.io/badge/docs-online-0f766e)](https://daoyuanli2816.github.io/can-i-finetune-this/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.1/LICENSE)

**这张显卡能微调这个 LLM 吗？下载权重之前，先把训练计划算清楚。**

你有一张消费级 NVIDIA 显卡，想用 LoRA / QLoRA 微调一个开放权重模型：
显存主要花在哪里？序列长度、batch size、LoRA rank 应该怎么选？预算紧张时应该先调整什么？

`canifinetune` 把这些问题转成显存分项、配置建议和可运行的训练配方，再通过本地短实验检验计划。
核心估算**不需要 PyTorch，也不下载模型权重**。

## 先得到一个结果

在虚拟环境中安装，无需先 clone 仓库：

```console
python -m pip install canifinetune==0.4.1
canifinetune estimate --model Qwen/Qwen2.5-1.5B-Instruct --method qlora --gpu-vram-gb 16 --seq-len 2048 --offline
canifinetune demo
```

该配置得到 **8.420 GiB** 的规划预算，在 16 GiB 预算下为 `YES`，启发式证据等级为 `medium`。
结果包含普通 loss 的 logits 分配和单列的安全余量；它不是训练一定不会 OOM 的保证。

`demo` 提供 [127.0.0.1:8765](http://127.0.0.1:8765) 本地交互页面：选择模型、训练设置、
总显存和当前可用显存，查看分项并复制匹配的命令。网页调用同一套 Python 估算核心。

<p align="center">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/can-i-finetune-this/v0.4.1/docs/demo-desktop.jpg" alt="本地交互演示：配置表单、8.42 GiB 显存分项与可复制命令" width="880">
</p>

## 选择你的使用路径

| 你想做什么 | 从这里开始 | 得到什么 |
| --- | --- | --- |
| **判断显存预算、寻找配置** | [工作流指南](https://daoyuanli2816.github.io/can-i-finetune-this/workflows/) | 分项估算、假设说明和候选配置 |
| **跑一次真实微调** | [中文 quickstart](https://daoyuanli2816.github.io/can-i-finetune-this/zh-CN/quickstart/) | 固定版本 0.5B QLoRA、真实更新、adapter 和重载检查 |
| **不下载 Hub 权重验证流程** | [离线 CPU smoke](https://daoyuanli2816.github.io/can-i-finetune-this/zh-CN/quickstart/#cpu) | 本地随机 tiny model，完成更新、保存和重载 |
| **理解或贡献测量证据** | [测量与限制](https://daoyuanli2816.github.io/can-i-finetune-this/evidence/) | 指标口径、原始记录与主动脱敏导出 |

## 它如何工作

<picture>
  <source media="(max-width: 760px)" srcset="https://raw.githubusercontent.com/DaoyuanLi2816/can-i-finetune-this/v0.4.1/docs/assets/architecture-mobile.svg">
  <img src="https://raw.githubusercontent.com/DaoyuanLi2816/can-i-finetune-this/v0.4.1/docs/assets/architecture.svg" alt="模型元数据、显存和共享配置进入估算与推荐；训练配方和 benchmark 加载权重并产生可核查的运行证据" width="1000">
</picture>

一套配置约定连接估算与执行。显存模型分别处理权重、量化、可训练参数的梯度、优化器状态、
activations、logits/loss 和开销。生成的配方使用统一 Transformers / PEFT 运行时，保留 system
及多轮对话，并记录请求与实际配置。benchmark 覆盖加载、首次优化器状态分配和声明的有限更新。

- **看清预算：**包括大词表 loss 缓冲区和 QLoRA 中未量化的部分。
- **把配置带到执行：**精度、attention、targets、优化器、checkpointing、量化及监督规则明确可查。
- **得到可用产物：**区分完整模型和 adapter，分阶段保存，失败不会留下伪成功状态，并提供重载命令。
- **保留证据来源：**区分进程峰值、安全余量、历史拟合、新观测和未经审核的社区记录。

[架构说明](https://daoyuanli2816.github.io/can-i-finetune-this/how-it-works/) ·
[训练与数据约定](https://daoyuanli2816.github.io/can-i-finetune-this/training/)

## 已经测过什么

在原生 Windows RTX 4080 上，四个预先固定的 Qwen2.5-0.5B-Instruct 配置覆盖 LoRA / QLoRA、
序列 256 / 512，每个执行三次更新；进程 reserved 峰值为 **1.666–2.398 GiB**。
新旧估算预测相同，**MAPE 为 46.2%**，四个点都偏保守，未证明估算精度提升。

这些是单 GPU、单模型的小规模前瞻观测，独立于开发期间使用的历史数据，不能给出通用 OOM 概率。
发布资格检查还从安装产物执行 CPU full / LoRA、CUDA LoRA / QLoRA 的更新、保存和重载。
随机 tiny smoke 和短实验证明流程可运行，不代表微调后的语言能力。

[新观测与原始记录](https://daoyuanli2816.github.io/can-i-finetune-this/validation-0.4.0/) ·
[历史基线](https://daoyuanli2816.github.io/can-i-finetune-this/rtx4080_baselines/)

## 安装、兼容性与贡献

Core 支持 Python 3.10–3.14；训练资格覆盖 Python 3.12、Torch 2.6 和文档中的最低 / 推荐依赖栈。
原生 Windows CPU / CUDA、Linux CPU 已验证；WSL GPU、其他 GPU、Flash Attention 和 Liger 未完成资格验证。
CPU 需使用 fp32；执行不支持预量化基座、远程模型代码或分布式训练。保存的是推理产物，不包含完整训练恢复状态。

[兼容性与迁移](https://daoyuanli2816.github.io/can-i-finetune-this/compatibility/) ·
[排错指南](https://daoyuanli2816.github.io/can-i-finetune-this/troubleshooting/) ·
[贡献指南](https://github.com/DaoyuanLi2816/can-i-finetune-this/blob/v0.4.1/CONTRIBUTING.md)

欢迎改进已支持的模型家族、复现有界测量或改善第一次使用体验。证据导出需主动选择、手动提交，默认脱敏；
社区上传保持未审核状态，不会自动进入拟合或权威准确率统计。

MIT。维护者：Daoyuan Li。
