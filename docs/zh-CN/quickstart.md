# 安装与第一次运行

Core 支持 Python 3.10–3.14；训练资格使用 Python 3.12。
先在新目录中创建虚拟环境，无需 clone 仓库。估算只用元数据，训练才需要权重。

## Windows PowerShell：估算与网页

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python -m pip install --index-url https://pypi.org/simple canifinetune==0.4.1
canifinetune estimate --model Qwen/Qwen2.5-1.5B-Instruct --method qlora --gpu-vram-gb 16 --seq-len 2048 --offline
canifinetune demo
```

估算输出为 **8.420 GiB / YES / medium**。打开打印出的 `http://127.0.0.1:8765`，
修改配置、查看分项和复制命令。在当前终端按 Ctrl+C 停止网页，或在另一个已激活环境的终端继续。

## Windows CUDA：真实 QLoRA

需要兼容 CUDA 12.4 wheel 的 NVIDIA 驱动，先检查当前空闲显存。
下面的原生 Windows RTX 4080 路径使用已验证的依赖组合：

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c https://raw.githubusercontent.com/DaoyuanLi2816/can-i-finetune-this/v0.4.1/constraints/train-recommended.txt "canifinetune[train]==0.4.1"
canifinetune recipe --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --method qlora --seq-len 256 --max-steps 2 --grad-accum 1 --output my-recipe
python my-recipe/train.py --config my-recipe/config.yaml
python my-recipe/eval_smoke.py --output-dir my-recipe/output --max-new-tokens 8
```

首次训练下载约 988 MB 权重及少量 tokenizer / metadata。示例执行两次真实更新，保存 adapter，
再重载固定版本基座与 adapter 进行简单生成。这是流程演示，不是语言能力评测。
替换数据时默认不会自动截断；过长、畸形或监督无效的记录会明确报错。

## Linux / WSL shell：离线 CPU 流程 {#cpu}

Linux CPU 已在 CI 中执行；本版本未完成 WSL GPU 资格验证。
随机 tiny model 在本地创建，无需下载 Hub 权重：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHONUTF8=1
python -m pip install --index-url https://pypi.org/simple canifinetune==0.4.1
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c https://raw.githubusercontent.com/DaoyuanLi2816/can-i-finetune-this/v0.4.1/constraints/train-recommended.txt "canifinetune[train]==0.4.1"
canifinetune smoke-model --output tiny-local
canifinetune recipe --model tiny-local --method lora --device cpu --base-dtype fp32 --optimizer adamw_torch --seq-len 128 --max-steps 2 --grad-accum 1 --offline --output cpu-recipe
python cpu-recipe/train.py --config cpu-recipe/config.yaml
python cpu-recipe/eval_smoke.py --output-dir cpu-recipe/output --max-new-tokens 2
```

Windows CPU 可先完成 PowerShell 环境设置，再使用相同的 CPU 安装和 smoke 命令。
CPU Torch wheel 无法运行 CUDA QLoRA。

## 如何确认成功

`output/run.json` 的 `status` 应为 `success`，更新数应符合配置，重载命令也必须返回成功。
full 保存 `output/model`，LoRA / QLoRA 保存 `output/adapter`。已有输出目录受到保护；
重新实验时请使用新目录。保存产物不包含完整 optimizer / RNG 恢复状态。

估算的 YES 是规划判断。benchmark 使用全长度合成 labels，可能比短示例数据更耗显存。
比较**进程 reserved 峰值**与不含单列安全余量的 process proxy；设备可用显存属于另一口径。

[兼容性](../compatibility.md) · [训练与数据约定](../training.md) ·
[测量与限制](../evidence.md) · [排错](../troubleshooting.md)
