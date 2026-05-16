# canifinetune benchmark comparison

| model | method | seq | bs | rank | quant | ckpt | opt | peak GB (meas) | estimated GB | tok/s | OOM? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | qlora | 1024 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 3.30 | 2.41 | 1995.33 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 1024 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 4.36 | 3.00 | 1351.69 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 2048 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 7.10 | 3.16 | 1470.40 | no |
| `Qwen/Qwen2.5-3B-Instruct` | qlora | 1024 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 5.54 | 3.86 | 1158.51 | no |
| `sshleifer/tiny-gpt2` | lora | 128 | 1 | 8 | bf16 | off | adamw_torch | 0.12 | 2.08 | 1735.45 | no |
