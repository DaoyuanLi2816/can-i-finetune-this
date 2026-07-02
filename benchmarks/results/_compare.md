# canifinetune benchmark comparison

| model | method | seq | bs | rank | quant | ckpt | opt | peak GB (meas) | estimated GB | tok/s | OOM? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | lora | 1024 | 1 | 16 | - | on | paged_adamw_8bit | 3.79 | 5.19 | 3761.51 | no |
| `Qwen/Qwen2.5-0.5B-Instruct` | qlora | 1024 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 3.30 | 5.01 | 3336.57 | no |
| `Qwen/Qwen2.5-0.5B-Instruct` | qlora | 2048 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 6.28 | 7.22 | 3445.37 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 1024 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 4.36 | 6.07 | 2482.60 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 1024 | 1 | 16 | nf4_double_quant | off | paged_adamw_8bit | 9.55 | 10.77 | 3002.86 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 1024 | 2 | 16 | nf4_double_quant | on | paged_adamw_8bit | 6.88 | 8.44 | 2661.52 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 2048 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 7.10 | 8.44 | 2326.95 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 4096 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 13.56 | 13.19 | 1661.93 | no |
| `Qwen/Qwen2.5-1.5B-Instruct` | qlora | 512 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 2.91 | 4.88 | 1497.69 | no |
| `Qwen/Qwen2.5-3B-Instruct` | qlora | 1024 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 5.54 | 7.31 | 1302.90 | no |
| `Qwen/Qwen2.5-7B-Instruct` | qlora | 1024 | 1 | 16 | nf4_double_quant | on | paged_adamw_8bit | 11.23 | 12.54 | 922.90 | no |
| `sshleifer/tiny-gpt2` | lora | 128 | 1 | 8 | - | on | paged_adamw_8bit | 0.14 | 2.16 | 1470.37 | no |
