# canifinetune benchmark report

Each section below corresponds to one benchmark result JSON. `estimated` is what the static estimator predicted, `measured` is what was observed on this machine.

## Qwen__Qwen2.5-0.5B-Instruct_lora_s1024_b1_r16_steps3.json  —  OK

**Configuration**


| Field | Value |
| --- | --- |
| model | `Qwen/Qwen2.5-0.5B-Instruct` |
| method | lora |
| seq_len | 1024 |
| micro_batch_size | 1 |
| steps | 3 |
| lora_rank | 16 |
| quantization | - |
| optimizer | paged_adamw_8bit |
| gradient_checkpointing | True |
| attention | sdpa |

**Environment**


| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 (15.99 GB) |
| torch | 2.6.0+cu124 |
| CUDA | 12.4 |
| bf16 | True |

**Memory: estimated vs measured**


| Metric | Value |
| --- | --- |
| estimated total | 5.19 GB |
| measured peak (reserved) | 3.79 GB |
| measured peak (allocated) | 3.03 GB |
| final allocated | 1.25 GB |
| tokens/sec | 3761.51 |
| avg step time | 0.2722 s |
| last-step loss | 13.0640 |

## Qwen__Qwen2.5-0.5B-Instruct_qlora_s1024_b1_r16_steps3.json  —  OK

**Configuration**


| Field | Value |
| --- | --- |
| model | `Qwen/Qwen2.5-0.5B-Instruct` |
| method | qlora |
| seq_len | 1024 |
| micro_batch_size | 1 |
| steps | 3 |
| lora_rank | 16 |
| quantization | nf4_double_quant |
| optimizer | paged_adamw_8bit |
| gradient_checkpointing | True |
| attention | sdpa |

**Environment**


| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 (15.99 GB) |
| torch | 2.6.0+cu124 |
| CUDA | 12.4 |
| bf16 | True |

**Memory: estimated vs measured**


| Metric | Value |
| --- | --- |
| estimated total | 5.01 GB |
| measured peak (reserved) | 3.30 GB |
| measured peak (allocated) | 3.12 GB |
| final allocated | 1.30 GB |
| tokens/sec | 3336.57 |
| avg step time | 0.3069 s |
| last-step loss | 13.0227 |

## Qwen__Qwen2.5-0.5B-Instruct_qlora_s2048_b1_r16_steps3.json  —  OK

**Configuration**


| Field | Value |
| --- | --- |
| model | `Qwen/Qwen2.5-0.5B-Instruct` |
| method | qlora |
| seq_len | 2048 |
| micro_batch_size | 1 |
| steps | 3 |
| lora_rank | 16 |
| quantization | nf4_double_quant |
| optimizer | paged_adamw_8bit |
| gradient_checkpointing | True |
| attention | sdpa |

**Environment**


| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 (15.99 GB) |
| torch | 2.6.0+cu124 |
| CUDA | 12.4 |
| bf16 | True |

**Memory: estimated vs measured**


| Metric | Value |
| --- | --- |
| estimated total | 7.21 GB |
| measured peak (reserved) | 6.28 GB |
| measured peak (allocated) | 5.52 GB |
| final allocated | 1.88 GB |
| tokens/sec | 3445.37 |
| avg step time | 0.5944 s |
| last-step loss | 12.9314 |

## Qwen__Qwen2.5-1.5B-Instruct_qlora_s1024_b1_r16_steps3.json  —  OK

**Configuration**


| Field | Value |
| --- | --- |
| model | `Qwen/Qwen2.5-1.5B-Instruct` |
| method | qlora |
| seq_len | 1024 |
| micro_batch_size | 1 |
| steps | 3 |
| lora_rank | 16 |
| quantization | nf4_double_quant |
| optimizer | paged_adamw_8bit |
| gradient_checkpointing | True |
| attention | sdpa |

**Environment**


| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 (15.99 GB) |
| torch | 2.6.0+cu124 |
| CUDA | 12.4 |
| bf16 | True |

**Memory: estimated vs measured**


| Metric | Value |
| --- | --- |
| estimated total | 6.05 GB |
| measured peak (reserved) | 4.36 GB |
| measured peak (allocated) | 4.04 GB |
| final allocated | 2.14 GB |
| tokens/sec | 2482.60 |
| avg step time | 0.4125 s |
| last-step loss | 13.0873 |

## Qwen__Qwen2.5-1.5B-Instruct_qlora_s1024_b1_r16_steps3_nockpt.json  —  OK

**Configuration**


| Field | Value |
| --- | --- |
| model | `Qwen/Qwen2.5-1.5B-Instruct` |
| method | qlora |
| seq_len | 1024 |
| micro_batch_size | 1 |
| steps | 3 |
| lora_rank | 16 |
| quantization | nf4_double_quant |
| optimizer | paged_adamw_8bit |
| gradient_checkpointing | False |
| attention | sdpa |

**Environment**


| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 (15.99 GB) |
| torch | 2.6.0+cu124 |
| CUDA | 12.4 |
| bf16 | True |

**Memory: estimated vs measured**


| Metric | Value |
| --- | --- |
| estimated total | 10.75 GB |
| measured peak (reserved) | 9.55 GB |
| measured peak (allocated) | 9.34 GB |
| final allocated | 2.18 GB |
| tokens/sec | 3002.86 |
| avg step time | 0.3410 s |
| last-step loss | 13.0907 |

## Qwen__Qwen2.5-1.5B-Instruct_qlora_s1024_b2_r16_steps2.json  —  OK

**Configuration**


| Field | Value |
| --- | --- |
| model | `Qwen/Qwen2.5-1.5B-Instruct` |
| method | qlora |
| seq_len | 1024 |
| micro_batch_size | 2 |
| steps | 2 |
| lora_rank | 16 |
| quantization | nf4_double_quant |
| optimizer | paged_adamw_8bit |
| gradient_checkpointing | True |
| attention | sdpa |

**Environment**


| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 (15.99 GB) |
| torch | 2.6.0+cu124 |
| CUDA | 12.4 |
| bf16 | True |

**Memory: estimated vs measured**


| Metric | Value |
| --- | --- |
| estimated total | 8.42 GB |
| measured peak (reserved) | 6.88 GB |
| measured peak (allocated) | 6.53 GB |
| final allocated | 2.73 GB |
| tokens/sec | 2661.52 |
| avg step time | 0.7695 s |
| last-step loss | 13.3183 |

## Qwen__Qwen2.5-1.5B-Instruct_qlora_s2048_b1_r16_steps3.json  —  OK

**Configuration**


| Field | Value |
| --- | --- |
| model | `Qwen/Qwen2.5-1.5B-Instruct` |
| method | qlora |
| seq_len | 2048 |
| micro_batch_size | 1 |
| steps | 3 |
| lora_rank | 16 |
| quantization | nf4_double_quant |
| optimizer | paged_adamw_8bit |
| gradient_checkpointing | True |
| attention | sdpa |

**Environment**


| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 (15.99 GB) |
| torch | 2.6.0+cu124 |
| CUDA | 12.4 |
| bf16 | True |

**Memory: estimated vs measured**


| Metric | Value |
| --- | --- |
| estimated total | 8.42 GB |
| measured peak (reserved) | 7.10 GB |
| measured peak (allocated) | 6.53 GB |
| final allocated | 2.74 GB |
| tokens/sec | 2326.95 |
| avg step time | 0.8801 s |
| last-step loss | 13.0152 |

## Qwen__Qwen2.5-1.5B-Instruct_qlora_s4096_b1_r16_steps2.json  —  OK

**Configuration**


| Field | Value |
| --- | --- |
| model | `Qwen/Qwen2.5-1.5B-Instruct` |
| method | qlora |
| seq_len | 4096 |
| micro_batch_size | 1 |
| steps | 2 |
| lora_rank | 16 |
| quantization | nf4_double_quant |
| optimizer | paged_adamw_8bit |
| gradient_checkpointing | True |
| attention | sdpa |

**Environment**


| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 (15.99 GB) |
| torch | 2.6.0+cu124 |
| CUDA | 12.4 |
| bf16 | True |

**Memory: estimated vs measured**


| Metric | Value |
| --- | --- |
| estimated total | 13.16 GB |
| measured peak (reserved) | 13.56 GB |
| measured peak (allocated) | 11.51 GB |
| final allocated | 3.92 GB |
| tokens/sec | 1661.93 |
| avg step time | 2.4646 s |
| last-step loss | 13.1373 |

## Qwen__Qwen2.5-1.5B-Instruct_qlora_s512_b1_r16_steps3.json  —  OK

**Configuration**


| Field | Value |
| --- | --- |
| model | `Qwen/Qwen2.5-1.5B-Instruct` |
| method | qlora |
| seq_len | 512 |
| micro_batch_size | 1 |
| steps | 3 |
| lora_rank | 16 |
| quantization | nf4_double_quant |
| optimizer | paged_adamw_8bit |
| gradient_checkpointing | True |
| attention | sdpa |

**Environment**


| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 (15.99 GB) |
| torch | 2.6.0+cu124 |
| CUDA | 12.4 |
| bf16 | True |

**Memory: estimated vs measured**


| Metric | Value |
| --- | --- |
| estimated total | 4.86 GB |
| measured peak (reserved) | 2.91 GB |
| measured peak (allocated) | 2.79 GB |
| final allocated | 1.85 GB |
| tokens/sec | 1497.69 |
| avg step time | 0.3419 s |
| last-step loss | 13.2013 |

## Qwen__Qwen2.5-3B-Instruct_qlora_s1024_b1_r16_steps2.json  —  OK

**Configuration**


| Field | Value |
| --- | --- |
| model | `Qwen/Qwen2.5-3B-Instruct` |
| method | qlora |
| seq_len | 1024 |
| micro_batch_size | 1 |
| steps | 2 |
| lora_rank | 16 |
| quantization | nf4_double_quant |
| optimizer | paged_adamw_8bit |
| gradient_checkpointing | True |
| attention | sdpa |

**Environment**


| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 (15.99 GB) |
| torch | 2.6.0+cu124 |
| CUDA | 12.4 |
| bf16 | True |

**Memory: estimated vs measured**


| Metric | Value |
| --- | --- |
| estimated total | 7.26 GB |
| measured peak (reserved) | 5.54 GB |
| measured peak (allocated) | 5.17 GB |
| final allocated | 3.16 GB |
| tokens/sec | 1302.90 |
| avg step time | 0.7859 s |
| last-step loss | 13.5585 |

## Qwen__Qwen2.5-7B-Instruct_qlora_s1024_b1_r16_steps2.json  —  OK

**Configuration**


| Field | Value |
| --- | --- |
| model | `Qwen/Qwen2.5-7B-Instruct` |
| method | qlora |
| seq_len | 1024 |
| micro_batch_size | 1 |
| steps | 2 |
| lora_rank | 16 |
| quantization | nf4_double_quant |
| optimizer | paged_adamw_8bit |
| gradient_checkpointing | True |
| attention | sdpa |

**Environment**


| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 (15.99 GB) |
| torch | 2.6.0+cu124 |
| CUDA | 12.4 |
| bf16 | True |

**Memory: estimated vs measured**


| Metric | Value |
| --- | --- |
| estimated total | 12.43 GB |
| measured peak (reserved) | 11.23 GB |
| measured peak (allocated) | 10.00 GB |
| final allocated | 7.89 GB |
| tokens/sec | 922.90 |
| avg step time | 1.1095 s |
| last-step loss | 13.6009 |

## sshleifer__tiny-gpt2_lora_s128_b1_r8_steps3.json  —  OK

**Configuration**


| Field | Value |
| --- | --- |
| model | `sshleifer/tiny-gpt2` |
| method | lora |
| seq_len | 128 |
| micro_batch_size | 1 |
| steps | 3 |
| lora_rank | 8 |
| quantization | - |
| optimizer | paged_adamw_8bit |
| gradient_checkpointing | True |
| attention | sdpa |

**Environment**


| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 (15.99 GB) |
| torch | 2.6.0+cu124 |
| CUDA | 12.4 |
| bf16 | True |

**Memory: estimated vs measured**


| Metric | Value |
| --- | --- |
| estimated total | 2.16 GB |
| measured peak (reserved) | 0.14 GB |
| measured peak (allocated) | 0.10 GB |
| final allocated | 0.03 GB |
| tokens/sec | 1470.37 |
| avg step time | 0.0871 s |
| last-step loss | 10.8219 |
