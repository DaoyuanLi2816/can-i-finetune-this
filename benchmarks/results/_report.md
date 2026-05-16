# canifinetune benchmark report

Each section below corresponds to one benchmark result JSON. `estimated` is what the static estimator predicted, `measured` is what was observed on this machine.

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
| estimated total | 2.41 GB |
| measured peak (reserved) | 3.30 GB |
| measured peak (allocated) | 3.12 GB |
| final allocated | 1.30 GB |
| tokens/sec | 1995.33 |
| avg step time | 0.5132 s |
| last-step loss | 13.0308 |

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
| estimated total | 3.00 GB |
| measured peak (reserved) | 4.36 GB |
| measured peak (allocated) | 4.04 GB |
| final allocated | 2.14 GB |
| tokens/sec | 1351.69 |
| avg step time | 0.7576 s |
| last-step loss | 13.0922 |

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
| estimated total | 3.16 GB |
| measured peak (reserved) | 7.10 GB |
| measured peak (allocated) | 6.53 GB |
| final allocated | 2.74 GB |
| tokens/sec | 1470.40 |
| avg step time | 1.3928 s |
| last-step loss | 13.0132 |

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
| estimated total | 3.86 GB |
| measured peak (reserved) | 5.54 GB |
| measured peak (allocated) | 5.17 GB |
| final allocated | 3.16 GB |
| tokens/sec | 1158.51 |
| avg step time | 0.8839 s |
| last-step loss | 13.5622 |

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
| quantization | bf16 |
| optimizer | adamw_torch |
| gradient_checkpointing | False |
| attention | eager |

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
| estimated total | 2.08 GB |
| measured peak (reserved) | 0.12 GB |
| measured peak (allocated) | 0.11 GB |
| final allocated | 0.04 GB |
| tokens/sec | 1735.45 |
| avg step time | 0.0738 s |
| last-step loss | 10.8217 |
