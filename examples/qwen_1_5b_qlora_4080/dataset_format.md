# Dataset format and supervision

Each non-blank JSONL line is exactly one of:

```jsonl
{"instruction":"Name a color.","input":"","output":"Blue."}
{"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Name a color."},{"role":"assistant","content":"Blue."},{"role":"user","content":"Another?"},{"role":"assistant","content":"Green."}]}
```

Instruction data uses `### Instruction`, optional `### Input`, and `### Response`
headers plus one EOS. Tokenize once with no automatic special-token insertion.
Completion boundaries use fast-tokenizer offsets in the complete text. Tokens
crossing the prompt/answer boundary are masked for assistant-only loss.

Chat data uses the model's native template with no generation prompt and no
additional BOS/EOS. Preserve every turn and initial system message. Require
non-empty text, alternating user/assistant and a final assistant answer. Tool,
multimodal and other role sequences are rejected. Do not mix messages and text
or instruction fields. A model without a chat template requires instruction data.

`loss_mode: all` (default) supervises every non-padding token after the causal
shift. `assistant` masks prompts; chat needs native `{% generation %}` spans and
working token masks. Unsupported templates fail instead of guessing boundaries.
For instruction records, only response-offset tokens and EOS are supervised.

`truncation: error` (default) rejects rows over seq_len=1024. Opt-in
`right` truncation must leave a response token after labels shift. Chat right
truncation also requires generation masks. A long prompt that loses the answer
fails even in all-token mode. Fixed right padding uses attention position, never
an EOS-ID comparison, so real EOS labels survive when PAD equals EOS.

Malformed/empty-answer/no-effective-supervision records fail with their line
number. `run.json` records row, blank-line, truncated-row and shifted-supervised
counts. No implicit row dropping or data uploads. The bundled sample is synthetic
and demonstrates execution only, not useful fine-tuning quality.
