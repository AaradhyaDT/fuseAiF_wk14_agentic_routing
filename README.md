# fuseAiF_wk14_agentic_routing

ShopAssist AI — Agentic Routing: Fine-Tuning Transformers for Intent Classification

Two candidate routing strategies for classifying customer support messages into 11 categories (ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, SUBSCRIPTION, ORDER, PAYMENT, REFUND, SHIPPING), benchmarked on the [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (26,872 examples, 80/10/10 stratified split, seed=0).

## Files

- `support_routing.ipynb` — all code: data prep, both approaches, comparison, recommendation, reflection
- `W14_ Assignment.md` — full assignment brief
- `LICENSE`

## Notebook structure

0. Setup
1. Data — load, dedup check, group-aware stratified split (prevents instruction-text leakage across splits), split distribution check
2. Approach 1 — Encoder-only fine-tuning (BERT, DistilBERT, RoBERTa, ModernBERT), AdamW + linear warmup, [CLS]-token classification head
3. Approach 2 — Decoder-only SLM fine-tuning via QLoRA (Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct), category name as generation target
4. Comparison — metrics table + bar chart across both approaches
5. Recommendation — generated from live test-set numbers (not hardcoded), addressed to engineering lead
6. Reflection — pretraining transfer, chit-chat handling

All four metrics (precision, recall, macro-F1, accuracy) plus GPU memory and inference latency are measured for every model.

## Status

**Approach 1 (encoders): complete.** All four candidates trained and evaluated on the held-out test set.

| Model | Macro-F1 | Accuracy | Peak Mem | Latency |
|---|---|---|---|---|
| distilbert-base-uncased | 1.0000 | 1.0000 | 580MB | 4.31ms |
| roberta-base | 1.0000 | 1.0000 | 1045MB | 12.18ms |
| bert-base-uncased | 0.9991 | 0.9993 | 919MB | 8.61ms |
| answerdotai/ModernBERT-base | 0.9997 | 0.9996 | 1269MB | 16.97ms |

Best by test macro-F1: **distilbert-base-uncased**.

**Approach 2 (SLMs): complete.** `train_slm`'s best-checkpoint reload previously crashed on `model.load_state_dict()` — the `"lora" in k` substring filter used to extract adapter weights also pulled in bitsandbytes' 4-bit quant buffers (`.absmax`, `.quant_map`, `.nested_absmax`, `.nested_quant_map`, `.quant_state.bitsandbytes__nf4`), which the strict reload then rejected as unexpected keys. Fixed by replacing the substring-filtered dict with PEFT's `get_peft_model_state_dict`/`set_peft_model_state_dict`, which extract adapter tensors by PEFT's own bookkeeping instead of key-name matching.

| Model | Macro-F1 | Accuracy | Peak Mem | Latency | Unparseable |
|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.0000 | 1.0000 | 1281MB | 234.65ms | 0.0% |
| Qwen2.5-0.5B-Instruct | 0.9989 | 0.9993 | 525MB | 172.47ms | 0.0% |

Best overall by test macro-F1: **distilbert-base-uncased** (encoder) — ties Qwen2.5-1.5B-Instruct at 1.0000 macro-F1 while running ~54x faster (4.31ms vs 234.65ms) and using ~2.2x less peak memory (580MB vs 1281MB). Sections 4 (comparison), 5 (recommendation), and 6 (reflection) have all executed against these results.

**Known gaps, not yet resolved:**
- The valid-preds-only classification report for the best SLM (final cell of §3.2) is unexecuted — it needs the raw `preds`/`labels` arrays from the original training run, which only existed in that Colab kernel's memory and weren't persisted to `slm_results.json`. `split/support_routing_partB_slms_v2.ipynb` now saves these going forward, so a fresh run won't hit this.
- `slm_results.json`'s `precision`/`recall`/`p50_ms`/`p95_ms` fields are `null` for the same reason (never printed, not recoverable after the fact).
- **Submission requirement not fully verified:** the notebook must run top-to-bottom without errors on a fresh kernel. The current committed notebook reflects merged Colab split-run history rather than one continuous fresh execution, so this hasn't been re-confirmed end-to-end.

## Setup

```
pip install -q transformers datasets peft accelerate evaluate scikit-learn matplotlib seaborn bitsandbytes huggingface_hub -U
```

GPU required for both approaches; SLM training additionally requires QLoRA (4-bit, via `bitsandbytes`) to fit on a free-tier T4 (~15GB). `.ipynb` is not meant to be executed via CI/local runner — run in Google Colab.
