# Fine-tuning an LLM for a Friendly Conversational Style

This was my assignment for an ML Engineering internship application: take a language
model through the full fine-tuning cycle — data, training, evaluation — and actually
understand every step, not just run someone else's script.

I'd never done this before, so this repo is also honestly a record of me learning LoRA
fine-tuning from scratch on a free Google Colab GPU, hitting real errors (bitsandbytes
version conflicts, GPU quota running out mid-generation), and working through them.

**Model:** `mistralai/Mistral-7B-Instruct-v0.3` · **Method:** LoRA · **Environment:** Google Colab (T4 GPU, free tier)

## What's in this repo

```
.
├── README.md                          # this file
├── dataset.jsonl                      # final dataset (Step 1)
├── generate_dataset.py                # dataset generation script
├── lora-adapter-friendly-style.zip    # trained LoRA adapter (Step 2)
├── loss_curve.png                     # training loss chart (Step 2)
└── (Colab notebooks for steps 1–3)
```

## Task checklist

| Step | Requirement | Status |
|---|---|---|
| 1. Data | ≥200 examples, JSONL, dedup, quality filtering | ✅ 350 examples |
| 2. Fine-tuning | LoRA/QLoRA, PEFT+Transformers, loss curve, adapter-only save | ✅ |
| 3. Evaluation | 10–20 examples, at least one metric, conclusions | ✅ 11 examples, ROUGE-L |

---

## Step 1 — Data

### Goal
Build an instruction-response dataset (minimum 200 examples) to later fine-tune a
model toward a warm, friendly, conversational tone.

### Model and dataset
- **Model used to synthesize data:** `mistralai/Mistral-7B-Instruct-v0.3`, loaded in
  4-bit (bitsandbytes/NF4) so it fits on a free Colab T4 GPU (16GB VRAM).
- **Seed dataset:** `tatsu-lab/alpaca` from HuggingFace (52,002 examples). I filtered
  down to entries with an empty `input` field (self-contained instructions), then took
  a sample of 350.
- **Language:** English — a deliberate choice. The task didn't actually require Russian
  specifically (I'd assumed it did at first), and English keeps the pipeline simpler
  and plays to Mistral's strengths, since it's trained mostly on English text.

### How the data was generated
I ran each instruction through Mistral-7B-Instruct with a system prompt asking it to
answer in a specific style:

> "You are a warm, friendly conversational assistant. Answer casually and kindly,
> like you're talking to a good friend. Keep it natural, avoid overly formal or
> robotic language."

So each pair is `(instruction, response)`, where the response is a *new* answer from
the model in the target style — not the original, drier alpaca answer. This matches
what the assignment asked for: "using a publicly available model... collect or
synthesize a dataset."

Every example was flushed to disk right after generation, so a dropped Colab session
wouldn't wipe out hours of progress (which, spoiler, mattered more than once).

### Cleaning and filtering
1. **Deduplication** — MD5 hash of the normalized instruction text. Since the 350
   source instructions were already unique in alpaca, nothing was actually duplicated
   (350 → 350).
2. **Quality filter** — response length between 20 and 2000 characters, no obvious
   refusal/out-of-character phrases ("as an AI", "as a digital assistant", etc.), and
   a basic repetition check. All 350 examples passed (350 → 350) — a good sign the
   system prompt was well-tuned.

### Result
350 instruction-response pairs in `dataset.jsonl` — comfortably above the 200 minimum.

---

## Step 2 — Fine-tuning (LoRA)

### Goal
Fine-tune Mistral-7B-Instruct with LoRA on `dataset.jsonl` so the model adopts the
friendly style on its own, with no prompt needed.

### Environment
Google Colab (T4 GPU), `transformers==4.44.2`, `accelerate==0.33.0`, `peft==0.11.1`,
`bitsandbytes==0.43.1`. Model loaded in 4-bit (NF4).

One real hiccup worth mentioning: `bitsandbytes` threw an internal kernel-registration
error partway through (`RuntimeError: already a kernel registered...`). A plain
"Restart runtime" didn't fix it — I had to fully disconnect and delete the runtime,
then reinstall pinned, known-compatible versions. Worth knowing if you hit the same wall.

### Data formatting
Each pair was converted into Mistral's chat format (`<s>[INST] instruction [/INST]
response</s>`) via `apply_chat_template`, then tokenized with `max_length=512`.
`labels` are created by the data collator (`DataCollatorForLanguageModeling`,
`mlm=False`) *after* batch padding — not copied manually beforehand, which was my
first attempt and caused a tensor-length mismatch error.

### LoRA configuration

| Parameter | Value |
|---|---|
| r (rank) | 16 |
| lora_alpha | 32 |
| target_modules | q_proj, k_proj, v_proj, o_proj |
| lora_dropout | 0.05 |
| task_type | CAUSAL_LM |

The model was prepped with `prepare_model_for_kbit_training` before wrapping it in
LoRA.

**Trainable parameters: 13,631,488 out of 7,261,655,040 — just 0.19%.** Seeing that
number was honestly the moment LoRA "clicked" for me — you're really only touching a
sliver of the model.

### Training setup

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Batch size / grad accumulation | 4 / 4 (effective batch 16) |
| Learning rate | 2e-4 |
| Precision | fp16 |

### Results
- Training time: ~16 minutes on T4, 66 steps
- **Loss: 1.12 → 0.42** (see `loss_curve.png`) — a steady decline with no signs of
  divergence or overfitting

### Saved artifact
Only the LoRA adapter was saved (~52 MB), not the full model (~14 GB) — as required.

---

## Step 3 — Evaluation

### Goal
Compare the base model against the fine-tuned one on unseen examples: did the style
actually transfer, and did the model keep answering correctly along the way?

### Test set
15 instructions from alpaca (indices 350–365) that were never used in training. Colab's
free GPU quota ran out mid-generation, so the final comparison uses **11 complete
pairs** instead of 15 — still within the assignment's 10–20 range.

### Method
- **Base model:** the same checkpoint with `model.disable_adapter()`, no style prompt.
- **Fine-tuned model:** the same checkpoint with the LoRA adapter active.
- Generation with `do_sample=False` for a reproducible, apples-to-apples comparison.
- **Metric: ROUGE-L** against the original alpaca answer — this checks that content
  was preserved, not style.

### Results

| Model | Average ROUGE-L |
|---|---|
| Base (no adapter) | 0.2273 |
| Fine-tuned (LoRA) | 0.2003 |

### Qualitative comparison
The base model answers formally and drops straight into a numbered list, no greeting,
no warmth. The fine-tuned model consistently opens with something friendly ("Hey
there!", "Absolutely!"), leans on casual phrasing and metaphors (comparing a project
manager to an orchestra conductor, SEO to a lighthouse), while keeping the same
underlying structure and facts.

Example (prompt: "Describe the role of a project manager"):
- Base: *"A Project Manager (PM) plays a crucial role in ensuring the successful
  completion of a project..."*
- Fine-tuned: *"Hey there! So, a project manager is like the conductor of an
  orchestra, but instead of music, they're orchestrating a team..."*

### Conclusions
- **It got better**, in the sense that actually mattered for this task: the model
  consistently picked up the friendly tone with zero prompting — the fine-tuning did
  what it was supposed to do.
- The small ROUGE-L drop (0.227 → 0.200) is an expected, acceptable trade-off — some
  of the response now goes toward greetings and stylistic flourishes instead of
  mirroring the original wording exactly. The drop is modest, and the underlying
  content and structure are clearly still there.
- Honest limitations: only 11 examples (due to the session dropping), and a single
  metric that measures lexical overlap, not meaning. A stronger evaluation would add
  BERTScore for semantic similarity and/or an LLM-as-a-judge score specifically for
  "how friendly does this sound," rather than relying on content-preservation alone.

---

## Reproducing this end to end
1. **Data:** Colab + T4 GPU → log into HuggingFace → load `tatsu-lab/alpaca` → load
   Mistral-7B-Instruct in 4-bit → generate responses with the style system prompt →
   dedupe + filter → save `dataset.jsonl`.
2. **Training:** load `dataset.jsonl` → format to chat template + tokenize → configure
   `LoraConfig` → train with `Trainer` → save the adapter.
3. **Evaluation:** generate answers from both the base and fine-tuned model on new
   prompts → compute ROUGE-L → compare qualitatively → write up conclusions.

## What I'd do differently with more time
- Save intermediate results to Google Drive from the start, not local Colab storage —
  would have avoided losing progress twice to session drops.
- Add a second metric (BERTScore) to cross-check ROUGE-L, since lexical overlap alone
  is a pretty blunt way to measure "did the meaning survive."
- Run the full 15-example eval set instead of 11, GPU quota permitting.
