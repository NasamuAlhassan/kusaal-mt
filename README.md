# Kusaal ↔ English Machine Translation

**The first machine translation model for Kusaal** — a Gur language spoken by ~400,000 people in northern Ghana and parts of Burkina Faso, with no prior NLP tools.

Built by a native Kusaal speaker from Bawku, Ghana. Fine-tuned on a parallel corpus assembled from scratch.

🤗 **Model on HuggingFace:** [PrinceAlhassanNasamu/kusaal-nllb-600M](https://huggingface.co/PrinceAlhassanNasamu/kusaal-nllb-600M)

---

## Why This Exists

Kusaal is not in Google Translate. It is not in Meta's NLLB-200 (which covers 200 languages). It has no speech recognizer, no text-to-speech system, no existing NLP dataset of any kind.

A Kusaal speaker navigating a hospital form, a legal document, or an agricultural advisory in northern Ghana has no digital translation tool. This project is the starting point.

---

## Model

Fine-tuned from `facebook/nllb-200-distilled-600M` on 20,212 Kusaal↔English sentence pairs.

| | |
|---|---|
| **Base model** | facebook/nllb-200-distilled-600M |
| **New language** | `kus_Latn` (Kusaal, Latin script) |
| **Embedding seed** | `dag_Latn` (Dagbani — closest Gur relative in NLLB) |
| **Training pairs** | 20,212 (bidirectional: 40,424 examples) |
| **Directions** | kus → eng and eng → kus in one model |
| **HuggingFace** | [PrinceAlhassanNasamu/kusaal-nllb-600M](https://huggingface.co/PrinceAlhassanNasamu/kusaal-nllb-600M) |

### Why Dagbani as the seed?

Kusaal and Dagbani are both Oti-Volta (Gur) languages. Instead of initialising the new `kus_Latn` embedding randomly, we copy Dagbani's embedding as a warm-start. The model begins with a linguistically grounded prior about what kind of language Kusaal is, which leads to faster convergence and better early translation quality than a cold random start.

---

## Dataset

Built from three sources:

| Source | Pairs | Domain |
|---|---|---|
| YouVersion (Bible KJV) | 15,615 | Religious / formal |
| GhanaNLP | 4,594 | Daily life, health, agriculture, greetings, numbers |
| Wikipedia (translated) | 3 | General |

**Splits:** Train 15,967 · Val 3,031 · Test 1,214

### Data pipeline

The raw data had two problems that would have silently hurt training:

1. **HTML pollution** — the YouVersion scraper stored `class="verse v2" > 2 In those days...` instead of clean text in 19,254 of 22,498 Bible pairs. Fixed by stripping the HTML pattern and re-sourcing clean English from the KJV JSON.

2. **CSV quoting corruption** — ~6,000 rows were silently dropped by pandas due to unescaped apostrophes in Bible text. Fixed by re-saving all files with `QUOTE_ALL`.

After cleaning: deduplication on Kusaal text, length ratio filter (max 6×), UTF-8 without BOM throughout.

---

## Usage

### Install

```bash
pip install transformers torch sentencepiece
```

### Quick inference

```python
import json, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download

MODEL_ID = "PrinceAlhassanNasamu/kusaal-nllb-600M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model.eval()

# Re-register language codes
# (HuggingFace save_pretrained does not persist lang_code_to_id — we save it separately)
lang_codes_path = hf_hub_download(repo_id=MODEL_ID, filename="kusaal_lang_codes.json")
with open(lang_codes_path) as f:
    lang_codes = json.load(f)

for lang, lid in lang_codes.items():
    if hasattr(tokenizer, "lang_code_to_id"): tokenizer.lang_code_to_id[lang] = lid
    if hasattr(tokenizer, "id_to_lang_code"): tokenizer.id_to_lang_code[lid] = lang

EOS = tokenizer.eos_token_id

def translate(text, direction="kus-eng"):
    src = "kus_Latn" if direction == "kus-eng" else "eng_Latn"
    tgt = "eng_Latn" if direction == "kus-eng" else "kus_Latn"
    src_id = tokenizer.convert_tokens_to_ids(src)
    tgt_id = tokenizer.convert_tokens_to_ids(tgt)

    raw = tokenizer(text.strip(), add_special_tokens=False, return_tensors="pt")
    ids = torch.cat([torch.tensor([[src_id]]), raw["input_ids"], torch.tensor([[EOS]])], dim=1)

    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            forced_bos_token_id=tgt_id,
            decoder_start_token_id=EOS,
            max_new_tokens=128,
            num_beams=4,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

print(translate("Laafi bɛ?", "kus-eng"))       # -> "Are you all right?"
print(translate("How are you?", "eng-kus"))     # -> Kusaal output
```

### Local interactive script

Clone the repo and run:

```bash
git clone https://github.com/NasamuAlhassan/kusaal-mt.git
cd kusaal-mt
python test_kusaal.py --model_dir path/to/kusaal-nllb-final
```

---

## Model Files

| File | Description |
|---|---|
| `model.safetensors` | Trained model weights (~2.3 GB) |
| `tokenizer.json` | Tokenizer vocabulary with `kus_Latn` added |
| `kusaal_lang_codes.json` | Language → token ID map (not persisted by HuggingFace standard) |
| `generation_config.json` | Default generation settings (num_beams=4, max_new_tokens=256) |
| `config.json` | Model architecture config |

> **Note on `kusaal_lang_codes.json`:** NLLB uses integer token IDs to identify languages at inference time via `forced_bos_token_id`. HuggingFace's `save_pretrained()` saves the token string in `added_tokens.json` but does not persist the `lang_code_to_id` dictionary. This file ensures any loader can correctly configure the model without guessing.

---

## Limitations

- **Domain bias:** 77% of training data is Bible text. The model handles religious and formal Kusaal better than modern conversational language.
- **Tokenization:** NLLB's SentencePiece tokenizer was built without Kusaal data. Kusaal words are over-segmented into small fragments, limiting word-level pattern learning.
- **Dataset size:** 20,212 pairs is small for MT. High-resource language pairs use hundreds of millions. Generalisation to unseen vocabulary is limited.
- **No native speaker evaluation yet:** BLEU measures n-gram overlap with reference translations. The only real quality test is assessment by fluent Kusaal speakers.

---

## What's Next

This translation model is step one of a larger planned project: a **trimodal Kusaal corpus** — parallel text in Kusaal and English, aligned Kusaal audio recordings, and structured linguistic annotations. The goal is to enable not just translation but speech recognition, text-to-speech, and language learning tools for Kusaal.

If you work in African NLP, speak Kusaal, or want to contribute data — open an issue or reach out.

---

## Training Details

| Parameter | Value |
|---|---|
| Max steps | 5,000 |
| Effective batch size | 32 (4 × 8 grad accumulation) |
| Learning rate | 5e-5 |
| Warmup steps | 500 (10%) |
| Max sequence length | 128 tokens |
| Evaluation metric | SacreBLEU |
| Best checkpoint selection | BLEU (not loss) |
| Precision | FP16 |
| Hardware | Google Colab T4 GPU |

---

## Citation

```bibtex
@misc{alhassan2026kusaal,
  author    = {Alhassan, Prince Nasamu},
  title     = {Kusaal-English Machine Translation: First NLP Model for Kusaal},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/NasamuAlhassan/kusaal-mt}
}
```

---

## About

Built by **Prince Nasamu Alhassan** — native Kusaal speaker from Bawku, Ghana. Mathematical Science with Computer Science student at the University of Ghana, Legon.

- GitHub: [@NasamuAlhassan](https://github.com/NasamuAlhassan)
- HuggingFace: [PrinceAlhassanNasamu](https://huggingface.co/PrinceAlhassanNasamu)
