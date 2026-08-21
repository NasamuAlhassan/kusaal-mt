# Kusaal ↔ English Machine Translation

**An open-source machine translation model for Kusaal** — a Gur language spoken by ~400,000 people in northern Ghana and parts of Burkina Faso.

Built by a native Kusaal speaker from Bawku, Ghana. Fine-tuned on a parallel corpus assembled from scratch, expanded through back-translation augmentation, and — as of August 2026 — a sentence-aligned corpus mined from Kusaal Wikipedia.

[**Model on Hugging Face**](https://huggingface.co/PrinceAlhassanNasamu/kusaal-nllb-600M) | [**Live demo**](https://huggingface.co/spaces/PrinceAlhassanNasamu/kusaal-mt) | [**Public benchmark**](https://huggingface.co/datasets/PrinceAlhassanNasamu/kusaal-wikipedia-benchmark)

**In this repo:** the [13,659-pair Kusaal Wikipedia parallel corpus](corpus/) (CC BY-SA 4.0), the [mining pipeline scripts](scripts/), and the [full pipeline report](docs/WIKI_CORPUS_PIPELINE_REPORT.md).

**Licensing:** code and documentation in this repository are CC BY 4.0 (root LICENSE); the corpus/ directory is **CC BY-SA 4.0** (see corpus/LICENSE), inheriting the ShareAlike terms of its Wikipedia source text.

---

## August 2026 update: Wikipedia fine-tune

The model was continued-trained on **13,659 new human-parallel sentence pairs**
mined from Kusaal Wikipedia and its English counterparts (1,170 comparable
article pairs, aligned by a shared-anchor + model-verification pipeline and
partially human-adjudicated), plus a 6,917-sentence back-translation pool
(real Kusaal, synthetic English, eng→kus direction only), mixed with the
original corpus as replay.

**Results** (beam 4, 1,000 sentences per test set per direction, sacreBLEU /
chrF++ word_order=2):

| Test set | Direction | BLEU before → after | chrF++ before → after |
|---|---|---|---|
| Wikipedia benchmark (new) | kus → eng | 42.17 → **47.73** | 59.92 → **64.09** |
| Wikipedia benchmark (new) | eng → kus | 19.43 → **32.26** | 43.58 → **54.09** |
| Original test set | kus → eng | 31.12 → 30.59 | 49.78 → 50.11 |
| Original test set | eng → kus | 19.96 → 20.21 | 42.32 → 42.96 |

The largest gain — **+12.8 BLEU for eng→kus on encyclopedic text** — comes
from the back-translation pool targeting the model's weaker direction.
Performance on the original (largely religious-register) test set is
unchanged: no catastrophic forgetting.

Qualitative fixes confirmed by a before/after probe set: Kusaal spelled-out
numerals now render as digits instead of KJV-style English ("one thousand
eight hundred seventy-four"), dates are exact, `Kristo biig` translates as
"a Christian" rather than the calque "son of Christ", and election
vocabulary no longer drifts into sports commentary.

The benchmark is **public and independently verifiable**:
[kusaal-wikipedia-benchmark](https://huggingface.co/datasets/PrinceAlhassanNasamu/kusaal-wikipedia-benchmark)
(1,000 frozen pairs, evaluation protocol and reference results included —
do not train on it).

**Evaluation notes (read before comparing numbers):**
- The original card reported BLEU 27.57 / 13.72 under a different decoding
  setup and the full 2,081-sentence test; the table above uses one uniform
  harness for before/after, so only within-table comparisons are valid.
- The Wikipedia benchmark was mined with MT assistance, which biases it
  toward sentences MT handles well; absolute scores on it read high. The
  before→after deltas are the meaningful signal.

**Remaining known weakness:** the Kusaal pronoun `o` is gender-neutral, and
the model still defaults to "he" even when context (e.g. `kul sid`, "married
a husband") indicates a female referent.

---

## Why This Exists

Kusaal is not in Google Translate. It is not in Meta's NLLB-200 (which covers 200 languages). It has no speech recognizer, no text-to-speech system, and very limited NLP resources.

A Kusaal speaker navigating a hospital form, a legal document, or an agricultural advisory in northern Ghana has no open digital translation tool. This project is a step toward changing that.

---

## Model

Fine-tuned from `facebook/nllb-200-distilled-600M`.

| | |
|---|---|
| **Base model** | facebook/nllb-200-distilled-600M |
| **New language** | `kus_Latn` (Kusaal, Latin script) |
| **Embedding seed** | `dag_Latn` (Dagbani — closest Gur relative in NLLB) |
| **Training data** | ~45,800 human-parallel + ~9,300 back-translation pairs¹ |
| **Directions** | kus → eng and eng → kus in one model |
| **Overall BLEU** | **39.2 (kus → eng) · 26.2 (eng → kus)** — average of the two benchmarks below |
| **BLEU (kus → eng)** | 47.73 wiki / 30.59 original domain |
| **BLEU (eng → kus)** | 32.26 wiki / 20.21 original domain |

### Why Dagbani as the seed?

Kusaal and Dagbani are both Oti-Volta (Gur) languages. Instead of initialising the new `kus_Latn` embedding randomly, we copy Dagbani's embedding as a warm-start. The model begins with a linguistically grounded prior about what kind of language Kusaal is, which leads to faster convergence and better early translation quality than a cold random start.

---

## Dataset

Built from multiple sources:

| Source | Pairs | Domain |
|---|---|---|
| YouVersion (Bible KJV) | ~29,257 | Religious / formal |
| **Kusaal Wikipedia harvest (Aug 2026)** | **13,659** | **Encyclopedic: politics, biography, geography, agriculture, science** |
| English-Kusaal Index | 3,504 | Index / vocabulary |
| GhanaNLP | 3,489 | Daily life, health, agriculture, greetings, numbers |
| Lexique Pro (Kusaal lexical database) | 2,775 | Dictionary |
| Wikipedia (original subset) | 1,136 | General |
| Back-translation augmentation | ~9,300 | Mixed |
| **Total (as collected)** | **~63,100** | |

¹ **One accounting, stated once:** source rows above are as-collected and
overlap; after deduplication and filtering, the six original sources yield
the 34,568-pair June 2026 corpus (~32,200 human-parallel + ~2,400
back-translation). Adding the Wikipedia harvest (13,659 pairs) and its
back-translation pool (6,917) gives **~45,800 human-parallel + ~9,300
back-translation pairs**; 1,500 wiki pairs (1,000 benchmark + 500
validation) are excluded from training, leaving ~53,600 pairs actually
trained on.

The Wikipedia harvest was aligned by a three-stage pipeline: anchor matching
(shared numbers, names, and loanwords), machine-translation verification
(embedding cosine + chrF++ against candidate sentences), and human/LLM
adjudication of borderline pairs. It is deduplicated against all prior
training data, and a frozen 1,000-pair benchmark plus 500-pair validation
split were held out before training.

### Data pipeline

The raw data had two problems that would have silently hurt training:

1. **HTML pollution** — the YouVersion scraper stored `class="verse v2" > 2 In those days...` instead of clean text in thousands of Bible pairs. Fixed by stripping the HTML pattern and re-sourcing clean text.

2. **CSV quoting corruption** — rows were silently dropped by pandas due to unescaped apostrophes in Bible text. Fixed by re-saving all files with `QUOTE_ALL`.

After cleaning: deduplication on Kusaal+English pairs, length ratio filter (max 6×), UTF-8 throughout, and apostrophe normalization (all variants folded to straight `'` U+0027 — Wikipedia text arrives with the saltillo `ꞌ` U+A78C, which the tokenizer has never seen).

---

## Usage

### Install

```bash
pip install transformers torch sentencepiece
```

### Quick inference

> ⚠️ **Do not use plain `tokenizer.src_lang = "kus_Latn"`.** Tested on
> transformers 5.x: it does not raise an error — it silently falls back to
> the `eng_Latn` prefix, because added language codes are not registered in
> the NLLB tokenizer's language list. Your Kusaal input gets encoded as if
> it were English and translations quietly degrade. Use the snippet below,
> which sets the language prefix explicitly.

```python
import json, re, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download

MODEL_ID = "PrinceAlhassanNasamu/kusaal-nllb-600M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model.eval()

# Re-register language codes (not saved by save_pretrained)
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
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

print(translate("Laafi bɛ?", "kus-eng"))       # -> "Are you all right?"
print(translate("How are you?", "eng-kus"))     # -> Kusaal output
```

---

## Model Files

| File | Description |
|---|---|
| `model.safetensors` | Trained model weights (~2.4 GB) |
| `tokenizer.json` | Tokenizer vocabulary with `kus_Latn` added |
| `kusaal_lang_codes.json` | Language → token ID map (required at inference) |
| `generation_config.json` | Default generation settings |
| `config.json` | Model architecture config |

> **Note on `kusaal_lang_codes.json`:** HuggingFace's `save_pretrained()` does not persist the `lang_code_to_id` dictionary. This file must be loaded at inference time to correctly set `forced_bos_token_id`.

---

## Limitations

- **Register coverage:** the majority of training data remains formal (Bible) text, now balanced by ~14K encyclopedic pairs. Everyday conversational Kusaal is still the weakest register.
- **Gender pronouns:** Kusaal `o` is gender-neutral; the model defaults to "he" and misses contextual cues for female referents.
- **Synthetic data noise:** back-translated pairs introduce noise — the model that generated them makes errors, and those errors appear in training. Filtering mitigates but does not eliminate this.
- **Tokenization:** NLLB's SentencePiece tokenizer was built without Kusaal data. Kusaal words are over-segmented into small fragments, limiting word-level pattern learning.
- **No formal native speaker evaluation:** BLEU/chrF++ measure n-gram overlap with reference translations. The only real quality test is assessment by fluent Kusaal speakers.

---

## What's Next

This model is a step toward a larger planned project: a **trimodal Kusaal corpus** — parallel text, aligned Kusaal audio recordings, and structured linguistic annotations. The goal is to enable speech recognition, text-to-speech, and language learning tools for Kusaal communities.

If you work in African NLP, speak Kusaal, or want to contribute data — open an issue or reach out.

---

## Technical Details

| Parameter | Value (Aug 2026 fine-tune) |
|---|---|
| Steps | 4,000 (continued from the June 2026 checkpoint) |
| Effective batch size | 32 (4 × 8 grad accumulation) |
| Learning rate | 3e-5, linear decay, 250 warmup steps |
| Max sequence length | 192 tokens |
| Optimizer | Adafactor |
| Checkpoint selection | validation loss on held-out wiki pairs |
| Precision | FP16 |
| Hardware | Kaggle T4 GPU (single) |

---

## Citation

```bibtex
@misc{alhassan2026kusaal,
  author    = {Alhassan, Prince Nasamu},
  title     = {Kusaal-English Machine Translation: An Open Model and Corpus},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/PrinceAlhassanNasamu/kusaal-nllb-600M}
}
```

---

## About

Built by **Prince Nasamu Alhassan** — native Kusaal speaker from Bawku, Ghana. Mathematical Science with Computer Science student at the University of Ghana, Legon. Independent NLP researcher affiliated with [GhanaNLP](https://ghananlp.org).

- GitHub: [@NasamuAlhassan](https://github.com/NasamuAlhassan)
- HuggingFace: [PrinceAlhassanNasamu](https://huggingface.co/PrinceAlhassanNasamu)
- Email: pnalhassan@gmail.com
