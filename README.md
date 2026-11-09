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
  setup. A November 2026 reconstruction of the June training kernel found
  those two numbers were computed on 200 rows of the *validation* split (the
  same split used to choose the checkpoint), not on the test split. The table
  above uses one uniform harness for before/after, so only within-table
  comparisons are valid; do not compare the old and new numbers.
- "No catastrophic forgetting" means the original-test-set changes (−0.53 to
  +0.25 BLEU) are within typical sampling noise for 1,000 sentences; no
  significance test was run.
- The Wikipedia benchmark was mined with MT assistance, which biases it
  toward sentences MT handles well; absolute scores on it read high. The
  before→after deltas are the meaningful signal.

**Remaining known weakness:** the Kusaal pronoun `o` is gender-neutral, and
the model still defaults to "he" even when context (e.g. `kul sid`, "married
a husband") indicates a female referent.

---

## Why This Exists

Kusaal is not in Google Translate and not in Meta's NLLB-200 (which covers 200 languages). Open NLP resources for it remain very limited: speech recognition and text-to-speech exist only as early research models (see the [ASR repository](https://github.com/NasamuAlhassan/kusaal-asr) and the Hugging Face profile), and nearly all parallel text is scripture.

A Kusaal speaker navigating a hospital form, a legal document, or an agricultural advisory in northern Ghana has no open digital translation tool. This project is a step toward changing that.

---

## Model

Fine-tuned from `facebook/nllb-200-distilled-600M`.

| | |
|---|---|
| **Base model** | facebook/nllb-200-distilled-600M |
| **New language** | `kus_Latn` (Kusaal, Latin script) |
| **Embedding seed** | Reported as `dag_Latn` (Dagbani) — **unverified**, see note below |
| **Training data** | ~45,800 human-parallel + ~9,300 back-translation pairs¹ |
| **Directions** | kus → eng and eng → kus in one model |
| **Overall BLEU** | **39.2 (kus → eng) · 26.2 (eng → kus)** — average of the two benchmarks below |
| **BLEU (kus → eng)** | 47.73 wiki / 30.59 original domain |
| **BLEU (eng → kus)** | 32.26 wiki / 20.21 original domain |

### Why Dagbani as the seed? (intended design — not verified)

Kusaal and Dagbani are both Oti-Volta (Gur) languages, so the intended design was to warm-start the new `kus_Latn` embedding from Dagbani rather than randomly. **Caveat (November 2026 audit):** NLLB-200 has no `dag_Latn` token, the only surviving training notebook that shows the seeding step prints `initialised from dag_Latn (id=3)` (id 3 is `<unk>`), and the original June training code was not recovered. No ablation against a random-init or other seed was ever run. Treat "Dagbani-seeded" as a design intent, not an established fact, and do not cite it as a finding.

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

### Data overlap and leakage — read before training or comparing (audit of November 2026)

Measured with exact + near-duplicate matching (character 5-gram MinHash and
fuzzy ratio ≥ 90 on normalised text); counts are from the audit's leakage reports.

- **`corpus/kusaal_wiki_parallel_corpus.csv` contains the 1,000 benchmark pairs and the 500 validation pairs** (it is gold + silver *before* the split). Do not train on the whole file. The exact frozen split is 12,159 train / 500 validation / 1,000 test (seed 42, `scripts/stage4_make_splits.py`). Dropping rows whose Kusaal sentence matches the [public benchmark](https://huggingface.co/datasets/PrinceAlhassanNasamu/kusaal-wikipedia-benchmark) leaves 12,657 rows (two benchmark sentences occur twice), which still includes the 500 validation rows.
- **Near-duplicates of benchmark items exist in the training split.** 246–277 of the 12,159 training rows near-match a benchmark item; 114 of them share the exact *English* sentence with a benchmark pair, because the mining pipeline sometimes anchored one English sentence to two different Kusaal sentences. The August models therefore saw many benchmark English sentences. **Absolute benchmark scores are optimistic (especially eng→kus); deltas between systems trained on the same data remain usable.**
- **Circularity.** `tekyerema-nllb600m-v1` was the verifier used to build the gold tier, so the benchmark favours sentences that model translates well.
- **Bible overlap with other benchmarks.** About 62% of the 1,000 NSANKU kus→eng verses appear in the June parallel corpus that the original model trained on, so any NSANKU score for these models is contaminated.
- **The June "original test set"** (1,956 rows) was not seen by `kusaal-nllb-600M`, but it *was* ingested (via the whole in-house corpus) into `tekyerema-nllb600m-v1`'s training data.

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
