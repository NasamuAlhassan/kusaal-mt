# Kusaal–English Wikipedia Parallel Corpus — Full Pipeline Report

**Project:** Mining a sentence-aligned Kusaal↔English parallel corpus from Wikipedia
**Author:** Prince Alhassan Nasamu ([@NasamuAlhassan](https://github.com/NasamuAlhassan), [PrinceAlhassanNasamu](https://huggingface.co/PrinceAlhassanNasamu) on HF), with Claude
**Period:** August 2026
**Goal:** Expand the training data for the Kusaal↔English NMT models (`kusaal-nllb-600M`, `tekyerema-nllb600m-v1`) to push BLEU/chrF++ higher, using Kusaal Wikipedia and its English counterparts as a comparable corpus.

**Bottom line:** from 1,170 comparable article pairs, the pipeline produced
**13,659 verified parallel sentence pairs** (8,268 gold + 5,391 silver),
a **6,917-sentence back-translation pool**, and 92 partial pairs held aside —
a ~40% expansion over all Kusaal parallel data the models had ever seen,
in encyclopedic register rather than scripture.

---

## Pipeline overview

```
kus.wikipedia.org ──(scrape, MediaWiki API)──> 1,170 article pairs
        │
        ▼
Stage 1 (local, seconds): clean + segment + ANCHOR alignment
        │   confident 10,440 · unsure 6,587 · unmatched 4,512
        ▼
Stage 2 (Colab T4, ~1h): tekyerema translates ALL 21,539 kus sentences
        │   mpnet cosine + chrF++ verify anchors & mine the unmatched
        │   accept 11,814 · uncertain 5,544 · reject 4,181
        ▼
Stage 3 (local, seconds): final cut + dedup vs existing training data
        │   gold 8,268 · silver 5,148 · adjudication 503 · BT pool 6,762
        ▼
Stage 3b: Claude adjudicates the 503 borderliners by hand
            y 252 → +243 to silver · p 92 aside · n 159 → BT pool
        ═══════════════════════════════════════════════════════════
        FINAL: gold 8,268 · silver 5,391 · BT 6,917 · partial 92
```

---

## 1. Scraping (Colab, CPU)

**Tool:** `colab_scrape_articles.ipynb` (also `colab_scrape_articles.py`).

**Method:** MediaWiki API only — no HTML scraping.
- Enumerated every main-namespace, non-redirect article on `kus.wikipedia.org`
  with `generator=allpages`, reading each article's English interlanguage link
  (`prop=langlinks&lllang=en`) in the same pass.
- Fetched full plain text of both sides via the TextExtracts API
  (`prop=extracts&explaintext=1&exsectionformat=wiki`). This strips infoboxes,
  tables, and reference lists at the source, and keeps `== Heading ==` markers
  for section handling later.
- Descriptive User-Agent per Wikimedia policy
  (`KusaalMT-corpus-builder/1.0 … github.com/NasamuAlhassan`), `maxlag=5`,
  politeness delays, retry/backoff, **resumable** (skips already-scraped pairs
  on rerun).

**Results:**

| Metric | Value |
|---|---|
| Kusaal articles total | 1,723 |
| With an English langlink → scraped | **1,170 pairs** |
| Without English link (skipped) | 550 |
| Failed (empty extracts: 2 Bible book pages + Main Page) | 3 |
| Kusaal text volume | 2.97 M chars |
| English text volume | 6.33 M chars |

**Character stats:** Kusaal articles min/median/mean/max = 47 / 1,660 / 2,534 / 47,779 chars; English = 59 / 2,055 / 5,410 / 97,839. No truncation even on the largest article (Nelson Mandela, 98K chars).

**Key discovery:** the corpus is far more parallel than "comparable" suggests.
Median EN/KUS length ratio is **1.1×**; 464 pairs (40%) fall in the 0.9–1.3
band and are essentially sentence-by-sentence translations of the English
article. 154 pairs have ratio ≥3 (Kusaal summarizes a long English article).

**The 550 unpaired articles:** ~48 Bible pages (`Wina'am Gbauŋ:*`), 10 Kusaal
folktales (`Sɔlima:*`), `AZAYA` reading-lesson pages, and local-topic stubs
with no English counterpart. Kept in `unpaired_kus_titles.txt` — usable later
as monolingual Kusaal.

---

## 2. Pre-alignment research (decisive groundwork)

### 2.1 Which model translates Kusaal?

Two candidates on the Hugging Face account:

| | `kusaal-nllb-600M` (public, Jul 2026) | **`tekyerema-nllb600m-v1`** (private, Aug 2026) |
|---|---|---|
| kus→eng quality | BLEU 27.57 | **chrF++ 32.15** on *out-of-domain everyday* holdout |
| eng→kus | BLEU 13.72 | chrF++ 26.92 |
| Training data | ~34.5K pairs | ~325K pairs, 6 language configs, 35,491 Kusaal rows |
| Inference API | manual `kusaal_lang_codes.json` re-registration hack | **standard NLLB usage** (`tok.src_lang="kus_Latn"`, `forced_bos_token_id`) |
| Language tokens | kus_Latn 256204 | kus_Latn 256204, dag_Latn 256205 (added_tokens.json) |

**Decision: `tekyerema-nllb600m-v1`** — newer, ~10× the data, standard API,
scored on deliberately out-of-domain text.

### 2.2 The apostrophe problem (silent quality killer)

The wiki text and the model's training data use **different apostrophes**:

| Character | Wiki scrape | tekyerema Kusaal training rows |
|---|---|---|
| `ꞌ` saltillo U+A78C | **32,754** | **0** |
| `'` straight U+0027 | 12,811 | **85,083** |
| `’` curly U+2019 | 975 | 1,900 |
| `‘` U+2018 | 36 | 850 |
| `ʼ` U+02BC | 36 | 5 |

Feeding saltillo text to the model would make most glottal-stop words
(`sʋꞌʋlʋm`, `Winaꞌam`) look like unseen vocabulary. **Rule adopted: fold every
variant to straight `'` (U+0027)** before anything touches the model.

### 2.3 Training-data overlap

tekyerema's `eng-kus` train split (35,491 rows) contains an
`inhouse-kusaal-wiki` source with **1,113 rows** — some wiki content was
already trained on. Consequence: the final cut **dedups against the entire
training Kusaal side** to avoid double-counting and test contamination.
Source breakdown of the training split: scripture 21,604 · dictionary 6,146 ·
opus-translatewiki 4,138 · ghananlp 2,490 · wiki 1,113.

---

## 3. Stage 1 — cleaning + anchor alignment (`stage1_anchor_align.py`, local)

**Cleaning (Stage 0 inside the script):**
- Unicode NFC; apostrophe folding (per §2.2); URL removal; curly→straight quotes.
- Junk sections dropped on both sides: References, External links, See also,
  Sources, Notes, Further reading, Bibliography, Footnotes, Gallery, Citations,
  Works cited.
- **Deglue fix** for stripped-link artifacts (`DistrictJohn Atta Mills`):
  split lowercase→uppercase glue only when ≥3 lowercase letters precede
  (leaves `McDonald`/`YouTube` alone), Kusaal side only → **113 fixes**.
- Sentence splitting: protected abbreviations (Mr./Dr./etc.), single-letter
  initials (`J. J. Rawlings`), and decimals; split at `.!?` + whitespace.
- Sentence filters: ≥15 chars, ≥3 tokens, ≥50% letters, ends in sentence
  punctuation; list/table lines dropped; per-article dedup.

**Output of cleaning: 21,539 Kusaal / 43,949 English sentences.**

**Anchor matching (per article, never across articles):**
- **Hard anchors:** numbers with ≥2 digits, capitalized tokens ≥3 chars.
  Shared hard tokens between a Kusaal and English sentence are almost always
  proper names, years, or figures — free named-entity matching, because
  ordinary vocabulary never coincides across the two languages.
- **Soft anchors:** shared lowercase tokens ≥4 chars — English **loanwords**
  in Kusaal prose (`security`, `mixtape`, `university`), weighted 0.5×.
- Scoring: idf-style weight `1/df_kus + 1/df_en` per shared anchor (an anchor
  in every sentence of an article is nearly worthless; a unique one is strong),
  plus a 0.3 positional bonus for similar relative position.
- **Confident tier requires:** a distinctive anchor (df ≤ 2 both sides) AND
  mutual-best match AND margin ≥ 0.5 over the runner-up AND (≥2 hard anchors
  OR score ≥ 2.2).

**Tuning iteration that mattered:** v1 allowed single-anchor confident pairs;
spot-checking showed those were ~50% wrong (a lone shared "Adam" or "Europe"
proves nothing). v2 added soft-anchor evidence + the "strong" requirement;
after the fix the weakest confident pairs sampled were all genuine.

**Stage 1 results (runtime ~15 s):**

| Tier | Count | Share |
|---|---|---|
| confident | 10,440 | 48% |
| unsure | 6,587 | 31% |
| unmatched | 4,512 | 21% |

Outputs: `anchor_pairs.csv` (UTF-8-BOM, QUOTE_ALL — Excel-safe),
`unmatched_kus.csv`, `stage1_stats.txt`.

---

## 4. Stage 2 — model verification & mining (GPU)

### 4.1 The Kaggle detour (abandoned, lessons learned)

Plan: fully automated via the Kaggle API (credentials already on the laptop).
Uploaded inputs as private dataset `alhassanprince/kusaal-stage2-data`,
pushed kernel `alhassanprince/kusaal-stage2-align`. Two failures:

1. **P100 incompatibility:** Kaggle assigned a Tesla P100 (compute capability
   6.0); current PyTorch requires sm_70+. Fix exists:
   `kaggle kernels push --accelerator NvidiaTeslaT4`.
2. **Secrets unreachable:** API-pushed kernels could not read the account's
   HF token secret (`UserSecretsClient` found nothing under any common label),
   even after attaching HF_TOKEN in the editor — attachments don't reliably
   survive API pushes. No API exists for attaching secrets.

**Decision:** move to Colab, where the scraped data already lived on Drive and
Colab Secrets are proven to work.

### 4.2 The Colab run (`colab_align_stage1_stage2.ipynb`, T4, ~1 h)

One notebook, run top to bottom:
1. Mount Drive; read HF token from Colab Secrets (`HF_TOKEN`).
2. **Re-run Stage 1 with identical code** (byte-identical outputs confirmed)
   so sentence indices line up with no upload step.
3. **Translate all 21,539 Kusaal sentences** with tekyerema
   (kus_Latn→eng_Latn, fp16, beam 4, batch 64, length-sorted;
   checkpointed to `translations.jsonl` after every batch → disconnect-safe).
4. **Score:** `sentence-transformers/all-mpnet-base-v2` embeds the machine
   English and all wiki English (normalized, cosine via dot product);
   **chrF++** (sacrebleu, word_order=2) as an independent surface-overlap
   signal.
   - Anchor pairs (confident + unsure): score the proposed pairing.
   - Unmatched sentences: **mine** — best cosine against every English
     sentence of the same article, with margin (best − second).
5. **Verdicts:** accept if cos ≥ 0.65 or (cos ≥ 0.55 and chrF ≥ 45);
   reject if cos < 0.40 and chrF < 25; else uncertain. Mined accepts
   additionally need margin ≥ 0.04.

**Calibration (validates the thresholds):**

| Distribution | p5 | p25 | p50 | p90/95/99 |
|---|---|---|---|---|
| anchor-confident cosine (true-match proxy) | 0.45 | 0.707 | 0.848 | p90 = 0.988 |
| random cross-article negatives | — | — | 0.095 | p95 = 0.395, p99 = 0.546 |

Only ~1% of random negatives exceed 0.55 → the 0.65 accept bar is conservative.

**Stage 2 verdict counts:**

| Origin | accept | uncertain | reject |
|---|---|---|---|
| anchor-confident (10,440) | 8,765 | 1,445 | 230 |
| anchor-unsure (6,587) | 2,739 | 2,391 | 1,457 |
| mined (4,512) | 310 | 1,708 | 2,494 |
| **Total** | **11,814** | **5,544** | **4,181** |

Outputs: `stage2_pairs.csv` (every pair + MT + scores + verdict),
`translations.jsonl`, `stage2_stats.txt`.

---

## 5. Stage 3 — final cut (`stage3_final_cut.py`, local)

**Routing rules:**
- **Gold** = anchor-confident ∧ accept.
- **Silver** = anchor-unsure accept ∪ anchor uncertain rescued at
  (cos ≥ 0.50 or chrF ≥ 40) ∪ mined accept with cos ≥ 0.70.
- **Adjudication** = anchor-confident rejects carrying digit anchors
  (suspected false rejects) ∪ mined borderliners (accepts < 0.70,
  uncertain ≥ 0.60).
- **Back-translation pool** = everything else with a sane MT
  (no repetition loops, length ratio 0.3–3.5, ≥10 chars):
  real Kusaal + synthetic English, for the eng→kus training direction.

**Filters applied to gold/silver:**
- identity rows dropped (Kusaal == English list items like film titles): part of 321 identity/length drops;
- length-ratio guard 1/6–6× (matching the original corpus convention);
- exact duplicate pairs: 21;
- **already in tekyerema training data (kus-side match): 331 dropped** — no contamination;
- BT-pool hygiene: 156 bad-MT drops, 29 in-training drops.

**Stage 3 output:** gold 8,268 · silver 5,148 · adjudication 503 · BT 6,762.
All 21,539 input sentences accounted for exactly.

---

## 6. Stage 3b — human(-ish) adjudication of the 503 borderliners

Every row judged by Claude with the Kusaal sentence, the machine translation,
and the English candidate side by side — including directly decoding Kusaal
spelled-out numerals (e.g. `tusa atan', kɔbisnu nɛ pisyuobʋ nɛ ayi` = 3,562)
against the English figures. Verdict scheme: **y** genuine pair · **p** partial
(one side states substantially more) · **n** not a pair.

**Verdicts:** y 252 (50%) · p 92 (18%) · n 159 (32%).
Applied by `stage3b_apply_adjudication.py`: 243 y-rows promoted to silver
(9 were duplicates), 92 partials to `partial_pairs.csv` (excluded from
training — they teach hallucination), 155 n-rows with sane MT joined the BT
pool (4 dropped).

**Findings from the adjudication:**
- **The Bible-register bug is real and rescueable.** ~85% of the 135
  anchor-confident "rejects" were true pairs whose similarity score was
  destroyed by tekyerema rendering Kusaal number-words KJV-style:
  `Ba da du'a Haruna … (4 February 1974)` → *"Aaron was born in the year one
  thousand eight hundred seventy-four…"*. The anchors (shared digits) were
  right; the verifier was misled by the MT, not the alignment.
- **Mined rows ran ~40% genuine** — e.g. WHO mission statements, Genesis
  narrative summaries ("Cain … takes Abel to a field and murders him"),
  and biographical formulae (`O mɔr pʋ'a nɛ biisa ayi'` = "married with two
  children").
- **A recurring template**: constituency articles pair Kusaal "they vote to
  choose one person as their MP" with English "It elects one MP by the first
  past the post system" — marked partial unless the Kusaal approximates the
  FPTP clause.
- **Model quirks worth fixing in the next fine-tune:** literal calques
  (`Kristo biig` → "son of Christ" instead of "Christian"); wrong gender
  pronouns (Kusaal `o` is gender-neutral); number-word arithmetic errors.

---

## 7. Final deliverables (`Desktop\MT\final_dataset\`)

| File | Rows | Content |
|---|---|---|
| `wiki_parallel_gold.csv` | **8,268** | anchor-confirmed + model-verified pairs; columns kus, eng, titles, origin, cos, chrf |
| `wiki_parallel_silver.csv` | **5,391** | rescued/mined/adjudicated pairs (origin column distinguishes) |
| `backtranslation_pool.csv` | **6,917** | real Kusaal + synthetic English (MT), for eng→kus augmentation |
| `partial_pairs.csv` | 92 | true correspondences with asymmetric content — reference only |
| `adjudication.csv` | 503 | full borderline set with review_verdict filled (y/p/n) |
| `stage3_stats.txt` | — | cut summary |

**Total new parallel data: 13,659 pairs** — vs ~34.5K in the original
kusaal-nllb corpus (77% scripture), i.e. roughly **+40% Kusaal parallel data,
nearly all of it non-religious register** (politics, biography, geography,
agriculture, science, culture).

### Scripts & notebooks (in `Desktop\MT\`)

| File | Role |
|---|---|
| `colab_scrape_articles.ipynb` / `.py` | Wikipedia article-pair scraper (Colab, CPU) |
| `stage1_anchor_align.py` | cleaning + anchor alignment (local) |
| `colab_align_stage1_stage2.ipynb` | Stage 1 rerun + Stage 2 GPU verification (Colab T4) |
| `stage3_final_cut.py` | verdicts → gold/silver/adjudication/BT (local, rerunnable) |
| `stage3b_apply_adjudication.py` | applies adjudication verdicts (local) |

### Intermediate data
- `Scraped Articles\wiki_pairs\` — article_pairs.jsonl (10 MB), unpaired/failed lists
- `stage1_output\` — anchor_pairs.csv, unmatched_kus.csv, stats
- `stage2\` — stage2_pairs.csv (21,539 scored rows), translations.jsonl (all 21,539 MTs — reusable), stats, executed notebook
- Drive: `MyDrive/kusaal_mt/wiki_pairs/` and `MyDrive/kusaal_mt/stage2/`
- Kaggle (dormant): private dataset `kusaal-stage2-data`, kernel `kusaal-stage2-align`

---

## 8. Method notes worth remembering

1. **Anchors before models.** Shared digits and proper names between a
   low-resource language and English are a free, high-precision aligner —
   48% of sentences matched confidently before any GPU was touched.
2. **Two independent verification signals beat one.** Embedding cosine and
   chrF++ fail differently; requiring agreement (or strong single-signal
   evidence) is what made verdicts trustworthy.
3. **Calibrate on your own confident tier.** The anchor-confident pairs gave a
   free true-match score distribution; random cross-article pairs gave the
   false distribution. No hand-labeled calibration set needed.
4. **Verify the verifier.** The MT-based check systematically executes
   the translator's biases (Bible-register numbers) — which is why
   anchor-evidence rejects went to adjudication instead of the bin.
5. **Rejects are not waste.** A rejected pair still contains a clean
   monolingual Kusaal sentence with an already-computed machine translation —
   the entire BT pool cost zero extra GPU time.
6. **Normalize orthography to the model's training convention** (apostrophes
   here), or every downstream score silently degrades.
7. **Kaggle API limits:** API-pushed kernels can't reach account secrets, and
   the default P100 no longer runs current PyTorch (`--accelerator
   NvidiaTeslaT4` if ever retried).

---

## 9. Agreed next step (not yet started)

**Fine-tuning recipe** for the next tekyerema round:
- gold + silver as genuine bidirectional pairs;
- BT pool tagged for **eng→kus direction only** (real Kusaal targets);
- hold out ~1,000 gold pairs as a **wiki-domain test set** — the first proper
  measurement of encyclopedic-register performance for Kusaal MT;
- dedup is already guaranteed by construction (§5);
- watch for: gender-pronoun corrections, number-word handling, calque fixes.
