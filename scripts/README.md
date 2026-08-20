# Pipeline scripts

The Kusaal Wikipedia mining and fine-tuning pipeline, in execution order.
These are the scripts exactly as run (August 2026) — file paths are local to
the machine they ran on and will need adjusting; the full methodology,
design decisions, and results are documented in
[`docs/WIKI_CORPUS_PIPELINE_REPORT.md`](../docs/WIKI_CORPUS_PIPELINE_REPORT.md).

| Script | Stage | Where it ran |
|---|---|---|
| `colab_scrape_articles.py` | Scrape 1,170 Kusaal–English article pairs via the MediaWiki API | Colab (CPU) |
| `stage1_anchor_align.py` | Clean, segment, and align sentences on shared anchors (numbers, names, loanwords) | local |
| — (`notebooks/colab_align_stage1_stage2.ipynb`) | Verify anchor pairs and mine unmatched sentences with MT + embedding similarity + chrF++ | Colab (T4) |
| `stage3_final_cut.py` | Route verdicts into gold / silver / adjudication / back-translation pool, dedup against prior training data | local |
| `stage3b_apply_adjudication.py` | Apply manual adjudication verdicts on ~500 borderline pairs | local |
| `stage4_make_splits.py` | Freeze train / validation / test splits (seed 42) | local |
| `finetune_main_model.py` | Continued fine-tuning + before/after evaluation (BLEU, chrF++, probe set) | Kaggle (T4) |
