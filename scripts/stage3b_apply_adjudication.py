# ==============================================================
# Stage 3b: apply Claude's adjudication verdicts.
#
# - Fills the review_verdict column in adjudication.csv
#   (y = genuine pair, p = partial/loose match, n = not a pair)
# - Promotes y-rows into wiki_parallel_silver.csv (origin
#   "adjudicated"), applying the same dedup/identity/length
#   filters as Stage 3
# - Writes p-rows to partial_pairs.csv for a later decision
# - n-rows with sane MT join the back-translation pool
# ==============================================================

import csv
import os
from collections import Counter

FINAL = r"C:\Users\Prince\Desktop\MT\final_dataset"
VERDICTS = (r"C:\Users\Prince\AppData\Local\Temp\claude"
            r"\C--Users-Prince-Desktop-MT\411acd46-b38c-4f3c-b292-ad8915e0adc0"
            r"\scratchpad\verdicts_full.txt")
TRAIN_PARQUET = (r"C:\Users\Prince\AppData\Local\Temp\claude"
                 r"\C--Users-Prince-Desktop-MT\411acd46-b38c-4f3c-b292-ad8915e0adc0"
                 r"\scratchpad\eng-kus-train.parquet")


def norm(s):
    return " ".join(s.split()).casefold()


verdicts = {}
for line in open(VERDICTS, encoding="utf-8"):
    idx, v = line.split()
    verdicts[int(idx)] = v

adj = list(csv.DictReader(open(f"{FINAL}/adjudication.csv", encoding="utf-8-sig")))
assert len(adj) == len(verdicts), (len(adj), len(verdicts))

# existing pairs for dedup
gold = list(csv.DictReader(open(f"{FINAL}/wiki_parallel_gold.csv", encoding="utf-8-sig")))
silver = list(csv.DictReader(open(f"{FINAL}/wiki_parallel_silver.csv", encoding="utf-8-sig")))
seen = {(norm(r["kus"]), norm(r["eng"])) for r in gold + silver}

existing_kus = set()
try:
    import pyarrow.parquet as pq
    t = pq.read_table(TRAIN_PARQUET).to_pydict()
    for sl, tl, s, tg in zip(t["source_lang"], t["target_lang"],
                             t["source_text"], t["target_text"]):
        kus = tg if tl == "kus" else (s if sl == "kus" else None)
        if kus:
            existing_kus.add(norm(kus))
except Exception as e:
    print("WARNING: no training-data dedup:", e)


def mt_sane(kus, mt):
    if len(mt) < 10:
        return False
    toks = mt.lower().split()
    if len(toks) >= 8 and max(Counter(toks).values()) / len(toks) > 0.34:
        return False
    return 0.3 <= len(mt) / max(len(kus), 1) <= 3.5


def pair_ok(kus, en):
    if norm(kus) == norm(en):
        return False
    return 1 / 6 <= len(en) / max(len(kus), 1) <= 6


promoted, partials, bt_extra = [], [], []
stats = Counter()
for i, r in enumerate(adj):
    v = verdicts[i]
    r["review_verdict"] = v
    kus, en, mt = r["kus_sentence"], r["en_sentence"], r["machine_translation"]
    if v == "y":
        key = (norm(kus), norm(en))
        if key in seen:
            stats["y: duplicate, skipped"] += 1
        elif norm(kus) in existing_kus:
            stats["y: already in training data"] += 1
        elif not pair_ok(kus, en):
            stats["y: identity/length filtered"] += 1
        else:
            seen.add(key)
            promoted.append({"kus": kus, "eng": en,
                             "kus_title": r["kus_title"],
                             "en_title": r["en_title"],
                             "origin": "adjudicated",
                             "cos": r["cos"], "chrf": r["chrf"]})
            stats["y: promoted to silver"] += 1
    elif v == "p":
        partials.append({"kus": kus, "eng": en,
                         "kus_title": r["kus_title"],
                         "en_title": r["en_title"],
                         "origin": r["origin"],
                         "cos": r["cos"], "chrf": r["chrf"]})
        stats["p: partial (kept aside)"] += 1
    else:
        if norm(kus) not in existing_kus and mt_sane(kus, mt):
            bt_extra.append({"kus": kus, "eng_synthetic": mt,
                             "kus_title": r["kus_title"],
                             "origin": r["origin"], "cos": r["cos"]})
            stats["n: added to bt pool"] += 1
        else:
            stats["n: dropped"] += 1


def write(path, rows_, cols, mode="w"):
    exists = mode == "a" and os.path.exists(path)
    with open(path, mode, encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        if not exists:
            w.writeheader()
        w.writerows(rows_)


pair_cols = ["kus", "eng", "kus_title", "en_title", "origin", "cos", "chrf"]
adj_cols = list(adj[0].keys())
write(f"{FINAL}/adjudication.csv", adj, adj_cols)
write(f"{FINAL}/wiki_parallel_silver.csv", silver + promoted, pair_cols)
write(f"{FINAL}/partial_pairs.csv", partials, pair_cols)
write(f"{FINAL}/backtranslation_pool.csv", bt_extra,
      ["kus", "eng_synthetic", "kus_title", "origin", "cos"], mode="a")

print("verdict counts:", dict(Counter(verdicts.values())))
for k, v in sorted(stats.items()):
    print(f"  {k}: {v}")
print(f"\nsilver now: {len(silver) + len(promoted)} pairs "
      f"(was {len(silver)}, +{len(promoted)})")
print(f"gold unchanged: {len(gold)} pairs")
print(f"partials: {len(partials)} -> partial_pairs.csv")
