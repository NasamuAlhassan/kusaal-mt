# ==============================================================
# Stage 3: final cut — turn stage2_pairs.csv into training-ready
# files. Local, no GPU, seconds.
#
# Reads : stage2/stage2_pairs.csv
#         stage2/anchor_pairs.csv        (for the anchors column)
#         eng-kus train parquet          (dedup vs existing training data)
# Writes: final_dataset/wiki_parallel_gold.csv    highest-precision pairs
#         final_dataset/wiki_parallel_silver.csv  good pairs, slightly looser
#         final_dataset/adjudication.csv          borderline rows for review
#         final_dataset/backtranslation_pool.csv  rejects as monolingual kus + MT
#         final_dataset/stage3_stats.txt
# ==============================================================

import csv
import os
import re
from collections import Counter

STAGE2 = r"C:\Users\Prince\Desktop\MT\stage2"
OUT = r"C:\Users\Prince\Desktop\MT\final_dataset"
TRAIN_PARQUET = (r"C:\Users\Prince\AppData\Local\Temp\claude"
                 r"\C--Users-Prince-Desktop-MT\411acd46-b38c-4f3c-b292-ad8915e0adc0"
                 r"\scratchpad\eng-kus-train.parquet")
os.makedirs(OUT, exist_ok=True)

rows = list(csv.DictReader(open(f"{STAGE2}/stage2_pairs.csv", encoding="utf-8-sig")))

# anchors column from stage 1 (stage2_pairs.csv doesn't carry it)
anchors_of = {}
for r in csv.DictReader(open(f"{STAGE2}/anchor_pairs.csv", encoding="utf-8-sig")):
    anchors_of[(r["kus_title"], r["kus_pos"])] = r["anchors"]

# kusaal side of the existing tekyerema training data, for dedup
existing_kus = set()
try:
    import pyarrow.parquet as pq
    t = pq.read_table(TRAIN_PARQUET).to_pydict()
    for sl, tl, s, tg in zip(t["source_lang"], t["target_lang"],
                             t["source_text"], t["target_text"]):
        kus = tg if tl == "kus" else (s if sl == "kus" else None)
        if kus:
            existing_kus.add(" ".join(kus.split()).casefold())
    print(f"loaded {len(existing_kus)} existing kusaal training sentences for dedup")
except Exception as e:
    print("WARNING: could not load training parquet for dedup:", e)


def norm(s):
    return " ".join(s.split()).casefold()


def has_digit_anchor(r):
    a = anchors_of.get((r["kus_title"], r["kus_pos"]), "")
    return bool(re.search(r"\b\d{2,}\b", a))


def mt_sane(kus, mt):
    """Cheap sanity filter for machine translations in the BT pool."""
    if len(mt) < 10:
        return False
    toks = mt.lower().split()
    if len(toks) >= 8 and max(Counter(toks).values()) / len(toks) > 0.34:
        return False                                    # repetition loop
    ratio = len(mt) / max(len(kus), 1)
    return 0.3 <= ratio <= 3.5


def pair_ok(kus, en):
    if norm(kus) == norm(en):
        return False                                    # identity/list row
    ratio = len(en) / max(len(kus), 1)
    return 1 / 6 <= ratio <= 6                          # corpus length convention


gold, silver, adjud, btpool = [], [], [], []
stats = Counter()
seen_pairs = set()

for r in rows:
    cos, ch = float(r["cos"]), float(r["chrf"])
    origin, verdict = r["origin"], r["verdict"]
    kus, en, mt = r["kus_sentence"], r["en_sentence"], r["machine_translation"]

    key = (norm(kus), norm(en))
    dup_new = key in seen_pairs
    dup_train = norm(kus) in existing_kus

    dest = None
    if verdict == "accept":
        if origin == "anchor-confident":
            dest = "gold"
        elif origin == "anchor-unsure":
            dest = "silver"
        elif origin == "mined":
            dest = "silver" if cos >= 0.70 else "adjud"
    elif verdict == "uncertain":
        if origin.startswith("anchor") and (cos >= 0.50 or ch >= 40):
            dest = "silver"
        elif origin == "mined" and cos >= 0.60:
            dest = "adjud"
        else:
            dest = "bt"
    else:  # reject
        if origin == "anchor-confident" and has_digit_anchor(r):
            dest = "adjud"                              # likely false reject
        else:
            dest = "bt"

    if dest in ("gold", "silver"):
        if dup_new:
            stats["dropped duplicate pair"] += 1
            continue
        if dup_train:
            stats["dropped: already in tekyerema training data"] += 1
            continue
        if not pair_ok(kus, en):
            stats["dropped by identity/length filter"] += 1
            continue
        seen_pairs.add(key)
        out = {"kus": kus, "eng": en, "kus_title": r["kus_title"],
               "en_title": r["en_title"], "origin": origin,
               "cos": r["cos"], "chrf": r["chrf"]}
        (gold if dest == "gold" else silver).append(out)
        stats[dest] += 1
    elif dest == "adjud":
        adjud.append({**r, "anchors": anchors_of.get(
            (r["kus_title"], r["kus_pos"]), ""), "review_verdict": ""})
        stats["adjudication"] += 1
    else:
        if dup_train:
            stats["bt pool: dropped (in training data)"] += 1
        elif mt_sane(kus, mt):
            btpool.append({"kus": kus, "eng_synthetic": mt,
                           "kus_title": r["kus_title"], "origin": origin,
                           "cos": r["cos"]})
            stats["backtranslation pool"] += 1
        else:
            stats["bt pool: dropped (bad MT)"] += 1


def write(path, rows_, cols):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows_)


pair_cols = ["kus", "eng", "kus_title", "en_title", "origin", "cos", "chrf"]
write(f"{OUT}/wiki_parallel_gold.csv", gold, pair_cols)
write(f"{OUT}/wiki_parallel_silver.csv", silver, pair_cols)
write(f"{OUT}/adjudication.csv", adjud,
      list(adjud[0].keys()) if adjud else ["origin"])
write(f"{OUT}/backtranslation_pool.csv", btpool,
      ["kus", "eng_synthetic", "kus_title", "origin", "cos"])

lines = [f"input rows                : {len(rows)}"]
lines += [f"{k:38s}: {v}" for k, v in sorted(stats.items())]
lines += ["",
          f"gold pairs   -> wiki_parallel_gold.csv   ({len(gold)})",
          f"silver pairs -> wiki_parallel_silver.csv ({len(silver)})",
          f"adjudication -> adjudication.csv         ({len(adjud)})",
          f"bt pool      -> backtranslation_pool.csv ({len(btpool)})"]
report = "\n".join(lines)
with open(f"{OUT}/stage3_stats.txt", "w", encoding="utf-8") as f:
    f.write(report + "\n")
print(report)
