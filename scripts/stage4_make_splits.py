# ==============================================================
# Stage 4a: freeze the fine-tuning data splits (run ONCE).
#
# Creates Desktop\MT\finetune_data\ :
#   wiki_test.csv   1,000 gold pairs — FROZEN benchmark, never trained on
#   wiki_val.csv      500 gold pairs — checkpoint selection
#   wiki_train.csv  remaining gold + all silver
#   bt_pool.csv     copy of backtranslation_pool.csv (eng->kus only)
#   probe_set.csv   8 qualitative probes (known failure modes)
#   README.md
#
# Fixed seed so the split is reproducible forever.
# ==============================================================

import csv
import os
import random
import shutil

FINAL = r"C:\Users\Prince\Desktop\MT\final_dataset"
OUT = r"C:\Users\Prince\Desktop\MT\finetune_data"
os.makedirs(OUT, exist_ok=True)

rng = random.Random(42)

gold = list(csv.DictReader(open(f"{FINAL}/wiki_parallel_gold.csv", encoding="utf-8-sig")))
silver = list(csv.DictReader(open(f"{FINAL}/wiki_parallel_silver.csv", encoding="utf-8-sig")))

rng.shuffle(gold)
test, val, gold_train = gold[:1000], gold[1000:1500], gold[1500:]
train = gold_train + silver
rng.shuffle(train)

PAIR_COLS = ["kus", "eng", "kus_title", "en_title", "origin", "cos", "chrf"]


def write(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PAIR_COLS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


write(f"{OUT}/wiki_test.csv", test)
write(f"{OUT}/wiki_val.csv", val)
write(f"{OUT}/wiki_train.csv", train)
shutil.copy(f"{FINAL}/backtranslation_pool.csv", f"{OUT}/bt_pool.csv")

# Qualitative probes: real pairs exercising known failure modes.
# (Some appear in training data — they are demo probes, not a metric.)
PROBES = [
    ("dates-numbers", "Ba da du'a Haruna nɛ yʋʋm tusir, kɔbiswai nɛ pisyɔpɔi nɛ anaasi, Nwadisa ayi la daba anaasi la daar (4 February 1974).",
     "Haruna was born on 4 February 1974."),
    ("dates-numbers", "O da paamnɛ vɔt bam 43,561 yi vɔt la wʋsa linɛ da an 57,478 la ni.",
     "He was elected with 43,561 votes out of 57,478 total valid votes cast."),
    ("dates-numbers", "Yʋʋm tusir, kɔbiswai nɛ piswai nɛ awai (1999) paae yʋʋm tusa ayi nɛ piinɛ atan (2003).",
     "From 1999 to 2003."),
    ("dates-numbers", "Nidib nwɛnɛ miliyɔŋa anu nɛ tusa pisyɔpɔi (5,070,000) yʋda da bɛ vɔɔtʋg gbaʋŋ la ni.",
     "Around 5,070,000 people were registered to vote."),
    ("calque", "O anɛ Kristo biig.", "He is a Christian."),
    ("gender", "O mɔr pʋ'a nɛ biisa ayi'.", "He is married with two children."),
    ("gender", "O kul sid ka mɔr biisa ayi.", "She is married and has two children."),
    ("entities", "Mahama anɛ ma'asim tiig na'adɔɔg nid (NDC).",
     "Mahama is a member of the National Democratic Congress (NDC)."),
]
with open(f"{OUT}/probe_set.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f, quoting=csv.QUOTE_ALL)
    w.writerow(["category", "kus", "eng_reference"])
    w.writerows(PROBES)

with open(f"{OUT}/README.md", "w", encoding="utf-8") as f:
    f.write(
        "# Kusaal wiki fine-tuning splits (frozen, seed 42)\n\n"
        f"- wiki_test.csv : {len(test)} gold pairs - benchmark, NEVER train on these\n"
        f"- wiki_val.csv  : {len(val)} gold pairs - checkpoint selection\n"
        f"- wiki_train.csv: {len(train)} pairs (gold {len(gold_train)} + silver {len(silver)})\n"
        "- bt_pool.csv   : real Kusaal + synthetic English - eng->kus direction ONLY\n"
        "- probe_set.csv : 8 qualitative probes (dates/numbers, calques, gender, entities)\n\n"
        "Source: Kusaal Wikipedia harvest, August 2026. Deduped against the\n"
        "tekyerema-pa-mt and kusaal-english-parallel-corpus training data.\n")

print(f"test={len(test)} val={len(val)} train={len(train)} "
      f"(gold_train={len(gold_train)}, silver={len(silver)})")
print("wrote", OUT)
