# ==============================================================
# Fine-tune kusaal-nllb-600M (the main model) on the wiki harvest.
# Kaggle script kernel, T4, no secrets needed (model + corpus public).
#
# Inputs : /kaggle/input/kusaal-finetune-data/  (wiki splits, BT pool, probes)
#          HF hub: PrinceAlhassanNasamu/kusaal-nllb-600M (public)
#                  PrinceAlhassanNasamu/kusaal-english-parallel-corpus (public)
# Outputs: /kaggle/working/
#          eval_results.json     baseline vs fine-tuned, BLEU + chrF++
#          comparison_table.csv  one row per (test set, direction, metric)
#          probe_results.txt     qualitative before/after probes
#          kusaal-nllb-600M-wiki/  the fine-tuned model
#
# SMOKE=True runs the whole pipeline in ~15 min (60 steps, 40-sentence
# evals) to validate end-to-end before committing a full run.
# ==============================================================

import csv
import json
import os
import random
import subprocess
import sys
import time

# Kaggle's "T4" is T4 x2; HF Trainer would wrap the model in DataParallel,
# which breaks M2M100's decoder-input handling. Train on one GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

try:
    import sacrebleu
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sacrebleu"],
                   check=True)
    import sacrebleu

# ---------------- config ----------------
SMOKE = False         # full run

MODEL_ID = "PrinceAlhassanNasamu/kusaal-nllb-600M"
CORPUS_ID = "PrinceAlhassanNasamu/kusaal-english-parallel-corpus"
import glob
_hits = glob.glob("/kaggle/input/**/wiki_train.csv", recursive=True)
assert _hits, ("wiki_train.csv not found under /kaggle/input - is the "
               "kusaal-finetune-data dataset attached and processed?")
DATA = os.path.dirname(_hits[0])
OUT = "/kaggle/working"
SAVE_DIR = f"{OUT}/kusaal-nllb-600M-wiki"

MAX_LEN = 192
MAX_STEPS = 60 if SMOKE else 4000
EVAL_STEPS = 30 if SMOKE else 500
EVAL_CAP = 40 if SMOKE else 1000     # sentences per test set per direction
GEN_BATCH = 24
LR = 3e-5
WARMUP = 10 if SMOKE else 250

print(f"SMOKE={SMOKE}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
rng = random.Random(42)

# ---------------- model ----------------
from huggingface_hub import hf_hub_download
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

# language token ids — the main model needs its lang-code re-registration
LANG_ID = {}
for code in ("kus_Latn", "eng_Latn"):
    tid = tokenizer.convert_tokens_to_ids(code)
    if tid is None or tid == tokenizer.unk_token_id:
        path = hf_hub_download(repo_id=MODEL_ID, filename="kusaal_lang_codes.json")
        LANG_ID = json.load(open(path))
        break
    LANG_ID[code] = tid
EOS = tokenizer.eos_token_id
print("lang ids:", LANG_ID, "eos:", EOS, flush=True)


# ---------------- data ----------------
def read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


wiki_train = read_csv(f"{DATA}/wiki_train.csv")
wiki_val = read_csv(f"{DATA}/wiki_val.csv")
wiki_test = read_csv(f"{DATA}/wiki_test.csv")
bt_pool = read_csv(f"{DATA}/bt_pool.csv")
probes = read_csv(f"{DATA}/probe_set.csv")

legacy_train = read_csv(hf_hub_download(
    repo_id=CORPUS_ID, filename="train.csv", repo_type="dataset"))
legacy_test = read_csv(hf_hub_download(
    repo_id=CORPUS_ID, filename="test.csv", repo_type="dataset"))
rng.shuffle(legacy_test)
legacy_test = legacy_test[:EVAL_CAP]

# training examples: (src_lang, tgt_lang, src_text, tgt_text)
examples = []
for r in wiki_train:
    examples.append(("kus_Latn", "eng_Latn", r["kus"], r["eng"]))
    examples.append(("eng_Latn", "kus_Latn", r["eng"], r["kus"]))
for r in legacy_train:
    if r.get("kusaal") and r.get("english"):
        examples.append(("kus_Latn", "eng_Latn", r["kusaal"], r["english"]))
        examples.append(("eng_Latn", "kus_Latn", r["english"], r["kusaal"]))
for r in bt_pool:                      # synthetic source -> REAL kus target only
    examples.append(("eng_Latn", "kus_Latn", r["eng_synthetic"], r["kus"]))
rng.shuffle(examples)

val_examples = []
for r in wiki_val:
    val_examples.append(("kus_Latn", "eng_Latn", r["kus"], r["eng"]))
    val_examples.append(("eng_Latn", "kus_Latn", r["eng"], r["kus"]))

print(f"train examples: {len(examples)}  (wiki {2*len(wiki_train)}, "
      f"legacy {2*len(legacy_train)}, bt {len(bt_pool)})", flush=True)


def encode(src_lang, tgt_lang, src, tgt):
    s = tokenizer(src, add_special_tokens=False)["input_ids"][:MAX_LEN - 2]
    t = tokenizer(tgt, add_special_tokens=False)["input_ids"][:MAX_LEN - 2]
    return {"input_ids": [LANG_ID[src_lang]] + s + [EOS],
            "labels": [LANG_ID[tgt_lang]] + t + [EOS]}


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return encode(*self.items[i])


# ---------------- evaluation ----------------
@torch.no_grad()
def translate_all(texts, src_lang, tgt_lang):
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    for bi in range(0, len(order), GEN_BATCH):
        idxs = order[bi:bi + GEN_BATCH]
        batch = [{"input_ids":
                  [LANG_ID[src_lang]] +
                  tokenizer(texts[i], add_special_tokens=False)["input_ids"][:MAX_LEN - 2] +
                  [EOS]} for i in idxs]
        padded = tokenizer.pad(batch, return_tensors="pt").to("cuda")
        gen = model.generate(**padded,
                             forced_bos_token_id=LANG_ID[tgt_lang],
                             decoder_start_token_id=EOS,
                             num_beams=4, max_new_tokens=MAX_LEN)
        for i, o in zip(idxs, tokenizer.batch_decode(gen, skip_special_tokens=True)):
            out[i] = o.strip()
    return out


def score(name, rows, kus_key, eng_key, results):
    rows = rows[:EVAL_CAP]
    kus = [r[kus_key] for r in rows]
    eng = [r[eng_key] for r in rows]
    for direction, srcs, refs, sl, tl in (
            ("kus-eng", kus, eng, "kus_Latn", "eng_Latn"),
            ("eng-kus", eng, kus, "eng_Latn", "kus_Latn")):
        t0 = time.time()
        hyps = translate_all(srcs, sl, tl)
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
        results[f"{name}/{direction}"] = {"bleu": round(bleu, 2),
                                          "chrf++": round(chrf, 2),
                                          "n": len(srcs)}
        print(f"  {name}/{direction}: BLEU {bleu:.2f}  chrF++ {chrf:.2f} "
              f"({len(srcs)} sents, {time.time()-t0:.0f}s)", flush=True)


def run_probes(tag, lines):
    lines.append(f"\n===== probes: {tag} =====")
    hyps = translate_all([p["kus"] for p in probes], "kus_Latn", "eng_Latn")
    for p, h in zip(probes, hyps):
        lines.append(f"[{p['category']}] KUS: {p['kus']}")
        lines.append(f"  REF: {p['eng_reference']}")
        lines.append(f"  HYP: {h}")


model = model.half().to("cuda")
results = {"model": MODEL_ID, "smoke": SMOKE, "baseline": {}, "finetuned": {}}
probe_lines = []

print("=== BASELINE evaluation ===", flush=True)
score("wiki_test", wiki_test, "kus", "eng", results["baseline"])
score("legacy_test", legacy_test, "kusaal", "english", results["baseline"])
run_probes("baseline", probe_lines)

# ---------------- training ----------------
model = model.float()               # train in fp32 master weights + fp16 autocast
args = TrainingArguments(
    output_dir=f"{OUT}/ckpt",
    max_steps=MAX_STEPS,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=LR,
    warmup_steps=WARMUP,
    lr_scheduler_type="linear",
    # no Trainer label smoothing: it pops `labels` before the forward pass,
    # leaving M2M100 unable to derive decoder_input_ids (obscure ValueError)
    optim="adafactor",
    fp16=True,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=EVAL_STEPS,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    logging_steps=25,
    report_to="none",
    seed=42,
    dataloader_num_workers=2,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=PairDataset(examples),
    eval_dataset=PairDataset(val_examples[:200] if SMOKE else val_examples),
    # labels-only collation: the model derives decoder_input_ids itself
    # (passing model= makes the collator pre-build them, which conflicts
    # with current transformers' M2M100 kwargs routing)
    data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
)
print("=== TRAINING ===", flush=True)
t0 = time.time()
trainer.train()
print(f"training done in {(time.time()-t0)/60:.1f} min", flush=True)

# ---------------- final evaluation ----------------
model = trainer.model.half().to("cuda")
print("=== FINE-TUNED evaluation ===", flush=True)
score("wiki_test", wiki_test, "kus", "eng", results["finetuned"])
score("legacy_test", legacy_test, "kusaal", "english", results["finetuned"])
run_probes("finetuned", probe_lines)

# ---------------- outputs ----------------
with open(f"{OUT}/eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

with open(f"{OUT}/comparison_table.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["test_set/direction", "metric", "baseline", "finetuned", "delta"])
    for key in results["baseline"]:
        for metric in ("bleu", "chrf++"):
            b = results["baseline"][key][metric]
            ft = results["finetuned"][key][metric]
            w.writerow([key, metric, b, ft, round(ft - b, 2)])

with open(f"{OUT}/probe_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(probe_lines) + "\n")
print("\n".join(probe_lines), flush=True)

model.float().save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
try:  # keep the lang-code file with the model — required at inference
    src = hf_hub_download(repo_id=MODEL_ID, filename="kusaal_lang_codes.json")
    import shutil
    shutil.copy(src, f"{SAVE_DIR}/kusaal_lang_codes.json")
except Exception as e:
    print("lang codes copy skipped:", e)

import shutil
shutil.rmtree(f"{OUT}/ckpt", ignore_errors=True)   # slim the kernel output

print("\n=== SUMMARY (baseline -> finetuned) ===")
for key in results["baseline"]:
    b, ft = results["baseline"][key], results["finetuned"][key]
    print(f"{key}: BLEU {b['bleu']} -> {ft['bleu']}  "
          f"chrF++ {b['chrf++']} -> {ft['chrf++']}")
print("DONE", flush=True)
