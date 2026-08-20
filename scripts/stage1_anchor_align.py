# ==============================================================
# Stage 1: cleaning + anchor-based sentence alignment baseline
#
# Reads  : Scraped Articles/wiki_pairs/article_pairs.jsonl
# Writes : stage1_output/anchor_pairs.csv    confident + unsure pairs, for review
#          stage1_output/unmatched_kus.csv   Kusaal sentences with no anchor match
#          stage1_output/stage1_stats.txt    run summary
#
# No GPU, no model — anchors only (shared numbers, years, proper
# names). The unsure tier and the unmatched file feed Stage 2.
# ==============================================================

import collections
import csv
import json
import os
import re
import unicodedata

IN_PATH = r"C:\Users\Prince\Desktop\MT\Scraped Articles\wiki_pairs\article_pairs.jsonl"
OUT_DIR = r"C:\Users\Prince\Desktop\MT\stage1_output"

# --- cleaning ---------------------------------------------------

# Model training data uses straight apostrophe U+0027 exclusively;
# the wiki mixes saltillo/curly variants. Fold them all down.
APOSTROPHE_MAP = str.maketrans({
    "\uA78C": "'", "\uA78B": "'", "\u02BC": "'", "\u02B9": "'",
    "\u2019": "'", "\u2018": "'", "\u2032": "'",
    "\u201C": '"', "\u201D": '"', "\u00A0": " ",
})

JUNK_SECTIONS = {
    "references", "external links", "see also", "sources", "notes",
    "further reading", "bibliography", "footnotes", "gallery",
    "citations", "works cited",
}

EN_ABBREVS = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "No.", "Jr.", "Sr.",
    "vs.", "etc.", "e.g.", "i.e.", "cf.", "ca.", "approx.", "Rev.",
    "Hon.", "Lt.", "Gen.", "Col.", "Capt.", "Sgt.", "a.m.", "p.m.",
    "Op.", "pp.", "Vol.", "Ltd.", "Inc.", "Co.",
]

DOT = "\x00"  # placeholder protecting non-boundary periods

deglue_count = 0


def normalize(text):
    text = unicodedata.normalize("NFC", text)
    text = text.translate(APOSTROPHE_MAP)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    return text


def deglue_kus(text):
    """Fix stripped-link artifacts like 'DistrictJohn Atta Mills':
    split lowercase->uppercase glue when >=3 lowercase letters precede
    (leaves McDonald/YouTube-style names alone). Kusaal side only."""
    global deglue_count
    fixed, n = re.subn(r"([a-zɛɔʋŋ']{3})([A-ZƐƆƲŊ])", r"\1 \2", text)
    deglue_count += n
    return fixed


def protect_dots(line, lang):
    if lang == "en":
        for ab in EN_ABBREVS:
            line = line.replace(ab, ab.replace(".", DOT))
    line = re.sub(r"\b([A-ZƐƆƲŊ])\.", r"\1" + DOT, line)   # initials: J. J. Rawlings
    line = re.sub(r"(?<=\d)\.(?=\d)", DOT, line)            # decimals: 3.5
    return line


def is_sentence(s):
    if len(s) < 15 or len(s) > 1200:
        return False
    if len(re.findall(r"[^\W\d_]", s)) < len(s) * 0.5:      # mostly letters
        return False
    if len(s.split()) < 3:
        return False
    return s[-1] in ".!?\"')"


def split_sentences(text, lang):
    """Clean one article and return its list of sentences, in order."""
    sents, seen = [], set()
    section = ""
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^=+\s*(.+?)\s*=+$", line)
        if m:
            section = m.group(1).lower()
            continue
        if section in JUNK_SECTIONS:
            continue
        if line[0] in "*-•|#":
            continue
        line = protect_dots(line, lang)
        for part in re.split(r"(?<=[.!?])\s+", line):
            part = part.replace(DOT, ".").strip()
            part = re.sub(r"\s+", " ", part)
            if is_sentence(part) and part not in seen:
                seen.add(part)
                sents.append(part)
    return sents


# --- anchors ----------------------------------------------------

def anchor_tokens(sent):
    """Split a sentence into hard anchors (numbers, capitalized words)
    and soft anchors (lowercase words >= 4 chars). Hard tokens shared
    between a Kusaal and an English sentence are almost always proper
    names, years, or figures; shared soft tokens are English loanwords
    ('security', 'university'), common in Kusaal wiki prose."""
    hard, soft = set(), set()
    for t in re.findall(r"[^\W_]+", sent):
        if any(c.isdigit() for c in t):
            if len(t) >= 2:                 # keep 14, 1960; drop bare 0-9
                hard.add(t)
        elif t[:1].isupper() and len(t) >= 3:
            hard.add(t)
        elif len(t) >= 4:
            soft.add(t)
    return hard, soft


def align_article(kus_sents, en_sents):
    """Anchor-match one article. Returns (rows, unmatched_indices)."""
    kus_hard, kus_soft = zip(*(anchor_tokens(s) for s in kus_sents))
    en_hard, en_soft = zip(*(anchor_tokens(s) for s in en_sents))

    df_kus = collections.Counter(a for s in kus_hard for a in s)
    df_en = collections.Counter(a for s in en_hard for a in s)
    dfs_kus = collections.Counter(a for s in kus_soft for a in s)
    dfs_en = collections.Counter(a for s in en_soft for a in s)

    def score(ki, ei):
        shared = kus_hard[ki] & en_hard[ei]
        shared_soft = kus_soft[ki] & en_soft[ei]
        if not shared and not shared_soft:
            return 0.0, shared, shared_soft
        s = sum(1 / df_kus[a] + 1 / df_en[a] for a in shared)
        s += 0.5 * sum(1 / dfs_kus[a] + 1 / dfs_en[a] for a in shared_soft)
        k_rel = ki / max(len(kus_sents) - 1, 1)
        e_rel = ei / max(len(en_sents) - 1, 1)
        s += 0.3 * (1 - abs(k_rel - e_rel))
        return s, shared, shared_soft

    # best English candidate for every Kusaal sentence (and reverse)
    best_for_en = {}
    scored = []
    for ki in range(len(kus_sents)):
        cands = []
        for ei in range(len(en_sents)):
            s, shared, shared_soft = score(ki, ei)
            if s > 0:
                cands.append((s, ei, shared, shared_soft))
        cands.sort(key=lambda c: (c[0], -c[1]))
        cands.reverse()
        scored.append(cands)
        if cands:
            s, ei = cands[0][0], cands[0][1]
            if s > best_for_en.get(ei, (0,))[0]:
                best_for_en[ei] = (s, ki)

    rows, unmatched = [], []
    for ki, cands in enumerate(scored):
        if not cands:
            unmatched.append(ki)
            continue
        s1, ei, shared, shared_soft = cands[0]
        s2 = cands[1][0] if len(cands) > 1 else 0.0
        margin = s1 - s2
        mutual = best_for_en.get(ei, (0, -1))[1] == ki
        distinctive = any(df_kus[a] <= 2 and df_en[a] <= 2 for a in shared)
        # a lone shared name ("Adam", "Europe") is not enough evidence:
        # confident also needs either 2+ hard anchors or a strong score
        strong = len(shared) >= 2 or s1 >= 2.2
        if distinctive and mutual and strong and (len(cands) == 1 or margin >= 0.5):
            tier = "confident"
        else:
            tier = "unsure"
        anchors = " ".join(sorted(shared))
        if shared_soft:
            anchors += f" (+{len(shared_soft)} loanwords)"
        rows.append({
            "tier": tier,
            "score": round(s1, 3),
            "margin": round(margin, 3),
            "mutual": "yes" if mutual else "no",
            "anchors": anchors.strip(),
            "kus_pos": ki,
            "en_pos": ei,
            "kus_sentence": kus_sents[ki],
            "en_sentence": en_sents[ei],
        })
    return rows, unmatched


# --- main -------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs_csv = os.path.join(OUT_DIR, "anchor_pairs.csv")
    unmatched_csv = os.path.join(OUT_DIR, "unmatched_kus.csv")
    stats_txt = os.path.join(OUT_DIR, "stage1_stats.txt")

    n_articles = 0
    n_kus_sents = n_en_sents = 0
    tier_counts = collections.Counter()
    all_rows, all_unmatched = [], []

    with open(IN_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            kus_text = deglue_kus(normalize(rec["kus_text"]))
            en_text = normalize(rec["en_text"])
            kus_sents = split_sentences(kus_text, "kus")
            en_sents = split_sentences(en_text, "en")
            n_articles += 1
            n_kus_sents += len(kus_sents)
            n_en_sents += len(en_sents)
            if not kus_sents or not en_sents:
                all_unmatched += [(rec["kus_title"], i, s)
                                  for i, s in enumerate(kus_sents)]
                continue
            rows, unmatched = align_article(kus_sents, en_sents)
            for r in rows:
                r["kus_title"] = rec["kus_title"]
                r["en_title"] = rec["en_title"]
                tier_counts[r["tier"]] += 1
            all_rows += rows
            all_unmatched += [(rec["kus_title"], ki, kus_sents[ki])
                              for ki in unmatched]

    cols = ["tier", "score", "margin", "mutual", "anchors",
            "kus_title", "kus_pos", "kus_sentence",
            "en_title", "en_pos", "en_sentence", "verdict"]
    with open(pairs_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in all_rows:
            r["verdict"] = ""
            w.writerow(r)

    with open(unmatched_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["kus_title", "kus_pos", "kus_sentence"])
        w.writerows(all_unmatched)

    lines = [
        f"articles processed        : {n_articles}",
        f"kusaal sentences (clean)  : {n_kus_sents}",
        f"english sentences (clean) : {n_en_sents}",
        f"glued-word fixes applied  : {deglue_count}",
        f"confident pairs           : {tier_counts['confident']}",
        f"unsure pairs              : {tier_counts['unsure']}",
        f"unmatched kusaal sentences: {len(all_unmatched)}",
        f"pairs csv                 : {pairs_csv}",
        f"unmatched csv             : {unmatched_csv}",
    ]
    report = "\n".join(lines)
    with open(stats_txt, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
