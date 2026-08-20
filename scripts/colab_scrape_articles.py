# ==============================================================
# Kusaal <-> English Wikipedia comparable-article scraper
# Paste this whole script into ONE Google Colab cell and run.
# CPU runtime is fine -- no GPU needed for this step.
#
# Output (on your Google Drive, under MyDrive/kusaal_mt/wiki_pairs/):
#   article_pairs.jsonl      one JSON record per Kusaal-English pair
#   unpaired_kus_titles.txt  Kusaal articles with no English langlink
#   failed_titles.txt        pairs that errored or came back empty
#
# The script is RESUMABLE: if Colab dies mid-run, just run the
# cell again -- already-scraped pairs are skipped.
# ==============================================================

import json
import os
import time

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm.auto import tqdm

from google.colab import drive

drive.mount("/content/drive")

OUT_DIR = "/content/drive/MyDrive/kusaal_mt/wiki_pairs"
os.makedirs(OUT_DIR, exist_ok=True)
PAIRS_PATH = os.path.join(OUT_DIR, "article_pairs.jsonl")
UNPAIRED_PATH = os.path.join(OUT_DIR, "unpaired_kus_titles.txt")
FAILED_PATH = os.path.join(OUT_DIR, "failed_titles.txt")

KUS_API = "https://kus.wikipedia.org/w/api.php"
EN_API = "https://en.wikipedia.org/w/api.php"

# Wikimedia asks for a descriptive User-Agent with contact info.
# Put your email or GitHub URL in here before running.
USER_AGENT = "KusaalMT-corpus-builder/1.0 (Kusaal-English MT research; https://github.com/NasamuAlhassan)"

SLEEP = 0.1  # politeness delay between requests (seconds)

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})
retries = Retry(total=5, backoff_factor=1.5,
                status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))


def api_get(url, **params):
    params.update({"format": "json", "formatversion": 2, "maxlag": 5})
    r = session.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if data.get("error", {}).get("code") == "maxlag":
        time.sleep(5)  # servers busy -- back off and retry once
        r = session.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
    time.sleep(SLEEP)
    return data


def list_kus_articles_with_en_links():
    """Return {kus_title: en_title or None} for every main-namespace,
    non-redirect article on kus.wikipedia.org."""
    pages = {}
    cont = {}
    while True:
        data = api_get(
            KUS_API,
            action="query",
            generator="allpages",
            gapnamespace=0,
            gapfilterredir="nonredirects",
            gaplimit=100,
            prop="langlinks",
            lllang="en",
            lllimit=500,
            **cont,
        )
        for p in data.get("query", {}).get("pages", []):
            pages.setdefault(p["title"], None)
            links = p.get("langlinks")
            if links:
                pages[p["title"]] = links[0]["title"]
        if "continue" not in data:
            break
        cont = data["continue"]
    return pages


def get_extract(api_url, title):
    """Fetch the full plain-text extract of one article.
    Returns (resolved_title, text) or (None, None) if missing/empty.
    Tables, infoboxes and reference lists are stripped by the API,
    which is what we want for sentence alignment."""
    data = api_get(
        api_url,
        action="query",
        prop="extracts",
        explaintext=1,
        exsectionformat="wiki",  # keeps "== Heading ==" markers for later
        redirects=1,
        titles=title,
    )
    plist = data.get("query", {}).get("pages", [])
    if not plist or plist[0].get("missing"):
        return None, None
    text = (plist[0].get("extract") or "").strip()
    if not text:
        return None, None
    return plist[0]["title"], text


# ---------- step 1: enumerate articles + their English links ----------
print("Listing all Kusaal Wikipedia articles and their English links...")
all_pages = list_kus_articles_with_en_links()
paired = {k: v for k, v in all_pages.items() if v}
unpaired = sorted(t for t, v in all_pages.items() if not v)

with open(UNPAIRED_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(unpaired))

print(f"Total Kusaal articles : {len(all_pages)}")
print(f"With an English link  : {len(paired)}")
print(f"Without (skipped)     : {len(unpaired)}  -> {UNPAIRED_PATH}")

# ---------- step 2: resume support ----------
done = set()
if os.path.exists(PAIRS_PATH):
    with open(PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["kus_title"])
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"Resuming: {len(done)} pairs already scraped, skipping those.")

# ---------- step 3: fetch both sides of every pair ----------
n_written = n_failed = 0
with open(PAIRS_PATH, "a", encoding="utf-8") as out, \
     open(FAILED_PATH, "a", encoding="utf-8") as failed:
    for kus_title, en_title in tqdm(sorted(paired.items()), desc="Scraping pairs"):
        if kus_title in done:
            continue
        try:
            _, kus_text = get_extract(KUS_API, kus_title)
            en_resolved, en_text = get_extract(EN_API, en_title)
        except Exception as e:
            failed.write(f"{kus_title}\t{en_title}\t{e}\n")
            n_failed += 1
            continue
        if not kus_text or not en_text:
            failed.write(f"{kus_title}\t{en_title}\tempty extract\n")
            n_failed += 1
            continue
        rec = {
            "kus_title": kus_title,
            "en_title": en_resolved,
            "kus_url": f"https://kus.wikipedia.org/wiki/{kus_title.replace(' ', '_')}",
            "en_url": f"https://en.wikipedia.org/wiki/{en_resolved.replace(' ', '_')}",
            "kus_chars": len(kus_text),
            "en_chars": len(en_text),
            "kus_text": kus_text,
            "en_text": en_text,
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        n_written += 1

print(f"\nDone. New pairs written: {n_written}, failed: {n_failed}")
print(f"Output: {PAIRS_PATH}")

# ---------- step 4: quick summary for review ----------
recs = []
with open(PAIRS_PATH, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        recs.append((r["kus_title"], r["en_title"], r["kus_chars"], r["en_chars"]))

if recs:
    ratios = sorted(en / max(kus, 1) for _, _, kus, en in recs)
    print(f"\nTotal pairs on disk   : {len(recs)}")
    print(f"Median EN/KUS length ratio: {ratios[len(ratios) // 2]:.1f}x")
    print(f"Smallest Kusaal article   : {min(r[2] for r in recs)} chars")
    print(f"Largest English article   : {max(r[3] for r in recs)} chars")
    print("\nSample pairs:")
    for kus_t, en_t, kus_c, en_c in recs[:5]:
        print(f"  {kus_t}  <->  {en_t}   ({kus_c} / {en_c} chars)")
