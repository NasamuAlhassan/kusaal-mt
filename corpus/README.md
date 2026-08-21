Kusaal-English Wikipedia Parallel Corpus, v1 (August 2026)
===========================================================

Compiled by Prince Nasamu Alhassan (pnalhassan@gmail.com)

13,659 sentence pairs mined from 1,135 Kusaal Wikipedia articles and their
English Wikipedia counterparts. The largest openly released parallel Kusaal
corpus outside the religious register (politics, biography, geography,
agriculture, science, culture).

FILE: kusaal_wiki_parallel_corpus.csv (UTF-8; opens directly in Excel)

Columns:
  kusaal            the Kusaal sentence (apostrophes normalized to straight ',
                    U+0027; original wiki text mostly used the saltillo U+A78C)
  english           the aligned English sentence
  tier              gold   = matched on shared anchors (dates, figures, names)
                             AND verified by machine translation similarity
                             (8,268 pairs; est. precision ~95%+)
                    silver = verified by machine translation similarity or
                             manual adjudication (5,391 pairs; slightly looser,
                             may include partial correspondences)
  kus_article       source article title on kus.wikipedia.org
  en_article        counterpart article title on en.wikipedia.org
  alignment_origin  which pipeline stage produced the pair
                    (anchor-confident / anchor-unsure / mined / adjudicated)

Method (in brief): all Kusaal Wikipedia articles were collected with their
English counterparts via interlanguage links; sentences were aligned in three
stages - (1) anchor matching on shared numbers, dates, proper names, and
English loanwords, (2) verification by machine translation plus semantic
similarity and character-overlap scoring, (3) manual adjudication of ~500
borderline cases. Aligned pairs are deduplicated and were used to fine-tune
the kusaal-nllb-600M translation model
(https://huggingface.co/PrinceAlhassanNasamu/kusaal-nllb-600M).

Source text license: both Wikipedias publish under CC BY-SA 4.0; this derived
corpus carries the same license, with attribution to the Kusaal and English
Wikipedia contributor communities.

Note: sentence pairs are correspondences between two independently edited
encyclopedias, not certified translations; a small residual error rate
remains, mostly in the silver tier.
