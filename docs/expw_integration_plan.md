# ExpW integration plan — legitimacy review and 7-class dataset design

Status: **plan only**. Nothing has been downloaded beyond 4.6 MB of label metadata,
no training has been run, and no application code has been touched.

Written against the current state of the project: `processed/fer_plus_v2.npz`
(Phase 6A) and `runs/improved_resnet18/` (Phase 6B, the served checkpoint).

---

## 1. Legitimacy review

### 1.1 Why ExpW at all

The two preferred datasets are closed to anyone but you:

| dataset | blocker | what it would take |
|---|---|---|
| **AffectNet** | A signed *AffectNet Academic Use License* submitted via request form. The site states plainly: "Only PIs, Lab Managers or Professors can request AffectNet+". | You (or your supervisor) sign the PDF and submit the form. Days to weeks. |
| **RAF-DB** | Email application to Shan Li (BUPT) before download; access is password-gated and limited to university-affiliated researchers. | You email and receive a password. Their host also refused HTTPS during this check (`ECONNREFUSED 43.139.48.164:443`). |

Neither can be obtained on your behalf — both require identifying you and
accepting terms in your name. The Kaggle and HuggingFace mirrors of both were
deliberately **not** used: they are third-party re-uploads of data whose owners
gate it behind a signed agreement, so the uploaders had no right to relicense.

### 1.2 What ExpW actually is

Zhang, Luo, Loy & Tang, *From Facial Expression Recognition to Interpersonal
Relation Prediction*, CUHK MMLab (arXiv:1609.06426, IJCV). Read from the paper
text directly:

> "we prepared a list of emotion-related keywords such as 'excited', 'afraid'
> and 'panic'. Then we appended different nouns related to a variety of
> occupations … and used them as queries for **Google image search**."

> "each of the face images was **manually annotated** as one of the seven basic
> expression categories: 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise',
> or 'neutral'."

91,793 labelled faces over 68,096 source images.

### 1.3 What the terms actually say

The shipped `readme.txt` (downloaded and read) contains **only** a label-format
description and a BibTeX entry. There is no licence grant, no restriction, no
agreement, no form, no password. The paper contains no licence, consent or
privacy statement either — searched and confirmed absent.

So the literal answer to "can ExpW be legitimately used" is: **the authors
distribute it openly and ask only to be cited, but they grant no explicit
licence.** Silence is not permission, and it is not prohibition either.

### 1.4 The controlling interpretation

ExpW is not licensed in isolation — it comes from a lab with a published and
consistent convention. CelebA, from the *same lab and two of the same authors*,
collected the same way, states:

> "The CelebA dataset is available for **non-commercial research purposes** only."
> "All images … are obtained from the Internet which are **not property of
> MMLAB**, The Chinese University of Hong Kong."
> "You agree **not to** reproduce, duplicate, copy, sell, trade, resell or
> exploit for any commercial purposes, any portion of the images **and any
> portion of derived data**."
> "You agree **not to** further copy, publish or distribute any portion … Except,
> for internal use at a single site within the same organization."

**Verdict: treat ExpW as if it carried the CelebA terms.** That is the lab's
stated intent for identically-sourced data, and it is the only defensible
reading of an otherwise silent release.

### 1.5 What that permits and forbids, for *this* project

| | verdict |
|---|---|
| Download and keep a local copy for this project | **Yes** — "internal use at a single site" |
| Train a model on it for academic / coursework purposes | **Yes**, with the citation |
| Redistribute the images, or commit them to a repo | **No** |
| Redistribute our 48×48 crops derived from it | **No** — "any portion of derived data" |
| Publish the *trained weights* academically | Grey. Weights are derived data; universal practice says yes, the CelebA wording is broad enough to argue no. Low practical risk for coursework. |
| Ship this in a **commercial** product | **No**, not without legal review. This is the one that matters, because this project is a deployed web service. |

Additional facts you should weigh, not hidden:

- **No consent.** Faces were scraped from Google image search. The subjects are
  identifiable people who never agreed. Under GDPR-style regimes facial data is
  biometric and special-category; a public demo carries more exposure than a
  local project does. No ExpW takedown or retraction was found, but that is the
  broader legal climate this class of dataset sits in.
- **Undocumented annotation quality.** The paper describes a rigorous five-
  annotator consensus process — *for the interpersonal relation labels only*.
  For the expression labels it says merely "manually annotated", with no
  annotator count and no consensus procedure. Treat ExpW expression labels as
  effectively single-annotator. A third party found it necessary to publish a
  "cleaned" re-release, [markson14/ExpWCleaned](https://github.com/markson14/ExpWCleaned),
  on the grounds that the original "is really dirty to some extent."
  Section 5 handles this explicitly.

**Bottom line:** legitimate for this project as academic work, with citation,
no redistribution of images or crops, and a legal review before any commercial
use. If this project is ever going commercial, stop here and get AffectNet or
RAF-DB properly licensed instead.

**Required citation** (put it in the README and any write-up):

```bibtex
@inproceedings{SOCIALRELATION_2017,
  author    = {Zhanpeng Zhang and Ping Luo and Chen Change Loy and Xiaoou Tang},
  title     = {From Facial Expression Recognition to Interpersonal Relation Prediction},
  booktitle = {arXiv:1609.06426v2},
  year      = {2016}
}
```

---

## 2. What the data actually gives us

Counted from the real `label.lst`, not from a paper's table. "≥48px" applies our
existing `SETTINGS.min_face_px` rule to the labelled box short side.

| ExpW class → ours | all faces | ≥48px | ≥48px & det. conf ≥40 | distinct source images |
|---|---|---|---|---|
| angry → **anger** | 3,671 | 3,546 | 2,619 | 3,395 |
| disgust → **disgust** | 3,995 | 3,722 | 2,611 | 3,452 |
| fear → **fear** | 1,088 | 1,045 | 732 | 998 |
| happy → **happiness** | 30,537 | 27,904 | 19,758 | 19,528 |
| sad → **sadness** | 10,559 | 10,046 | 5,916 | 9,383 |
| surprise → **surprise** | 7,060 | 6,697 | 4,905 | 6,283 |
| neutral → **neutral** | 34,883 | 31,698 | 18,310 | 26,690 |
| **total** | **91,793** | **84,658** | **54,851** | 68,096 |

Face box short side: p10 = 48px, median = 96px, p75 = 161px. 12,448 source
images contain more than one labelled face (max 112).

Against the deficit that started all this — 185 disgust and 570 fear in our
current training split — ExpW offers roughly **20× the disgust** and **2× the
fear**, in the wild rather than lab-posed.

---

## 3. Label mapping

Exact and total; no class is merged, split, dropped or invented.

| ExpW id | ExpW name | our class | our index |
|---|---|---|---|
| 0 | angry | anger | 4 |
| 1 | disgust | disgust | 5 |
| 2 | fear | fear | 6 |
| 3 | happy | happiness | 1 |
| 4 | sad | sadness | 3 |
| 5 | surprise | surprise | 2 |
| 6 | neutral | neutral | 0 |

Our index column is `class_names` order from the checkpoint
(`neutral, happiness, surprise, sadness, anger, disgust, fear`) and is asserted
at load time against `processed/fer_plus_v2.npz`, never hardcoded twice.

There is no contempt class in ExpW, so the contempt problem that FER+ forced on
us does not recur. Nothing needs remapping or renormalising.

---

## 4. Preprocessing — identical to the live camera path

This is the part that decides whether the merge helps or hurts. ExpW images are
full-colour web photos at arbitrary resolution; ours are 48×48 grayscale. If we
convert them differently from how the webcam path converts frames, we are
training on a different distribution than we serve.

So ExpW goes through **the same code**, not an equivalent copy of it:

```
ExpW origin .jpg
  └─ ml.detector.FaceDetector.detect()          BlazeFace short-range, conf ≥ 0.70
      └─ match detections to the labelled box by IoU, keep best (IoU ≥ 0.40)
          └─ reject if det.short_side < SETTINGS.min_face_px (48)
              └─ ml.preprocessing.square_face_crop(margin=1.15, offset=0.079)
                  └─ ml.preprocessing.to_fer48()   BT.601 gray, INTER_AREA → 48×48 uint8
                      └─ stored
```

Decisions and why:

- **Re-detect with BlazeFace rather than trusting ExpW's own box.** ExpW boxes
  come from a 2016 detector with different framing conventions. Our calibrated
  crop (margin 1.150, upward offset 0.079) was fitted to *BlazeFace* boxes
  against FER+ framing in Phase 4A.1. Feeding it a foreign box would apply a
  calibration to something it was not calibrated for.
- **The labelled box is used only to choose *which* face.** Up to 112 faces per
  image, each separately labelled — IoU matching is how we attach the right
  label to the right face. IoU < 0.40 means we cannot be sure which face the
  label refers to, so the row is dropped rather than guessed at.
- **Faces BlazeFace misses are dropped, not rescued.** A face our production
  detector cannot find is a face the live camera will never send to the
  classifier. Training on it teaches something we can never use.
- **Expected yield is unknown and must be measured.** ≥48px puts the ceiling at
  84,658 faces. BlazeFace short-range is tuned for near-camera faces, so
  in-the-wild profiles, occlusions and crowds will cost more — a realistic guess
  is 60–80% of the ceiling, but that is a guess and the prepare script reports
  the true number per class before anything is trained.

Everything is derived through `ml/` imports. No preprocessing logic is
reimplemented, so the two paths cannot drift apart later.

---

## 5. Label quality and soft targets

Our trainer consumes soft targets — FER+ gives a genuine 10-annotator
distribution. ExpW gives one label from an undocumented number of annotators
(§1.5), for a dataset with a known "dirty" reputation.

Three options, and the recommendation:

| option | verdict |
|---|---|
| Synthesise a plausible-looking vote distribution for ExpW rows | **Rejected.** That is fabricating annotator agreement that does not exist. |
| Store one-hot and treat identically to FER+ | Honest but overconfident: a single-annotator label would carry more certainty than a unanimous 10-vote FER+ label. |
| **Store one-hot, mark the source, smooth it harder** | **Recommended.** |

Concretely: `soft` = one-hot, plus a `source` array (`ferplus` / `expw`) and a
per-source label-smoothing setting — `eps_ferplus = 0.05` (unchanged),
`eps_expw = 0.10`. This says "we know this label is one person's opinion" in the
only place that can act on it, the loss, without inventing data. Both values are
CLI-exposed so the assumption is testable rather than baked in.

A per-source sample weight γ is also exposed (default 1.0). If ExpW's noise
turns out to hurt, γ < 1 is the dial, and the ablation is one run.

---

## 6. Splits, duplicates and leakage

Three separate hazards, three separate mechanisms.

**6.1 The evaluation yardstick does not move.** `fer_plus_v2.npz`'s val and test
splits stay exactly as they are — byte-identical to `fer_clean.npz`, the same
4,390 + 4,388 images every number in `runs/comparison/` was measured on. **No
ExpW image ever enters them.** Break this and every comparison made so far
becomes meaningless.

**6.2 ExpW gets its own held-out slice**, 90/10, **split by source image name,
not by face row**. Two faces cropped from one photo share lighting, camera and
compression; letting them straddle the split would leak. This slice is the
first honest measurement of disgust we will ever have had: ~370 disgust and
~100 fear images, against the 17 and 86 in the FER+ test split.

**6.3 Cross-dataset duplicate check.** This is not paranoia — FER2013 was itself
assembled from Google image search, and ExpW was assembled from Google image
search. Overlap is plausible. Every prepared ExpW crop is checked against
**every** FER+ row:

- exact: md5 of the final 48×48 crop;
- near: the 64-bit dHash already in `fer_plus_v2.npz`, at Hamming distance ≤ 6
  (equality alone is too strict — two detectors cropping the same face produce
  slightly different pixels);
- any ExpW crop matching a FER+ **val or test** row is dropped outright;
- any matching a FER+ **train** row is dropped as a redundant duplicate;
- the counts are reported per class, and a non-trivial collision count is itself
  a finding worth knowing.

**6.4 Residual risk, stated not hidden.** Identity leakage is *not* solved. The
same person (a celebrity, a stock-photo model) can appear in several unrelated
ExpW photos, and nothing above catches that. Solving it needs face-embedding
clustering — out of scope unless you want it, and flagged in the final report
rather than quietly ignored.

---

## 7. The domain-correlation problem, and how we avoid it

The tempting move — take only ExpW's disgust and fear — is the one that breaks
the model. If almost every ExpW-sourced image is labelled disgust or fear, then
"looks like it came from ExpW" becomes a near-perfect predictor of those labels,
and the network will learn that instead of learning the expression. It would
score well offline and fail on your webcam.

Two defences:

**7.1 Structural.** Every ExpW image is reduced to a 48×48 BT.601 grayscale crop
through the identical BlazeFace + calibrated-crop path as FER+ (§4). Colour,
resolution, background framing and camera signature — most of what distinguishes
the two corpora — are destroyed by that bottleneck on purpose.

**7.2 Compositional.** ExpW contributes to *every* class, capped, not just the
deficient ones. Default cap 3,500 per class:

| class | current train | + ExpW (capped) | merged | ExpW share |
|---|---|---|---|---|
| neutral | 9,506 | 3,500 | 13,006 | 27% |
| happiness | 6,479 | 3,500 | 9,979 | 35% |
| surprise | 2,780 | 3,500 | 6,280 | 56% |
| sadness | 3,151 | 3,500 | 6,651 | 53% |
| anger | 2,216 | 3,500 | 5,716 | 61% |
| **disgust** | **185** | **3,500** | **3,685** | 95% |
| **fear** | **570** | **1,045** | **1,615** | 65% |
| **total** | **24,887** | **22,045** | **46,932** | |

Worst-class imbalance falls from **51:1** (neutral:disgust) to **8:1**
(neutral:fear); disgust goes from 51:1 to 3.5:1. The cap is a CLI argument, and
the counts above are ceilings — §4's detector yield will reduce them, which is
why the prepare script reports real numbers before any training decision.

Disgust stays 95% ExpW-sourced, because 185 FER+ disgust images is all that
exists. That residual correlation cannot be composed away, which is why it gets
measured:

**7.3 The honesty check.** Before training the real model, train a small
classifier to predict *the source dataset* from the 48×48 crop. If it can tell
ExpW from FER+ at well above chance, §7.1 failed and the merged model's disgust
performance cannot be trusted. Cheap (minutes), and it is the difference between
knowing and hoping.

---

## 8. Training and evaluation plan

Deliberately minimal changes to `train_improved.py` — `--dataset`,
`--source-smoothing`, `--source-weight`. The model, schedule, augmentation and
selection rule are untouched, so any change in the result is attributable to the
data.

Runs, all ResNet-18, all selected on FER+ val macro-F1:

| run | data | question |
|---|---|---|
| F0 | current `fer_plus_v2` | control — re-confirms `improved_resnet18` |
| F1 | merged, cap 3,500, `eps_expw=0.10` | does ExpW help? |
| F2 | merged, cap 3,500, `eps_expw=0.05` | is the extra smoothing earning its place? |
| F3 | merged, disgust+fear+anger only | quantifies the domain-shortcut hazard of §7 |
| F4 | merged, cap 3,500, `--sampler-exponent 0.25` | less oversampling needed once real data exists? |

Evaluation, unchanged where it matters:

- **Selection:** FER+ val macro-F1. Same rule, same split, comparable to every
  existing run.
- **Primary report:** FER+ val + test (8,778 images) — accuracy, macro-F1,
  balanced accuracy, per-class P/R/F1, confusion matrix, and the
  expressive-read-as-neutral/happiness rate, all already implemented in
  `compare_models.py`.
- **Rare-class report:** the ExpW holdout. This is where disgust and fear become
  measurable at all. Reported separately and never mixed into the FER+ numbers.
- **Statistical honesty:** the existing paired bootstrap. Disgust moved 17 test
  images last round; a point estimate without an interval is not a result.
- **Real-video check:** `verify_checkpoint.py`, unchanged.

Success is **not** "accuracy went up". It is: disgust and fear recall and F1 up
on the ExpW holdout, the FER+ numbers not regressing, the domain probe near
chance, and the confusion matrix showing less mass leaking into neutral and
happiness.

---

## 9. Files this creates

| path | what | in repo? |
|---|---|---|
| `data/expw_raw/origin.7z.001…008` | the download, ~7.56 GiB (7×1000 MiB + 738 MiB) | **no** — add to `.gitignore` |
| `data/expw_raw/images/` | extracted originals, ~8 GB | **no** — deletable once prepared |
| `data/expw_raw/label.lst` | 4.6 MB label file (already fetched) | no |
| `prepare_expw.py` | §4 + §6 → `processed/expw_v1.npz` | yes |
| `merge_datasets.py` | §7 caps → `processed/fer_expw_v1.npz` | yes |
| `domain_probe.py` | §7.3 check | yes |
| `processed/expw_v1.npz` | 48×48 crops, labels, source image ids, dhash, split | no (derived ExpW data — see §1.5) |
| `processed/fer_expw_v1.npz` | merged training artifact | no |
| `processed/expw_stats.json`, `merge_stats.json` | counts, drops, collisions | yes |

Peak disk ~16 GB during preparation, ~200 MB after cleanup. 355 GB free — fine.
Runtime: download at your link speed, plus ~35–50 min for BlazeFace over 68k
images, plus minutes for merge and probe.

**Nothing in `frontend/`, `backend/`, `ml/` or `configuration/` is modified.**
`ml/` is imported, never edited. The served checkpoint does not change until a
merged model wins on its merits.

---

## 10. What I need from you

1. **The commercial question (§1.5).** If this project stays academic, ExpW is
   fine. If it might be commercialised, say so now — the answer changes to
   "licence AffectNet or RAF-DB properly instead", and there is no point
   spending 7.5 GB and an afternoon first.
2. **Go / no-go on the ~7.56 GiB download.** It cannot be fetched in parts; the
   archive is one monolithic 8-volume 7z.
3. **Optional:** whether you want identity-clustering de-duplication (§6.4). It
   is real work and probably unnecessary at this scale.

Worth doing regardless of the answer: **apply for AffectNet and RAF-DB now.**
Both are free for academics and take days to arrive. AffectNet in particular has
~134k disgust and ~25k fear images with better-documented annotation — strictly
better than ExpW on every axis except availability today.
