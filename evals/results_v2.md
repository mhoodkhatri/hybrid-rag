# Evaluation v2: results

Generated tables: [`retrieval_v2.md`](retrieval_v2.md) (retrieval, all 4 modes, with per-question
ranks) and [`e2e_report.md`](e2e_report.md) (end-to-end, with every failure printed in full).
Raw per-question records: [`e2e_results.jsonl`](e2e_results.jsonl). This page summarises both and
adds the hand spot-check.

## How the test set was made (and kept honest)

- **173 questions, written independently.** Four fresh agents that knew nothing about this
  project or its retrieval methods wrote them. Three each got a disjoint slice of the manual's
  content pages (p.5–15, 16–26, 27–37) and wrote **123 answerable** questions in mixed styles
  (paraphrased, exact code/part, numeric, two-page), each labelled with page(s) and a verbatim
  phrase. A fourth wrote **50 unanswerable** ones: 15 irrelevant, 15 fake codes/parts/clauses,
  10 false premises, 5 on-topic but not in the manual, 5 prompt-injection attempts (each with a
  unique "canary" word that only appears if the model obeys).
- **Validated before any run.** Every one of the 154 page+phrase labels was checked to exist
  verbatim in an indexed chunk (strict mode: each page of a two-page question separately). Every
  invented code was checked to be absent from the manual (several are deliberate near-misses:
  NTS-CF-300 vs the real NTS-CF-200, Clause 7.9 when 7.1–7.8 exist). The only edit: one exact
  duplicate question (A19 = A18) was removed.
- **Frozen.** SHA-256 `c7ac549980ee776fbc391ba7b8d5861b6b866facf3af95fb3bde8b08285d3f8a`
  (in `eval_set_v2.json.sha256`), recorded before the first run; both eval scripts refuse to run
  if the file changes. No retrieval setting was changed during this round.
- Type is set by rule (a digit in the question → exact-term): 79 conceptual, 44 exact-term.
- v1 (`data/eval_set.json`, `eval_results.md`) is untouched.

## 1. Retrieval (free, no LLM): 123 answerable questions

Brackets: bootstrap 95% confidence intervals (10,000 resamples).

| Mode | Hit@1 | Hit@4 | MRR@4 | Recall@10 |
|---|---|---|---|---|
| Semantic only | 0.39 [0.31–0.48] | 0.68 [0.60–0.76] | 0.52 [0.44–0.59] | 0.84 [0.77–0.89] |
| BM25 only | 0.57 [0.48–0.66] | 0.80 [0.72–0.87] | 0.66 [0.58–0.73] | 0.85 [0.78–0.90] |
| Hybrid (no rerank) | 0.54 [0.46–0.63] | 0.81 [0.74–0.88] | 0.66 [0.59–0.73] | 0.91 [0.86–0.95] |
| Hybrid + Rerank | **0.63** [0.54–0.72] | **0.85** [0.79–0.91] | **0.73** [0.66–0.80] | **0.93** [0.89–0.97] |

Paired tests (pre-specified before the run):

| Comparison | Hit@4 difference [95% CI] | McNemar p | Verdict |
|---|---|---|---|
| Semantic → Hybrid | +0.13 [+0.06, +0.20] | 0.001 | **Real improvement** |
| Hybrid → Hybrid + Rerank | +0.04 [−0.03, +0.12] | 0.41 | Not proven (could be noise) |
| Semantic → Hybrid + Rerank | +0.17 [+0.10, +0.25] | <0.001 | **Real improvement** |

(Hit@1 for Hybrid → Rerank: +0.09 [−0.02, +0.20], p = 0.13; also not proven.)

**What this means**
- **Hybrid clearly beats semantic-only search**, and the difference survives the statistics.
- **The reranker's extra gain is not proven.** It's ahead on every metric, but with 123
  questions the intervals include zero.
- **Semantic search did much worse than in v1** (Hit@4 0.68 vs 0.88). v1 was small and partly
  written by the project author; this is the more trustworthy number.
- **BM25 alone was about as good as Hybrid** at Hit@4 (0.80 vs 0.81) and slightly better at
  Hit@1 (0.57 vs 0.54); this comparison was not pre-specified, so it wasn't tested. A likely reason:
  the question writers read the page text, so even "paraphrased" questions share many of its
  words, which favours keyword search. Real users who haven't read the manual may phrase things
  differently. Hybrid's advantage over BM25 shows mainly in Recall@10 (0.91 vs 0.85).

## 2. End-to-end (real DeepSeek answers): all 173 questions, Hybrid + Rerank

About 470 small API calls (answer + claim check + grader per question). One temporary connection
error happened during the run; that question was retried by resuming, with nothing paid twice.

| Measure | Result |
|---|---|
| Answerable: **correct** | **99/123 = 80%** [73–87%] (+11 partial, 13 wrong) |
| Answerable: false "I don't know" | 15/123 = 12% [7–19%] |
| Correct when retrieval found the labelled page (top 4) / when it didn't | 96/105 = 91% / 3/18 = 17% |
| Unanswerable: handled correctly (declined / corrected premise / resisted) | **50/50 = 100%** |
| Injection resistance (canary word leaked) | 0 of 5 leaked; 5/5 handled correctly |
| Citations valid (point at a retrieved chunk) | 156/156 = 100% |
| Answers with no citation and no "I don't know" | 0/173 |
| Claims judged supported by the in-app claim check | 146/149 = 98% |
| Claim check failed to run | 0/173 |
| Latency, answer complete (p50 / p95) | 2.0 s / 4.4 s (+ claim check 1.2 s / 3.6 s) |

By unanswerable category: irrelevant 15/15, fake identifier 15/15, false premise 10/10, not in
manual 5/5, injection 5/5.

**What this means**
- **When it fails, it fails safely.** 13 of the 15 false "I don't know" answers happened because
  retrieval didn't fetch the right page (17% correct when retrieval missed vs 91% when it hit). The
  model refused rather than guessed. **Retrieval, not the LLM, is the main thing to improve.**
- **It never made up an answer** to an unanswerable question, and never obeyed an injection.
- The other 9 non-correct answers were mostly *incomplete* (a missing second part), not false.

## 3. Hand spot-check of the grader

I read 20 graded answers drawn at random (seed 2026: 10 graded correct, 5 graded partial/wrong,
5 unanswerable) and judged them myself against the manual, checking every detail not in the
labelled phrase (e.g. part number PN-5230-D, the 60-month warranty, 41.5 bar cutout).

- **I agreed with the grader on 20 of 20** (correct vs. not correct).
- Two borderline cases: C09 was graded *wrong*; it declined but gave related information, so I'd
  call it partial at most. U38 (spraying water on the compressor) was graded *correct* for saying
  "I don't know"; acceptable, but the ideal answer would also warn that the manual forbids it.

**But I found grader mistakes outside the sample.** Looking at the "partial" answers where
retrieval succeeded, at least two are **wrongly marked down**: B12 ("Settings > Water > Flow
Switch") and B16 ("18 percent" imbalance) state facts that *are* in the manual, on retrieved and
cited pages (p.31, p.28). The grader only sees the *labelled* pages, so it calls correct details
from other pages "unsupported". The in-app claim check marked both as supported, correctly. So the
true accuracy is probably a little higher than 80% (at least 101/123 = 82%). Following the
no-edits-after-results rule, the frozen grades stand.

## Limitations (please read)

- **The grader is the same model family that wrote the answers** (deepseek-flash), so it may
  share blind spots and be lenient toward its own style. The 20/20 spot-check agreement is
  reassuring but small.
- **The grader only sees the labelled pages**, which causes the false "partial"s above.
- **The in-app claim check rarely catches wrong answers** (it flagged 2 of 9 that the grader
  marked partial/wrong). That's expected: most failures are *omissions* or refusals, and the
  claim check only asks "is each cited sentence backed by its passage?", not "is the answer
  complete?".
- **Unanswerable 50/50 may be partly because the questions were easy to refuse.** Most fake
  codes and off-topic questions retrieve nothing relevant, so declining is natural.
- One document, one domain, English only. Question writers saw the page text (see the BM25 note).
- Latency was measured from one home connection, sequentially, on a CPU laptop.
