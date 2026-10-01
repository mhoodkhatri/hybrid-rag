# Retrieval evaluation

- Corpus: aquatherm_x200_manual.pdf (69 chunks of ~350 tokens)
- Questions: 24 (13 conceptual / 11 exact-term), hand-labeled
- Metrics at K=4. Semantic/BM25 each fetch top 10/10; RRF k=60; reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Eval set SHA-256 (first 16 hex): `0f77fb3173cd67fd`

## All questions

| Mode | Hit@4 | MRR@4 | Hit@4 conceptual | MRR@4 conceptual | Hit@4 exact-term | MRR@4 exact-term |
|---|---|---|---|---|---|---|
| Semantic only | 0.88 | 0.64 | 0.85 | 0.58 | 0.91 | 0.70 |
| BM25 only | 0.67 | 0.53 | 0.62 | 0.40 | 0.73 | 0.68 |
| Hybrid (no rerank) | 0.96 | 0.79 | 1.00 | 0.82 | 0.91 | 0.75 |
| Hybrid + Rerank | 0.92 | 0.79 | 0.85 | 0.76 | 1.00 | 0.83 |

## Independent questions (8)

| Mode | Hit@4 | MRR@4 | Hit@4 conceptual | MRR@4 conceptual | Hit@4 exact-term | MRR@4 exact-term |
|---|---|---|---|---|---|---|
| Semantic only | 0.75 | 0.62 | 0.80 | 0.60 | 0.67 | 0.67 |
| BM25 only | 0.75 | 0.56 | 0.80 | 0.60 | 0.67 | 0.50 |
| Hybrid (no rerank) | 0.88 | 0.81 | 1.00 | 0.90 | 0.67 | 0.67 |
| Hybrid + Rerank | 1.00 | 0.83 | 1.00 | 1.00 | 1.00 | 0.56 |

## Project questions (16)

| Mode | Hit@4 | MRR@4 | Hit@4 conceptual | MRR@4 conceptual | Hit@4 exact-term | MRR@4 exact-term |
|---|---|---|---|---|---|---|
| Semantic only | 0.94 | 0.65 | 0.88 | 0.57 | 1.00 | 0.72 |
| BM25 only | 0.62 | 0.52 | 0.50 | 0.28 | 0.75 | 0.75 |
| Hybrid (no rerank) | 1.00 | 0.78 | 1.00 | 0.77 | 1.00 | 0.78 |
| Hybrid + Rerank | 0.88 | 0.77 | 0.75 | 0.60 | 1.00 | 0.94 |

## Per-question rank of first relevant chunk (– = not in top K)

| # | Author | Type | Question | Semantic only | BM25 only | Hybrid (no rerank) | Hybrid + Rerank |
|---|---|---|---|---|---|---|---|
| 1 | project | conceptual | Why does a dirty outdoor coil make the chiller lose cooling power? | – | – | 3 | – |
| 2 | project | conceptual | What happens if the water pressure at the inlet of the circulating pump gets too low? | 2 | 1 | 1 | 1 |
| 3 | project | conceptual | There is a puddle under the machine but the pipes seem fine. What could it be? | 1 | – | 2 | 1 |
| 4 | project | conceptual | Why is it bad for the compressor to switch on and off very often? | 4 | 2 | 1 | 3 |
| 5 | project | conceptual | How can the temperature readings tell me the system is running low on refrigerant? | 1 | – | 3 | 1 |
| 6 | project | conceptual | Why would the compressor's motor get too hot when there isn't enough refrigerant? | 3 | 2 | 1 | – |
| 7 | project | conceptual | Is it safe to touch the electronics right after turning off the main power switch? | 2 | – | 1 | 2 |
| 8 | project | conceptual | What protective equipment should I wear before hooking up gauges to the cooling circuit? | 1 | 4 | 1 | 1 |
| 9 | project | exact_term | What does error code E-4021 mean? | 2 | – | 4 | 1 |
| 10 | project | exact_term | What does P-4021 indicate? | 4 | 1 | 1 | 1 |
| 11 | project | exact_term | Which model uses fan motor assembly PN-7731-B? | 1 | 1 | 1 | 1 |
| 12 | project | exact_term | What is the maximum water-side pressure of the X200-HD? | 1 | – | 2 | 2 |
| 13 | project | exact_term | Under Clause 7.4, how long do I have to return a failed compressor? | 2 | 1 | 2 | 1 |
| 14 | project | exact_term | Which firmware version is needed for auto restart after a mains failure? | 1 | 1 | 1 | 1 |
| 15 | project | exact_term | How often must task M-05 be done? | 1 | 1 | 1 | 1 |
| 16 | project | exact_term | What is the Modbus baud rate on terminals X6:1 and X6:2? | 2 | 1 | 1 | 1 |
| 17 | independent | conceptual | How long do I need to wait after isolating the unit before I can open the compressor drive cover? | 2 | 1 | 1 | 1 |
| 18 | independent | conceptual | There's a puddle of water around the base frame of the chiller. Is the process loop leaking? | 1 | 2 | 1 | 1 |
| 19 | independent | exact_term | Display is showing P-1001, what does that mean and how do I clear it? | – | – | – | 3 |
| 20 | independent | conceptual | Chiller not keeping up, suction pressure is low, superheat high and subcooling low. What's the likely cause? | 1 | 1 | 1 | 1 |
| 21 | independent | conceptual | Why did a fault that normally auto-resets turn into a lockout I have to acknowledge manually? | 2 | – | 1 | 1 |
| 22 | independent | exact_term | We keep getting E-2105 alarms, what causes that? | 1 | 2 | 1 | 1 |
| 23 | independent | exact_term | How much cooling capacity do we lose installing the X200 at around 1,600 m altitude? | 1 | 1 | 1 | 3 |
| 24 | independent | conceptual | What does short cycling mean on the compressor? | – | 2 | 2 | 1 |

_Small eval set: "project" questions were written and labeled by the project author; "independent" questions were written by a separate agent from randomly drawn pages (seed 2026). Treat differences of one or two questions as noise._
