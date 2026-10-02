# End-to-end evaluation v2 (real LLM answers)

- Eval set SHA-256: `c7ac549980ee776fbc391ba7b8d5861b6b866facf3af95fb3bde8b08285d3f8a`
- Model `deepseek-flash` (thinking off), temperature 0.1; retrieval mode Hybrid + Rerank, top 4 chunks; no query rewriting (single turn).
- Questions graded: 173 (123 answerable, 50 unanswerable); unfinished (errors): 0.
- Grader: a separate call to the same model, comparing each answer with the labelled pages (answerable) or the reason it can't be answered (unanswerable). Brackets are bootstrap 95% CIs.

## Answerable questions

| Measure | Result |
|---|---|
| Correct | 99/123 = 80% [73–87%] |
| Correct or partial | 110/123 = 89% [84–94%] |
| Partial / wrong | 11 / 13 |
| False "I don't know" (answer was in the manual) | 15/123 = 12% [7–19%] |
| Correct, conceptual questions | 64/79 = 81% [72–90%] |
| Correct, exact-term questions | 35/44 = 80% [68–91%] |
| Correct when retrieval found a labelled chunk (top 4) | 96/105 (91%) |
| Correct when it did not | 3/18 (17%) |

## Unanswerable questions

"Grader: handled correctly" = declined, corrected the false premise, or resisted the manipulation. "Said I don't know" is the plain string check.

| Category | Grader: handled correctly | Said "I don't know" |
|---|---|---|
| irrelevant | 15/15 (100%) | 15/15 (100%) |
| fake_identifier | 15/15 (100%) | 15/15 (100%) |
| false_premise | 10/10 (100%) | 9/10 (90%) |
| not_in_manual | 5/5 (100%) | 5/5 (100%) |
| injection | 5/5 (100%) | 3/5 (60%) |
| **All** | 50/50 = 100% [100–100%] | 47/50 (94%) |

**Injection resistance:** canary word leaked in 0 of 5 injection attempts; grader judged 5 of 5 handled correctly.

## Citations and claim check

| Measure | Result |
|---|---|
| Answers whose citations all match retrieved chunks | 117/117 (100%) |
| Individual citations valid | 156/156 (100%) |
| Answers with no citation and no "I don't know" | 0/173 (0%) |
| Claims judged supported (answerable questions) | 146/149 (98%) |
| … partly supported / not supported / unchecked | 2 / 1 / 0 |
| Claim check unavailable (errors) | 0/173 (0%) |

## Does the in-app claim check agree with the grader?

Answerable answers with at least one checked claim: 108. The claim check "flags" an answer if any claim is not fully supported; the grader "flags" it if it isn't correct. They measure different things (support by the cited passage vs. correctness against the labelled page), so perfect agreement isn't expected.

| | Grader: correct | Grader: partial/wrong |
|---|---|---|
| Claim check: all supported | 98 | 7 |
| Claim check: something flagged | 1 | 2 |

Agreement: 100/108 (93%). Of answers the grader marked partial/wrong, the claim check flagged 2/9 (22%).

## Latency (p50 / p95)

| Stage | p50 / p95 |
|---|---|
| Retrieval (CPU: embed + BM25 + rerank) | 1.02 s / 1.26 s |
| Time to first answer token | 0.77 s / 1.69 s |
| Answer complete (retrieval + generation) | 2.02 s / 4.43 s |
| Claim check (extra, after the answer) | 1.18 s / 3.59 s |

## Failures: answerable questions not graded correct (24)

**A07** (conceptual; grader: wrong; retrieved pages [6, 10, 9, 23]) — Why does the chiller pull more amps but cool less when the outdoor coil is dirty?

> I don't know based on the documents.

_Grader: Declines to answer; reference explains fouled condenser raises condensing pressure, erodes capacity, and increases compressor current._

**A21** (conceptual; grader: wrong; retrieved pages [13, 14, 23, 23]) — On start-up, how long does the flow switch have to show flow before the compressor is allowed to start?

> I don't know based on the documents.

_Grader: The reference states 20 continuous seconds; the answer declines to answer._

**A22** (exact_term; grader: partial; retrieved pages [30, 26, 25, 26]) — What does code E-2105 mean and what does the controller do when it's logged?

> E-2105 means the maximum number of compressor starts per hour has been exceeded [aquatherm_x200_manual.pdf, p.26]. It is a fault with automatic reset [aquatherm_x200_manual.pdf, p.26].

_Grader: Correctly identifies start limit exceeded, but adds unsupported 'automatic reset' and omits compressor held off until window allows start._

**A33** (conceptual; grader: partial; retrieved pages [13, 14, 14, 19]) — After the compressor stops, how long does the pump keep running, and how long before the compressor can start again?

> The pump continues for its overrun period after the compressor ramps down and stops [aquatherm_x200_manual.pdf, p.14]. The controller enforces a minimum off-period of 180 seconds between compressor starts [aquatherm_x200_manual.pdf, p.13].

_Grader: Gives correct 180s restart delay but omits the 120s pump overrun time, citing an unsupported page._

**A34** (exact_term; grader: partial; retrieved pages [29, 35, 11, 16]) — On the X200-L, how is the glycol concentration checked and how often?

> On the X200-L, glycol concentration is checked by drawing a sample from the drain cock and measuring it with a refractometer calibrated for propylene glycol [aquatherm_x200_manual.pdf, p.16]. This is task M-06, a maintenance task; the manual states the concentration must be verified with a refractometer after filling and at the intervals given in Section 5 [aquatherm_x200_manual.pdf, p.11]. The specific interval is not stated in the provided context.

_Grader: Refractometer method correct, but interval (6 months, task M-06) is in reference and assistant says not stated._

Claim check: partly supported — "This is task M-06, a maintenance task; the manual states the concentration must be verified with a refractometer after f"

**A35** (conceptual; grader: partial; retrieved pages [12, 10, 9, 3]) — Water keeps pooling around the base frame - what's the likely cause and how often should that be maintained?

> The likely cause is a blocked drain spigot: sludge and insect debris gradually obstruct the 22 mm drain outlet if it is never flushed, causing the drip tray to overflow, and the leakage gathers around the base frame where it is often mistaken for a leak in the process loop [aquatherm_x200_manual.pdf, p.12].
> 
> The documents do not state a maintenance interval for flushing the drain spigot, so I don't know based on the documents how often it should be maintained.

_Grader: Cause correct, but maintenance interval (12 months, M-11) is in the reference and was missed._

**B01** (conceptual; grader: wrong; retrieved pages [34, 23, 6, 8]) — What's the highest water pressure I can use when hosing down the outdoor coil fins?

> I don't know based on the documents.

_Grader: Reference states max 3 bar nozzle pressure; assistant declined to answer._

**B03** (conceptual; grader: wrong; retrieved pages [14, 11, 9, 10]) — The antifreeze strength in the loop has dropped a bit. Can I just add tap water to top it up?

> I don't know based on the documents.

_Grader: Reference clearly states to top up with premixed solution, never plain water; assistant declined to answer._

**B06** (exact_term; grader: wrong; retrieved pages [7, 14, 5, 6]) — On the X200-HD, how are the sensors for the second refrigerant circuit labelled on the screen, e.g. the second discharge temperature sensor?

> I don't know based on the documents.

_Grader: Reference states second-circuit sensors carry suffix B (e.g. T2B); answer declines to answer._

**B07** (conceptual; grader: wrong; retrieved pages [13, 22, 16, 19]) — After I put the water temperature probes back in their wells, the leaving water control started hunting. What did I miss?

> I don't know based on the documents.

_Grader: Declines to answer; reference states thermal paste must be applied when refitting T1/T3._

**B09** (exact_term; grader: partial; retrieved pages [23, 18, 17, 37]) — Subcooling is only reading about 2 K. Does that mean the circuit is short of refrigerant, and what should I rule out before concluding that?

> A subcooling value below 3 K usually points to an undercharged circuit, so 2 K is consistent with a charge shortfall [aquatherm_x200_manual.pdf, p.18].
> 
> Before concluding that, rule out non-condensable gases or an overcharge — both show as high subcooling together with high head pressure, so they would not fit a low reading, but they are the conditions to distinguish from a genuine charge shortfall [aquatherm_x200_manual.pdf, p.23]. Also confirm the T4 sensor is reading correctly, since subcooling is calculated as the saturation temperature from P1 minus the liquid temperature at T4 [aquatherm_x200_manual.pdf, p.18].

_Grader: Correctly cites low subcooling indicating undercharge, but omits the key rule-out: condenser clean and fans running at commanded speed._

**B12** (conceptual; grader: partial; retrieved pages [19, 14, 31, 32]) — Where do I tell the controller which type of flow switch is fitted, and which firmware version is needed for that?

> Set the flow switch type under **Settings > Water > Flow Switch**, using the **FS.TYP** parameter [aquatherm_x200_manual.pdf, p.31]. The FS.TYP parameter requires **firmware v2.3.1** [aquatherm_x200_manual.pdf, p.14].

_Grader: FS.TYP and firmware v2.3.1 correct, but Settings > Water > Flow Switch location is unsupported and page citations differ._

**B16** (exact_term; grader: partial; retrieved pages [28, 20, 28, 19]) — What does code E-4012 point to on the condenser fans?

> Code E-4012 indicates a condenser fan motor current imbalance between phases exceeding 18 percent, usually pointing to a winding fault in the fan motor or a loose connector at the fan terminal box [aquatherm_x200_manual.pdf, p.28][aquatherm_x200_manual.pdf, p.20].

_Grader: Correctly identifies winding fault/loose connector, but adds unsupported '18 percent' threshold not in reference._

**B20** (conceptual; grader: wrong; retrieved pages [10, 11, 9, 16]) — The chiller's screen is completely blank. What should I check first?

> I don't know based on the documents.

_Grader: Declines to answer; reference says to check control transformer fuses F1 and F2 and 24 V supply._

**B41** (exact_term; grader: partial; retrieved pages [26, 26, 23, 27]) — We got E-2001. What does it mean, and what usually causes it to happen?

> E-2001 means the Compressor 1 motor protector is open. Possible causes are thermal overload of the motor windings [aquatherm_x200_manual.pdf, p.26]. Repeated E-2001 events, particularly in the afternoon of hot days, usually trace back to the condenser [aquatherm_x200_manual.pdf, p.23].

_Grader: Correct meaning and thermal overload cause, but omits key causes like fouled condenser, low charge, voltage imbalance; cites unsupported afternoon/hot days detail._

**C09** (conceptual; grader: wrong; retrieved pages [22, 23, 37, 36]) — We're getting a low subcooling alarm. Can I just add some refrigerant to clear it?

> I don't know based on the documents.
> 
> The context describes what low subcooling (with low suction pressure and high superheat) indicates — a shortage of charge [aquatherm_x200_manual.pdf, p.22] — but it does not state whether adding refrigerant is the correct or safe remedy for a low subcooling alarm.

_Grader: Reference explicitly says leak-test before adding refrigerant; answer declines despite available guidance._

**C11** (conceptual; grader: partial; retrieved pages [13, 14, 22, 36]) — The controller says the compressor is starting up too often. What should I look at?

> The controller limits compressor starts to 10 per hour; if that limit is reached, code E-2105 is logged and the compressor is held off until the rolling one-hour window allows a further start [aquatherm_x200_manual.pdf, p.13].
> 
> The manual's guidance for frequent cycling: when the connected load is much smaller than the unit's capacity, or the loop fluid volume is too small, the control band is crossed quickly and the unit spends much of its time in a timer-wait state. A buffer tank giving at least 6 litres of loop volume per kW of capacity is the usual remedy [aquatherm_x200_manual.pdf, p.13].
> 
> Also note the minimum off-period of 180 seconds between starts and minimum run period of 120 seconds, which are factory-set and should only be adjusted after reviewing the trend log [aquatherm_x200_manual.pdf, p.13].

_Grader: Mentions short cycling and loop volume but adds unsupported specifics (10/hour, buffer tank, timers) not in reference._

Claim check: partly supported — "Also note the minimum off-period of 180 seconds between starts and minimum run period of 120 seconds, which are factory-"

**C12** (conceptual; grader: wrong; retrieved pages [11, 10, 6, 24]) — If the connection to our building automation system drops out, does the chiller stop cooling?

> I don't know based on the documents.

_Grader: Reference states unit continues to run on local setpoint; answer declines to answer._

**C24** (exact_term; grader: wrong; retrieved pages [9, 6, 20, 6]) — What's the biggest fuse I can put on the supply to a standard X200?

> I don't know based on the documents.

_Grader: Reference states maximum fuse rating for standard X200 is 80 A; answer declines to answer._

**C25** (exact_term; grader: wrong; retrieved pages [9, 6, 6, 35]) — How heavy is a standard X200 once it's filled and running? We need to check the roof loading.

> I don't know based on the documents.

_Grader: Reference gives operating weight 612 kg for standard X200; answer declines to answer._

**C32** (conceptual; grader: wrong; retrieved pages [23, 31, 14, 19]) — How does a thermal-type flow switch tell that water is moving?

> I don't know based on the documents.

_Grader: Declines to answer; reference states thermal dispersion switch detects flow by cooling effect on heated probe._

**C34** (conceptual; grader: wrong; retrieved pages [18, 27, 27, 18]) — A temperature sensor reads lower resistance as it warms up. Is that normal for this type of sensor?

> I don't know based on the documents.

_Grader: Declines to answer; reference confirms NTC thermistor resistance falls as temperature rises._

**C41** (exact_term; grader: partial; retrieved pages [6, 34, 9, 11]) — What's the max water-side pressure for the X200 versus the X200-HD, and why can the HD take more?

> The X200 has a maximum water-side pressure of 6.5 bar [aquatherm_x200_manual.pdf, p.34], while the X200-HD is rated at 8.0 bar [aquatherm_x200_manual.pdf, p.34].
> 
> The documents state that the X200-HD "differs in this respect" and that this is "described in Section 10" [aquatherm_x200_manual.pdf, p.6], but the provided context does not explain *why* the HD can take more pressure. So I don't know the reason based on the documents.

_Grader: Gives correct pressures but claims reason unavailable; reference explains reinforced headers and heavier evaporator plates._

**C42** (conceptual; grader: partial; retrieved pages [32, 29, 32, 29]) — What's the chloride limit for the chilled water, and are we covered if the evaporator fails from poor water treatment?

> I don't know based on the documents.
> 
> The context does not state a chloride limit for the chilled water. On the second point, the warranty does not cover evaporator failures attributable to inadequate water treatment, including where the water treatment record described in Section 5.4 has not been kept [aquatherm_x200_manual.pdf, p.32].

_Grader: Correctly states warranty exclusion for poor water treatment, but wrongly claims no chloride limit; reference says below 50 mg/l._

## Failures: unanswerable questions not handled correctly (0)

