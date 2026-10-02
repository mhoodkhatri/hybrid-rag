# Retrieval evaluation v2 (with confidence intervals)

- Corpus: aquatherm_x200_manual.pdf (69 chunks of ~350 tokens)
- Answerable questions: 123 (79 conceptual / 44 exact-term; type set by rule: a digit means exact-term)
- Semantic/BM25 each fetch top 10/10; RRF k=60; reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`. Hit/MRR at K=4 (what the LLM sees); Recall at 10.
- Eval set SHA-256: `c7ac549980ee776fbc391ba7b8d5861b6b866facf3af95fb3bde8b08285d3f8a`
- Brackets are bootstrap 95% confidence intervals (10,000 resamples of the questions, seed 2026).

## All answerable questions

| Mode | Hit@1 | Hit@4 | MRR@4 | Recall@10 |
|---|---|---|---|---|
| Semantic only | 0.39 [0.31–0.48] | 0.68 [0.60–0.76] | 0.52 [0.44–0.59] | 0.84 [0.77–0.89] |
| BM25 only | 0.57 [0.48–0.66] | 0.80 [0.72–0.87] | 0.66 [0.58–0.73] | 0.85 [0.78–0.90] |
| Hybrid (no rerank) | 0.54 [0.46–0.63] | 0.81 [0.74–0.88] | 0.66 [0.59–0.73] | 0.91 [0.86–0.95] |
| Hybrid + Rerank | 0.63 [0.54–0.72] | 0.85 [0.79–0.91] | 0.73 [0.66–0.80] | 0.93 [0.89–0.97] |

## Conceptual questions (79)

| Mode | Hit@1 | Hit@4 | MRR@4 | Recall@10 |
|---|---|---|---|---|
| Semantic only | 0.41 [0.30–0.51] | 0.66 [0.56–0.76] | 0.52 [0.42–0.61] | 0.85 [0.77–0.92] |
| BM25 only | 0.57 [0.46–0.68] | 0.76 [0.66–0.85] | 0.64 [0.55–0.74] | 0.84 [0.75–0.91] |
| Hybrid (no rerank) | 0.56 [0.44–0.67] | 0.81 [0.72–0.89] | 0.66 [0.57–0.75] | 0.92 [0.86–0.97] |
| Hybrid + Rerank | 0.67 [0.57–0.77] | 0.82 [0.73–0.90] | 0.73 [0.64–0.82] | 0.94 [0.88–0.98] |

## Exact-term questions (44)

| Mode | Hit@1 | Hit@4 | MRR@4 | Recall@10 |
|---|---|---|---|---|
| Semantic only | 0.36 [0.23–0.50] | 0.73 [0.59–0.86] | 0.52 [0.40–0.64] | 0.82 [0.72–0.91] |
| BM25 only | 0.57 [0.41–0.70] | 0.86 [0.75–0.95] | 0.69 [0.57–0.80] | 0.86 [0.76–0.94] |
| Hybrid (no rerank) | 0.52 [0.39–0.66] | 0.82 [0.70–0.93] | 0.66 [0.54–0.77] | 0.89 [0.80–0.97] |
| Hybrid + Rerank | 0.57 [0.41–0.70] | 0.91 [0.82–0.98] | 0.73 [0.62–0.82] | 0.93 [0.85–0.99] |

## Paired comparisons (same questions, both modes)

Difference = challenger − baseline, with a paired-bootstrap 95% CI. McNemar's exact test uses only the questions where exactly one mode hits; p < 0.05 means the difference is unlikely to be chance. Six McNemar tests are run, so a cautious (Bonferroni) threshold is p < 0.008.

| Baseline → challenger | Metric | Baseline | Challenger | Difference [95% CI] | Only baseline hit / only challenger hit | McNemar p |
|---|---|---|---|---|---|---|
| Semantic only → Hybrid (no rerank) | Hit@1 | 0.39 | 0.54 | +0.15 [+0.07, +0.24] | 6 / 25 | 0.001 |
| Semantic only → Hybrid (no rerank) | Hit@4 | 0.68 | 0.81 | +0.13 [+0.06, +0.20] | 3 / 19 | 0.001 |
| Semantic only → Hybrid (no rerank) | MRR@4 | 0.52 | 0.66 | +0.14 [+0.08, +0.20] | – | – |
| Hybrid (no rerank) → Hybrid + Rerank | Hit@1 | 0.54 | 0.63 | +0.09 [-0.02, +0.20] | 16 / 27 | 0.126 |
| Hybrid (no rerank) → Hybrid + Rerank | Hit@4 | 0.81 | 0.85 | +0.04 [-0.03, +0.12] | 9 / 14 | 0.405 |
| Hybrid (no rerank) → Hybrid + Rerank | MRR@4 | 0.66 | 0.73 | +0.07 [-0.00, +0.15] | – | – |
| Semantic only → Hybrid + Rerank | Hit@1 | 0.39 | 0.63 | +0.24 [+0.15, +0.34] | 8 / 38 | 0.000 |
| Semantic only → Hybrid + Rerank | Hit@4 | 0.68 | 0.85 | +0.17 [+0.10, +0.25] | 3 / 24 | 0.000 |
| Semantic only → Hybrid + Rerank | MRR@4 | 0.52 | 0.73 | +0.21 [+0.14, +0.29] | – | – |

## Per-question rank of first relevant chunk in the top 10 (– = not found)

| # | Author | Type | Question | Pages | Semantic only | BM25 only | Hybrid (no rerank) | Hybrid + Rerank |
|---|---|---|---|---|---|---|---|---|
| 1 | independent-A | conceptual | If our local regulations are stricter than a procedure in this manual, which one do I follow? | 5 | 1 | 1 | 1 | 1 |
| 2 | independent-A | exact_term | What changed in edition v2.1 of the service manual? | 5 | 1 | 1 | 1 | 1 |
| 3 | independent-A | conceptual | Why do I have to give the whole part number, letter at the end included, when I order a spare? | 5 | – | 3 | 5 | 2 |
| 4 | independent-A | conceptual | Which model should we buy if we need to chill a glycol mix below freezing? | 6 | 2 | 5 | 2 | 1 |
| 5 | independent-A | conceptual | What does the electronic expansion valve go by when deciding how much refrigerant to feed into the evaporator? | 6 | 1 | 1 | 1 | 1 |
| 6 | independent-A | conceptual | Over what range of speed does the compressor vary to follow the cooling load? | 7 | 1 | – | 4 | 1 |
| 7 | independent-A | conceptual | Why does the chiller pull more amps but cool less when the outdoor coil is dirty? | 7 | 3 | 1 | 1 | 6 |
| 8 | independent-A | conceptual | How can I tell from the nameplate what year and week the unit was built? | 7 | 1 | 1 | 1 | 1 |
| 9 | independent-A | conceptual | After switching off the isolator, how long do I need to wait before opening up the compressor drive? | 8 | 9 | 1 | 2 | 1 |
| 10 | independent-A | exact_term | On the X200-L, which fuse besides the main isolator must be isolated before touching the compressor terminal box? | 8 | 2 | 1 | 1 | 1 |
| 11 | independent-A | conceptual | How far away from the chiller do sparks and flames have to be kept while the refrigerant circuit is open? | 9 | 5 | 1 | 1 | 1 |
| 12 | independent-A | conceptual | We want to put the chiller inside a closed plant room - what kind of ventilation does the room need? | 9 | 2 | 1 | 1 | 1 |
| 13 | independent-A | conceptual | How much free space do I need along the long sides and above the fans? | 10 | 2 | 1 | 1 | 1 |
| 14 | independent-A | conceptual | What happens if the installer swaps the water inlet and outlet pipes? | 10 | 1 | 1 | 1 | 1 |
| 15 | independent-A | conceptual | The tilt indicator on the crate was triggered - how many days do we have to report it to Norvale? | 10 | 9 | 1 | 4 | 1 |
| 16 | independent-A | conceptual | Our external pump makes a rattling noise and the flow readings jump around - what's causing that? | 11 | 2 | 1 | 2 | 1 |
| 17 | independent-A | conceptual | Which terminals give the volt-free signal to start the external pump? | 11 | 2 | 1 | 2 | 1 |
| 18 | independent-A | conceptual | What is the maximum allowed voltage imbalance between phases on the supply? | 11 | 1 | 1 | 1 | 1 |
| 19 | independent-A | conceptual | Which form do I send back to Norvale after commissioning to start the warranty? | 12 | 8 | 1 | 3 | 6 |
| 20 | independent-A | conceptual | On start-up, how long does the flow switch have to show flow before the compressor is allowed to start? | 13 | 6 | 2 | 2 | 5 |
| 21 | independent-A | exact_term | What does code E-2105 mean and what does the controller do when it's logged? | 13 | 9 | 1 | 2 | 5 |
| 22 | independent-A | conceptual | Our chiller keeps turning on and off because the load is small - what's the usual fix? | 13 | – | 1 | 3 | 3 |
| 23 | independent-A | conceptual | Which firmware version do I need for the chiller to restart by itself after a mains failure? | 14 | 1 | 1 | 1 | 1 |
| 24 | independent-A | exact_term | The display is showing P-4012 - what does it mean? | 14 | – | 1 | 4 | 1 |
| 25 | independent-A | exact_term | What does advisory code P-1001 indicate? | 15 | 1 | 3 | 1 | 2 |
| 26 | independent-A | conceptual | What happens to a warranty claim if the logbook is missing an entry for a maintenance task? | 15 | 4 | 1 | 2 | 1 |
| 27 | independent-A | exact_term | How often is task M-15 due? | 15 | 1 | 1 | 1 | 2 |
| 28 | independent-A | conceptual | Where does the water inlet strainer have to be installed, and how often should it be cleaned? | 10, 15 | 1 | 1 | 1 | 1 |
| 29 | independent-A | conceptual | At what pressures do the electronic high-pressure cutout and the backup mechanical switch open? | 8, 9 | 1 | 1 | 1 | 1 |
| 30 | independent-A | exact_term | What's the maximum water pressure for a standard X200, and what should be fitted if the loop is topped up from the mains? | 6, 9 | 3 | 1 | 2 | 1 |
| 31 | independent-A | conceptual | During commissioning, what needs to be done with the flow switch type setting and what values can it take? | 12, 14 | – | 4 | 9 | 1 |
| 32 | independent-A | conceptual | After the compressor stops, how long does the pump keep running, and how long before the compressor can start again? | 11, 13 | 1 | 1 | 1 | 1 |
| 33 | independent-A | exact_term | On the X200-L, how is the glycol concentration checked and how often? | 11, 15 | 3 | 1 | 1 | 3 |
| 34 | independent-A | conceptual | Water keeps pooling around the base frame - what's the likely cause and how often should that be maintained? | 12, 15 | 10 | 1 | 1 | 1 |
| 35 | independent-A | conceptual | Which firmware behaviour does the latest manual edition cover, and which firmware version is recommended? | 5, 14 | 1 | 1 | 1 | 1 |
| 36 | independent-A | exact_term | On the X200-L, why is the crankcase heater still live with the isolator open, and how long should it be on before the first start? | 8, 12 | 3 | 1 | 1 | 3 |
| 37 | independent-A | exact_term | How many compressors and how many condenser fans does the X200-HD have? | 6, 7 | 2 | 4 | 2 | 1 |
| 38 | independent-B | conceptual | What's the highest water pressure I can use when hosing down the outdoor coil fins? | 16 | – | 1 | 4 | 8 |
| 39 | independent-B | exact_term | During the M-09 flow switch test, which alarm code should show up when I throttle the outlet valve? | 16 | 1 | 1 | 1 | 1 |
| 40 | independent-B | conceptual | The antifreeze strength in the loop has dropped a bit. Can I just add tap water to top it up? | 16 | – | 1 | 3 | 6 |
| 41 | independent-B | exact_term | How often do I need to swap the PN-2208-F cartridge? | 17 | 2 | 2 | 2 | 1 |
| 42 | independent-B | conceptual | The compressor oil sample tested acidic. What has to be done about it? | 17 | 1 | 2 | 1 | 1 |
| 43 | independent-B | exact_term | On the X200-HD, how are the sensors for the second refrigerant circuit labelled on the screen, e.g. the second discharge temperature sensor? | 18 | 9 | 1 | 2 | – |
| 44 | independent-B | conceptual | After I put the water temperature probes back in their wells, the leaving water control started hunting. What did I miss? | 18 | 7 | 5 | 5 | 5 |
| 45 | independent-B | conceptual | What liquid subcooling range should a healthy circuit show at design conditions? | 18 | 2 | 1 | 1 | 1 |
| 46 | independent-B | exact_term | Subcooling is only reading about 2 K. Does that mean the circuit is short of refrigerant, and what should I rule out before concluding that? | 18, 19 | 3 | 2 | 1 | 2 |
| 47 | independent-B | conceptual | At what suction pressure does the controller trip on low pressure, and for how long does it have to stay there? | 19 | 5 | – | 10 | 2 |
| 48 | independent-B | conceptual | The compressor has been without crankcase heater power for most of a day. How long must the heater run before I can start it? | 19 | 1 | 1 | 1 | 1 |
| 49 | independent-B | conceptual | Where do I tell the controller which type of flow switch is fitted, and which firmware version is needed for that? | 19 | 8 | 3 | 2 | 1 |
| 50 | independent-B | conceptual | Why does the outside air temperature sensor have a cover over it, and does it matter if it's left off? | 19 | 2 | – | 7 | 4 |
| 51 | independent-B | exact_term | What superheat setpoint does the expansion valve control to on an X200-L compared with the other models? | 20 | 1 | 1 | 1 | 1 |
| 52 | independent-B | conceptual | We swapped the main controller board and it came with no settings. How do we get our parameters back? | 20 | 8 | 1 | 5 | 4 |
| 53 | independent-B | exact_term | What does code E-4012 point to on the condenser fans? | 20 | 1 | 1 | 1 | 2 |
| 54 | independent-B | exact_term | I need a replacement P3 differential pressure transducer. What part number do I order? | 21 | 1 | 1 | 1 | 1 |
| 55 | independent-B | conceptual | What's the part number for the gasket set that seals the evaporator headers, and which units does it fit? | 21 | 1 | – | 3 | 1 |
| 56 | independent-B | exact_term | Which fan motor assembly in the spares list is the heavy duty one for the X200-HD? | 21 | 5 | 1 | 2 | 1 |
| 57 | independent-B | conceptual | The chiller's screen is completely blank. What should I check first? | 22 | 1 | – | 3 | 7 |
| 58 | independent-B | conceptual | Leaving water is staying warm with the compressor flat out, and the temperature drop across the evaporator is bigger than design. What does that suggest? | 22 | 1 | 1 | 1 | 1 |
| 59 | independent-B | exact_term | E-1002 comes up every time we power the chiller on. What does it mean, and where exactly should the phases be swapped? | 22, 25 | 2 | 2 | 3 | 1 |
| 60 | independent-B | conceptual | Low suction and high superheat but normal subcooling makes me suspect a blocked filter-drier. How can I confirm the drier is restricted while the unit is running? | 22, 17 | 1 | 1 | 1 | 1 |
| 61 | independent-B | exact_term | What condition makes the controller raise E-4210? | 23 | – | 6 | – | 1 |
| 62 | independent-B | conceptual | After a refrigerant repair, how deep a vacuum should be pulled to keep non-condensables out? | 23 | 2 | 3 | 1 | 1 |
| 63 | independent-B | conceptual | Can I cool a tripped compressor down with a hose so the overload resets faster? | 23 | 2 | 5 | 1 | 2 |
| 64 | independent-B | exact_term | Our chiller was built in late 2025 and keeps throwing nuisance flow alarms at start-up even though the strainer is clean. Could the flow switch configuration be wrong? | 19, 23 | 2 | 1 | 1 | 1 |
| 65 | independent-B | conceptual | There's a rhythmic knocking coming from the fan section. What's the likely cause and how should I check the fans? | 23, 16 | 1 | 1 | 1 | 1 |
| 66 | independent-B | conceptual | A flow alarm has gradually started appearing over the last few weeks on an established system. What's the usual cause, and how do I clean it? | 23, 16 | 1 | 1 | 1 | 1 |
| 67 | independent-B | exact_term | We still get afternoon E-2001 trips after cleaning the condenser. How much phase voltage imbalance is enough to cause motor protector trips? | 23, 19 | 3 | 1 | 2 | 1 |
| 68 | independent-B | conceptual | What are the default Modbus address and baud rate for connecting the chiller to our BMS? | 24 | 1 | 1 | 1 | 1 |
| 69 | independent-B | exact_term | Which terminals on the TCU-5 are the Modbus RTU A and B lines? | 24 | 1 | 1 | 1 | 1 |
| 70 | independent-B | conceptual | The building management system keeps reporting timeouts when talking to the chiller. What should I look at? | 24 | 2 | 1 | 1 | 1 |
| 71 | independent-B | conceptual | How many times can a self-resetting fault happen within an hour before it turns into a manual lockout? | 25 | 4 | 3 | 1 | 1 |
| 72 | independent-B | conceptual | What's the difference between alarm codes starting with P and ones starting with E? Does a P code shut the chiller down? | 25 | 1 | 1 | 1 | 1 |
| 73 | independent-B | exact_term | What does E-1010 mean? | 25 | 3 | 3 | 1 | 2 |
| 74 | independent-B | exact_term | We're getting E-1001, and one listed cause is a loose supply terminal. What torque should the incoming supply terminals be tightened to? | 25, 16 | 5 | 1 | 1 | 3 |
| 75 | independent-B | conceptual | Above what supply voltage does the chiller trip on overvoltage? | 26 | 2 | 1 | 1 | 1 |
| 76 | independent-B | conceptual | The compressor drive keeps overheating. What should I clean or check? | 26 | 1 | 1 | 1 | 3 |
| 77 | independent-B | exact_term | If E-2010 is active on our X200-HD, does the whole chiller stop? | 26 | – | 2 | 6 | 2 |
| 78 | independent-B | exact_term | We got E-2001. What does it mean, and what usually causes it to happen? | 26, 19 | 1 | 2 | 1 | 2 |
| 79 | independent-B | conceptual | For a warranty claim, what water quality records should we keep? | 16, 17 | 1 | 1 | 1 | 1 |
| 80 | independent-B | exact_term | The evaporator differential pressure (P3) has been creeping up at the same flow. What does that suggest, and what alarm will it raise if it goes too high? | 16, 19 | 3 | 3 | 4 | 1 |
| 81 | independent-C | exact_term | What does fault E-3104 mean and what part do I fit to fix it? | 27 | 1 | 7 | 2 | 2 |
| 82 | independent-C | conceptual | The chiller keeps saying the oil didn't make it back to the compressor during its cycle. What should I check first? | 27 | 6 | 3 | 1 | 1 |
| 83 | independent-C | exact_term | While E-3106 is active, what does the expansion valve do? | 27 | – | 1 | 6 | 2 |
| 84 | independent-C | conceptual | At what discharge pressure does the electronic high-pressure cutout trip? | 28 | 2 | 2 | 2 | 7 |
| 85 | independent-C | exact_term | For E-3201, what should I check before swapping out the P1 transducer? | 28 | 1 | 1 | 1 | 1 |
| 86 | independent-C | conceptual | Low side pressure stays too low even after the startup period. What are the likely culprits? | 28 | 5 | 8 | 2 | 4 |
| 87 | independent-C | conceptual | How long does the flow switch have to stay open before the chiller flags a loss of water flow? | 29 | 5 | – | 10 | 1 |
| 88 | independent-C | conceptual | How low does subcooling have to be, and for how long, before the low subcooling fault comes up? | 29 | 8 | 3 | 3 | 2 |
| 89 | independent-C | conceptual | We're getting a low subcooling alarm. Can I just add some refrigerant to clear it? | 29 | – | – | – | – |
| 90 | independent-C | exact_term | What leaving water temperature triggers E-4120 freeze protection on an X200-HD? | 29 | 10 | 1 | 2 | 1 |
| 91 | independent-C | conceptual | The controller says the compressor is starting up too often. What should I look at? | 30 | 8 | – | – | – |
| 92 | independent-C | conceptual | If the connection to our building automation system drops out, does the chiller stop cooling? | 30 | 6 | 3 | 5 | 10 |
| 93 | independent-C | exact_term | We got E-5001 on the controller. What's the fix, and what gets replaced if it keeps happening? | 30 | 1 | 3 | 1 | 1 |
| 94 | independent-C | conceptual | After a firmware update the screen says the flow switch type isn't set. Where do I set it and what are the options? | 31 | 1 | 1 | 1 | 1 |
| 95 | independent-C | exact_term | Which firmware upgrade paths cause the P-4021 advisory to appear? | 31 | 2 | 1 | 1 | 1 |
| 96 | independent-C | conceptual | How do I clear an alarm that won't reset on its own, and what access level do I need? | 31 | 3 | 7 | 2 | 1 |
| 97 | independent-C | conceptual | How long does the standard warranty last? | 32 | 1 | 1 | 1 | 1 |
| 98 | independent-C | exact_term | Under Clause 7.2, how soon after commissioning do I have to register the unit to get the longer compressor warranty? | 32 | 1 | 1 | 1 | 1 |
| 99 | independent-C | conceptual | Which form serves as proof that the unit was commissioned? | 32 | – | 10 | – | 1 |
| 100 | independent-C | conceptual | How quickly will a technician come out for a breakdown that's under warranty? | 33 | 1 | 1 | 1 | 1 |
| 101 | independent-C | conceptual | What's the labour rate multiplier for a weekend or holiday callout? | 33 | 1 | 1 | 1 | 1 |
| 102 | independent-C | conceptual | What do I need to send in when filing a warranty claim? | 33 | 2 | 1 | 1 | 2 |
| 103 | independent-C | exact_term | What's the lowest water flow rate allowed for the X200-HD? | 34 | 2 | – | 5 | 1 |
| 104 | independent-C | exact_term | What's the biggest fuse I can put on the supply to a standard X200? | 34 | – | – | – | – |
| 105 | independent-C | exact_term | How heavy is a standard X200 once it's filled and running? We need to check the roof loading. | 34 | 8 | – | – | 8 |
| 106 | independent-C | conceptual | What pH range should the chilled water be kept within? | 35 | 2 | 6 | 1 | 2 |
| 107 | independent-C | exact_term | Our site is at 1,600 m. How much do I need to knock off the rated capacity? | 35 | 1 | 1 | 1 | 1 |
| 108 | independent-C | exact_term | On a really hot day, how much cooling output do we lose for each degree above 35? | 35 | 3 | 3 | 1 | 2 |
| 109 | independent-C | conceptual | What's the point of the heater on the compressor's crankcase? | 36 | 1 | 1 | 1 | 1 |
| 110 | independent-C | conceptual | What does the FS.TYP setting control? | 36 | – | 3 | 7 | 6 |
| 111 | independent-C | exact_term | What does the A2L safety class say about the refrigerant? | 36 | 2 | 2 | 2 | 1 |
| 112 | independent-C | conceptual | How does a thermal-type flow switch tell that water is moving? | 37 | – | – | – | – |
| 113 | independent-C | conceptual | What kind of oil goes in these compressors, and why do I need to keep it away from moisture? | 37 | 5 | – | 8 | 1 |
| 114 | independent-C | conceptual | A temperature sensor reads lower resistance as it warms up. Is that normal for this type of sensor? | 37 | – | – | – | – |
| 115 | independent-C | conceptual | After a compressor is swapped under warranty, how long do I have to send the old one back, and how long must other failed parts be kept? | 32, 33 | 1 | 1 | 1 | 1 |
| 116 | independent-C | exact_term | What's the X200-L's rated cooling capacity, and at what water and glycol conditions is it rated? | 34, 35 | 1 | 1 | 1 | 1 |
| 117 | independent-C | conceptual | What is short cycling, and which advisory warns that it's happening? | 37, 30 | 1 | 1 | 1 | 1 |
| 118 | independent-C | conceptual | At what discharge temperature does the controller start slowing the compressor down, and at what temperature does it trip completely? | 31, 29 | 1 | 3 | 1 | 1 |
| 119 | independent-C | conceptual | The controller clock battery alarm is showing. Which maintenance task covers it, and is a new battery covered under warranty? | 30, 32 | 2 | 2 | 1 | 3 |
| 120 | independent-C | conceptual | We lost evaporator flow. Which part should I inspect first, and if its screen is damaged, is that a warranty item? | 29, 32 | 1 | 8 | 3 | 1 |
| 121 | independent-C | exact_term | What's the max water-side pressure for the X200 versus the X200-HD, and why can the HD take more? | 34, 35 | 1 | – | 6 | 2 |
| 122 | independent-C | conceptual | What's the chloride limit for the chilled water, and are we covered if the evaporator fails from poor water treatment? | 35, 32 | 3 | 2 | 1 | 1 |
| 123 | independent-C | exact_term | How cold can the X200-L's leaving fluid run, and what sets its freeze protection limit? | 34, 29 | 6 | 4 | 8 | 2 |
