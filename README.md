# GTA VI launch-day Twitch forecast

Entry for make.fun's `gta6-twitch-ccu-forecast-01`: forecast how many people
watch Grand Theft Auto VI live on Twitch in each hour (2–24) of its release
day, 2026-11-19 (UTC), given the counts recorded up to the end of hour 1.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads the case's
`comparables` — past big game launches with hourly Twitch category counts —
and builds the forecast from them at run time. For each comparable it takes
the release-day profile as ratios to that launch's own hour-1 print, in log
space, and combines them per hour with a weighted median. The weights encode
two regularities in the data:

1. **Size similarity.** How a launch day unfolds depends strongly on how big
   hour 1 already is. Launches with a huge hour 1 (a global simultaneous
   unlock, or a launch the whole world watched at once) are already near
   their peak and stay flat — Elden Ring's hour-1 print of 875k *was* its
   peak; Cyberpunk 2077 (852k) peaked only 1.26x above hour 1. Launches with
   a small hour 1 can grow several-fold through the day. Each comparable is
   weighted by a Gaussian kernel on log hour-1 size (bandwidth 1.0) around
   the target's hour-1 print.
2. **Unlock pattern.** A rolling midnight-local launch (consoles unlock at
   midnight in each timezone, the Americas hours after Europe and Asia)
   climbs when the Americas come online; a single simultaneous global unlock
   peaks at once. The target's pattern is read from the case's notes, and
   comparables with the same pattern (and a clean, non-early-access launch)
   are boosted 2x, while comparables with the opposite pattern are weighted
   0.4x and early-access-contaminated / weak-tier launches 0.5x — their
   release days are distorted by days of early play.

The spread of each hour comes from the same weighted pool: sigma is 1.4826 ×
the weighted median absolute deviation of that hour's log-ratios, floored at
0.30 (a launch bigger than anything in the pool deserves honest uncertainty)
and capped at 1.20. Quantiles are the median ratio × exp(z × sigma), scaled
by the target's hour-1 print. If a case carries no usable comparables, the
program falls back to the persistence baseline (every hour stays at hour 1's
print, random-walk spread); it never crashes without writing a valid output.

## Validation

Scored locally against SullyGnome's hourly averages (the same source the
practice is graded on — the house baseline reproduces at 0.3871, matching
the published ~0.387):

| case | loss | typical miss | 90% coverage |
|---|---|---|---|
| Marvel's Wolverine | 0.352 | 87% | 0.87 |
| Red Dead Redemption 2 | 0.169 | 31% | 0.96 |
| God of War Ragnarök | 0.144 | 21% | 1.00 |
| Marvel's Spider-Man 2 | 0.196 | 31% | 0.78 |
| **mean** | **0.215** | | |

Leave-one-out over the 37 comparables with an hour-1 print: mean pinball
loss 0.396, median 0.258 (baseline 0.930 / 0.325 on the same set). Runtime
~0.06 s per case, deterministic (byte-identical across runs and hash seeds).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes one `*.out` per case.
