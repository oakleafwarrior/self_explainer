# Should we retrain on the paper's filtering?

Written 2026-08-21, **substantially revised 2026-08-22** after `stratify_by_position.py` was
re-run at N = 4,096 and 8,192 (those two were blocked on the floor-record repair, not on the
seeds). The revision changed the finding, not just the numbers — see "What the ladder actually
shows". Not committed. Decision doc, not a plan of record.

## The question

Our pool patches at prompt-template tokens (`"or"`, `"else"`, `"and"`, `"with"`, `"Respond"`,
`"nothing"`, `"one"`, `"of"`, `"assistant"`) — 43.0% of rows. The paper's Table 7 reports five
token types (Subject Final, Relation, Orig/Other/Changed Answer Option), all content positions,
and none of ours map to a template token. So the paper appears to have filtered them out and we
did not.

So: rebuild the dataset filtered to the paper's categories and retrain everything?

## What the ladder actually shows

The 2026-08-21 draft of this doc claimed the content-token lift was "significant at every N with
a floor". **That was written with 4,096 and 8,192 missing, and it does not survive them.**

`lift (Cfull R-Q − floor)` on content tokens, `exact_match`:

| N | 256 | 512 | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 |
|---|---|---|---|---|---|---|---|
| lift | +0.070 | +0.091 | +0.065 | +0.048 | **+0.013** | **+0.011** | +0.073 |
| CI excludes 0 | yes | yes | yes | yes | **no** | **no** | yes |

At 4,096 and 8,192 nothing is significant in any bucket on any comparison. From 512 onward the
lift is a **monotone decay to zero**: the activation helps while training data is scarce and
becomes redundant once the model can learn the task from the prompt alone. That is an ordinary
sample-efficiency story and it needs no composition effect to explain it.

The one point that breaks the decay is **N=16,384**, which is also the only single-seed point in
the ladder (`MULTI_SEED_N` excludes it, `se_config.py:113`). Four seeds are in flight as of
2026-08-22 and land ~3 h out. Until they do, the anomaly to be suspicious of is 16,384 — not
4,096/8,192.

**This is not an eval-set artifact.** `bootstrap_paired.py --audit` reports one fingerprint over
the whole tree, so every N is scored on identical held-out items and the ladder is comparable
across N. The `ACT_DATASET_PREFIX` 10,000 → 20,000 split that `--audit` caught earlier
(a2b5913) is not in play here.

### The contrast, which is what the mechanism claim rests on

`delta (R-Q − R-id)` on content tokens: **+0.008 at 4,096, −0.003 at 8,192.** Flat, well inside
the ±0.033 seed band (NB03 per-seed sd ~0.027, pooled). The R-id/R-Q contrast — the thing the
project actually concludes about — is unmoved at the two N where the ladder is now densest.

## What the eval-side fix bought

Restricting *scoring* to content positions was free and is done. It answers "does the activation
carry information at the positions the paper patches" — **yes, but only at low N**, which is a
narrower claim than the 08-21 draft made.

What it does not fix is the training mixture. Every model in the tree was trained on data where
43% of examples carried an activation from a position with nothing to say. That plausibly
teaches a prior that the injected channel is noise, which would suppress the effect under study.

**That is still the only real argument for retraining, and the new ladder sharpens it into
something falsifiable.** The 08-21 framing — "the measured lift is a floor on the true one" —
was a claim about level, and level is hard to read against an unknown true value. The decay
gives a shape instead:

> Unfiltered, the content-token lift decays to zero by N≈4,096. If 43% dead training examples
> are what kills it, filtered training should push that decay point to **higher** N — a small
> DiD at 512 where the effect is already strong, and a large one at 8,192 where it has vanished.

A flat DiD across the ladder means the decay is intrinsic to the task, not to the mixture.

Note this reverses an intuition worth stating explicitly: **8,192 is the sharpest probe point
precisely because the unfiltered lift there is zero.** The DiD is `lift_filtered − 0.011`, so a
real filtered lift appears with no baseline to disentangle it from. At 512 the unfiltered lift
is already +0.091 and there is far less headroom.

## The cost of doing it properly

| item | cost |
|---|---|
| floor + `Cfull` R-id + `Cfull` R-Q, full sweep | ~36 h |
| `C0` ladder, `Cfull-rand`, NB04's 72-run ladder, NB05 | on top |
| `ACT_DATASET_PREFIX` raise + eval-window migration | required — see below |
| every existing run invalidated | the whole tree |

Filtering removes 43% of the pool, so 20,000 raw rows yield 11,394 content+relation. Keeping
N=16,384 would need the prefix raised to ~35,000, which moves the eval window and invalidates
every eval in the tree (the mechanics are in `n100k_scale_plan.md` Appendix A). A full filtered
retrain is effectively restarting the project's compute.

## Recommendation: a targeted probe, ~6.5 h, nothing invalidated

`notebooks/claude_unchecked/NB03b_filtered_training_probe.ipynb` (unreviewed as of 2026-08-22).

Filter the **training** rows only and hold the eval set fixed. Train on the content+relation
rows of the training region; keep eval as the current last 1,024 rows, scored restricted to
content. Filtered and unfiltered arms are then scored on **identical items** and pair directly
against every run already on disk. The notebook asserts that identity rather than trusting it.

Headroom, from `audit_dataset_balance.py` (2026-08-21):

```
content + relation in pool        11,394  (57.0%)
  of which fall in the eval split    577
  available to train on           10,817   -> N_TRAIN up to 8,192 is reachable
```

**Runs:** floor, `Cfull · R-id`, `Cfull · R-Q`, filtered-trained, at **N = 512, 2,048 and
8,192**, one seed each — 9 runs, ~6.5 h. The ladder is the point: it tests the shape prediction
above, which a single N cannot. It is also cheaper than the alternative spend — 10,752 training
examples per arm against 24,576 for three seeds at 8,192 alone.

Writes under `task="patching_filtered"`, a sibling of `patching` in the runs tree, so
`bootstrap_paired.collect` can never confuse the two.

**The floor is retrained on the filtered mixture, not reused.** Reaching 8,192 filtered rows
walks ~14.4k raw rows instead of 8.2k, and held-out leakage rises with depth (+0.124 at 8,192,
+0.148 at 16,384). That shift is common to the floor and the `Cfull` arms and cancels in the
lift — but only if the floor saw the same mixture.

## Reading the probe

The row that decides the retrain is `delta (R-Q − R-id)` on content, not `lift`. Filtered
training could raise both lifts substantially and leave every conclusion intact.

| Result at content positions | Action |
|---|---|
| `delta` DiD inside ±0.033 at every N | **No resweep.** Report the stratified result; record the composition difference as a stated deviation |
| `lift` DiD large, `delta` DiD flat | **No resweep.** Absolute numbers understated; say so and cite the probe's size |
| `delta` DiD outside the band at one N, no trend | **Not yet.** Add `C.seeds_for(n)` there first |
| `delta` DiD outside the band and growing as N falls | **Resweep.** The case the retrain exists for |
| `delta` changes sign under filtering | **Resweep**, whatever the magnitude |
| Any of the above but driven by a floor shift | **No resweep.** Pool-depth leakage; diagnose the floor |

Read the absolute per-arm table before any DiD, for that last row.

## Caveats to keep in the writeup

- **That the paper filtered by position type is an inference**, from Table 7 reporting exactly
  five content categories and none of ours mapping to a template token. The paper does not say
  "we dropped template positions"; G.2 describes the filter as balancing has-changed across
  (token, layer) cells. It is possible template positions fell out of that balancing rather than
  being excluded by design. Write it as "our pool includes patch positions absent from the
  paper's Table 7 taxonomy", not as "the paper excluded template tokens".
- **The content-token lift is a low-N phenomenon.** Any statement that the activation carries
  information at the paper's positions has to carry "at N ≤ 2,048" with it, pending 16,384.
- **Held-out leakage is 0.689 against the paper's 0.814**, so surface leakage is not why the
  floor is high — but our estimate was still rising at the largest fit size and has not
  converged, and the partitions differ (~2,000 cells against their 20). Loose comparison.
- **The train-side filter changes which examples a given N sees**, so filtered N=8,192 and
  unfiltered N=8,192 are not the same data, only the same count. That is the intended contrast,
  but it is a mixture comparison, not an ablation of individual rows.
- **One seed per point on the filtered side.** The probe is sized to decide whether to spend
  36 h, not to be a result in its own right.

## Note on step 2 — `Cfull-rand` seeds

`Cfull-rand` is the init control, and it reverses sign exactly where the identity-init arms do
not:

| N | identity-init (R-Q − R-id) | random-init (R-Q − R-id) |
|---|---|---|
| 512 | +0.017 | **−0.040** |
| 2,048 | +0.017 | **−0.021** |
| 16,384 | +0.026 | **+0.025** |

The low-N R-Q advantage flips under random init — consistent with it being the initialization /
response-bias artifact the paired bootstrap already identified — while the N=16,384 advantage
survives both inits. Every one of those readings rests on a **single seed**, and the 16,384 row
shares the single-seed caveat that now hangs over the lift ladder's one anomalous point.

**Run seeds 68 and 69 at N = 512 and 2,048 only: 8 runs, ~4 h.** That is where the sign flips and
where the claim is currently undefended. Skip the extra `Cfull-rand` seeds at N=16,384 — the two
inits already corroborate each other there, and the four main seeds land first and will say more.

NB03 §3's cell never passed `seed`, so every `Cfull-rand` run went to the default and the three
seed iterations resolved to one directory that `resume` then skipped. Fixed 2026-08-22.

## Order

1. **Four N=16,384 seeds land (~3 h from 2026-08-22).** They decide whether the ladder is a clean
   monotone decay or has a real high-N resurgence. That question is now larger than the filter
   one: a resurgence at 16,384 after decay to zero at 8,192 needs explaining on its own terms.
2. Re-run `bootstrap_paired.py` and `stratify_by_position.py` across all N.
3. `Cfull-rand` seeds at N = 512, 2,048 (~4 h).
4. The filtered-training probe, 9 runs (~6.5 h).
5. Decide on the full retrain from (4), not before.
