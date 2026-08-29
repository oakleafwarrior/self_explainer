"""Every arm placed on one horizontal axis, in doublings of training data.

Common-slope ANCOVA:  y = b*x + sum_a c_a * 1[arm=a],  x = log2(N_train)
Offset of arm a relative to E_self|Cfull|R-id is Delta_a = -(c_a - c_ref)/b,
i.e. how far RIGHT arm a's learning curve sits. Positive = needs more data.
Slope is identified mainly by the two self arms (7 sizes); the cross-model arms
(2-3 sizes) contribute intercepts. The shared-slope assumption is tested.
"""
import json
import numpy as np, pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260829)
ROOT = "/Users/aolhava/Desktop/summer26/self_explainer"
M = "exact_match"

core = pd.read_csv(f"{ROOT}/reports/core_sweep.csv")
rand = pd.read_csv(f"{ROOT}/reports/cfull_rand_runs.csv")
ridge = pd.read_csv(f"{ROOT}/reports/ridge_recipe.csv")
ridge = ridge[ridge.n_train.notna() & (ridge.unparseable_rate < 0.5)].copy()

def arm_of(r):
    if r.explainer.endswith("8B"):
        return f"E_self | {r.capacity} | " + ("R-Q" if r.rotation == "Q" else "R-id")
    return f"E_cross 4B | {r.label.split(' ·')[0]}"

d = pd.concat([core, rand, ridge], ignore_index=True)
d["arm"] = d.apply(arm_of, axis=1)
d["x"] = np.log2(d.n_train.astype(float))
d = d[d[M] > 0]

REF = "E_self | Cfull | R-id"
arms = [REF] + sorted(a for a in d.arm.unique() if a != REF)
A = pd.get_dummies(d.arm).reindex(columns=arms).values.astype(float)
X = np.column_stack([d.x.values, A])           # slope + one intercept per arm
y = d[M].values
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta
dof = len(y) - X.shape[1]
cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
b, sb2 = beta[0], cov[0, 0]

# slope homogeneity: allow each arm its own slope, F-test against common slope
Xf = np.column_stack([A, A * d.x.values[:, None]])
bf, *_ = np.linalg.lstsq(Xf, y, rcond=None)
rf = y - Xf @ bf
dof_f = len(y) - np.linalg.matrix_rank(Xf)
F = ((resid @ resid - rf @ rf) / (dof - dof_f)) / (rf @ rf / dof_f)
p_slope = stats.f.sf(F, dof - dof_f, dof_f)

def fieller(u, v, s_uu, s_vv, s_uv, dfree):
    t = stats.t.ppf(0.975, dfree)
    g = t ** 2 * s_vv / v ** 2
    if g >= 1:
        return u / v, [-np.inf, np.inf]
    r = u / v
    disc = s_uu - 2 * r * s_uv + r ** 2 * s_vv - g * (s_uu - s_uv ** 2 / s_vv)
    centre = (r - g * s_uv / s_vv) / (1 - g)
    half = (t / abs(v)) * np.sqrt(max(disc, 0.0)) / (1 - g)
    return r, [centre - half, centre + half]

iref = 1 + arms.index(REF)
rows = []
for a in arms:
    j = 1 + arms.index(a)
    # Delta_a = -(c_a - c_ref)/b ;  U = c_ref - c_a,  V = b
    u = beta[iref] - beta[j]
    s_uu = cov[iref, iref] + cov[j, j] - 2 * cov[iref, j]
    s_uv = cov[iref, 0] - cov[j, 0]
    delta, ci = fieller(u, b, s_uu, sb2, s_uv, dof)
    sub = d[d.arm == a]
    rows.append(dict(arm=a, n_runs=int(len(sub)), n_sizes=int(sub.n_train.nunique()),
                     sizes=sorted(int(v) for v in sub.n_train.unique()),
                     delta_log2=float(delta), ci95=[float(c) for c in ci],
                     data_multiplier=float(2 ** delta),
                     data_multiplier_ci95=[float(2 ** c) for c in ci]))

out = dict(metric=M, reference_arm=REF,
           common_slope_per_doubling=float(b), slope_se=float(np.sqrt(sb2)),
           r2=float(1 - resid @ resid / ((y - y.mean()) ** 2).sum()),
           slope_homogeneity_F=float(F), slope_homogeneity_p=float(p_slope),
           note=("Delta > 0 = arm sits to the RIGHT of E_self|Cfull|R-id, "
                 "i.e. needs more training data for the same score."),
           arms=sorted(rows, key=lambda r: r["delta_log2"]))
with open(f"{ROOT}/reports/horizontal_ladder.json", "w") as fh:
    json.dump(out, fh, indent=2)

print(f"common slope {b:.4f}/doubling (se {np.sqrt(sb2):.4f})  R2={out['r2']:.4f}")
print(f"slope homogeneity across arms: F={F:.2f} p={p_slope:.3f}\n")
print(f"{'arm':40s} {'sizes':>6s} {'runs':>5s}  {'Delta(log2)':>12s}  {'95% CI':>18s}  {'xData':>6s}")
for r in out["arms"]:
    print(f"{r['arm']:40s} {r['n_sizes']:6d} {r['n_runs']:5d}  {r['delta_log2']:+12.3f}  "
          f"[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]  {r['data_multiplier']:6.2f}")


# ---- Does the rotation offset itself depend on the adapter's initialization? ----
# The rotation x init interaction, expressed horizontally. If frame adaptation were a
# representational cost this contrast should be ~0 and the rotation offset stable; if
# it is an optimization-scale effect the offset should wander with the init.
def contrast_delta(name_a_q, name_a_id, name_b_q, name_b_id):
    ia, ja = 1 + arms.index(name_a_q), 1 + arms.index(name_a_id)
    ib, jb = 1 + arms.index(name_b_q), 1 + arms.index(name_b_id)
    w = np.zeros(len(beta))
    w[ia] -= 1; w[ja] += 1        # -(c_aQ - c_aid)
    w[ib] += 1; w[jb] -= 1        # +(c_bQ - c_bid)
    u = w @ beta
    s_uu = w @ cov @ w
    s_uv = w @ cov[:, 0]
    d, ci = fieller(u, b, s_uu, sb2, s_uv, dof)
    return dict(delta=float(d), ci95=[float(c) for c in ci])

pairs = dict(
    rotation_in_Cfull=("E_self | Cfull | R-Q", "E_self | Cfull | R-id"),
    rotation_in_Cfull_rand=("E_self | Cfull-rand | R-Q", "E_self | Cfull-rand | R-id"),
    rotation_in_C0=("E_self | C0 | R-Q", "E_self | C0 | R-id"),
)
within = {}
for k, (q, i) in pairs.items():
    iq, ii = 1 + arms.index(q), 1 + arms.index(i)
    w = np.zeros(len(beta)); w[ii] += 1; w[iq] -= 1
    d, ci = fieller(w @ beta, b, w @ cov @ w, sb2, w @ cov[:, 0], dof)
    within[k] = dict(delta=float(d), ci95=[float(c) for c in ci])

inter = contrast_delta(*pairs["rotation_in_Cfull"], *pairs["rotation_in_Cfull_rand"])
out["rotation_offset_within_arm"] = within
out["rotation_x_init_interaction"] = inter
out["rotation_x_init_note"] = (
    "Difference of the two rotation offsets (Cfull identity-init minus Cfull-rand "
    "orthogonal-init). A CI covering 0 means the rotation's horizontal price is not "
    "distinguishable between initializations, i.e. its sign is not pinned down.")
with open(f"{ROOT}/reports/horizontal_ladder.json", "w") as fh:
    json.dump(out, fh, indent=2)

print("\nrotation offset measured inside each capacity arm:")
for k, v in within.items():
    print(f"  {k:26s} {v['delta']:+.3f}  [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]")
print(f"  interaction (Cfull - Cfull-rand) {inter['delta']:+.3f}  "
      f"[{inter['ci95'][0]:+.3f},{inter['ci95'][1]:+.3f}]")


# ---- The capacity axis, expressed horizontally, for comparison with rotation ----
# C0 -> Cfull is the study's other manipulation. Putting it in the same units as the
# rotation is the only way to say which axis actually moves the reader.
cap = {}
for rot in ("R-id", "R-Q"):
    i0, ifull = 1 + arms.index(f"E_self | C0 | {rot}"), 1 + arms.index(f"E_self | Cfull | {rot}")
    w = np.zeros(len(beta)); w[ifull] += 1; w[i0] -= 1      # -(c_C0 - c_Cfull)/b
    d, ci = fieller(w @ beta, b, w @ cov @ w, sb2, w @ cov[:, 0], dof)
    cap[f"capacity_C0_minus_Cfull_within_{rot}"] = dict(delta=float(d),
                                                        ci95=[float(c) for c in ci])
out["capacity_offset"] = cap
out["capacity_note"] = ("Horizontal cost of dropping the full-rank input map (Cfull -> C0), "
                        "measured inside each rotation condition. Directly comparable to "
                        "rotation_offset_within_arm.")
with open(f"{ROOT}/reports/horizontal_ladder.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("\ncapacity axis (Cfull -> C0), same units:")
for k, v in cap.items():
    print(f"  {k:38s} {v['delta']:+.3f}  [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]")
