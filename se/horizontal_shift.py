"""Horizontal (sample-complexity) offset between the R-id and R-Q learning curves.

Reanalysis of reports/core_sweep.csv + cfull_rand_runs.csv. No new runs.

Convention: y_Q(x) = y_id(x - Delta), with x = log2(N_train).
  Delta > 0 -> R-Q needs MORE data (the unfamiliar frame costs examples)
  Delta < 0 -> R-Q needs LESS data
Data multiplier D = 2**Delta. Cost in examples at size N is N*(D-1).
"""
import json, itertools
import numpy as np, pandas as pd
from scipy import stats, optimize

RNG = np.random.default_rng(20260829)
ROOT = "/Users/aolhava/Desktop/summer26/self_explainer"
METRICS = ["exact_match", "has_changed_f1", "content_match"]

df = pd.concat([pd.read_csv(f"{ROOT}/reports/core_sweep.csv"),
                pd.read_csv(f"{ROOT}/reports/cfull_rand_runs.csv")], ignore_index=True)
df["x"] = np.log2(df.n_train)
df["q"] = (df.rotation == "Q").astype(float)


# ---------- shared-shape curve family -------------------------------------
def shape_fn(kind):
    """f(z; theta): the learning curve both arms are assumed to share."""
    if kind == "loglin":
        return (lambda th, z: th[0] + th[1] * z), [0.0, 0.06]
    if kind == "quad":
        return (lambda th, z: th[0] + th[1] * z + th[2] * z ** 2), [0.0, 0.06, 0.0]
    if kind == "sat":                       # saturating: y_inf - A*2^(-k z)
        return (lambda th, z: th[0] - th[1] * np.exp2(-th[2] * z)), [0.9, 3.0, 0.25]
    raise ValueError(kind)


def fit_model(d, metric, kind="loglin", horiz=True, vert=False):
    """Fit y = f(x - Delta*q) + gamma*q over the chosen shape family.

    horiz/vert toggle which arm effect is free, giving the nested family
    null / vertical-only / horizontal-only / both.
    """
    f, th0 = shape_fn(kind)
    x, q, y = d.x.values, d.q.values, d[metric].values
    nth = len(th0)
    p0 = th0 + ([0.0] if horiz else []) + ([0.0] if vert else [])

    def unpack(p):
        th = p[:nth]
        i = nth
        D = p[i] if horiz else 0.0
        i += int(horiz)
        gm = p[i] if vert else 0.0
        return th, D, gm

    def resid(p):
        th, D, gm = unpack(p)
        return f(th, x - D * q) + gm * q - y

    res = optimize.least_squares(resid, p0, x_scale="jac")
    rss = float(res.fun @ res.fun)
    k = len(p0)
    n = len(y)
    dof = n - k
    cov = (rss / dof) * np.linalg.inv(res.jac.T @ res.jac)
    th, D, gm = unpack(res.x)
    out = dict(kind=kind, horiz=horiz, vert=vert, rss=rss, k=k, dof=dof,
               aicc=n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1),
               r2=1 - rss / float(((y - y.mean()) ** 2).sum()),
               delta=float(D), gamma=float(gm), theta=res.x[:nth].tolist())
    if horiz:
        out["delta_se"] = float(np.sqrt(cov[nth, nth]))
        t = stats.t.ppf(0.975, dof)
        out["delta_ci95"] = [D - t * out["delta_se"], D + t * out["delta_se"]]
    return out


def fieller_loglin(d, metric):
    """Exact ratio CI for Delta in the parallel-lines case (y = a + b x + c q,
    Delta = -c/b). Avoids the delta-method's failure when b is uncertain."""
    X = np.column_stack([np.ones(len(d)), d.x.values, d.q.values])
    y = d[metric].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    dof = len(y) - 3
    cov = (r @ r / dof) * np.linalg.inv(X.T @ X)
    u, v = -beta[2], beta[1]
    s_uu, s_vv, s_uv = cov[2, 2], cov[1, 1], -cov[1, 2]
    t = stats.t.ppf(0.975, dof)
    g = t ** 2 * s_vv / v ** 2
    if g >= 1:
        return dict(delta=u / v, ci95=[-np.inf, np.inf], g=g)
    rat = u / v
    disc = s_uu - 2 * rat * s_uv + rat ** 2 * s_vv - g * (s_uu - s_uv ** 2 / s_vv)
    centre = (rat - g * s_uv / s_vv) / (1 - g)
    half = (t / abs(v)) * np.sqrt(max(disc, 0.0)) / (1 - g)
    return dict(delta=float(rat), ci95=[float(centre - half), float(centre + half)],
                g=float(g), slope=float(v), slope_se=float(np.sqrt(s_vv)),
                vgap=float(beta[2]), vgap_se=float(np.sqrt(s_uu)))


def interaction_test(d, metric):
    """Parallelism check: does R-Q have a different slope in log2 N?"""
    y = d[metric].values
    X0 = np.column_stack([np.ones(len(d)), d.x, d.q])
    X1 = np.column_stack([X0, d.x * d.q])
    rss = []
    for X in (X0, X1):
        bt, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ bt
        rss.append(float(r @ r))
    dof1 = len(y) - 4
    F = (rss[0] - rss[1]) / (rss[1] / dof1)
    return float(F), float(stats.f.sf(F, 1, dof1))


def nonparametric_shift(d, metric):
    """Model-free: invert the R-id cell-mean curve at each R-Q cell mean.
    Delta = x_Q - x_equivalent  (positive => R-Q needs more data)."""
    m = d.groupby(["rotation", "x"])[metric].mean().unstack(0)
    xi, yi = m.index.values, m["identity"].values
    o = np.argsort(yi)
    xs, ys = xi[o], yi[o]
    out = []
    for xq, yq in zip(m.index.values, m["Q"].values):
        if ys[0] <= yq <= ys[-1]:
            xeq, extrap = float(np.interp(yq, ys, xs)), False
        else:
            if yq > ys[-1]:
                s = (xs[-1] - xs[-2]) / (ys[-1] - ys[-2]); xeq = xs[-1] + s * (yq - ys[-1])
            else:
                s = (xs[1] - xs[0]) / (ys[1] - ys[0]); xeq = xs[0] + s * (yq - ys[0])
            extrap = True
        out.append(dict(n_train=int(2 ** xq), delta=float(xq - xeq),
                        equiv_n=float(2 ** xeq), extrapolated=extrap))
    return out


# ---------- uncertainty ---------------------------------------------------
def est_delta(d, metric):
    return fieller_loglin(d, metric)["delta"]


def paired_seed_boot(d, metric):
    """Seed is the unit of randomness and is matched across arms and sizes.
    With 3 seeds all 27 resamples are enumerated exactly."""
    seeds = sorted(d.seed.unique())
    vals = []
    for combo in itertools.product(seeds, repeat=len(seeds)):
        vals.append(est_delta(pd.concat([d[d.seed == s] for s in combo],
                                        ignore_index=True), metric))
    return np.array(vals)


def wild_cluster_boot(d, metric, cluster, B=4000):
    """Rademacher wild bootstrap with weights drawn per cluster. Clustering on
    n_train is the conservative choice: the eval set is shared across every
    cell, so residuals are correlated within a training size."""
    X = np.column_stack([np.ones(len(d)), d.x.values, d.q.values])
    y = d[metric].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted, resid = X @ beta, y - X @ beta
    keys = d[cluster].values
    uniq = np.unique(keys)
    idx = {k: keys == k for k in uniq}
    out = np.empty(B)
    dd = d.copy()
    for i in range(B):
        w = np.empty(len(y))
        for k, draw in zip(uniq, RNG.choice([-1.0, 1.0], size=len(uniq))):
            w[idx[k]] = draw
        dd[metric] = fitted + resid * w
        out[i] = est_delta(dd, metric)
    return out


def per_size_sign_test(np_shifts):
    """Most conservative view: each training size is one exchangeable unit."""
    s = np.array([e["delta"] for e in np_shifts])
    pos = int((s > 0).sum())
    return dict(n_sizes=len(s), n_positive=pos, median=float(np.median(s)),
                sign_test_p=float(stats.binomtest(pos, len(s), 0.5).pvalue))


def boot_ci(v):
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                frac_positive=float((v > 0).mean()), n=int(v.size))


# ---------- assemble ------------------------------------------------------
def analyse(d, metric, tag, full=True):
    fi = fieller_loglin(d, metric)
    F, p = interaction_test(d, metric)
    nps = nonparametric_shift(d, metric)
    r = dict(arm=tag, metric=metric, n_runs=int(len(d)),
             n_sizes=int(d.n_train.nunique()),
             slope_per_doubling=fi["slope"], slope_se=fi["slope_se"],
             mean_vertical_gap_q_minus_id=fi["vgap"],
             vertical_gap_se=fi["vgap_se"],
             delta_log2=fi["delta"], delta_fieller_ci95=fi["ci95"],
             fieller_g=fi["g"], parallel_F=F, parallel_p=p,
             data_multiplier=float(2 ** fi["delta"]),
             data_multiplier_ci95=[float(2 ** c) for c in fi["ci95"]],
             nonparametric_per_size=nps,
             per_size_sign_test=per_size_sign_test(nps))
    if full:
        r["delta_wild_cluster_by_ntrain"] = boot_ci(wild_cluster_boot(d, metric, "n_train"))
        r["delta_wild_cluster_by_seed"] = boot_ci(wild_cluster_boot(d, metric, "seed"))
        if d.groupby(["n_train", "rotation"]).seed.nunique().min() == d.seed.nunique():
            r["delta_paired_seed_boot"] = boot_ci(paired_seed_boot(d, metric))
        # nested model comparison over each shape family
        r["model_comparison"] = {}
        for kind in ("loglin", "quad", "sat"):
            fam = {}
            for name, (h, v) in dict(null=(False, False), vertical_only=(False, True),
                                     horizontal_only=(True, False),
                                     both=(True, True)).items():
                fam[name] = fit_model(d, metric, kind, horiz=h, vert=v)
            best = min(fam, key=lambda k: fam[k]["aicc"])
            r["model_comparison"][kind] = dict(
                best_by_aicc=best,
                delta_aicc_vs_best={k: fam[k]["aicc"] - fam[best]["aicc"] for k in fam},
                rss={k: fam[k]["rss"] for k in fam},
                horizontal_only=fam["horizontal_only"],
                both=fam["both"])
    return r


core = df[(df.capacity == "Cfull") & (df.init == "identity")]
c0 = df[df.capacity == "C0"]
crand = df[df.capacity == "Cfull-rand"]

report = {
    "convention": ("y_Q(x) = y_id(x - Delta) with x = log2(N_train). "
                   "Delta > 0 means R-Q needs MORE examples. "
                   "Data multiplier D = 2**Delta; cost at size N is N*(D-1)."),
    "source": ["reports/core_sweep.csv", "reports/cfull_rand_runs.csv"],
    "results": [],
}
for m in METRICS:
    report["results"].append(analyse(core, m, "E_self | Cfull | identity-init"))
report["results"].append(analyse(c0, "exact_match", "E_self | C0 (paper config)", full=False))
report["results"].append(analyse(crand, "exact_match", "E_self | Cfull-rand (init control)", full=False))

prim = report["results"][0]
lo, hi = prim["delta_fieller_ci95"]
report["example_cost_exact_match"] = [
    dict(n_train=int(n),
         point=float(n * (2 ** prim["delta_log2"] - 1)),
         ci95=[float(n * (2 ** lo - 1)), float(n * (2 ** hi - 1))])
    for n in sorted(core.n_train.unique())]

g = core.groupby(["n_train", "rotation"])["exact_match"].agg(["mean", "std", "count"])
report["vertical_gaps_exact_match"] = [
    dict(n_train=int(n), r_id=float(g.loc[(n, "identity"), "mean"]),
         r_q=float(g.loc[(n, "Q"), "mean"]),
         gap_q_minus_id=float(g.loc[(n, "Q"), "mean"] - g.loc[(n, "identity"), "mean"]),
         sd_id=float(g.loc[(n, "identity"), "std"]), sd_q=float(g.loc[(n, "Q"), "std"]),
         seeds=int(g.loc[(n, "Q"), "count"]))
    for n in sorted(core.n_train.unique())]

with open(f"{ROOT}/reports/horizontal_shift.json", "w") as fh:
    json.dump(report, fh, indent=2, default=float)
print("wrote reports/horizontal_shift.json")
