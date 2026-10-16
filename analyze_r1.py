import os, sys, json, argparse, itertools
import numpy as np, pandas as pd
from scipy import stats
import analyze as A
import matplotlib.pyplot as plt

ALGOS = ["GLS", "IGLS", "RGLS", "HGLS", "ALNS"]
GLSF = ["GLS", "IGLS", "RGLS", "HGLS"]
CLASSES = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
MARGIN = 1.0          # equivalence margin for paired fuel differences, in percent
NBOOT = 10000
NPERM = 10000
rng = np.random.default_rng(2026)


def load(path):
    d = pd.read_csv(path)
    if "error" in d:
        d = d[d["error"].isna() | (d["error"] == "")]
    for c in ("config", "search_time", "iters", "evals", "accepted", "start_time", "budget"):
        if c not in d:
            d[c] = np.nan
    d["config"] = d["config"].fillna("fuel_free")
    d["class"] = d["instance"].map(A.cls)
    return d


def phi_tuple(txt):
    from ecvrptw import PHI_BASE, cmem_phi
    txt = str(txt).strip("() ")
    if txt == "base":
        return PHI_BASE
    if txt == "cmem":
        return cmem_phi()
    return tuple(float(v) for v in txt.split(","))


def aggregate(d):
    g = (d.groupby(["instance", "class", "algo"])
           .agg(nv=("nv", "mean"), dist=("dist", "mean"), fuel=("fuel", "mean"),
                emis=("emis", "mean"), IF=("IF", "mean"), plow=("plow", "mean"),
                pmed=("pmed", "mean"), phigh=("phigh", "mean"), lam=("lam", "mean"),
                search_time=("search_time", "mean"), iters=("iters", "mean"),
                evals=("evals", "mean"), accepted=("accepted", "mean"),
                start_time=("start_time", "mean"), runs=("fuel", "size"))
           .reset_index())
    best = g.groupby("instance")["fuel"].transform("min")
    g["RelGapZ"] = 100 * (g["fuel"] - best) / best
    return g


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    prev = 0.0
    for r, i in enumerate(order):
        prev = max(prev, min(1.0, (len(p) - r) * p[i]))
        adj[i] = prev
    return adj


def wilcox(x, alternative="two-sided"):
    x = np.asarray(x, float)
    if np.allclose(x, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(x, alternative=alternative).pvalue)
    except ValueError:
        return 1.0


def boot_mean_ci(x):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(NBOOT, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pairwise(g, algos, col="fuel"):
    # paired percent differences a vs b per instance, with CI, power and equivalence
    piv = g.pivot(index="instance", columns="algo", values=col).dropna()
    rows = []
    for a, b in itertools.combinations(algos, 2):
        d = (100 * (piv[a] - piv[b]) / piv[b]).values
        lo, hi = boot_mean_ci(d)
        sd = float(np.std(d, ddof=1))
        mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * sd / np.sqrt(len(d)) * np.sqrt(np.pi / 3)
        p_lo = wilcox(d + MARGIN, "greater")
        p_hi = wilcox(d - MARGIN, "less")
        rows.append(dict(a=a, b=b, n=len(d), mean_pct=float(d.mean()), median_pct=float(np.median(d)),
                         ci_lo=lo, ci_hi=hi, sd_pct=sd, wins_a=int((d < -1e-9).sum()),
                         ties=int((np.abs(d) <= 1e-9).sum()), wins_b=int((d > 1e-9).sum()),
                         p=wilcox(d), mde_pct=float(mde), p_tost=max(p_lo, p_hi)))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].values)
    out["p_tost_holm"] = holm(out["p_tost"].values)
    out["equivalent_at_margin"] = out["p_tost_holm"] < 0.05
    return out


def friedman(g, algos, col):
    piv = g.pivot(index="instance", columns="algo", values=col).dropna()[algos]
    try:
        chi, p = stats.friedmanchisquare(*[piv[a].values for a in algos])
    except ValueError:
        chi, p = float("nan"), 1.0
    return dict(outcome=col, methods=len(algos), n=len(piv), chi2=float(chi),
                df=len(algos) - 1, p=float(p),
                **{"rank_" + k: float(v) for k, v in piv.rank(axis=1).mean().items()})


def signature(g, algos=ALGOS):
    d = g[g.algo.isin(algos)].copy()
    for c in ("dist", "IF", "fuel", "plow", "nv"):
        mean = d.groupby("instance")[c].transform("mean")
        d["d_" + c] = np.where(mean != 0, 100 * (d[c] - mean) / mean, 0.0)
    rows = []
    for a in algos:
        s = d[d.algo == a]
        row = dict(algo=a)
        for c in ("dist", "IF", "fuel", "plow", "nv"):
            v = s["d_" + c].values
            row[c] = float(v.mean())
            row[c + "_se"] = float(np.std(v, ddof=1) / np.sqrt(len(v)))
            row[c + "_p"] = wilcox(v)
        rows.append(row)
    return pd.DataFrame(rows), d


def coupling(dev):
    # correlation of distance and intensity deviations with the instance as the unit:
    # permutation of method labels within instances, bootstrap over instances
    piv_d = dev.pivot(index="instance", columns="algo", values="d_dist")
    piv_i = dev.pivot(index="instance", columns="algo", values="d_IF")
    X, Y = piv_d.values, piv_i.values
    r_obs = float(np.corrcoef(X.ravel(), Y.ravel())[0, 1])
    cnt = 0
    for _ in range(NPERM):
        Yp = np.array([row[rng.permutation(len(row))] for row in Y])
        if abs(np.corrcoef(X.ravel(), Yp.ravel())[0, 1]) >= abs(r_obs):
            cnt += 1
    boots = []
    n = X.shape[0]
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        boots.append(np.corrcoef(X[idx].ravel(), Y[idx].ravel())[0, 1])
    return dict(r=r_obs, n_obs=int(X.size), n_instances=int(n),
                p_perm=(cnt + 1) / (NPERM + 1),
                ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)))


def centroids(g):
    c = g.groupby(["class", "algo"])[["IF", "phigh", "plow", "lam"]].mean().reset_index()
    out = dict(n=len(c))
    for k in ("phigh", "plow", "lam"):
        r = stats.spearmanr(c[k], c["IF"])
        out["rho_" + k] = float(r.statistic)
        out["p_" + k] = float(r.pvalue)
    X = np.column_stack([np.ones(len(c)), c["phigh"], c["lam"]])
    beta, *_ = np.linalg.lstsq(X, c["IF"].values, rcond=None)
    pred = X @ beta
    out["beta"] = [float(b) for b in beta]
    out["r2"] = float(1 - ((c["IF"] - pred) ** 2).sum() / ((c["IF"] - c["IF"].mean()) ** 2).sum())
    return out, c


def table3(g, algos=ALGOS):
    rows = []
    for a in algos:
        s = g[g.algo == a]
        rows.append(dict(algo=a, NV=s["nv"].median(), D=s["dist"].median(), Z=s["fuel"].median(),
                         E=s["emis"].median(), IF=s["IF"].median(), RelGapZ=s["RelGapZ"].median(),
                         plow=s["plow"].mean(), pmed=s["pmed"].mean(), phigh=s["phigh"].mean(),
                         lam=s["lam"].median(), search_s=s["search_time"].median(),
                         iters=s["iters"].median(), evals=s["evals"].median(),
                         accepted=s["accepted"].median()))
    t = pd.DataFrame(rows)
    t["share_sum"] = t["plow"] + t["pmed"] + t["phigh"]
    return t


def table4(g):
    t = (g.groupby(["class", "algo"])
          .agg(n=("instance", "nunique"), NV=("nv", "median"), D=("dist", "median"),
               Z=("fuel", "median"), IF=("IF", "median"), RelGapZ=("RelGapZ", "median"),
               plow=("plow", "mean"), phigh=("phigh", "mean"), iters=("iters", "median"))
          .reset_index())
    return t


def effort(d):
    e = (d.groupby(["class", "algo"])
          .agg(runs=("iters", "size"), zero_iter=("iters", lambda s: int((s == 0).sum())),
               iters=("iters", "median"), evals=("evals", "median"),
               accepted=("accepted", "median"), search_s=("search_time", "median"),
               search_max=("search_time", "max"), start_s=("start_time", "median"))
          .reset_index())
    return e


def anytime(path, out):
    d = load(path)
    rows = []
    for b, sub in d.groupby("budget"):
        g = aggregate(sub)
        f = friedman(g, ALGOS, "fuel")
        f.update(budget=b, zero_iter_runs=int((sub["iters"] == 0).sum()),
                 runs=len(sub), search_max=float(sub["search_time"].max()))
        for a in ALGOS:
            f["iters_" + a] = float(sub[sub.algo == a]["iters"].median())
            f["fuel_" + a] = float(g[g.algo == a]["fuel"].mean())
        rows.append(f)
    t = pd.DataFrame(rows).sort_values("budget")
    t.to_csv(os.path.join(out, "table6_budget.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for a in ALGOS:
        ax[0].plot(t["budget"], t["fuel_" + a], marker=A.MK[a], color=A.COL[a], lw=1, ms=4, label=a)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Search budget per run (s)")
    ax[0].set_ylabel("Mean fuel (L)")
    ax[0].legend(frameon=False, fontsize=7)
    ax[1].plot(t["budget"], t["p"], marker="o", color="black", lw=1, ms=4)
    ax[1].axhline(0.05, ls="--", lw=0.7, color="#777777")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Search budget per run (s)")
    ax[1].set_ylabel("Friedman p-value, fuel")
    fig.savefig(os.path.join(out, "fig6_budget.png"))
    plt.close(fig)
    return t


def sensitivity(path, out):
    d = load(path)
    d["spread"] = d["phi"].map(lambda s: round(phi_tuple(s)[2] - phi_tuple(s)[0], 3))
    g = (d.groupby(["spread", "instance", "algo"])[["fuel", "dist", "IF", "plow", "phigh"]]
          .mean().reset_index())
    t = g.groupby(["spread", "algo"])[["dist", "fuel", "IF", "plow", "phigh"]].mean().reset_index()
    pv = []
    for sp, sub in g.groupby("spread"):
        piv = sub.pivot(index="instance", columns="algo", values="fuel").dropna()
        algs = list(piv.columns)
        if len(algs) == 2:
            dd = 100 * (piv[algs[0]] - piv[algs[1]]) / piv[algs[1]]
            pv.append(dict(spread=sp, contrast="%s vs %s" % tuple(algs),
                           mean_pct=float(dd.mean()), p=wilcox(dd.values)))
    t.to_csv(os.path.join(out, "table7_sensitivity.csv"), index=False)
    pd.DataFrame(pv).to_csv(os.path.join(out, "table7_sensitivity_tests.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for a in sorted(t["algo"].unique()):
        s = t[t.algo == a].sort_values("spread")
        ax[0].plot(s["spread"], s["IF"], marker=A.MK.get(a, "o"), color=A.COL.get(a, "#555"), lw=1, ms=4, label=a)
        ax[1].plot(s["spread"], s["plow"], marker=A.MK.get(a, "o"), color=A.COL.get(a, "#555"), lw=1, ms=4, label=a)
    ax[0].set_xlabel("Speed-factor spread $\\phi_{high}-\\phi_{low}$")
    ax[0].set_ylabel("Fuel intensity (L/km)")
    ax[1].set_xlabel("Speed-factor spread $\\phi_{high}-\\phi_{low}$")
    ax[1].set_ylabel("Distance at low speed (%)")
    ax[0].legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(out, "fig7_sensitivity.png"))
    plt.close(fig)
    return t, pd.DataFrame(pv)


def ablation(path, out):
    d = load(path)
    rows, tests = [], []
    g = (d.groupby(["config", "instance", "class"])[["nv", "dist", "fuel", "IF", "plow"]]
          .mean().reset_index())
    for c, s in g.groupby("config"):
        rows.append(dict(config=c, n=len(s), NV=s["nv"].mean(), D=s["dist"].median(),
                         Z=s["fuel"].median(), IF=s["IF"].median(), plow=s["plow"].mean()))
    for a, b, label in (("fuel_fixed", "dist_fixed", "load-aware routing"),
                        ("fuel_free", "fuel_fixed", "speed selection"),
                        ("fuel_free", "dist_fixed", "both")):
        pa = g[g.config == a].set_index("instance")
        pb = g[g.config == b].set_index("instance")
        idx = pa.index.intersection(pb.index)
        if len(idx) == 0:
            continue
        for col in ("fuel", "dist", "IF"):
            dd = (100 * (pa.loc[idx, col] - pb.loc[idx, col]) / pb.loc[idx, col]).values
            lo, hi = boot_mean_ci(dd)
            tests.append(dict(effect=label, a=a, b=b, outcome=col, n=len(idx), mean_pct=float(dd.mean()),
                              median_pct=float(np.median(dd)), ci_lo=lo, ci_hi=hi, p=wilcox(dd)))
        nvd = pa.loc[idx, "nv"] - pb.loc[idx, "nv"]
        tests.append(dict(effect=label, a=a, b=b, outcome="nv", n=len(idx), mean_pct=float(nvd.mean()),
                          median_pct=float(np.median(nvd)), ci_lo=np.nan, ci_hi=np.nan,
                          p=wilcox(nvd.values)))
    t, tt = pd.DataFrame(rows), pd.DataFrame(tests)
    t.to_csv(os.path.join(out, "table_ablation.csv"), index=False)
    tt.to_csv(os.path.join(out, "table_ablation_tests.csv"), index=False)
    return t, tt


def comparison(path, out, tag):
    # the full set of main-comparison statistics for one results file
    d = load(path)
    g = aggregate(d)
    res = dict(file=os.path.basename(path), runs=len(d), instances=int(g.instance.nunique()),
               zero_iter_runs=int((d["iters"] == 0).sum()),
               search_time_max=float(d["search_time"].max()) if d["search_time"].notna().any() else None)
    fr = [friedman(g, ALGOS, c) for c in ("fuel", "dist", "IF", "plow", "nv")]
    fr += [friedman(g, GLSF, c) for c in ("fuel", "dist", "IF", "plow", "nv")]
    pd.DataFrame(fr).to_csv(os.path.join(out, tag + "_friedman.csv"), index=False)
    pw = pd.concat([pairwise(g, GLSF).assign(family="GLS designs"),
                    pd.concat([pairwise(g, [a, "ALNS"]) for a in GLSF]).assign(family="ALNS-Eco vs GLS")])
    pw["p_holm"] = np.nan
    pw["p_tost_holm"] = np.nan
    for fam, idx in pw.groupby("family").groups.items():
        pw.loc[idx, "p_holm"] = holm(pw.loc[idx, "p"].values)
        pw.loc[idx, "p_tost_holm"] = holm(pw.loc[idx, "p_tost"].values)
    pw["equivalent_at_margin"] = pw["p_tost_holm"] < 0.05
    pw.to_csv(os.path.join(out, tag + "_pairwise_fuel.csv"), index=False)
    sig, dev = signature(g)
    sig.to_csv(os.path.join(out, tag + "_signature.csv"), index=False)
    res["coupling"] = coupling(dev)
    cen, cdf = centroids(g)
    res["centroids"] = cen
    cdf.to_csv(os.path.join(out, tag + "_centroids.csv"), index=False)
    table3(g).to_csv(os.path.join(out, tag + "_table3.csv"), index=False)
    table4(g).to_csv(os.path.join(out, tag + "_table4.csv"), index=False)
    effort(d).to_csv(os.path.join(out, tag + "_effort.csv"), index=False)
    g.to_csv(os.path.join(out, tag + "_per_instance.csv"), index=False)
    wins = g.loc[g.groupby("instance")["fuel"].idxmin(), "algo"].value_counts().to_dict()
    res["wins"] = {k: int(v) for k, v in wins.items()}
    res["equal_fleet_instances"] = int((g.groupby("instance")["nv"].nunique() == 1).sum())
    res["friedman"] = fr
    return res, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/r1_main.csv")
    ap.add_argument("--anytime", default="results/r1_anytime.csv")
    ap.add_argument("--sens", default="results/r1_sens.csv")
    ap.add_argument("--cmem", default="results/r1_cmem.csv")
    ap.add_argument("--ablation", default="results/r1_ablation.csv")
    ap.add_argument("--out", default="results/analysis_r1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = {}

    res, g = comparison(args.main, args.out, "main")
    summary["main"] = res
    A.fig_iso(g, os.path.join(args.out, "fig3_iso.png"))
    A.fig_gaps(g, os.path.join(args.out, "fig4_gaps.png"))
    A.fig_signature(g, os.path.join(args.out, "fig5_signature.png"))

    if os.path.exists(args.anytime):
        summary["budget"] = anytime(args.anytime, args.out).to_dict("records")
    if os.path.exists(args.sens):
        t, pv = sensitivity(args.sens, args.out)
        summary["sensitivity_tests"] = pv.to_dict("records")
    if os.path.exists(args.cmem):
        summary["cmem"], _ = comparison(args.cmem, args.out, "cmem")
    if os.path.exists(args.ablation):
        t, tt = ablation(args.ablation, args.out)
        summary["ablation"] = t.to_dict("records")
        summary["ablation_tests"] = tt.to_dict("records")

    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1, default=float)
    m = summary["main"]
    print("main: %d runs, %d zero-iteration runs, max search time %s" %
          (m["runs"], m["zero_iter_runs"], m["search_time_max"]))
    for f in m["friedman"]:
        print("  Friedman %-4s over %d methods: chi2=%.2f p=%.4g" % (f["outcome"], f["methods"], f["chi2"], f["p"]))
    c = m["coupling"]
    print("  distance-intensity coupling r=%.3f, permutation p=%.4g, 95%% CI [%.3f, %.3f]" %
          (c["r"], c["p_perm"], c["ci_lo"], c["ci_hi"]))
    print("  centroids n=%d: rho(IF, phigh)=%.3f rho(IF, plow)=%.3f rho(IF, lam)=%.3f" %
          (m["centroids"]["n"], m["centroids"]["rho_phigh"], m["centroids"]["rho_plow"], m["centroids"]["rho_lam"]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
