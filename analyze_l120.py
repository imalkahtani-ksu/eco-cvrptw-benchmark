import json, os
import numpy as np, pandas as pd
from scipy import stats
import analyze_r1 as R

# series-2 instances at 120 s against the same instances and seeds at 30 s
d120 = R.load("results/r1_series2_120.csv")
d30 = R.load("results/r1_main.csv")
d30 = d30[d30.instance.isin(d120.instance.unique()) & d30.seed.isin(d120.seed.unique())]
out = dict(runs_120=len(d120), instances=int(d120.instance.nunique()),
           zero_iter_120=int((d120.iters == 0).sum()), max_search_120=float(d120.search_time.max()))
for tag, d in (("30", d30), ("120", d120)):
    g = R.aggregate(d)
    for algos, fam in ((R.ALGOS, "five"), (R.GLSF, "gls")):
        for col in ("fuel", "dist", "IF"):
            f = R.friedman(g, algos, col)
            out["fr_%s_%s_%s" % (tag, fam, col)] = [f["chi2"], f["p"]]
    sig, dev = R.signature(g)
    out["sig_" + tag] = sig.set_index("algo")[["fuel", "fuel_p", "IF", "IF_p", "dist", "dist_p"]].round(4).to_dict("index")
    out["iters_" + tag] = d.groupby("algo")["iters"].median().to_dict()
    out["fuel_mean_" + tag] = g.groupby("algo")["fuel"].mean().round(2).to_dict()
    pw = R.pairwise(g, R.GLSF)
    out["pairwise_" + tag] = pw[["a", "b", "mean_pct", "ci_lo", "ci_hi", "p_holm"]].round(4).to_dict("records")
# improvement from 30 s to 120 s, paired by instance, seed and method
k = ["instance", "seed", "algo"]
m = d30.set_index(k)[["fuel"]].join(d120.set_index(k)[["fuel"]], rsuffix="_120", how="inner")
m["impr"] = 100 * (m.fuel_120 - m.fuel) / m.fuel
out["improvement_pct"] = m.groupby("algo")["impr"].mean().round(3).to_dict()
out["improvement_all"] = float(m["impr"].mean())
os.makedirs("results/analysis_r1", exist_ok=True)
json.dump(out, open("results/analysis_r1/l120_summary.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:3500])
