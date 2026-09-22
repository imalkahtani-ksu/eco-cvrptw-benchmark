import io, json, os
import numpy as np, pandas as pd

FUEL_USD = 0.95
CARBON_USD_T = 80.0
XI = 2.60

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    res = json.load(io.open(os.path.join(HERE, "case_results.json"), encoding="utf-8"))
    runs = pd.read_csv(os.path.join(HERE, "case_runs.csv"))
    b = res["baseline"]

    byalgo = runs.groupby("algo")[["nv", "dist", "fuel", "IF", "plow", "lam"]].mean()
    best_algo = byalgo["fuel"].idxmin()
    sub = runs[runs.algo == best_algo]
    o = dict(nv=sub["nv"].mean(), dist=sub["dist"].mean(), fuel=sub["fuel"].mean(),
             plow=sub["plow"].mean(), lam=sub["lam"].mean())
    o["IF"] = o["fuel"] / o["dist"]
    o["emis"] = XI * o["fuel"]
    sd_fuel, sd_dist = sub["fuel"].std(), sub["dist"].std()

    dist_term = (b["dist"] - o["dist"]) * b["IF"]
    int_term = o["dist"] * (b["IF"] - o["IF"])
    tot_save = b["fuel"] - o["fuel"]


    def money(st):
        return st["fuel"] * FUEL_USD, st["emis"] / 1000.0 * CARBON_USD_T


    bf, bc = money(b)
    of, oc = money(o)

    T8 = [["Metric", "Manual plan", "Optimized plan", "Change", "Change (%)"]]


    def add(lbl, x, y, f="%.2f", pct=True):
        T8.append([lbl, f % x, f % y, f % (y - x),
                   ("%+.2f" % (100 * (y - x) / x)) if pct and x else "-"])


    add("Vehicles used", b["nv"], o["nv"], "%.0f")
    add("Total distance (km)", b["dist"], o["dist"])
    add("Fuel intensity IF (L/km)", b["IF"], o["IF"], "%.4f")
    add("Total fuel (L/day)", b["fuel"], o["fuel"])
    add("Total CO2e (kg/day)", b["emis"], o["emis"])
    add("Distance-weighted payload", b["lam"], o["lam"], "%.3f")
    add("Fuel cost (USD/day)", bf, of)
    add("Carbon cost (USD/day)", bc, oc)
    add("Total cost (USD/day)", bf + bc, of + oc)

    T9 = [["Component", "Fuel (L/day)", "Share of manual fuel (%)"],
          ["Saving from shorter routes", "%+.2f" % -dist_term,
           "%+.2f" % (-100 * dist_term / b["fuel"])],
          ["Penalty from higher intensity", "%+.2f" % -int_term,
           "%+.2f" % (-100 * int_term / b["fuel"])],
          ["Net fuel saving", "%+.2f" % -tot_save,
           "%+.2f" % (-100 * tot_save / b["fuel"])]]

    summary = dict(best_algo=best_algo, baseline=b, optimized=o,
                   sd_fuel=float(sd_fuel), sd_dist=float(sd_dist),
                   dist_term=float(dist_term), int_term=float(int_term),
                   tot_save=float(tot_save),
                   pct_dist=100 * (o["dist"] - b["dist"]) / b["dist"],
                   pct_fuel=100 * (o["fuel"] - b["fuel"]) / b["fuel"],
                   pct_IF=100 * (o["IF"] - b["IF"]) / b["IF"],
                   cost_save=(bf + bc) - (of + oc),
                   T8=T8, T9=T9,
                   n_cust=res["n_customers"], demand=res["total_demand"],
                   cap=res["capacity_m3"],
                   byalgo=byalgo.round(4).to_dict())
    json.dump(summary, io.open(os.path.join(HERE, "case_final.json"), "w", encoding="utf-8"), indent=1)

    lines = ["best method: %s (mean of 5 seeds)" % best_algo,
             "baseline  : nv=%.0f dist=%.2f fuel=%.3f IF=%.4f lam=%.3f"
             % (b["nv"], b["dist"], b["fuel"], b["IF"], b["lam"]),
             "optimized : nv=%.0f dist=%.2f fuel=%.3f IF=%.4f lam=%.3f (sd fuel %.2f)"
             % (o["nv"], o["dist"], o["fuel"], o["IF"], o["lam"], sd_fuel),
             "",
             "distance  %+.2f%%   intensity %+.2f%%   fuel %+.2f%%"
             % (summary["pct_dist"], summary["pct_IF"], summary["pct_fuel"]),
             "decomposition: shorter routes %+.2f L, intensity penalty %+.2f L, net %+.2f L"
             % (-dist_term, int_term, -tot_save),
             "cost saving: %.2f USD/day" % summary["cost_save"]]
    io.open(os.path.join(HERE, "case_summary.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
