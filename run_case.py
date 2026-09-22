import io, os, sys, json, random, time, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zone_d as Z
from ecvrptw import build, Solution, XI
from algos import run_gls, run_alns, route_elimination, Search, OPFN, OPS

FUEL_USD = 0.95
CARBON_USD_T = 80.0
SEEDS = [1, 2, 3, 4, 5]
BUDGET = 45.0


def solve(ev, seed, algo):
    if algo == "ALNS":
        return run_alns(ev, seed, max_sec=BUDGET)
    return run_gls(ev, seed, algo, max_sec=BUDGET)


def best_solution(ev, algo, seed):
    if algo == "ALNS":
        _, sol = run_alns(ev, seed, max_sec=BUDGET, return_sol=True)
    else:
        _, sol = run_gls(ev, seed, algo, max_sec=BUDGET, return_sol=True)
    return sol


def main():
    inst, df = Z.load()
    ev = Z.ZoneEval(inst)
    out = []

    base = Z.sweep_plan(inst, ev)
    bs = Z.stats(ev, base)
    out.append("BASELINE (sweep + nearest neighbour): " + json.dumps(
        {k: round(v, 4) for k, v in bs.items()}))

    rows = []
    for algo in ("GLS", "IGLS", "RGLS", "HGLS", "ALNS"):
        for s in SEEDS:
            r = solve(ev, s, algo)
            r.update(algo=algo, seed=s)
            rows.append(r)
            out.append("  %-5s seed %d: nv=%d dist=%.2f fuel=%.3f IF=%.4f" %
                       (algo, s, r["nv"], r["dist"], r["fuel"], r["IF"]))
    with io.open("case_runs.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()),
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    best = min(rows, key=lambda r: (r["nv"], r["fuel"]))
    out.append("")
    out.append("BEST: %s seed %d  nv=%d dist=%.2f fuel=%.3f"
               % (best["algo"], best["seed"], best["nv"], best["dist"], best["fuel"]))

    sol = best_solution(ev, best["algo"], best["seed"])
    os_ = Z.stats(ev, sol)
    out.append("OPTIMIZED (rebuilt): " + json.dumps(
        {k: round(v, 4) for k, v in os_.items()}))

    def money(st):
        return (st["fuel"] * FUEL_USD, st["emis"] / 1000.0 * CARBON_USD_T)

    bf, bc = money(bs)
    of, oc = money(os_)
    sysrows = [["Metric", "Manual plan", "Optimized plan", "Change", "Percent change"]]

    def add(label, a, b, fmt="%.2f", pct=True):
        ch = b - a
        p = (100 * ch / a) if (pct and a) else ""
        sysrows.append([label, fmt % a, fmt % b, fmt % ch,
                        ("%.2f" % p) if p != "" else ""])
    add("Number of routes", bs["nv"], os_["nv"], "%.0f")
    add("Total distance (km)", bs["dist"], os_["dist"])
    add("Average intensity IF (L/km)", bs["IF"], os_["IF"], "%.4f")
    add("Total fuel (L)", bs["fuel"], os_["fuel"])
    add("Total CO2e (kg)", bs["emis"], os_["emis"])
    add("Fuel cost (USD)", bf, of)
    add("Carbon cost (USD)", bc, oc)
    add("Total cost (USD)", bf + bc, of + oc)
    add("Distance at low speed (%)", bs["plow"], os_["plow"], "%.1f")
    add("Distance-weighted payload", bs["lam"], os_["lam"], "%.3f")

    dist_part = (bs["dist"] - os_["dist"]) * bs["IF"]
    int_part = os_["dist"] * (bs["IF"] - os_["IF"])
    out.append("")
    out.append("fuel saving = %.3f L; distance term %.3f L (%.2f pp), "
               "intensity term %.3f L (%.2f pp)"
               % (bs["fuel"] - os_["fuel"], dist_part,
                  100 * dist_part / bs["fuel"], int_part,
                  100 * int_part / bs["fuel"]))

    pairs = []
    for r, it in zip(sol.routes, sol.info):
        pairs.append((it["dist"], it["fuel"], len(r)))
    pairs.sort(reverse=True)
    routerows = [["Route", "Containers", "Distance (km)", "Fuel (L)",
                  "IF (L/km)", "CO2e (kg)"]]
    for i, (dd, ff, nn) in enumerate(pairs[:5], 1):
        routerows.append(["O%d" % i, nn, "%.2f" % dd, "%.3f" % ff,
                          "%.4f" % (ff / dd), "%.2f" % (XI * ff)])

    json.dump({"baseline": bs, "optimized": os_,
               "best_algo": best["algo"], "best_seed": best["seed"],
               "fuel_usd": FUEL_USD, "carbon_usd_t": CARBON_USD_T,
               "capacity_m3": Z.CAPACITY,
               "dist_term_L": dist_part, "int_term_L": int_part,
               "system_rows": sysrows, "route_rows": routerows,
               "n_customers": inst.n, "total_demand": float(inst.dem.sum())},
              io.open("case_results.json", "w", encoding="utf-8"), indent=1)
    io.open("case_log.txt", "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
