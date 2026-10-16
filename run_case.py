import io, os, sys, json, csv, time, argparse
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALGOS = ("GLS", "IGLS", "RGLS", "HGLS", "ALNS")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "case_study")


def work(task):
    # one seed: build the common start once, then run every method from it
    seed, budget = task
    import zone_d as Z
    from algos import run_gls, run_alns, start_solution
    inst, _ = Z.load()
    ev = Z.ZoneEval(inst)
    t = time.time()
    start = start_solution(ev, seed)
    start_time = time.time() - t
    rows = []
    for a in ALGOS:
        if a == "ALNS":
            r, sol = run_alns(ev, start, seed, max_sec=budget, return_sol=True)
        else:
            r, sol = run_gls(ev, start, seed, a, max_sec=budget, return_sol=True)
        r.update(algo=a, seed=seed, budget=budget, start_time=start_time,
                 start_nv=start.nv, start_fuel=start.fuel,
                 routes=json.dumps([[int(c) for c in x] for x in sol.routes]))
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=float, default=45.0)
    ap.add_argument("--procs", type=int, default=5)
    args = ap.parse_args()

    import zone_d as Z
    from algos import _stats
    inst, _ = Z.load()
    ev = Z.ZoneEval(inst)
    base = Z.sweep_plan(inst, ev)
    bs = Z.stats(ev, base)
    print("constructed baseline:", {k: round(v, 4) for k, v in bs.items()})

    rows = []
    with mp.Pool(args.procs) as pool:
        for rr in pool.imap_unordered(work, [(s, args.budget) for s in range(1, args.seeds + 1)]):
            rows.extend(rr)
            for r in rr:
                print("  %-5s seed %d: nv=%d dist=%.2f fuel=%.3f IF=%.4f iters=%d"
                      % (r["algo"], r["seed"], r["nv"], r["dist"], r["fuel"], r["IF"], r["iters"]))
    rows.sort(key=lambda r: (r["algo"], r["seed"]))
    fields = ["algo", "seed", "budget", "nv", "dist", "fuel", "emis", "IF", "plow", "pmed",
              "phigh", "lam", "start_time", "start_nv", "start_fuel", "search_time",
              "iters", "evals", "accepted", "routes"]
    with io.open(os.path.join(OUT, "case_runs.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    best = min(rows, key=lambda r: (r["nv"], r["fuel"]))
    json.dump({"baseline": bs, "baseline_routes": [[int(c) for c in x] for x in base.routes],
               "best_run": {k: best[k] for k in ("algo", "seed", "nv", "dist", "fuel", "IF")},
               "load_profile": "collection", "budget_s": args.budget,
               "capacity_m3": Z.CAPACITY, "n_customers": inst.n,
               "total_demand": float(inst.dem.sum())},
              io.open(os.path.join(OUT, "case_results.json"), "w", encoding="utf-8"), indent=1)
    print("wrote case_study/case_runs.csv and case_study/case_results.json")


if __name__ == "__main__":
    main()
