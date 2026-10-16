import os, sys, csv, time, argparse
import multiprocessing as mp

# Why does fuel routing with speed selection end worse than fuel routing at fixed medium speed?
# For each instance and seed: build both starting solutions, run GLS from each, run free-speed GLS
# from the fixed-speed start, and re-score fixed-speed solutions with free speed selection.
INSTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instances")
INST = "C101,C108,C203,C206,R104,R110,R202,R208,RC103,RC107,RC202,RC206"


def rescore(ev, sol):
    # evaluate the same routes under another evaluator; None if a route is infeasible there
    tot_f = tot_d = 0.0
    worse = 0
    for r in sol.routes:
        if not r:
            continue
        it = ev.route_eval(r)
        if it is None:
            return None
        tot_f += it["fuel"]
        tot_d += it["dist"]
    return tot_f, tot_d


def greedy_vs_medium(ev_free, ev_fix, sol):
    # routes where the greedy speed assignment burns more fuel than driving all arcs at medium
    n = worse = 0
    extra = 0.0
    for r in sol.routes:
        if not r:
            continue
        a, b = ev_free.route_eval(r), ev_fix.route_eval(r)
        if a is None or b is None:
            continue
        n += 1
        if a["fuel"] > b["fuel"] + 1e-9:
            worse += 1
            extra += a["fuel"] - b["fuel"]
    return n, worse, extra


def work(task):
    name, seed, budget = task
    from ecvrptw import read_solomon, Eval
    from algos import start_solution, run_gls
    I = read_solomon(os.path.join(INSTDIR, name + ".txt"), name)
    fix = Eval(I, obj="fuel", fixed=1)
    free = Eval(I, obj="fuel", fixed=None)
    s_fix = start_solution(fix, seed)
    s_free = start_solution(free, seed)
    r_fix, sol_fix = run_gls(fix, s_fix, seed, "GLS", max_sec=budget, return_sol=True)
    r_free, sol_free = run_gls(free, s_free, seed, "GLS", max_sec=budget, return_sol=True)
    # free-speed search from the fixed-speed start: re-evaluate the start under free speeds first
    from ecvrptw import build
    s_warm = s_fix.clone()
    build(free, s_warm)
    r_warm, sol_warm = run_gls(free, s_warm, seed, "GLS", max_sec=budget, return_sol=True)
    cross = rescore(free, sol_fix)
    n, worse, extra = greedy_vs_medium(free, fix, sol_fix)
    return dict(instance=name, seed=seed,
                start_fix_nv=s_fix.nv, start_fix_fuel=s_fix.fuel, start_fix_dist=s_fix.dist,
                start_free_nv=s_free.nv, start_free_fuel=s_free.fuel, start_free_dist=s_free.dist,
                fix_nv=r_fix["nv"], fix_fuel=r_fix["fuel"], fix_dist=r_fix["dist"], fix_iters=r_fix["iters"], fix_evals=r_fix["evals"],
                free_nv=r_free["nv"], free_fuel=r_free["fuel"], free_dist=r_free["dist"], free_iters=r_free["iters"], free_evals=r_free["evals"],
                warm_nv=r_warm["nv"], warm_fuel=r_warm["fuel"], warm_dist=r_warm["dist"],
                cross_fuel=cross[0] if cross else "", cross_dist=cross[1] if cross else "",
                routes=n, greedy_worse_routes=worse, greedy_extra_fuel=extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/diag_speed.csv")
    ap.add_argument("--instances", default=INST)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    tasks = [(n, s, a.budget) for n in a.instances.split(",") for s in range(1, a.seeds + 1)]
    rows = []
    t0 = time.time()
    with mp.Pool(a.procs) as pool:
        for r in pool.imap_unordered(work, tasks):
            rows.append(r)
            print("  %d/%d %s seed %d  (%.0fs)" % (len(rows), len(tasks), r["instance"], r["seed"], time.time() - t0))
            sys.stdout.flush()
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["instance"], r["seed"])))
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
