import os, sys, csv, time, argparse
import multiprocessing as mp

INSTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instances")
ALGOS = ("GLS", "IGLS", "RGLS", "HGLS", "ALNS")

# objective and speed handling of each experimental configuration
CONFIGS = {
    "fuel_free": dict(obj="fuel", fixed=None),   # the E-CVRPTW as studied
    "fuel_fixed": dict(obj="fuel", fixed=1),     # fuel objective, every arc at medium speed
    "dist_fixed": dict(obj="dist", fixed=1),     # classical VRPTW: distance, medium speed
}
_CACHE = {}


def _inst(name):
    if name not in _CACHE:
        from ecvrptw import read_solomon
        _CACHE[name] = read_solomon(os.path.join(INSTDIR, name + ".txt"), name)
    return _CACHE[name]


def parse_phi(txt):
    from ecvrptw import PHI_BASE, cmem_phi
    if txt == "base":
        return PHI_BASE
    if txt == "cmem":
        return cmem_phi()
    return tuple(float(v) for v in txt.split(","))


def work(task):
    # one task = one instance, seed, fuel curve and configuration; the starting
    # solution is built once and every method and budget starts from a copy of it
    name, seed, phi_txt, config, algos, budgets = task
    from ecvrptw import Eval
    from algos import run_gls, run_alns, start_solution
    phi = parse_phi(phi_txt)
    cfg = CONFIGS[config]
    I = _inst(name)
    ev = Eval(I, phi=phi, obj=cfg["obj"], fixed=cfg["fixed"])
    head = dict(instance=name, seed=seed, phi=phi_txt, config=config)
    rows = []
    try:
        t = time.time()
        start = start_solution(ev, seed)
        start_time = time.time() - t
    except Exception as e:
        return [dict(head, algo=a, budget=b, error="start: " + repr(e)[:100])
                for a in algos for b in budgets]
    for a in algos:
        for b in budgets:
            row = dict(head, algo=a, budget=b, start_time=start_time,
                       start_nv=start.nv, start_dist=start.dist, start_fuel=start.fuel)
            try:
                if a == "ALNS":
                    r = run_alns(ev, start, seed, max_sec=b)
                else:
                    r = run_gls(ev, start, seed, a, max_sec=b)
                row.update(r, error="")
            except Exception as e:
                row.update(error=repr(e)[:120])
            rows.append(row)
    return rows


FIELDS = ["instance", "algo", "seed", "phi", "config", "budget", "nv", "dist", "fuel",
          "emis", "IF", "plow", "pmed", "phigh", "lam", "start_time", "start_nv",
          "start_dist", "start_fuel", "search_time", "iters", "evals", "accepted",
          "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/r1_main.csv")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budgets", default="30")
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--instances", default="")
    ap.add_argument("--algos", default=",".join(ALGOS))
    ap.add_argument("--phis", default="base", help="base, cmem, or a,b,c; separate several with ;")
    ap.add_argument("--configs", default="fuel_free")
    args = ap.parse_args()

    if args.instances:
        names = args.instances.split(",")
    else:
        names = sorted(f[:-4] for f in os.listdir(INSTDIR) if f.endswith(".txt"))
    algos = args.algos.split(",")
    budgets = [float(v) for v in args.budgets.split(",")]
    phis = args.phis.split(";")
    configs = args.configs.split(",")
    tasks = [(n, s, p, c, algos, budgets)
             for n in names for s in range(1, args.seeds + 1) for p in phis for c in configs]
    nruns = len(tasks) * len(algos) * len(budgets)
    print("%d tasks, %d runs (%d instances x %d seeds x %d phi x %d configs x %d algos x %d budgets)"
          % (len(tasks), nruns, len(names), args.seeds, len(phis), len(configs),
             len(algos), len(budgets)))
    sys.stdout.flush()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    done = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        with mp.Pool(args.procs) as pool:
            for rows in pool.imap_unordered(work, tasks, chunksize=1):
                for r in rows:
                    w.writerow(r)
                fh.flush()
                done += 1
                el = time.time() - t0
                print("  task %d/%d  elapsed %.0fs  eta %.0fs"
                      % (done, len(tasks), el, el / done * (len(tasks) - done)))
                sys.stdout.flush()
    print("WROTE %s in %.0fs" % (args.out, time.time() - t0))


if __name__ == "__main__":
    main()
