import os, sys, csv, time, itertools, argparse
import multiprocessing as mp

INSTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inst")
ALGOS = ("GLS", "IGLS", "RGLS", "HGLS", "ALNS")
_CACHE = {}


def _inst(name):
    if name not in _CACHE:
        from ecvrptw import read_solomon
        _CACHE[name] = read_solomon(os.path.join(INSTDIR, name + ".txt"), name)
    return _CACHE[name]


def work(task):
    name, algo, seed, phi, budget = task
    from ecvrptw import Eval, PHI_BASE
    from algos import run_gls, run_alns
    I = _inst(name)
    ev = Eval(I, phi=phi)
    try:
        if algo == "ALNS":
            r = run_alns(ev, seed, max_sec=budget)
        else:
            r = run_gls(ev, seed, algo, max_sec=budget)
    except Exception as e:
        return dict(instance=name, algo=algo, seed=seed, phi=str(phi),
                    error=repr(e)[:120], nv=-1, dist=0, fuel=0, emis=0, IF=0,
                    plow=0, pmed=0, phigh=0, lam=0, runtime=0, iters=0,
                    budget=budget)
    r.update(instance=name, algo=algo, seed=seed, phi=str(phi),
             budget=budget, error="")
    return r


FIELDS = ["instance", "algo", "seed", "phi", "nv", "dist", "fuel", "emis",
          "IF", "plow", "pmed", "phigh", "lam", "runtime", "iters", "budget",
          "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_main.csv")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=float, default=15.0)
    ap.add_argument("--budgets", default="")
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--instances", default="")
    ap.add_argument("--algos", default=",".join(ALGOS))
    ap.add_argument("--phis", default="")
    args = ap.parse_args()

    if args.instances:
        names = args.instances.split(",")
    else:
        names = sorted(f[:-4] for f in os.listdir(INSTDIR) if f.endswith(".txt"))
    algos = args.algos.split(",")

    from ecvrptw import PHI_BASE
    if args.phis:
        phis = []
        for chunk in args.phis.split(";"):
            phis.append(tuple(float(v) for v in chunk.split(",")))
    else:
        phis = [PHI_BASE]

    budgets = ([float(v) for v in args.budgets.split(",")]
               if args.budgets else [args.budget])
    tasks = [(n, a, s, p, b)
             for n in names for a in algos
             for s in range(1, args.seeds + 1) for p in phis for b in budgets]
    print("tasks: %d  (%d instances x %d algos x %d seeds x %d phi)"
          % (len(tasks), len(names), len(algos), args.seeds, len(phis)))
    sys.stdout.flush()

    t0 = time.time()
    done = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        with mp.Pool(args.procs) as pool:
            for r in pool.imap_unordered(work, tasks, chunksize=1):
                w.writerow(r)
                done += 1
                if done % 25 == 0 or done == len(tasks):
                    el = time.time() - t0
                    eta = el / done * (len(tasks) - done)
                    print("  %d/%d  elapsed %.0fs  eta %.0fs"
                          % (done, len(tasks), el, eta))
                    sys.stdout.flush()
                    fh.flush()
    print("WROTE %s in %.0fs" % (args.out, time.time() - t0))


if __name__ == "__main__":
    main()
