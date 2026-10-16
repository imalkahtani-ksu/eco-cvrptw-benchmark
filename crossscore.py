import os, sys, csv, time
import multiprocessing as mp

# value of the speed decision on given routes: solve with fuel routing at fixed medium speed,
# then re-score the same routes with speed selection allowed
INSTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instances")


def work(task):
    name, seed, budget = task
    from ecvrptw import read_solomon, Eval, build
    from algos import start_solution, run_gls, _stats
    I = read_solomon(os.path.join(INSTDIR, name + ".txt"), name)
    fix = Eval(I, obj="fuel", fixed=1)
    free = Eval(I, obj="fuel", fixed=None)
    start = start_solution(fix, seed)
    r, sol = run_gls(fix, start, seed, "GLS", max_sec=budget, return_sol=True)
    cs = sol.clone()
    build(free, cs)
    st = _stats(free, cs)
    return dict(instance=name, seed=seed, nv=r["nv"], dist=r["dist"], fuel_fixed=r["fuel"],
                fuel_rescored=cs.fuel, IF_fixed=r["IF"], IF_rescored=st["IF"],
                plow_rescored=st["plow"], pmed_rescored=st["pmed"], phigh_rescored=st["phigh"])


def main():
    names = sorted(f[:-4] for f in os.listdir(INSTDIR) if f.endswith(".txt"))
    tasks = [(n, s, 30.0) for n in names for s in range(1, 6)]
    rows = []
    t0 = time.time()
    with mp.Pool(10) as pool:
        for r in pool.imap_unordered(work, tasks):
            rows.append(r)
            if len(rows) % 20 == 0:
                print("  %d/%d (%.0fs)" % (len(rows), len(tasks), time.time() - t0))
                sys.stdout.flush()
    with open("results/r1_crossscore.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["instance"], r["seed"])))
    print("WROTE results/r1_crossscore.csv in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
