import os, sys, time, subprocess

# every experiment of the revision, in order; results go to results/r1_*.csv and
# each step writes its own log in results/logs
PROCS = "10"
ANY = "C103,C107,C205,C208,R103,R109,R204,R210,RC102,RC106,RC203,RC207"
SENS = "C101,C108,C203,C206,R104,R110,R202,R208,RC103,RC107,RC202,RC206"
PHIS = "1.0,1.0,1.0;0.93,1.0,1.09;0.85,1.0,1.18;0.75,1.0,1.30"

STEPS = [
    ("main", ["runner.py", "--out", "results/r1_main.csv", "--seeds", "5", "--budgets", "30", "--procs", PROCS]),
    ("case", ["run_case.py", "--seeds", "5", "--budget", "45", "--procs", "5"]),
    ("budget", ["runner.py", "--out", "results/r1_anytime.csv", "--seeds", "3", "--budgets", "3,8,20,60",
                "--instances", ANY, "--procs", PROCS]),
    ("sens", ["runner.py", "--out", "results/r1_sens.csv", "--seeds", "3", "--budgets", "30",
              "--algos", "GLS,HGLS", "--instances", SENS, "--phis", PHIS, "--procs", PROCS]),
    ("ablation", ["runner.py", "--out", "results/r1_ablation.csv", "--seeds", "5", "--budgets", "30",
                  "--algos", "GLS", "--configs", "dist_fixed,fuel_fixed,fuel_free", "--procs", PROCS]),
    ("cmem", ["runner.py", "--out", "results/r1_cmem.csv", "--seeds", "5", "--budgets", "30",
              "--phis", "cmem", "--procs", PROCS]),
    ("analysis", ["analyze_r1.py", "--main", "results/r1_main.csv", "--anytime", "results/r1_anytime.csv",
                  "--sens", "results/r1_sens.csv", "--cmem", "results/r1_cmem.csv",
                  "--ablation", "results/r1_ablation.csv", "--out", "results/analysis_r1"]),
    ("case_tables", ["case_tables.py"]),
]


def keep_awake():
    # ask Windows not to idle-sleep while this process runs; released when it exits
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def main():
    only = sys.argv[1:]
    os.makedirs("results/logs", exist_ok=True)
    keep_awake()
    t_all = time.time()
    for name, cmd in STEPS:
        if only and name not in only:
            continue
        t = time.time()
        print("[%s] %s ..." % (time.strftime("%H:%M"), name))
        sys.stdout.flush()
        with open("results/logs/%s.log" % name, "w", encoding="utf-8") as fh:
            rc = subprocess.call([sys.executable] + cmd, stdout=fh, stderr=subprocess.STDOUT)
        print("[%s] %s finished in %.0f min, exit code %d" % (time.strftime("%H:%M"), name, (time.time() - t) / 60, rc))
        sys.stdout.flush()
        if rc != 0:
            print("stopping: see results/logs/%s.log" % name)
            with open("results/logs/STATUS.txt", "w") as fh:
                fh.write("FAILED at step %s, %s\n" % (name, time.strftime("%Y-%m-%d %H:%M")))
            sys.exit(rc)
    with open("results/logs/STATUS.txt", "w") as fh:
        fh.write("ALL STEPS FINISHED %s (%.1f h)\n" % (time.strftime("%Y-%m-%d %H:%M"), (time.time() - t_all) / 3600))


if __name__ == "__main__":
    main()
