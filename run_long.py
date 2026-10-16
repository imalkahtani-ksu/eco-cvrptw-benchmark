import os, sys, time, subprocess

# longer-budget check on the series-2 classes; waits for any running re-scoring job to finish
S2 = "C201,C202,C203,C204,C205,C206,C207,C208,R201,R202,R203,R204,R205,R206,R207,R208,R209,R210,R211,RC201,RC202,RC203,RC204,RC205,RC206,RC207,RC208"
LOG = "results/logs/crossscore.log"


def main():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    while os.path.exists(LOG) and "WROTE" not in open(LOG).read() and "Traceback" not in open(LOG).read():
        time.sleep(20)
    with open("results/logs/series2_120.log", "w") as fh:
        rc = subprocess.call([sys.executable, "runner.py", "--out", "results/r1_series2_120.csv", "--seeds", "3",
                              "--budgets", "120", "--instances", S2, "--procs", "10"], stdout=fh, stderr=subprocess.STDOUT)
    with open("results/logs/STATUS_series2.txt", "w") as fh:
        fh.write("series2 120 s %s, exit code %d, %s\n" % ("finished" if rc == 0 else "FAILED", rc, time.strftime("%Y-%m-%d %H:%M")))


if __name__ == "__main__":
    main()
