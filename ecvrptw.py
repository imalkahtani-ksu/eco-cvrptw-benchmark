import math, random, time
import numpy as np

SPEEDS = (30.0, 50.0, 70.0)
U_REF = 50.0
TMUL = tuple(U_REF / u for u in SPEEDS)
PHI_BASE = (0.85, 1.00, 1.18)
RHO0, RHO1 = 0.20, 0.50
XI = 2.60
NLEV = 3


class Instance:
    __slots__ = ("name", "n", "x", "y", "dem", "rdy", "due", "srv", "Q",
                 "maxveh", "d", "near")

    def __init__(self, name, rows, Q, maxveh, nnear=20):
        self.name = name
        self.n = len(rows) - 1
        a = np.array(rows, dtype=float)
        self.x, self.y = a[:, 1], a[:, 2]
        self.dem, self.rdy, self.due, self.srv = a[:, 3], a[:, 4], a[:, 5], a[:, 6]
        self.Q, self.maxveh = Q, maxveh
        dx = self.x[:, None] - self.x[None, :]
        dy = self.y[:, None] - self.y[None, :]
        self.d = np.sqrt(dx * dx + dy * dy)
        order = np.argsort(self.d, axis=1)
        self.near = [[j for j in order[i] if j != i and j != 0][:nnear]
                     for i in range(self.n + 1)]


def read_solomon(path, name):
    rows, Q, maxveh = [], None, None
    with open(path) as fh:
        lines = [l.rstrip("\n") for l in fh]
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("NUMBER"):
            p = lines[i + 1].split()
            maxveh, Q = int(p[0]), float(p[1])
            i += 2
            continue
        p = s.split()
        if len(p) >= 7:
            try:
                rows.append([float(v) for v in p[:7]])
            except ValueError:
                pass
        i += 1
    return Instance(name, rows, Q, maxveh)


class Eval:

    def __init__(self, inst, phi=PHI_BASE):
        self.I = inst
        self.phi = tuple(phi)
        self.drho = RHO1 - RHO0
        self.calls = 0
        self.dd = inst.d.tolist()
        self.dem = inst.dem.tolist()
        self.rdy = inst.rdy.tolist()
        self.due = inst.due.tolist()
        self.srv = inst.srv.tolist()
        self.Q = float(inst.Q)

    def _prop(self, seq, arcd, lev):
        rdy, due, srv = self.rdy, self.due, self.srv
        t = 0.0
        m = len(seq) - 1
        for k in range(m):
            t += arcd[k] * TMUL[lev[k]]
            j = seq[k + 1]
            if j == 0:
                return (t <= due[0] + 1e-6), k
            if t < rdy[j]:
                t = rdy[j]
            if t > due[j] + 1e-6:
                return False, k
            t += srv[j]
        return True, -1

    def route_eval(self, r):
        self.calls += 1
        if not r:
            return {"dist": 0.0, "fuel": 0.0, "lev": [], "arcd": [], "load": []}
        dem, Q, dd = self.dem, self.Q, self.dd
        tot = 0.0
        for c in r:
            tot += dem[c]
        if tot > Q + 1e-9:
            return None

        seq = [0]
        seq.extend(r)
        seq.append(0)
        m = len(seq) - 1
        arcd = [0.0] * m
        load = [0.0] * m
        base = [0.0] * m
        cur = tot
        drho, R0 = self.drho, RHO0
        for k in range(m):
            a = dd[seq[k]][seq[k + 1]]
            arcd[k] = a
            load[k] = cur
            base[k] = (R0 + (cur / Q) * drho) * a
            cur -= dem[seq[k + 1]] if seq[k + 1] != 0 else 0.0

        lev = [0] * m
        ok, kbad = self._prop(seq, arcd, lev)
        if not ok:
            phi = self.phi
            for _ in range(2 * m + 4):
                best, bestk = None, -1
                for k in range(kbad + 1):
                    L = lev[k]
                    if L >= NLEV - 1:
                        continue
                    dt = arcd[k] * (TMUL[L] - TMUL[L + 1])
                    if dt <= 1e-12:
                        continue
                    ratio = (base[k] * (phi[L + 1] - phi[L])) / dt
                    if best is None or ratio < best:
                        best, bestk = ratio, k
                if bestk < 0:
                    return None
                lev[bestk] += 1
                ok, kbad = self._prop(seq, arcd, lev)
                if ok:
                    break
            if not ok:
                return None

        phi = self.phi
        fuel = 0.0
        dist = 0.0
        for k in range(m):
            fuel += base[k] * phi[lev[k]]
            dist += arcd[k]
        return {"dist": dist, "fuel": fuel, "lev": lev, "arcd": arcd,
                "load": load}


class Solution:
    __slots__ = ("routes", "info", "fuel", "dist", "nv")

    def __init__(self):
        self.routes, self.info = [], []
        self.fuel = self.dist = 0.0
        self.nv = 0

    def clone(self):
        s = Solution()
        s.routes = [r[:] for r in self.routes]
        s.info = list(self.info)
        s.fuel, s.dist, s.nv = self.fuel, self.dist, self.nv
        return s

    def recompute(self, ev):
        self.fuel = sum(i["fuel"] for i in self.info)
        self.dist = sum(i["dist"] for i in self.info)
        self.nv = sum(1 for r in self.routes if r)


def build(ev, sol):
    sol.info = []
    for r in sol.routes:
        it = ev.route_eval(r)
        if it is None:
            return False
        sol.info.append(it)
    sol.recompute(ev)
    return True


def cam_construct(ev, rng):
    I = ev.I
    unserved = set(range(1, I.n + 1))
    routes = []
    while unserved:
        seed = max(unserved, key=lambda c: (I.d[0, c], -I.due[c]))
        r = [seed]
        if ev.route_eval(r) is None:
            unserved.discard(seed)
            continue
        unserved.discard(seed)
        improved = True
        while improved and unserved:
            improved = False
            best, bc, bpos = None, -1, -1
            cand = list(unserved)
            if len(cand) > 45:
                cand = rng.sample(cand, 45)
            for c in cand:
                for pos in range(len(r) + 1):
                    nr = r[:pos] + [c] + r[pos:]
                    it = ev.route_eval(nr)
                    if it is None:
                        continue
                    cost = it["fuel"]
                    if best is None or cost < best:
                        best, bc, bpos = cost, c, pos
            if bc > 0:
                r.insert(bpos, bc)
                unserved.discard(bc)
                improved = True
        routes.append(r)
        if len(routes) > I.maxveh + 12:
            break
    for c in list(unserved):
        routes.append([c])
        unserved.discard(c)
    s = Solution()
    s.routes = [r for r in routes if r]
    build(ev, s)
    return s
