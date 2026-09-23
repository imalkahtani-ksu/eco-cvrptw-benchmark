import math, os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecvrptw import Instance, Eval, Solution, build, SPEEDS, U_REF, TMUL, XI

HERE = os.path.dirname(os.path.abspath(__file__))
NODES_CSV = os.path.join(HERE, "case_study", "zone_d_nodes.csv")
R_EARTH = 6371.0088
CAPACITY = 5.0
TSCALE = 60.0 / U_REF


def haversine_matrix(lat, lon):
    la = np.radians(lat)[:, None]
    lo = np.radians(lon)[:, None]
    dla = la - la.T
    dlo = lo - lo.T
    a = np.sin(dla / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlo / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load(depot_row=0, path=NODES_CSV):
    d = pd.read_csv(path)
    d.columns = [str(c).strip() for c in d.columns]
    d = d.sort_values("Point").reset_index(drop=True)
    lat = d["Latitude_x"].to_numpy(float)
    lon = d["Longitude_y"].to_numpy(float)
    D = haversine_matrix(lat, lon)

    inst = Instance.__new__(Instance)
    inst.name = "CaseStudyZone"
    inst.n = len(d) - 1
    inst.x, inst.y = lon, lat
    inst.dem = d["Demand_m3"].to_numpy(float).copy()
    inst.dem[depot_row] = 0.0
    inst.rdy = d["Start_min"].to_numpy(float)
    inst.due = d["End_min"].to_numpy(float).copy()
    inst.srv = d["Service_min"].to_numpy(float).copy()
    inst.srv[depot_row] = 0.0
    inst.due[depot_row] = 960.0
    inst.Q = CAPACITY
    inst.maxveh = 60
    inst.d = D
    order = np.argsort(D, axis=1)
    inst.near = [[j for j in order[i] if j != i and j != 0][:20]
                 for i in range(len(d))]
    return inst, d


class ZoneEval(Eval):

    def _prop(self, seq, arcd, lev):
        rdy, due, srv = self.rdy, self.due, self.srv
        t = rdy[seq[0]] if seq[0] != 0 else 480.0
        m = len(seq) - 1
        for k in range(m):
            t += arcd[k] * TSCALE * TMUL[lev[k]]
            j = seq[k + 1]
            if j == 0:
                return (t <= due[0] + 1e-6), k
            if t < rdy[j]:
                t = rdy[j]
            if t > due[j] + 1e-6:
                return False, k
            t += srv[j]
        return True, -1


def sweep_plan(inst, ev):
    lat, lon = inst.y, inst.x
    ang = np.arctan2(lat - lat[0], lon - lon[0])
    cust = sorted(range(1, inst.n + 1), key=lambda c: ang[c])
    routes, cur, load = [], [], 0.0
    for c in cust:
        if load + inst.dem[c] > inst.Q + 1e-9:
            routes.append(cur); cur, load = [], 0.0
        cur.append(c); load += inst.dem[c]
    if cur:
        routes.append(cur)
    seq = []
    for r in routes:
        rem, out, here = set(r), [], 0
        while rem:
            nxt = min(rem, key=lambda c: inst.d[here][c])
            out.append(nxt); rem.discard(nxt); here = nxt
        seq.append(out)
    final = []
    for r in seq:
        piece = []
        for c in r:
            trial = piece + [c]
            if ev.route_eval(trial) is None:
                if piece:
                    final.append(piece)
                piece = [c]
            else:
                piece = trial
        if piece:
            final.append(piece)
    s = Solution()
    s.routes = [r for r in final if r]
    build(ev, s)
    return s


def stats(ev, sol):
    tot = 0.0
    sh = [0.0, 0.0, 0.0]
    lam = 0.0
    for it, r in zip(sol.info, sol.routes):
        if not r:
            continue
        for k, a in enumerate(it["arcd"]):
            sh[it["lev"][k]] += a
            lam += a * (it["load"][k] / ev.Q)
            tot += a
    if tot <= 0:
        return {}
    return dict(nv=sol.nv, dist=sol.dist, fuel=sol.fuel, emis=XI * sol.fuel,
                IF=sol.fuel / sol.dist, plow=100 * sh[0] / tot,
                pmed=100 * sh[1] / tot, phigh=100 * sh[2] / tot, lam=lam / tot)
