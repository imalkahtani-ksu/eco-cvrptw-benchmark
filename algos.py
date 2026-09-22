import math, random, time
from ecvrptw import Eval, Solution, build, cam_construct, XI

OPS = ("2opt", "3opt", "relocate", "oropt", "crossexch")
NOPS = len(OPS)


class Search:
    def __init__(self, ev, sol, rng, alpha_mode="adaptive", omega=0.3):
        self.ev, self.rng = ev, rng
        self.sol = sol
        self.pen = {}
        self.alpha = 0.0
        self.alpha_mode = alpha_mode
        self.omega = omega
        self.best = sol.clone()
        self.nfeat = max(1, ev.I.n)
        self.set_alpha(sol)

    def set_alpha(self, sol):
        if self.alpha_mode == "fixed" and self.alpha > 0:
            return
        self.alpha = self.omega * sol.fuel / self.nfeat

    def route_pen(self, r):
        if not r:
            return 0.0
        pen, dd = self.pen, self.ev.dd
        s = 0.0
        prev = 0
        for c in r:
            k = (prev, c) if prev < c else (c, prev)
            p = pen.get(k)
            if p:
                s += p * dd[prev][c]
            prev = c
        k = (0, prev)
        p = pen.get(k)
        if p:
            s += p * dd[prev][0]
        return s

    def pcost(self, info, r):
        return info["fuel"] + self.alpha * self.route_pen(r)

    def total_pcost(self, sol):
        return sum(self.pcost(sol.info[i], sol.routes[i])
                   for i in range(len(sol.routes)))

    def penalise(self, sol):
        pen, dd = self.pen, self.ev.dd
        best, arcs = -1.0, []
        for r in sol.routes:
            if not r:
                continue
            prev = 0
            for c in list(r) + [0]:
                k = (prev, c) if prev < c else (c, prev)
                u = dd[prev][c] / (1.0 + pen.get(k, 0))
                if u > best + 1e-12:
                    best, arcs = u, [k]
                elif u > best - 1e-12:
                    arcs.append(k)
                prev = c
        for k in arcs[:3]:
            pen[k] = pen.get(k, 0) + 1

    def accept(self, sol):
        if (sol.nv, round(sol.fuel, 6)) < (self.best.nv, round(self.best.fuel, 6)):
            self.best = sol.clone()
            return True
        return False


def _apply(S, idxs, newroutes):
    ev, sol = S.ev, S.sol
    olds = sum(S.pcost(sol.info[i], sol.routes[i]) for i in idxs)
    infos = []
    for r in newroutes:
        it = ev.route_eval(r)
        if it is None:
            return None
        infos.append(it)
    news = sum(it["fuel"] for it in infos) + S.alpha * sum(
        S.route_pen(r) for r in newroutes)
    if news >= olds - 1e-9:
        return None
    return (olds - news, infos)


def op_2opt(S):
    sol, rng = S.sol, S.rng
    order = list(range(len(sol.routes)))
    rng.shuffle(order)
    for ri in order:
        r = sol.routes[ri]
        L = len(r)
        if L < 4:
            continue
        for a in range(L - 1):
            for b in range(a + 2, min(L, a + 20)):
                nr = r[:a + 1] + r[a + 1:b + 1][::-1] + r[b + 1:]
                res = _apply(S, [ri], [nr])
                if res:
                    g, infos = res
                    sol.routes[ri], sol.info[ri] = nr, infos[0]
                    sol.recompute(S.ev)
                    return g
    return 0.0


def op_3opt(S):
    sol, rng = S.sol, S.rng
    order = list(range(len(sol.routes)))
    rng.shuffle(order)
    for ri in order:
        r = sol.routes[ri]
        L = len(r)
        if L < 5:
            continue
        for a in range(L):
            for ln in (1, 2, 3):
                if a + ln > L:
                    break
                chain = r[a:a + ln]
                rest = r[:a] + r[a + ln:]
                for pos in range(len(rest) + 1):
                    if pos == a:
                        continue
                    for ch in (chain, chain[::-1]):
                        nr = rest[:pos] + ch + rest[pos:]
                        res = _apply(S, [ri], [nr])
                        if res:
                            g, infos = res
                            sol.routes[ri], sol.info[ri] = nr, infos[0]
                            sol.recompute(S.ev)
                            return g
    return 0.0


def op_relocate(S):
    sol, rng, I = S.sol, S.rng, S.ev.I
    where = {}
    for ri, r in enumerate(sol.routes):
        for k, c in enumerate(r):
            where[c] = (ri, k)
    custs = list(where)
    rng.shuffle(custs)
    for c in custs:
        ri, k = where[c]
        r1 = sol.routes[ri]
        nr1 = r1[:k] + r1[k + 1:]
        for nb in I.near[c]:
            if nb not in where:
                continue
            rj, kk = where[nb]
            if rj == ri:
                continue
            r2 = sol.routes[rj]
            for pos in (kk, kk + 1):
                nr2 = r2[:pos] + [c] + r2[pos:]
                res = _apply(S, [ri, rj], [nr1, nr2])
                if res:
                    g, infos = res
                    sol.routes[ri], sol.info[ri] = nr1, infos[0]
                    sol.routes[rj], sol.info[rj] = nr2, infos[1]
                    sol.routes = [x for x in sol.routes if x] or [[]]
                    build(S.ev, sol)
                    return g
    return 0.0


def op_oropt(S):
    sol, rng, I = S.sol, S.rng, S.ev.I
    order = list(range(len(sol.routes)))
    rng.shuffle(order)
    for ri in order:
        r1 = sol.routes[ri]
        if len(r1) < 2:
            continue
        for a in range(len(r1) - 1):
            for ln in (2, 3):
                if a + ln > len(r1):
                    break
                chain = r1[a:a + ln]
                nr1 = r1[:a] + r1[a + ln:]
                head = chain[0]
                for nb in I.near[head]:
                    rj = None
                    for jj, rr in enumerate(sol.routes):
                        if jj != ri and nb in rr:
                            rj, kk = jj, rr.index(nb)
                            break
                    if rj is None:
                        continue
                    r2 = sol.routes[rj]
                    for pos in (kk, kk + 1):
                        nr2 = r2[:pos] + chain + r2[pos:]
                        res = _apply(S, [ri, rj], [nr1, nr2])
                        if res:
                            g, infos = res
                            sol.routes[ri], sol.info[ri] = nr1, infos[0]
                            sol.routes[rj], sol.info[rj] = nr2, infos[1]
                            sol.routes = [x for x in sol.routes if x] or [[]]
                            build(S.ev, sol)
                            return g
    return 0.0


def op_crossexch(S):
    sol, rng = S.sol, S.rng
    idx = [i for i, r in enumerate(sol.routes) if r]
    if len(idx) < 2:
        return 0.0
    pairs = [(a, b) for ii, a in enumerate(idx) for b in idx[ii + 1:]]
    rng.shuffle(pairs)
    for ri, rj in pairs[:14]:
        r1, r2 = sol.routes[ri], sol.routes[rj]
        for a in range(len(r1)):
            for la in (1, 2, 3):
                if a + la > len(r1):
                    break
                for b in range(len(r2)):
                    for lb in (1, 2, 3):
                        if b + lb > len(r2):
                            break
                        A, B = r1[a:a + la], r2[b:b + lb]
                        nr1 = r1[:a] + B + r1[a + la:]
                        nr2 = r2[:b] + A + r2[b + lb:]
                        res = _apply(S, [ri, rj], [nr1, nr2])
                        if res:
                            g, infos = res
                            sol.routes[ri], sol.info[ri] = nr1, infos[0]
                            sol.routes[rj], sol.info[rj] = nr2, infos[1]
                            sol.routes = [x for x in sol.routes if x]
                            build(S.ev, sol)
                            return g
    return 0.0


def route_elimination(ev, sol, rng, tries=99):
    changed = False
    for _ in range(tries):
        live = [i for i, r in enumerate(sol.routes) if r]
        if len(live) < 2:
            break
        got = False
        for target in sorted(live, key=lambda i: len(sol.routes[i])):
            cust = sol.routes[target][:]
            trial = [r[:] for i, r in enumerate(sol.routes) if i != target and r]
            ok = True
            for c in cust:
                best = None
                for ri, r in enumerate(trial):
                    b = ev.route_eval(r)
                    if b is None:
                        continue
                    for pos in range(len(r) + 1):
                        it = ev.route_eval(r[:pos] + [c] + r[pos:])
                        if it is None:
                            continue
                        d = it["fuel"] - b["fuel"]
                        if best is None or d < best[0]:
                            best = (d, ri, pos)
                if best is None:
                    ok = False
                    break
                trial[best[1]].insert(best[2], c)
            if ok:
                sol.routes = trial
                build(ev, sol)
                changed = got = True
                break
        if not got:
            break
    return changed


OPFN = {"2opt": op_2opt, "3opt": op_3opt, "relocate": op_relocate,
        "oropt": op_oropt, "crossexch": op_crossexch}


def _stats(ev, sol):
    tot = 0.0
    sh = [0.0, 0.0, 0.0]
    lam = 0.0
    Q = ev.Q
    for it, r in zip(sol.info, sol.routes):
        if not r:
            continue
        for k, a in enumerate(it["arcd"]):
            sh[it["lev"][k]] += a
            lam += a * (it["load"][k] / Q)
            tot += a
    if tot <= 0:
        return dict(plow=0.0, pmed=0.0, phigh=0.0, lam=0.0, IF=0.0)
    return dict(plow=100 * sh[0] / tot, pmed=100 * sh[1] / tot,
                phigh=100 * sh[2] / tot, lam=lam / tot,
                IF=sol.fuel / sol.dist if sol.dist > 0 else 0.0)


def run_gls(ev, seed, variant, max_iter=100000, max_sec=15.0, theta=0.01,
            seg=40, rho=0.2):
    rng = random.Random(seed)
    t0 = time.time()
    sol = cam_construct(ev, rng)
    route_elimination(ev, sol, rng)
    mode = "fixed" if variant == "GLS" else "adaptive"
    S = Search(ev, sol, rng, alpha_mode=mode)
    S.best = sol.clone()

    w = {o: 1.0 for o in OPS}
    ssum = {o: 0.0 for o in OPS}
    scnt = {o: 0 for o in OPS}
    rr = 0
    it = 0
    since_improve = 0
    failed = set()

    while it < max_iter and time.time() - t0 < max_sec:
        it += 1
        if variant in ("GLS", "IGLS"):
            op = OPS[rr % NOPS]
        elif variant == "RGLS":
            op = rng.choice(OPS)
        else:
            tot = sum(w.values())
            x, acc, op = rng.random() * tot, 0.0, OPS[0]
            for o in OPS:
                acc += w[o]
                if x <= acc:
                    op = o
                    break

        tic = time.perf_counter()
        gain = OPFN[op](S)
        cpu = max(1e-6, time.perf_counter() - tic)

        improved_best = False
        if gain > 0.0:
            S.sol.recompute(ev)
            improved_best = S.accept(S.sol)
            failed.discard(op)
            since_improve = 0
            if variant == "IGLS":
                pass
            elif variant == "GLS":
                rr += 1
        else:
            failed.add(op)
            since_improve += 1
            if variant in ("GLS", "IGLS"):
                rr += 1
            if len(failed) >= NOPS:
                S.penalise(S.sol)
                build(ev, S.sol)
                failed.clear()
                if S.alpha_mode == "adaptive":
                    S.set_alpha(S.sol)

        if variant == "HGLS":
            scnt[op] += 1
            ssum[op] += (4.0 if improved_best else (2.0 if gain > 0 else 0.0)) / cpu
            if it % seg == 0:
                mx = max(1e-9, max(ssum[o] / max(1, scnt[o]) for o in OPS))
                for o in OPS:
                    obs = (ssum[o] / scnt[o]) if scnt[o] else 0.0
                    w[o] = max(0.08, (1 - rho) * w[o] + rho * (obs / mx))
                    ssum[o], scnt[o] = 0.0, 0
                if since_improve > seg:
                    w["3opt"] = min(3.0, w["3opt"] + 0.35)
                    w["crossexch"] = min(3.0, w["crossexch"] + 0.35)

        if it % 120 == 0:
            if route_elimination(ev, S.sol, rng, tries=3):
                S.sol.recompute(ev)
                S.accept(S.sol)
        if since_improve > 1500:
            break

    best = S.best
    best.routes = [r for r in best.routes if r]
    build(ev, best)
    route_elimination(ev, best, rng)
    st = _stats(ev, best)
    return dict(nv=best.nv, dist=best.dist, fuel=best.fuel,
                emis=XI * best.fuel, runtime=time.time() - t0,
                iters=it, **st)


def _greedy_insert(ev, routes, pool, rng, regret=2):
    I = ev.I
    pool = list(pool)
    while pool:
        best = None
        for c in pool:
            costs = []
            for ri, r in enumerate(routes):
                base = ev.route_eval(r)
                if base is None:
                    continue
                for pos in range(len(r) + 1):
                    it = ev.route_eval(r[:pos] + [c] + r[pos:])
                    if it is None:
                        continue
                    costs.append((it["fuel"] - base["fuel"], ri, pos))
            costs.sort()
            if not costs:
                val, pick = -1e18, (None, len(routes), 0)
            elif len(costs) < regret:
                val, pick = 1e12 - costs[0][0], (costs[0][1], costs[0][2])
                pick = (pick[0], pick[1])
            else:
                val = costs[regret - 1][0] - costs[0][0]
                pick = (costs[0][1], costs[0][2])
            if best is None or val > best[0]:
                best = (val, c, pick, bool(costs))
        _, c, pick, feasible = best
        if feasible:
            ri, pos = pick
            routes[ri].insert(pos, c)
        else:
            routes.append([c])
        pool.remove(c)
    return routes


def run_alns(ev, seed, max_iter=4000, max_sec=12.0):
    rng = random.Random(seed)
    t0 = time.time()
    cur = cam_construct(ev, rng)
    route_elimination(ev, cur, rng)
    best = cur.clone()
    I = ev.I
    DES = ("random", "worst", "related")
    wd = {d: 1.0 for d in DES}
    T = max(1e-6, 0.05 * cur.fuel)
    it = 0
    while it < max_iter and time.time() - t0 < max_sec:
        it += 1
        cand = cur.clone()
        allc = [c for r in cand.routes for c in r]
        if len(allc) < 5:
            break
        q = rng.randint(max(2, len(allc) // 25), max(3, len(allc) // 8))
        tot = sum(wd.values())
        x, acc, dsel = rng.random() * tot, 0.0, DES[0]
        for d in DES:
            acc += wd[d]
            if x <= acc:
                dsel = d
                break
        if dsel == "random":
            rem = rng.sample(allc, q)
        elif dsel == "worst":
            sc = []
            for ri, r in enumerate(cand.routes):
                for k, c in enumerate(r):
                    nr = r[:k] + r[k + 1:]
                    a, b = ev.route_eval(r), ev.route_eval(nr)
                    if a and b:
                        sc.append((a["fuel"] - b["fuel"], c))
            sc.sort(reverse=True)
            rem = [c for _, c in sc[:q]]
        else:
            s0 = rng.choice(allc)
            rem = [s0] + [c for c in I.near[s0] if c in allc][:q - 1]
        rs = set(rem)
        cand.routes = [[c for c in r if c not in rs] for r in cand.routes]
        cand.routes = [r for r in cand.routes if r]
        _greedy_insert(ev, cand.routes, rem, rng)
        if not build(ev, cand):
            continue
        d = (cand.nv, cand.fuel) < (cur.nv, cur.fuel)
        if d or rng.random() < math.exp(-(cand.fuel - cur.fuel) / T):
            cur = cand
            wd[dsel] = 0.9 * wd[dsel] + 0.1 * (3.0 if d else 1.0)
        else:
            wd[dsel] = 0.9 * wd[dsel] + 0.1 * 0.2
        if it % 60 == 0:
            route_elimination(ev, cur, rng, tries=3)
        if (cur.nv, cur.fuel) < (best.nv, best.fuel):
            best = cur.clone()
        T *= 0.999
    best.routes = [r for r in best.routes if r]
    build(ev, best)
    route_elimination(ev, best, rng)
    st = _stats(ev, best)
    return dict(nv=best.nv, dist=best.dist, fuel=best.fuel,
                emis=XI * best.fuel, runtime=time.time() - t0,
                iters=it, **st)
