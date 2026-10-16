# eco-cvrptw-benchmark

Eco-speed benchmark instances, solver and experiment logs for the
emission-capacitated vehicle routing problem with time windows (E-CVRPTW).

Fuel on an arc depends on the payload carried and on the cruising speed
selected for that arc:

    rate(q, s) = [rho0 + (q / Q) * (rho1 - rho0)] * phi[s]      L/km
    F(i,j,s)   = rate * d(i,j)

with `rho0 = 0.20`, `rho1 = 0.50`, three cruising levels at 30, 50 and 70 km/h
and `phi = (0.85, 1.00, 1.18)`. Emissions are `E = 2.60 * F` kg CO2e. Capacity
and time windows are hard constraints; fleet size is minimized before fuel.

## Contents

| Path | What it is |
|---|---|
| `ecvrptw.py` | Model, instance reader, evaluator, construction heuristic |
| `algos.py` | GLS, IGLS, RGLS, HGLS and the ALNS-Eco baseline |
| `runner.py` | Parallel experiment driver |
| `analyze.py` | Statistics and figures |
| `zone_d.py`, `run_case.py` | Municipal collection-zone case study |
| `instances/` | The 56 Solomon instances used |
| `results/` | Raw logs of all 2,408 runs |
| `case_study/` | Case-study collection register and results |
| `figures/` | Figures 3 to 7 of the paper |

## Eco-speed instances

The instance files are the original Solomon files. The eco-speed extension is
applied at load time by `ecvrptw.py`: coordinates, demands, service times and
time windows are untouched, and the fuel coefficients, cruising levels and
carbon factor are the constants at the top of that file. Changing the model
therefore means changing those constants, not editing the instances.

The medium cruising level reproduces the classical Solomon convention in which
travel time equals distance, so the extension is backward compatible: with
`phi = (1, 1, 1)` and every arc at the medium level, the model reduces to the
standard distance-minimizing VRPTW.

## Reproducing the experiments

```
pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs every experiment of the paper in order and then the analysis;
each step logs to `results/logs/`. A single step can be run on its own, for
example `python run_all.py main`. The steps are:

| Step | What it runs |
|---|---|
| `main` | all 56 instances, five methods, five seeds, 30 s search budget |
| `case` | the municipal case study, five seeds, 45 s search budget |
| `budget` | 12 instances at search budgets of 3, 8, 20 and 60 s |
| `sens` | GLS and HGLS under four speed-factor vectors |
| `ablation` | distance routing at fixed speed, fuel routing at fixed speed, fuel routing with speed selection |
| `cmem` | the main comparison under the U-shaped speed-fuel curve of the comprehensive modal emission model |
| `analysis` | all statistics, tables and figures, written to `results/analysis_r1/` |

For every instance and seed the starting solution (Cheapest Arc construction
followed by fleet minimization) is built once, outside the budget, and all
methods start from a copy of it. The search budget is enforced on every route
evaluation, so no method can exceed it by more than one evaluation. Each run
records the construction time, the search time, the number of iterations, the
number of route evaluations and the number of accepted moves.

The full programme is about 4,700 runs and takes roughly four and a half hours
with 10 worker processes. Use one worker per physical core: running more
processes than cores lets them slow each other down, so that a wall-clock
budget no longer buys the same amount of search.

The logs of the original submission are kept in `results/results_*.csv`.

## Case study

Vehicles in the case study collect rather than deliver: each leaves the depot
empty and its payload grows with every container it empties.

`case_study/zone_d_nodes.csv` is the collection-point register supplied by the
municipal utility that operates the study zone: coordinates, service windows,
demands and service times for 110 containers plus the depot. The utility is not
named in the paper, which refers to the zone as the study zone.

Road distances were not available, so inter-node distance is the great-circle
distance between registered coordinates. Both plans compared use the same
metric. Reported kilometers are therefore lower bounds on road distance.

The operator's own route sheets were not available either, so the comparison
baseline is a sweep-and-nearest-neighbour plan built by `zone_d.sweep_plan`,
which stands in for an unoptimized hand-built route sheet. It is not a record
of what the utility actually drove.

## Citation

M. Alkahtani. Fuel-intensity dissociation in emission-aware vehicle routing: an eco-speed benchmark and controlled metaheuristic comparison.

## License

MIT for the code. The Solomon instances are redistributed under their original
terms.
