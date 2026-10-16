# eco-cvrptw-benchmark

Eco-speed benchmark instances, solver, experiment logs and analysis for the paper

> M. Alkahtani. *Separating distance and fuel-intensity effects in a controlled
> comparison of metaheuristics for emission-aware vehicle routing.*

The archive contains everything needed to check or repeat the study: the code,
the benchmark instances, the case-study register, the per-run output of every
experiment reported in the paper, the analysis scripts and their output, and
the logs of the original submission.

## The model

The emission-aware capacitated vehicle routing problem with time windows
(E-CVRPTW) minimizes fleet size first and fuel second. Fuel on an arc depends
on the payload carried and on the cruising level selected for that arc:

    rate(q, s) = [rho0 + (q / Q) * (rho1 - rho0)] * phi[s]      L/km
    F(i,j,s)   = rate * d(i,j)

with `rho0 = 0.20`, `rho1 = 0.50`, three cruising levels at 30, 50 and 70 km/h
and `phi = (0.85, 1.00, 1.18)`. Emissions are `E = 2.60 * F` kg CO2e. Capacity
and time windows are hard constraints. The model imposes no emission cap.

The fuel coefficients and speed factors are a controlled synthetic model chosen
to make the benchmark reproducible; they are not calibrated to any vehicle.
`ecvrptw.cmem_phi()` gives the alternative U-shaped speed factors of the
comprehensive modal emission model used in the paper's sensitivity analysis.

## Eco-speed instances and units

The instance files in `instances/` are the original 56 Solomon files. The
eco-speed extension is applied at load time by `ecvrptw.py`: coordinates,
demands, service times and time windows are untouched.

The benchmark is dimensionless, like the Solomon instances. One time unit is
the time needed to travel one distance unit at the medium level, so the travel
time of an arc of length `d` is `5d/3`, `d` and `5d/7` at the low, medium and
high levels. The medium level reproduces the original Solomon travel times. The
case study uses physical units instead: kilometers and minutes, with travel time
`60 d / u(s)`.

Load is propagated in delivery form for the benchmark (a vehicle leaves the depot
carrying the demand of its route) and in collection form for the case study (a
vehicle leaves empty and its payload grows with every container it empties).

## Contents

| Path | What it is |
|---|---|
| `ecvrptw.py` | Model, instance reader, route evaluator, construction heuristic |
| `algos.py` | GLS, IGLS, RGLS, HGLS, ALNS-Eco, fleet minimization, common starting solution |
| `runner.py` | Parallel experiment driver |
| `run_all.py` | Runs every main experiment of the paper in order, then the analysis |
| `zone_d.py`, `run_case.py`, `case_tables.py` | Municipal case study |
| `crossscore.py` | Ablation: fixed-speed solutions re-scored with speed selection allowed |
| `run_long.py`, `analyze_l120.py` | Series-2 instances at a 120 s search budget, compared with 30 s |
| `diag_speed.py` | Diagnostic of the search with speed selection |
| `analyze_r1.py`, `analyze.py` | Statistics, tables and figures |
| `instances/` | The 56 Solomon instances |
| `case_study/` | Case-study collection register and case-study results |
| `results/r1_*.csv` | Per-run output of every experiment in the paper |
| `results/analysis_r1/` | Statistics, tables and figures produced by the analysis |
| `results/logs/` | Console logs of every experiment |
| `results/original_submission/` | Logs and figures of the original submission (see below) |
| `figures/` | Figures 3 to 7 of the paper |

## Experiments and runs

| File | Experiment | Runs |
|---|---|---|
| `results/r1_main.csv` | 56 instances, five methods, five seeds, 30 s search budget | 1,400 |
| `results/r1_anytime.csv` | 12 instances, five methods, three seeds, budgets of 3, 8, 20 and 60 s | 720 |
| `results/r1_sens.csv` | 12 instances, GLS and HGLS, three seeds, four speed-factor vectors | 288 |
| `results/r1_cmem.csv` | as the main comparison, under the U-shaped speed-fuel curve | 1,400 |
| `results/r1_ablation.csv` | GLS: distance routing at medium speed, fuel routing at medium speed, fuel routing with speed selection | 840 |
| `results/r1_crossscore.csv` | GLS fuel routing at medium speed, re-scored with speed selection | 280 |
| `results/r1_series2_120.csv` | 27 series-2 instances, five methods, three seeds, 120 s search budget | 405 |
| `case_study/case_runs.csv` | case study, five methods, five seeds, 45 s search budget | 25 |
| `results/diag_speed.csv` | 12 instances, two seeds, three GLS configurations | 72 |

Every row is one run. In the search experiments each row records the instance, method, seed, speed-factor
vector, configuration and budget; the fleet size, distance, fuel, emissions,
fuel intensity, speed shares and distance-weighted payload of the best solution;
and the search effort: construction time of the starting solution, search time,
iterations, route evaluations and accepted moves. The re-scoring file records, for each
run, the fuel of the same routes at medium speed and with speed selection.

## Protocol

For every instance and seed the starting solution (Cheapest Arc construction
followed by fleet minimization to convergence) is built once, outside the
budget, and every method starts from a copy of it. The search budget is
enforced on every route evaluation, so no method can exceed it by more than one
evaluation. All runs used 10 worker processes, one per physical core, on an
Intel Core 7 240H processor (10 cores, 16 threads) with 32 GB of RAM under
Windows 11.

## Reproducing the results

```
pip install -r requirements.txt
python run_all.py                 # main, case, budget, sens, ablation, cmem, analysis, case tables
python crossscore.py
python run_long.py && python analyze_l120.py
```

`python analyze_r1.py` and `python case_tables.py` recompute every statistic,
table and figure from the files in `results/` and `case_study/` without re-running
any experiment. The full set of experiments takes about seven hours with 10
worker processes. Use one worker per physical core: more processes than cores
slow each other down, so that a wall-clock budget no longer buys the same
amount of search.

Software used for the reported runs: Python 3.11.9, numpy 2.4.4, pandas 2.3.3,
scipy 1.17.1, matplotlib 3.10.8 (pinned in `requirements.txt`).

## What is and is not reproducible exactly

- The starting solution of each instance and seed is deterministic and is
  reproduced exactly.
- The search is limited by wall-clock time, so a repeated run does not perform
  exactly the same number of iterations and does not return a bit-identical
  solution. Repeated runs give similar results, and the statistics in the paper
  are computed over five independent runs per instance and method.
- The per-run results of every benchmark run are archived, but not their route
  sequences; route sequences are archived for the case-study runs only
  (`case_study/case_runs.csv`).

## Original submission and the budget correction

`results/original_submission/` keeps the logs and figures of the original
submission unchanged, so that the correction made during the revision can be
inspected. In those runs the 30 s budget clock started before construction but
was checked only inside the search loop, so construction and the initial fleet
minimization always ran to completion. On the long-route series-2 classes these
steps alone exceeded the budget, and 288 of the 1,400 main runs performed no
search. All experiments of the revised paper were run again with the corrected
protocol described above. The case-study files in that folder use the delivery
load profile of the original submission; the revised case study uses collection
loads.

## Case study

`case_study/zone_d_nodes.csv` is the collection-point register supplied by the
municipal utility that operates the study zone: coordinates, service windows,
demands and service times for 110 containers plus the depot. It is released with
the operator's permission. The utility is not named in the paper.

Road distances were not available, so inter-node distance is the great-circle
distance between registered coordinates. The operator's route sheets were not
available either, so the comparison baseline is a constructed
sweep-and-nearest-neighbor plan (`zone_d.sweep_plan`); it is not a record of
what the utility drove. Vehicle capacity (5.0 m3) is an assumption.

## License

Code: MIT (see `LICENSE`). Results, logs and the case-study register: CC BY 4.0.
The Solomon instances are redistributed under their original terms.
