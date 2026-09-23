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

python runner.py --out results/results_main.csv --seeds 5 --budget 30 --procs 14

python runner.py --out results/results_sens.csv --seeds 3 --budget 30 --algos GLS,HGLS \
  --instances C101,C108,C203,C206,R104,R110,R202,R208,RC103,RC107,RC202,RC206 \
  --phis "1.0,1.0,1.0;0.93,1.0,1.09;0.85,1.0,1.18;0.75,1.0,1.30"

python runner.py --out results/results_anytime.csv --seeds 3 --budgets "3,8,20,60" \
  --instances C103,C107,C205,C208,R103,R109,R204,R210,RC102,RC106,RC203,RC207

python analyze.py results/results_main.csv results/results_sens.csv
python run_case.py
```

The main sweep is 1,400 runs and takes about an hour on 14 cores. Logs of the
runs reported in the paper are already in `results/`, so `analyze.py` can be run
without repeating them.

## Case study

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
