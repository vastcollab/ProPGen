# Paper experiments

Each directory holds everything needed for one result: the sweep definition,
the figure notebook, any analysis code, and a README with the exact commands
and measured runtimes.

| Result | Directory | Data needed | Full-scale runtime (1 core) |
| --- | --- | --- | --- |
| Rugged-landscape schematic | `landscape_diagram/` | none | 3 s |
| Persister resuscitation and partitioning | `persister/` | none (closed form) | 2 s |
| Mean fitness from a non-equilibrium start | `meanfit/` | 5 runs | 1 s |
| Absolute-fitness non-invariance (DGP vs PrGP) | `absfit/` | 60 runs | 16 s |
| Phenotypic buoying | `buoy/` | 110 runs | 30 s |
| Phenotypic bridges / valley crossing | `bridge/` | 6,300 runs | 54 core-hours |

Aggregated summaries for every experiment are committed under `results/`, so
all figure notebooks run from a fresh clone without re-running any simulation.

Re-running a sweep writes per-trial output to `results/<name>/raw/`, which is
gitignored.
