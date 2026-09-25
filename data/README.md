# Input files

Every model is defined by three plain-text matrices. They are **whitespace
delimited** (not comma delimited), may contain `#` comment lines, and are read
by `propgen.read_matrix`, which tolerates a UTF-8 byte-order mark and CRLF line
endings.

| File | Shape | Meaning |
| --- | --- | --- |
| `adjacency.txt` | `(Ng, Ng)` | genotype mutation graph `A`. Symmetric, 0/1. Every genotype needs at least one neighbour; a single-genotype model uses a self-loop. |
| `pheno_probs.txt` | `(Ng, Np)` | genotype-to-phenotype map `phi`. Row `g` is the probability distribution over phenotypes for genotype `g`, and must sum to 1. |
| `repro_probs.txt` | `(Np,)` | probability that an individual expressing each phenotype divides in one generation. Must lie in `[0, 1]`. |

## Index conventions

Genotypes are rows, phenotypes are columns. Flattened `(g, p)` vectors — most
importantly the equilibrium frequency `f_eq` returned by `propgen.equilibrium`
— are **row-major**, so element `g * Np + p` is the pair `(g, p)`.

A landscape is validated on construction. Malformed input raises `ValueError`
immediately rather than producing silently wrong dynamics.

## What is here

| Directory | Model |
| --- | --- |
| `bridge/` | 3-genotype path `0 -- 1 -- 2`; genotype 1 is the fitness valley. |
| `buoy/` | 2 genotypes, 2 phenotypes; genotype 0 acts as a source for the low-fitness phenotype. |
| `absfit/` | 2 genotypes with both a deterministic (`pheno_probs_dgp.txt`) and a probabilistic (`pheno_probs_prgp.txt`) map. |
| `meanfit/` | 2 genotypes, 2 phenotypes, both mixing. |
| `persister/` | 1 genotype, 2 phenotypes, with separate maps and rates for the antibiotic-free and antibiotic environments. |

Some entries are the starting point of a sweep rather than a fixed value —
`bridge/pheno_probs.txt` row 1 and `absfit/repro_probs.txt`, for example. Each
file says so in its header comment, and the sweep that varies it is named
there.

## Using your own

Write the three files and point a config at them:

```yaml
landscape:
  adjacency:   my_model/adjacency.txt
  pheno_probs: my_model/pheno_probs.txt
  repro_probs: my_model/repro_probs.txt
```

Paths are resolved relative to the config file. Or build a `Landscape`
directly from NumPy arrays without any files — see the repository README.
