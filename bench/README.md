# Benchmarks

This folder hosts `BenchmarkTools` suites for `SelfConsistentHartreeFock`.

## Structure

- `runbenchmarks.jl` – entrypoint that assembles the suites, tunes them, runs them, and
  persists the median results as `benchmarks_output.json`.
- `fixtures.jl` – shared helpers for constructing reusable benchmark fixtures.
- `step_control.jl` – micro-benchmarks around the adaptive step controller.
- `solve.jl` – benchmarks that time the full HFB solver and the mean-field variant.
- `continuation.jl` – benchmarks targeting continuation sweeps across detuning values.
- `Project.toml` – local environment declaring `BenchmarkTools` and this package as deps.

## Running

Instantiate the benchmark environment and run the suite:

```bash
julia --project=bench -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=bench bench/runbenchmarks.jl
```

The second command prints a tuned run and saves the medians to
`bench/benchmarks_output.json`.
