using BenchmarkTools
using SelfConsistentHartreeFock

const SUITE = BenchmarkGroup()

include("fixtures.jl")
include("step_control.jl")
include("solve.jl")
include("continuation.jl")

benchmark_step_control!(SUITE)
benchmark_solver!(SUITE)
benchmark_continuation!(SUITE)

BenchmarkTools.tune!(SUITE)
results = BenchmarkTools.run(SUITE; verbose=true)
display(median(results))

BenchmarkTools.save(joinpath(@__DIR__, "benchmarks_output.json"), median(results))
