module SelfConsistentHartreeFock

export Params,
    SolverConfig,
    ContinuationTrace,
    solve,
    solve_meanfield,
    sweep_delta,
    converged,
    continuation_trace

include("types.jl")
include("physics.jl")
include("state.jl")
include("step_control.jl")
include("solver.jl")
include("continuation.jl")

end # module
