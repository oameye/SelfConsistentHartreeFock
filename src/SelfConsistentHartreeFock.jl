module SelfConsistentHartreeFock

export Option, Params, solve, solve_meanfield, diagnostics, converged, sweep_delta

include("types.jl")
include("solver.jl")
include("diagnostics.jl")
include("continuation.jl")

end # module
