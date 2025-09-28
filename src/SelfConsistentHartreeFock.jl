module SelfConsistentHartreeFock

export Option, Params, solve, solve_meanfield, converged, sweep_delta

include("types.jl")
include("solver.jl")
include("continuation.jl")

end # module
