module test_correctness

using Test
using SelfConsistentHartreeFock

using JLD2
current_path = @__DIR__
data = load(joinpath(@__DIR__, "correction_test.jld2"))
Δrange = data["Δrange"]

αv = map(Δrange) do _Δ
    p = Params(; Δ=_Δ, K=1.0, F=0.9)
    res = solve_meanfield(p)
    abs(res.α)
end

@test isapprox(data["branch"], αv)

end
