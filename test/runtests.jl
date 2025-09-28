using Test

@testset "SelfConsistentHartreeFock.jl" begin
    @testset "solver" include("test_solver.jl")

    @testset "correctness" include("test_correctness.jl")
    @testset "continuation" include("test_continuation.jl")
    @testset "bifurcation-regression" include("test_bifurcation_regression.jl")
    @testset "code-linting" include("test_code_linting.jl")
    @testset "best-practises" include("test_best_practises.jl")
end
