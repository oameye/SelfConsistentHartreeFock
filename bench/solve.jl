using BenchmarkTools
using SelfConsistentHartreeFock
using SelfConsistentHartreeFock: solve, solve_meanfield

using .BenchFixtures: ensure_group!, meanfield_fixture, solver_fixture

function benchmark_solver!(suite::BenchmarkGroup)
    solver = solver_fixture()
    params = solver.params
    config = solver.config
    α0 = solver.α0

    meanfield = meanfield_fixture()

    solve(α0, params, config)
    solve_meanfield(meanfield.α0, meanfield.params, meanfield.config)

    solver_group = ensure_group!(suite, "solver")

    solver_group["solve_full"] = @benchmarkable solve($α0, $params, $config) seconds = 5
    solver_group["solve_default_guess"] =
        @benchmarkable solve($params; config=$config) seconds = 5

    meanfield_group = ensure_group!(solver_group, "meanfield")
    meanfield_group["solve_meanfield"] = @benchmarkable solve_meanfield(
        $(meanfield.α0), $(meanfield.params), $(meanfield.config)
    ) seconds = 5

    return nothing
end
