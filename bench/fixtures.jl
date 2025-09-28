module BenchFixtures

using BenchmarkTools: BenchmarkGroup
using SelfConsistentHartreeFock
using SelfConsistentHartreeFock:
    Params,
    SolverConfig,
    SolverState,
    step_once,
    sweep_delta
using SelfConsistentHartreeFock: _residual_norm

export solver_fixture,
    meanfield_fixture,
    step_control_fixture,
    continuation_fixture,
    ensure_group!

function ensure_group!(suite::BenchmarkGroup, key::AbstractString)::BenchmarkGroup
    if haskey(suite, key)
        group = suite[key]
        group isa BenchmarkGroup || error("Existing entry for $(key) is not a BenchmarkGroup")
        return group
    else
        suite[key] = BenchmarkGroup()
        return suite[key]
    end
end

function solver_fixture()
    params = Params(; Δ=-1.2, K=1.0, F=0.9 + 0.0im)
    config = SolverConfig(; max_iter=2000, tol=1e-10, step_fraction=0.4, backtrack=6)
    α0 = ComplexF64(0.1 + 0.0im)
    return (params=params, config=config, α0=α0)
end

function meanfield_fixture()
    base = solver_fixture()
    config = SolverConfig(base.config; keep_nm_zero=true)
    return (params=base.params, config=config, α0=base.α0)
end

function step_control_fixture()
    base = solver_fixture()
    params = base.params
    config = base.config
    α0 = base.α0
    initial = SolverState(α0, 0.15, ComplexF64(0.05))
    candidate = step_once(params, initial, config)
    residual = _residual_norm(initial, candidate)
    return (
        params=params,
        config=config,
        state=initial,
        candidate=candidate,
        residual=residual,
        fraction=config.step.fraction,
    )
end

function continuation_fixture()
    base = solver_fixture()
    params = base.params
    config = base.config
    Δs = collect(range(-1.5, 0.3; length=32))
    starting = Params(; Δ=first(Δs), K=params.K, F=params.F)
    α0 = base.α0
    sweep_delta(Δs, starting, α0, config)
    return (Δs=Δs, params=params, base=starting, α0=α0, config=config)
end

end # module
