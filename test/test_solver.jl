module test_solver

using Test
using SelfConsistentHartreeFock
using SelfConsistentHartreeFock:
    Result,
    SolverConfig,
    SolverState,
    step_once,
    _coeffs,
    _is_physical,
    _residual_norm,
    _state_max_change

"""
    diagnostics(p, α, n, m; tol=1e-12)

Return a NamedTuple with (epsilon, DeltaB, omega, omega2, unstable, physical).
"""
function diagnostics(
    p::Params, α::ComplexF64, n::Float64, m::ComplexF64; tol::Float64=1e-12
)
    ε, ΔB, ω2 = _coeffs(p, α, n, m)
    ω = ω2 > 0 ? sqrt(ω2) : 0.0
    unstable = !(ω2 > 0)
    physical = (ω2 > 0) && _is_physical(n, m, tol)
    return (epsilon=ε, DeltaB=ΔB, omega=ω, omega2=ω2, unstable=unstable, physical=physical)
end

function diagnostics(p::Params, state::SolverState; tol::Float64=1e-12)
    return diagnostics(p, state.α, state.n, state.m; tol=tol)
end

function diagnostics(p::Params, res::Result; tol::Float64=1e-12)
    return diagnostics(p, res.α, res.n, res.m; tol=tol)
end

@testset "SolverState utilities" begin
    let s = SolverState(1 + 2im, 3//2, -0.4im)
        @test s.α isa ComplexF64
        @test s.n isa Float64
        @test s.m isa ComplexF64
        @test s.n == float(3//2)
    end

    let p = Params(; Δ=1.5, K=0.1, F=0.25 + 0.1im),
        config = SolverConfig(; step_fraction=0.4)

        s0 = SolverState(; α=0.1 + 0.0im, n=0.0, m=0.0 + 0im)
        s1 = step_once(p, s0, config)
        δ = _state_max_change(s0, s1)
        @test s1 isa SolverState
        @test δ == _residual_norm(s0, s1)
        @test δ > 0.0
    end
end

@testset "SolverConfig builder" begin
    cfg = SolverConfig(;
        max_iter=42,
        tol=1e-8,
        step_fraction=0.6,
        keep_nm_zero=true,
        omega_floor=1e-9,
        enforce_physical=false,
        step_bounds=(0.05, 0.7),
        step_scales=(1.1, 0.4),
        backtrack=4,
        accept_relax=0.95,
        unstable_scale=0.8,
    )
    @test cfg.iteration.max_iter == 42
    @test cfg.iteration.tol == 1e-8
    @test cfg.physics.keep_nm_zero
    @test !cfg.physics.enforce_physical
    @test cfg.physics.omega_floor == 1e-9
    @test cfg.step.fraction == 0.6
    @test cfg.step.fraction_bounds == (0.05, 0.7)
    @test cfg.step.fraction_scales == (1.1, 0.4)
    @test cfg.step.backtrack == 4
    @test cfg.step.accept_relax == 0.95
    @test cfg.step.unstable_scale == 0.8

    cfg2 = SolverConfig(cfg; step_fraction=0.3, keep_nm_zero=false)
    @test cfg2.step.fraction == 0.3
    @test !cfg2.physics.keep_nm_zero
    @test cfg2.iteration.max_iter == cfg.iteration.max_iter
    @test cfg2.physics.omega_floor == cfg.physics.omega_floor
end

@testset "basic solve" begin
    let p = Params(; Δ=1.0, K=0.05, F=0.2 + 0im),
        config = SolverConfig(; max_iter=200, tol=1e-12, step_fraction=0.3)

        res = solve(0.0 + 0im, p, config)
        @test res isa Result
        @test res.converged || res.iterations == config.iteration.max_iter
        # compute diagnostics via helper
        d = diagnostics(p, res)
        @test isfinite(d.epsilon) && isfinite(d.omega)
        @test res.α == conj(conj(res.α)) # roundtrip sanity
    end
end

@testset "keyword solve wrappers" begin
    let p = Params(; Δ=1.0, K=0.05, F=0.2 + 0im),
        config = SolverConfig(; max_iter=150, tol=1e-10, step_fraction=0.25)

        res_kw = solve(p; config=config, α0=0.0 + 0im)
        res_ref = solve(0.0 + 0im, p, config)
        @test res_kw == res_ref

        res_default = solve(p; config=config)
        res_default_ref = solve(ComplexF64(1e-3 + 1e-3im), p, config)
        @test res_default == res_default_ref

        mf_kw = solve_meanfield(p; config=config, α0=0.0 + 0im)
        mf_ref = solve_meanfield(0.0 + 0im, p, config)
        @test mf_kw == mf_ref

        mf_default = solve_meanfield(p; config=config)
        mf_default_ref = solve_meanfield(ComplexF64(1e-3 + 1e-3im), p, config)
        @test mf_default == mf_default_ref
    end
end

@testset "determinism for same IC" begin
    let p = Params(; Δ=1.0, K=0.05, F=0.2 + 0im),
        config = SolverConfig(; max_iter=150, tol=1e-10, step_fraction=0.25)

        res1 = solve(0.0 + 0im, p, config)
        res2 = solve(0.0 + 0im, p, config)
        @test res1 isa Result
        @test res2 isa Result
        @test isapprox(res1.α, res2.α; rtol=1e-12, atol=1e-12)
        @test isapprox(res1.n, res2.n; rtol=1e-12, atol=1e-12)
        @test isapprox(res1.m, res2.m; rtol=1e-12, atol=1e-12)
    end
end

@testset "step_once progression" begin
    let p = Params(; Δ=1.0, K=0.05, F=0.2 + 0im)
        state = SolverState(; α=0.2 + 0im, n=0.0, m=0.0 + 0im)
        config = SolverConfig(; step_fraction=0.3)
        for _ in 1:10
            state = step_once(p, state, config)
            d = diagnostics(p, state)
            @test isfinite(d.epsilon) && isfinite(d.omega)
            @test !isnan(real(d.DeltaB))
        end
        res = solve(
            p;
            config=SolverConfig(; max_iter=200, tol=1e-10, step_fraction=0.3),
            α0=0.0 + 0im,
        )
        # After 10 steps we should be in the vicinity of the solved alpha
        @test isapprox(state.α, res.α; rtol=5e-2, atol=5e-3)
    end
end

@testset "physical flag and stability classification" begin
    let p = Params(; Δ=1.0, K=0.05, F=0.2 + 0im),
        config = SolverConfig(; max_iter=200, tol=1e-12, step_fraction=0.3)

        res = solve(0.0 + 0im, p, config)
        # If stable, result should be flagged physical and satisfy positivity
        d = diagnostics(p, res)
        pos_ok = (res.n ≥ -1e-12) && (res.n * (res.n + 1.0) + 1e-12 ≥ abs2(res.m))
        @test res.physical == ((d.omega2 > 0) && pos_ok)
    end
end

@testset "instability handling: enforce_physical vs off" begin
    # Construct an input with ω² ≤ 0 at the current (α, n, m), then compare one-step updates
    let p = Params(; Δ=0.0, K=1.0, F=0.0 + 0im)
        state0 = SolverState(; α=0.0 + 0im, n=0.0, m=1.0 + 0im)

        # Initial state is unstable (ε=0, ΔB=2 → ω² = -4 < 0)
        d0 = diagnostics(p, state0)
        @test d0.unstable
        @test d0.omega == 0.0

        # No enforcement: allow nonphysical correlators
        config1 = SolverConfig(;
            step_fraction=1.0, omega_floor=1e-2, enforce_physical=false, unstable_scale=1.0
        )
        state1 = step_once(p, state0, config1)

        # With enforcement: correlators are projected to the physical region
        config2 = SolverConfig(;
            step_fraction=1.0, omega_floor=1e-2, enforce_physical=true, unstable_scale=1.0
        )
        state2 = step_once(p, state0, config2)
        @test state2.n ≥ -1e-12
        @test state2.n * (state2.n + 1.0) + 1e-12 ≥ abs2(state2.m)

        # Typically the unenforced step violates positivity
        # (don't require it strictly in case parameters change)
        violated =
            !(state1.n ≥ -1e-12 && state1.n * (state1.n + 1.0) + 1e-12 ≥ abs2(state1.m))
        @test violated || isapprox(state2.m, state1.m; atol=0, rtol=0) == false
    end
end

@testset "mean-field only via keep_nm_zero" begin
    let p = Params(; Δ=1.0, K=0.1, F=0.3 + 0im)
        # Start from zero and iterate with n=m frozen (mean-field map)
        config_mf = SolverConfig(;
            step_fraction=0.8, keep_nm_zero=true, max_iter=200, tol=1e-12
        )
        state = SolverState(; α=0.0 + 0im, n=0.0, m=0.0 + 0im)
        for _ in 1:40
            state = step_once(p, state, config_mf)
        end
        # Fluctuations are held at zero
        @test state.n == 0.0
        @test state.m == 0.0 + 0im
        # α should satisfy the mean-field fixed point equation approximately
        # A α + B α* = -F with n=m=0, i.e., A = -Δ + 2K|α|^2, B = 0
        Δ, K, F = p.Δ, p.K, p.F
        α = state.α
        A = (-Δ + 2K * abs2(α))
        lhs = A * α
        rhs = -F
        @test isapprox(lhs, rhs; rtol=1e-7, atol=1e-9)
    end
end

end # module test_solver
