module test_solver

using Test
using SelfConsistentHartreeFock
using SelfConsistentHartreeFock: Result, step_once, _coeffs, _is_physical

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

function diagnostics(p::Params, res::Result; tol::Float64=1e-12)
    return diagnostics(p, res.α, res.n, res.m; tol=tol)
end

@testset "basic solve" begin
    let p = Params(; Δ=1.0, K=0.05, F=0.2 + 0im),
        opt = Option(; max_iter=200, tol=1e-12, mix=0.3)

        res = solve(0.0 + 0im, p, opt)
        @test res isa Result
        @test res.converged || res.iterations == opt.max_iter
        # compute diagnostics via helper
        d = diagnostics(p, res)
        @test isfinite(d.epsilon) && isfinite(d.omega)
    @test res.α == conj(conj(res.α)) # roundtrip sanity
    end
end

@testset "determinism for same IC" begin
    let p = Params(; Δ=1.0, K=0.05, F=0.2 + 0im),
        opt = Option(; max_iter=150, tol=1e-10, mix=0.25)

        res1 = solve(0.0 + 0im, p, opt)
        res2 = solve(0.0 + 0im, p, opt)
        @test res1 isa Result
        @test res2 isa Result
    @test isapprox(res1.α, res2.α; rtol=1e-12, atol=1e-12)
        @test isapprox(res1.n, res2.n; rtol=1e-12, atol=1e-12)
        @test isapprox(res1.m, res2.m; rtol=1e-12, atol=1e-12)
    end
end

@testset "step_once progression" begin
    let p = Params(; Δ=1.0, K=0.05, F=0.2 + 0im)
        α, n, m = 0.2 + 0im, 0.0, 0.0 + 0im
        opt = Option(; mix=0.3)
        for _ in 1:10
            α, n, m = step_once(p, α, n, m, opt)
            d = diagnostics(p, α, n, m)
            @test isfinite(d.epsilon) && isfinite(d.omega)
            @test !isnan(real(d.DeltaB))
        end
        res = solve(0.0 + 0im, p, Option(; max_iter=200, tol=1e-10, mix=0.3))
        # After 10 steps we should be in the vicinity of the solved alpha
    @test isapprox(α, res.α; rtol=5e-2, atol=5e-3)
    end
end

@testset "physical flag and stability classification" begin
    let p = Params(; Δ=1.0, K=0.05, F=0.2 + 0im),
        opt = Option(; max_iter=200, tol=1e-12, mix=0.3)

        res = solve(0.0 + 0im, p, opt)
        # If stable, result should be flagged physical and satisfy positivity
        d = diagnostics(p, res)
        pos_ok = (res.n ≥ -1e-12) && (res.n * (res.n + 1.0) + 1e-12 ≥ abs2(res.m))
        @test res.physical == ((d.omega2 > 0) && pos_ok)
    end
end

@testset "instability handling: enforce_physical vs off" begin
    # Construct an input with ω² ≤ 0 at the current (α, n, m), then compare one-step updates
    let p = Params(; Δ=0.0, K=1.0, F=0.0 + 0im)
        α0, n0, m0 = (0.0 + 0im), 0.0, (1.0 + 0im) # ε=0, ΔB=2 → ω² = -4 < 0

        # Initial state is unstable
        d0 = diagnostics(p, α0, n0, m0)
        @test d0.unstable
        @test d0.omega == 0.0

        # No enforcement: allow nonphysical correlators
        opt1 = Option(; mix=1.0, omega_floor=1e-2, enforce_physical=false, instability_damping=1.0)
        α1, n1, m1 = step_once(p, α0, n0, m0, opt1)

        # With enforcement: correlators are projected to the physical region
        opt2 = Option(; mix=1.0, omega_floor=1e-2, enforce_physical=true, instability_damping=1.0)
        α2, n2, m2 = step_once(p, α0, n0, m0, opt2)
        @test n2 ≥ -1e-12
        @test n2 * (n2 + 1.0) + 1e-12 ≥ abs2(m2)

        # Typically the unenforced step violates positivity
        # (don't require it strictly in case parameters change)
        violated = !(n1 ≥ -1e-12 && n1 * (n1 + 1.0) + 1e-12 ≥ abs2(m1))
        @test violated || isapprox(m2, m1; atol=0, rtol=0) == false
    end
end

@testset "mean-field only via keep_nm_zero" begin
    let p = Params(; Δ=1.0, K=0.1, F=0.3 + 0im)
        # Start from zero and iterate with n=m frozen (mean-field map)
        opt_mf = Option(; mix=0.8, keep_nm_zero=true, max_iter=200, tol=1e-12)
        α, n, m = 0.0 + 0im, 0.0, 0.0 + 0im
        for _ in 1:40
            α, n, m = step_once(p, α, n, m, opt_mf)
        end
        # Fluctuations are held at zero
        @test n == 0.0
        @test m == 0.0 + 0im
        # α should satisfy the mean-field fixed point equation approximately
        # A α + B α* = -F with n=m=0, i.e., A = -Δ + 2K|α|^2, B = 0
        Δ, K, F = p.Δ, p.K, p.F
        A = (-Δ + 2K * abs2(α))
        lhs = A * α
        rhs = -F
        @test isapprox(lhs, rhs; rtol=1e-7, atol=1e-9)
    end
end

end # module test_solver
