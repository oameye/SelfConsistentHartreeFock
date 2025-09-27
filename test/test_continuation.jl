module test_continuation

using Test
using SelfConsistentHartreeFock

# Check that sweep_delta performs a deterministic warm-start equivalent to
# explicitly calling solve_meanfield with the previous α as the new initial
# condition. We test both increasing and decreasing sweeps.
@testset "sweep_delta warm-start determinism (mean-field)" begin
    let Δs = collect(range(2.0, 2.3; length=7)),
        base = Params(; Δ=Δs[1], K=1.0, F=0.9),
    opt = Option(; max_iter=2000, tol=1e-12, mix=0.5,
              keep_nm_zero=true, backtrack=6, mix_bounds=(0.05, 0.9))

        α0 = 1e-3 + 0im
        res = sweep_delta(Δs, base, α0, opt)
        @test length(res) == length(Δs)
        # step-by-step determinism: re-solve with previous α as initial
        for i in eachindex(Δs)
            p = Params(; Δ=Δs[i], K=base.K, F=base.F)
            if i == 1
                r = solve_meanfield(α0, p, opt)
            else
                r = solve_meanfield(res[i-1].α, p, opt)
            end
            @test isapprox(res[i].α, r.α; rtol=1e-12, atol=1e-12)
            @test isapprox(res[i].n, r.n; rtol=1e-12, atol=1e-12)
            @test isapprox(res[i].m, r.m; rtol=1e-12, atol=1e-12)
            @test res[i].converged || res[i].iterations == opt.max_iter
        end
    end
end

@testset "sweep_delta warm-start determinism (down)" begin
    let Δs = collect(range(2.3, 2.0; length=7)),
        base = Params(; Δ=Δs[1], K=1.0, F=0.9),
    opt = Option(; max_iter=2000, tol=1e-12, mix=0.5,
              keep_nm_zero=true, backtrack=6, mix_bounds=(0.05, 0.9))

        α0 = 1e-3 + 0im
        res = sweep_delta(Δs, base, α0, opt)
        @test length(res) == length(Δs)
        for i in eachindex(Δs)
            p = Params(; Δ=Δs[i], K=base.K, F=base.F)
            if i == 1
                r = solve_meanfield(α0, p, opt)
            else
                r = solve_meanfield(res[i-1].α, p, opt)
            end
            @test isapprox(res[i].α, r.α; rtol=1e-12, atol=1e-12)
            @test isapprox(res[i].n, r.n; rtol=1e-12, atol=1e-12)
            @test isapprox(res[i].m, r.m; rtol=1e-12, atol=1e-12)
            @test res[i].converged || res[i].iterations == opt.max_iter
        end
    end
end

end # module test_continuation
