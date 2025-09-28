module test_continuation

using Test
using SelfConsistentHartreeFock
using SelfConsistentHartreeFock:
    ContinuationTrace,
    continuation_trace,
    continuation_completed,
    continuation_failures,
    continuation_results,
    continuation_requested,
    continuation_stopindex,
    continuation_stopped

# Check that sweep_delta performs a deterministic warm-start equivalent to
# explicitly calling solve_meanfield with the previous α as the new initial
# condition. We test both increasing and decreasing sweeps.
@testset "sweep_delta warm-start determinism (mean-field)" begin
    let Δs = collect(range(2.0, 2.3; length=7)),
        base = Params(; Δ=Δs[1], K=1.0, F=0.9),
        config = SolverConfig(;
            max_iter=2000,
            tol=1e-12,
            step_fraction=0.5,
            keep_nm_zero=true,
            backtrack=6,
            step_bounds=(0.05, 0.9),
        )

        α0 = 1e-3 + 0im
        res = sweep_delta(Δs, base, α0, config)
        @test length(res) == length(Δs)
        # step-by-step determinism: re-solve with previous α as initial
        for i in eachindex(Δs)
            p = Params(; Δ=Δs[i], K=base.K, F=base.F)
            if i == 1
                r = solve_meanfield(α0, p, config)
            else
                r = solve_meanfield(res[i - 1].α, p, config)
            end
            @test isapprox(res[i].α, r.α; rtol=1e-12, atol=1e-12)
            @test isapprox(res[i].n, r.n; rtol=1e-12, atol=1e-12)
            @test isapprox(res[i].m, r.m; rtol=1e-12, atol=1e-12)
            @test res[i].converged || res[i].iterations == config.iteration.max_iter
        end
    end
end

@testset "sweep_delta warm-start determinism (down)" begin
    let Δs = collect(range(2.3, 2.0; length=7)),
        base = Params(; Δ=Δs[1], K=1.0, F=0.9),
        config = SolverConfig(;
            max_iter=2000,
            tol=1e-12,
            step_fraction=0.5,
            keep_nm_zero=true,
            backtrack=6,
            step_bounds=(0.05, 0.9),
        )

        α0 = 1e-3 + 0im
        res = sweep_delta(Δs, base, α0, config)
        @test length(res) == length(Δs)
        for i in eachindex(Δs)
            p = Params(; Δ=Δs[i], K=base.K, F=base.F)
            if i == 1
                r = solve_meanfield(α0, p, config)
            else
                r = solve_meanfield(res[i - 1].α, p, config)
            end
            @test isapprox(res[i].α, r.α; rtol=1e-12, atol=1e-12)
            @test isapprox(res[i].n, r.n; rtol=1e-12, atol=1e-12)
            @test isapprox(res[i].m, r.m; rtol=1e-12, atol=1e-12)
            @test res[i].converged || res[i].iterations == config.iteration.max_iter
        end
    end
end

@testset "continuation trace metadata" begin
    let Δs = collect(range(2.0, 2.2; length=5)),
        base = Params(; Δ=Δs[1], K=1.0, F=0.9),
        config = SolverConfig(;
            max_iter=3000, tol=1e-12, step_fraction=0.5, keep_nm_zero=true, backtrack=6
        )

        α0 = 1e-3 + 0im
        trace = continuation_trace(Δs, base, α0, config)
        @test trace isa ContinuationTrace
        @test continuation_requested(trace) == Float64.(Δs)
        @test continuation_completed(trace) == length(Δs)
        @test isempty(continuation_failures(trace))
        @test !continuation_stopped(trace)

        collected = collect(trace)
        @test collected == continuation_results(trace)

        res = sweep_delta(Δs, base, α0, config)
        @test res == continuation_results(trace)
    end
end

@testset "continuation trace failure handling" begin
    let Δs = [2.0, 2.1, 2.2],
        base = Params(; Δ=Δs[1], K=1.0, F=0.9),
        config = SolverConfig(; max_iter=0, tol=1e-12, step_fraction=0.3, keep_nm_zero=true)

        α0 = 1e-3 + 0im
        trace = continuation_trace(Δs, base, α0, config; stop_on_failure=true)
        @test continuation_stopped(trace)
        @test continuation_stopindex(trace) == 1
        @test continuation_completed(trace) == 1
        @test !continuation_results(trace)[1].converged
        @test !isempty(continuation_failures(trace))

        trace_all = continuation_trace(Δs, base, α0, config; stop_on_failure=false)
        @test !continuation_stopped(trace_all)
        @test continuation_completed(trace_all) == length(Δs)
        @test length(continuation_failures(trace_all)) == length(Δs)
        @test continuation_requested(trace_all) == Float64.(Δs)
        @test continuation_stopindex(trace_all) === nothing
    end
end

end # module test_continuation
