module test_bifurcation_regression

using Test
using SelfConsistentHartreeFock

@testset "bifurcation regression: cold vs continuation and hysteresis" begin
    let Δs = collect(range(2.0, 2.3; length=41)),
        base = Params(; Δ=Δs[1], K=1.0, F=0.9),
    opt = Option(; max_iter=4000, tol=1e-10, mix=0.5,
              keep_nm_zero=true, backtrack=8, mix_bounds=(0.05, 0.9),
                      accept_relax=0.995)

        α0 = 1e-3 + 0im

        # Cold sweep: same tiny init at every Δ
        cold = map(Δs) do Δ
            p = Params(; Δ=Δ, K=base.K, F=base.F)
            solve_meanfield(α0, p, opt)
        end
    α_cold = getfield.(cold, :α)

        # Continuation up
        up = sweep_delta(Δs, base, α0, opt)
    α_up = getfield.(up, :α)

        # Continuation down (then reverse to match Δs ordering)
        Δdown = reverse(Δs)
        base_dn = Params(; Δ=Δdown[1], K=base.K, F=base.F)
        dn_raw = sweep_delta(Δdown, base_dn, α0, opt)
    α_down = reverse(getfield.(dn_raw, :α))

        diff_cold_up = maximum(abs.(abs.(α_cold) .- abs.(α_up)))
        diff_up_down = maximum(abs.(abs.(α_up) .- abs.(α_down)))

        # Expect a measurable difference near the turning point
        @test diff_cold_up > 1e-4
        # Expect hysteresis (branches differ) in bi-stable window
        @test diff_up_down > 1e-5
    end
end

end # module test_bifurcation_regression
