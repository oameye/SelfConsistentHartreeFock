using BenchmarkTools
using SelfConsistentHartreeFock
using SelfConsistentHartreeFock: Params, continuation_trace, sweep_delta

using .BenchFixtures: continuation_fixture, ensure_group!, meanfield_fixture

function benchmark_continuation!(suite::BenchmarkGroup)
    fixture = continuation_fixture()
    Δs = fixture.Δs
    base = fixture.base
    α0 = fixture.α0
    config = fixture.config

    # Warm-up
    sweep_delta(Δs, base, α0, config)
    continuation_trace(Δs, base, α0, config)

    continuation_group = ensure_group!(suite, "continuation")

    continuation_group["sweep_delta"] = @benchmarkable sweep_delta(
        $Δs, $base, $α0, $config
    ) seconds = 8

    continuation_group["continuation_trace"] = @benchmarkable continuation_trace(
        $Δs, $base, $α0, $config; stop_on_failure=false
    ) seconds = 8

    meanfield = meanfield_fixture()
    Δs_meanfield = collect(range(-1.5, 0.6; length=28))
    base_meanfield = Params(; Δ=first(Δs_meanfield), K=meanfield.params.K, F=meanfield.params.F)
    sweep_delta(Δs_meanfield, base_meanfield, meanfield.α0, meanfield.config)

    meanfield_group = ensure_group!(continuation_group, "meanfield")
    meanfield_group["sweep_delta"] = @benchmarkable sweep_delta(
        $Δs_meanfield, $base_meanfield, $(meanfield.α0), $(meanfield.config)
    ) seconds = 6

    return nothing
end
