using BenchmarkTools
using SelfConsistentHartreeFock
using SelfConsistentHartreeFock: step_once, _advance_adaptive
using SelfConsistentHartreeFock: _residual_norm

using .BenchFixtures: ensure_group!, step_control_fixture

function benchmark_step_control!(suite::BenchmarkGroup)
    fixture = step_control_fixture()
    params = fixture.params
    config = fixture.config
    state = fixture.state
    candidate = fixture.candidate
    residual = fixture.residual
    fraction = fixture.fraction

    micro = ensure_group!(suite, "micro")
    step_group = ensure_group!(micro, "step_control")

    step_group["step_once"] = @benchmarkable step_once($params, $state, $config)

    step_group["residual_norm"] = @benchmarkable _residual_norm($state, $candidate)

    prev_residual = residual * 1.2
    step_group["advance_adaptive"] = @benchmarkable _advance_adaptive(
        $params, $state, $prev_residual, $fraction, $config
    )

    return nothing
end
