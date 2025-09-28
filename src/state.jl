"""
    step_once(p::Params, state::SolverState, config::SolverConfig)

Perform a single fixed-point iteration returning an updated solver state.
"""
function step_once(p::Params, state::SolverState, config::SolverConfig)
    step_cfg = config.step
    physics = config.physics

    ε, ΔB, ω2 = _coeffs(p, state.α, state.n, state.m)

    n_target, m_target = _correlators(ε, ΔB, ω2, physics)
    α_target = _alpha_update(p, state.α, n_target, m_target)

    effective_fraction = _effective_fraction(ω2, step_cfg)
    α_next = _blend_value(state.α, α_target, effective_fraction)

    if physics.keep_nm_zero
        n_next = 0.0
        m_next = ComplexF64(0.0)
    else
        n_next = _blend_value(state.n, n_target, effective_fraction)
        m_next = _blend_value(state.m, m_target, effective_fraction)
    end

    return SolverState(α_next, n_next, m_next)
end

function _residual_norm(state::SolverState, candidate::SolverState)::Float64
    return max(
        abs(candidate.α - state.α),
        max(abs(candidate.n - state.n), abs(candidate.m - state.m)),
    )
end

function _state_max_change(state::SolverState, candidate::SolverState)::Float64
    return _residual_norm(state, candidate)
end

_blend_value(old::T, new::T, fraction::Float64) where {T<:Number} = (1 - fraction) * old + fraction * new

function _effective_fraction(ω2::Float64, step_cfg::AdaptiveStepConfig)::Float64
    return (ω2 > 0) ? step_cfg.fraction : (step_cfg.fraction * step_cfg.unstable_scale)
end
