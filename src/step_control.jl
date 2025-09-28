function _accept_step(rt::Float64, prev::Float64, accept_relax::Float64)::Bool
    return rt < prev * accept_relax
end

function _update_fraction_after_success(
    fraction_curr::Float64,
    resid_new::Float64,
    prev_resid::Float64,
    fraction_min::Float64,
    fraction_max::Float64,
    fraction_inc::Float64,
)::Float64
    if isfinite(prev_resid) && resid_new ≤ prev_resid * 0.7
        return min(fraction_max, fraction_curr * fraction_inc)
    else
        return clamp(fraction_curr, fraction_min, fraction_max)
    end
end

function _backtracked_fraction(
    fraction_curr::Float64, fraction_min::Float64, fraction_dec::Float64, backtrack::Int
)::Float64
    return max(fraction_min, fraction_curr * fraction_dec^backtrack)
end

function _advance_adaptive(
    p::Params,
    state::SolverState,
    prev_residual::Float64,
    current_fraction::Float64,
    config::SolverConfig,
)::Tuple{SolverState,Float64,Float64}
    step = config.step
    fraction_min, fraction_max = step.fraction_bounds
    fraction_inc, fraction_dec = step.fraction_scales

    trial_fraction = current_fraction
    max_backtrack_attempts = max(step.backtrack, 1)

    for _ in 1:max_backtrack_attempts
        candidate_state = step_once(p, state, _with_step_fraction(config, trial_fraction))
        residual_candidate = _residual_norm(state, candidate_state)

        if _accept_step(residual_candidate, prev_residual, step.accept_relax)
            updated_fraction = _update_fraction_after_success(
                current_fraction,
                residual_candidate,
                prev_residual,
                fraction_min,
                fraction_max,
                fraction_inc,
            )
            return candidate_state, residual_candidate, updated_fraction
        end

        trial_fraction = max(fraction_min, trial_fraction * fraction_dec)
    end

    fallback_fraction = _backtracked_fraction(
        current_fraction, fraction_min, fraction_dec, step.backtrack
    )
    fallback_state = step_once(p, state, _with_step_fraction(config, fallback_fraction))
    residual_fallback = _residual_norm(state, fallback_state)

    return fallback_state, residual_fallback, current_fraction
end
