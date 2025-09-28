"""
    solve(α0, p::Params, opt::Option)

Self-consistent Hartree-Fock-Bogoliubov (HFB) at T = 0 for the single-mode driven
Kerr oscillator.
"""
function solve(α0::Complex, p::Params, opt::Option=Option())
    @assert opt.mix > 0 && opt.mix ≤ 1 "mix must be in (0,1]"

    # Initialize state variables
    α = ComplexF64(α0)
    n = 0.0
    m = ComplexF64(0.0)

    # Initialize convergence tracking
    converged = false
    iterations = 0
    prev_residual = Inf

    # Initialize adaptive mixing parameters
    mix_min, mix_max = opt.mix_bounds
    current_mix = clamp(opt.mix, mix_min, mix_max)

    # Main iteration loop
    for iteration in 1:(opt.max_iter)
        # Advance one step using adaptive method
        α_new, n_new, m_new, residual, current_mix = _advance_adaptive(
            p, α, n, m, prev_residual, current_mix, opt
        )

        # Check convergence criteria
        max_change = max(abs(α_new - α), abs(n_new - n), abs(m_new - m))

        # Update state
        α, n, m = α_new, n_new, m_new
        iterations = iteration
        prev_residual = residual

        if max_change < opt.tol
            converged = true
            break
        end
    end

    # Determine solution properties
    _, _, ω2 = _coeffs(p, α, n, m)
    unstable = !(ω2 > 0)
    physical = (ω2 > 0) && _is_physical(n, m, max(opt.tol, 1e-12))

    return Result(; α, n, m, converged, iterations, unstable, physical)
end

"""
    solve_meanfield(α0, p::Params, opt::Option)

Freeze fluctuations at n=m=0 and solve the mean-field fixed point.
"""
function solve_meanfield(α0::Complex, p::Params, opt::Option=Option())
    # Create options with fluctuations frozen at zero
    meanfield_opt = _with_meanfield_constraints(opt)
    return solve(α0, p, meanfield_opt)
end

function solve_meanfield(p::Params, opt::Option=Option())
    return solve_meanfield(1e-3 + 1e-3im, p, opt)
end

"""
    step_once(p::Params, α, n, m, opt::Option)

Perform a single fixed-point iteration.
"""
function step_once(p::Params, α::ComplexF64, n::Float64, m::ComplexF64, opt::Option)
    @assert opt.mix > 0 && opt.mix ≤ 1 "mix must be in (0,1]"
    @assert opt.omega_floor > 0 "omega_floor must be positive"
    @assert opt.instability_damping > 0 && opt.instability_damping ≤ 1 "instability_damping in (0,1]"

    # Calculate system coefficients
    ε, ΔB, ω2 = _coeffs(p, α, n, m)

    # Compute new correlator values
    n_target, m_target = _correlators(ε, ΔB, ω2, opt)
    α_target = _alpha_update(p, α, n_target, m_target)

    # Apply mixing with stability-dependent factor
    effective_mix = _effective_mix(ω2, opt)
    α_next = _mix_value(α, α_target, effective_mix)

    if opt.keep_nm_zero
        n_next = 0.0
        m_next = ComplexF64(0.0)
    else
        n_next = _mix_value(n, n_target, effective_mix)
        m_next = _mix_value(m, m_target, effective_mix)
    end

    return α_next, n_next, m_next
end

# --- adaptive mixing helpers ---

function _residual_norm(
    α::ComplexF64, n::Float64, m::ComplexF64, α2::ComplexF64, n2::Float64, m2::ComplexF64
)::Float64
    return max(abs(α2 - α), max(abs(n2 - n), abs(m2 - m)))
end

# --- internal helpers ---

function _effective_mix(ω2::Float64, opt::Option)::Float64
    return (ω2 > 0) ? opt.mix : (opt.mix * opt.instability_damping)
end

function _accept_step(rt::Float64, prev::Float64, accept_relax::Float64)::Bool
    return rt < prev * accept_relax
end

function _update_mix_after_success(
    mix_curr::Float64,
    resid_new::Float64,
    prev_resid::Float64,
    mix_min::Float64,
    mix_max::Float64,
    mix_inc::Float64,
)::Float64
    if isfinite(prev_resid) && resid_new ≤ prev_resid * 0.7
        return min(mix_max, mix_curr * mix_inc)
    else
        return clamp(mix_curr, mix_min, mix_max)
    end
end

function _backtracked_mix(
    mix_curr::Float64, mix_min::Float64, mix_dec::Float64, backtrack::Int
)::Float64
    return max(mix_min, mix_curr * mix_dec^backtrack)
end

function _correlators(
    ε::Float64, ΔB::ComplexF64, ω2::Float64, opt::Option
)::Tuple{Float64,ComplexF64}
    if opt.keep_nm_zero
        return 0.0, ComplexF64(0.0)
    end

    # Use regularized frequency to avoid division by zero
    effective_ω = ω2 > 0 ? sqrt(ω2) : opt.omega_floor

    # Calculate correlators from Bogoliubov transformation
    n_new = (ε / (2.0 * effective_ω)) - 0.5
    m_new = -(ΔB / (2.0 * effective_ω))

    # Ensure physical constraints if required
    if opt.enforce_physical
        n_new, m_new = _project_to_physical(n_new, m_new)
    end

    return n_new, m_new
end

_mix_value(old::T, new::T, mix::Float64) where {T<:Number} = (1 - mix) * old + mix * new



function _advance_adaptive(
    p::Params, α::ComplexF64, n::Float64, m::ComplexF64, prev_residual::Float64,
    current_mix::Float64, opt::Option,
)::Tuple{ComplexF64,Float64,ComplexF64,Float64,Float64}
    mix_min, mix_max = opt.mix_bounds
    mix_inc, mix_dec = opt.mix_scales

    # Attempt backtracking to find acceptable step size
    trial_mix = current_mix
    max_backtrack_attempts = max(opt.backtrack, 1)

    for attempt in 1:max_backtrack_attempts
        # Try step with current mixing parameter
        α_candidate, n_candidate, m_candidate = step_once(
            p, α, n, m, _with_mix(opt, trial_mix)
        )
        residual_candidate = _residual_norm(
            α, n, m, α_candidate, n_candidate, m_candidate
        )

        # Accept step if residual improvement is sufficient
        if _accept_step(residual_candidate, prev_residual, opt.accept_relax)
            updated_mix = _update_mix_after_success(
                current_mix, residual_candidate, prev_residual, mix_min, mix_max, mix_inc
            )
            return α_candidate, n_candidate, m_candidate, residual_candidate, updated_mix
        end

        # Reduce mixing parameter for next backtracking attempt
        trial_mix = max(mix_min, trial_mix * mix_dec)
    end

    # Backtracking failed - force step with maximum reduction
    fallback_mix = _backtracked_mix(current_mix, mix_min, mix_dec, opt.backtrack)
    α_fallback, n_fallback, m_fallback = step_once(
        p, α, n, m, _with_mix(opt, fallback_mix)
    )
    residual_fallback = _residual_norm(α, n, m, α_fallback, n_fallback, m_fallback)

    return α_fallback, n_fallback, m_fallback, residual_fallback, current_mix
end

"""
    _coeffs(p, α, n, m) -> (ε, ΔB, ω2)

Compute quadratic coefficients and ω² for the Gaussian Hamiltonian.
"""
function _coeffs(p::Params, α::ComplexF64, n::Float64, m::ComplexF64)
    Δ = p.Δ
    K = p.K
    ε = -Δ + 4.0 * K * (abs2(α)) + 2.0 * K * n
    ΔB = 2.0 * K * (α^2 + m)
    ω2 = ε^2 - abs2(ΔB)
    return ε, ΔB, ω2
end

function _alpha_update(p::Params, α::ComplexF64, n_new::Float64, m_new::ComplexF64)
    Δ = p.Δ
    K = p.K
    F = p.F
    A = (-Δ + 4.0 * K * n_new) + 2.0 * K * abs2(α)
    B = 2.0 * K * m_new
    denom = abs2(A) - abs2(B)
    if abs(denom) < 1e-18
        denom += 1e-18
    end
    return (B * conj(F) - conj(A) * F) / denom
end

function _is_physical(n::Float64, m::ComplexF64, tol::Float64)::Bool
    return (n ≥ -tol) && (n * (n + 1.0) + tol ≥ abs2(m))
end

function _project_to_physical(n::Float64, m::ComplexF64)
    np = max(n, 0.0)
    bound = np * (np + 1.0)
    m2 = abs2(m)
    if m2 > bound && m2 > 0
        scale = sqrt(max(bound, 0.0) / m2)
        return np, m * scale
    else
        return np, m
    end
end
