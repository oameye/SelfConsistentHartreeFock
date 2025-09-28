"""
    solve(α0, p::Params, config::SolverConfig)

Self-consistent Hartree-Fock-Bogoliubov (HFB) at T = 0 for the single-mode driven
Kerr oscillator.
"""
function solve(α0::Complex, p::Params, config::SolverConfig=SolverConfig())
    step = config.step
    iteration = config.iteration

    # Initialize state variables
    state = SolverState(α0, 0.0, 0.0)

    # Initialize convergence tracking
    converged = false
    iterations = 0
    prev_residual = Inf

    # Initialize adaptive step parameters
    fraction_min, fraction_max = step.fraction_bounds
    current_fraction = clamp(step.fraction, fraction_min, fraction_max)

    # Main iteration loop
    for iteration_count in 1:(iteration.max_iter)
        new_state, residual, current_fraction = _advance_adaptive(
            p, state, prev_residual, current_fraction, config
        )

        # Check convergence criteria
        max_change = _state_max_change(state, new_state)

        # Update state
        state = new_state
        iterations = iteration_count
        prev_residual = residual

        if max_change < iteration.tol
            converged = true
            break
        end
    end

    # Determine solution properties
    _, _, ω2 = _coeffs(p, state.α, state.n, state.m)
    unstable = !(ω2 > 0)
    physical = (ω2 > 0) && _is_physical(state.n, state.m, max(iteration.tol, 1e-12))

    return Result(
        ; α=state.α,
          n=state.n,
          m=state.m,
          converged,
          iterations,
          unstable,
          physical,
    )
end

"""
    solve_meanfield(α0, p::Params, config::SolverConfig)

Freeze fluctuations at n=m=0 and solve the mean-field fixed point.
"""
function solve_meanfield(α0::Complex, p::Params, config::SolverConfig=SolverConfig())
    meanfield_config = _with_meanfield_constraints(config)
    return solve(α0, p, meanfield_config)
end

function solve_meanfield(p::Params, config::SolverConfig)
    return solve_meanfield(p; config=config)
end

function solve(p::Params; config::SolverConfig=SolverConfig(), α0::Complex=_default_initial_guess(p, config))
    return solve(α0, p, config)
end

function solve(p::Params, config::SolverConfig)
    return solve(p; config=config)
end

function solve_meanfield(
    p::Params; config::SolverConfig=SolverConfig(), α0::Complex=_default_initial_guess(p, config)
)
    meanfield_config = _with_meanfield_constraints(config)
    return solve(α0, p, meanfield_config)
end

_default_initial_guess(::Params, ::SolverConfig) = ComplexF64(1e-3 + 1e-3im)
