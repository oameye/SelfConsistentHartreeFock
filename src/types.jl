"""
    Params; Δ, K, F

Model parameters for the single-mode driven Kerr oscillator
    H = -Δ a† a + k a† a† a a + F (a† + a)

Fields
- Δ::Float64          # detuning Δ
- K::Float64          # Kerr nonlinearity
- F::ComplexF64       # drive amplitude (complex allowed)
"""
Base.@kwdef struct Params
    Δ::Float64
    K::Float64
    F::ComplexF64
end

"""
    IterationConfig; max_iter=1000, tol=1e-10

Iteration controls for the fixed-point solver.
"""
struct IterationConfig
    max_iter::Int
    tol::Float64

    function IterationConfig(max_iter::Int, tol::Float64)
        max_iter >= 0 ||
            throw(ArgumentError("max_iter must be nonnegative, got $(max_iter)"))
        tol > 0 || throw(ArgumentError("tol must be positive, got $(tol)"))
        return new(max_iter, tol)
    end
end

function IterationConfig(; max_iter::Int=1000, tol::Real=1e-10)
    return IterationConfig(max_iter, Float64(tol))
end

function IterationConfig(
    cfg::IterationConfig; max_iter::Int=cfg.max_iter, tol::Real=cfg.tol
)
    return IterationConfig(; max_iter=max_iter, tol=tol)
end

"""
    PhysicsConfig; keep_nm_zero=false, omega_floor=1e-14, enforce_physical=true

Physical constraints applied during solver updates.
"""
struct PhysicsConfig
    keep_nm_zero::Bool
    omega_floor::Float64
    enforce_physical::Bool

    function PhysicsConfig(keep_nm_zero::Bool, omega_floor::Float64, enforce_physical::Bool)
        omega_floor > 0 ||
            throw(ArgumentError("omega_floor must be positive, got $(omega_floor)"))
        return new(keep_nm_zero, omega_floor, enforce_physical)
    end
end

function PhysicsConfig(;
    keep_nm_zero::Bool=false, omega_floor::Real=1e-14, enforce_physical::Bool=true
)
    return PhysicsConfig(keep_nm_zero, Float64(omega_floor), enforce_physical)
end

function PhysicsConfig(
    cfg::PhysicsConfig;
    keep_nm_zero::Bool=cfg.keep_nm_zero,
    omega_floor::Real=cfg.omega_floor,
    enforce_physical::Bool=cfg.enforce_physical,
)
    return PhysicsConfig(; keep_nm_zero, omega_floor, enforce_physical)
end

"""
    AdaptiveStepConfig; fraction=0.2, fraction_bounds=(0.02, 0.9), fraction_scales=(1.2, 0.5)

Adaptive step-size controller for the solver's under-relaxation scheme.
"""
struct AdaptiveStepConfig
    fraction::Float64
    fraction_bounds::NTuple{2,Float64}
    fraction_scales::NTuple{2,Float64}
    backtrack::Int
    accept_relax::Float64
    unstable_scale::Float64

    function AdaptiveStepConfig(
        fraction::Float64,
        fraction_bounds::NTuple{2,Float64},
        fraction_scales::NTuple{2,Float64},
        backtrack::Int,
        accept_relax::Float64,
        unstable_scale::Float64,
    )
        _check_unit_interval(:step_fraction, fraction)
        bounds = _validate_fraction_bounds(fraction_bounds)
        _check_positive(:step_scale_increase, fraction_scales[1])
        _check_positive(:step_scale_decrease, fraction_scales[2])
        _check_nonnegative(:backtrack, backtrack)
        _check_unit_interval(:accept_relax, accept_relax)
        _check_unit_interval(:unstable_scale, unstable_scale)
        return new(
            fraction, bounds, fraction_scales, backtrack, accept_relax, unstable_scale
        )
    end
end

function _to_float64_pair(value)::NTuple{2,Float64}
    length(value) == 2 || throw(ArgumentError("expected length-2 collection"))
    return (Float64(value[1]), Float64(value[2]))
end

function _check_positive(name::Symbol, value::Real)
    value > 0 || throw(ArgumentError("$name must be positive, got $(value)"))
    return value
end

function _check_nonnegative(name::Symbol, value::Real)
    value >= 0 || throw(ArgumentError("$name must be nonnegative, got $(value)"))
    return value
end

function _check_unit_interval(name::Symbol, value::Real)
    (0 < value <= 1) || throw(ArgumentError("$name must lie in (0, 1], got $(value)"))
    return value
end

function _validate_fraction_bounds(bounds::NTuple{2,Float64})
    lower, upper = bounds
    _check_nonnegative(:step_bounds_min, lower)
    _check_unit_interval(:step_bounds_max, upper)
    lower <= upper || throw(ArgumentError("step_bounds must satisfy min <= max"))
    return bounds
end

function AdaptiveStepConfig(;
    fraction::Real=0.2,
    fraction_bounds=(0.02, 0.9),
    fraction_scales=(1.2, 0.5),
    backtrack::Int=6,
    accept_relax::Real=0.99,
    unstable_scale::Real=0.3,
)
    frac = Float64(fraction)
    bounds = _to_float64_pair(fraction_bounds)
    scales = _to_float64_pair(fraction_scales)
    acc = Float64(accept_relax)
    unstable = Float64(unstable_scale)
    return AdaptiveStepConfig(frac, bounds, scales, backtrack, acc, unstable)
end

function AdaptiveStepConfig(
    cfg::AdaptiveStepConfig;
    fraction::Real=cfg.fraction,
    fraction_bounds=cfg.fraction_bounds,
    fraction_scales=cfg.fraction_scales,
    backtrack::Int=cfg.backtrack,
    accept_relax::Real=cfg.accept_relax,
    unstable_scale::Real=cfg.unstable_scale,
)
    return AdaptiveStepConfig(;
        fraction, fraction_bounds, fraction_scales, backtrack, accept_relax, unstable_scale
    )
end

"""
    SolverConfig(; ...)

Aggregate configuration for the solver, combining iteration, physics, and adaptive step control.
"""
struct SolverConfig
    iteration::IterationConfig
    physics::PhysicsConfig
    step::AdaptiveStepConfig
end

function SolverConfig(;
    iteration::IterationConfig=IterationConfig(),
    physics::PhysicsConfig=PhysicsConfig(),
    step::AdaptiveStepConfig=AdaptiveStepConfig(),
    max_iter::Int=iteration.max_iter,
    tol::Real=iteration.tol,
    keep_nm_zero::Bool=physics.keep_nm_zero,
    omega_floor::Real=physics.omega_floor,
    enforce_physical::Bool=physics.enforce_physical,
    step_fraction::Real=step.fraction,
    step_bounds=step.fraction_bounds,
    step_scales=step.fraction_scales,
    backtrack::Int=step.backtrack,
    accept_relax::Real=step.accept_relax,
    unstable_scale::Real=step.unstable_scale,
)
    iter_cfg = IterationConfig(iteration; max_iter=max_iter, tol=tol)
    physics_cfg = PhysicsConfig(physics; keep_nm_zero, omega_floor, enforce_physical)
    step_cfg = AdaptiveStepConfig(
        step;
        fraction=step_fraction,
        fraction_bounds=step_bounds,
        fraction_scales=step_scales,
        backtrack,
        accept_relax,
        unstable_scale,
    )
    return SolverConfig(iter_cfg, physics_cfg, step_cfg)
end

function SolverConfig(config::SolverConfig; kwargs...)
    return SolverConfig(;
        iteration=config.iteration, physics=config.physics, step=config.step, kwargs...
    )
end

Base.@kwdef struct SolverState
    α::ComplexF64
    n::Float64
    m::ComplexF64
end

function SolverState(α::Complex, n::Real, m::Complex)
    return SolverState(; α=ComplexF64(α), n=float(n), m=ComplexF64(m))
end

function _with_meanfield_constraints(config::SolverConfig)::SolverConfig
    return SolverConfig(config; keep_nm_zero=true)
end

function _with_step_fraction(config::SolverConfig, fraction::Float64)::SolverConfig
    return SolverConfig(config; step_fraction=fraction)
end

"""
    Result

Result of the self-consistent HFB solver.
"""
Base.@kwdef struct Result
    α::ComplexF64
    n::Float64
    m::ComplexF64
    converged::Bool
    iterations::Int
    unstable::Bool
    physical::Bool
end

converged(result::Result) = result.converged
iterations(result::Result) = result.iterations
unstable(result::Result) = result.unstable
physical(result::Result) = result.physical
