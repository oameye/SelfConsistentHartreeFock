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
Base.@kwdef struct IterationConfig
    max_iter::Int = 1000
    tol::Float64 = 1e-10
end

IterationConfig(cfg::IterationConfig; max_iter::Int=cfg.max_iter, tol::Real=cfg.tol) =
    IterationConfig(; max_iter=max_iter, tol=float(tol))

"""
    PhysicsConfig; keep_nm_zero=false, omega_floor=1e-14, enforce_physical=true

Physical constraints applied during solver updates.
"""
Base.@kwdef struct PhysicsConfig
    keep_nm_zero::Bool = false
    omega_floor::Float64 = 1e-14
    enforce_physical::Bool = true
end

PhysicsConfig(
    cfg::PhysicsConfig;
    keep_nm_zero::Bool=cfg.keep_nm_zero,
    omega_floor::Real=cfg.omega_floor,
    enforce_physical::Bool=cfg.enforce_physical,
) = PhysicsConfig(
    ; keep_nm_zero=keep_nm_zero,
      omega_floor=float(omega_floor),
      enforce_physical=enforce_physical,
)

"""
    AdaptiveStepConfig; fraction=0.2, fraction_bounds=(0.02, 0.9), fraction_scales=(1.2, 0.5)

Adaptive step-size controller for the solver's under-relaxation scheme.
"""
Base.@kwdef struct AdaptiveStepConfig
    fraction::Float64 = 0.2
    fraction_bounds::NTuple{2,Float64} = (0.02, 0.9)
    fraction_scales::NTuple{2,Float64} = (1.2, 0.5)
    backtrack::Int = 6
    accept_relax::Float64 = 0.99
    unstable_scale::Float64 = 0.3
end

function _to_float64_pair(value)::NTuple{2,Float64}
    length(value) == 2 || throw(ArgumentError("expected length-2 collection"))
    return (Float64(value[1]), Float64(value[2]))
end

AdaptiveStepConfig(
    cfg::AdaptiveStepConfig;
    fraction::Real=cfg.fraction,
    fraction_bounds=cfg.fraction_bounds,
    fraction_scales=cfg.fraction_scales,
    backtrack::Int=cfg.backtrack,
    accept_relax::Real=cfg.accept_relax,
    unstable_scale::Real=cfg.unstable_scale,
) = AdaptiveStepConfig(
    ; fraction=float(fraction),
      fraction_bounds=_to_float64_pair(fraction_bounds),
      fraction_scales=_to_float64_pair(fraction_scales),
      backtrack=backtrack,
      accept_relax=float(accept_relax),
      unstable_scale=float(unstable_scale),
)

"""
    SolverConfig(; ...)

Aggregate configuration for the solver, combining iteration, physics, and adaptive step control.
"""
struct SolverConfig
    iteration::IterationConfig
        physics::PhysicsConfig
        step::AdaptiveStepConfig
end

function SolverConfig(
    ; iteration::IterationConfig=IterationConfig(),
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
    physics_cfg = PhysicsConfig(
        physics;
        keep_nm_zero=keep_nm_zero,
        omega_floor=omega_floor,
        enforce_physical=enforce_physical,
    )
        step_cfg = AdaptiveStepConfig(
                step;
                fraction=step_fraction,
                fraction_bounds=step_bounds,
                fraction_scales=step_scales,
                backtrack=backtrack,
                accept_relax=accept_relax,
                unstable_scale=unstable_scale,
    )
        return SolverConfig(iter_cfg, physics_cfg, step_cfg)
end

SolverConfig(config::SolverConfig; kwargs...) = SolverConfig(
    ; iteration=config.iteration,
      physics=config.physics,
            step=config.step,
      kwargs...,
)

Base.@kwdef struct SolverState
    α::ComplexF64
    n::Float64
    m::ComplexF64
end

SolverState(α::Complex, n::Real, m::Complex) = SolverState(
    ; α=ComplexF64(α), n=float(n), m=ComplexF64(m),
)

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
