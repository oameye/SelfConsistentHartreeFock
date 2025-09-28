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
    Option; max_iter=1000, tol=1e-10, mix=0.2

Configuration for the HFB solver (T = 0 only). Uses adaptive mixing with backtracking
for improved convergence.

Fields
- max_iter::Int     # max fixed-point iterations
- tol::Float64      # convergence tolerance for (α, n, m)
- mix::Float64      # initial under-relaxation factor in (0,1] (adaptively adjusted)
- keep_nm_zero::Bool  # if true, keep n and m fixed at 0 (mean-field only)
"""
Base.@kwdef struct Option
    max_iter::Int = 1000
    tol::Float64 = 1e-10
    mix::Float64 = 0.2
    keep_nm_zero::Bool = false
    # stability/physicality handling
    omega_floor::Float64 = 1e-14
    enforce_physical::Bool = true
    instability_damping::Float64 = 0.3
    # adaptive mixing / backtracking
    mix_bounds::NTuple{2,Float64} = (0.02, 0.9)    # (mix_min, mix_max)
    mix_scales::NTuple{2,Float64} = (1.2, 0.5)     # (mix_increase, mix_decrease)
    backtrack::Int = 6
    accept_relax::Float64 = 0.99
end

function _with_meanfield_constraints(opt::Option)::Option
    return Option(;
        max_iter=opt.max_iter,
        tol=opt.tol,
        mix=opt.mix,
        keep_nm_zero=true,
        omega_floor=opt.omega_floor,
        enforce_physical=opt.enforce_physical,
        instability_damping=opt.instability_damping,
        mix_bounds=opt.mix_bounds,
        mix_scales=opt.mix_scales,
        backtrack=opt.backtrack,
        accept_relax=opt.accept_relax,
    )
end

function _with_mix(opt::Option, mix::Float64)::Option
    return Option(;
        max_iter=opt.max_iter,
        tol=opt.tol,
        mix=mix,
        keep_nm_zero=opt.keep_nm_zero,
        omega_floor=opt.omega_floor,
        enforce_physical=opt.enforce_physical,
        instability_damping=opt.instability_damping,
        mix_bounds=opt.mix_bounds,
        mix_scales=opt.mix_scales,
        backtrack=opt.backtrack,
        accept_relax=opt.accept_relax,
    )
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
