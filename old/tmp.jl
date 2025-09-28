# ====================== HFB Kerr (Zero Temperature) ======================
# Self-consistent Hartree–Fock–Bogoliubov solver for the driven Kerr mode
# Model: H = -Δ a† a + K a† a† a a + F (a† + a)
# Split a = α + d, decouple, and solve for (α, n, m) self-consistently.
#
# Conventions:
#   Ω = Δ - 4K(|α|^2 + n)
#   G = K(α^2 + m)
#   ω̃ = sqrt(Ω^2 - |G|^2)   (must be real > 0)
#   n = 1/2 (Ω/ω̃ - 1)
#   m = G / (2ω̃)
# ========================================================================

# ---------------------- Types / Options ----------------------------------

struct Parameters
    Δ::Float64        # detuning
    K::Float64        # Kerr nonlinearity
    F::ComplexF64     # coherent drive (can be complex)
end

struct Options
    max_iter::Int
    tol::Float64
    η::Float64
    verbose::Bool
end

function Options(; max_iter=10_000, tol=1e-12, η=0.0, verbose=false)
    Options(max_iter, tol, η, verbose, α0, n0, m0)
end
# ---------------------- Core Helpers -------------------------------------

# Solve linear-term cancellation system:
# (A  B)(α  ) = (-F  ),  (B* A*)(α*) = (-F)
function solve_alpha(A::ComplexF64, B::ComplexF64, F::ComplexF64)
    D = abs2(A) - abs2(B)
    if isapprox(D, 0.0; atol=1e-18, rtol=1e-15)
        error("Ill-conditioned α-equation: |A|^2 - |B|^2 ≈ 0 (near threshold).")
    end
    (B - conj(A)) * F / D
end

# One HFB iteration step (T = 0):
# returns (n_new, m_new, α_new, Ω, G, ω̃)
function iteration_step(HFBParams, α::ComplexF64, n::Float64, m::ComplexF64)
    Δ, K, F = Params
    Ω = Δ - 4K * (abs2(α) + n)
    G = K * (α^2 + m)
    discr = Ω^2 - abs2(G)
    ω̃ = sqrt(discr)  # NaN if discr < 0

    n_new = 0.5 * (Ω / ω̃ - 1.0)
    m_new = G / (2.0 * ω̃)

    A = -Δ + 2K * abs2(α) + 4K * n_new
    B = 2K * m_new
    α_new = solve_alpha(complex(A), complex(B), F)

    return n_new, m_new, α_new, Ω, G, ω̃
end

# ---------------------- Solver -------------------------------------------

"""
    solve_hfb(par::HFBParams; opt::HFBOptions = HFBOptions())

Run the zero-temperature self-consistent HFB iteration.

Returns a NamedTuple:
    (α, n, m, Ω, G, ω̃, stable, converged, iters, residuals)
"""
function solve_hfb(par::Parameters; opt::Options=Options())
    Δ, K, F = par.Δ, par.K, par.F
    α, n, m = opt.α0, opt.n0, opt.m0

    Ω = 0.0
    G = 0.0 + 0.0im
    ω̃ = NaN
    stable = true
    converged = false

    for it in 1:opt.max_iter
        n̂, m̂, α̂, Ω̂, Ĝ, ω̃̂ = hfb_step(Δ, K, F, α, n, m)

        # require real, positive ω̃
        if !(isfinite(ω̃̂) && ω̃̂ > 0)
            stable = false
            Ω, G, ω̃ = Ω̂, Ĝ, ω̃̂
            return (
                α=α,
                n=n,
                m=m,
                Ω=Ω,
                G=G,
                ω̃=ω̃,
                stable=stable,
                converged=false,
                iters=it,
                residuals=(δα=NaN, δn=NaN, δm=NaN),
            )
        end

        # under-relaxed updates
        α_new = (1 - opt.η) * α + opt.η * α̂
        n_new = (1 - opt.η) * n + opt.η * n̂
        m_new = (1 - opt.η) * m + opt.η * m̂

        # residuals
        δα = abs(α_new - α) / max(abs(α), 1e-16)
        δn = abs(n_new - n) / max(abs(n) + 1.0, 1.0)
        δm = abs(m_new - m) / max(abs(m) + 1.0, 1.0)

        α, n, m = α_new, n_new, m_new
        Ω, G, ω̃ = Ω̂, Ĝ, ω̃̂

        opt.verbose && @info "HFB iter=$it" δα δn δm Ω ω̃ absG = abs(G)

        if δα < opt.tol && δn < opt.tol && δm < opt.tol
            converged = true
            return (
                α=α,
                n=n,
                m=m,
                Ω=Ω,
                G=G,
                ω̃=ω̃,
                stable=stable,
                converged=converged,
                iters=it,
                residuals=(δα=δα, δn=δn, δm=δm),
            )
        end
    end

    (
        α=α,
        n=n,
        m=m,
        Ω=Ω,
        G=G,
        ω̃=ω̃,
        stable=stable,
        converged=converged,
        iters=opt.max_iter,
        residuals=(δα=NaN, δn=NaN, δm=NaN),
    )
end

# ---------------------- Minimal example -----------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    # Example parameters
    Δ = 1.0
    K = 0.02
    F = 0.25 + 0im

    par = Parameters(Δ, K, F)
    opt = Options(; η=0.6, tol=1e-12, verbose=false)

    res = solve_hfb(par; opt=opt)

    println("Converged: ", res.converged, "  in ", res.iters, " iters")
    println("Stable:    ", res.stable, "  ω̃ = ", res.ω̃)
    println("|α| = ", abs(res.α), "   arg(α) = ", angle(res.α))
    println("n   = ", res.n)
    println("|m| = ", abs(res.m), "   arg(m) = ", angle(res.m))
    println("Ω   = ", res.Ω, "   |G| = ", abs(res.G))
end
