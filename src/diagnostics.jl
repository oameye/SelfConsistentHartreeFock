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

"""
    diagnostics(p, α, n, m; tol=1e-12)

Return a NamedTuple with (epsilon, DeltaB, omega, omega2, unstable, physical).
"""
function diagnostics(
    p::Params, α::ComplexF64, n::Float64, m::ComplexF64; tol::Float64=1e-12
)
    ε, ΔB, ω2 = _coeffs(p, α, n, m)
    ω = ω2 > 0 ? sqrt(ω2) : 0.0
    unstable = !(ω2 > 0)
    physical = (ω2 > 0) && _is_physical(n, m, tol)
    return (epsilon=ε, DeltaB=ΔB, omega=ω, omega2=ω2, unstable=unstable, physical=physical)
end

function diagnostics(p::Params, res::Result; tol::Float64=1e-12)
    return diagnostics(p, res.α, res.n, res.m; tol=tol)
end
