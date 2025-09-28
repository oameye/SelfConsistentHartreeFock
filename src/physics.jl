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

function _correlators(
    ε::Float64, ΔB::ComplexF64, ω2::Float64, physics::PhysicsConfig
)::Tuple{Float64,ComplexF64}
    if physics.keep_nm_zero
        return 0.0, ComplexF64(0.0)
    end

    effective_ω = ω2 > 0 ? sqrt(ω2) : physics.omega_floor

    n_new = (ε / (2.0 * effective_ω)) - 0.5
    m_new = -(ΔB / (2.0 * effective_ω))

    if physics.enforce_physical
        n_new, m_new = _project_to_physical(n_new, m_new)
    end

    return n_new, m_new
end
