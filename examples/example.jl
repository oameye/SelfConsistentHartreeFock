using SelfConsistentHartreeFock, QuantumToolbox
using Plots
using LaTeXStrings

Δrange = range(-0.01, 0.03, 300)
Kval = 0.001
Fval = 0.01

# Configuration shared across both sweeps
config_gaussian = SolverConfig(
    ; max_iter=20_000,
      tol=eps(),
      step_fraction=0.5,
      backtrack=8,
      step_bounds=(0.05, 0.9),
      accept_relax=0.995,
      keep_nm_zero=false,
      unstable_scale=0.2,
)

config_mean_field = SolverConfig(
    ; max_iter=20_000,
      tol=eps(),
      step_fraction=0.5,
      backtrack=8,
      step_bounds=(0.05, 0.9),
      accept_relax=0.995,
      keep_nm_zero=true,
      unstable_scale=0.2,
)
param = Params(; Δ=first(Δrange), K=Kval, F=Fval)

# Continuation sweep (up): reuse α from the previous solution as Δ increases
cont_up = sweep_delta(collect(Δrange), param, 1e-3 + 0im, config_gaussian)
cont_up_mean = sweep_delta(collect(Δrange), param, 1e-3 + 0im, config_mean_field)
α_up = getfield.(cont_up, :α)
α_mean = getfield.(cont_up_mean, :α)
n_up = getfield.(cont_up, :n)
m_up = getfield.(cont_up, :m)

# Continuation sweep (down): start at the largest Δ and sweep backward,
# then reverse to align with Δrange for plotting
Δdown = reverse(collect(Δrange))
cont_dn = sweep_delta(Δdown, param, 1e-3 + 0im, config_gaussian)
α_down = reverse(getfield.(cont_dn, :α))
n_down = reverse(getfield.(cont_dn, :n))
m_down = reverse(getfield.(cont_dn, :m))

function ρ_ss(Δ, F, K, γ; kwargs...)
    N = 50 # cutoff of the Hilbert space dimension
    a = destroy(N) # annihilation operator
    nth=0.01

    H = - Δ * a' * a + K * a' * a' * a * a + F * (a'  +  a)
    c_ops = [sqrt(γ)*a]

    solver = SteadyStateLinearSolver()
    ρ_ss = steadystate(H, c_ops; solver,kwargs...) # Hamiltonian and collapse operators
    # real(QT.expect(a' * a, ρ_ss))
end

ρv = map(_Δ -> ρ_ss(_Δ, Fval, Kval, 0.0005), Δrange)

a = destroy(50) # a
n_quantum = ρ -> real(expect(a' * a, ρ))
n_q = n_quantum.(ρv)
plot(n_q)
# g2m = ρ -> real(QT.expect(a' * a' * a * a, ρ)/QT.expect(a' * a , ρ)^2)

# Plot comparison
plt = plot(
    Δrange,
    abs.(α_up);
    label="continuation up",
    lw=2,
    xlabel="",
    ylabel=L"|α|^2",
    title="meanfield",
)
plot!(plt, Δrange, abs.(α_down); label="continuation down", lw=2, ls=:dot)
plot!(plt, Δrange, n_q; label="quantum steady state", lw=2, ls=:dash, color=:black)
plot!(plt, Δrange, abs.(α_mean); label="meanfield", lw=2, ls=:dashdot, color=:red)
plt2 = plot(
    Δrange,
    n_up;
    label="continuation up",
    lw=2,
    xlabel="",
    ylabel=L"\langle d^\dagger d \rangle",
    title="fluctuations amplitude",
)
plot!(plt2, Δrange, n_down; label="continuation down", lw=2, ls=:dot)

plt3 = plot(
    Δrange,
    abs.(m_up);
    label="continuation up",
    lw=2,
    xlabel="Δ",
    ylabel=L"|\langle d d \rangle|",
    title="anomalous correlations amplitude",
)
plot!(plt3, Δrange, abs.(m_down); label="continuation down", lw=2, ls=:dot)

plot(plt, plt2, plt3; layout=(3, 1), size=(600, 700), legend=:topleft)
