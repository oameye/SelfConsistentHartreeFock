using SelfConsistentHartreeFock
using Plots

Δrange = range(-3.0, 6, 600)

# Options shared across both sweeps
opt = Option(; max_iter=20_000, tol=eps(), mix=0.5, backtrack=8,
    mix_bounds=(0.05, 0.9), accept_relax=0.995, keep_nm_zero=true,
    instability_damping=0.2)

# Cold sweep: independent solves for each Δ with the same small initial α
cold_results = map(Δrange) do _Δ
    p = Params(; Δ=_Δ, K=1.0, F=0.9)
    solve_meanfield(1e-3 + 0im, p, opt)
end
α_cold = getfield.(cold_results, :α)

# Continuation sweep (up): reuse α from the previous solution as Δ increases
base_up = Params(; Δ=first(Δrange), K=1.0, F=0.9)
cont_up = sweep_delta(collect(Δrange), base_up, 1e-3 + 0im, opt)
α_up = getfield.(cont_up, :α)

# Continuation sweep (down): start at the largest Δ and sweep backward,
# then reverse to align with Δrange for plotting
Δdown = reverse(collect(Δrange))
base_dn = Params(; Δ=last(Δrange), K=1.0, F=0.9)
cont_dn_raw = sweep_delta(Δdown, base_dn, 1e-3 + 0im, opt)
α_down = reverse(getfield.(cont_dn_raw, :α))

# Plot comparison
plt = plot(Δrange, abs.(α_cold); label="cold sweep", lw=2, ls=:dash, color=:black)
plot!(plt, Δrange, abs.(α_up); label="continuation up", lw=2)
plot!(plt, Δrange, abs.(α_down); label="continuation down", lw=2, ls=:dot)
xlabel!(plt, "Δ")
ylabel!(plt, "|α| (mean-field)")
title!(plt, "Mean-field branches: cold vs continuation")
display(plt)
