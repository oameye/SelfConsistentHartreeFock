using QuantumCumulants, HarmonicBalance, Plots

h = FockSpace(:cavity)
@qnumbers a::Destroy(h)
@variables Δ::Real K::Real F::Real
param = [Δ, K, F]

H_RWA = -Δ * a' * a + K * (a'^2 * a^2) + F * (a' + a)
ops = [a, a']

eqs = complete(meanfield(ops, H_RWA, [a]; rates=[0], order=1))

fixed = (K => 1, F => 0.9)
Δrange = range(3, 6, 1000)
varied = (Δ => Δrange)
problem = HarmonicSteadyState.HomotopyContinuationProblem(eqs, param, varied, fixed)
problem.system.expressions

result = get_steady_states(problem, WarmUp())
plot(result; y="sqrt(aᵣ^2 + aᵢ^2)")

branches = get_branches(result, "sqrt(aᵣ^2 + aᵢ^2)", class="stable")

using DrWatson
data = Dict("branch"=> branches[1],"Δrange" => Δrange)
save(projectdir("test","correction_test.jld2"), data)
