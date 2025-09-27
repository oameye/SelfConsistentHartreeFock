"""
    sweep_delta(Δs, base, α0, opt)

Sweep detuning values with warm-start continuation.
"""
function sweep_delta(
    Δs::AbstractVector{T}, base::Params, α0::Complex, opt::Option=Option(),
) where {T<:Real}
    results = Vector{Result}(undef, length(Δs))
    αinit = ComplexF64(α0)
    for (i, Δi) in pairs(Δs)
        pᵢ = Params(; Δ=float(Δi), K=base.K, F=base.F)
        res = opt.keep_nm_zero ? solve_meanfield(αinit, pᵢ, opt) : solve(αinit, pᵢ, opt)
        results[i] = res
    αinit = res.α
    end
    return results
end
