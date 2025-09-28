"""
    ContinuationTrace

Container returned by [`continuation_trace`](@ref) that records requested detuning
values, the solver results that were obtained, and where the sweep stopped when
early termination was requested.
"""
Base.@kwdef struct ContinuationTrace
    requested::Vector{Float64}
    results::Vector{Result}
    stop_index::Union{Int,Nothing}=nothing
end

continuation_completed(trace::ContinuationTrace)::Int = length(trace.results)

function continuation_failures(trace::ContinuationTrace)
    return findall(r -> !converged(r), trace.results)
end

continuation_stopped(trace::ContinuationTrace)::Bool = trace.stop_index !== nothing

continuation_results(trace::ContinuationTrace) = trace.results

continuation_requested(trace::ContinuationTrace) = trace.requested

continuation_stopindex(trace::ContinuationTrace) = trace.stop_index

Base.eltype(::Type{ContinuationTrace}) = Result

Base.IndexStyle(::Type{ContinuationTrace}) = IndexLinear()

Base.length(trace::ContinuationTrace) = length(trace.results)

Base.firstindex(::ContinuationTrace) = 1

Base.lastindex(trace::ContinuationTrace) = length(trace)

function Base.getindex(trace::ContinuationTrace, idx::Int)
    return trace.results[idx]
end

function Base.iterate(trace::ContinuationTrace)
    return iterate(trace.results)
end

function Base.iterate(trace::ContinuationTrace, state)
    return iterate(trace.results, state)
end

"""
    continuation_trace(Δs, base, α0, config; stop_on_failure=false)

Run a continuation over detuning values and return trace metadata.
"""
function continuation_trace(
    Δs::AbstractVector{T},
    base::Params,
    α0::Complex,
    config::SolverConfig=SolverConfig();
    stop_on_failure::Bool=false,
) where {T<:Real}
    requested = Float64[Float64(Δi) for Δi in Δs]
    results = Vector{Result}()
    sizehint!(results, length(requested))
    αinit = ComplexF64(α0)
    stop_index = nothing

    for (i, Δi) in enumerate(requested)
        pᵢ = Params(; Δ=Δi, K=base.K, F=base.F)
        is_meanfield = config.physics.keep_nm_zero
        res = is_meanfield ? solve_meanfield(αinit, pᵢ, config) : solve(αinit, pᵢ, config)
        push!(results, res)
        αinit = res.α

        if stop_on_failure && !res.converged
            stop_index = i
            break
        end
    end

    completed = length(results)
    completed_requested = completed == 0 ? Float64[] : requested[1:completed]
    return ContinuationTrace(
        ; requested=completed_requested, results=results, stop_index=stop_index,
    )
end

"""
    sweep_delta(Δs, base, α0, config)

Sweep detuning values with warm-start continuation.
"""
function sweep_delta(
    Δs::AbstractVector{T}, base::Params, α0::Complex, config::SolverConfig=SolverConfig(),
) where {T<:Real}
    trace = continuation_trace(Δs, base, α0, config; stop_on_failure=false)
    return continuation_results(trace)
end
