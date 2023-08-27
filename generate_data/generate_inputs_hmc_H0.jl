"""
HMC-H0 
"""

const problem = ARGS[1]
const Nchains = parse(Int, ARGS[2])
const Njumps = parse(Int, ARGS[3])
const Nsteps = parse(Int, ARGS[4])
const sigma = parse(Float64, ARGS[5])
const dt = parse(Float64, ARGS[6])
const Nf = parse(Int, ARGS[7])
const method = ARGS[8]
const with_rejection = parse(Bool, ARGS[9])
const keep_itm_states = parse(Bool, ARGS[10])
const output_dir = ARGS[11]
const use_float64x4 = parse(Bool, ARGS[12])

println("problem =        ", problem)
println("Nchains =        ", Nchains)
println("Njumps/chain =   ", Njumps)
println("Nsteps/jump =    ", Nsteps)
println("sigma =          ", sigma)
println("dt =             ", dt)
println("Nf =             ", Nf)
println("h =              ", dt/Nf)
println("method =         ", method)
println("with_rejection = ", with_rejection)
println("keep_itm_states =", keep_itm_states)
println("output_dir =     ", output_dir)
println("use_float64x4 =  ", use_float64x4)


using MultiFloats
using Distributed 
addprocs(40);
println("# workers = 40")
@everywhere include("../tools/setups/$($problem).jl")


if problem == "fpu"
    const OMEGA = 50.
    const param = use_float64x4 ? (Float64x4(OMEGA)^2)/2. : (OMEGA^2)/2.
    
    kwargs = Dict(:omega => use_float64x4 ? Float64x4(OMEGA) : OMEGA)
    println("\n", kwargs)
end 

p0, q0 = initial_condition(; kwargs...)
H0 = compute_H(p0, q0; kwargs...)
K0 = compute_K(p0)
U0 = compute_U(q0; kwargs...)

println("\nInitial condition:")
println("p0 = ", p0)
println("q0 = ", q0)
println("H0 = ", H0)
println("K0 = ", K0)
println("U0 = ", U0)


@everywhere begin 
    include("../tools/ode_solver.jl")
    include("./generate_data_utils.jl")
    using Random 
    using Distributions
    using ProgressMeter 

    phi_dt(p, q) = ode_solve(A!, methods[$method], p, q, 0.0, $dt, $Nf, false, $param)
    
    function sample_p_new(p::AbstractArray{T, 1}, q::AbstractArray{T, 1}) where T<:AbstractFloat
        U = compute_U(q; $kwargs...)
        K_new = normal($H0-U, abs($sigma*$H0))
        while K_new <= 0
            K_new = normal($H0-U, abs($sigma*$H0))
        end
        
        if $problem == "lennardjones"
            p_new = nSphereSampling(length(p)) * sqrt(2*K_new/MASS)
        else
            p_new = nSphereSampling(length(p)) * sqrt(2*K_new)
        end  
        
        return p_new
    end

    function hmc_H0_transition(p::AbstractArray{T, 1}, q::AbstractArray{T, 1}; 
            nsteps::Int, keep_intermediate_states::Bool) where T<:AbstractFloat
        
        res = []
        
        # step 1: momentum refreshment
        p = sample_p_new(p, q)
        
        # step 2: integration in time 
        for n in 1:nsteps
            if keep_intermediate_states
                push!(res, (p, q))
            end
            p, q = phi_dt(p, q)
        end 
        
        push!(res, (p, q))
        return res
    end

    
    function rhmc_H0_transition(p::AbstractArray{T, 1}, q::AbstractArray{T, 1}; 
            nsteps::Int, keep_intermediate_states::Bool) where T<:AbstractFloat
        
        res = []
        
        # step 1: momentum refreshment
        p = sample_p_new(p, q)
        
        # step 2: integration in time with accept/reject mechanism
        for n in 1:nsteps
            if keep_intermediate_states
                push!(res, (p, q))
            end
            p_new, q_new = phi_dt(p, q)
            
            dH = compute_H(p_new, q_new; $kwargs...) - compute_H(p, q; $kwargs...)
            dH = Float64(dH)
            dist = Binomial(1, min(1, exp(-dH)))
            gamma = rand(dist)
            if gamma == 0
                p = -p
                println("reject")
            else
                p = p_new
                q = q_new
            end 
        end 
        
        push!(res, (p, q))
        return res
    end
    
    
    function chain(p0::AbstractArray{T, 1}, q0::AbstractArray{T, 1}, transition_func::Function,
            num_transitions::Int, seed::Int) where T<:AbstractFloat
        Random.seed!(seed)
        
        samples = []
        p = p0
        q = q0

        for i in 1:num_transitions
            res = transition_func(p, q)
            samples = vcat(samples, res)
            p, q = res[end]
        end
        return samples 
    end
end


function chain_ensemble(p0::AbstractArray{T, 1}, q0::AbstractArray{T, 1}, transition_func::Function;
        num_chains::Int=1, num_transitions::Int=1) where T<:AbstractFloat

    seeds = 1:num_chains
    res = @showprogress pmap(s->chain(p0, q0, transition_func, num_transitions, s), seeds)

    return vcat(res...)
end
    

println("\nSampling chains ...")

if with_rejection 
    res = chain_ensemble(p0, q0, (p, q)->rhmc_H0_transition(p, q; nsteps=Nsteps, keep_intermediate_states=keep_itm_states), 
        num_chains=Nchains, num_transitions=Njumps)
else
    res = chain_ensemble(p0, q0, (p, q)->hmc_H0_transition(p, q; nsteps=Nsteps, keep_intermediate_states=keep_itm_states), 
        num_chains=Nchains, num_transitions=Njumps)
end
    
P = hcat([p for (p, q) in res]...)
Q = hcat([q for (p, q) in res]...)

println("Done.")

save_csv("$output_dir/U0.csv", P, Q, use_float64x4 ? Float64x4 : Float64)
