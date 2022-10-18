using Distributed 
addprocs(40);
println("# workers = 40")

const problem = ARGS[1]
const Nchains = parse(Int, ARGS[2])
const Njumps = parse(Int, ARGS[3])
const Nsteps = parse(Int, ARGS[4])
const sigma = parse(Float64, ARGS[5])
const dt = parse(Float64, ARGS[6])
const h = parse(Float64, ARGS[7])
const with_rejection = parse(Bool, ARGS[8])
const keep_itm_states = parse(Bool, ARGS[9])
const output_dir = ARGS[10]

println("problem =        ", problem)
println("Nchains =        ", Nchains)
println("Njumps/chain =   ", Njumps)
println("Nsteps/jump =    ", Nsteps)
println("sigma =          ", sigma)
println("dt =             ", dt)
println("h =              ", h)
println("with_rejection = ", with_rejection)
println("keep_itm_states =", keep_itm_states)
println("output_dir =     ", output_dir)


@everywhere begin 
    using DifferentialEquations
    using Random 
    using Distributions

    include("./setups/$($problem).jl")
    include("./generate_data_utils.jl")

    function ode_solve(A, method, p0, q0, t0, H, nsteps, retfull)
        
        h = H/nsteps 
        prob = SecondOrderODEProblem((du,u,p,t)->A(u), p0, q0, (t0, t0+H));
        sol = solve(prob, method, tstops=t0:h:(t0+H), adaptive=false);
        if retfull 
            P = hcat([u.x[1] for u in sol.u]...)   
            Q = hcat([u.x[2] for u in sol.u]...)
            return P, Q
        else 
            p = sol[end].x[1]
            q = sol[end].x[2]
            return p, q
        end
    end
    
    const dt = $dt;
    const h = $h;
    const sigma = $sigma;
    
    phi_dt(p, q) = ode_solve(A, CalvoSanz4(), p, q, 0.0, dt, round(Int, dt/h), false)


    function sample_p_new(p::AbstractArray{T, 1}, q::AbstractArray{T, 1}) where T<:AbstractFloat
        U = compute_U(q)
        K_new = normal(H0-U, abs(sigma*H0))
        while K_new <= 0
            K_new = normal(H0-U, abs(sigma*H0))
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
            
            dH = compute_H(p_new, q_new) - compute_H(p, q)
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
    
    
    function chain_ensemble(p0::AbstractArray{T, 1}, q0::AbstractArray{T, 1}, transition_func::Function;
            num_chains::Int=1, num_transitions::Int=1) where T<:AbstractFloat

        seeds = 1:num_chains
        res = pmap(s->chain(p0, q0, transition_func, num_transitions, s), seeds)

        return vcat(res...)
    end
    
    
#     function chain(p0::AbstractArray{T, 1}, q0::AbstractArray{T, 1}, n_steps::Int, n_jumps::Int, seed::Int) where T<:AbstractFloat
#         Random.seed!(seed)
        
#         n_samples = (n_steps+1) * n_jumps 
#         P = zeros(length(p0), n_samples)
#         Q = zeros(length(q0), n_samples)
        
#         for i in 1:n_jumps
#             c = (n_steps+1) * i - n_steps 
#             P[:, c] = sample_p_new(p0, q0)
#             Q[:, c] = q0
#             for j in 1:n_steps
#                 P[:, c+j], Q[:, c+j] = phi_dt(P[:, c+j-1], Q[:, c+j-1])
#             end
#             p0, q0 = P[:, c+n_steps], Q[:, c+n_steps]
#         end
        
#         return P, Q
#     end
end

println("\nSampling initial states ...")

if problem == "lennardjones"
    p0 = v0
    q0 = x0
end 

if with_rejection 
    res = chain_ensemble(p0, q0, (p, q)->rhmc_H0_transition(p, q; nsteps=Nsteps, keep_intermediate_states=keep_itm_states), 
        num_chains=Nchains, num_transitions=Njumps)
else
    res = chain_ensemble(p0, q0, (p, q)->hmc_H0_transition(p, q; nsteps=Nsteps, keep_intermediate_states=keep_itm_states), 
        num_chains=Nchains, num_transitions=Njumps)
end
    
# chain_seeds = 1:Nchains;
# res = pmap(s->chain(p0, q0, Nsteps, Njumps, s), chain_seeds);
P_init = hcat([p for (p, q) in res]...)
Q_init = hcat([q for (p, q) in res]...)

println("Done.")

save("$output_dir/U0.csv", P_init, Q_init)
