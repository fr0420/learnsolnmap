"""
Hamiltonian Monte Carlo methods.
"""

module HMC

using Distributed
using Logging
using ProgressMeter
using Random
include("sampling.jl")


"""
Generate a Markov chain with a specified transition function.

Arguments:
    v0: initial velocity
    x0: initial position
    transition_func: transition function
    num_transitions: number of transitions
    seed: random seed
"""
function chain(
    v0::AbstractArray{T, 1}, 
    x0::AbstractArray{T, 1}, 
    transition_func::Function,
    num_transitions::Int, 
    seed::Int
    ) where T<:AbstractFloat

    Random.seed!(seed)        
    samples = Vector{Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}}()
    v, x = v0, x0

    for i in 1:num_transitions
        res = transition_func(v, x)
        
        if res isa Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}
            push!(samples, res)
            v, x = res
        elseif res isa Vector{Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}}
            append!(samples, res)
            v, x = res[end]
        else
            error("Invalid return type from transition function. Transition function must return a tuple or a vector of tuples of (AbstractArray{T, 1}, AbstractArray{T, 1}).")
        end
    end

    return samples 
end


"""
Sample an ensemble of chains with different random seeds.

Arguments: 
    v0: initial velocity
    x0: initial position
    transition_func: transition function
    num_chains: number of chains
    num_transitions: number of transitions
"""
function chain_ensemble(
    v0::AbstractArray{T, 1}, 
    x0::AbstractArray{T, 1}, 
    transition_func::Function;
    num_chains::Int = 1, 
    num_transitions::Int = 1
    ) where T<:AbstractFloat

    seeds = 1:num_chains
    res = @showprogress pmap(s->chain(v0, x0, transition_func, num_transitions, s), seeds)

    return reduce(vcat, res)
end


"""
Sample an ensemble of chains with different initial conditions.

Arguments: 
    initial_conditions: vector of initial conditions
    transition_func: transition function
    num_transitions: number of transitions
"""
function chain_ensemble(
    initial_conditions::Vector{<:Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}},
    transition_func::Function;
    num_transitions::Int = 1
    ) where T<:AbstractFloat

    num_chains = length(initial_conditions) 
    seeds = 1:num_chains
    res = @showprogress pmap(1:num_chains) do i
        v0, x0 = initial_conditions[i]
        chain(v0, x0, transition_func, num_transitions, seeds[i])
    end

    return reduce(vcat, res)
end


"""
Sample initial conditions by sampling velocities with perturbed kinetic energy.

Arguments:
    v0: initial velocity
    x0: initial position
    mass: mass vector
    num_samples: number of samples
    epsilon: perturbation factor
"""
function sample_initial_conditions(
    v0::AbstractArray{T, 1}, 
    x0::AbstractArray{T, 1}, 
    mass::AbstractArray{Float64, 1};
    num_samples::Int = 1, 
    epsilon::Float64 = 1e-2,
    ) where T<:AbstractFloat

    K0 = 0.5 * v0' * (mass .* v0)
    K_new = truncated_normal(K0, epsilon*K0, T(0.), T(Inf), num_samples)
    
    vhat = nSphereSampling(length(v0), num_samples)
    v = vhat ./ sqrt.(mass) .* sqrt.(2 * K_new')

    return [(v[:, i], x0) for i in 1:num_samples]
end


"""
HMC-H0 transition function.

Arguments:
    v: current velocity
    x: current position
    mass: mass vector
    compute_H: Hamiltonian function
    phi_dt: time integration function with stepsize dt
    nsteps: number of time steps
    with_rejection: whether to use accept/reject mechanism
"""
function hmc_H0_transition(
    v::AbstractArray{T, 1}, 
    x::AbstractArray{T, 1},
    mass::AbstractArray{Float64, 1},
    compute_H::Function,
    phi_dt::Function;
    nsteps::Int = 1,
    with_rejection::Bool = false,
    ) where T<:AbstractFloat
        
    res = Vector{Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}}(undef, nsteps)

    # step 1: momentum refreshment
    refresh_momentum(v, mass; preserve_kinetic_energy=true)

    # step 2: time integration with accept/reject mechanism
    for n in 1:nsteps
        v_temp, x_temp = phi_dt(v, x)

        if with_rejection
            acceptance = compute_acceptance_hmc_H0(v, x, v_temp, x_temp, compute_H)
            update_state!(v, x, v_temp, x_temp, acceptance)
        else
            update_state!(v, x, v_temp, x_temp)
        end
        res[n] = (copy(v), copy(x))
    end 

    return res
end


"""
Randomized HMC-H0 transition function.

Arguments:
    v: current velocity
    x: current position
    mass: mass vector
    compute_H: Hamiltonian function
    phi_h: time integration function with sub-stepsize h and variable number of sub-steps
    mean_dt_div_h: mean number of sub-steps
    nsteps: number of time steps
    with_rejection: whether to use accept/reject mechanism
"""
function rhmc_H0_transition(
    v::AbstractArray{T, 1}, 
    x::AbstractArray{T, 1},
    mass::AbstractArray{Float64, 1},
    compute_H::Function,
    phi_h::Function,
    mean_dt_div_h::Float64;
    nsteps::Int = 1,
    with_rejection::Bool = false,
    ) where T<:AbstractFloat
        
    res = Vector{Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}}(undef, nsteps)

    # step 1: momentum refreshment
    refresh_momentum(v, mass; preserve_kinetic_energy=true)

    # step 2: random time integration with accept/reject mechanism
    for n in 1:nsteps
        v_temp, x_temp = random_time_integration(v, x, phi_h, mean_dt_div_h)

        if with_rejection
            acceptance = compute_acceptance_hmc_H0(v, x, v_temp, x_temp, compute_H)
            update_state!(v, x, v_temp, x_temp, acceptance)
        else
            update_state!(v, x, v_temp, x_temp)
        end
        res[n] = (copy(v), copy(x))
    end 

    return res
end


"""
HMC transition function (Bou-Rabee and Sanz-Serna, 2018).
    
Arguments: 
    v: current velocity
    x: current position
    mass: mass vector
    compute_H: Hamiltonian function
    phi_dt: time integration function with stepsize dt
    nsteps: number of time steps
    beta: inverse temperature
    with_rejection: whether to use accept/reject mechanism
"""
function hmc_transition(
    v::AbstractArray{T, 1}, 
    x::AbstractArray{T, 1},
    mass::AbstractArray{Float64, 1},
    compute_H::Function,
    phi_dt::Function;
    nsteps::Int = 1,
    beta::Float64 = 1.,
    with_rejection::Bool = true,
    ) where T<:AbstractFloat

    res = Vector{Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}}(undef, nsteps)

    # step 1: momentum refreshment
    refresh_momentum(v, mass; beta=beta)

    # step 2: time integration with accept/reject mechanism
    for n in 1:nsteps
        v_temp, x_temp = phi_dt(v, x)

        if with_rejection
            acceptance = compute_acceptance_hmc(v, x, v_temp, x_temp, compute_H)
            update_state!(v, x, v_temp, x_temp, acceptance)
        else
            update_state!(v, x, v_temp, x_temp)
        end
        res[n] = (copy(v), copy(x))
    end 

    return res
end


"""
Randomized HMC transition function (Bou-Rabee and Sanz-Serna, 2018).

Arguments:
    v: current velocity
    x: current position
    mass: mass vector
    compute_H: Hamiltonian function
    phi_h: time integration function with sub-stepsize h and variable number of sub-steps
    mean_dt_div_h: mean number of sub-steps
    nsteps: number of time steps
    beta: inverse temperature
    with_rejection: whether to use accept/reject mechanism
"""
function rhmc_transition(
    v::AbstractArray{T, 1}, 
    x::AbstractArray{T, 1},
    mass::AbstractArray{Float64, 1},
    compute_H::Function,
    phi_h::Function,
    mean_dt_div_h::Float64;
    nsteps::Int = 1,
    beta::Float64 = 1.,
    with_rejection::Bool = true,
    ) where T<:AbstractFloat

    res = Vector{Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}}(undef, nsteps)

    # step 1: momentum refreshment    
    refresh_momentum(v, mass; beta=beta)

    # step 2: random time integration with accept/reject mechanism
    for n in 1:nsteps
        v_temp, x_temp = random_time_integration(v, x, phi_h, mean_dt_div_h)

        if with_rejection
            acceptance = compute_acceptance_hmc(v, x, v_temp, x_temp, compute_H)
            update_state!(v, x, v_temp, x_temp, acceptance)
        else
            update_state!(v, x, v_temp, x_temp)
        end
        res[n] = (copy(v), copy(x))
    end 

    return res
end


### Helper functions 

"""Refresh the current momentum (velocity)."""
function refresh_momentum(v::AbstractArray{T, 1}, mass::AbstractArray{Float64, 1}; beta::Float64 = 1.0, preserve_kinetic_energy::Bool = false) where T <: AbstractFloat
    if preserve_kinetic_energy
        # Preserve kinetic energy: scale the random unit vector by 1/sqrt(mass) * sqrt(2K)
        K = 0.5 * v' * (mass .* v)
        vhat = nSphereSampling(length(v))
        v .= vhat ./ sqrt.(mass) * sqrt(2 * K)
    else
        # Standard momentum refresh: scale the Gaussian random vector by 1/sqrt(mass*beta).
        vhat = randn(length(v))
        v .= vhat ./ sqrt.(mass) / sqrt(beta)
    end
end


"""Random time integration with a geometrically distributed number of sub-steps."""
function random_time_integration(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}, phi_h::Function, mean_dt_div_h::Float64) where T <: AbstractFloat
    
    # Generate a random time step using a geometric distribution
    m = geometric(1. / mean_dt_div_h)

    # Perform 'm' sub-steps of the integrator phi_h
    v_temp, x_temp = phi_h(v, x, m)

    return v_temp, x_temp
end


"""Compute the acceptance probability for HMC algorithms."""
function compute_acceptance_hmc(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}, v_temp::AbstractArray{T, 1}, x_temp::AbstractArray{T, 1}, compute_H::Function) where T<:AbstractFloat
    dH = compute_H(v_temp, x_temp) - compute_H(v, x)
    acceptance = min(1, exp(-dH))
    return Float64(acceptance)
end


"""Compute the acceptance probability for HMC-H0 algorithms."""
function compute_acceptance_hmc_H0(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}, v_temp::AbstractArray{T, 1}, x_temp::AbstractArray{T, 1}, compute_H::Function) where T<:AbstractFloat
    # TODO: implement accept/reject mechanism for HMC-H0 algorithms
    acceptance = 1.  # placeholder for now
    return acceptance
end


"""Update the state based on the acceptance probability."""
function update_state!(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}, v_temp::AbstractArray{T, 1}, x_temp::AbstractArray{T, 1}, acceptance::Float64) where T<:AbstractFloat
    gamma = bernoulli(acceptance)
    if gamma == 1
        copyto!(v, v_temp)
        copyto!(x, x_temp)
    else
        v .= -v
        @info "Transition rejected (acceptance rate = $acceptance). State updated by reversing the momentum."
    end
end


"""Update the state."""
function update_state!(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}, v_temp::AbstractArray{T, 1}, x_temp::AbstractArray{T, 1}) where T<:AbstractFloat
    copyto!(v, v_temp)
    copyto!(x, x_temp)
end


export hmc_H0_transition, rhmc_H0_transition, hmc_transition, rhmc_transition, chain_ensemble

end