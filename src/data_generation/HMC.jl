"""
Hamiltonian Monte Carlo methods.
"""

module HMC

using Distributed
using Logging
using ProgressMeter
using Random
using LinearAlgebra
include("sampling.jl")


"""
Generate a Markov chain with a specified transition function.

Arguments:
    v0: initial velocity
    x0: initial position
    transition_func: transition function
    num_transitions: number of transitions
    seed: random seed

Returns:
    samples: vector of state tuples
    num_rejections: number of rejections
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
    num_rejections = 0 

    v, x = copy(v0), copy(x0)

    for i in 1:num_transitions
        res = transition_func(v, x)
        
        if res isa Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}
            push!(samples, res)
            update_state!(v, x, res)
        elseif res isa Tuple{Vector{Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}}, Int}
            append!(samples, res[1])
            num_rejections += res[2]
            update_state!(v, x, res[1][end])
        else
            error("Invalid return type from transition function. Must return either:
                  - Tuple(next_v, next_x) or
                  - Tuple(Vector of state tuples, num_rejections)")
        end
    end

    return samples, num_rejections
end


"""
Sample an ensemble of chains with different random seeds.

Arguments: 
    v0: initial velocity
    x0: initial position
    transition_func: transition function
    num_chains: number of chains
    num_transitions: number of transitions

Returns:
    samples: vector of state tuples
    total_rejections: total number of rejections
"""
function chain_ensemble(
    v0::AbstractArray{T, 1}, 
    x0::AbstractArray{T, 1}, 
    transition_func::Function;
    num_chains::Int = 1, 
    num_transitions::Int = 1
    ) where T<:AbstractFloat

    seeds = 1:num_chains

    results = @showprogress pmap(seeds) do s
        chain(v0, x0, transition_func, num_transitions, s)
    end

    samples = reduce(vcat, [r[1] for r in results])
    total_rejections = sum([r[2] for r in results])

    return samples, total_rejections
end


"""
Sample an ensemble of chains with different initial conditions.

Arguments: 
    initial_conditions: vector of initial conditions
    transition_func: transition function
    num_transitions: number of transitions

Returns:
    samples: vector of state tuples
    total_rejections: total number of rejections
"""
function chain_ensemble(
    initial_conditions::Vector{<:Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}},
    transition_func::Function;
    num_transitions::Int = 1
    ) where T<:AbstractFloat

    num_chains = length(initial_conditions) 
    seeds = 1:num_chains

    results = @showprogress pmap(1:num_chains) do i
        v0, x0 = initial_conditions[i]
        chain(v0, x0, transition_func, num_transitions, seeds[i])
    end

    samples = reduce(vcat, [r[1] for r in results])
    total_rejections = sum([r[2] for r in results])

    return samples, total_rejections
end


"""
Sample initial conditions by sampling velocities with perturbed kinetic energy.

Arguments:
    v0: initial velocity
    x0: initial position
    mass: mass vector
    num_samples: number of samples
    epsilon: perturbation factor
    seed: random seed

Returns:
    initial_conditions: vector of state tuples
"""
function sample_initial_conditions(
    v0::AbstractArray{T, 1}, 
    x0::AbstractArray{T, 1}, 
    mass::AbstractArray{Float64, 1};
    num_samples::Int = 1, 
    epsilon::Float64 = 1e-2,
    seed::Int = 1
    ) where T<:AbstractFloat

    Random.seed!(seed)
    K0 = 0.5 * v0' * (mass .* v0)
    K_new = truncated_normal(K0, epsilon*K0, T(0.), T(Inf), num_samples)
    
    vhat = nSphereSampling(length(v0), num_samples)
    v = vhat ./ sqrt.(mass) .* sqrt.(2 * K_new')

    return [(v[:, i], x0) for i in 1:num_samples]
end


"""
Sample initial conditions by sampling velocities with perturbed kinetic energy, 
total momentum and angular momentum. (for n-body problems)

Arguments:
    v0: initial velocity
    x0: initial position
    mass: mass vector
    num_samples: number of samples
    epsilon: perturbation factor
    seed: random seed

Returns:
    initial_conditions: vector of state tuples
"""
function sample_initial_conditions_3body(
    v0::AbstractArray{T, 1}, 
    x0::AbstractArray{T, 1}, 
    mass::AbstractArray{Float64, 1};
    num_samples::Int = 1, 
    epsilon::Float64 = 1e-2,
    seed::Int = 1
    ) where T<:AbstractFloat

    Random.seed!(seed)
    K0 = 0.5 * v0' * (mass .* v0)

    q1 = @view x0[1:2]
    q2 = @view x0[3:4]
    q3 = @view x0[5:6]
    p0 = mass .* v0
    p1 = @view p0[1:2]
    p2 = @view p0[3:4]
    p3 = @view p0[5:6]
    p_tot = p1 + p2 + p3
    Lz = q1[1]*p1[2] - q1[2]*p1[1] + q2[1]*p2[2] - q2[2]*p2[1] + q3[1]*p3[2] - q3[2]*p3[1]
    M = Diagonal(T.(mass))
    A = [1.0 0.0 1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0 0.0 1.0; -q1[2] q1[1] -q2[2] q2[1] -q3[2] q3[1]]
    b = [p_tot[1], p_tot[2], Lz]
    eps_c = T(epsilon)
    eps_b = epsilon * ones(T, 3)
    phat = ellipsoidLinearConstraintsSampling(M/2, K0, A, b, eps_c, eps_b, num_samples)
    v = phat ./ mass

    return [(v[:, i], x0) for i in 1:num_samples]
end



"""
Sample initial conditions by sampling velocities with perturbed kinetic energy, 
total momentum and angular momentum. (for n-body problems)

Arguments:
    v0: initial velocity
    x0: initial position
    mass: mass vector
    num_samples: number of samples
    epsilon: perturbation factor
    seed: random seed

Returns:
    initial_conditions: vector of state tuples
"""
function sample_initial_conditions_3body_equilateral(
    ::Type{T} = Float64;
    num_samples::Int = 1, 
    nu::Float64 = 0.2,
    min_radius::Float64 = 0.9,
    max_radius::Float64 = 1.2,
    seed::Int = 1
) where T<:AbstractFloat

    nu = T(nu)
    min_radius = T(min_radius)
    max_radius = T(max_radius)

    Random.seed!(seed)
    
    # positions that form a equilateral triangle
    q1 = ShellSampling(num_samples, [(min_radius, max_radius), (min_radius, max_radius)])
    q2 = rotate2d(q1, 2*pi/3)
    q3 = rotate2d(q2, 2*pi/3)
    r = sqrt.(sum(q1.^2, dims=1))

    # velocities that yield a circular orbit
    v1 = rotate2d(q1, pi/2)
    v1 ./= r.^1.5 
    v1 .*= sqrt(sin(pi/3)/(2*cos(pi/6)^2))
    v2 = rotate2d(v1, 2*pi/3)
    v3 = rotate2d(v2, 2*pi/3)

    # make the circular orbits slightly chaotic
    v1 .*= 1 .+ nu.*(2 .* rand(2, num_samples) .- 1)
    v2 .*= 1 .+ nu.*(2 .* rand(2, num_samples) .- 1)
    v3 .*= 1 .+ nu.*(2 .* rand(2, num_samples) .- 1)
    
    v = vcat(v1, v2, v3)
    x = vcat(q1, q2, q3)

    return [(v[:, i], x[:, i]) for i in 1:num_samples]
end


function rotate2d(v::AbstractArray{T, 2}, theta::Float64) where T<:AbstractFloat
    R = [cos(theta) -sin(theta); sin(theta) cos(theta)]
    return R * v
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

Returns:
    res: vector of state tuples
    num_rejections: number of rejections
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
    rejection_counter = [0]

    # step 1: momentum refreshment
    refresh_momentum_hmc_H0!(v, x, mass)

    # step 2: time integration with accept/reject mechanism
    for n in 1:nsteps
        v_temp, x_temp = phi_dt(v, x)

        if with_rejection
            acceptance = compute_acceptance_hmc_H0(v, x, v_temp, x_temp, compute_H, mass)
            update_state!(v, x, v_temp, x_temp, acceptance, rejection_counter)
        else
            update_state!(v, x, v_temp, x_temp)
        end
        res[n] = (copy(v), copy(x))
    end 

    return res, rejection_counter[1]
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

Returns:
    res: vector of state tuples
    num_rejections: number of rejections
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
    rejection_counter = [0]

    # step 1: momentum refreshment
    refresh_momentum_hmc_H0!(v, x, mass)

    # step 2: random time integration with accept/reject mechanism
    for n in 1:nsteps
        v_temp, x_temp = random_time_integration(v, x, phi_h, mean_dt_div_h)

        if with_rejection
            acceptance = compute_acceptance_hmc_H0(v, x, v_temp, x_temp, compute_H, mass)
            update_state!(v, x, v_temp, x_temp, acceptance, rejection_counter)
        else
            update_state!(v, x, v_temp, x_temp)
        end
        res[n] = (copy(v), copy(x))
    end 

    return res, rejection_counter[1]
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

Returns:
    res: vector of state tuples
    num_rejections: number of rejections
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
    rejection_counter = [0]

    # step 1: momentum refreshment
    refresh_momentum_hmc!(v, mass; beta=beta)

    # step 2: time integration with accept/reject mechanism
    for n in 1:nsteps
        v_temp, x_temp = phi_dt(v, x)

        if with_rejection
            acceptance = compute_acceptance_hmc(v, x, v_temp, x_temp, compute_H)
            update_state!(v, x, v_temp, x_temp, acceptance, rejection_counter)
        else
            update_state!(v, x, v_temp, x_temp)
        end
        res[n] = (copy(v), copy(x))
    end 

    return res, rejection_counter[1]
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

Returns:
    res: vector of state tuples
    num_rejections: number of rejections
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
    rejection_counter = [0]

    # step 1: momentum refreshment    
    refresh_momentum_hmc!(v, mass; beta=beta)

    # step 2: random time integration with accept/reject mechanism
    for n in 1:nsteps
        v_temp, x_temp = random_time_integration(v, x, phi_h, mean_dt_div_h)

        if with_rejection
            acceptance = compute_acceptance_hmc(v, x, v_temp, x_temp, compute_H)
            update_state!(v, x, v_temp, x_temp, acceptance, rejection_counter)
        else
            update_state!(v, x, v_temp, x_temp)
        end
        res[n] = (copy(v), copy(x))
    end 

    return res, rejection_counter[1]
end


### Helper functions 

"""Refresh the current momentum (velocity) for HMC algorithms."""
function refresh_momentum_hmc!(v::AbstractArray{T, 1}, mass::AbstractArray{Float64, 1}; beta::Float64 = 1.0) where T <: AbstractFloat
    # Scale a Gaussian random vector by 1/sqrt(mass*beta)
    v .= randn(length(v)) ./ sqrt.(mass * beta)
end


"""Refresh the current momentum (velocity) for HMC-H0 algorithms."""
function refresh_momentum_hmc_H0!(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}, mass::AbstractArray{Float64, 1}) where T <: AbstractFloat

    # Scale a random unit vector by 1/sqrt(mass) * sqrt(2K)
    K = 0.5 * v' * (mass .* v)
    # v .= nSphereSampling(length(v)) ./ sqrt.(mass) * sqrt(2 * K)

    # Temporary solution for 3-body problem: preserve the total momentum and angular momentum
    q1 = @view x[1:2]
    q2 = @view x[3:4]
    q3 = @view x[5:6]
    p = mass .* v
    p1 = @view p[1:2]
    p2 = @view p[3:4]
    p3 = @view p[5:6]
    p_tot = p1 + p2 + p3
    Lz = q1[1]*p1[2] - q1[2]*p1[1] + q2[1]*p2[2] - q2[2]*p2[1] + q3[1]*p3[2] - q3[2]*p3[1]
    M = Diagonal(T.(mass))
    A = [1.0 0.0 1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0 0.0 1.0; -q1[2] q1[1] -q2[2] q2[1] -q3[2] q3[1]]
    b = [p_tot[1], p_tot[2], Lz]
    v .= ellipsoidLinearConstraintsSampling(M, 2*K, A, b) ./ mass
end


"""Random time integration with a geometrically distributed number of sub-steps."""
function random_time_integration(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}, phi_h::Function, mean_dt_div_h::Float64) where T <: AbstractFloat
    
    # Generate a random number of sub-steps
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
function compute_acceptance_hmc_H0(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}, v_temp::AbstractArray{T, 1}, x_temp::AbstractArray{T, 1}, compute_H::Function, mass::AbstractArray{Float64, 1}) where T<:AbstractFloat
    # TODO: implement accept/reject mechanism for HMC-H0 algorithms
    H_temp = compute_H(v_temp, x_temp) 
    H = compute_H(v, x)
    K_temp = 0.5 * v_temp' * (mass .* v_temp)
    return abs(H_temp - H) < 1e-10 ? acceptance = 1.0 : acceptance = 0.0
    # return abs(H_temp - H) < 1e-10 && K_temp < 3.0 ? acceptance = 1.0 : acceptance = 0.0
    # return abs(H_temp - H) < 1e-10 && K_temp < 3.0 && K_temp > 0.3 ? acceptance = 1.0 : acceptance = 0.0
end


"""Update the state based on the acceptance probability."""
function update_state!(
    v::AbstractArray{T, 1}, x::AbstractArray{T, 1}, v_temp::AbstractArray{T, 1}, x_temp::AbstractArray{T, 1}, 
    acceptance::Float64, num_rejections::Vector{Int}
) where T<:AbstractFloat
    gamma = bernoulli(acceptance)
    if gamma == 1
        copyto!(v, v_temp)
        copyto!(x, x_temp)
    else
        # v .= -v
        num_rejections .+= 1
        # @info "Transition rejected (acceptance rate = $acceptance). State updated by reversing the momentum."
    end
end


"""Update the state."""
function update_state!(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}, v_temp::AbstractArray{T, 1}, x_temp::AbstractArray{T, 1}) where T<:AbstractFloat
    copyto!(v, v_temp)
    copyto!(x, x_temp)
end


"""Update the state."""
function update_state!(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}, state_tuple::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}) where T<:AbstractFloat
    copyto!(v, state_tuple[1])
    copyto!(x, state_tuple[2])
end


export hmc_H0_transition, rhmc_H0_transition, hmc_transition, rhmc_transition, 
    chain_ensemble, sample_initial_conditions, sample_initial_conditions_3body, sample_initial_conditions_3body_equilateral

end


if ARGS == ["--run"]

    # Example usage
    using .HMC

    samples = HMC.sample_initial_conditions_3body_equilateral(num_samples=10)
    println(samples)
end