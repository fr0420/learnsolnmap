"""
Alpha-particle problem 
"""

Base.@kwdef struct AlphaParticle <: AutonomousODESystem
    epsilon::Float64
    B0::Float64 = 1.0
    a1::Float64 = 0.3
    a2::Float64 = 0.3
    k_x1::Float64 = 3.0
    k_y1::Float64 = 1.0
    k_x2::Float64 = 1.0
    k_y2::Float64 = 3.0
end

mass(prob::AlphaParticle) = [1., 1.]

function initial_condition(prob::AlphaParticle, T::Type)
    u0 = T.([sqrt(2.), 0., 2.5, 3.0])
    # u0 = T.([1., -1., 2.5, 3.0])
    return u0
end 

function nondimensionalize(prob::AlphaParticle, u::AbstractArray{T, 1}) where T<:AbstractFloat
    return u
end

function dimensionalize(prob::AlphaParticle, u_nd::AbstractArray{T, 1}) where T<:AbstractFloat
    return u_nd
end

B(prob::AlphaParticle, x, y) = prob.B0 + prob.a1 * cos(prob.k_x1 * x + prob.k_y1 * y) + prob.a2 * cos(prob.k_x2 * x + prob.k_y2 * y)

function compute_du!(prob::AlphaParticle, du, u)     
    # u = [vx, vy, x, y]
    du[1] = B(prob, u[3], u[4]) * u[2]
    du[2] = -B(prob, u[3], u[4]) * u[1]
    du[3] = prob.epsilon * u[1]
    du[4] = prob.epsilon * u[2]

    nothing 
end

function compute_energy(prob::AlphaParticle, u)
    # u = [vx, vy, x, y]
    return 0.5 * (u[1]^2 + u[2]^2)
end

function embed_state_procrustes(prob::AlphaParticle, u::AbstractArray{T, 1}) where T<:AbstractFloat
    return u[1:2]
end

function align_state_procrustes(prob::AlphaParticle, u::AbstractArray{T, 1}, corrector::Any) where T<:AbstractFloat
    return [corrector(u[1:2]); u[3:4]]
end

function embed_state_interpolative(prob::AlphaParticle, u::AbstractArray{T, 1}) where T<:AbstractFloat
    return u
end

function align_state_interpolative(prob::AlphaParticle, u::AbstractArray{T, 1}, corrector::Any) where T<:AbstractFloat
    return corrector(u)
end