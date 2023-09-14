"""
One-body Kepler problem in 2D
"""

module OneBodyKepler

using LinearAlgebra


function A!(ddu, du, u, p, t) 
    """Compute ddu, second order time derivative of u, in place"""
    ddu[:] = - u ./ norm(u)^3
    nothing 
end


function A(q::AbstractArray{T, 1}) where T<:AbstractFloat
    """Compute second order time derivative of q"""
    return - q ./ norm(q)^3
end


function compute_K(p::AbstractArray{T, 1}) where T<:AbstractFloat
    """Compute kinetic energy"""
    return 0.5 * p' * p
end


function compute_U(q::AbstractArray{T, 1}) where T<:AbstractFloat
    """Compute potential energy"""
    return - 1. / norm(q)
end


function compute_H(p::AbstractArray{T, 1}, q::AbstractArray{T, 1}) where T<:AbstractFloat
    """Compute Hamiltonian"""
    return compute_K(p) + compute_U(q)
end


function initial_condition(T::Type; ecc::Float64)
    """Generate initial condition"""
    ecc = convert(T, ecc)
    q0 = [1-ecc,    0.]
    p0 = [0.,       sqrt((1+ecc)/(1-ecc))]
    return p0, q0
end 

export A!, A_static, compute_K, compute_U, compute_H, initial_condition

end