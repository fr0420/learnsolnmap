"""
Two-body Kepler problem in 3D
"""

using LinearAlgebra


const g12 = 1e-5;
const e1 = 0.4;
const e2 = 0.5;


function A(q::AbstractArray{T, 1}) where T<:AbstractFloat
    """Compute second order time derivative of q"""
    
    q1 = q[1:3]
    q2 = q[4:6]
    
    q_ddot = zero(q)
    q_ddot[1:3] = - q1 ./ norm(q1)^3 - g12*(q1-q2) ./ norm(q1-q2)^3
    q_ddot[4:6] = - q2 ./ norm(q2)^3 + g12*(q1-q2) ./ norm(q1-q2)^3
    
    return q_ddot
end


function compute_K(p::AbstractArray{T, 1}) where T<:AbstractFloat
    """Compute kinetic energy"""
    return 0.5 * p' * p
end


function compute_U(q::AbstractArray{T, 1}) where T<:AbstractFloat
    """Compute potential energy"""
    q1 = q[1:3]
    q2 = q[4:6]
    U = - (1/norm(q1) + 1/norm(q2) + g12/norm(q1-q2))
    return U
end


function compute_H(p::AbstractArray{T, 1}, q::AbstractArray{T, 1}) where T<:AbstractFloat
    """Compute Hamiltonian"""
    return compute_K(p) + compute_U(q)
end


# Initial conditions 
q0 = [1-e1, 0., 0., cos(pi/4)*(1-e2), 0., sin(pi/4)*(1-e2)];
p0 = [0., sqrt((1+e1)/(1-e1)), 0., 0., sqrt((1+e2)/(1-e2)), 0.];

# Initial energy 
K0 = compute_K(p0);
U0 = compute_U(q0);
H0 = compute_H(p0, q0);