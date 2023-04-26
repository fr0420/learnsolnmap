"""
Fermi-Pasta-Ulam Problem 
"""


const m = 3;


function A(q::AbstractArray{T, 1}; omega::T=300.) where T<:AbstractFloat
    """Compute second order time derivative of q"""
    
    dq_stiff = q[2:2:end] - q[1:2:end]
    q_pad = vcat([0], q, [0]) 
    dq_soft = q_pad[2:2:end] - q_pad[1:2:end]
    
    halfomegasquared = 0.5 * omega^2

    a_r = - halfomegasquared * dq_stiff + 4 * dq_soft[2:end].^3
    a_l = halfomegasquared * dq_stiff - 4 * dq_soft[1:end-1].^3
    
    q_ddot = zero(q)
    q_ddot[1:2:end] = a_l
    q_ddot[2:2:end] = a_r
    
    return q_ddot
end


function compute_K(p::AbstractArray{T, 1}) where T<:AbstractFloat
    """Compute kinetic energy"""
    return 0.5 * p' * p
end



function compute_U(q::AbstractArray{T, 1}; omega::T=300.) where T<:AbstractFloat
    """Compute potential energy"""
    
    dq_stiff = q[2:2:end] - q[1:2:end]
    q_pad = vcat([0], q, [0]) 
    dq_soft = q_pad[2:2:end] - q_pad[1:2:end]
    
    U = 0.25 * omega^2 * sum(dq_stiff.^2) + sum(dq_soft.^4) 
    return U
end


function compute_H(p::AbstractArray{T, 1}, q::AbstractArray{T, 1}; omega::T=300.) where T<:AbstractFloat
    """Compute total energy / Hamiltonian"""
    
    return compute_K(p) + compute_U(q; omega=omega)
end


function compute_I(p::AbstractArray{T, 1}, q::AbstractArray{T, 1}; omega::T=300.) where T<:AbstractFloat
    """Compute energy of stiff springs"""
    
    dq_stiff = q[2:2:end] - q[1:2:end]
    dp_stiff = p[2:2:end] - p[1:2:end]
    
    return 0.25 * dp_stiff.^2 + 0.25 * omega^2 * dq_stiff.^2
end


function initial_condition(;omega::T=300.) where T<:AbstractFloat
    """Generate initial condition"""
    
    q0 = zeros(T, 2*m)
    p0 = zeros(T, 2*m)
    q0[1] = (1 - 1/omega)/sqrt(2)
    q0[2] = (1 + 1/omega)/sqrt(2)
    p0[2] = sqrt(2)
        
    return p0, q0
end 



using NL2sol

const Lengthz = 4*m+1;


function construct_z(p::AbstractArray{T, 1}, q::AbstractArray{T, 1}; omega::T=300.) where T<:AbstractFloat
    z = zeros(T, Lengthz)
    z[1:2*m] = p
    
    dq_stiff = q[2:2:end] - q[1:2:end]
    q_pad = vcat([0], q, [0]) 
    dq_soft = q_pad[2:2:end] - q_pad[1:2:end]
    
    z[2*m+1:3*m] = omega / sqrt(2) * dq_stiff
    z[3*m+1:end] = sqrt(2) * dq_soft.^2
    
    return z        
end


function compute_dq(q::AbstractArray{T, 1}; omega::T=300.) where T<:AbstractFloat
    dq = zeros(T, 2*m+1)
    
    dq_stiff = q[2:2:end] - q[1:2:end]
    q_pad = vcat([0], q, [0]) 
    dq_soft = q_pad[2:2:end] - q_pad[1:2:end]
    
    dq[1:m] = omega / sqrt(2) * dq_stiff
    dq[m+1:end] = sqrt(2) * dq_soft.^2
    
    return dq
end 


function Jac_dq(q::AbstractArray{T, 1}; omega::T=300.) where T<:AbstractFloat
    J = zeros(T, (2*m+1, 2*m))
    
    q_pad = vcat([0], q, [0]) 
    dq_soft = q_pad[2:2:end] - q_pad[1:2:end]
    
    for i in 1:m
        J[i, 2*i-1] = - omega / sqrt(2)
        J[i, 2*i] = omega / sqrt(2)
    end 
    
    for i in 2:m
        J[m+i, 2*i-2] = - 2 * sqrt(2) * dq_soft[i]
        J[m+i, 2*i-1] = 2 * sqrt(2) * dq_soft[i]
    end
    
    J[m+1, 1] = 2 * sqrt(2) * dq_soft[1]
    J[end, end] = - 2 * sqrt(2) * dq_soft[end]
    
    return J
end

                
function recover_canonical_vars(z::AbstractArray{T, 1}; omega::T=300.) where T<:AbstractFloat
    
    p = z[1:2*m]
    dq = z[2*m+1:end]
    
    res = nl2sol((q,r)->compute_dq(q; omega=omega).-dq, (q,r)->Jac_dq(q; omega=omega), q0, 2*m+1; quiet=true)
    q = res.minimum 
    
    return p, q
end
