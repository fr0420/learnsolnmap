"""
Fermi-Pasta-Ulam Problem 
"""

using StaticArrays
using MultiFloats
using NL2sol
# using NonlinearSolve

const m = 3;


function A!(ddu, du, u, p, t) 
    """Compute ddu, second order time derivative of u, in place"""
    
    halfomegasquared = p
    
#     u_odd = @view u[1:2:end]
#     u_even = @view u[2:2:end]
#     ddu_odd = @view ddu[1:2:end]
#     ddu_even = @view ddu[2:2:end]
    
#     @. ddu_odd = halfomegasquared * (u_even .- u_odd)
#     @. ddu_even = - ddu_odd
    
#     u_odd = @view u[3:2:end]
#     u_even = @view u[2:2:end-1]
#     ddu_odd = @view ddu[3:2:end]
#     ddu_even = @view ddu[2:2:end-1]
    
#     ddu[1] -= 4 * u[1].^3 
#     @. ddu_odd -= 4 * (u_odd .- u_even).^3 
#     @. ddu_even += 4 * (u_odd .- u_even).^3 
#     ddu[end] += 4 * (-u[end]).^3 
    
    ddu[1] = halfomegasquared * (u[2] - u[1]) - 4 * u[1].^3
    ddu[2] = - halfomegasquared * (u[2] - u[1]) + 4 * (u[3] - u[2]).^3
    ddu[3] = halfomegasquared * (u[4] - u[3]) - 4 * (u[3] - u[2]).^3 
    ddu[4] = - halfomegasquared * (u[4] - u[3]) + 4 * (u[5] - u[4]).^3
    ddu[5] = halfomegasquared * (u[6] - u[5]) - 4 * (u[5] - u[4]).^3 
    ddu[6] = - halfomegasquared * (u[6] - u[5]) + 4 * (- u[6]).^3
    
    nothing 
end


function A_static(du, u, p, t) 
    """Compute ddu, second order time derivative of u, using StaticArrays"""
    
    halfomegasquared = p
    
    ddu1 = halfomegasquared * (u[2] - u[1]) - 4 * u[1].^3
    ddu2 = - halfomegasquared * (u[2] - u[1]) + 4 * (u[3] - u[2]).^3
    ddu3 = halfomegasquared * (u[4] - u[3]) - 4 * (u[3] - u[2]).^3 
    ddu4 = - halfomegasquared * (u[4] - u[3]) + 4 * (u[5] - u[4]).^3
    ddu5 = halfomegasquared * (u[6] - u[5]) - 4 * (u[5] - u[4]).^3 
    ddu6 = - halfomegasquared * (u[6] - u[5]) + 4 * (- u[6]).^3
    
    SA[ddu1, ddu2, ddu3, ddu4, ddu5, ddu6]
end


# function A(q::AbstractArray{T, 1}; omega::T=300.) where T<:AbstractFloat
#     """Compute second order time derivative of q"""
    
#     dq_stiff = q[2:2:end] - q[1:2:end]
#     q_pad = vcat([0], q, [0]) 
#     dq_soft = q_pad[2:2:end] - q_pad[1:2:end]
    
#     halfomegasquared = 0.5 * omega^2

#     a_r = - halfomegasquared * dq_stiff + 4 * dq_soft[2:end].^3
#     a_l = halfomegasquared * dq_stiff - 4 * dq_soft[1:end-1].^3
    
#     q_ddot = zero(q)
#     q_ddot[1:2:end] = a_l
#     q_ddot[2:2:end] = a_r
    
#     return q_ddot
# end


function compute_K(p::AbstractArray{T, 1}) where T<:AbstractFloat
    """Compute kinetic energy"""
    return 0.5 * p' * p
end



function compute_U(q::AbstractArray{T, 1}; omega::Float64=300.) where T<:AbstractFloat
    """Compute potential energy"""
    omega = convert(T, omega)

    dq_stiff = q[2:2:end] - q[1:2:end]
    q_pad = vcat([0], q, [0]) 
    dq_soft = q_pad[2:2:end] - q_pad[1:2:end]
    
    U = 0.25 * omega^2 * sum(dq_stiff.^2) + sum(dq_soft.^4) 
    return U
end


function compute_H(p::AbstractArray{T, 1}, q::AbstractArray{T, 1}; omega::Float64=300.) where T<:AbstractFloat
    """Compute total energy / Hamiltonian"""
    
    return compute_K(p) + compute_U(q; omega=omega)
end


function compute_I(p::AbstractArray{T, 1}, q::AbstractArray{T, 1}; omega::Float64=300.) where T<:AbstractFloat
    """Compute energy of stiff springs"""
    omega = convert(T, omega)

    dq_stiff = q[2:2:end] - q[1:2:end]
    dp_stiff = p[2:2:end] - p[1:2:end]
    
    return 0.25 * dp_stiff.^2 + 0.25 * omega^2 * dq_stiff.^2
end


function initial_condition(T; omega::Float64=300.)
    """Generate initial condition"""
    omega = convert(T, omega)
    sqrt2 = sqrt(convert(T, 2.0))

    q0 = zeros(T, 2*m)
    p0 = zeros(T, 2*m)
    q0[1] = (1 - 1/omega)/sqrt2
    q0[2] = (1 + 1/omega)/sqrt2
    p0[2] = sqrt2
        
    return p0, q0
end 


function compute_dq(q::AbstractArray{T, 1}; omega::Float64=300.) where T<:AbstractFloat
    omega = convert(T, omega)
    sqrt2 = sqrt(convert(T, 2.0))

    dq = zeros(T, 2*m+1)
    
    dq_stiff = q[2:2:end] - q[1:2:end]
    q_pad = vcat([0], q, [0]) 
    dq_soft = q_pad[2:2:end] - q_pad[1:2:end]
    
    dq[1:m] = omega / sqrt2 * dq_stiff
    dq[m+1:end] = sqrt2 * dq_soft.^2
    
    return dq
end 


function Jac_dq(q::AbstractArray{T, 1}; omega::Float64=300.) where T<:AbstractFloat
    omega = convert(T, omega)
    sqrt2 = sqrt(convert(T, 2.0))

    J = zeros(T, (2*m+1, 2*m))
    
    q_pad = vcat([0], q, [0]) 
    dq_soft = q_pad[2:2:end] - q_pad[1:2:end]
    
    for i in 1:m
        J[i, 2*i-1] = - omega / sqrt2
        J[i, 2*i] = omega / sqrt2
    end 
    
    for i in 2:m
        J[m+i, 2*i-2] = - 2 * sqrt2 * dq_soft[i]
        J[m+i, 2*i-1] = 2 * sqrt2 * dq_soft[i]
    end
    
    J[m+1, 1] = 2 * sqrt2 * dq_soft[1]
    J[end, end] = - 2 * sqrt2 * dq_soft[end]
    
    return J
end


function construct_z(p::AbstractArray{T, 1}, q::AbstractArray{T, 1}; omega::Float64=300.) where T<:AbstractFloat

    z = zeros(T, 4*m+1)
    z[1:2*m] = p
    z[2*m+1:end] = compute_dq(q; omega=omega)    
    
    return z        
end


function recover_canonical_vars(z::AbstractArray{T, 1}, q_guess::AbstractArray{T, 1}; omega::Float64=300.) where T<:AbstractFloat
    
    p = z[1:2*m]
    dq = z[2*m+1:end]
    
    if T <: MultiFloat
        dq = convert.(Float64, dq)
        q_guess = convert.(Float64, q_guess)
    end 
    
    function residual(q, r)
        r[:] = compute_dq(q; omega=omega) .- dq
        return r
    end

    function jacobian(q, jac)
        jac[:, :] = Jac_dq(q; omega=omega)
        return jac
    end

    res = nl2sol(residual, jacobian, q_guess, 2*m+1; quiet=true)
    q = res.minimum

    # f(u, p) = compute_dq(u; omega=omega) - dq
    # jac(u, p) = Jac_dq(u; omega=omega)
    # prob = NonlinearProblem(NonlinearFunction(f; jac=jac), q_guess, nothing)
    # sol = solve(prob, NewtonRaphson())
    # q = sol.u
    # println("residual:", sol.resid)

    if T <: MultiFloat
        q = convert.(T, q)
    end

    return p, q
end
