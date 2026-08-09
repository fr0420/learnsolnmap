"""
Fermi-Pasta-Ulam Problem 
"""

using StaticArrays

Base.@kwdef struct FPU <: SeparableHamiltonianSystem
    omega::Float64      # frequency of stiff springs  
    m::Int64 = 3        # number of stiff springs
    c0::Float64 = 0.5 * omega^2
end

mass(prob::FPU) = ones(2*prob.m) 

function initial_condition(prob::FPU, T::Type)
    omega = convert(T, prob.omega)
    sqrt2 = sqrt(convert(T, 2.0))

    q0 = zeros(T, 2*prob.m)
    p0 = zeros(T, 2*prob.m)
    q0[1] = (1 - 1/omega)/sqrt2
    q0[2] = (1 + 1/omega)/sqrt2
    p0[2] = sqrt2
    # p0[2] = 1.0

    return p0, q0
end 

function nondimensionalize(prob::FPU, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v, x
end

function dimensionalize(prob::FPU, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v, x
end

function compute_ddx!(prob::FPU, ddx, dx, x)     
    
    ddx[1] = prob.c0 * (x[2] - x[1]) - 4 * x[1].^3
    ddx[2] = - prob.c0 * (x[2] - x[1]) + 4 * (x[3] - x[2]).^3
    ddx[3] = prob.c0 * (x[4] - x[3]) - 4 * (x[3] - x[2]).^3 
    ddx[4] = - prob.c0 * (x[4] - x[3]) + 4 * (x[5] - x[4]).^3
    ddx[5] = prob.c0 * (x[6] - x[5]) - 4 * (x[5] - x[4]).^3 
    ddx[6] = - prob.c0 * (x[6] - x[5]) + 4 * (- x[6]).^3
    
    nothing 
end

function compute_ddx_static(prob::FPU, dx, x, params, t) 
    
    ddx1 = prob.c0 * (x[2] - x[1]) - 4 * x[1].^3
    ddx2 = - prob.c0 * (x[2] - x[1]) + 4 * (x[3] - x[2]).^3
    ddx3 = prob.c0 * (x[4] - x[3]) - 4 * (x[3] - x[2]).^3 
    ddx4 = - prob.c0 * (x[4] - x[3]) + 4 * (x[5] - x[4]).^3
    ddx5 = prob.c0 * (x[6] - x[5]) - 4 * (x[5] - x[4]).^3 
    ddx6 = - prob.c0 * (x[6] - x[5]) + 4 * (- x[6]).^3
    
    SA[ddx1, ddx2, ddx3, ddx4, ddx5, ddx6]
end

function compute_ddx(prob::FPU, x::AbstractArray{T, 1}) where T<:AbstractFloat
    
    dx_stiff = x[2:2:end] - x[1:2:end]
    x_pad = vcat([0], x, [0]) 
    dx_soft = x_pad[2:2:end] - x_pad[1:2:end]
    
    a_r = - prob.c0 * dx_stiff + 4 * dx_soft[2:end].^3
    a_l = prob.c0 * dx_stiff - 4 * dx_soft[1:end-1].^3
    
    ddx = zero(x)
    ddx[1:2:end] = a_l
    ddx[2:2:end] = a_r
    
    return ddx
end

function compute_K(prob::FPU, v::AbstractArray{T, 1}) where T<:AbstractFloat
    return 0.5 * v' * v
end

function compute_U(prob::FPU, x::AbstractArray{T, 1}) where T<:AbstractFloat
    omega = convert(T, prob.omega)

    dx_stiff = x[2:2:end] - x[1:2:end]
    x_pad = vcat([0], x, [0]) 
    dx_soft = x_pad[2:2:end] - x_pad[1:2:end]
    
    U = 0.25 * omega^2 * sum(dx_stiff.^2) + sum(dx_soft.^4) 
    return U
end

"""Compute energy of stiff springs."""
function compute_I(prob::FPU, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    omega = convert(T, prob.omega)

    dx_stiff = x[2:2:end] - x[1:2:end]
    dv_stiff = v[2:2:end] - v[1:2:end]
    
    return 0.25 * dv_stiff.^2 + 0.25 * omega^2 * dx_stiff.^2
end

function to_slow_fast_variables(prob::FPU, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    xs = (x[2:2:end] + x[1:2:end]) / sqrt(T(2.))
    xf = (x[2:2:end] - x[1:2:end]) / sqrt(T(2.))
    vs = (v[2:2:end] + v[1:2:end]) / sqrt(T(2.))
    vf = (v[2:2:end] - v[1:2:end]) / sqrt(T(2.))
    return vs, vf, xs, xf
end

function to_original_variables(prob::FPU, vs::AbstractArray{T, 1}, vf::AbstractArray{T, 1}, 
    xs::AbstractArray{T, 1}, xf::AbstractArray{T, 1}) where T<:AbstractFloat
    x = zeros(T, 2*prob.m)
    v = zeros(T, 2*prob.m)
    x[1:2:end] = (xs - xf) / sqrt(T(2.))
    x[2:2:end] = (xs + xf) / sqrt(T(2.))
    v[1:2:end] = (vs - vf) / sqrt(T(2.))
    v[2:2:end] = (vs + vf) / sqrt(T(2.))
    return v, x
end


function embed_state_interpolative(prob::FPU, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}) where T<:AbstractFloat
    return vcat(u...)
end

function align_state_interpolative(prob::FPU, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}, corrector::Any) where T<:AbstractFloat
    znew = corrector(embed_state_interpolative(prob, u))
    v = znew[1:2*prob.m]
    x = znew[2*prob.m+1:end]
    return (v, x)
end

# function embed_state_interpolative(prob::FPU, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}) where T<:AbstractFloat
#     vs, vf, xs, xf = to_slow_fast_variables(prob, u...)
#     scaled_xf = xf * prob.omega
#     return vcat(vs, xs, vf, scaled_xf)
# end

# function align_state_interpolative(prob::FPU, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}, corrector::Any) where T<:AbstractFloat
#     znew = corrector(embed_state_interpolative(prob, u))
#     vs = znew[1:prob.m]
#     vf = znew[prob.m+1:2*prob.m]
#     xs = znew[2*prob.m+1:3*prob.m]
#     scaled_xf = znew[3*prob.m+1:end]
#     xf = scaled_xf / prob.omega
#     v, x = to_original_variables(prob, vs, vf, xs, xf)
#     return (v, x)
# end


# function embed_state_procrustes(prob::FPU, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}) where T<:AbstractFloat
#     vs, vf, xs, xf = to_slow_fast_variables(prob, u...)
#     scaled_xf = xf * prob.omega
#     return (vcat(vs, xs), vcat(vf, scaled_xf))
# end

# function align_state_procrustes(prob::FPU, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}, corrector::Any) where T<:AbstractFloat
#     znew = corrector(embed_state_procrustes(prob, u))
#     vs = znew[1][1:prob.m]
#     xs = znew[1][prob.m+1:end] 
#     vf = znew[2][1:prob.m]
#     scaled_xf = znew[2][prob.m+1:end]
#     xf = scaled_xf / prob.omega
#     v, x = to_original_variables(prob, vs, vf, xs, xf)
#     return (v, x)
# end


function embed_state_procrustes(prob::FPU, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}) where T<:AbstractFloat
    vs, vf, xs, xf = to_slow_fast_variables(prob, u...)
    scaled_xf = xf * prob.omega
    return vcat(vf, scaled_xf)
end

function align_state_procrustes(prob::FPU, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}, corrector::Any) where T<:AbstractFloat
    vs, _, xs, _ = to_slow_fast_variables(prob, u...)
    znew = corrector(embed_state_procrustes(prob, u))
    vf = znew[1:prob.m]
    scaled_xf = znew[prob.m+1:2*prob.m]
    xf = scaled_xf / prob.omega
    v, x = to_original_variables(prob, vs, vf, xs, xf)
    return (v, x)
end


# function embed_state(prob::FPU, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}) where T<:AbstractFloat
#     vs, vf, xs, xf = to_slow_fast_variables(prob, u...)
#     scaled_xf = xf * prob.omega
#     return vcat(vs, vf, xs, scaled_xf)
# end

# function align_state(prob::FPU, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}, corrector::Any) where T<:AbstractFloat
#     znew = corrector(embed_state(prob, u))
#     vs = znew[1:prob.m]
#     vf = znew[prob.m+1:2*prob.m]
#     xs = znew[2*prob.m+1:3*prob.m]
#     scaled_xf = znew[3*prob.m+1:end]
#     xf = scaled_xf / prob.omega
#     v, x = to_original_variables(prob, vs, vf, xs, xf)
#     return (v, x)
# end


# function embed_state(prob::FPU, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}) where T<:AbstractFloat
#     construct_z(prob, u...)
# end

# function align_state(prob::FPU, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}, corrector::Any) where T<:AbstractFloat
#     znew = corrector(embed_state(prob, u))
#     recover_v_x(prob, znew, u[2])
# end
