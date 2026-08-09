"""
Nonlinearly Coupled Oscillators
"""

Base.@kwdef struct NonlinearCoupledOscillators <: SeparableHamiltonianSystem
    epsilon::Float64      # coupling constant
end

mass(prob::NonlinearCoupledOscillators) = [1., 1/prob.epsilon]

function initial_condition(prob::NonlinearCoupledOscillators, T::Type)
    v0 = zeros(T, 2)
    x0 = ones(T, 2) * 1.5
    # v0 = T.([0.0, 0.0])
    # v0 = T.([1.0, 0.1122464])
    # x0 = zeros(T, 2)
    # x0 = T.([1.0, 11.45256])
    # x0 = T.([-1.45433, 5.0])
    # x0 = T.([1.5, 9.])

    return v0, x0
end 

function nondimensionalize(prob::NonlinearCoupledOscillators, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v .* mass(prob), x
end

function dimensionalize(prob::NonlinearCoupledOscillators, v_nd::AbstractArray{T, 1}, x_nd::AbstractArray{T, 1}) where T<:AbstractFloat
    return v_nd ./ mass(prob), x_nd
end

function compute_ddx!(prob::NonlinearCoupledOscillators, ddx, dx, x)     
    
    sine = sin(2*x[1] + 2*x[2])
    cosine = cos(2*x[1] + 2*x[2])

    ddx[1] = - x[1] - prob.epsilon * x[2] * (sine + 2*x[1]*cosine)
    ddx[2] = - prob.epsilon^2 * x[2] - prob.epsilon^2 * x[1] * (sine + 2*x[2]*cosine)
    
    nothing 
end

function compute_K(prob::NonlinearCoupledOscillators, v::AbstractArray{T, 1}) where T<:AbstractFloat
    return 0.5 * (v[1]^2 + v[2]^2 / prob.epsilon)
end

function compute_U(prob::NonlinearCoupledOscillators, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return 0.5 * (x[1]^2 + x[2]^2 * prob.epsilon) + prob.epsilon * x[1] * x[2] * sin(2*x[1] + 2*x[2])
end

function vx_to_pq(prob::NonlinearCoupledOscillators, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v .* mass(prob), x
end

function pq_to_vx(prob::NonlinearCoupledOscillators, p::AbstractArray{T, 1}, q::AbstractArray{T, 1}) where T<:AbstractFloat
    return p ./ mass(prob), q
end

# function embed_state_procrustes(prob::NonlinearCoupledOscillators, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}) where T<:AbstractFloat
#     p, q = vx_to_pq(prob, u...)
#     return vcat(p, q)
# end

# function align_state_procrustes(prob::NonlinearCoupledOscillators, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}, corrector::Any) where T<:AbstractFloat
#     znew = corrector(embed_state_procrustes(prob, u))
#     p = znew[1:2]
#     q = znew[3:4]
#     v, x = pq_to_vx(prob, p, q)
#     return (v, x)
# end


function embed_state_procrustes(prob::NonlinearCoupledOscillators, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}) where T<:AbstractFloat
    p, q = vx_to_pq(prob, u...)
    pf = p[1]
    qf = q[1]
    return [pf, qf]
end

function align_state_procrustes(prob::NonlinearCoupledOscillators, u::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}, corrector::Any) where T<:AbstractFloat
    p, q = vx_to_pq(prob, u...)
    znew = corrector(embed_state_procrustes(prob, u))
    pf = znew[1]
    qf = znew[2]
    p[1] = pf
    q[1] = qf
    v, x = pq_to_vx(prob, p, q)
    return (v, x)
end
