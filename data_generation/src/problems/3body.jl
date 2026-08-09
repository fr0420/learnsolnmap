"""
Three-body problem in 3D
"""

Base.@kwdef struct ThreeBody <: SeparableHamiltonianSystem
    G::Float64 = 1.0
    m1::Float64 = 100.0
    m2::Float64 = 1.0
    m3::Float64 = 0.001
end

mass(prob::ThreeBody) = [
    prob.m1, prob.m1, prob.m1, prob.m2, prob.m2, prob.m2, prob.m3, prob.m3, prob.m3
]

function initial_condition(prob::ThreeBody, T::Type)
    x0 = zeros(T, 9)
    v0 = zeros(T, 9)

    x0[1:3] =   [-1.00102,      0.,     0.    ]
    x0[4:6] =   [100.,    0.,     0.    ]
    x0[7:9] =   [102.,    0.,     0.    ]

    v0[1:3] =   [0.,     -0.010001,     -0.000001   ]
    v0[4:6] =   [0.,     1.,     0.   ]
    v0[7:9] =   [0.,     0.1,    0.1   ]

    return v0, x0
end 

function nondimensionalize(prob::ThreeBody, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v, x
end

function dimensionalize(prob::ThreeBody, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v, x
end

function compute_ddx!(prob::ThreeBody, ddx, dx, x)   
    x1 = @view x[1:3]
    x2 = @view x[4:6]
    x3 = @view x[7:9]

    r12_cubed = norm(x1 .- x2)^3
    r13_cubed = norm(x1 .- x3)^3
    r23_cubed = norm(x2 .- x3)^3

    ddx[1:3] = prob.m2 * (x1 .- x2) ./ r12_cubed + prob.m3 * (x1 .- x3) ./ r13_cubed
    ddx[4:6] = prob.m1 * (x2 .- x1) ./ r12_cubed + prob.m3 * (x2 .- x3) ./ r23_cubed
    ddx[7:9] = prob.m1 * (x3 .- x1) ./ r13_cubed + prob.m2 * (x3 .- x2) ./ r23_cubed

    ddx[:] *= -prob.G

    nothing 
end

function compute_ddx(prob::ThreeBody, x::AbstractArray{T, 1}) where T<:AbstractFloat    
    x1 = @view x[1:3]
    x2 = @view x[4:6]
    x3 = @view x[7:9]
    
    r12_cubed = norm(x1 .- x2)^3
    r13_cubed = norm(x1 .- x3)^3
    r23_cubed = norm(x2 .- x3)^3

    ddx = zero(x)
    ddx[1:3] = prob.m2 * (x1 .- x2) ./ r12_cubed + prob.m3 * (x1 .- x3) ./ r13_cubed
    ddx[4:6] = prob.m1 * (x2 .- x1) ./ r12_cubed + prob.m3 * (x2 .- x3) ./ r23_cubed
    ddx[7:9] = prob.m1 * (x3 .- x1) ./ r13_cubed + prob.m2 * (x3 .- x2) ./ r23_cubed

    ddx[:] *= -prob.G

    return ddx
end

function compute_K(prob::ThreeBody, v::AbstractArray{T, 1}) where T<:AbstractFloat
    v1 = @view v[1:3]
    v2 = @view v[4:6]
    v3 = @view v[7:9]
    return 0.5 * (prob.m1 * v1' * v1 + prob.m2 * v2' * v2 + prob.m3 * v3' * v3)
end

function compute_U(prob::ThreeBody, x::AbstractArray{T, 1}) where T<:AbstractFloat
    x1 = @view x[1:3]
    x2 = @view x[4:6]
    x3 = @view x[7:9]
    return - (prob.m1 * prob.m2 / norm(x1-x2) + prob.m1 * prob.m3 / norm(x1-x3) + prob.m2 * prob.m3 / norm(x2-x3)) * prob.G
end
