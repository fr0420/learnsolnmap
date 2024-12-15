"""
Three-body problem in 2D
"""

Base.@kwdef struct ThreeBody2D <: SeparableHamiltonianSystem
    G::Float64 = 1.0
    m1::Float64 = 100.0
    m2::Float64 = 1.0
    m3::Float64 = 0.001
end

mass(prob::ThreeBody2D) = [
    prob.m1, prob.m1, prob.m2, prob.m2, prob.m3, prob.m3
]

function initial_condition(prob::ThreeBody2D, T::Type)
    x0 = zeros(T, 6)
    v0 = zeros(T, 6)

    # x0[1:2] =   [-1.00102,      0.,     ]
    # x0[3:4] =   [100.,    0.,     ]
    # x0[5:6] =   [102.,    0.,     ]

    # v0[1:2] =   [0.,     -0.010001,]
    # v0[3:4] =   [0.,     1., ]
    # v0[5:6] =   [0.,     0.1,  ]

    # x0[1:2] =   [-1.00102,      0.,     ]
    # x0[3:4] =   [100.,    0.,     ]
    # x0[5:6] =   [102.,    0.,     ]

    # v0[1:2] =   [0.,     -0.010015,]
    # v0[3:4] =   [0.,     1., ]
    # v0[5:6] =   [0.,     1.5,  ]
    
    # # equal mass system 
    # x0[1:2] =   [1.,      0.,     ]
    # x0[3:4] =   [-0.5,    sqrt(3)/2,     ]
    # x0[5:6] =   [-0.5,    -sqrt(3)/2,     ]

    # v0[1:2] =   [0.0001,     1,]
    # v0[3:4] =   [-sqrt(3)/2,    -0.5, ]
    # v0[5:6] =   [sqrt(3)/2,     -0.5,  ]
    # v0[:] *= 3^(-0.25)

    # v0[1:2] *= 1.1
    # v0[3:4] *= 0.9
    # v0[5:6] *= 1.15

    # equal mass, non-equilateral configuration
    v0[1:6] = [0.07549985 -0.81340487  0.1986117   0.27531269 -0.15989042  0.35567132]  
    x0[1:6] = [-1.38436497 -0.62820671  0.4143637  -0.82314002  1.57423108  0.48634039]

    return v0, x0
end 

function nondimensionalize(prob::ThreeBody2D, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v, x
end

function dimensionalize(prob::ThreeBody2D, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v, x
end

function compute_ddx!(prob::ThreeBody2D, ddx, dx, x)   
    x1 = @view x[1:2]
    x2 = @view x[3:4]
    x3 = @view x[5:6]

    r12_cubed = norm(x1 .- x2)^3
    r13_cubed = norm(x1 .- x3)^3
    r23_cubed = norm(x2 .- x3)^3

    ddx[1:2] = prob.m2 * (x1 .- x2) ./ r12_cubed + prob.m3 * (x1 .- x3) ./ r13_cubed
    ddx[3:4] = prob.m1 * (x2 .- x1) ./ r12_cubed + prob.m3 * (x2 .- x3) ./ r23_cubed
    ddx[5:6] = prob.m1 * (x3 .- x1) ./ r13_cubed + prob.m2 * (x3 .- x2) ./ r23_cubed

    ddx[:] *= -prob.G

    nothing 
end

function compute_ddx(prob::ThreeBody2D, x::AbstractArray{T, 1}) where T<:AbstractFloat    
    x1 = @view x[1:2]
    x2 = @view x[3:4]
    x3 = @view x[5:6]
    
    r12_cubed = norm(x1 .- x2)^3
    r13_cubed = norm(x1 .- x3)^3
    r23_cubed = norm(x2 .- x3)^3

    ddx = zero(x)
    ddx[1:2] = prob.m2 * (x1 .- x2) ./ r12_cubed + prob.m3 * (x1 .- x3) ./ r13_cubed
    ddx[3:4] = prob.m1 * (x2 .- x1) ./ r12_cubed + prob.m3 * (x2 .- x3) ./ r23_cubed
    ddx[5:6] = prob.m1 * (x3 .- x1) ./ r13_cubed + prob.m2 * (x3 .- x2) ./ r23_cubed

    ddx[:] *= -prob.G

    return ddx
end

function compute_K(prob::ThreeBody2D, v::AbstractArray{T, 1}) where T<:AbstractFloat
    v1 = @view v[1:2]
    v2 = @view v[3:4]
    v3 = @view v[5:6]
    return 0.5 * (prob.m1 * v1' * v1 + prob.m2 * v2' * v2 + prob.m3 * v3' * v3)
end

function compute_U(prob::ThreeBody2D, x::AbstractArray{T, 1}) where T<:AbstractFloat
    x1 = @view x[1:2]
    x2 = @view x[3:4]
    x3 = @view x[5:6]
    return - (prob.m1 * prob.m2 / norm(x1-x2) + prob.m1 * prob.m3 / norm(x1-x3) + prob.m2 * prob.m3 / norm(x2-x3)) * prob.G
end
