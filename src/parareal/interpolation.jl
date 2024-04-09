using GenericLinearAlgebra

"""Linear Interpolation"""

struct Linear{T<:AbstractFloat}
    A::AbstractArray{T, 2}  # m x n
    b::AbstractArray{T, 1}  # m x 1
    xc::AbstractArray{T, 1}  # n x 1
    yc::AbstractArray{T, 1}  # m x 1
    rank::Integer
    condition_number::T
    use_bias::Bool
    Ur::AbstractArray{T, 2}  # (n+1) x r if use_bias otherwise n x r
    residual::T
end

function Linear(
    X::AbstractArray{T, 2},  # n x N
    Y::AbstractArray{T, 2},  # m x N
    use_bias::Bool,
    tol::T
) where T<:AbstractFloat

    xc = zeros(T, size(X, 1))
    yc = zeros(T, size(Y, 1))

    return Linear(X, Y, xc, yc, use_bias, tol)
end

function Linear(
    X::AbstractArray{T, 2},  # n x N
    Y::AbstractArray{T, 2},  # m x N
    xc::AbstractArray{T, 1},  # n x 1
    yc::AbstractArray{T, 1},  # m x 1
    use_bias::Bool,
    tol::T
) where T<:AbstractFloat

    X = X .- xc
    Y = Y .- yc 
    
    # if iszero(X)
    #     throw(DomainError(X, "X must not be null matrix"))
    # end

    X = use_bias ? vcat(X, ones(T, size(X, 2))') : X  # (n+1) x N if use_bias otherwise n x N
    sol = GenericLinearAlgebra.svd(X)
    r = sum(sol.S/sol.S[1] .> tol)
    cn = sol.S[1]/sol.S[r]
    L = Y * sol.V[:, 1:r] * Diagonal(1 ./ sol.S[1:r]) * sol.U[:, 1:r]'  # m x (n+1) if use_bias otherwise m x n
    Ur = sol.U[:, 1:r]  # (n+1) x r if use_bias otherwise n x r
    residual = norm(L * X - Y)^2

    A = use_bias ? L[:, 1:end-1] : L  # m x n
    b = use_bias ? L[:, end] : zero(yc)  # m x 1

    return Linear(A, b, xc, yc, r, cn, use_bias, Ur, residual)
end

(linear::Linear)(x) = linear.yc + linear.A * (x-linear.xc) + linear.b

function range_space_projection_ratio(x, linear)

    z = linear.use_bias ? [x-linear.xc; 1] : x-linear.xc
    z_proj = linear.Ur' * z

    return (z_proj' * z_proj) / (z' * z)
end