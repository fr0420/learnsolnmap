using GenericLinearAlgebra
using LinearAlgebra

"""
    struct Linear{T}(A, b, xc, yc, rank, condition_number, use_bias, Ur, residual)

A linear interpolation model defined by a matrix `A` and a bias vector `b`, along with 
center offsets `xc` and `yc`. The model is determined via SVD, and the effective rank, condition 
number, and range-space basis `Ur` are stored.

Fields:
- `A::AbstractArray{T, 2}`: m x n matrix
- `b::AbstractArray{T, 1}`: m-element vector
- `xc::AbstractArray{T, 1}`: n-element vector, x center offset
- `yc::AbstractArray{T, 1}`: m-element vector, y center offset
- `rank::Integer`: effective rank of the solution
- `condition_number::T`: condition number from the SVD
- `use_bias::Bool`: flag for using bias term
- `Ur::AbstractArray{T, 2}`: basis for the range space ((n+1) x r if use_bias, otherwise n x r)
- `residual::T`: squared residual error
"""
struct Linear{T<:AbstractFloat}
    A::AbstractArray{T, 2}
    b::AbstractArray{T, 1}
    xc::AbstractArray{T, 1}
    yc::AbstractArray{T, 1}
    rank::Integer
    condition_number::T
    use_bias::Bool
    Ur::AbstractArray{T, 2}
    residual::T
end

"""
    linear_interpolation(X, Y; xc=nothing, yc=nothing, use_bias=true, tol=eps(T))

Construct a `Linear` interpolation model from the design matrix `X` and response matrix `Y` with optional
center offsets `xc` and `yc`, bias term `use_bias`, and tolerance `tol` for determining the effective rank.

- `X::AbstractArray{T, 2}`: n x N design matrix.
- `Y::AbstractArray{T, 2}`: m x N response matrix.
- `xc::Union{AbstractArray{T, 1}, Nothing}` (keyword, default `nothing`): n-element vector offset for X.
- `yc::Union{AbstractArray{T, 1}, Nothing}` (keyword, default `nothing`): m-element vector offset for Y.
- `use_bias::Bool` (keyword, default `true`): Include a bias term.
- `tol::Float64` (keyword, default `1e-14`): Tolerance for determining the effective rank.
"""
function linear_interpolation(
    X::AbstractArray{T, 2}, 
    Y::AbstractArray{T, 2};
    xc::Union{AbstractArray{T, 1}, Nothing}=nothing,
    yc::Union{AbstractArray{T, 1}, Nothing}=nothing, 
    use_bias::Bool=true, 
    tol::Float64=1e-14
) where T<:AbstractFloat

    # Ensure dimensions match
    n, N = size(X)
    m, N_Y = size(Y)
    @assert N == N_Y "X and Y must have the same number of columns"

    # Set default offsets
    xc = isnothing(xc) ? zeros(T, n) : xc
    yc = isnothing(yc) ? zeros(T, m) : yc
    @assert length(xc) == n "Length of xc must match number of rows in X"
    @assert length(yc) == m "Length of yc must match number of rows in Y"

    # Center the data
    X_centered = X .- xc
    Y_centered = Y .- yc 

    # Append bias row if required
    X_design = use_bias ? vcat(X_centered, ones(T, 1, N)) : X_centered  # (n+1) x N if use_bias otherwise n x N

    # Perform SVD on the design matrix
    svd_sol = GenericLinearAlgebra.svd(X_design)

    # Determine effective rank using the tolerance
    r = count(s -> s / svd_sol.S[1] > tol, svd_sol.S)
    r = max(r, 1)  # ensure at least rank 1
    condition_num = svd_sol.S[1] / svd_sol.S[r]

    # Compute least-squares solution using the SVD components
    L = Y_centered * svd_sol.V[:, 1:r] * Diagonal(1 ./ svd_sol.S[1:r]) * svd_sol.U[:, 1:r]'  # m x (n+1) if use_bias otherwise m x n
    Ur = svd_sol.U[:, 1:r]
    residual = norm(L * X_design - Y_centered)^2

    # Separate the linear operator into A and b components
    A = use_bias ? L[:, 1:end-1] : L        # m x n
    b = use_bias ? L[:, end] : zeros(T, m)  # m x 1

    return Linear(A, b, xc, yc, r, condition_num, use_bias, Ur, residual)
end

# Define the callable behavior for the Linear instance
(linear::Linear)(x::AbstractArray{T, 1}) where T<:AbstractFloat = 
    linear.yc .+ linear.A * (x .- linear.xc) .+ linear.b

"""
    range_space_projection_ratio(x, linear)

Compute the ratio of the squared norm of the projection of the (offset) input `x` onto the range space 
of the interpolation model to the squared norm of the (offset) input. This ratio can be interpreted 
as a measure of how well the input aligns with the range space.

For bias usage, the vector `z` is formed by appending 1 to (x - xc).
"""
function range_space_projection_ratio(x::AbstractArray{T, 1}, linear::Linear{T}) where T<:AbstractFloat
    z = linear.use_bias ? vcat(x .- linear.xc, one(T)) : (x .- linear.xc)  # (n+1) x 1 if use_bias otherwise n x 1
    z_proj = linear.Ur' * z  # r x 1
    return dot(z_proj, z_proj) / dot(z, z)
end

# Main function for testing
function main()
    println("Linear Interpolation Example")
    
    # Create a simple dataset
    n, m, N = 2, 1, 5
    X = collect(range(-1, 1, length=N))'   # 1 x N
    X = vcat(X, X.^2)                      # 2 x N (x and x^2)
    Y = 3 .* X[1:1, :] .- 2 .* X[2:2, :] .+ 1  # y = 3x - 2x^2 + 1
    
    # Create the model
    model = linear_interpolation(X, Y)
    
    # Display model parameters
    println("Model parameters:")
    println("A = ", model.A)
    println("b = ", model.b)
    println("rank = ", model.rank)
    println("condition number = ", model.condition_number)
    
    # Test predictions
    println("\nPredictions at training points:")
    for i in 1:N
        x = X[:, i]
        y_true = Y[:, i]
        y_pred = model(x)
        println("x = $x, y_true = $y_true, y_pred = $y_pred")
    end
    
    # Test interpolation
    println("\nInterpolation at new points:")
    x_new = [0.5, 0.5^2]
    y_expected = 3 * 0.5 - 2 * 0.5^2 + 1
    y_pred = model(x_new)
    println("x = $x_new, y_expected = $y_expected, y_pred = $y_pred")
end

if ARGS == ["--run"]
    main()
end
