"""
Sampling functions.
"""

using Distributions
using LinearAlgebra
using Random


"""Sample random points in a high-dimensional box."""
function BoxSampling(n_samples::Int, bounds::Vector{Tuple{T, T}}) where T<:AbstractFloat

    # get dimension from bounds
    n = length(bounds)

    # generate random points for each dimension
    X = zeros(T, n, n_samples)
    for i in 1:n
        X[i, :] = rand(T, n_samples) * (bounds[i][2] - bounds[i][1]) .+ bounds[i][1]
    end

    return X
end

"""Sample random points in a high-dimensional shell."""
function ShellSampling(n_samples::Int, radius_bounds::Vector{Tuple{T, T}}) where T<:AbstractFloat

    # get dimension from bounds
    n = length(radius_bounds)

    # generate random directions
    X = randn(T, n, n_samples)
    X ./= sqrt.(sum(abs2, X, dims=1))

    # generate random radii
    r = BoxSampling(n_samples, radius_bounds)

    # scale the directions by the radii
    X .*= r

    return X
end

"""Sample a random point on a unit n-sphere."""
function nSphereSampling(n::Int)

    # generate n Gaussian random variables 
    x = randn(n)
    
    # normalize 
    x /= sqrt(sum(abs2, x)) 
    
    return x
end

"""Generate uniformly distributed random points on a unit n-sphere."""
function nSphereSampling(n::Int, n_samples::Int)
    
    # generate n Gaussian random variables 
    X = randn(n, n_samples)
    
    # normalize 
    X = X ./ sqrt.(sum(abs2, X, dims=1)) 
    
    return X
end

"""Generate uniformly distributed random points in a unit n-ball."""
function nBallSampling(n::Int, n_samples::Int)
    
    # generate n Gaussian random variables 
    X = randn(n, n_samples)
    
    # generate random radius 
    r = rand(1, n_samples) .^ (1/n)
    
    # normalize the vector and multiply by radius 
    X = X ./ sqrt.(sum(abs2, X, dims=1)) 
    X = X .* r
    
    return X
end

"""Generate uniformly distributed random points in a unit n-cube."""
function nCubeSampling(n::Int, n_samples::Int)
    
    X = rand(n, n_samples) .- 0.5
    
    return X
end

"""Generate uniformly distributed random points in a n-dimensional spherical shell (1 < r < 1+epsilon)."""
function nShellSampling(n::Int, n_samples::Int, epsilon::T) where T<:AbstractFloat
    
    # generate n Gaussian random variables 
    X = randn(n, n_samples)
    
    # generate random radius between [1, 1+epsilon]
    u = rand(1, n_samples)
    r = (u * (1. + epsilon)^n + (1 .- u)) .^ (1/n)
    
    # normalize the vector and multiply by radius 
    X = X ./ sqrt.(sum(abs2, X, dims=1)) 
    X = X .* r
    
    return X
end

"""Generate random variables from a normal distribution."""
function normal(mean::T, std::T, n_samples::Int=1) where T<:AbstractFloat
    samples = randn(n_samples) * std .+ mean
    return n_samples == 1 ? samples[1] : samples
end

"""Generate random variables from a truncated normal distribution."""
function truncated_normal(mean::T, std::T, lower::T, upper::T, n_samples::Int=1) where T<:AbstractFloat
    
    if std < 0
        throw(ArgumentError("Standard deviation must be non-negative."))
    elseif std == 0
        if lower <= mean <= upper
            return n_samples == 1 ? mean : fill(mean, n_samples)
        else
            throw(ArgumentError("Mean must lie within the specified bounds when std is 0."))
        end
    elseif lower >= upper
        throw(ArgumentError("Lower bound must be less than upper bound."))
    end

    dist = Truncated(Normal(Float64(mean), Float64(std)), Float64(lower), Float64(upper))  
    samples = rand(dist, n_samples)
    samples = T.(samples) 
    return n_samples == 1 ? samples[1] : samples
end

"""Generate random variables from a Bernoulli distribution."""
function bernoulli(p::T, n_samples::Int=1) where T<:AbstractFloat
    dist = Binomial(1, Float64(p))
    samples = rand(dist, n_samples)
    return n_samples == 1 ? samples[1] : samples
end

"""Generate random variables from a geometric distribution."""
function geometric(p::T, n_samples::Int=1) where T<:AbstractFloat
    dist = Geometric(Float64(p))
    samples = rand(dist, n_samples)
    return n_samples == 1 ? samples[1] : samples
end

"""
Sample random points on the surface of an n-dimensional ellipsoid defined by the quadratic 
equation x^T M x = c.

Arguments:
    M: ellipsoid matrix (n x n)
    c: ellipsoid constant
    n_samples: number of samples
"""
function ellipsoidSampling(M::AbstractArray{T, 2}, c::T, n_samples::Int=1) where T<:AbstractFloat
    
    # check input validity
    if !issymmetric(M)
        throw(ArgumentError("Matrix M must be symmetric"))
    end
    if c <= 0
        throw(ArgumentError("Constant c must be positive"))
    end
    
    # get dimension from matrix size
    n = size(M, 1)
    if size(M, 2) != n
        throw(ArgumentError("Matrix M must be square"))
    end
    
    # compute eigendecomposition of M
    F = eigen(M)
    if any(F.values .<= 0)
        throw(ArgumentError("Matrix M must be positive definite"))
    end
    
    # generate random points on the unit sphere
    X = nSphereSampling(n, n_samples)
    
    # scale and rotate points to satisfy the ellipsoid equation
    scaling = sqrt(c) ./ sqrt.(F.values)
    scaled_X = scaling .* X
    X = F.vectors * scaled_X

    return n_samples == 1 ? X[:, 1] : X
end

"""
Sample random points on the intersection of an ellipsoid surface x^T M x = c and 
a set of linear constraints Ax = b.

Arguments:
    M: ellipsoid matrix (n x n)
    c: ellipsoid constant
    A: constraint matrix (m x n)
    b: constraint vector (m)
    n_samples: number of samples
"""
function ellipsoidLinearConstraintsSampling(
    M::AbstractArray{T, 2},
    c::T,
    A::AbstractArray{T, 2},
    b::AbstractArray{T, 1},
    n_samples::Int=1
) where T<:AbstractFloat

    # check input validity
    n = size(M, 1)
    m = size(A, 1)
    if !issymmetric(M)
        throw(ArgumentError("Matrix M must be symmetric"))
    end
    if c <= 0
        throw(ArgumentError("Constant c must be positive"))
    end
    if size(A, 2) != n
        throw(ArgumentError("Constraint matrix A must have same number of columns as M"))
    end
    if length(b) != m
        throw(ArgumentError("Length of b must match number of constraints"))
    end
    
    # find a particular point satisfying Ax = b and is closest to the origin measured in the M norm
    # x_p = M^(-1)A^T (AM^(-1)A^T)^(-1)b
    Minv = inv(M)
    MinvAT = Minv * A'
    x_p = MinvAT * (A * MinvAT \ b)
    
    # verify x_p satisfies linear constraints
    if !isapprox(A * x_p, b, rtol=1e-10)
        throw(ArgumentError("Numerical issues in finding particular solution"))
    end
    
    # determine if intersection exists by checking if x_p^T M x_p <= c
    energy_p = x_p' * M * x_p
    if energy_p > c
        throw(ArgumentError("No intersection exists: constraints do not intersect ellipsoid"))
    elseif energy_p == c  # tangent to ellipsoid at x_p
        return n_samples == 1 ? x_p : repeat(x_p, 1, n_samples)
    end
    
    # find basis for nullspace of A 
    null_basis = nullspace_qr(A)
    if size(null_basis, 2) == 0
        throw(ArgumentError("No degrees of freedom left after constraints"))
    end
    
    # generate random directions in nullspace
    k = size(null_basis, 2)  # dimension of nullspace
    Z = null_basis * nSphereSampling(k, n_samples)  # (n, n_samples)
    
    # for each direction, solve quadratic equation for correct scaling
    X = zeros(n, n_samples)
    for i in 1:n_samples
        z = Z[:,i]
        
        # solve quadratic equation
        a_quad = z' * M * z
        b_quad = 2 * x_p' * M * z
        c_quad = x_p' * M * x_p - c
        
        # want the positive solution for intersection
        discriminant = b_quad^2 - 4*a_quad*c_quad
        if discriminant < 0
            throw(ArgumentError("Numerical error: discriminant < 0"))
        end
        scale = (-b_quad + sqrt(discriminant)) / (2*a_quad)
        X[:,i] = x_p + scale * z
    end
    
    return n_samples == 1 ? X[:, 1] : X
end

"""Helper function to compute the nullspace of a matrix."""
function nullspace_qr(A; atol=eps(real(eltype(A))))
    F = qr(A', ColumnNorm())
    r = sum(abs.(diag(F.R)) .> atol)
    return F.Q[:, (r+1):end]
end

"""
Sample random points on the intersection of an ellipsoid surface x^T M x = c and 
a set of linear constraints Ax = b within tolerances.

Arguments:
    M: ellipsoid matrix (n x n)
    c: ellipsoid constant
    A: constraint matrix (m x n)
    b: constraint vector (m)
    eps_c: tolerance for ellipsoid constraint
    eps_b: tolerance for linear constraints
    n_samples: number of samples
"""
function ellipsoidLinearConstraintsSampling(
    M::AbstractArray{T, 2},
    c::T,
    A::AbstractArray{T, 2},
    b::AbstractArray{T, 1},
    eps_c::T,
    eps_b::AbstractArray{T, 1},
    n_samples::Int=1
) where T<:AbstractFloat

    # check input validity
    n = size(M, 1)
    m = size(A, 1)
    if !issymmetric(M)
        throw(ArgumentError("Matrix M must be symmetric"))
    end
    if c <= 0
        throw(ArgumentError("Constant c must be positive"))
    end
    if size(A, 2) != n
        throw(ArgumentError("Constraint matrix A must have same number of columns as M"))
    end
    if length(b) != m
        throw(ArgumentError("Length of b must match number of constraints"))
    end
    
    # find a particular point satisfying Ax = b and is closest to the origin measured in the M norm
    # x_p = M^(-1)A^T (AM^(-1)A^T)^(-1)b
    Minv = inv(M)
    MinvAT = Minv * A'
    x_p = MinvAT * (A * MinvAT \ b)

    # verify x_p satisfies linear constraints
    if !isapprox(A * x_p, b, rtol=1e-10)
        throw(ArgumentError("Numerical issues in finding particular solution"))
    end
    
    # determine if intersection exists by checking if x_p^T M x_p <= c
    energy_p = x_p' * M * x_p
    if energy_p > c
        throw(ArgumentError("No intersection exists: constraints do not intersect ellipsoid"))
    elseif energy_p == c  # tangent to ellipsoid at x_p
        return n_samples == 1 ? x_p : repeat(x_p, 1, n_samples)
    end
    
    # find basis for nullspace of A 
    null_basis = nullspace_qr(A)
    if size(null_basis, 2) == 0
        throw(ArgumentError("No degrees of freedom left after constraints"))
    end
    k = size(null_basis, 2)  # dimension of nullspace
    
    # generate samples
    X = zeros(T, n, n_samples)
    accepted = 0
    attempts = 0
    max_attempts = n_samples * 100  # prevent infinite loop

    while accepted < n_samples && attempts < max_attempts
        attempts += 1
        
        # pick a random direction in nullspace
        z = randn(k)
        z /= norm(z)
        w = null_basis * z

        # sample perturbations for linear constraints within their tolerances
        delta = rand(m) .* 2 .* eps_b .- eps_b
        
        # update particular solution for perturbed constraints
        x_p_perturbed = MinvAT * (A * MinvAT \ (b + delta))
        
        # solve quadratic equation with perturbed target value
        c_perturbed = c + rand() * 2 * eps_c - eps_c
        a_quad = w' * M * w
        b_quad = 2 * x_p_perturbed' * M * w
        c_quad = x_p_perturbed' * M * x_p_perturbed - c_perturbed
        
        discriminant = b_quad^2 - 4*a_quad*c_quad
        if discriminant < 0
            continue
        end
        scale = (-b_quad + sqrt(discriminant)) / (2*a_quad)
        
        x = x_p_perturbed + scale * w

        # verify all constraints within tolerances
        if abs(x' * M * x - c) > eps_c
            continue
        end
        if any(abs.(A * x - b) .> eps_b)
            continue
        end
        
        accepted += 1
        X[:, accepted] = x
    end
    
    if accepted < n_samples
        @warn "Only generated $accepted points out of $n_samples requested within tolerance after $max_attempts attempts"
        return X[:, 1:accepted]
    end
    
    @info "Generated $accepted points within tolerance after $attempts attempts"
    return n_samples == 1 ? X[:, 1] : X
end


if ARGS == ["--run"]

    using MultiFloats

    # # Example usage
    # M = [1.0 0.0 0.5 0.0; 0.0 1.0 0.0 0.5; 0.5 0.0 1.0 0.0; 0.0 0.5 0.0 1.0]
    # c = 1.0
    # num_samples = 4
    # ellipsoid_samples = ellipsoidSampling(M, c, num_samples)
    # println(ellipsoid_samples)

    # # Verify that the samples lie on the ellipsoid
    # for i in 1:num_samples
    #     lhs = ellipsoid_samples[:, i]' * M * ellipsoid_samples[:, i]
    #     @assert abs(lhs - c) < 1e-6 "Sample does not lie on the ellipsoid, lhs = $lhs"

    #     p1 = ellipsoid_samples[1:2, i]
    #     p2 = ellipsoid_samples[3:4, i]
    #     p3 = -(p1 + p2)
    #     lhs = 0.5 * (sum(abs2, p1) + sum(abs2, p2) + sum(abs2, p3)) 
    #     @assert abs(lhs - c) < 1e-6 "Sample does not lie on the ellipsoid, lhs = $lhs"
    # end

    # Example usage
    # samples = BoxSampling(5, [(-1.0, 1.0), (-2.0, 2.0)])
    # samples = ShellSampling(5, [(1.0, 2.0), (1.0, 2.0)])
    # println(samples)

    # # Example usage
    # M = [2.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    # c = 1.0
    # A = [1.0 1.0 0.0 0.0]  # Two constraints
    # b = [0.5]
    # n = 5
    # eps_c = 1e-2
    # eps_b = [1e-2]
    # # points = ellipsoidLinearConstraintsSampling(M, c, A, b, n)
    # # points = ellipsoidLinearConstraintsSampling(M, c, A, b, eps_c, eps_b, n)
    # points = ellipsoidLinearConstraintsSampling(
    #     Float64x4.(M), Float64x4(c), Float64x4.(A), Float64x4.(b), Float64x4(eps_c), Float64x4.(eps_b), n)
    # println(points)
    # println(eltype(points))

    # # Verification
    # for i in 1:size(points, 2)
    #     x = points[:,i]
    #     # @assert isapprox(x' * M * x, c, rtol=1e-10)     # On ellipsoid
    #     # @assert isapprox(A * x, b, rtol=1e-10)          # Satisfies all constraints
    #     @assert abs(x' * M * x - c) <= eps_c     # Within ellipsoid tolerance
    #     for j in 1:size(A,1)
    #         @assert abs(A[j,:]' * x - b[j]) <= eps_b[j]  # Within linear constraints tolerances
    #     end
    # end

end