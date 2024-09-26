"""
Sampling functions.
"""

using Distributions


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