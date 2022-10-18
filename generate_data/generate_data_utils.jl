using DataFrames
using CSV


function nSphereSampling(n::Int)
    """Sample a random point on a unit n-sphere"""
    
    # generate n Gaussian random variables 
    x = randn(n)
    
    # normalize 
    x /= sqrt(sum(abs2, x)) 
    
    return x
end


function nSphereSampling(n::Int, n_samples::Int)
    """Generate uniformly distributed random points on a unit n-sphere"""
    
    # generate n Gaussian random variables 
    X = randn(n, n_samples)
    
    # normalize 
    X = X ./ sqrt.(sum(abs2, X, dims=1)) 
    
    return X
end


function nBallSampling(n::Int, n_samples::Int)
    """Generate uniformly distributed random points in a unit n-ball"""
    
    # generate n Gaussian random variables 
    X = randn(n, n_samples)
    
    # generate random radius 
    r = rand(1, n_samples) .^ (1/n)
    
    # normalize the vector and multiply by radius 
    X = X ./ sqrt.(sum(abs2, X, dims=1)) 
    X = X .* r
    
    return X
end


function nCubeSampling(n::Int, n_samples::Int)
    """Generate uniformly distributed random points in a unit n-cube"""
    
    X = rand(n, n_samples) .- 0.5
    
    return X
end


function nShellSampling(n::Int, n_samples::Int, epsilon::T) where T<:AbstractFloat
    """Generate uniformly distributed random points in a n-dimensional spherical shell (1 < r < 1+epsilon)"""
    
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


function normal(mean::T, std::T) where T<:AbstractFloat
    """Generate a random variable from a normal distribution"""

    return randn() * std .+ mean
end


function normal(mean::T, std::T, n_samples::Int) where T<:AbstractFloat
    """Generate random variables from a normal distribution"""

    return randn(n_samples) * std .+ mean
end


function save(filepath, P, Q)

    df_P = DataFrame(P', "p" .* string.(1:size(P, 1)))
    df_Q = DataFrame(Q', "q" .* string.(1:size(Q, 1)))     
    
    mkpath(dirname(filepath))
    CSV.write(filepath, hcat(df_P, df_Q))
    
end
