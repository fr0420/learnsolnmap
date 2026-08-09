"""
Saving utils.
"""

using DataFrames, CSV
using Logging
using MultiFloats 
using TOML


function save_toml_config(dir, config)
    filepath = joinpath(dir, "config.toml")
    @info "Saving config file at $filepath ..."

    mkpath(dir)
    open(filepath, "w") do io
        TOML.print(io, config, sorted=true)
    end
end


function save_config(dir, config)
    mkpath(dir)
    CSV.write(joinpath(dir, "config.csv"), config)
end


"""Save V and X matrices to a csv file."""
function save_csv(
    filepath::String, 
    V::AbstractArray{T, 2},  # shape = (d, N)
    X::AbstractArray{T, 2},  # shape = (d, N)
    ) where T<:AbstractFloat
    
    df_V = DataFrame(V', "v" .* string.(1:size(V, 1)))
    df_X = DataFrame(X', "x" .* string.(1:size(X, 1)))     
    
    mkpath(dirname(filepath))
    CSV.write(filepath, hcat(df_V, df_X))
end

"""Save U matrix to a csv file."""
function save_csv(
    filepath::String, 
    U::AbstractArray{T, 2},  # shape = (d, N)
    ) where T<:AbstractFloat
    
    df_U = DataFrame(U', "u" .* string.(1:size(U, 1)))  
    
    mkpath(dirname(filepath))
    CSV.write(filepath, df_U)
end

# """Save V and X matrices and Dt values to a csv file."""
# function save_csv(
#     filepath::String, 
#     V::AbstractArray{T, 2},  # shape = (d, N)
#     X::AbstractArray{T, 2},  # shape = (d, N)
#     Dt::AbstractArray{Float64, 1}, # shape = (N,)
#     ) where T<:AbstractFloat
    
#     df_V = DataFrame(V', "v" .* string.(1:size(V, 1)))
#     df_X = DataFrame(X', "x" .* string.(1:size(X, 1)))     
#     df_Dt = DataFrame([Dt], ["Dt"])
    
#     mkpath(dirname(filepath))
#     CSV.write(filepath, hcat(df_V, df_X, df_Dt))
# end

"""Save a vector of (v, x) tuples to a csv file."""
function save_csv(
    filepath::String, 
    tuples::Vector{<:Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}},
    ) where T<:AbstractFloat
    
    V = hcat([v for (v, x) in tuples]...)
    X = hcat([x for (v, x) in tuples]...)

    df_V = DataFrame(V', "v" .* string.(1:size(V, 1)))
    df_X = DataFrame(X', "x" .* string.(1:size(X, 1)))     
    
    mkpath(dirname(filepath))
    CSV.write(filepath, hcat(df_V, df_X))
end

"""Save a vector of vectors to a csv file."""
function save_csv(
    filepath::String, 
    vectors::Vector{<:AbstractArray{T, 1}},
    ) where T<:AbstractFloat
    
    U = hcat(vectors...)
    df_U = DataFrame(U', "u" .* string.(1:size(U, 1)))    
    
    mkpath(dirname(filepath))
    CSV.write(filepath, df_U)
end

"""Save a dictionary of vectors to a csv file."""
function save_csv(
    filepath::String, 
    dict::Dict{String, <:Vector})
    
    mkpath(dirname(filepath))
    CSV.write(filepath, DataFrame(dict))
end

# """Read U matrix or V and X matrices from a csv file."""
# function read_csv(filepath::String, dtype::Type)
    
#     if dtype <: MultiFloat
#         df = CSV.read(filepath, DataFrame, types=BigFloat)
#         df = convert.(dtype, df) 
#     else
#         df = CSV.read(filepath, DataFrame, types=dtype)
#     end

#     if startswith.(names(df), "u") |> sum != 0 
#         # return U matrix
#         df_U = df[:, startswith.(names(df), "u")]
#         U = permutedims(Matrix(df_U))
#         return U
#     elseif startswith.(names(df), "v") |> sum != 0 
#         # return V and X matrices
#         df_V = df[:, startswith.(names(df), "v")]
#         df_X = df[:, startswith.(names(df), "x")]
#         ncol(df_V) == ncol(df_X) || error("V and X matrices must have the same number of columns.")
#         V = permutedims(Matrix(df_V))
#         X = permutedims(Matrix(df_X))
#         return V, X
#     else
#         error("Invalid csv file format.")
#     end 

# end

"""Read states from a csv file. States are either a vector of vectors or a vector of (v, x) tuples."""
function read_csv(filepath::String, dtype::Type)
    
    if dtype <: MultiFloat
        df = CSV.read(filepath, DataFrame, types=BigFloat)
        df = convert.(dtype, df) 
    else
        df = CSV.read(filepath, DataFrame, types=dtype)
    end

    if startswith.(names(df), "u") |> sum != 0 
        # return u vectors
        df_U = df[:, startswith.(names(df), "u")]
        return Array.(eachrow(df_U))

    elseif startswith.(names(df), "v") |> sum != 0 
        # return (v, x) tuples
        df_V = df[:, startswith.(names(df), "v")]
        df_X = df[:, startswith.(names(df), "x")]
        ncol(df_V) == ncol(df_X) || error("V and X matrices must have the same number of columns.")
        # return [(Array(df_V[i, :]), Array(df_X[i, :])) for i in 1:nrow(df)]
        return Array.(eachrow(df_V)), Array.(eachrow(df_X))
    else
        error("Invalid csv file format.")
    end 

end

"""Save Dt values to a csv file."""
function save_Dt(
    filepath::String, 
    Dt::AbstractArray{Float64, 1}, # shape = (N,)
    )
    mkpath(dirname(filepath))
    df_Dt = DataFrame([Dt], ["Dt"])
    CSV.write(filepath, df_Dt)
end

"""Read Dt values from a csv file."""
function read_Dt(filepath::String)
    df = CSV.read(filepath, DataFrame)
    return df.Dt
end