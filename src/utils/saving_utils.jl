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

"""Save (v, x) tuples to a csv file."""
function save_csv(
    filepath::String, 
    tuples::Vector{Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}},
    ) where T<:AbstractFloat
    
    V = hcat([v for (v, x) in tuples]...)
    X = hcat([x for (v, x) in tuples]...)

    df_V = DataFrame(V', "v" .* string.(1:size(V, 1)))
    df_X = DataFrame(X', "x" .* string.(1:size(X, 1)))     
    
    mkpath(dirname(filepath))
    CSV.write(filepath, hcat(df_V, df_X))
end

"""Save a dictionary of vectors to a csv file."""
function save_csv(
    filepath::String, 
    dict::Dict{String, Vector})
    
    mkpath(dirname(filepath))
    CSV.write(filepath, DataFrame(dict))
end

"""Read V and X matrices from a csv file."""
function read_csv(filepath::String, dtype::Type)
    
    if dtype <: MultiFloat
        df = CSV.read(filepath, DataFrame, types=BigFloat)
        df = convert.(dtype, df) 
    else
        df = CSV.read(filepath, DataFrame, types=dtype)
    end
    
    if ncol(df) % 2 == 0  # [V, X]
        dim = Int(ncol(df)/2)
        df_V = df[:, 1:dim]
        df_X = df[:, dim+1:end]
    else  # [V, X, Dt]
        dim = Int((ncol(df)-1)/2)
        df_V = df[:, 1:dim]
        df_X = df[:, dim+1:2*dim]
        df_Dt = df[:, end]
    end   
    
    V = permutedims(Matrix(df_V))
    X = permutedims(Matrix(df_X))
    
    return V, X
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