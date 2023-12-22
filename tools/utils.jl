using DataFrames, CSV
using MultiFloats 


function save_config(dir, config)
    mkpath(dir)
    CSV.write(joinpath(dir, "config.csv"), config)
end


function save_all_iterations(
    dir::String, 
    p_all::AbstractArray{T, 3}, 
    q_all::AbstractArray{T, 3}, 
    dtype::Type) where T<:AbstractFloat
    for k in axes(p_all, 3)
        subdir = joinpath(dir, "k=$(k-1)")
        path = joinpath(subdir, "u.csv")
        save_csv(path, p_all[:, :, k], q_all[:, :, k], dtype)
    end
end


function save_all_iterations(
    dir::String, 
    p_all::AbstractArray{T, 3}, 
    q_all::AbstractArray{T, 3}, 
    dtype::Type,
    diagnostics::Vector{Dict{String, Vector}}) where T<:AbstractFloat
    for k in axes(p_all, 3)
        subdir = joinpath(dir, "k=$(k-1)")
        path = joinpath(subdir, "u.csv")
        save_csv(path, p_all[:, :, k], q_all[:, :, k], dtype)
    end
    for k in 1:length(diagnostics)
        subdir = joinpath(dir, "k=$(k-1)")
        path = joinpath(subdir, "diagnostics.csv")
        save_csv(path, diagnostics[k])
    end
end


function save_csv(
    filepath::String, 
    P::AbstractArray{T, 2}, 
    Q::AbstractArray{T, 2}, 
    dtype::Type) where T<:AbstractFloat
    """Save P and Q matrices to a csv file"""
    
    P = convert.(dtype, P)
    Q = convert.(dtype, Q)
    
    df_P = DataFrame(P', "p" .* string.(1:size(P, 1)))
    df_Q = DataFrame(Q', "q" .* string.(1:size(Q, 1)))     
    
    mkpath(dirname(filepath))
    CSV.write(filepath, hcat(df_P, df_Q))
    
end


function save_csv(
    filepath::String, 
    dict::Dict{String, Vector})
    """Save a dictionary of vectors to a csv file"""
    
    mkpath(dirname(filepath))
    CSV.write(filepath, DataFrame(dict))
    
end


function read_csv(filepath::String, dtype::Type)
    """Read P and Q matrices from a csv file"""
    
    if dtype == Float64x4
        df = CSV.read(filepath, DataFrame, types=BigFloat)
        df = convert.(Float64x4, df) 
    else
        df = CSV.read(filepath, DataFrame, types=dtype)
    end
    
    dim = Int(ncol(df)/2)
    df_P = df[:, 1:dim]
    df_Q = df[:, dim+1:end]    
    
    P = Matrix(df_P)'
    Q = Matrix(df_Q)'
    
    return P, Q
end
