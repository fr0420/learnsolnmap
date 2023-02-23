using DataFrames
using CSV


function save_config(dir, config)
    mkpath(dir)
    CSV.write(joinpath(dir, "config.csv"), config)
end


function save_solutions(dir, t, p_all, q_all)
    for k in 1:size(p_all, 3)
        subdir = joinpath(dir, "k=$(k-1)")
        mkpath(subdir)
        
        p = transpose(p_all[:, :, k])
        q = transpose(q_all[:, :, k])
        
        save_solutions_of_iteration_k(subdir, t, p, q)
    end
end


function save_solutions_of_iteration_k(dir, t, p, q)
    df_t = DataFrame((time=t))
    
    df_p = DataFrame(p, "p" .* string.(1:size(p, 2)))
    CSV.write(joinpath(dir, "p.csv"), hcat(df_t, df_p))
    
    df_q = DataFrame(q, "q" .* string.(1:size(q, 2)))
    CSV.write(joinpath(dir, "q.csv"), hcat(df_t, df_q))
    
end


function read_data(dir)
    df_p = DataFrame(CSV.File(joinpath(dir, "p.csv")))
    df_q = DataFrame(CSV.File(joinpath(dir, "q.csv")))
    p = Matrix(df_p[:, 2:end])
    q = Matrix(df_q[:, 2:end])
    t = df_p[:, 1]
    return p, q, t
end


function postprocess(dir, compute_scalar, scalar_name)
    for subdir in readdir(dir)
        if startswith(subdir, "k")
            subdir = joinpath(dir, subdir)
            p, q, t = read_data(subdir)
            scalar = map(compute_scalar, eachslice(p, dims=1), eachslice(q, dims=1))
            df_scalar = DataFrame(scalar_name=>scalar)
            df_t = DataFrame((time=t))
            CSV.write(joinpath(subdir, "$scalar_name.csv"), hcat(df_t, df_scalar))
        end
    end
end