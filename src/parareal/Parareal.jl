"""
Parareal algorithms.
"""

module Parareal

using Distributed
using Logging
using LinearAlgebra
using Optim
using ProgressMeter 
include("procrustes.jl")
include("interpolation.jl")
include("../utils/saving_utils.jl")


function save(output_dir, k, u)
    if ~isempty(output_dir)
        filepath = joinpath(output_dir, "k=$k/u.csv")
        @info "Saving iter $k solution at $filepath ..."
        save_csv(filepath, u)
    end
end

function save(output_dir, k, u, diagnostics)
    if ~isempty(output_dir)
        filepath = joinpath(output_dir, "k=$k/u.csv")
        filepath2 = joinpath(output_dir, "k=$k/diagnostics.csv")
        @info "Saving iter $k solution at $filepath ..."
        save_csv(filepath, u)
        @info "Saving iter $k diagnostics at $filepath2 ..."
        save_csv(filepath2, diagnostics)
    end
end

function save_F_G(output_dir, k, F, G)
    if ~isempty(output_dir)
        filepath = joinpath(output_dir, "k=$k/F.csv")
        @info "Saving iter $k fine solutions F at $filepath ..."
        save_csv(filepath, F)
        filepath2 = joinpath(output_dir, "k=$k/G.csv")
        @info "Saving iter $k coarse solutions G at $filepath2 ..."
        save_csv(filepath2, G)
    end
end

function concatenate_states(states::Vector{<:AbstractArray{T, 1}}) where T<:AbstractFloat
    return hcat(states...)
end

function concatenate_states(states::Vector{<:Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}}) where T<:AbstractFloat
    return (hcat(map(first, states)...), hcat(map(last, states)...))
end


"""
Plain parareal algorithm.

Arguments:
- `u0::T`: The initial state, which can be either a single array or a tuple of arrays.
- `fine_solve::Function`: Function to compute the fine solution over one time interval.
- `coarse_solve::Function`: Function to compute the coarse solution over one time interval.
- `N::Integer`: The number of time intervals.
- `niters::Integer`: The number of parareal iterations.
- `output_dir::String (keyword, default empty)`: Directory for saving intermediate results.
- `batch::Bool` (keyword, default `false`): Whether coarse solver can accept a batch of states.

Returns:
- A vector of state histories. Each entry is a vector of states (one per time point)
  representing the solution after that parareal iteration.
  
Assumptions:
- Both `fine_solve` and `coarse_solve` accept a state `u` (of type `T`)
  and return a new state of the same type.
"""
function plain(
    u0::T, 
    fine_solve::Function,
    coarse_solve::Function,
    N::Integer,
    niters::Integer;
    output_dir::String = "",
    batch::Bool = false) where T
    
    # Initialize arrays
    states = Vector{T}(undef, N + 1)
    states_history = Vector{Vector{T}}(undef, niters + 1)

    @info "Computing initial coarse solution (iteration 0)..."
    elapsed_time = @elapsed begin
        states[1] = deepcopy(u0)
        @showprogress for n in 1:N
            states[n+1] = coarse_solve(states[n])
        end
    end
    @info "Initial coarse solution computed in $(elapsed_time) seconds."

    # Record the iteration 0 solution
    states_history[1] = deepcopy(states)
    save(output_dir, 0, states)

    # Main parareal iterations
    for k in 1:niters
        @info "Starting parareal iteration $k..."
        elapsed_time = @elapsed begin
            new_states = similar(states)
            new_states[1] = deepcopy(u0)
            
            # Compute fine solutions in parallel for each interval
            elapsed_time_fine = @elapsed F = @showprogress pmap(fine_solve, states[1:end-1])
            @info "Fine solutions computed in $(elapsed_time_fine) seconds."

            # Compute coarse solutions: batch if applicable, else pmap over intervals
            if batch
                elapsed_time_coarse = @elapsed G = coarse_solve(states[1:end-1])
            else
                elapsed_time_coarse = @elapsed G = @showprogress pmap(coarse_solve, states[1:end-1])
            end
            @info "Coarse solutions computed in $(elapsed_time_coarse) seconds."

            # Propagate the state using the parareal correction
            @showprogress for n in 1:N
                new_states[n+1] = coarse_solve(new_states[n]) .- G[n] .+ F[n]
            end
        end
        @info "Iteration $k complete in $(elapsed_time) seconds."
        states = deepcopy(new_states)
        states_history[k+1] = deepcopy(states)
        save(output_dir, k, states)
    end

    return states_history
end


"""
Procrustes parareal algorithm.

Arguments:
- `u0::T`: The initial state, which can be either a single array or a tuple of arrays.
- `fine_solve::Function`: Function to compute the fine solution over one time interval.
- `coarse_solve::Function`: Function to compute the coarse solution over one time interval.
- `N::Integer`: The number of time intervals.
- `niters::Integer`: The number of parareal iterations.
- `embed_state::Function`: Function to embed a state into a representation suitable for the procrustes problem.
- `align_state::Function`: Function to align a state using the computed procrustes transformation.
- `output_dir::String` (keyword, default empty): Directory for saving intermediate results.
- `with_additive::Bool` (keyword, default `true`): Whether to use an additive correction.
- `use_scaling::Bool` (keyword, default `false`): Whether to use scaling in the procrustes problem.
- `k_most_recent::Integer` (keyword, default `1`): Number of most recent iterations to consider for procrustes alignment.
- `window_size::Integer` (keyword, default `-1`): Size of the window for local procrustes alignment.
- `batch::Bool` (keyword, default `false`): Whether coarse solver can accept a batch of states.


Returns:
- A vector of state histories. Each entry is a vector of states (one per time point)
  representing the solution after that parareal iteration.
  
Assumptions:
- Both `fine_solve` and `coarse_solve` accept a state `u` (of type `T`)
  and return a new state of the same type.
- `embed_state` accepts a state `u` (of type `T`) and returns a vector of scalars.
- `align_state` accepts a state `u` (of type `T`) and a procrustes transformation `pa`
  and returns a new state of the same type.
"""
function procrustes(
    u0::T,
    fine_solve::Function,
    coarse_solve::Function,
    N::Integer,
    niters::Integer,
    embed_state::Function,
    align_state::Function;
    output_dir::String = "",
    with_additive::Bool=true,
    use_scaling::Bool=false,
    k_most_recent::Integer=1,
    window_size::Integer=-1,
    batch::Bool=false) where T
    
    # Validate window size and determine if we should use a global procrustes correction
    if window_size == -1 || window_size == N
        use_global = true
    elseif window_size > 0 && window_size < N
        use_global = false
    else
        error("Invalid window size: $window_size")
    end

    # Initialize arrays
    states = Vector{T}(undef, N + 1)
    states_history = Vector{Vector{T}}(undef, niters + 1)
    F_history = Vector{Vector{T}}(undef, niters)
    G_history = Vector{Vector{T}}(undef, niters)

    @info "Computing initial coarse solution (iteration 0)..."
    elapsed_time = @elapsed begin
        states[1] = deepcopy(u0)
        @showprogress for n in 1:N
            states[n+1] = coarse_solve(states[n])
        end
    end
    @info "Initial coarse solution computed in $(elapsed_time) seconds."

    # Record the iteration 0 solution
    states_history[1] = deepcopy(states)
    save(output_dir, 0, states)

    # Main procrustes parareal iterations
    for k in 1:niters
        @info "Starting parareal iteration $k..."
        elapsed_time = @elapsed begin
            new_states = similar(states)
            new_states[1] = deepcopy(u0)
            
            # Compute fine solutions in parallel for each interval
            elapsed_time_fine = @elapsed F = @showprogress pmap(fine_solve, states[1:end-1])
            @info "Fine solutions computed in $(elapsed_time_fine) seconds."

            # Compute coarse solutions: batch if applicable, else pmap over intervals
            if batch
                elapsed_time_coarse = @elapsed G = coarse_solve(states[1:end-1])
            else
                elapsed_time_coarse = @elapsed G = @showprogress pmap(coarse_solve, states[1:end-1])
            end
            @info "Coarse solutions computed in $(elapsed_time_coarse) seconds."
            
            # Update F, G history
            F_history[k] = deepcopy(F)
            G_history[k] = deepcopy(G)

            if k == 1  # save F, G for the first iteration for diagnostics 
                save_F_G(output_dir, k, F, G)
            end 

            # Determine iteration boundaries centered around k
            iter_start = max(1, k - k_most_recent + 1)
            iter_end = k
            
            if use_global
                # Prepare data matrices
                F_global = vcat(F_history[iter_start:iter_end]...)
                G_global = vcat(G_history[iter_start:iter_end]...)
                Fh_global = concatenate_states(embed_state.(F_global))
                Gh_global = concatenate_states(embed_state.(G_global))

                # Compute the global procrustes transformation
                pa = procrustes_alignment(Fh_global, Gh_global, use_scaling)
                # @show pa 
                # pa = PAH(Fh, Gh)
                # pa = HPA(Fh, Gh)
                # pa = DoublePA(Fh, Gh)
            else
                determine_pa = n -> begin
                    # For each n,    
                    # Determine window boundaries centered around n
                    w_start = max(1, n - div(window_size, 2))
                    w_end   = min(N, n + div(window_size, 2))

                    # Prepare data matrices
                    F_local = vcat([F_history[l][w_start:w_end] for l in iter_start:iter_end]...)
                    G_local = vcat([G_history[l][w_start:w_end] for l in iter_start:iter_end]...)
                    Fh_local = concatenate_states(embed_state.(F_local))
                    Gh_local = concatenate_states(embed_state.(G_local))

                    # Compute the local procrustes transformation
                    procrustes_alignment(Fh_local, Gh_local, use_scaling)
                end
                pas = @showprogress pmap(determine_pa, 1:N)
            end
            
            # Propagate the state using the procrustes parareal correction
            if with_additive
                @showprogress for n in 1:N
                    pa = use_global ? pa : pas[n]
                    new_states[n+1] = align_state(coarse_solve(new_states[n]), pa) .- align_state(G[n], pa) .+ F[n]
                end
            else
                @showprogress for n in 1:N
                    pa = use_global ? pa : pas[n]
                    new_states[n+1] = align_state(coarse_solve(new_states[n]), pa)
                end
            end
        end
        @info "Iteration $k complete in $(elapsed_time) seconds."
        states = deepcopy(new_states)
        states_history[k+1] = deepcopy(states)
        save(output_dir, k, states)
    end

    return states_history
end


function interpolative(
    u0::T,
    fine_solve::Function,
    coarse_solve::Function,
    N::Integer,
    niters::Integer,
    embed_state::Function,
    align_state::Function;
    output_dir::String = "",
    tol::Float64=1e-14,
    use_bias::Bool=true,
    centering::Bool=true,
    k_most_recent::Integer=-1,
    window_size::Integer=-1,
    k_nearest::Integer=-1,
    batch::Bool=false) where T
    
    # Get dimension and dtype of the embedded state 
    d = length(embed_state(u0))
    dtype = eltype(embed_state(u0))

    # Validate window size and determine if we should use a global interpolative correction 
    # when centering is disabled (when centering is enabled, we always use local interpolation)
    if window_size == -1 || window_size == N
        use_global = true
    elseif window_size > 0 && window_size < N
        use_global = false
    else
        error("Invalid window size: $window_size")
    end

    # Initialize arrays
    states = Vector{T}(undef, N + 1)
    states_history = Vector{Vector{T}}(undef, niters + 1)
    F_history = Vector{Vector{T}}(undef, niters)
    G_history = Vector{Vector{T}}(undef, niters)

    # Initialize diagnostics vector
    diagnostic_keys = ["num_singular", "condition_number", "residual", "is_exception", "range_space_projection_ratio"]
    diagnostics = [Dict{String, Vector{Any}}() for _ in 1:niters]
    for k in 1:niters
        diagnostics[k]["num_singular"] = zeros(Int, N)
        diagnostics[k]["condition_number"] = zeros(dtype, N)
        diagnostics[k]["residual"] = zeros(dtype, N)
        diagnostics[k]["is_exception"] = falses(N)
        diagnostics[k]["range_space_projection_ratio"] = zeros(dtype, N)
    end

    @info "Computing initial coarse solution (iteration 0)..."
    elapsed_time = @elapsed begin
        states[1] = deepcopy(u0)
        @showprogress for n in 1:N
            states[n+1] = coarse_solve(states[n])
        end
    end
    @info "Initial coarse solution computed in $(elapsed_time) seconds."

    # Record the iteration 0 solution
    states_history[1] = deepcopy(states)
    save(output_dir, 0, states)

    # Main interpolative parareal iterations
    for k in 1:niters
        @info "Starting parareal iteration $k..."
        elapsed_time = @elapsed begin
            new_states = similar(states)
            new_states[1] = deepcopy(u0)
            
            # Compute fine solutions in parallel for each interval
            elapsed_time_fine = @elapsed F = @showprogress pmap(fine_solve, states[1:end-1])
            @info "Fine solutions computed in $(elapsed_time_fine) seconds."

            # Compute coarse solutions: batch if applicable, else pmap over intervals
            if batch
                elapsed_time_coarse = @elapsed G = coarse_solve(states[1:end-1])
            else
                elapsed_time_coarse = @elapsed G = @showprogress pmap(coarse_solve, states[1:end-1])
            end
            @info "Coarse solutions computed in $(elapsed_time_coarse) seconds."
            
            # Update F, G history
            F_history[k] = deepcopy(F)
            G_history[k] = deepcopy(G)
            
            # Determine iteration boundaries centered around k
            iter_start = k_most_recent > 0 ? max(1, k - min(d+1, k_most_recent-1)) : max(1, k - (d+1))
            iter_end = k

            # Perform linear interpolation for n = 1, ..., N in parallel
            determine_linear = n -> begin    

                # Determine window boundaries centered around n
                if use_global
                    w_start = 1
                    w_end = N
                else
                    w_start = max(1, n - div(window_size, 2))
                    w_end = min(N, n + div(window_size, 2))
                end

                # Prepare data matrices X_n, Y_n                    
                X_n = vcat([states_history[l][w_start:w_end] for l in iter_start:iter_end]...)
                F_n = vcat([F_history[l][w_start:w_end] for l in iter_start:iter_end]...)
                G_n = vcat([G_history[l][w_start:w_end] for l in iter_start:iter_end]...)
                Y_n = [f .- g for (f, g) in zip(F_n, G_n)]
                # X_n = G_n
                # Y_n = F_n
                X_n = concatenate_states(embed_state.(X_n))  # d x (d+1)ws
                Y_n = concatenate_states(embed_state.(Y_n))  # d x (d+1)ws

                if centering
                    # Choose centering point
                    xc_n = embed_state(states[n])
                    yc_n = embed_state(F[n] .- G[n])
                    # xc_n = embed_state(G[n])
                    # yc_n = embed_state(F[n])

                    # Compute linear interpolation with centering
                    linear = linear_interpolation(X_n, Y_n, xc=xc_n, yc=yc_n, use_bias=use_bias, tol=tol)
                else
                    # Compute linear interpolation without centering
                    linear = linear_interpolation(X_n, Y_n, use_bias=use_bias, tol=tol)
                end

                return linear
            end
            linears = @showprogress map(determine_linear, 1:N)  # in practice map is faster than pmap here (why? maybe related to data copy overhead)
                
            # Propagate the state using the interpolative parareal correction
            @showprogress for n in 1:N
                linear = linears[n]
                
                # Record diagnostics
                diagnostics[k]["num_singular"][n] = linear.rank
                diagnostics[k]["condition_number"][n] = linear.condition_number
                diagnostics[k]["residual"][n] = linear.residual

                # Compute range space projection ratio
                proj_ratio = range_space_projection_ratio(embed_state(new_states[n]), linear)
                diagnostics[k]["range_space_projection_ratio"][n] = proj_ratio

                # Determine if we should use standard parareal or interpolative parareal
                use_standard = linear.rank <= 1
                # use_standard = linear.rank <= 1 || proj_ratio < 0.5 || isnan(proj_ratio)
                
                if use_standard
                    diagnostics[k]["is_exception"][n] = true
                    new_states[n+1] = coarse_solve(new_states[n]) .- G[n] .+ F[n]
                else
                    corr = align_state(new_states[n], linear) .- align_state(states[n], linear)
                    new_states[n+1] = coarse_solve(new_states[n]) .- G[n] .+ F[n] .+ corr
                    # new_states[n+1] = align_state(coarse_solve(new_states[n]), linear) .- align_state(G[n], linear) .+ F[n]
                end
            end
        end
        @info "Iteration $k complete in $(elapsed_time) seconds."
        states = deepcopy(new_states)
        states_history[k+1] = deepcopy(states)
        save(output_dir, k, states, diagnostics[k])
    end

    return states_history
end


# "Interpolation based theta parareal algorithm."
# function interpolative(
#         p0::AbstractArray{T, 1},
#         q0::AbstractArray{T, 1},
#         fine_solve::Function,
#         coarse_solve::Function,
#         N::Integer,
#         niters::Integer,
#         transform_func::Function;
#         output_dir::String="",
#         tol::T=1e-14,
#         use_bias::Bool=true,
#         centering::Bool=true,
#         k_most_recent::Integer=-1,
#         k_nearest::Integer=-1) where T<:AbstractFloat
    
#     # get dimension d
#     @assert length(p0) == length(q0)
#     d = length(p0)
    
#     # initialize arrays 
#     p = zeros(T, d, N+1)
#     q = zeros(T, d, N+1)
#     pnew = zero(p)
#     qnew = zero(q)
#     p_all = zeros(T, d, N+1, niters+1)
#     q_all = zeros(T, d, N+1, niters+1)

#     # initialize diagnostics vector
#     diagnostics = [Dict(
#         "num_singular"=>zeros(Integer, N), 
#         "condition_number"=>zeros(T, N),
#         "residual"=>zeros(T, N),
#         "is_exception"=>zeros(Bool, N),
#         "range_space_projection_ratio"=>zeros(T, N)) for k in 1:niters]

#     # solve for solutions at iteration 0 
#     @info "Starting iter 0 ..."
#     elapsed_time = @elapsed begin
#         p[:, 1] = p0
#         q[:, 1] = q0
#         @showprogress for n in 1:N
#             p[:, n+1], q[:, n+1] = coarse_solve(p[:, n], q[:, n])
#         end
#     end
#     @info "Done. Elapsed time = $elapsed_time seconds."
    
#     p_all[:, :, 1] = p
#     q_all[:, :, 1] = q
#     save(output_dir, 0, p, q)

#     # initialize W_n for n = 1, ..., N
#     W = zeros(T, 2*d, 2*d+1, N)
#     for i in 1:(2*d+1)
#         W[:, i, :] = [p[:, 1:N]; q[:, 1:N]]
#     end
    
#     if niters == 0
#         return p_all, q_all, diagnostics
#     end 

#     ### for k = 1
#     @info "Starting iter 1 ..."
#     elapsed_time = @elapsed begin
#         pnew[:, 1] = p0
#         qnew[:, 1] = q0

#         F = @showprogress pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
#         G = @showprogress pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))

#         @showprogress for n in 1:N
#             pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n]) .- G[n] .+ F[n] 
#         end
#     end
#     @info "Done. Elapsed time = $elapsed_time seconds."
    
#     p[:, :] = pnew[:, :]
#     q[:, :] = qnew[:, :]
#     p_all[:, :, 2] = p
#     q_all[:, :, 2] = q

#     # update W 
#     W[:, 2:end, :] = W[:, 1:end-1, :]
#     W[:, 1, :] = [p[:, 1:N]; q[:, 1:N]]

#     # initialize K_n for n = 1, ..., N
#     K = zeros(T, 2*d, 2*d+1, N)
#     for n in 1:N
#         dp, dq = F[n] .- G[n]
#         for i in 1:(2*d+1)
#             K[:, i, n] = [dp; dq]
#         end
#     end

#     # record num_singular_vals(W_n) and condition_number(W_n) for n = 1, ..., N
#     diagnostics[1]["num_singular"][:] .= 1
#     diagnostics[1]["condition_number"][:] .= 1.
#     diagnostics[1]["residual"][:] .= 0.
#     diagnostics[1]["is_exception"][:] .= false
#     diagnostics[1]["range_space_projection_ratio"][:] .= 0.

#     save(output_dir, 1, p, q, diagnostics[1])

#     ### for k >= 2 
#     for k in 2:niters
#         @info "Starting iter $k ..."
#         elapsed_time = @elapsed begin
#             pnew[:, 1] = p0
#             qnew[:, 1] = q0
            
#             F = @showprogress pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
#             G = @showprogress pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
            
#             # update K 
#             K[:, 2:end, :] = K[:, 1:end-1, :]
#             for n in 1:N
#                 dp, dq = F[n] .- G[n]
#                 K[:, 1, n] = [dp; dq]
#             end
            
#             # linear_maps = @showprogress pmap((X, Y) -> Linear(X, Y, true, tol), eachslice(W, dims=3), eachslice(K, dims=3))
#             @showprogress for n in 1:N
#                 # perform linear interpolation
#                 if k_most_recent > 0  # TODO: parallelize 
#                     X = W[:, 1:k_most_recent, n]
#                     Y = K[:, 1:k_most_recent, n]
#                 elseif k_nearest > 0
#                     dist = [norm(W[:, i, n] - [pnew[:, n]; qnew[:, n]]) for i in 1:(2*d+1)]
#                     idx = sortperm(dist)[1:k_nearest]
#                     @show idx
#                     X = W[:, idx, n]
#                     Y = K[:, idx, n]
#                 else  # TODO: parallelize 
#                     X = W[:, :, n]
#                     Y = K[:, :, n]
#                 end
#                 X = mapslices(transform_func, X, dims=1)
#                 Xc = transform_func(W[:, 1, n])
#                 Yc = K[:, 1, n]
#                 linear = centering ? Linear(X, Y, Xc, Yc, use_bias, tol) : Linear(X, Y, use_bias, tol)

#                 # record num_singular_vals(W_n) and condition_number(W_n) 
#                 diagnostics[k]["num_singular"][n] = linear.rank
#                 diagnostics[k]["condition_number"][n] = linear.condition_number
#                 diagnostics[k]["residual"][n] = linear.residual
#                 xnew = transform_func([pnew[:, n]; qnew[:, n]])
#                 diagnostics[k]["range_space_projection_ratio"][n] = range_space_projection_ratio(xnew, linear)
                
#                 # corr = linear.A * transform_func([pnew[:, n]; qnew[:, n]] - [p[:, n]; q[:, n]])
#                 ynew = linear(xnew)
#                 corr = linear.A * (xnew - Xc)  # equivalent to linear(xnew) - linear(Xc)

#                 if linear.rank == 1 
#                     diagnostics[k]["is_exception"][n] = true
#                     pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n]) .- G[n] .+ F[n]
#                 elseif norm(ynew) > 2 * maximum(norm.(eachcol(K[:, :, n])))  # TODO: modify condition for exception 
#                     diagnostics[k]["is_exception"][n] = true
#                     pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n]) .- G[n] .+ F[n]
#                 else
#                     pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n]) .- G[n] .+ F[n]
#                     pnew[:, n+1] += corr[1:d]
#                     qnew[:, n+1] += corr[d+1:2*d]
#                 end 
#             end
#         end
#         @info "Done. Elapsed time = $elapsed_time seconds."

#         p[:, :] = pnew[:, :]
#         q[:, :] = qnew[:, :]
#         p_all[:, :, k+1] = p
#         q_all[:, :, k+1] = q

#         # update W
#         W[:, 2:end, :] = W[:, 1:end-1, :]
#         W[:, 1, :] = [p[:, 1:N]; q[:, 1:N]]
        
#         save(output_dir, k, p, q, diagnostics[k])
#     end
    
#     return p_all, q_all, diagnostics
# end


# "Interpolation based theta parareal algorithm"
# function interpolative2(
#         p0::AbstractArray{T, 1},
#         q0::AbstractArray{T, 1},
#         t_grid::AbstractArray{Float64, 1},
#         fine_solve::Function,
#         coarse_solve::Function;
#         niters::Integer=3,
#         tol::T=1e-14) where T<:AbstractFloat
    
#     # get dimension d
#     @assert length(p0) == length(q0)
#     d = length(p0)
    
#     # initialize arrays 
#     p = zeros(T, d, N+1)
#     q = zeros(T, d, N+1)
#     pnew = zero(p)
#     qnew = zero(q)
#     p_all = zeros(T, d, N+1, niters+1)
#     q_all = zeros(T, d, N+1, niters+1)

#     # solve for solutions at iteration 0 
#     p[:, 1] = p0
#     q[:, 1] = q0
#     @showprogress for n in 1:N
#         p[:, n+1], q[:, n+1] = coarse_solve(p[:, n], q[:, n])
#     end
#     p_all[:, :, 1] = p
#     q_all[:, :, 1] = q

#     # initialize diagnostics vector
#     diagnostics = [Dict(
#         "num_singular"=>zeros(Integer, N), 
#         "condition_number"=>zeros(T, N),
#         "interp_err"=>zeros(T, N),
#         "is_exception"=>zeros(Bool, N),
#         "range_space_projection_ratio"=>zeros(T, N)) for k in 1:niters]

#     if niters == 0
#         return p_all, q_all
#     end 
    
#     ### for k = 1
#     println("iter 1")

#     # record num_singular_vals(W_n) and condition_number(W_n) for n = 1, ..., N
#     diagnostics[1]["num_singular"][:] .= 1
#     diagnostics[1]["condition_number"][:] .= 1.
#     diagnostics[1]["interp_err"][:] .= 0.
#     diagnostics[1]["is_exception"][:] .= false
#     diagnostics[1]["range_space_projection_ratio"][:] .= 0.

#     pnew[:, 1] = p0
#     qnew[:, 1] = q0

#     F = @showprogress pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
#     G = @showprogress pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))

#     # initialize X_n, Y_n for n = 1, ..., N
#     X = zeros(T, 2*d, 2*d+1, N)
#     Y = zeros(T, 2*d, 2*d+1, N)
#     for n in 1:N
#         Fu_p, Fu_q = F[n]
#         Cu_p, Cu_q = G[n]
#         for i in 1:(2*d+1)
#             X[:, i, n] = [Cu_p; Cu_q]
#             Y[:, i, n] = [Fu_p; Fu_q]
#         end
#     end

#     @showprogress for n in 1:N
#         pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n]) .- G[n] .+ F[n] 
#     end

#     p[:, :] = pnew[:, :]
#     q[:, :] = qnew[:, :]

#     p_all[:, :, 2] = p
#     q_all[:, :, 2] = q

#     ### for k >= 2 
#     for k in 2:niters
        
#         println("iter ", k)
        
#         pnew[:, 1] = p0
#         qnew[:, 1] = q0
         
#         F = @showprogress pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
#         G = pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
        
#         # update X, Y
#         X[:, 2:end, :] = X[:, 1:end-1, :]
#         Y[:, 2:end, :] = Y[:, 1:end-1, :]
#         for n in 1:N
#             Fu_p, Fu_q = F[n]
#             Cu_p, Cu_q = G[n]
#             X[:, 1, n] = [Cu_p; Cu_q]
#             Y[:, 1, n] = [Fu_p; Fu_q]
#         end
        
#         # linear_maps = @showprogress pmap((X, Y) -> Linear(X, Y, true, tol), eachslice(W, dims=3), eachslice(K, dims=3))

#         @showprogress for n in 1:N            
#             # dist = [norm(W[:, i, n] - [pnew[:, n]; qnew[:, n]]) for i in 1:(2*d+1)]
#             # idx = sortperm(dist)[1:5]
#             # println(idx)

#             # linear = linear_maps[n]
#             linear = Linear(X[:, :, n], Y[:, :, n], X[:, 1, n], Y[:, 1, n], false, tol)
#             # linear = Linear(X[:, 1:5, n], Y[:, 1:5, n], X[:, 1, n], Y[:, 1, n], true, tol)
#             # linear = Linear(W[:, idx, n], K[:, idx, n], W[:, 1, n], K[:, 1, n], true, tol)

#             # record num_singular_vals(W_n) and condition_number(W_n) 
#             diagnostics[k]["num_singular"][n] = linear.rank
#             println("m:", linear.rank)
#             diagnostics[k]["condition_number"][n] = linear.condition_number
#             interp_err = norm(Y[:, 1, n] - linear(X[:, 1, n]))
#             diagnostics[k]["interp_err"][n] = interp_err
#             # println("interp_err:", interp_err)
#             # println("A_norm:", norm(linear.A))
#             # println("A_det:", det(linear.A))

#             if linear.rank <= 1 
#                 pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n]) .- G[n] .+ F[n]
#             else
#                 Cunew_p, Cunew_q = coarse_solve(pnew[:, n], qnew[:, n])

#                 ratio = range_space_projection_ratio([Cunew_p; Cunew_q], linear)
#                 println("ratio:", ratio)
#                 diagnostics[k]["range_space_projection_ratio"][n] = ratio

#                 # if norm(corr) > 2 * maximum(norm.(eachcol(K[:, :, n])))
#                 if ratio < 0.98
#                     println("is exception")
#                     diagnostics[k]["is_exception"][n] = true
#                     pnew[:, n+1], qnew[:, n+1] = (Cunew_p, Cunew_q) .- G[n] .+ F[n]
#                 else
#                     diagnostics[k]["is_exception"][n] = false
#                     dp, dq = (Cunew_p, Cunew_q) .- G[n]
#                     corr = linear.A * [dp; dq]    
#                     pnew[:, n+1], qnew[:, n+1] = F[n]
#                     pnew[:, n+1] += corr[1:d]
#                     qnew[:, n+1] += corr[d+1:2*d]
#                 end
#             end 
#         end

#         p[:, :] = pnew[:, :]
#         q[:, :] = qnew[:, :]
#         p_all[:, :, k+1] = p
#         q_all[:, :, k+1] = q
#     end
    
#     return p_all, q_all, diagnostics
# end


# "Parareal algorithm with symplectic correction"
# function sympcorr(
#         p0::AbstractArray{T, 1},
#         q0::AbstractArray{T, 1},
#         t_grid::AbstractArray{Float64, 1},
#         fine_solve::Function,
#         coarse_solve::Function,
#         phi::Function;
#         objective::Function,
#         niters::Integer=3,
#         with_additive::Bool=true) where T<:AbstractFloat
    
#     # get dimension d
#     @assert length(p0) == length(q0)
#     d = length(p0)
    
#     # initialize arrays 
#     p = zeros(T, d, N+1)
#     q = zeros(T, d, N+1)
#     pnew = zero(p)
#     qnew = zero(q)
#     p_all = zeros(T, d, N+1, niters+1)
#     q_all = zeros(T, d, N+1, niters+1)
    
#     # solve for solutions at iteration 0 
#     p[:, 1] = p0
#     q[:, 1] = q0
#     @showprogress for n in 1:N
#         p[:, n+1], q[:, n+1] = coarse_solve(p[:, n], q[:, n], t_grid[n], dt[n])
#     end
#     p_all[:, :, 1] = p
#     q_all[:, :, 1] = q

#     # begin parareal iterations 
#     for k in 1:niters
        
#         println("iter ", k)
        
#         pnew[:, 1] = p0
#         qnew[:, 1] = q0
        
#         F = @showprogress pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2), t_grid[1:end-1], dt)
# #         G = pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2), t_grid[1:end-1], dt)
#         G = [coarse_solve(p[:, n], q[:, n], t_grid[n], dt[n]) for n in 1:N]
        
#         H = sum(dt)/N
#         res = optimize(h -> objective(h, F, G), -H, H)
#         h = Optim.minimizer(res)
#         println(h)
        
#         if with_additive
#             @showprogress for n in 1:N
#                 pnew[:, n+1], qnew[:, n+1] = phi(coarse_solve(pnew[:, n], qnew[:, n], t_grid[n], dt[n])..., h) .- phi(G[n]..., h) .+ F[n] 
#             end
#         else
#             @showprogress for n in 1:N
#                 pnew[:, n+1], qnew[:, n+1] = phi(coarse_solve(pnew[:, n], qnew[:, n], t_grid[n], dt[n])..., h)
#             end
#         end

#         p = pnew
#         q = qnew

#         p_all[:, :, k+1] = p
#         q_all[:, :, k+1] = q
#     end
    
#     return p_all, q_all
# end

export plain, procrustes, interpolative

end
