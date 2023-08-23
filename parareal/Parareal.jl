module Parareal

using Distributed
using Optim
using GenericLinearAlgebra 
using ProgressMeter 


"Plain parareal algorithm"
function plain(
        p0::AbstractArray{T, 1},
        q0::AbstractArray{T, 1},
        t_grid::AbstractArray{Float64, 1},
        fine_solve::Function,
        coarse_solve::Function;
        niters::Integer=3) where T<:AbstractFloat
    
    # get dimension d
    @assert length(p0) == length(q0)
    d = length(p0)
    
    # get time intervals 
    dt = t_grid[2:end] - t_grid[1:end-1]
    N = length(dt)
    
    # initialize arrays 
    p = zeros(T, d, N+1)
    q = zeros(T, d, N+1)
    pnew = zero(p)
    qnew = zero(q)
    p_all = zeros(T, d, N+1, niters+1)
    q_all = zeros(T, d, N+1, niters+1)

    # solve for solutions at iteration 0 
    p[:, 1] = p0
    q[:, 1] = q0
    @showprogress for n in 1:N
        p[:, n+1], q[:, n+1] = coarse_solve(p[:, n], q[:, n], t_grid[n], dt[n])
    end
    p_all[:, :, 1] = p
    q_all[:, :, 1] = q
    
    # begin parareal iterations 
    for k in 1:niters
        
        println("iter ", k)
        
        pnew[:, 1] = p0
        qnew[:, 1] = q0
         
        F = @showprogress pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2), t_grid[1:end-1], dt)
#         G = pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2), t_grid[1:end-1], dt)
        G = [coarse_solve(p[:, n], q[:, n], t_grid[n], dt[n]) for n in 1:N]
        
        @showprogress for n in 1:N
            pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n], t_grid[n], dt[n]) .+ F[n] .- G[n]
        end

        p = pnew
        q = qnew

        p_all[:, :, k+1] = p
        q_all[:, :, k+1] = q
    end
    
    return p_all, q_all
end


"Parareal algorithm with symplectic correction"
function sympcorr(
        p0::AbstractArray{T, 1},
        q0::AbstractArray{T, 1},
        t_grid::AbstractArray{Float64, 1},
        fine_solve::Function,
        coarse_solve::Function,
        phi::Function;
        objective::Function,
        niters::Integer=3,
        with_additive::Bool=true) where T<:AbstractFloat
    
    # get dimension d
    @assert length(p0) == length(q0)
    d = length(p0)
    
    # get time intervals 
    dt = t_grid[2:end] - t_grid[1:end-1]
    N = length(dt)
    
    # initialize arrays 
    p = zeros(T, d, N+1)
    q = zeros(T, d, N+1)
    pnew = zero(p)
    qnew = zero(q)
    p_all = zeros(T, d, N+1, niters+1)
    q_all = zeros(T, d, N+1, niters+1)
    
    # solve for solutions at iteration 0 
    p[:, 1] = p0
    q[:, 1] = q0
    @showprogress for n in 1:N
        p[:, n+1], q[:, n+1] = coarse_solve(p[:, n], q[:, n], t_grid[n], dt[n])
    end
    p_all[:, :, 1] = p
    q_all[:, :, 1] = q

    # begin parareal iterations 
    for k in 1:niters
        
        println("iter ", k)
        
        pnew[:, 1] = p0
        qnew[:, 1] = q0
        
        F = @showprogress pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2), t_grid[1:end-1], dt)
#         G = pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2), t_grid[1:end-1], dt)
        G = [coarse_solve(p[:, n], q[:, n], t_grid[n], dt[n]) for n in 1:N]
        
        H = sum(dt)/N
        res = optimize(h -> objective(h, F, G), -H, H)
        h = Optim.minimizer(res)
        println(h)
        
        if with_additive
            @showprogress for n in 1:N
                pnew[:, n+1], qnew[:, n+1] = phi(coarse_solve(pnew[:, n], qnew[:, n], t_grid[n], dt[n])..., h) .+ F[n] .- phi(G[n]..., h)
            end
        else
            @showprogress for n in 1:N
                pnew[:, n+1], qnew[:, n+1] = phi(coarse_solve(pnew[:, n], qnew[:, n], t_grid[n], dt[n])..., h)
            end
        end

        p = pnew
        q = qnew

        p_all[:, :, k+1] = p
        q_all[:, :, k+1] = q
    end
    
    return p_all, q_all
end


"Parareal algorithm with phase correction"
function phasecorr(
        p0::AbstractArray{T, 1},
        q0::AbstractArray{T, 1},
        t_grid::AbstractArray{Float64, 1},
        fine_solve::Function,
        coarse_solve::Function,
        Lambda::Function,
        Theta::Function;
        niters::Integer=3,
        with_additive::Bool=true) where T<:AbstractFloat
    
    # get dimension d
    @assert length(p0) == length(q0)
    d = length(p0)
    
    # get time intervals 
    dt = t_grid[2:end] - t_grid[1:end-1]
    N = length(dt)
    
    # initialize arrays 
    p = zeros(T, d, N+1)
    q = zeros(T, d, N+1)
    pnew = zero(p)
    qnew = zero(q)
    p_all = zeros(T, d, N+1, niters+1)
    q_all = zeros(T, d, N+1, niters+1)
    
    # solve for solutions at iteration 0 
    p[:, 1] = p0
    q[:, 1] = q0
    @showprogress for n in 1:N
        p[:, n+1], q[:, n+1] = coarse_solve(p[:, n], q[:, n], t_grid[n], dt[n])
    end
    p_all[:, :, 1] = p
    q_all[:, :, 1] = q

    # begin parareal iterations 
    for k in 1:niters
        
        println("iter ", k)
        
        pnew[:, 1] = p0
        qnew[:, 1] = q0
        
        F = @showprogress pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2), t_grid[1:end-1], dt)
#         G = pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2), t_grid[1:end-1], dt)
        G = [coarse_solve(p[:, n], q[:, n], t_grid[n], dt[n]) for n in 1:N]
        
        Fh = hcat([Lambda(p, q) for (p, q) in F]...)
        Gh = hcat([Lambda(p, q) for (p, q) in G]...)
        
        # solve procrustes problem 
        M = Fh * Gh'
        sol = GenericLinearAlgebra.svd(M)
        Omega = sol.U * sol.Vt
        
        if with_additive
            @showprogress for n in 1:N
                pnew[:, n+1], qnew[:, n+1] = Theta(coarse_solve(pnew[:, n], qnew[:, n], t_grid[n], dt[n])..., Omega) .+ F[n] .- Theta(G[n]..., Omega)
            end
        else
            @showprogress for n in 1:N
                pnew[:, n+1], qnew[:, n+1] = Theta(coarse_solve(pnew[:, n], qnew[:, n], t_grid[n], dt[n])..., Omega)
            end
        end

        p = pnew
        q = qnew

        p_all[:, :, k+1] = p
        q_all[:, :, k+1] = q
    end
    
    return p_all, q_all
end


"Interpolation based theta parareal algorithm"
function interpolative(
        p0::AbstractArray{T, 1},
        q0::AbstractArray{T, 1},
        t_grid::AbstractArray{Float64, 1},
        fine_solve::Function,
        coarse_solve::Function;
        niters::Integer=3,
        tol::T=1e-14) where T<:AbstractFloat
    
    # get dimension d
    @assert length(p0) == length(q0)
    d = length(p0)
    
    # get time intervals 
    dt = t_grid[2:end] - t_grid[1:end-1]
    N = length(dt)
    
    # initialize arrays 
    p = zeros(T, d, N+1)
    q = zeros(T, d, N+1)
    pnew = zero(p)
    qnew = zero(q)
    p_all = zeros(T, d, N+1, niters+1)
    q_all = zeros(T, d, N+1, niters+1)

    # solve for solutions at iteration 0 
    p[:, 1] = p0
    q[:, 1] = q0
    for n in 1:N
        p[:, n+1], q[:, n+1] = coarse_solve(p[:, n], q[:, n], t_grid[n], dt[n])
    end
    p_all[:, :, 1] = p
    q_all[:, :, 1] = q
    
    # initialize W_n for n = 1, ..., N
    W = zeros(T, 2*d+1, 2*d+1, N)
    for i in 1:(2*d+1)
        W[1:end-1, i, :] = [p[:, 1:N]; q[:, 1:N]]
    end
    W[end, :, :] .= 1.

    if niters == 0
        return p_all, q_all
    end 
    
    ### for k = 1
    println("iter 1")

    pnew[:, 1] = p0
    qnew[:, 1] = q0

    F = pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2), t_grid[1:end-1], dt)
    G = pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2), t_grid[1:end-1], dt)
    #         G = [coarse_solve(p[:, n], q[:, n], t_grid[n], dt[n]) for n in 1:N]

    # initialize K_n for n = 1, ..., N
    K = zeros(T, 2*d, 2*d+1, N)
    for n in 1:N
        dp, dq = F[n] .- G[n]
        for i in 1:(2*d+1)
            K[:, i, n] = [dp; dq]
        end
    end

    for n in 1:N
        pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n], t_grid[n], dt[n]) .+ F[n] .- G[n]
    end

    p = pnew
    q = qnew

    p_all[:, :, 2] = p
    q_all[:, :, 2] = q

    # update W 
    W[:, 2:end, :] = W[:, 1:end-1, :]
    W[1:end-1, 1, :] = [p[:, 1:N]; q[:, 1:N]]


    ### for k >= 2 
    for k in 2:niters
        
        println("iter ", k)
        
        pnew[:, 1] = p0
        qnew[:, 1] = q0
         
        F = pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2), t_grid[1:end-1], dt)
        G = pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2), t_grid[1:end-1], dt)
#         G = [coarse_solve(p[:, n], q[:, n], t_grid[n], dt[n]) for n in 1:N]
        
        # update K 
        K[:, 2:end, :] = K[:, 1:end-1, :]
        for n in 1:N
            dp, dq = F[n] .- G[n]
            K[:, 1, n] = [dp; dq]
        end
        
        for n in 1:N
            res = svd(W[:, :, n])
            m = sum(res.S/res.S[1] .> tol)
            if m == 1 
                pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n], t_grid[n], dt[n]) .+ F[n] .- G[n]
            else 
                I = K[:, :, n] * res.V[:, 1:m] * Diagonal(1 ./ res.S[1:m]) * res.U[:, 1:m]'
                corr = I * [pnew[:, n]; qnew[:, n]; 1.]
                if norm(corr) > 2 * maximum(norm.(eachcol(K[:, :, n])))
                    pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n], t_grid[n], dt[n]) .+ F[n] .- G[n]
                else
                    pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n], t_grid[n], dt[n])
                    pnew[:, n+1] += corr[1:d]
                    qnew[:, n+1] += corr[d+1:2*d]
                end
            end 
        end

        p = pnew
        q = qnew

        p_all[:, :, k+1] = p
        q_all[:, :, k+1] = q
        
        # update W
        W[:, 2:end, :] = W[:, 1:end-1, :]
        W[1:end-1, 1, :] = [p[:, 1:N]; q[:, 1:N]]
    end
    
    return p_all, q_all
end


export plain, sympcorr, phasecorr, interpolative

end

