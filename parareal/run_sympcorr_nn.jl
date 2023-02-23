"""
Parareal algorithm with symplectic correction 
"""

const problem = ARGS[1];
const T_init = parse(Float64, ARGS[2]);
const T_end = parse(Float64, ARGS[3]);
const N = parse(Int, ARGS[4]);
const Nf = parse(Int, ARGS[5]);
const niters = parse(Int, ARGS[6]);
const fine_method = ARGS[7];
const phi_method = ARGS[8];
const distance_func = ARGS[9];
const with_additive = parse(Bool, ARGS[10]);
const output_dir = ARGS[11];
const num_workers = parse(Int, ARGS[12]);
const checkpoint_path = ARGS[13];

println("problem =       ", problem)
println("T_init =        ", T_init)
println("T_end =         ", T_end)
println("N =             ", N)
println("Nf =            ", Nf)
println("niters =        ", niters)
println("fine method =   ", fine_method)
println("phi method =    ", phi_method)
println("distance func = ", distance_func)
println("with additive = ", with_additive)
println("output_dir =    ", output_dir)
println("NN checkpoint path = ", checkpoint_path)
println("# workers =     ", num_workers)

DeltaT = (T_end-T_init) / N
deltat = DeltaT / Nf

println("Delta T com =     ", DeltaT)
println("fine stepsize =   ", deltat)

using Distributed  # for parallel computing
addprocs(num_workers);
println("\nworkers: ", workers())

include("utils.jl")
include("Parareal.jl")
using .Parareal
using Dates
using DataStructures
using LinearAlgebra


config = OrderedDict(
    "problem"=>problem, 
    "T_init"=>T_init, "T_end"=>T_end, 
    "N"=>N, "Nf"=>Nf, "niters"=>niters,
    "fine method"=>fine_method, "NN checkpoint path"=>checkpoint_path, "phi method"=>phi_method,
    "distance func"=>distance_func, "with additive"=>with_additive,
    "Delta T com"=>DeltaT, "fine stepsize"=>deltat,
)
save_config(output_dir, config)


@everywhere begin 
    
    include("./setups/$($problem).jl")
    
    using DifferentialEquations
    
    function ode_solve(A, method, p0, q0, t0, H, nsteps)

        h = H/nsteps 
        prob = SecondOrderODEProblem((du,u,p,t)->A(u), p0, q0, (t0, t0+H));
        sol = solve(prob, method, tstops=t0:h:(t0+H), adaptive=false);
        p = sol[end].x[1]
        q = sol[end].x[2]

        return p, q
    end

    methods = Dict(
        "VelocityVerlet"=>VelocityVerlet(), 
        "CalvoSanz4"=>CalvoSanz4()
    )
    
    fine_solve(p0, q0, t0, H) = ode_solve(A, methods[$fine_method], p0, q0, t0, H, $Nf)
    
end

    using PyCall
    py"""
    import sys
    sys.path.insert(0, "/workspace/projects_rui/learnsolnmap")
    """
    torch = pyimport("torch");
    model = pyimport("model");

    nn_func = model.LitModel.load_from_checkpoint(checkpoint_path, strict=false).double();
    
    function coarse_solve(p0, q0, t0, H)
        dim = length(p0);
        u0 = torch.tensor(vcat(p0, q0));
        u = nn_func(u0);
        u = u.detach().numpy();
        p = u[1:dim];
        q = u[dim+1:end];

        return p, q
    end


function objective(h::T, 
                   F::AbstractArray{Tuple{Array{T, 1}, Array{T, 1}}, 1}, 
                   G::AbstractArray{Tuple{Array{T, 1}, Array{T, 1}}, 1},
                   phi::Function,
                   dist_func::String,
                   construct_z::Union{Function, Nothing}=nothing) where T<:AbstractFloat
    G_corr = [phi(G[n]..., h) for n in 1:length(G)]
    
    if dist_func == "H_norm"
        Fh = [construct_z(p, q) for (p, q) in F]
        Gh = [construct_z(p, q) for (p, q) in G_corr]
        return norm(Fh.-Gh)
    elseif dist_func == "l2_norm"
        return norm([f.-g for (f, g) in zip(F, G_corr)])
    else
    end
        
end


if problem == "lennardjones"
    p0 = v0
    q0 = x0
end 

println("p0: $p0")
println("q0: $q0")    

t = collect(T_init:DeltaT:T_end)
phi(p0, q0, h) = ode_solve(A, methods[phi_method], p0, q0, 0.0, h, 1)
func(h, F, G) = objective(h, F, G, phi, distance_func, construct_z)
# func(h, F, G) = objective(h, F, G, phi, distance_func)

p_all, q_all = Parareal.sympcorr(p0, q0, t, fine_solve, coarse_solve, phi, 
        objective=func, niters=niters, with_additive=with_additive)

save_solutions(output_dir, t, p_all, q_all)
