"""
Parareal algorithm 
"""

const problem = ARGS[1];
const T_init = parse(Float64, ARGS[2]);
const T_end = parse(Float64, ARGS[3]);
const N = parse(Int, ARGS[4]);
const Nf = parse(Int, ARGS[5]);
const niters = parse(Int, ARGS[6]);
const fine_method = ARGS[7];
const output_dir = ARGS[8];
const num_workers = parse(Int, ARGS[9]);
const use_float64x4 = parse(Bool, ARGS[10]);
const checkpoint_path = ARGS[11];

DeltaT = (T_end-T_init) / N
deltat = DeltaT / Nf

println("problem =         ", problem)
println("T_init =          ", T_init)
println("T_end =           ", T_end)
println("N =               ", N)
println("Nf =              ", Nf)
println("niters =          ", niters)
println("fine method =     ", fine_method)
println("Delta T com =     ", DeltaT)
println("fine stepsize =   ", deltat)
println("use_float64x4 =   ", use_float64x4)
println("output_dir =      ", output_dir)
println("NN checkpoint path = ", checkpoint_path)
println("# workers =       ", num_workers)



include("utils.jl")
include("Parareal.jl")
using .Parareal
using Dates
using DataStructures
using Distributed  # for parallel computing
addprocs(num_workers);
println("\nworkers: ", workers())


config = OrderedDict(
    "problem"=>problem, 
    "T_init"=>T_init, "T_end"=>T_end, 
    "N"=>N, "Nf"=>Nf, "niters"=>niters,
    "fine method"=>fine_method,  "use_float64x4"=>use_float64x4, 
    "NN checkpoint path"=>checkpoint_path, 
    "Delta T com"=>DeltaT, "fine stepsize"=>deltat, 
)
save_config(output_dir, config)


@everywhere begin 
    include("../tools/setups/$($problem).jl")
    include("../tools/ode_solver.jl")
    using ProgressMeter 

    if $problem == "fpu"
        param = $use_float64x4 ? (Float64x4(300.)^2)/2. : (300. ^2)/2.
    end 
    
    fine_solve(p0, q0, t0, H) = ode_solve(A!, methods[$fine_method], p0, q0, t0, H, $Nf, false, param)
end

using PyCall
py"""
import sys
sys.path.insert(0, "/workspace/projects_rui/learnsolnmap")
"""
torch = pyimport("torch")
model = pyimport("model")

checkpoint = torch.load(checkpoint_path)
# nn_func = model.SolutionMap(checkpoint["hyper_parameters"]...).double()
nn_func = model.SolutionMap.load_from_checkpoint(checkpoint_path, strict=false).double()
nn_func.load_state_dict(checkpoint["state_dict"], strict=false)

function coarse_solve(p0, q0, t0, H)
    dim = length(p0);
    T = typeof(p0[1])
    if T <: MultiFloat
        p0 = convert.(Float64, p0)
        q0 = convert.(Float64, q0)
    end
    u0 = torch.tensor(vcat(p0, q0));
    u = nn_func(u0);
    u = u.detach().numpy();
    p = u[1:dim];
    q = u[dim+1:end];
    if T <: MultiFloat
        p = convert.(T, p)
        q = convert.(T, q)
    end
    return p, q
end


if problem == "fpu"
    kwargs = Dict(:omega => use_float64x4 ? Float64x4(300.) : 300.)
end 

p0, q0 = initial_condition(; kwargs...)

println("\nInitial condition:")
println("p0: ", p0)
println("q0: ", q0)
println("H0: ", compute_H(p0, q0; kwargs...))
println("K0: ", compute_K(p0))
println("U0: ", compute_U(q0; kwargs...))


println("\nRunning plain parareal ...")
t = collect(T_init:DeltaT:T_end)    
p_all, q_all = Parareal.plain(p0, q0, t, fine_solve, coarse_solve, niters=niters)
save_solutions(output_dir, t, p_all, q_all)
    