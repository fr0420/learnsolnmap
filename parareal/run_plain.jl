"""
Parareal algorithm 
"""

const problem = ARGS[1];
const T_init = parse(Float64, ARGS[2]);
const T_end = parse(Float64, ARGS[3]);
const N = parse(Int, ARGS[4]);
const Nf = parse(Int, ARGS[5]);
const Nc = parse(Int, ARGS[6]);
const niters = parse(Int, ARGS[7]);
const fine_method = ARGS[8];
const coarse_method = ARGS[9];
const output_dir = ARGS[10];
const num_workers = parse(Int, ARGS[11]);

println("problem =       ", problem)
println("T_init =        ", T_init)
println("T_end =         ", T_end)
println("N =             ", N)
println("Nf =            ", Nf)
println("Nc =            ", Nc)
println("niters =        ", niters)
println("fine method =   ", fine_method)
println("coarse method = ", coarse_method)
println("output_dir =    ", output_dir)
println("# workers =     ", num_workers)

DeltaT = (T_end-T_init) / N
deltat = DeltaT / Nf
dT = DeltaT / Nc

println("Delta T com =     ", DeltaT)
println("fine stepsize =   ", deltat)
println("coarse stepsize = ", dT)


using Distributed  # for parallel computing
addprocs(num_workers);
println("\nworkers: ", workers())

include("utils.jl")
include("Parareal.jl")
using .Parareal
using Dates
using DataStructures


config = OrderedDict(
    "problem"=>problem, 
    "T_init"=>T_init, "T_end"=>T_end, 
    "N"=>N, "Nf"=>Nf, "Nc"=>Nc, "niters"=>niters,
    "fine method"=>fine_method, "coarse method"=>coarse_method, 
    "Delta T com"=>DeltaT, "fine stepsize"=>deltat, "coarse stepsize"=>dT
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
    coarse_solve(p0, q0, t0, H) = ode_solve(A, methods[$coarse_method], p0, q0, t0, H, $Nc)
    
end


if problem == "lennardjones"
    p0 = v0
    q0 = x0
end 

println("p0: $p0")
println("q0: $q0")    


t = collect(T_init:DeltaT:T_end)    
p_all, q_all = Parareal.plain(p0, q0, t, fine_solve, coarse_solve, niters=niters)
save_solutions(output_dir, t, p_all, q_all)
    