"""
Sequential algorithm
"""

const problem = ARGS[1];
const T_init = parse(Float64, ARGS[2]);
const T_end = parse(Float64, ARGS[3]);
const N = parse(Int, ARGS[4]);
const Nf = parse(Int, ARGS[5]);
const fine_method = ARGS[6];
const output_dir = ARGS[7];
const use_float64x4 = parse(Bool, ARGS[8]);
const fpu_omega = parse(Float64, ARGS[9]);

DeltaT = (T_end-T_init) / N
deltat = DeltaT / Nf

println("problem =         ", problem)
println("omega =           ", fpu_omega)
println("T_init =          ", T_init)
println("T_end =           ", T_end)
println("N =               ", N)
println("Nf =              ", Nf)
println("fine method =     ", fine_method)
println("Delta T com =     ", DeltaT)
println("fine stepsize =   ", deltat)
println("use_float64x4 =   ", use_float64x4)
println("output_dir =      ", output_dir)


include("../tools/utils.jl")
include("Parareal.jl")
using .Parareal
using Dates
using DataStructures
include("../tools/setups/$problem.jl")
include("../tools/ode_solver.jl")


config = OrderedDict(
    "problem"=>problem, "omega"=>fpu_omega,
    "T_init"=>T_init, "T_end"=>T_end, 
    "N"=>N, "Nf"=>Nf, 
    "fine method"=>fine_method, "use_float64x4"=>use_float64x4,
    "Delta T com"=>DeltaT, "fine stepsize"=>deltat
)
save_config(output_dir, config)

dtype = use_float64x4 ? Float64x4 : Float64

if problem == "fpu"
    kwargs = Dict(:omega => fpu_omega)
    param = convert(dtype, (fpu_omega^2)/2.)
end 
    
fine_solve(p0, q0, t0, H) = ode_solve(A!, methods[fine_method], p0, q0, t0, H, Nf, false, param)

p0, q0 = initial_condition(dtype; kwargs...)

println("\nInitial condition:")
println("p0: ", p0)
println("q0: ", q0)
println("H0: ", compute_H(p0, q0; kwargs...))
println("K0: ", compute_K(p0))
println("U0: ", compute_U(q0; kwargs...))


println("\nRunning sequential ...")
t = collect(T_init:DeltaT:T_end)    
p_all, q_all = Parareal.plain(p0, q0, t, fine_solve, fine_solve, niters=0)
save_all_iterations(output_dir, p_all, q_all, dtype)
