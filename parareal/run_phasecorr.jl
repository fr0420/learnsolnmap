"""
Parareal algorithm with phase correction 
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
const with_additive = parse(Bool, ARGS[10]);
const output_dir = ARGS[11];
const num_workers = parse(Int, ARGS[12]);
const use_float64x4 = parse(Bool, ARGS[13]);
const fpu_omega = parse(Float64, ARGS[14]);

DeltaT = (T_end-T_init) / N
deltat = DeltaT / Nf
dT = DeltaT / Nc

println("problem =         ", problem)
println("omega =           ", fpu_omega)
println("T_init =          ", T_init)
println("T_end =           ", T_end)
println("N =               ", N)
println("Nf =              ", Nf)
println("Nc =              ", Nc)
println("niters =          ", niters)
println("fine method =     ", fine_method)
println("coarse method =   ", coarse_method)
println("with additive =   ", with_additive)
println("Delta T com =     ", DeltaT)
println("fine stepsize =   ", deltat)
println("coarse stepsize = ", dT)
println("use_float64x4 =   ", use_float64x4)
println("output_dir =      ", output_dir)
println("# workers =       ", num_workers)


include("../tools/utils.jl")
include("Parareal.jl")
using .Parareal
using Dates
using DataStructures
using Distributed  # for parallel computing
addprocs(num_workers);
println("\nworkers: ", workers())


config = OrderedDict(
    "problem"=>problem, "omega"=>fpu_omega,
    "T_init"=>T_init, "T_end"=>T_end, 
    "N"=>N, "Nf"=>Nf, "Nc"=>Nc, "niters"=>niters,
    "fine method"=>fine_method, "coarse method"=>coarse_method, "use_float64x4"=>use_float64x4,
    "with additive"=>with_additive,
    "Delta T com"=>DeltaT, "fine stepsize"=>deltat, "coarse stepsize"=>dT
)
save_config(output_dir, config)

@everywhere begin 
    include("../tools/setups/$($problem).jl")
    include("../tools/ode_solver.jl")
    using ProgressMeter 

    if $problem == "fpu"
        param = ($fpu_omega^2)/2.
    end 
    
    coarse_solve(p0, q0, t0, H) = ode_solve(A!, methods[$coarse_method], p0, q0, t0, H, $Nc, false, param)
    if $use_float64x4
        fine_solve(p0, q0, t0, H) = ode_solve(A!, methods[$fine_method], p0, q0, t0, H, $Nf, false, param)
    else
        function fine_solve(p0, q0, t0, H)
            p0 = Float64x4.(p0)
            q0 = Float64x4.(q0)
            p, q = ode_solve(A!, methods[$fine_method], p0, q0, t0, H, $Nf, false, param)
            return Float64.(p), Float64.(q)
        end
    end
end


dtype = use_float64x4 ? Float64x4 : Float64

if problem == "fpu"
    kwargs = Dict(:omega => fpu_omega)
end 

p0, q0 = initial_condition(dtype; kwargs...)

println("\nInitial condition:")
println("p0: ", p0)
println("q0: ", q0)
println("H0: ", compute_H(p0, q0; kwargs...))
println("K0: ", compute_K(p0))
println("U0: ", compute_U(q0; kwargs...))


function Lambda(
        p::AbstractArray{T, 1}, 
        q::AbstractArray{T, 1}) where T<:AbstractFloat
    return construct_z(p, q; kwargs...)
end


function Theta(
        p::AbstractArray{T, 1}, 
        q::AbstractArray{T, 1}, 
        Omega::AbstractArray{T, 2}) where T<:AbstractFloat
    znew = Omega * construct_z(p, q; kwargs...)
    pnew, qnew = recover_canonical_vars(znew, q; kwargs...)
    return pnew, qnew
end
        
println("\nRunning parareal with phase correction ...")
t = collect(T_init:DeltaT:T_end)
p_all, q_all = Parareal.phasecorr(p0, q0, t, fine_solve, coarse_solve, Lambda, Theta, niters=niters, with_additive=with_additive)
save_all_iterations(output_dir, p_all, q_all, dtype)
