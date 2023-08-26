"""
Parareal algorithm 
"""

using Distributed  # for parallel computing
addprocs(4)
println("workers: ", workers())

include("../tools/utils.jl")
include("../tools/python_model.jl")
include("Parareal.jl")
@everywhere include("../tools/ode_solver.jl")
@everywhere include("../tools/setups/fpu.jl")

using ArgParse
using DataStructures
using .Parareal
@everywhere using .FPU

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "problem"
            help = "ode problem name"
            arg_type = String
            required = true
        "T_init"
            help = "init time"
            arg_type = Float64
            required = true
        "T_end"
            help = "end time"
            arg_type = Float64
            required = true
        "N"
            help = "number of intervals"
            arg_type = Int
            required = true
        "niters"
            help = "number of parareal iterations"
            arg_type = Int
            required = true
        "--Nf"
            help = "number of fine steps per interval"
            arg_type = Int
            default = 128
        "--Nc"
            help = "number of coarse steps per interval"
            arg_type = Int
            default = 64
        "--fine_method", "-f"
            help = "fine solver method"
            arg_type = String
            default = "VelocityVerlet"
        "--coarse_method", "-c"
            help = "coarse solver method"
            arg_type = String
            default = "VelocityVerlet"
        "--nn_checkpoint_path"
            help = "NN checkpoint path"
            arg_type = String
            default = ""
        "--output_dir"
            help = "output directory"
            arg_type = String
            default = "./"
        "--use_float64x4"
            help = "use Float64x4 for solutions"
            action = :store_true
        "--fpu_omega"
            help = "omega parameter of FPU problem"
            arg_type = Float64
            default = 300.
    end

    return parse_args(s)
end

@everywhere function ode_solve_wrapper(
    p0::AbstractArray{T, 1}, 
    q0::AbstractArray{T, 1}, 
    t0::Float64, 
    H::Float64; 
    func::Function,
    method::String, 
    nsteps::Integer, 
    param::Any, 
    T2::Type) where T<:AbstractFloat

    if T != T2
        p0 = convert.(T2, p0)
        q0 = convert.(T2, q0)
    end
    p, q = ode_solve(func, METHODS[method], p0, q0, t0, H, nsteps, false, param)
    if T != T2
        p = convert.(T, p)
        q = convert.(T, q)
    end
    return p, q
end

function main()
    parsed_args = parse_commandline()
    
    problem = parsed_args["problem"]
    T_init = parsed_args["T_init"]
    T_end = parsed_args["T_end"]
    N = parsed_args["N"]
    Nf = parsed_args["Nf"]
    Nc = parsed_args["Nc"]
    niters = parsed_args["niters"]
    fine_method = parsed_args["fine_method"]
    coarse_method = parsed_args["coarse_method"]
    nn_checkpoint_path = parsed_args["nn_checkpoint_path"]
    output_dir = parsed_args["output_dir"]
    use_float64x4 = parsed_args["use_float64x4"]
    fpu_omega = parsed_args["fpu_omega"]
    
    DeltaT = (T_end-T_init) / N

    println("problem =         ", problem)
    println("fpu_omega =       ", fpu_omega)
    println("use_float64x4 =   ", use_float64x4)
    println("output_dir =      ", output_dir)
    println("T_init =          ", T_init)
    println("T_end =           ", T_end)
    println("N =               ", N)
    println("Delta_T_com =     ", DeltaT)
    println("niters =          ", niters)
    println("fine_method =     ", fine_method)
    println("Nf =              ", Nf)
    println("fine_stepsize =   ", DeltaT / Nf)

    if isempty(nn_checkpoint_path)
        println("coarse_method =   ", coarse_method)
        println("Nc =              ", Nc)
        println("coarse_stepsize = ", DeltaT / Nc)

        config = OrderedDict(
            "problem"=>problem, "omega"=>fpu_omega,
            "T_init"=>T_init, "T_end"=>T_end, 
            "N"=>N, "Nf"=>Nf, "Nc"=>Nc, "niters"=>niters,
            "fine_method"=>fine_method, "coarse_method"=>coarse_method, "use_float64x4"=>use_float64x4,
            "Delta_T_com"=>DeltaT, "fine_stepsize"=>DeltaT / Nf, "coarse_stepsize"=>DeltaT / Nc
        )
        save_config(output_dir, config)
    else
        println("NN_checkpoint_path = ", nn_checkpoint_path)

        config = OrderedDict(
            "problem"=>problem, "omega"=>fpu_omega,
            "T_init"=>T_init, "T_end"=>T_end, 
            "N"=>N, "Nf"=>Nf, "niters"=>niters,
            "fine_method"=>fine_method,  "use_float64x4"=>use_float64x4, 
            "Delta_T_com"=>DeltaT, "fine_stepsize"=>DeltaT / Nf, 
            "NN_checkpoint path"=>nn_checkpoint_path, 
        )
        save_config(output_dir, config)
    end

    if problem == "fpu"
        kwargs = Dict(:omega => fpu_omega)
        param = (fpu_omega^2)/2.
    end 

    fine_solve = (p0, q0, t0, H) -> ode_solve_wrapper(
        p0, q0, t0, H; func=A!, method=fine_method, nsteps=Nf, param=param, T2=Float64x4
        )

    if isempty(nn_checkpoint_path)
        coarse_solve = (p0, q0, t0, H) -> ode_solve_wrapper(
        p0, q0, t0, H; func=A!, method=coarse_method, nsteps=Nc, param=param, T2=use_float64x4 ? Float64x4 : Float64
        )
    else
        coarse_solve = (p0, q0, t0, H) -> nn_solve(p0, q0, load_nn(nn_checkpoint_path))
    end

    p0, q0 = initial_condition(use_float64x4 ? Float64x4 : Float64; kwargs...)

    println("\nInitial condition:")
    println("p0: ", p0)
    println("q0: ", q0)
    println("H0: ", compute_H(p0, q0; kwargs...))
    println("K0: ", compute_K(p0))
    println("U0: ", compute_U(q0; kwargs...))

    println("\nRunning plain parareal ...")
    t = collect(T_init:DeltaT:T_end)    
    p_all, q_all = Parareal.plain(p0, q0, t, fine_solve, coarse_solve, niters=niters)
    save_all_iterations(output_dir, p_all, q_all, use_float64x4 ? Float64x4 : Float64)
end

main()
