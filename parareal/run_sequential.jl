"""
Sequential algorithm
"""

include("../tools/utils.jl")
include("../tools/ode_solver.jl")
include("../tools/setups/fpu.jl")
include("Parareal.jl")
using ArgParse
using DataStructures
using Printf
using .FPU
using .Parareal


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
        "--Nf"
            help = "number of fine steps per interval"
            arg_type = Int
            default = 128
        "--fine_method", "-f"
            help = "fine solver method"
            arg_type = String
            default = "VelocityVerlet"
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

function main()
    parsed_args = parse_commandline()
    
    problem = parsed_args["problem"]
    T_init = parsed_args["T_init"]
    T_end = parsed_args["T_end"]
    N = parsed_args["N"]
    Nf = parsed_args["Nf"]
    fine_method = parsed_args["fine_method"]
    output_dir = parsed_args["output_dir"]
    use_float64x4 = parsed_args["use_float64x4"]
    fpu_omega = parsed_args["fpu_omega"]

    DeltaT = (T_end-T_init) / N
    deltat = DeltaT / Nf

    config = OrderedDict(
        "problem"=>problem, "omega"=>fpu_omega, "use_float64x4"=>use_float64x4,
        "T_init"=>T_init, "T_end"=>T_end, "N"=>N, "Delta_T_com"=>DeltaT, 
        "fine_method"=>fine_method, "Nf"=>Nf, "fine_stepsize"=>deltat
    )
    save_config(output_dir, config)

    for (key, val) in config
        @printf("%-20s%s\n", "$key =", val)
    end
    @printf("%-20s%s\n", "output_dir =", output_dir)

    if problem == "fpu"
        kwargs = Dict(:omega => fpu_omega)
        param = (fpu_omega^2)/2.
    end 

    fine_solve = (p0, q0, t0, H) -> ode_solve(A!, METHODS[fine_method], p0, q0, t0, H, Nf, false, param)

    p0, q0 = initial_condition(use_float64x4 ? Float64x4 : Float64; kwargs...)

    println("\nInitial condition:")
    println("p0: ", p0)
    println("q0: ", q0)
    println("H0: ", compute_H(p0, q0; kwargs...))
    println("K0: ", compute_K(p0))
    println("U0: ", compute_U(q0; kwargs...))

    println("\nRunning sequential ...")
    t = collect(T_init:DeltaT:T_end)    
    p_all, q_all = Parareal.plain(p0, q0, t, fine_solve, fine_solve, niters=0)
    save_all_iterations(output_dir, p_all, q_all, use_float64x4 ? Float64x4 : Float64)
end

main()
