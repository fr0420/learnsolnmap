"""
Generate targets 
"""

using Distributed 
addprocs(80);
println("workers: ", workers())

include("../tools/utils.jl")
@everywhere include("../tools/ode_solver.jl")
@everywhere include("../tools/setups/fpu.jl")

using ArgParse
using DataStructures
using Printf
@everywhere using ProgressMeter 
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
        "data_dir"
            help = "data directory"
            arg_type = String
            required = true
        "--Nf"
            help = "number of fine steps per interval"
            arg_type = Int
            default = 128
        "--fine_method", "-f"
            help = "fine solver method"
            arg_type = String
            default = "McAte5"
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
    data_dir = parsed_args["data_dir"]
    use_float64x4 = parsed_args["use_float64x4"]
    fpu_omega = parsed_args["fpu_omega"]

    # assert U0.csv exists in data directory
    init_states_file = joinpath(data_dir, "U0.csv")
    @assert isfile(init_states_file)

    Dt = (T_end-T_init) / N

    config = OrderedDict(
        "problem"=>problem, "use_float64x4"=>use_float64x4,
        "T_init"=>T_init, "T_end"=>T_end, "N"=>N, "Dt"=>Dt,
        "fine_method"=>fine_method, "Nf"=>Nf, "fine_stepsize"=>Dt / Nf,
        "num_initial_states"=>countlines(init_states_file)-1)

    if problem == "fpu"
        config["omega"] = fpu_omega
        param = (fpu_omega^2)/2.
    end

    save_config(data_dir, config)

    for (key, val) in config
        @printf("%-25s%s\n", "$key =", val)
    end
    @printf("%-25s%s\n", "data_dir =", data_dir)

    phi_Dt = (p, q) -> ode_solve(A!, METHODS[fine_method], p, q, 0.0, Dt, Nf, false, param)

    for n in 1:N
        input_file = joinpath(data_dir, "U$(n-1).csv")
        output_file = joinpath(data_dir, "U$n.csv")

        if isfile(output_file)
            println("n=$n computed.")
            continue
        end

        println("\nComputing n=$n ...")
        
        P_init, Q_init = read_csv(input_file, use_float64x4 ? Float64x4 : Float64)
        res = @showprogress pmap(phi_Dt, eachslice(P_init, dims=2), eachslice(Q_init, dims=2))
        P_final = hcat([p for (p, q) in res]...)
        Q_final = hcat([q for (p, q) in res]...)
        
        println("Done.")
        
        save_csv(output_file, P_final, Q_final, use_float64x4 ? Float64x4 : Float64)
    end
end

main()
