"""
Run parareal algorithm. 
"""

using Distributed  # for parallel computing
addprocs(40)

include("../utils/parsing_utils.jl")
include("../utils/logging_utils.jl")
include("../utils/saving_utils.jl")
@everywhere include("Parareal.jl")
@everywhere include("../utils/python_model.jl")
@everywhere include("../utils/ode_solver.jl")
@everywhere include("../problems/problems.jl")

using ArgParse
using TOML
using .Parareal


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "toml_config"
            help = "TOML config file"
            arg_type = String
            required = true
        "--output_dir"
            help = "output directory"
            arg_type = String
            default = "."
    end

    return parse_args(s)
end

function get_solvers(config::Dict{String, Any}, prob::SeparableHamiltonianSystem)
    solver_kwargs = get_solver_kwargs(config)
    fine_solve = u0 -> ode_solve(
        (ddx, dx, x, p, t) -> compute_ddx!(prob, ddx, dx, x),
        METHODS[config["fine_method"]],
        u0,
        0.,
        config["Delta_t"],
        config["Nf"], 
        false,
        target_type=Float64x4,
        solver_kwargs...
    )

    if ~haskey(config, "nn_ckpt_path")
        coarse_solve = u0 -> ode_solve(
            (ddx, dx, x, p, t) -> compute_ddx!(prob, ddx, dx, x),
            METHODS[config["coarse_method"]],
            u0,
            0.,
            config["Delta_t"],
            config["Nc"], 
            false,
            target_type=config["use_float64x4"] ? Float64x4 : Float64,
            solver_kwargs...
        )
    else
        nn_func = load_nn(config["nn_ckpt_path"], config["nn_model_name"])
        p = Dict()
        nn_solver = NNForward(nn_func, config["Delta_t"], config["Nc"], p)
        coarse_solve = u0 -> nn_solver(u0)
    end

    return fine_solve, coarse_solve
end

function get_solvers(config::Dict{String, Any}, prob::AutonomousODESystem)
    solver_kwargs = get_solver_kwargs(config)
    fine_solve = u0 -> ode_solve(
        (du, u, p, t) -> compute_du!(prob, du, u), 
        METHODS[config["fine_method"]],
        u0,
        0.,
        config["Delta_t"],
        config["Nf"], 
        false,
        target_type=Float64x4,
        solver_kwargs...
    )

    if ~haskey(config, "nn_ckpt_path")
        coarse_solve = u0 -> ode_solve(
            (du, u, p, t) -> compute_du!(prob, du, u), 
            METHODS[config["coarse_method"]],
            u0,
            0.,
            config["Delta_t"],
            config["Nc"], 
            false,
            target_type=config["use_float64x4"] ? Float64x4 : Float64,
            solver_kwargs...
        )
    else
        nn_func = load_nn(config["nn_ckpt_path"], config["nn_model_name"])
        p = Dict("epsilon"=>prob.epsilon)
        nn_solver = NNForward(nn_func, config["Delta_t"], config["Nc"], p)
        coarse_solve = u0 -> nn_solver(u0)
    end

    return fine_solve, coarse_solve
end

function main()
    parsed_args = parse_commandline()
    config = TOML.parsefile(parsed_args["toml_config"])
    output_dir = parsed_args["output_dir"]

    # create logger 
    logger = get_default_logger(output_dir)
    global_logger(logger)

    # save a copy of config file in the output directory 
    save_toml_config(output_dir, config)

    @info "Using $(nworkers()) workers ..."

    @info "Instantiating problem ..."
    prob = get_problem(config["problem"])
    @info "Problem: $(prob)"

    @info "Instantiating integrators ..."
    fine_solve, coarse_solve = get_solvers(config["integration"], prob)

    @info "Generating initial state ..."
    if prob isa AutonomousODESystem
        u0 = initial_condition(prob, config["integration"]["use_float64x4"] ? Float64x4 : Float64)
        @info "u0: $u0"
    elseif prob isa SeparableHamiltonianSystem
        v0, x0 = initial_condition(prob, config["integration"]["use_float64x4"] ? Float64x4 : Float64)
        u0 = (v0, x0)
        @info "v0: $v0 \nx0: $x0\nH0: $(compute_H(prob, v0, x0))\nK0: $(compute_K(prob, v0))\nU0: $(compute_U(prob, x0))"
    end

    @info "Running parareal!"
    alg_name = config["algorithm"]["_name_"]
    alg_kwargs = to_kwargs(config["algorithm"])
    if alg_name == "plain"
        elapsed_time = @elapsed Parareal.plain(
            u0, fine_solve, coarse_solve, config["integration"]["N"], config["integration"]["niters"];
            output_dir=output_dir, alg_kwargs...)
    elseif alg_name == "procrustes"
        elapsed_time = @elapsed Parareal.procrustes(
            u0, fine_solve, coarse_solve, config["integration"]["N"], config["integration"]["niters"],
            u -> embed_state_procrustes(prob, u), 
            (u, corrector) -> align_state_procrustes(prob, u, corrector); 
            output_dir=output_dir, alg_kwargs...)
    elseif alg_name == "interpolative"
        elapsed_time = @elapsed Parareal.interpolative(
            u0, fine_solve, coarse_solve, config["integration"]["N"], config["integration"]["niters"], 
            u -> embed_state_interpolative(prob, u), 
            (u, corrector) -> align_state_interpolative(prob, u, corrector); 
            output_dir=output_dir, alg_kwargs...)
    elseif alg_name == "sequential"
        elapsed_time = @elapsed Parareal.plain(
            u0, fine_solve, fine_solve, config["integration"]["N"], 0;
            output_dir=output_dir, alg_kwargs...)
    end
    @info "Done running parareal. Elapsed time = $elapsed_time seconds."
end

main()
