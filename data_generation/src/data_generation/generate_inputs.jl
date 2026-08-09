"""
Generate inputs.
"""

using Distributed  # for parallel computing
addprocs(40)

include("../utils/parsing_utils.jl")
include("../utils/saving_utils.jl")
@everywhere include("../utils/logging_utils.jl")
@everywhere include("HMC.jl")
@everywhere include("../utils/ode_solver.jl")
@everywhere include("../problems/problems.jl")

using ArgParse
using TOML
@everywhere using .HMC


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

function setup_logging(output_dir)
    @everywhere begin
        logger = get_default_logger($output_dir)
        global_logger(logger)
    end
end

function load_configuration(file_path)
    TOML.parsefile(file_path)
end

function save_configuration(output_dir, config)
    save_toml_config(output_dir, config)
end

function get_solvers(config::Dict{String, Any}, prob::SeparableHamiltonianSystem)
    solver_kwargs = get_solver_kwargs(config)
    phi_dt = (v0, x0) -> ode_solve(
        (ddx, dx, x, p, t) -> compute_ddx!(prob, ddx, dx, x), 
        METHODS[config["method"]], (v0, x0), 0., config["dt"], config["nsteps"], false; solver_kwargs...)
    phi_h = (v0, x0, nsteps) -> ode_solve(
        (ddx, dx, x, p, t) -> compute_ddx!(prob, ddx, dx, x), 
        METHODS[config["method"]], (v0, x0), 0., nsteps * config["dt"]/config["nsteps"], nsteps, false; solver_kwargs...)
    mean_dt_div_h = Float64(config["nsteps"])
    return phi_dt, phi_h, mean_dt_div_h
end

function get_initial_state(prob::SeparableHamiltonianSystem, use_float64x4::Bool)
    v0, x0 = initial_condition(prob, use_float64x4 ? Float64x4 : Float64)
    H0 = compute_H(prob, v0, x0)
    @info "v0: $v0 \nx0: $x0\nH0: $H0\nK0: $(compute_K(prob, v0))\nU0: $(compute_U(prob, x0))"
    return v0, x0, H0
end

function select_transition_algorithm(config, prob, phi_dt, phi_h, mean_dt_div_h)
    if config["_name_"] == "hmc-H0" 
        transition = (v, x) -> hmc_H0_transition(v, x, mass(prob),
            (v, x) -> compute_H(prob, v, x), phi_dt;
            nsteps=config["n_steps_per_trans"],
            with_rejection=config["with_rejection"]
        )
    elseif config["_name_"] == "rhmc-H0"
        transition = (v, x) -> rhmc_H0_transition(v, x, mass(prob),
            (v, x) -> compute_H(prob, v, x), phi_h, mean_dt_div_h;
            nsteps=config["n_steps_per_trans"],
            with_rejection=config["with_rejection"]
        )
    elseif config["_name_"] == "hmc"
        transition = (v, x) -> hmc_transition(v, x, mass(prob),
            (v, x) -> compute_H(prob, v, x), phi_dt;
            nsteps=config["n_steps_per_trans"],
            beta=config["beta"],
            with_rejection=config["with_rejection"]
        )
    elseif config["_name_"] == "rhmc"
        transition = (v, x) -> rhmc_transition(v, x, mass(prob),
            (v, x) -> compute_H(prob, v, x), phi_h, mean_dt_div_h;
            nsteps=config["n_steps_per_trans"],
            beta=config["beta"],
            with_rejection=config["with_rejection"]
        )
    elseif config["_name_"] == "trajensemble"
        transition = phi_dt
    else
        error("Unknown algorithm name: $(config["_name_"])")
    end
    return transition
end

function run_sampling(config, transition, prob, v0, x0)
    if config["_name_"] == "hmc-H0" || config["_name_"] == "rhmc-H0" || config["_name_"] == "trajensemble"
        init_conditions = sample_initial_conditions(v0, x0, mass(prob); 
            num_samples=config["n_chains"], 
            epsilon=config["epsilon"]
        )
        # init_conditions = sample_initial_conditions_3body(v0, x0, mass(prob); 
        #     num_samples=config["n_chains"], 
        #     epsilon=config["epsilon"]
        # )
        # init_conditions = sample_initial_conditions_3body_equilateral(eltype(v0); 
        #     num_samples=config["n_chains"], 
        #     nu=config["epsilon"]
        # )
        elapsed_time = @elapsed samples, total_rejections = chain_ensemble(init_conditions, transition; 
            num_transitions=config["n_trans_per_chain"]
        )
    elseif config["_name_"] == "hmc" || config["_name_"] == "rhmc"
        elapsed_time = @elapsed samples, total_rejections = chain_ensemble(v0, x0, transition; 
            num_chains=config["n_chains"],
            num_transitions=config["n_trans_per_chain"]
        )
    end
    return elapsed_time, samples, total_rejections
end

function main()
    parsed_args = parse_commandline()
    config = load_configuration(parsed_args["toml_config"])
    output_dir = parsed_args["output_dir"]
    
    setup_logging(output_dir)
    save_configuration(output_dir, config)

    @info "Using $(nworkers()) workers ..."

    @info "Instantiating problem ..."
    prob = get_problem(config["problem"])
    @info "Problem: $(prob)"

    @info "Instantiating integrator ..."
    phi_dt, phi_h, mean_dt_div_h = get_solvers(config["integration"], prob)

    @info "Generating initial state ..."
    v0, x0, H0 = get_initial_state(prob, config["integration"]["use_float64x4"])

    @info "Sampling chains!"
    transition = select_transition_algorithm(config["algorithm"], prob, phi_dt, phi_h, mean_dt_div_h)
    elapsed_time, samples, total_rejections = run_sampling(config["algorithm"], transition, prob, v0, x0)

    @info "Done generating inputs. Elapsed time = $elapsed_time seconds. Number of samples = $(length(samples)). Total rejections = $total_rejections."
    
    filepath = joinpath(output_dir, "U0.csv")
    @info "Saving results at $filepath ..."
    save_csv(filepath, samples)
end

main()