"""
Run ODE solver. 
"""

include("../utils/parsing_utils.jl")
include("../utils/logging_utils.jl")
include("../utils/saving_utils.jl")
include("../utils/ode_solver.jl")
include("../problems/problems.jl")

using ArgParse
using TOML
using ProgressMeter 


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

function get_solver(config::Dict{String, Any}, prob::SeparableHamiltonianSystem)
    solver_kwargs = get_solver_kwargs(config)
    phi_dt = u0 -> ode_solve(
        (ddx, dx, x, p, t) -> compute_ddx!(prob, ddx, dx, x), 
        METHODS[config["method"]], u0, 0., config["dt"], config["nsteps"], config["retfull"]; solver_kwargs...)
    return phi_dt
end

function get_solver(config::Dict{String, Any}, prob::AutonomousODESystem)
    solver_kwargs = get_solver_kwargs(config)
    phi_dt = u0 -> ode_solve(
        (du, u, p, t) -> compute_du!(prob, du, u), 
        METHODS[config["method"]], u0, 0., config["dt"], config["nsteps"], config["retfull"]; solver_kwargs...)
    return phi_dt
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

    @info "Instantiating problem ..."
    prob = get_problem(config["problem"])
    @info "Problem: $(prob)"
    
    @info "Instantiating integrator ..."
    fine_solve = get_solver(config["integration"], prob)

    @info "Generating initial state ..."
    if prob isa AutonomousODESystem
        u0 = initial_condition(prob, config["integration"]["use_float64x4"] ? Float64x4 : Float64)
        @info "u0: $u0"
    elseif prob isa SeparableHamiltonianSystem
        v0, x0 = initial_condition(prob, config["integration"]["use_float64x4"] ? Float64x4 : Float64)
        u0 = (v0, x0)
        @info "v0: $v0 \nx0: $x0\nH0: $(compute_H(prob, v0, x0))\nK0: $(compute_K(prob, v0))\nU0: $(compute_U(prob, x0))"
    end

    @info "Running sequential!"
    N = config["integration"]["N"]
    dt = config["integration"]["dt"]
    retfull = config["integration"]["retfull"]
    times = Vector()
    push!(times, 0.)
    if prob isa AutonomousODESystem
        T = typeof(u0[1])
        states = Vector{AbstractArray{T, 1}}()
        push!(states, u0)
        elapsed_time = @elapsed begin
            u = copy(u0)
            @showprogress for n in 1:N
                res = fine_solve(u)
                if ~retfull
                    push!(states, res)
                    copyto!(u, res)
                else
                    append!(states, res[1][2:end])
                    append!(times, res[2][2:end] .+ (n-1)*dt)
                    copyto!(u, res[1][end])
                end
            end
        end
    elseif prob isa SeparableHamiltonianSystem
        T = typeof(v0[1])
        states = Vector{Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}}()
        push!(states, u0)

        elapsed_time = @elapsed begin
            # v, x = copy(v0), copy(x0)
            u = deepcopy(u0)
            @showprogress for n in 1:N
                res = fine_solve(u)
                if ~retfull
                    push!(states, res)
                    u = deepcopy(res)
                else
                    append!(states, res[1][2:end])
                    append!(times, res[2][2:end] .+ (n-1)*dt)
                    u = deepcopy(res[1][end])
                end
            end
        end
    end
    @info "Done. Elapsed time = $elapsed_time seconds."

    filepath = joinpath(output_dir, "u.csv")
    @info "Saving results at $filepath ..."
    save_csv(filepath, states)

    if retfull
        filepath = joinpath(output_dir, "t.csv")
        @info "Saving times at $filepath ..."
        save_csv(filepath, Dict("t" => times))
    end
end

main()
