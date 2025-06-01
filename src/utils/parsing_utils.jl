"""
Config parsing utils.
"""

function to_kwargs(config::Dict{String, Any})
    kwargs = Dict{Symbol, Any}()
    for (k, v) in config
        if k != "_name_"
            kwargs[Symbol(k)] = v
        end
    end
    return kwargs
end

function get_problem(config::Dict{String, Any})
    name = config["_name_"]
    kwargs = to_kwargs(config)
    
    prob = Dict(
        "fpu"=>FPU, 
        "1body"=>OneBodyKepler, 
        "2body"=>TwoBodyKepler,
        "3body"=>ThreeBody,
        "3body-2d"=>ThreeBody2D,
        "argoncrystal"=>ArgonCrystal,
        "nbody"=>NBody,
        "nco"=>NonlinearCoupledOscillators,
        "alphaparticle"=>AlphaParticle)[name](; kwargs...)

    return prob
end

function get_solver_kwargs(config::Dict{String, Any})
    solver_kwargs = Dict{Symbol, Any}()
    solver_params = ["abstol", "reltol", "maxiters", "adaptive"]

    for param in solver_params
        if haskey(config, param)
            solver_kwargs[Symbol(param)] = config[param]
        end
    end
    return solver_kwargs
end
