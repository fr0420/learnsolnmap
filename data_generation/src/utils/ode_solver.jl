"""
Wrapper functions for solving ODEs.

Available solvers: 
    https://docs.sciml.ai/DiffEqDocs/latest/solvers/ode_solve/
    https://docs.sciml.ai/DiffEqDocs/latest/solvers/dynamical_solve/
"""

using DifferentialEquations
using MultiFloats 
MultiFloats.use_bigfloat_transcendentals()


# Extension methods for MultiFloats compatibility
Base.round(x::MultiFloat{Float64, N}, y::RoundingMode) where {N} = MultiFloat{Float64, N}(Base.round(BigFloat(x),y))
Base.trunc(x::Type{T}, y::MultiFloat{Float64, N}) where {T <: Integer, N} = Base.trunc(x, BigFloat(y))
Base.:^(x::MultiFloat{Float64, N}, y::AbstractFloat) where {N} = MultiFloat{Float64, N}((BigFloat(x)^y))


METHODS = Dict(
    # Symplectic integrators
    "SymplecticEuler"=>SymplecticEuler(),
    "VelocityVerlet"=>VelocityVerlet(), 
    "CalvoSanz4"=>CalvoSanz4(),
    "McAte5"=>McAte5(),
    "KahanLi6"=>KahanLi6(),
    "KahanLi8"=>KahanLi8(),
    # Runge-Kutta-Nystrom integrators
    "DPRKN4"=>DPRKN4(),
    "ERKN5"=>ERKN5(),
    "DPRKN5"=>DPRKN5(),
    "DPRKN12"=>DPRKN12(),
    # Explicit Runge-Kutta integrators
    "Euler"=>Euler(),
    "Midpoint"=>Midpoint(),
    "DP5"=>DP5(),
    "Tsit5"=>Tsit5(),
    "Vern7"=>Vern7(),
    "Vern9"=>Vern9(),
    # For stiff problems
    "ImplicitMidpoint"=>ImplicitMidpoint(),
    "Rodas5"=>Rodas5(),
    "AutoVern7(Rodas5)"=>AutoVern7(Rodas5())
)


# Helper function to prepare solver keyword arguments
function prepare_solver_kwargs(h::T, solver_kwargs) where T<:AbstractFloat
    default_kwargs = (
        dt = h,
        # tstops = tstops,
        abstol = 1e-14,
        reltol = 1e-14,
        maxiters = 1e7,
        adaptive = false
    )
    merge(default_kwargs, solver_kwargs)
end

# Helper function to run the solver and return results
function run_solver(prob, method, h, retfull, solver_kwargs)
    kwargs = prepare_solver_kwargs(h, solver_kwargs)
    if retfull
        sol = solve(prob, method; kwargs..., save_everystep=true)
        return sol
    else
        sol = solve(prob, method; kwargs..., save_everystep=false)
        return sol[end]
    end
end


"""
    ode_solve(A::Function, method::OrdinaryDiffEqAlgorithm, u0::AbstractArray{T, 1}, 
              t0::Float64, H::Float64, nsteps::Integer, retfull::Bool;
              target_type::Union{Nothing, Type}=nothing, solver_kwargs...) where T<:AbstractFloat

Solves a first-order ODE using the specified numerical method.

Arguments:
- `A`: Function representing the ODE's right-hand side.
- `method`: A numerical method of type `OrdinaryDiffEqAlgorithm` used for solving the ODE.
- `u0`: Initial state vector.
- `t0`: Initial time.
- `H`: Total time span; the final time is `t0 + H`.
- `nsteps`: Number of steps into which the time span is divided.
- `retfull`: Boolean flag that, if `true`, returns the full history (all time points and corresponding states); if `false`, returns only the final state.
- `target_type` (optional): If provided and different from the type `T`, the initial conditions will be converted to this type before solving.
- `solver_kwargs...`: Additional keyword arguments passed to the solver.

Returns:
- If `retfull` is `true`: a tuple `(us, ts)`, where `us` is an array of solution states at each time step and `ts` is an array of time points.
- If `retfull` is `false`: the final state.
"""
function ode_solve(
    A::Function, 
    method::OrdinaryDiffEqAlgorithm, 
    u0::AbstractArray{T, 1}, 
    t0::Float64, 
    H::Float64,
    nsteps::Integer,
    retfull::Bool;
    target_type::Union{Nothing, Type} = nothing,
    solver_kwargs...
) where T<:AbstractFloat
    
    # If a target type is provided and is different than T, convert the initial conditions
    if target_type !== nothing && T != target_type
        u0 = convert.(target_type, u0)
    end

    # Define the ODE problem
    t0 = convert(eltype(u0), t0)
    H = convert(eltype(u0), H)
    prob = ODEProblem(A, u0, (t0, t0 + H), nothing)

    # Run the solver
    h = H / nsteps
    sol = run_solver(prob, method, h, retfull, solver_kwargs)

    # If target type conversion was done at input, convert back the output to original type T
    if target_type !== nothing && T != target_type
        if retfull
            return [convert.(T, u) for u in sol.u], sol.t
        else
            return convert.(T, sol)
        end
    else
        return retfull ? (sol.u, sol.t) : sol
    end
end


"""
    ode_solve(A::Function, method::OrdinaryDiffEqAlgorithm, 
              u0::Tuple{<:AbstractArray{T, 1}, <:AbstractArray{T, 1}},
              t0::Float64, H::Float64, nsteps::Integer, retfull::Bool;
              target_type::Union{Nothing, Type}=nothing, solver_kwargs...) where T<:AbstractFloat

Solves a second-order ODE using the specified numerical method.

Arguments:
- `A`: Function representing the second-order ODE's right-hand side.
- `method`: A numerical method of type `OrdinaryDiffEqAlgorithm` used for solving the ODE.
- `u0`: A tuple `(v0, x0)` where `v0` is the initial velocity vector and `x0` is the initial position vector.
- `t0`: Initial time.
- `H`: Total time span; the final time is `t0 + H`.
- `nsteps`: Number of steps into which the time span is divided.
- `retfull`: Boolean flag that, if `true`, returns the full history (all time points and corresponding states); if `false`, returns only the final state.
- `target_type` (optional): If provided and different from the type `T`, the initial conditions will be converted to this type before solving.
- `solver_kwargs...`: Additional keyword arguments passed to the solver.

Returns:
- If `retfull` is `true`: a tuple `(us, ts)`, where `us` is an array of tuples `(v, x)` representing the states at each time step, and `ts` is an array of time points.
- If `retfull` is `false`: a tuple `(v, x)` representing the final state.
"""
function ode_solve(
    A::Function, 
    method::OrdinaryDiffEqAlgorithm, 
    u0::Tuple{<:AbstractArray{T, 1}, <:AbstractArray{T, 1}},
    t0::Float64, 
    H::Float64,
    nsteps::Integer,
    retfull::Bool;
    target_type::Union{Nothing, Type} = nothing,
    solver_kwargs...
) where T<:AbstractFloat
    
    # Unpack the initial conditions
    v0, x0 = u0

    # If a target type is provided and is different than T, convert the initial conditions
    if target_type !== nothing && T != target_type
        v0 = convert.(target_type, v0)
        x0 = convert.(target_type, x0)
    end

    # Define the ODE problem
    t0 = convert(eltype(v0), t0)
    H = convert(eltype(v0), H)
    prob = SecondOrderODEProblem(A, v0, x0, (t0, t0 + H), nothing)

    # Run the solver
    h = H / nsteps
    sol = run_solver(prob, method, h, retfull, solver_kwargs)
    
    if retfull
        # Each element of sol.u is expected to have field `x` as a tuple (v, x)
        if target_type !== nothing && T != target_type
            return [(convert.(T, u.x[1]), convert.(T, u.x[2])) for u in sol.u], sol.t
        else
            return [u.x for u in sol.u], sol.t
        end
    else
        if target_type !== nothing && T != target_type
            return (convert.(T, sol.x[1]), convert.(T, sol.x[2]))
        else
            return sol.x
        end
    end
end
