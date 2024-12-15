"""
Wrapper function for solving second-order ODEs.

Available solvers: https://docs.sciml.ai/DiffEqDocs/latest/solvers/dynamical_solve/
"""

using DifferentialEquations
using MultiFloats 
MultiFloats.use_bigfloat_transcendentals()


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
    "Tsit5"=>Tsit5(),
    # For stiff problems
    "ImplicitMidpoint"=>ImplicitMidpoint(),
    "Rodas5"=>Rodas5(),
    "AutoVern7(Rodas5)"=>AutoVern7(Rodas5())
)


"""
Solve a second-order ODE system.

Arguments:
    A: The system function
    method: The ODE solver method
    v0: Initial velocity
    x0: Initial position
    t0: Initial time
    H: Total time to integrate
    nsteps: Number of steps
    retfull: Whether to return the full history
    param: Additional parameters to pass to the system function
    solver_kwargs: Additional keyword arguments to pass to the solver

Returns:
    If retfull=true: Tuple of velocity and position histories
    If retfull=false: Final velocity and position
"""
function ode_solve(
    A::Function, 
    method::OrdinaryDiffEqAlgorithm, 
    v0::AbstractArray{T, 1}, 
    x0::AbstractArray{T, 1}, 
    t0::Float64, 
    H::Float64,
    nsteps::Integer,
    retfull::Bool;
    param::Any=nothing,
    solver_kwargs...
) where T<:AbstractFloat

    t0 = convert(T, t0)
    H = convert(T, H)
    
    prob = SecondOrderODEProblem(A, v0, x0, (t0, t0+H), param)
    h = H/nsteps 
    tstops = t0 .+ (0:nsteps) * h
    
    default_kwargs = (
        dt = h,
        # tstops = tstops,
        abstol = 1e-14,
        reltol = 1e-14,
        maxiters = 1e5
    )
    merged_kwargs = merge(default_kwargs, solver_kwargs)

    if retfull
        sol = solve(prob, method; merged_kwargs..., save_everystep=true)
        V = hcat([u.x[1] for u in sol.u]...)   
        X = hcat([u.x[2] for u in sol.u]...)
        return V, X
    else 
        sol = solve(prob, method; merged_kwargs..., save_everystep=false)
        return sol[end].x[1], sol[end].x[2]
    end
end


# Extension methods for MultiFloats compatibility
Base.round(x::MultiFloat{Float64, N}, y::RoundingMode) where {N} = MultiFloat{Float64, N}(Base.round(BigFloat(x),y))
Base.trunc(x::Type{Integer}, y::MultiFloat{Float64, N}) where {N} = Base.trunc(x::Type{Integer}, BigFloat(y))
Base.:^(x::MultiFloat{Float64, N}, y::AbstractFloat) where {N} = MultiFloat{Float64, N}((BigFloat(x)^y))