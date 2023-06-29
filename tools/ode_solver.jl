using DifferentialEquations
using MultiFloats 


methods = Dict(
    "VelocityVerlet"=>VelocityVerlet(), 
    "CalvoSanz4"=>CalvoSanz4(),
    "McAte5"=>McAte5(),
    "KahanLi6"=>KahanLi6(),
    "KahanLi8"=>KahanLi8(),
    "DPRKN12"=>DPRKN12(),
)


function ode_solve(
        A::Function, 
        method::OrdinaryDiffEqAlgorithm, 
        p0::AbstractArray{T, 1}, 
        q0::AbstractArray{T, 1}, 
        t0::Float64, 
        H::Float64,
        nsteps::Integer,
        retfull::Bool,
        param::Any) where T<:AbstractFloat
        
    t0 = convert(T, t0)
    H = convert(T, H)
        
    h = H/nsteps 
    prob = SecondOrderODEProblem(A, p0, q0, (t0, t0+H), param);
    if retfull 
        sol = solve(prob, method, tstops=t0:h:(t0+H), adaptive=false, save_everystep=true);
        P = hcat([u.x[1] for u in sol.u]...)   
        Q = hcat([u.x[2] for u in sol.u]...)
        return P, Q
    else 
        sol = solve(prob, method, tstops=t0:h:(t0+H), adaptive=false, save_everystep=false);
        p = sol[end].x[1]
        q = sol[end].x[2]
        return p, q
    end
end

    
Base.round(x::MultiFloat{Float64, 4}, y::RoundingMode) = MultiFloat{Float64, 4}(Base.round(Float64(x),y))
Base.trunc(x::Type{Int64}, y::MultiFloat{Float64, 4}) = Base.trunc(x::Type{Int64}, Float64(y))
