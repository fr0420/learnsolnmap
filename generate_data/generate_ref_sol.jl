include("./generate_data_utils.jl")

const problem = ARGS[1];
const N = parse(Int64, ARGS[2]);
const Dt = parse(Float64, ARGS[3]);
const Nf = parse(Int64, ARGS[4]);
const method = ARGS[5];
const output_dir = ARGS[6];
const use_bigfloat = parse(Bool, ARGS[7]);


println("problem =    ", problem)
println("N =          ", N)
println("Delta t =    ", Dt)
println("Nf =         ", Nf)
println("h =          ", Dt/Nf)
println("method =     ", method)
println("output_dir = ", output_dir)
println("use_bigfloat =", use_bigfloat)


using DifferentialEquations
using ProgressMeter

function ode_solve(A::Function, 
            method::OrdinaryDiffEqAlgorithm, 
            p0::AbstractArray{T, 1}, 
            q0::AbstractArray{T, 1}, 
            t0::Float64, 
            H::Float64,
            nsteps::Integer,
            retfull::Bool) where T<:AbstractFloat
    
    t0 = convert(T, t0)
    H = convert(T, H)
    
    h = H/nsteps 
    prob = SecondOrderODEProblem((du,u,p,t)->A(u), p0, q0, (t0, t0+H));
    sol = solve(prob, method, tstops=t0:h:(t0+H), adaptive=false);
    if retfull 
        P = hcat([u.x[1] for u in sol.u]...)   
        Q = hcat([u.x[2] for u in sol.u]...)
        return P, Q
    else 
        p = sol[end].x[1]
        q = sol[end].x[2]
        return p, q
    end
end


methods = Dict(
    "VelocityVerlet"=>VelocityVerlet(),
    "CalvoSanz4"=>CalvoSanz4(),
    "KahanLi6"=>KahanLi6(),
    "KahanLi8"=>KahanLi8(),
)


include("./setups/$problem.jl")

if problem == "fpu"
    kwargs = Dict(:omega => use_bigfloat ? BigFloat(300.) : 300.)
end 

phi_Dt(p, q) = ode_solve(q->A(q; kwargs...), methods[method], p, q, 0.0, Dt, Nf, false)

function phi_Dt_N(
        p0::AbstractArray{T, 1}, 
        q0::AbstractArray{T, 1}, 
        N::Integer) where T<:AbstractFloat
    
    progressbar = Progress(N)
    
    d = length(p0)
    P = zeros(T, d, N+1)
    Q = zeros(T, d, N+1)
    P[:, 1] = p0
    Q[:, 1] = q0
    for n in 1:N
        P[:, n+1], Q[:, n+1] = phi_Dt(P[:, n], Q[:, n])
        next!(progressbar)
    end

    return P, Q
end 


p0, q0 = initial_condition(; kwargs...)
# p0 /= sqrt(2)

println("p0: ", p0)
println("q0: ", q0)
println("H0: ", compute_H(p0, q0; kwargs...))
println("K0: ", compute_K(p0))
println("U0: ", compute_U(q0; kwargs...))

println("\nIntegrate forward N=$N steps ...")

P, Q = phi_Dt_N(p0, q0, N)

println("Done.")

save_csv("$output_dir/true_sol.csv", P, Q, BigFloat)
save_csv("$output_dir/true_sol_f64.csv", P, Q, Float64)
