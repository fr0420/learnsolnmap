include("./generate_data_utils.jl")

const problem = ARGS[1];
const N = parse(Int, ARGS[2]);
const Dt = parse(Float64, ARGS[3]);
const h = parse(Float64, ARGS[4]);
const output_dir = ARGS[5];

println("problem =    ", problem)
println("N =          ", N)
println("Delta t =    ", Dt)
println("h =          ", h)
println("output_dir = ", output_dir)


using DifferentialEquations
    
include("./setups/$problem.jl")

function ode_solve(A, method, p0, q0, t0, H, nsteps, retfull)
        
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

phi_Dt(p, q) = ode_solve(A, CalvoSanz4(), p, q, 0.0, Dt, round(Int, Dt/h), false)
    
function phi_Dt_N(
        p0::AbstractArray{T, 1}, 
        q0::AbstractArray{T, 1}, 
        N::Integer) where T<:AbstractFloat
    
    d = length(p0)
    P = zeros(T, d, N+1)
    Q = zeros(T, d, N+1)
    P[:, 1] = p0
    Q[:, 1] = q0
    for n in 1:N
        P[:, n+1], Q[:, n+1] = phi_Dt(P[:, n], Q[:, n])
    end

    return P, Q
end 

if problem == "lennardjones"
    p0 = v0
    q0 = x0
end 

# p0 /= sqrt(2)

println("p0: $p0")
println("q0: $q0")

println("\nIntegrate forward N=$N steps ...")

P, Q = phi_Dt_N(p0, q0, N)

println("Done.")

save("$output_dir/true_sol.csv", P, Q);