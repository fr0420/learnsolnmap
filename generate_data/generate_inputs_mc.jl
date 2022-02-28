include("./generate_data_utils.jl")
using Distributed 
addprocs(40);
println("# workers = 40")

const problem = ARGS[1];
const Nsamples = parse(Int, ARGS[2]);
const sigma = parse(Float64, ARGS[3]);
const dt = parse(Float64, ARGS[4]);
const h = parse(Float64, ARGS[5]);
const output_dir = ARGS[6];

println("problem =    ", problem)
println("Nsamples =   ", Nsamples)
println("sigma =      ", sigma)
println("dt =         ", dt)
println("h =          ", h)
println("output_dir = ", output_dir)


@everywhere begin 
    using DifferentialEquations
    
    include("./setups/$($problem).jl")

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
    
    const dt = $dt;
    const h = $h;
    
    phi_dt(p, q) = ode_solve(A, CalvoSanz4(), p, q, 0.0, dt, round(Int, dt/h), false)
end

function sample_p_new(p::AbstractArray{T, 1}, q::AbstractArray{T, 1}) where T<:AbstractFloat
    U = compute_U(q)
    K_new = Normal(H0-U, abs(sigma*H0))
    while K_new <= 0
        K_new = Normal(H0-U, abs(sigma*H0))
    end
    p_new = nSphereSampling(length(p)) * sqrt(2*K_new)
    return p_new
end


println("\nSampling initial states ...")

if problem == "lennardjones"
    p0 = v0
    q0 = x0
end 


P = zeros(length(p0), Nsamples)
Q = zeros(length(q0), Nsamples)
P[:, 1] = p0
Q[:, 1] = q0

for i in 1:2:(Nsamples-2)
    P[:, i+1], Q[:, i+1] = phi_dt(P[:, i], Q[:, i])
    P[:, i+2] = sample_p_new(P[:, i+1], Q[:, i+1])
    Q[:, i+2] = Q[:, i+1]
end 



println("Done.")

save("$output_dir/U0.csv", P, Q);
