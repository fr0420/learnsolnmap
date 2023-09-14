include("./generate_data_utils.jl")
using Distributed 
addprocs(40);
println("# workers = 40")

const problem = ARGS[1];
const Nlevelsets = parse(Int, ARGS[2]);
const Ntraj = parse(Int, ARGS[3]);
const K = parse(Int, ARGS[4]);
const sigma = parse(Float64, ARGS[5]);
const dt = parse(Float64, ARGS[6]);
const h = parse(Float64, ARGS[7]);
const output_dir = ARGS[8];

println("problem =    ", problem)
println("Nlevelsets = ", Nlevelsets)
println("Ntraj =      ", Ntraj)
println("K =          ", K)
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
    
    phi_dt(p, q) = ode_solve(A, CalvoSanz4(), p, q, 0.0, $dt, round(Int, $dt/$h), false)
    
    function phi_dt_K(
            p0::AbstractArray{T, 1}, 
            q0::AbstractArray{T, 1}, 
            K::Integer) where T<:AbstractFloat
    
        d = length(p0)
        P = zeros(T, d, K+1)
        Q = zeros(T, d, K+1)
        P[:, 1] = p0
        Q[:, 1] = q0
        for k in 1:K
            P[:, k+1], Q[:, k+1] = phi_dt(P[:, k], Q[:, k])
        end

        return P, Q
    end 
end



println("\nSampling initial states ...")

if problem == "lennardjones"
    p0 = v0
    q0 = x0
end 

kinetic_energies = zeros(Nlevelsets);
count = 1;
while count <= Nlevelsets
    ke = Normal(K0, abs(sigma*H0));
    if ke >= 0.
        kinetic_energies[count] = ke
        global count += 1
    end
end

radius = sqrt.(2*repeat(kinetic_energies, Ntraj));
P0 = nSphereSampling(length(p0), Ntraj*Nlevelsets) .* radius';
# P0 = nShellSampling(length(p0), Ntraj, 1e-2) * norm(p0);
res = pmap(p->phi_dt_K(p, q0, K), eachslice(P0, dims=2));

P_init = hcat([P for (P, Q) in res]...);
Q_init = hcat([Q for (P, Q) in res]...);

println("Done.")

save("$output_dir/U0.csv", P_init, Q_init);
