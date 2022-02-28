include("./generate_data_utils.jl")
using Distributed 
addprocs(40);
println("# workers = 40")

const problem = ARGS[1];
const Dt = parse(Float64, ARGS[2]);
const h = parse(Float64, ARGS[3]);
const input_file = ARGS[4]
const output_file = ARGS[5];

println("problem =     ", problem)
println("Delta t =     ", Dt)
println("h =           ", h)
println("input_file =  ", input_file)
println("output_file = ", output_file)


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
    
    const Dt = $Dt;
    const h = $h;
    
    phi_Dt(p, q) = ode_solve(A, CalvoSanz4(), p, q, 0.0, Dt, round(Int, Dt/h), false)
    
end



println("\nImporting initial states ...")

df_init = DataFrame(CSV.File(input_file));
dim = div(size(df_init)[2], 2);
P_init = Matrix(df_init[:, 1:dim])'; 
Q_init = Matrix(df_init[:, dim+1:end])';

println("Done.")


println("\nComputing final states ...")

res = pmap(phi_Dt, eachslice(P_init, dims=2), eachslice(Q_init, dims=2));
P_final = hcat([p for (p, q) in res]...);
Q_final = hcat([q for (p, q) in res]...);

println("Done.")

save(output_file, P_final, Q_final);

