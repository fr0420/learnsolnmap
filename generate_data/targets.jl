"""
Generate targets 
"""

const problem = ARGS[1];
const Dt = parse(Float64, ARGS[2]);
const Nf = parse(Int64, ARGS[3]);
const method = ARGS[4];
const input_file = ARGS[5]
const output_file = ARGS[6];
const use_float64x4 = parse(Bool, ARGS[7]);

println("problem =     ", problem)
println("Delta t =     ", Dt)
println("Nf =          ", Nf)
println("h =           ", Dt/Nf)
println("method =      ", method)
println("input_file =  ", input_file)
println("output_file = ", output_file)
println("use_float64x4 =", use_float64x4)


include("./generate_data_utils.jl")
using Distributed 
addprocs(40);
println("# workers = 40")
@everywhere using MultiFloats

if problem == "fpu"
    const OMEGA = 50.
    const param = use_float64x4 ? (Float64x4(OMEGA)^2)/2. : (OMEGA^2)/2.
    
end 

@everywhere begin 
    include("../tools/ode_solver.jl")
    include("../tools/setups/$($problem).jl")
    using ProgressMeter 
    
    phi_Dt(p, q) = ode_solve(A!, methods[$method], p, q, 0.0, $Dt, $Nf, false, $param)
end



println("\nImporting initial states ...")

P_init, Q_init = read_csv(input_file, use_float64x4 ? Float64x4 : Float64)

println(size(P_init)[2], " states imported.")


println("\nComputing final states ...")

res = @showprogress pmap(phi_Dt, eachslice(P_init, dims=2), eachslice(Q_init, dims=2))
P_final = hcat([p for (p, q) in res]...)
Q_final = hcat([q for (p, q) in res]...)

println("Done.")

save_csv(output_file, P_final, Q_final, use_float64x4 ? Float64x4 : Float64)

