"""
Benchmark long time stability and accuracy of an ode solver.
"""

include("../../src/utils/ode_solver.jl")
include("../../src/utils/saving_utils.jl")
include("../../src/problems/problems.jl")

using ArgParse
using Plots, LaTeXStrings
using Printf
using ProgressMeter


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "problem"
            help = "ode problem name"
            arg_type = String
            required = true
        "Dt"
            help = "time interval"
            arg_type = Float64
            required = true
        "N"
            help = "number of intervals"
            arg_type = Int
            required = true
        "method"
            help = "solver method"
            arg_type = String
            required = true
        "--nsteps"
            help = "number of steps per interval"
            arg_type = Int
            default = 2^7
        "--ref_method"
            help = "reference solver method"
            arg_type = String
            default = "DPRKN12"
        "--ref_nsteps"
            help = "number of steps per interval for reference solver"
            arg_type = Int
            default = 2^10
        "--plot"
            help = "make plots"
            action = :store_true
        "--output_dir"
            help = "output directory"
            arg_type = String
            default = "."
    end

    return parse_args(s)
end


function phi_Dt_N(
    v0::AbstractArray{T, 1}, 
    x0::AbstractArray{T, 1},
    phi_Dt::Function, 
    N::Integer) where T<:AbstractFloat
    
    d = length(v0)
    V = zeros(T, d, N+1)
    X = zeros(T, d, N+1)
    V[:, 1] = v0
    X[:, 1] = x0
        
    @showprogress for n in 1:N
        V[:, n+1], X[:, n+1] = phi_Dt(V[:, n], X[:, n])
    end

    return V, X
end


function plot_errors(sol1_filepath, sol2_filepath, compute_err;
    ylabel="error", title="", filepath="./error.png", ymin=1e-20, ymax=1e1)
    
    V1, X1 = read_csv(sol1_filepath, Float64x4)
    V2, X2 = read_csv(sol2_filepath, Float64x4)
    
    N = size(V1, 2)-1

    errors = []
    for n in 2:N+1
        err = compute_err((V1[:, n], X1[:, n]), (V2[:, n], X2[:, n]))
        push!(errors, err)
    end

    plot(dpi=300)
    plot!((1:N), errors[1:N], label="")
    plot!(ylim=[ymin, ymax])
    plot!(xaxis=:log, yaxis=:log)
    plot!(xlabel="n", ylabel=ylabel)
    plot!(title=title)
    plot!(legend=:bottomright)
    plot!(margin=10Plots.mm)

    savefig(filepath)
    println("Saved figure to $filepath")
end


function plot_errors_against_ref(sol_filepath_list, ref_sol_filepath, compute_err, label_list;
    ylabel="error", title="", filepath="./error.png", ymin=1e-20, ymax=1e1, legend=:bottomright)
    V_ref, X_ref = read_csv(ref_sol_filepath, Float64x4)
    N = size(V_ref, 2)-1

    plot(dpi=300)
    ls_list = [:solid, :dash, :dot]
    for (sol_filepath, label, ls) in zip(sol_filepath_list, label_list, ls_list)
        V, X = read_csv(sol_filepath, Float64x4)
        errors = []
        for n in 2:N+1
            err = compute_err((V[:, n], X[:, n]), (V_ref[:, n], X_ref[:, n]))
            push!(errors, err)
        end
        plot!((1:N), errors[1:N], seriescolor=:blue, linestyle=ls, label=label)
    end
    plot!(ylim=[ymin, ymax])
    plot!(xaxis=:log, yaxis=:log)
    plot!(xlabel="n", ylabel=ylabel)
    plot!(title=title)
    plot!(legend=legend)
    plot!(margin=10Plots.mm)

    savefig(filepath)
    println("Saved figure to $filepath")
end


function main()
    parsed_args = parse_commandline()
    
    problem = parsed_args["problem"]
    Dt = parsed_args["Dt"]
    N = parsed_args["N"]
    method = parsed_args["method"]
    nsteps = parsed_args["nsteps"]
    ref_method = parsed_args["ref_method"]
    ref_nsteps = parsed_args["ref_nsteps"]
    output_dir = parsed_args["output_dir"]

    println("problem = $problem")
    println("Dt = $Dt")
    println("output_dir = $output_dir")
    println("method = $method")
    println("nsteps = $nsteps")
    println("ref_method = $ref_method")
    println("ref_nsteps = $ref_nsteps")
    
    # set problem parameters
    if problem == "fpu"
        prob = FPU(; omega=300.)
    elseif problem == "kepler-1body"
        prob = OneBodyKepler(; ecc=0.5)
    elseif problem == "kepler-2body"
        prob = TwoBodyKepler(; g12=1e-5, ecc1=0.4, ecc2=0.5)
    elseif problem == "nbody"
        prob = NBody()
    elseif problem == "argoncrystal"
        prob = ArgonCrystal()
    elseif problem == "nco"
        prob = NonlinearCoupledOscillators(; epsilon=0.01)
    end 

    # generate initial state
    v0, x0 = initial_condition(prob, Float64)
    v0_f128, x0_f128 = initial_condition(prob, Float64x2)
    v0_f256, x0_f256 = initial_condition(prob, Float64x4)

    println("\nInitial condition:")
    println("v0: ", v0)
    println("x0: ", x0)
    println("H0: ", compute_H(prob, v0, x0))
    println("K0: ", compute_K(prob, v0))
    println("U0: ", compute_U(prob, x0))

    # define solvers 
    phi_Dt = (v, x) -> ode_solve((ddx, dx, x, p, t) -> compute_ddx!(prob, ddx, dx, x), METHODS[method], v, x, 0.0, Dt, nsteps, false)
    phi_Dt_ref = (v, x) -> ode_solve((ddx, dx, x, p, t) -> compute_ddx!(prob, ddx, dx, x), METHODS[ref_method], v, x, 0.0, Dt, ref_nsteps, false)
    
    # compute solutions
    filepath_f64 = "$output_dir/$method/float64/N=$(N)_Dt=$(@sprintf("%.2e", Dt))_nsteps=$nsteps/sol.csv"
    filepath_f128 = "$output_dir/$method/float128/N=$(N)_Dt=$(@sprintf("%.2e", Dt))_nsteps=$nsteps/sol.csv"
    filepath_f256 = "$output_dir/$method/float256/N=$(N)_Dt=$(@sprintf("%.2e", Dt))_nsteps=$nsteps/sol.csv"
    filepath_ref = "$output_dir/$ref_method/float256/N=$(N)_Dt=$(@sprintf("%.2e", Dt))_nsteps=$ref_nsteps/sol.csv"

    println("Computing solution in float64 precision ...")
    if ~ispath(filepath_f64)
        V, X = phi_Dt_N(v0, x0, phi_Dt, N)
        save_csv(filepath_f64, V, X)
    else
        println("Solution file exists. Skipping integration.")
    end
    println("Solution saved at $filepath_f64")
    
    println("Computing solution in float128 precision ...")
    if ~ispath(filepath_f128)
        V, X = phi_Dt_N(v0_f128, x0_f128, phi_Dt, N)
        save_csv(filepath_f128, V, X)
    else
        println("Solution file exists. Skipping integration.")
    end
    println("Solution saved at $filepath_f128")

    println("Computing solution in float256 precision ...")
    if ~ispath(filepath_f256)
        V, X = phi_Dt_N(v0_f256, x0_f256, phi_Dt, N)
        save_csv(filepath_f256, V, X)
    else
        println("Solution file exists. Skipping integration.")
    end
    println("Solution saved at $filepath_f256")

    println("Computing reference solution in float256 precision ...")
    if ~ispath(filepath_ref)
        V, X = phi_Dt_N(v0_f256, x0_f256, phi_Dt_ref, N)
        save_csv(filepath_ref, V, X)
    else
        println("Solution file exists. Skipping integration.")
    end
    println("Solution saved at $filepath_ref")

    # make plots
    if parsed_args["plot"]
        
        h_label = "h=Dt/2^$(@sprintf("%d", log2(nsteps)))"
        ref_h_label = "h=Dt/2^$(@sprintf("%d", log2(ref_nsteps)))"

        # round err vs N 
        plot_errors(
            filepath_f64, filepath_f256, 
            (sol1, sol2) -> compute_traj_err(prob, sol1, sol2)[1];
            ylabel=L"|| \Phi^n_{\mathrm{float64}} u_0 - \Phi^n_{\mathrm{float256}} u_0 || ", 
            title="$problem N=$N Dt=$Dt\n$method $h_label", 
            filepath="$output_dir/$method/N=$(N)_Dt=$(@sprintf("%.2e", Dt))_nsteps=$(nsteps)_round_err.png", 
            ymin=1e-30, ymax=1e4)

        # round err vs N 
        plot_errors(
            filepath_f128, filepath_f256, 
            (sol1, sol2) -> compute_traj_err(prob, sol1, sol2)[1];
            ylabel=L"|| \Phi^n_{\mathrm{float128}} u_0 - \Phi^n_{\mathrm{float256}} u_0 || ", 
            title="$problem N=$N Dt=$Dt\n$method $h_label", 
            filepath="$output_dir/$method/N=$(N)_Dt=$(@sprintf("%.2e", Dt))_nsteps=$(nsteps)_round_err2.png", 
            ymin=1e-35, ymax=1e4)
        
        # traj err vs N 
        plot_errors_against_ref(
            [filepath_f64, filepath_f128, filepath_f256], filepath_ref, 
            (sol, ref_sol) -> compute_traj_err(prob, sol, ref_sol)[2], 
            ["$method $h_label float64", "$method $h_label float128", "$method $h_label float256"];
            ylabel=L"|| \Phi u_0 - \Phi_{\mathrm{ref}} u_0 || / || \Phi_{\mathrm{ref}} u_0 || ", 
            title="$problem N=$N Dt=$Dt\nRef: $ref_method $ref_h_label float256", 
            filepath="$output_dir/$method/N=$(N)_Dt=$(@sprintf("%.2e", Dt))_nsteps=$(nsteps)_global_traj_err.png", 
            ymin=1e-35, ymax=1e1, legend=:bottomright)
        
        # H err vs N 
        plot_errors_against_ref(
            [filepath_f64, filepath_f128, filepath_f256], filepath_ref, 
            (sol, ref_sol) -> compute_H_err(prob, sol, ref_sol)[2],
            ["$method $h_label float64", "$method $h_label float128", "$method $h_label float256"];
            ylabel=L"|H(\Phi u_0) - H(\Phi_{\mathrm{ref}} u_0)| / |H(\Phi_{\mathrm{ref}} u_0)|", 
            title="$problem N=$N Dt=$Dt\nRef: $ref_method $ref_h_label float256", 
            filepath="$output_dir/$method/N=$(N)_Dt=$(@sprintf("%.2e", Dt))_nsteps=$(nsteps)_global_H_err.png", 
            ymin=1e-35, ymax=1e1, legend=:topright)
    end
end

main()
