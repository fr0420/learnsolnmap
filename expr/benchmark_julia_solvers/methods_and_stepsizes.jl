"""
Benchmark accuracy and runtime of various ode solvers on a fixed time interval [0, Dt].
"""

include("../../src/utils/ode_solver.jl")
include("../../src/problems/problems.jl")

using ArgParse
using DataFrames, CSV
using Plots, LaTeXStrings
using Printf


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


function plot_errors(
    errors, nsteps_list, method_list; ylabel="error", title="", filepath="./error.png", ymin=1e-20, ymax=1e1)

    plot(dpi=300)
    plot!(nsteps_list, errors, marker=:circle, labels=permutedims(method_list))
    plot!(ylim=[ymin, ymax])
    plot!(xaxis=:log, yaxis=:log)
    plot!(xlabel="nsteps (= Dt/h)", ylabel=ylabel)
    plot!(title=title)
    plot!(legend=:bottomleft)
    plot!(margin=10Plots.mm)

    savefig(filepath)
    println("Saved figure to $filepath")
end


function plot_runtimes(runtimes, nsteps_list, method_list; title="", filepath="./runtime.png")

    plot(dpi=300)
    plot!(nsteps_list, runtimes, marker=:circle, labels=permutedims(method_list))
    plot!(xaxis=:log, yaxis=:log)
    plot!(xlabel="nsteps (= Dt/h)", ylabel="runtime (sec)")
    plot!(title=title)
    plot!(legend=:bottomright)
    plot!(margin=10Plots.mm)

    savefig(filepath)
    println("Saved figure to $filepath")
end


function main()
    parsed_args = parse_commandline()
    
    problem = parsed_args["problem"]
    Dt = parsed_args["Dt"]
    output_dir = parsed_args["output_dir"]

    # set methods and step sizes / number of steps
    # method_list = ["VelocityVerlet", "CalvoSanz4", "McAte5", "KahanLi6", "KahanLi8"]
    method_list = ["VelocityVerlet", "CalvoSanz4", "DPRKN4"]
    nsteps_list = [1, 10, 100, 1000, 10000]
    # nsteps_list = 2 .^ [0, 1, 2, 3, 4, 5, 6]
    # nsteps_list = 2 .^ [8, 10, 12, 14, 16]
    ref_method = "DPRKN12"
    ref_nsteps = 2^14
    n_methods = length(method_list)
    n_nsteps = length(nsteps_list)

    println("problem = $problem")
    println("Dt = $Dt")
    println("output_dir = $output_dir")
    println("method_list = $method_list")
    println("nsteps_list = $nsteps_list")
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
    v0_f256, x0_f256 = initial_condition(prob, Float64x4)

    println("\nInitial condition:")
    println("v0: ", v0)
    println("x0: ", x0)
    println("H0: ", compute_H(prob, v0, x0))
    println("K0: ", compute_K(prob, v0))
    println("U0: ", compute_U(prob, x0))

    # define solvers in float64 and float64x4 respectively  
    phi_Dt_f64 = (method, nsteps) -> ode_solve(
        (ddx, dx, x, p, t) -> compute_ddx!(prob, ddx, dx, x), METHODS[method], v0, x0, 0.0, Dt, nsteps, false)
    phi_Dt_f256 = (method, nsteps) -> ode_solve(
        (ddx, dx, x, p, t) -> compute_ddx!(prob, ddx, dx, x), METHODS[method], v0_f256, x0_f256, 0.0, Dt, nsteps, false)
    
    # compute reference solution 
    println("\nComputing reference solution ...")
    ref_runtime = @elapsed ref_sol = phi_Dt_f256(ref_method, ref_nsteps)
    
    println("Computing solutions ...")
    table = []
    for i in 1:n_methods
        for j in 1:n_nsteps
            m = method_list[i]
            nsteps = nsteps_list[j]

            # compute solution
            runtime_f64 = @elapsed sol_f64 = phi_Dt_f64(m, nsteps)
            runtime_f256 = @elapsed sol_f256 = phi_Dt_f256(m, nsteps) 
            if j == 1  # re-measure runtime since the function gets compiled on the first call
                runtime_f64 = @elapsed phi_Dt_f64(m, nsteps)
                runtime_f256 = @elapsed phi_Dt_f256(m, nsteps)
            end

            # compute errors
            sol_f64 = (Float64x4.(sol_f64[1]), Float64x4.(sol_f64[2]))
            rounding_err = compute_traj_err(prob, sol_f64, sol_f256)[1]
            abs_traj_err_f64, rel_traj_err_f64 = compute_traj_err(prob, sol_f64, ref_sol)
            abs_H_err_f64, rel_H_err_f64 = compute_H_err(prob, sol_f64, ref_sol)
            abs_traj_err_f256, rel_traj_err_f256 = compute_traj_err(prob, sol_f256, ref_sol)
            abs_H_err_f256, rel_H_err_f256 = compute_H_err(prob, sol_f256, ref_sol)
            
            row_f64 = (
                method=m, nsteps=nsteps, precision="float64",
                abs_traj_err=abs_traj_err_f64,
                rel_traj_err=rel_traj_err_f64,
                abs_H_err=abs_H_err_f64,
                rel_H_err=rel_H_err_f64,
                rounding_err=rounding_err,
                runtime=runtime_f64,
            )
            row_f256 = (
                method=m, nsteps=nsteps, precision="float256",
                abs_traj_err=abs_traj_err_f256,
                rel_traj_err=rel_traj_err_f256,
                abs_H_err=abs_H_err_f256,
                rel_H_err=rel_H_err_f256,
                rounding_err=missing,
                runtime=runtime_f256
            )
            push!(table, row_f64)
            push!(table, row_f256)
        end
    end
    row_ref = (
        method=ref_method, nsteps=ref_nsteps, precision="float256",
        abs_traj_err=missing,
        rel_traj_err=missing,
        abs_H_err=missing,
        rel_H_err=missing,
        rounding_err=missing,
        runtime=ref_runtime
    )
    push!(table, row_ref)

    df = DataFrame(table)
    println(df)

    # save table 
    filepath = "$output_dir/Dt=$(@sprintf("%.2e", Dt)).csv"
    CSV.write(filepath, df)
    println("Saved table to $filepath")

    # plot error or runtime against nsteps 
    if parsed_args["plot"]

        precision = "float256"
        rel_traj_errors = [
            df[df.precision .== precision .&& df.method .== m, :rel_traj_err] for m in method_list
        ]
        rel_H_errors = [
            df[df.precision .== precision .&& df.method .== m, :rel_H_err] for m in method_list
        ]
        runtimes = [
            df[df.precision .== precision .&& df.method .== m, :runtime] for m in method_list
        ]
        rounding_errors = [
            df[df.precision .== "float64" .&& df.method .== m, :rounding_err] for m in method_list
        ]

        ref_label = "float256 $(ref_method) nsteps=2^$(log2(ref_nsteps))"
        
        plot_errors(
            rel_traj_errors, nsteps_list, method_list, 
            ylabel=L"|| \Phi u_0 - \Phi_{\mathrm{ref}} u_0 || / || \Phi_{\mathrm{ref}} u_0 || ", 
            title="$problem Dt=$Dt $precision\nRef: $ref_label", 
            filepath="$output_dir/Dt=$(@sprintf("%.2e", Dt))_rel_traj_err.png", 
            ymin=1e-30, ymax=1e-1)

        plot_errors(
            rel_H_errors, nsteps_list, method_list, 
            ylabel=L"|H(\Phi u_0) - H(\Phi_{\mathrm{ref}} u_0)| / |H(\Phi_{\mathrm{ref}} u_0)|",
            title="$problem Dt=$Dt $precision\nRef: $ref_label", 
            filepath="$output_dir/Dt=$(@sprintf("%.2e", Dt))_rel_H_err.png",
            ymin=1e-30, ymax=1e-1)

        plot_runtimes(
            runtimes, nsteps_list, method_list,
            title="$problem Dt=$Dt $precision",
            filepath="$output_dir/Dt=$(@sprintf("%.2e", Dt))_runtime.png")

        plot_errors(
            rounding_errors, nsteps_list, method_list, 
            ylabel=L"|| \Phi_{\mathrm{float64}} u_0 - \Phi_{\mathrm{float256}} u_0 || ", 
            title="$problem Dt=$Dt", 
            filepath="$output_dir/Dt=$(@sprintf("%.2e", Dt))_round_err.png", 
            ymin=1e-15, ymax=1e-10)
    end
end

main()
