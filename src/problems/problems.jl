using LinearAlgebra

abstract type SeparableHamiltonianSystem end

"""Compute total energy / Hamiltonian."""
function compute_H(
    prob::SeparableHamiltonianSystem, 
    v::AbstractArray{T, 1}, 
    x::AbstractArray{T, 1}
    ) where T<:AbstractFloat
    return compute_K(prob, v) + compute_U(prob, x)
end

"""Compute trajectory error between two solutions."""
function compute_traj_err(
    prob::SeparableHamiltonianSystem,
    sol1::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}, 
    sol2::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}
    ) where T<:AbstractFloat
    sol1 = nondimensionalize(prob, sol1...)
    sol2 = nondimensionalize(prob, sol2...)
    abs_err = norm(sol1 .- sol2)
    rel_err = abs_err / norm(sol2)
    return abs_err, rel_err
end

"""Compute energy error between two solutions."""
function compute_H_err(
    prob::SeparableHamiltonianSystem,
    sol1::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}, 
    sol2::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}
    ) where T<:AbstractFloat
    H1 = compute_H(prob, sol1...)
    H2 = compute_H(prob, sol2...)
    abs_err = abs(H1 - H2)
    rel_err = abs_err / abs(H2)
    return abs_err, rel_err
end

include("./fpu.jl")
include("./argoncrystal.jl")
include("./1body.jl")
include("./2body.jl")
include("./3body.jl")
include("./3body_2d.jl")
include("./nbody.jl")
include("./nco.jl")

if ARGS == ["--run"]
    using MultiFloats
    MultiFloats.use_bigfloat_transcendentals()

    # prob = FPU(; omega=50.)
    # prob = ArgonCrystal()
    # prob = OneBodyKepler()
    # prob = TwoBodyKepler(; g12=1e-5)
    # prob = ThreeBody()
    prob = ThreeBody2D(m1=1., m2=1., m3=1., G=1.)
    # prob = NBody()
    # prob = NonlinearCoupledOscillators(; epsilon=0.01)
    println(prob)

    v0, x0 = initial_condition(prob, Float64)
    println(v0, x0)
    println(compute_H(prob, v0, x0))
    println(compute_K(prob, v0))
    println(compute_U(prob, x0))

    # ddx = zero(x0)
    # compute_ddx!(prob, ddx, v0, x0)
    # println(ddx)

    # v1 = ddx * 0.001 + v0
    # x1 = v0 * 0.001 + x0
    # println(compute_H(prob, v1, x1))
    # println(compute_H_err(prob, (v1, x1), (v0, x0)))

    # v0_ptb, x0_ptb = v0 .+ 1e-5, x0 .+ 1e-5
    v0_ptb, x0_ptb = copy(v0), copy(x0)
    # v0_ptb[5] += 1e-2
    x0_ptb[1] += 1e-2
    println(compute_H_err(prob, (v0, x0), (v0_ptb, x0_ptb)))
    println(compute_traj_err(prob, (v0, x0), (v0_ptb, x0_ptb)))
end