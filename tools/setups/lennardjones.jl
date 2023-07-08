"""
Molecular dynamics with Lennard-Jones potential  
"""

using LinearAlgebra

const Natoms = 7;
const d = 2; 
const MASS = 66.34e-27;
const kB = 1.380658e-23;
const EPSILON = 119.8kB;
const SIGMA = 0.341;

const twoSIGMA12 = 2SIGMA^(12);
const SIGMA6 = SIGMA^(6);
const twentyfourEPSILONdivbyMASS = 24EPSILON/MASS;
const halfMASSdivbyNatomsdivbykB = 0.5MASS/(Natoms*kB);

const Lengthz = Int(2Natoms + Natoms*(Natoms-1)/2);
const twoSqrtEPSILON = 2 * sqrt(EPSILON);
const SqrtHalfMASS = sqrt(MASS/2);


function distance_matrix(X::AbstractArray{T, 2}) where T<:AbstractFloat
    n = size(X, 2)         
    return [norm(X[:, i]-X[:, j]) for i in 1:n, j in 1:n]
end

function A(x::Array{T, 1}) where T<:AbstractFloat
    x_reshaped = reshape(x, (d, Natoms))
    dist = distance_matrix(x_reshaped)
    xddot = zero(x_reshaped)
    
    for i in 1:Natoms
        for j in 1:Natoms
            if i == j
                continue
            else
                r = dist[i, j]
                fac = twoSIGMA12*r^(-14) - SIGMA6*r^(-8) 
                xddot[:, i] += fac * (x_reshaped[:, i] - x_reshaped[:, j])
            end
        end
    end
    xddot *= twentyfourEPSILONdivbyMASS
    return vec(xddot)
end


LJ_potential(r) = 4EPSILON * ((SIGMA/r)^(12) - (SIGMA/r)^(6))


function compute_K(v::AbstractArray{T, 1}) where T<:AbstractFloat
    """Compute kinetic energy"""
    return 0.5 * MASS * v' * v
end



function compute_U(x::AbstractArray{T, 1}) where T<:AbstractFloat
    """Compute potential energy"""
    
    x_reshaped = reshape(x, (d, Natoms))
    dist = distance_matrix(x_reshaped)
    U = sum([LJ_potential(dist[i, j]) for i in 1:Natoms for j in 1:i-1])
    return U
end


function compute_H(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    """Compute total energy / Hamiltonian"""
    return compute_K(v) + compute_U(x)
end


# Initial conditions
x0 = vec([0.0 0.0 0.02 0.39 0.34 0.17 0.36 -0.21 -0.02 -0.4 -0.35 -0.16 -0.31 0.21]);
v0 = vec([-30.0 -20.0 50.0 -90.0 -70.0 -60.0 90.0 40.0 80.0 90.0 -40.0 100.0 -80.0 -60.0]);  # H0 = -1260 kB
# v0 = vec([-130.0 -20.0 150.0 -90.0 -70.0 -60.0 90.0 40.0 80.0 90.0 -40.0 100.0 -80.0 -60.0]);  # H0 = -1174 kB
# v0 = vec([0.0 -20.0 20.0 -90.0 -50.0 -60.0 70.0 40.0 80.0 90.0 -40.0 20.0 -80.0 20.0]); # H0 = -1312 kB

# Initial energy 
K0 = compute_K(v0);
U0 = compute_U(x0);
H0 = compute_H(v0, x0);


w(r::T) where T<:AbstractFloat = (SIGMA / r)^6


function construct_z(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    z = zeros(T, Lengthz)
    z[1:2Natoms] = v * SqrtHalfMASS
    x_reshaped = reshape(x, (d, Natoms))
    dist = distance_matrix(x_reshaped)
    z[2Natoms+1:end] = [twoSqrtEPSILON * (w(dist[i, j])-0.5) for i in 1:Natoms for j in 1:i-1]
    return z
end


function vec2symmat(vec::Array{T, 1}, n::Integer) where T<:AbstractFloat
    n*(n-1)/2 == length(vec) || error("length of vector is not valid")
    
    mat = zeros(T, n, n)
    k = 1 
    
    for i in 1:n
        for j in 1:i-1
            mat[i, j] = vec[k]
            mat[j, i] = vec[k]
            k += 1
        end
    end
    
    return mat
end


function classicalMDS(D::Array{T, 2}, d::Integer, tol::T=1e-8) where T<:AbstractFloat
    n = size(D, 1)
    J = Matrix(1.0I, n, n) - ones(n) * ones(n)' / n
    G = -0.5*J*D*J
    G = (G + G')/2
    sol = eigen(G, sortby = x -> -abs(x))
#     print(sol.values)
    if ~all(sol.values .> -tol)  # check if G is psd
       error("G is not positive semidefinite.") 
    end
    X = (sol.vectors[:, 1:d] * Diagonal(sqrt.(sol.values[1:d])))'
    return X
end


function align(X::AbstractArray{T, 2}, Y::AbstractArray{T, 2}) where T<:AbstractFloat
    d = size(X, 1)
    n = size(X, 2)
    xc = sum(X, dims=2) / n
    yc = sum(Y, dims=2) / n
    Xc = X .- xc
    Yc = Y .- yc
    sol = svd(Xc * Yc')
    R = (sol.U * sol.Vt)'
    return R * Xc .+ yc
end


function recover_canonical_vars(z::Array{T, 1}) where T<:AbstractFloat

    v = z[1:2Natoms] ./ SqrtHalfMASS
    W = z[2Natoms+1:end] ./ twoSqrtEPSILON .+ 0.5    
    dist_sq_vec = sign.(W).*(W./sign.(W)).^(-1/3) * SIGMA^2
    D = vec2symmat(dist_sq_vec, Natoms)
    x = classicalMDS(D, d)
#     xhat = align(classicalMDS(D, d), reshape(x, (d, Natoms)))
    x = vec(x)
    return v, x
end