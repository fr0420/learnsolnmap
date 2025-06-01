using PyCall
using BenchmarkTools

py"""
import sys
sys.path.insert(0, "/workspace/projects_rui/learnsolnmap/deep_learning/src")
"""
torch = pyimport("torch")
checkpoint_utils = pyimport("utils.checkpoint_utils")


function load_nn(checkpoint_path, model_name="SFTaylorBasedT0CenteredSolutionMap")
    model, _ = checkpoint_utils.load_model_from_ckpt(checkpoint_path, model_name)
    model.to("cuda:0")
    model.to(torch.float64)
    return model
end 

# function nn_solve(
#     p0::AbstractArray{T, 1}, 
#     q0::AbstractArray{T, 1}, 
#     torch_model::PyObject,
#     torch_dtype::PyObject) where T<:AbstractFloat
        
#     # Concatenate input arrays
#     u0 = vcat(p0, q0)

#     # Convert Julia array to a PyTorch tensor
#     u0_torch = torch.tensor(u0, dtype=torch_dtype)
    
#     # Ensure tensor is on the correct device (CPU/GPU)
#     u0_torch = u0_torch.to(torch_model.device)
    
#     # Perform the model forward pass
#     u_torch = torch_model(u0_torch)
#     p_torch, q_torch = u_torch.chunk(2, dim=-1)
    
#     # Detach tensors and move to CPU, then convert back to Julia arrays
#     p = p_torch.detach().cpu().numpy()
#     q = q_torch.detach().cpu().numpy()
    
#     # Ensure type T
#     p = convert.(T, p)
#     q = convert.(T, q)

#     return p, q
# end


struct NNForward
    torch_model::PyObject
    H::Float64
    nsteps::Integer
    p::Dict{String, Float64}
end


function (f::NNForward)(u0::AbstractArray{T, 1}) where T<:AbstractFloat
    # Extract device and dtype from the model
    device = f.torch_model.device
    dtype = f.torch_model.dtype

    # Convert a Julia array to a PyTorch tensor 
    u0_torch = torch.tensor(u0, dtype=dtype, device=device).unsqueeze(0)

    # Compute step size
    h = f.H / f.nsteps

    # Create the time tensor and the parameter tensors
    t_torch = torch.full((u0_torch.shape[1], 1), h, dtype=dtype, device=device)
    p_torch = Dict(k => torch.full((u0_torch.shape[1], 1), v, dtype=dtype, device=device)
                    for (k, v) in f.p)

    # Perform forward pass
    u_torch = u0_torch.clone()
    for _ in 1:f.nsteps
        u_torch = f.torch_model(u_torch, t_torch, p_torch)
    end
    
    # Detach the tensor, move it to CPU and convert to a Julia array
    u = u_torch.squeeze(0).detach().cpu().numpy()
    
    # Ensure the result is of type T
    u = convert.(T, u)

    return u
end


function (f::NNForward)(u0::Vector{<:AbstractArray{T, 1}}) where T<:AbstractFloat
    # Extract device and dtype from the model
    device = f.torch_model.device
    dtype = f.torch_model.dtype

    # Convert Julia arrays to a PyTorch tensor
    u0_torch = torch.tensor(hcat(u0...)', dtype=dtype, device=device)

    # Compute step size
    h = f.H / f.nsteps

    # Create the time tensor and the parameter tensors
    t_torch = torch.full((u0_torch.shape[1], 1), h, dtype=dtype, device=device)
    p_torch = Dict(k => torch.full((u0_torch.shape[1], 1), v, dtype=dtype, device=device)
                    for (k, v) in f.p)

    # Perform forward pass
    u_torch = u0_torch.clone()
    for _ in 1:f.nsteps
        u_torch = f.torch_model(u_torch, t_torch, p_torch)
    end
    
    # Detach the tensor, move it to CPU and convert to Julia arrays
    u = u_torch.detach().cpu().numpy()
    
    # Ensure the result is of type T
    u = convert.(T, u)

    # Convert the result to a vector of arrays
    u = [u[i, :] for i in 1:size(u, 1)]

    return u
end


function (f::NNForward)(u0::Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}) where T<:AbstractFloat
    
    # Extract dimension from the first tuple
    dim = length(u0[1])

    # Concatenate the tuple elements into a single array
    u0 = vcat(u0[1], u0[2])

    # Pass the concatenated input to the existing function
    u = f(u0)

    # Split the result back into two arrays
    return (u[1:dim], u[dim+1:end])
end


function (f::NNForward)(u0::Vector{<:Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}}) where T<:AbstractFloat
    
    # Extract dimension from the first tuple
    dim = length(u0[1][1])

    # Concatenate each tuple element into a single array
    u0 = [vcat(tuple[1], tuple[2]) for tuple in u0]

    # Pass the concatenated input to the existing function
    u = f(u0)

    # Split the result back into vectors of tuples 
    u = [(u[i][1:dim], u[i][dim+1:end]) for i in 1:size(u, 1)]
    return u
end



if ARGS == ["--run"]

    ### Alphaparticle ###
    # Load the model
    checkpoint_path = "/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20250321-121238_f0bbt0/checkpoints/latest-epoch01249.ckpt"
    model = load_nn(checkpoint_path, "TaylorBasedIdentityEnforcedSolutionMap")
    println(model)
    println(model.device)
    println(model.dtype)

    # Create an instance of the NNForward struct
    nn_solver = NNForward(model, 20., 4, Dict("epsilon"=>0.15))

    # Define input data
    # u0 = [sqrt(2.), 0., 3.0, 2.5]
    u0 = [[sqrt(2.), 0., 3.0, 2.5], [1.0, 1.0, 3.0, 2.5]]
    println(typeof(u0))

    # Warm-up call
    elapsed_time = @elapsed u = nn_solver(u0)
    println(elapsed_time)
    println(u)
    println(typeof(u))

    # Benchmark the forward pass in Julia
    @btime nn_solver($u0)


    # ### FPUT ###
    # # Load the model
    # checkpoint_path = "/workspace/projects_rui/learnsolnmap/deep_learning/logs/train/runs/20250406-182238_8x29ca/checkpoints/latest-epoch00499.ckpt"
    # model = load_nn(checkpoint_path, "SFTaylorBasedT0CenteredSolutionMap")
    # println(model)
    # println(model.device)
    # println(model.dtype)

    # # Create an instance of the NNForward struct
    # nn_solver = NNForward(model, 1.0, 1, Dict())

    # # Define input data
    # u0 = ([sqrt(2.), 0., 0., 0., 0., 0.], [1.0, 1.0, 0., 0., 0., 0.])
    # # u0 = [
    # #     ([sqrt(2.), 0., 0., 0., 0., 0.], [1.0, 1.0, 0., 0., 0., 0.]),
    # #     ([sqrt(2.), 0., 0., 0., 0., 0.], [1.0, 1.0, 0., 0., 0., 0.])
    # # ]
    # println(typeof(u0))

    # # Warm-up call
    # elapsed_time = @elapsed u = nn_solver(u0)
    # println(elapsed_time)
    # println(u)
    # println(typeof(u))

    # # Benchmark the forward pass in Julia
    # @btime nn_solver($u0)

end