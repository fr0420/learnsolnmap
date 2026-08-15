import torch
from modules.fixed_timestep import FixedStepSolutionMap
from modules.variable_timestep import IdentityEnforcedSolutionMap, T0CenteredSolutionMap, StackedSolutionMap
from modules.variable_timestep_taylor import TaylorBasedIdentityEnforcedSolutionMap, TaylorBasedT0CenteredSolutionMap
from modules.variable_timestep_taylor_sf import SFTaylorBasedIdentityEnforcedSolutionMap, SFTaylorBasedT0CenteredSolutionMap

def load_model_from_ckpt(ckpt_path, model_name="T0CenteredSolutionMap", strict=True, load_net_T0_only=False):
    """
    Loads a model from a checkpoint file.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        model_name (str): Name of the model to load.
        strict (bool): Whether to strictly load the state dictionary.
        load_net_T0_only (bool): Whether to load only the net_T0 module for TaylorBasedT0CenteredSolutionMap. Temporary fix.

    Returns:
        model (SolutionMap): The model loaded with trained weights and set to evaluation mode.
        dtype (torch.dtype): The data type of the model parameters.
    """
    try:
        # Load checkpoint from the given path
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        print(f"Error: No checkpoint found at {ckpt_path}")
        return None
    except Exception as e:
        print(f"Error loading the checkpoint: {e}")
        return None

    # Extract model hyperparameters and state dictionary
    hyper_params = checkpoint.get("hyper_parameters", {})
    state_dict = checkpoint.get("state_dict", {})

    # Temporary fix for the loss function in hyperparameters
    loss_target = hyper_params["loss"]["_target_"]
    if loss_target.startswith("losses") and loss_target.split(".")[1] != "losses":
        hyper_params["loss"]["_target_"] = "losses." + hyper_params["loss"]["_target_"]

    # Temporary fix for inconsistent use_dudt usage in net_T0 and net_residual
    if model_name == "TaylorBasedT0CenteredSolutionMap":
        if not hyper_params.get("use_dudt", False): 
            if "net_residual" in hyper_params and "net_T0" in hyper_params:
                if hyper_params["net_residual"].get("input_dim", 0) - hyper_params["net_T0"].get("input_dim", 0) != 1:
                    hyper_params["net_residual"]["input_dim"] = hyper_params["net_T0"]["input_dim"] + 1

    # Instantiate the model using hyperparameters
    model = {
        "FixedStepSolutionMap": FixedStepSolutionMap,
        "IdentityEnforcedSolutionMap": IdentityEnforcedSolutionMap,
        "T0CenteredSolutionMap": T0CenteredSolutionMap,
        "StackedSolutionMap": StackedSolutionMap,
        "TaylorBasedIdentityEnforcedSolutionMap": TaylorBasedIdentityEnforcedSolutionMap,
        "TaylorBasedT0CenteredSolutionMap": TaylorBasedT0CenteredSolutionMap,
        "SFTaylorBasedIdentityEnforcedSolutionMap": SFTaylorBasedIdentityEnforcedSolutionMap,
        "SFTaylorBasedT0CenteredSolutionMap": SFTaylorBasedT0CenteredSolutionMap,
        # "ODEEmbeddedIdentityEnforcedSolutionMap": TaylorBasedIdentityEnforcedSolutionMap,
        # "ODEEmbeddedT0CenteredSolutionMap": TaylorBasedT0CenteredSolutionMap,
        "VariableDtSolutionMap": IdentityEnforcedSolutionMap,
    }.get(model_name)(**hyper_params)

    # If state_dict is empty, return without loading weights
    if not state_dict:
        print("Warning: No state dictionary found in the checkpoint.")
        return model

    # Infer model dtype from the state_dict (assume all parameters have the same dtype)
    dtype = next(iter(state_dict.values())).dtype if state_dict else None
    model.to(dtype=dtype)

    # Load model state from the checkpoint
    if load_net_T0_only and model_name in [
        "T0CenteredSolutionMap",
        "TaylorBasedT0CenteredSolutionMap", 
        "SFTaylorBasedT0CenteredSolutionMap",
    ]:
        net_T0_state_dict = {k.replace("net_T0.", ""): v for k, v in state_dict.items() if k.startswith("net_T0.")}
        message = model.net_T0.load_state_dict(net_T0_state_dict, strict=strict)
        print(message)
    else:
        message = model.load_state_dict(state_dict, strict=strict)
    if message is not None:
        if message.missing_keys:
            print(f"Warning: Missing keys in the state dictionary: {message.missing_keys}")
        if message.unexpected_keys:
            print(f"Warning: Unexpected keys in the state dictionary: {message.unexpected_keys}")

    # Set the model to evaluation mode
    model.eval()

    return model, dtype
