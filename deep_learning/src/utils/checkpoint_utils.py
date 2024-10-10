import torch
from modules.fixed_timestep import FixedStepSolutionMap
from modules.variable_timestep import IdentityEnforcedSolutionMap, T0CenteredSolutionMap


def load_model_from_ckpt(ckpt_path, model_name="SolutionMap"):
    """
    Loads a model from a checkpoint file.

    Args:
        ckpt_path (str): Path to the checkpoint file.

    Returns:
        model (SolutionMap): The model loaded with trained weights and set to evaluation mode.
        dtype (torch.dtype): The data type of the model parameters.
    """
    try:
        # Load checkpoint from the given path
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except FileNotFoundError:
        print(f"Error: No checkpoint found at {ckpt_path}")
        return None
    except Exception as e:
        print(f"Error loading the checkpoint: {e}")
        return None

    # Extract model hyperparameters and state dictionary
    hyper_params = checkpoint.get("hyper_parameters", {})
    state_dict = checkpoint.get("state_dict", {})

    # Instantiate the model using hyperparameters
    model = {
        "FixedStepSolutionMap": FixedStepSolutionMap,
        "IdentityEnforcedSolutionMap": IdentityEnforcedSolutionMap,
        "T0CenteredSolutionMap": T0CenteredSolutionMap,
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
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, dtype
