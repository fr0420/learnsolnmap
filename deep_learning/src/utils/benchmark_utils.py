from typing import List

import torch
import torch.utils.benchmark as benchmark
import pandas as pd

from modules.solnmap import BaseSolutionMap


def time_forward(model: BaseSolutionMap, nsteps_list: List[int] = [0, 1,]):
    """Benchmark forward pass time of a SolutionMap model."""

    # compare runtime for different batch sizes and devices 
    batch_sizes = [1, 64, 128, 512, 1024, 2048]
    devices = [torch.device("cpu")] 
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:1"))
    
    results = []
    for b in batch_sizes:
        for device in devices:
            u = model.problem.random_states(b).to(device).to(model.dtype)
            t = torch.ones(b, 1, dtype=model.dtype).to(device) * 8.
            p = {key: torch.ones(b, 1, dtype=model.dtype).to(device) * model.problem.default_params[key] 
                 for key in model.problem_param_keys}

            model.to(device)
            label = "forward time"
            sub_label=f"u: {u.shape}, t: {t.shape}"

            for nsteps in nsteps_list:
                results.append(benchmark.Timer(
                    stmt="model.predict_sequence(u, t, p, sequence_len=nsteps+1)",
                    globals={"u": u, "t": t, "p": p, "model": model, "nsteps": nsteps},
                    label=label,
                    sub_label=sub_label,
                    description=f"{device}: nsteps={nsteps}",
                ).blocked_autorange(min_run_time=1))

    return benchmark.Compare(results)


def time_backward(model: BaseSolutionMap, nsteps_list: List[int] = [0, 1,]):
    """Benchmark backward pass time of a SolutionMap model."""

    # compare runtime for different batch sizes
    batch_sizes = [1, 64, 128, 512]

    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:1")
    model.to(device)
    model.update_loss_hparams({"data_misfit": {"strength": 1.0}})

    def backward(model, batch):
        out = model.model_step(batch, batch_idx=0)
        loss = out["loss"]
        loss.backward()
        return

    results = []
    for b in batch_sizes:
        u = model.problem.random_states(b).to(device).to(model.dtype)
        t = torch.ones(b, 1, dtype=model.dtype).to(device) * 10.
        p = {key: torch.ones(b, 1, dtype=model.dtype).to(device) * model.problem.default_params[key] 
                 for key in model.problem_param_keys}
        label = f"backward time ({device})"
        sub_label=f"u: {u.shape}, t: {t.shape}"

        for nsteps in nsteps_list:
            batch = {"input": u, "Dt": t, "params": p, "target_seq": [u for _ in range(nsteps+1)]}
            model.set_seq_weights([1.0] * (nsteps+1))
            results.append(benchmark.Timer(
                stmt="backward(model, batch)",
                globals={"backward": backward, "model": model, "batch": batch},
                label=label,
                sub_label=sub_label,
                description=f"nsteps={nsteps}",
            ).blocked_autorange(min_run_time=1))

    return benchmark.Compare(results)


def outputs_stats(model: BaseSolutionMap, nsteps: int = 5):
    """Benchmark outputs mean and variance for a SolutionMap model."""
    
    data = {}
    
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    model.to(device)
    # u0 = model._prepare_random_states(batch_size=128)
    u0 = model.problem.default_initial_states().to(model.dtype).to(device)
    t = torch.ones(len(u0), 1, dtype=model.dtype).to(u0) * 1.
    p = {key: torch.ones(len(u0), 1, dtype=model.dtype).to(u0) * model.problem.default_params[key] for key in model.problem_param_keys}

    v0, x0 = u0.chunk(2, dim=-1)
    col = pd.Series(
        torch.stack([torch.mean(v0), torch.mean(x0), torch.var(v0), torch.var(x0)]).detach().cpu().numpy(),
        index=["v_mean", "x_mean", "v_var", "x_var"])
    data["input"] = col

    pred_seq = model.predict_sequence(u0, t, p, sequence_len=nsteps+1)
    for i, u in enumerate(pred_seq):
        v, x = u.chunk(2, dim=-1)
        col = pd.Series(
            torch.stack([torch.mean(v), torch.mean(x), torch.var(v), torch.var(x)]).detach().cpu().numpy(),
            index=["v_mean", "x_mean", "v_var", "x_var"])
        data[f"output_{i}"] = col
    
    return pd.DataFrame(data=data)


# def outputs_stats_var_dt(model: BaseVariableDtSolutionMap, nsteps: int = 5):
#     """Benchmark outputs mean and variance for a VariableDtSolutionMap model."""
    
#     data = {}
    
#     # use default initial states as inputs
#     u0 = model.problem.default_initial_states().to(model.dtype)

#     device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
#     model.to(device)
#     u0 = u0.to(device)

#     v0, x0 = u0.chunk(2, dim=-1)
#     col = pd.Series(
#         torch.stack([torch.mean(v0), torch.mean(x0), torch.var(v0), torch.var(x0)]).detach().cpu().numpy(),
#         index=["v_mean", "x_mean", "v_var", "x_var"])
#     data["input"] = col

#     Dt = torch.ones(u0.shape[0], 1).to(u0)
#     pred_seq = model.predict_sequence(u0, Dt, sequence_len=nsteps+1)
#     for i, u in enumerate(pred_seq):
#         v, x = u.chunk(2, dim=-1)
#         col = pd.Series(
#             torch.stack([torch.mean(v), torch.mean(x), torch.var(v), torch.var(x)]).detach().cpu().numpy(),
#             index=["v_mean", "x_mean", "v_var", "x_var"])
#         data[f"output_{i}"] = col
    
#     return pd.DataFrame(data=data)

# def time_backward(model: BaseSolutionMap, max_nsteps: int = 5):
#     """Benchmark backward pass time of a SolutionMap model."""

#     # compare runtime for different batch sizes
#     batch_sizes = [1, 64, 128, 512]

#     # use default initial states as inputs
#     u0 = model.problem.default_initial_states().to(model.dtype)[0, :]

#     device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
#     model.to(device)
#     u0 = u0.to(device)

#     def backward(model, batch):
#         out = model.model_step(batch, batch_idx=0)
#         loss = out["loss"]
#         loss.backward()
#         return

#     results = []
#     for b in batch_sizes:
#         x = u0.repeat(b, 1)
#         batch = [x for _ in range(max_nsteps+1)]
#         label = f"backward time ({device})"
#         sub_label=f"{x.shape}"

#         for i in range(max_nsteps):
#             seq_weights = torch.zeros(max_nsteps)
#             seq_weights[i] = 1.
#             model.set_seq_weights(seq_weights)
#             results.append(benchmark.Timer(
#                 stmt="backward(model, batch)",
#                 globals={"backward": backward, "model": model, "batch": batch},
#                 label=label,
#                 sub_label=sub_label,
#                 description=f"step {i+1}",
#             ).blocked_autorange(min_run_time=1))

#         seq_weights = torch.ones(max_nsteps)
#         model.set_seq_weights(seq_weights)
#         results.append(benchmark.Timer(
#             stmt="backward(model, batch)",
#             globals={"backward": backward, "model": model, "batch": batch},
#             label=label,
#             sub_label=sub_label,
#             description=f"all steps",
#         ).blocked_autorange(min_run_time=1))

#     return benchmark.Compare(results)