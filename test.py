import os 
import numpy as np
import pandas as pd 
import torch
from model import SolutionMap, CorrectionOperator, CorrectionOperator2


def load_model(model_name, checkpoint_path):
    """Load model from checkpoint"""

    model_class = {
        "SolutionMap": SolutionMap, 
        "CorrectionOperator": CorrectionOperator, 
        "CorrectionOperator2": CorrectionOperator2
    }[model_name]
    
    # old approach: load_from_checkpoint, then convert to double (this is problematic) 
    # model = model_class.load_from_checkpoint(checkpoint_path, strict=False).double()
    
    # new approach: initilize model, convert to double, then load_state_dict
    checkpoint = torch.load(checkpoint_path)
    model = model_class(**checkpoint["hyper_parameters"])
    model = model.double()
    info = model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    print(f"Loaded {model_name} from {checkpoint_path}")
    print(info)
    print("\n", model)
        
    return model 


def run_model(model, u0, N, columns, output_path):
    """Run model iteratively for N times given a initial state u0"""
    
    print(f"\nInitial condition u0: {u0}")
    print(f"Integrate forward for N={N} steps ...")

    sol = torch.zeros(N+1, len(u0)).type_as(u0)
    sol[0] = u0
    
    for n in range(N):
        sol[n+1] = model(sol[n])
#         sol[n+1] = model(sol[n].unsqueeze(0))[0]
#         sol[n+1] = model(sol[n])[0]
        
    df_sol = pd.DataFrame(sol.detach().numpy(), columns=columns)
    df_sol.to_csv(output_path, index=False)
    
    print("Done.")
    print(f"Results saved to {output_path}")

    return sol
    

def main(checkpoint_paths, u0_list, u0_labels, nsteps, columns, output_dir):

    models = [load_model("SolutionMap", ckpt_path) for ckpt_path in checkpoint_paths]
    
    for u0, label in zip(u0_list, u0_labels):

        path = os.path.join(output_dir, label) 
        os.makedirs(path, exist_ok=True)

        for i, model in enumerate(models): 
            run_model(model, u0, nsteps, columns, os.path.join(path, f'NN{i+1}_sol.csv'))
    
    return None


checkpoint_paths = [
#     "/work/08170/rfang/maverick2/learnsolnmap/solutionmap/7fxuc2dj/checkpoints/epoch999-val_loss4.501e-05.ckpt",
    "/work/08170/rfang/maverick2/learnsolnmap/solutionmap/2q2mr8th/checkpoints/epoch1727-val_loss3.816e-05.ckpt",
    "/work/08170/rfang/maverick2/learnsolnmap/solutionmap/wixcf1d7/checkpoints/epoch1732-val_loss3.433e-05.ckpt",
]


cols = ["p1", "p2", "p3", "p4", "p5", "p6", "q1", "q2", "q3", "q4", "q5", "q6"]
omega = 300
u0_list = [torch.tensor([0., np.sqrt(2), 0., 0., 0., 0., (1 - 1/omega)/np.sqrt(2), (1 + 1/omega)/np.sqrt(2), 0., 0., 0., 0.], dtype=torch.float64),
           torch.tensor([0., 1., 0., 0., 0., 0., (1 - 1/omega)/np.sqrt(2), (1 + 1/omega)/np.sqrt(2), 0., 0., 0., 0.], dtype=torch.float64),
           torch.tensor([0., 2., 0., 0., 0., 0., (1 - 1/omega)/np.sqrt(2), (1 + 1/omega)/np.sqrt(2), 0., 0., 0., 0.], dtype=torch.float64)]
u0_labels = ["H0-2.00003", "H0-1.50003", "H0-3.00003"]

main(checkpoint_paths, u0_list, u0_labels, 1000, cols, "./res/fpu") 

"""
cols = list(map(lambda e: "v{}".format(e), range(1, 15))) + list(map(lambda e: "x{}".format(e), range(1, 15)))
x0 = torch.tensor([0.0, 0.0, 0.02, 0.39, 0.34, 0.17, 0.36, -0.21, -0.02, -0.4, -0.35, -0.16, -0.31, 0.21])

# H0 = -1260 kB
v0 = torch.tensor([-30.0, -20.0, 50.0, -90.0, -70.0, -60.0, 90.0, 40.0, 80.0, 90.0, -40.0, 100.0, -80.0, -60.0])
run_model(model, torch.cat((v0/100, x0)), 1000, cols, "./res/lj/H0--1260/NN_sol.csv")

# H0 = -1174 kB
v0 = torch.tensor([-130.0, -20.0, 150.0, -90.0, -70.0, -60.0, 90.0, 40.0, 80.0, 90.0, -40.0, 100.0, -80.0, -60.0])
run_model(model, torch.cat((v0/100, x0)), 1000, cols, "./res/lj/H0--1174/NN_sol.csv")

# H0 = -1312 kB 
v0 = torch.tensor([0.0, -20.0, 20.0, -90.0, -50.0, -60.0, 70.0, 40.0, 80.0, 90.0, -40.0, 20.0, -80.0, -20.0])
run_model(model, torch.cat((v0/100, x0)), 1000, cols, "./res/lj/H0--1312/NN_sol.csv")
"""


