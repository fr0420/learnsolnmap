import os 
import numpy as np
import pandas as pd 
import torch
from model import LitModel


def run_model(model, u0, nsteps, columns, output_path):
    sol = torch.zeros(nsteps+1, len(u0)).double()
    sol[0] = u0

    for n in range(nsteps):
        sol[n+1] = model(sol[n])
        
    df_sol = pd.DataFrame(sol.detach().numpy(), columns=columns)
    df_sol.to_csv(output_path, index=False)
    
    return sol


checkpoint_path = "/workspace/projects_rui/learnsolnmap/solutionmap/1wd6gswh/checkpoints/epoch998-val_loss6.859e-05.ckpt"
model = LitModel.load_from_checkpoint(checkpoint_path, strict=False).double()


# cols = ["p1", "p2", "p3", "p4", "p5", "p6", "q1", "q2", "q3", "q4", "q5", "q6"]
# omega = 300 

# u0 = torch.tensor([0., np.sqrt(2), 0., 0., 0., 0., (1 - 1/omega)/np.sqrt(2), (1 + 1/omega)/np.sqrt(2), 0., 0., 0., 0.])  # H0 = 2.00003
# run_model(model, u0, 1000, cols, "./res/H0-2.00003/NN1_sol.csv")

# u0 = torch.tensor([0., 1., 0., 0., 0., 0., (1 - 1/omega)/np.sqrt(2), (1 + 1/omega)/np.sqrt(2), 0., 0., 0., 0.])  # H0 = 1.50003
# run_model(model, u0, 1000, cols, "./res/H0-1.50003/NN1_sol.csv")

# u0 = torch.tensor([0., 2., 0., 0., 0., 0., (1 - 1/omega)/np.sqrt(2), (1 + 1/omega)/np.sqrt(2), 0., 0., 0., 0.])  # H0 = 3.00003
# run_model(model, u0, 1000, cols, "./res/H0-3.00003/NN1_sol.csv")


cols = list(map(lambda e: "v{}".format(e), range(1, 15))) + list(map(lambda e: "x{}".format(e), range(1, 15)))
v0 = torch.tensor([-30.0, -20.0, 50.0, -90.0, -70.0, -60.0, 90.0, 40.0, 80.0, 90.0, -40.0, 100.0, -80.0, -60.0])
x0 = torch.tensor([0.0, 0.0, 0.02, 0.39, 0.34, 0.17, 0.36, -0.21, -0.02, -0.4, -0.35, -0.16, -0.31, 0.21])
u0 = torch.cat((v0/100., x0))
run_model(model, u0, 1000, cols, "./res/NN_sol.csv")
