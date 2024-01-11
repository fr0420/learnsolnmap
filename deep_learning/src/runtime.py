import timeit
import numpy as np
import torch
from .models import SolutionMap, CorrectionOperator, CorrectionOperator2

checkpoint_path = "/work/08170/rfang/maverick2/learnsolnmap/solutionmap/2itmmglz/checkpoints/epoch139-val_loss5.260e-05.ckpt"
model = CorrectionOperator2.load_from_checkpoint(checkpoint_path, strict=False, WS_strength=0.).double()
u0 = torch.rand(12).double()
u0 = u0.unsqueeze(0)

def time(N=100):
	SETUP_CODE = "from __main__ import model, u0"
	TEST_CODE = "model(u0)"
 
	times = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=7, number=N)
	
	times = np.array(times)/N
	print("min = {:.3e} sec".format(np.amin(times)))
	print("mean = {:.3e} sec".format(np.mean(times)))
	print("std = {:.3e} sec".format(np.std(times)))


print("\nGPU time:")
model.to("cuda")
u0 = u0.to("cuda")
time()

print("\nCPU time:")
model.to("cpu")
u0 = u0.to("cpu")
time()


