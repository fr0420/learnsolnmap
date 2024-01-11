import torch
from torch import nn
from pytorch_lightning.callbacks import Callback


class FixedSequenceWeights(Callback):
    def __init__(self, weights):
        self.weights = torch.tensor(weights)
        
    def on_fit_start(self, trainer, pl_module):
        pl_module.set_seq_weights(self.weights)
        

class RandomizedSequenceWeights(Callback):
    def __init__(self, sequence_len, n1, n2):
        assert (n1 + n2) < sequence_len and n1 >= 0 and n2 >= 0
        self.sequence_len = sequence_len
        self.n1 = n1
        self.n2 = n2 
        self.prob = torch.tensor([0] * n1 + [1] * (sequence_len - n1), dtype=torch.float)
        self.base_weights = torch.tensor([1] * n1 + [0] * (sequence_len - n1))
#         self.base_factors = 1./ torch.arange(1, sequence_len+1)
        self.base_factors = 1.
        
    def on_fit_start(self, trainer, pl_module):
        pl_module.set_seq_weights(self.base_weights*self.base_factors)
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        indices = torch.multinomial(self.prob, self.n2)
        random_weights = nn.functional.one_hot(indices, num_classes=self.sequence_len).sum(dim=0)
        pl_module.set_seq_weights((random_weights + self.base_weights)*self.base_factors)
