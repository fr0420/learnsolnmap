from torch.optim.lr_scheduler import _LRScheduler
import math


class CustomCyclicLR(_LRScheduler):
    def __init__(self, optimizer, steps_per_cycle=2000, base_lr=0.001, max_lr=0.01, 
                 base_lr_scale_fn=None, max_lr_scale_fn=None, scale_mode='cycle', 
                 pct_start=0.3, anneal_strategy='cos', last_epoch=-1, verbose=False):
        
        self.steps_per_cycle = steps_per_cycle
        self.base_lr = base_lr 
        self.max_lr = max_lr
        
        # Validate scale_mode
        if scale_mode not in ['cycle', 'iterations']:
            raise ValueError("scale_mode must by one of 'cycle' or 'iterations', instead got {}".format(scale_mode))
        self.scale_mode = scale_mode 
        
        self.base_lr_scale_fn = self._default_scale_fn(scale_mode) if base_lr_scale_fn is None else base_lr_scale_fn
        self.max_lr_scale_fn = self._default_scale_fn(scale_mode) if max_lr_scale_fn is None else max_lr_scale_fn
        
        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError("Expected float between 0 and 1 pct_start, but got {}".format(pct_start))
        self.pct_start = pct_start
        
        # Validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError("anneal_strategy must by one of 'cos' or 'linear', instead got {}".format(anneal_strategy))
        elif anneal_strategy == 'cos':
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = self._annealing_linear
            
        super().__init__(optimizer, last_epoch, verbose)
    
    def _default_scale_fn(self, scale_mode):
        if scale_mode == 'cycle':
            return lambda cycle: 0.9**cycle
        elif scale_mode == 'iterations':
            return lambda iteration: 0.999995**iteration
        
    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start
    
    def get_lr(self):
        step_num = self.last_epoch 
        cycle_idx = math.floor(step_num / self.steps_per_cycle)
        
        if self.scale_mode == 'cycle': 
            init_lr = self.base_lr * self.base_lr_scale_fn(cycle_idx)
            final_lr = self.base_lr * self.base_lr_scale_fn(cycle_idx+1)
            peak_lr = self.max_lr * self.max_lr_scale_fn(cycle_idx+self.pct_start)
        elif self.scale_mode == 'iterations':
            init_lr = self.base_lr * self.base_lr_scale_fn(cycle_idx*self.steps_per_cycle)
            final_lr = self.base_lr * self.base_lr_scale_fn((cycle_idx+1)*self.steps_per_cycle)
            peak_lr = self.max_lr * self.max_lr_scale_fn((cycle_idx+self.pct_start)*self.steps_per_cycle)
            
        x = step_num / self.steps_per_cycle - cycle_idx 
        
        if x < self.pct_start:
            pct = x / self.pct_start 
            lr = self.anneal_func(init_lr, peak_lr, pct)
        else:
            pct = (x - self.pct_start) / (1 - self.pct_start) 
            lr = self.anneal_func(peak_lr, final_lr, pct)
        return [lr] * len(self.optimizer.param_groups)
