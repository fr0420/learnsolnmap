from torch.optim.lr_scheduler import _LRScheduler
import math


def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def annealing_linear(start, end, pct):
    "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return (end - start) * pct + start
    
    
class CustomCyclicLR(_LRScheduler):
    def __init__(self, optimizer, steps_per_cycle=2000, mult_factor=1.0, base_lr=0.001, max_lr=0.01, 
                 base_lr_scale_fn=None, max_lr_scale_fn=None, scale_mode='cycle', 
                 pct_start=0.3, anneal_strategy='cos', last_epoch=-1, verbose=False):
        
        self.cycle_counter = 0
        self.step_counter = 0
        self.cur_cycle_length = steps_per_cycle 
        self.mult_factor = mult_factor 
        
        self.base_lr = base_lr 
        self.max_lr = max_lr
        
        # Validate scale_mode
        if scale_mode not in ['cycle', 'iterations']:
            raise ValueError("scale_mode must be one of 'cycle' or 'iterations', instead got {}".format(scale_mode))
        self.scale_mode = scale_mode 
        
        self.base_lr_scale_fn = self._default_scale_fn() if base_lr_scale_fn is None else base_lr_scale_fn
        self.max_lr_scale_fn = self._default_scale_fn() if max_lr_scale_fn is None else max_lr_scale_fn
        
        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError("Expected float between 0 and 1 pct_start, but got {}".format(pct_start))
        self.pct_start = pct_start
        
        # Validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError("anneal_strategy must be one of 'cos' or 'linear', instead got {}".format(anneal_strategy))
        elif anneal_strategy == 'cos':
            self.anneal_func = annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = annealing_linear
            
        super().__init__(optimizer, last_epoch, verbose)
    
    def _default_scale_fn(self):
        if self.scale_mode == 'cycle':
            return lambda cycle: 0.9**cycle
        elif self.scale_mode == 'iterations':
            return lambda iteration: 0.999995**iteration
            
    def get_lr(self):
        step_num = self.last_epoch 
        
        if step_num > self.step_counter + self.cur_cycle_length: 
            self.cycle_counter += 1 
            self.step_counter += self.cur_cycle_length 
            self.cur_cycle_length *= self.mult_factor
        
        if self.scale_mode == 'cycle': 
            init_lr = self.base_lr * self.base_lr_scale_fn(self.cycle_counter)
            final_lr = self.base_lr * self.base_lr_scale_fn(self.cycle_counter+1)
            peak_lr = self.max_lr * self.max_lr_scale_fn(self.cycle_counter+self.pct_start)
        elif self.scale_mode == 'iterations':
            init_lr = self.base_lr * self.base_lr_scale_fn(self.step_counter)
            final_lr = self.base_lr * self.base_lr_scale_fn(self.step_counter+self.cur_cycle_length)
            peak_lr = self.max_lr * self.max_lr_scale_fn(self.step_counter+self.pct_start*self.cur_cycle_length)
            
        x = (step_num - self.step_counter) / self.cur_cycle_length 
        
        if x < self.pct_start:
            pct = x / self.pct_start 
            lr = self.anneal_func(init_lr, peak_lr, pct)
        else:
            pct = (x - self.pct_start) / (1 - self.pct_start) 
            lr = self.anneal_func(peak_lr, final_lr, pct)
        return [lr] * len(self.optimizer.param_groups)
    
    def state_dict(self):
        state = super().state_dict()
        # We are dropping `base_lr_scale_fn` and `max_lr_scale_fn` attributes because 
        # lambda functions can't be pickled
        state.pop("base_lr_scale_fn")
        state.pop("max_lr_scale_fn")
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_lr_scale_fn = self._default_scale_fn()
        self.max_lr_scale_fn = self._default_scale_fn()
        