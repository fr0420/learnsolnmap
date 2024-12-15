import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return end + (start - end) * (math.cos(math.pi * pct) + 1) / 2.0


def annealing_linear(start, end, pct):
    "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return (end - start) * pct + start
    
    
# class CustomCyclicLR(_LRScheduler):
#     def __init__(self, optimizer, steps_per_cycle=2000, mult_factor=1.0, base_lr=0.001, max_lr=0.01, 
#                  base_lr_scale_fn=None, max_lr_scale_fn=None, scale_mode='cycle', 
#                  pct_start=0.3, anneal_strategy='cos', last_epoch=-1, verbose=False):
        
#         self.cycle_counter = 0
#         self.step_counter = 0
#         self.cur_cycle_length = steps_per_cycle 
#         self.mult_factor = mult_factor 
        
#         self.base_lr = base_lr 
#         self.max_lr = max_lr
        
#         # Validate scale_mode
#         if scale_mode not in ['cycle', 'iterations']:
#             raise ValueError("scale_mode must be one of 'cycle' or 'iterations', instead got {}".format(scale_mode))
#         self.scale_mode = scale_mode 
        
#         self.base_lr_scale_fn = self._default_scale_fn() if base_lr_scale_fn is None else base_lr_scale_fn
#         self.max_lr_scale_fn = self._default_scale_fn() if max_lr_scale_fn is None else max_lr_scale_fn
        
#         # Validate pct_start
#         if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
#             raise ValueError("Expected float between 0 and 1 pct_start, but got {}".format(pct_start))
#         self.pct_start = pct_start
        
#         # Validate anneal_strategy
#         if anneal_strategy not in ['cos', 'linear']:
#             raise ValueError("anneal_strategy must be one of 'cos' or 'linear', instead got {}".format(anneal_strategy))
#         elif anneal_strategy == 'cos':
#             self.anneal_func = annealing_cos
#         elif anneal_strategy == 'linear':
#             self.anneal_func = annealing_linear
            
#         super().__init__(optimizer, last_epoch, verbose)
    
#     def _default_scale_fn(self):
#         if self.scale_mode == 'cycle':
#             return lambda cycle: 0.9**cycle
#         elif self.scale_mode == 'iterations':
#             return lambda iteration: 0.999995**iteration
            
#     def get_lr(self):
#         step_num = self.last_epoch 
        
#         if step_num > self.step_counter + self.cur_cycle_length: 
#             self.cycle_counter += 1 
#             self.step_counter += self.cur_cycle_length 
#             self.cur_cycle_length *= self.mult_factor
        
#         if self.scale_mode == 'cycle': 
#             init_lr = self.base_lr * self.base_lr_scale_fn(self.cycle_counter)
#             final_lr = self.base_lr * self.base_lr_scale_fn(self.cycle_counter+1)
#             peak_lr = self.max_lr * self.max_lr_scale_fn(self.cycle_counter+self.pct_start)
#         elif self.scale_mode == 'iterations':
#             init_lr = self.base_lr * self.base_lr_scale_fn(self.step_counter)
#             final_lr = self.base_lr * self.base_lr_scale_fn(self.step_counter+self.cur_cycle_length)
#             peak_lr = self.max_lr * self.max_lr_scale_fn(self.step_counter+self.pct_start*self.cur_cycle_length)
            
#         x = (step_num - self.step_counter) / self.cur_cycle_length 
        
#         if x < self.pct_start:
#             pct = x / self.pct_start 
#             lr = self.anneal_func(init_lr, peak_lr, pct)
#         else:
#             pct = (x - self.pct_start) / (1 - self.pct_start) 
#             lr = self.anneal_func(peak_lr, final_lr, pct)
#         return [lr] * len(self.optimizer.param_groups)
    
#     def state_dict(self):
#         state = super().state_dict()
#         # We are dropping `base_lr_scale_fn` and `max_lr_scale_fn` attributes because 
#         # lambda functions can't be pickled
#         state.pop("base_lr_scale_fn")
#         state.pop("max_lr_scale_fn")
#         return state

#     def load_state_dict(self, state_dict):
#         super().load_state_dict(state_dict)
#         self.base_lr_scale_fn = self._default_scale_fn()
#         self.max_lr_scale_fn = self._default_scale_fn()
    


class CustomCyclicLR(_LRScheduler):
    """
    Custom cyclic learning rate scheduler with adjustable scaling and annealing strategies.
    """

    def __init__(
            self, 
            optimizer: torch.optim.Optimizer, 
            steps_per_cycle: int = 2000, 
            mult_factor: float = 1.0,
            base_lr: float = 0.001, 
            max_lr: float = 0.01, 
            scale_mode: str = "cycle",
            scale_fn_type: str = "exponential", 
            scale_fn_factor: float = 0.9, 
            pct_start: float = 0.3, 
            anneal_strategy: float = "cos", 
            last_epoch: int = -1, 
            verbose: bool = False
        ) -> None:
        """
        Initialize the scheduler.

        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            steps_per_cycle (int): Number of steps per cycle. Default is 2000.
            mult_factor (float): Multiplicative factor for increasing cycle length. Default is 1.0.
            base_lr (float): Initial learning rate at the start of each cycle. Default is 0.001.
            max_lr (float): Peak learning rate within a cycle. Default is 0.01.
            scale_mode (str): Mode for scaling ('cycle' or 'iterations'). Default is 'cycle'.
            scale_fn_type (str): Scaling function type ('exponential' or 'linear'). Default is 'exponential'.
            scale_fn_factor (float): Factor used in the scaling function. Default is 0.9.
            pct_start (float): Percentage of the cycle for increasing learning rate. Default is 0.3.
            anneal_strategy (str): Annealing strategy ('cos' or 'linear'). Default is 'cos'.
            last_epoch (int): Index of last epoch. Default is -1.
            verbose (bool): If True, prints a message to stdout for each update. Default is False.

        Raises:
            ValueError: If an invalid value is passed for `scale_mode`, `scale_fn_type`, 
                        `pct_start`, or `anneal_strategy`.
        """
        self.steps_per_cycle = steps_per_cycle
        self.mult_factor = mult_factor
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.scale_mode = self._validate_scale_mode(scale_mode)
        self.pct_start = self._validate_pct_start(pct_start)
        self.anneal_func = self._get_anneal_func(anneal_strategy)
        self.scale_fn_type = self._validate_scale_fn_type(scale_fn_type)
        self.scale_fn_factor = scale_fn_factor

        self.cycle_counter = 0
        self.step_counter = 0
        self.cur_cycle_length = steps_per_cycle

        super().__init__(optimizer, last_epoch, verbose)

    def _validate_scale_mode(self, scale_mode):
        if scale_mode not in ["cycle", "iterations"]:
            raise ValueError(f"scale_mode must be 'cycle' or 'iterations', got '{scale_mode}'")
        return scale_mode

    def _validate_pct_start(self, pct_start):
        if not (0.0 <= pct_start <= 1.0):
            raise ValueError(f"pct_start must be between 0 and 1, got {pct_start}")
        return pct_start

    def _validate_scale_fn_type(self, scale_fn_type):
        if scale_fn_type not in ["exponential", "linear"]:
            raise ValueError(f"scale_fn_type must be 'exponential' or 'linear', got '{scale_fn_type}'")
        return scale_fn_type

    def _get_anneal_func(self, strategy):
        if strategy == "cos":
            return annealing_cos
        elif strategy == "linear":
            return annealing_linear
        else:
            raise ValueError(f"Invalid anneal_strategy: {strategy}")

    def _default_scale_fn(self, step):
        """
        Generates the scaling factor based on the type and current step/cycle.

        Args:
            step (int): Current step or cycle count.

        Returns:
            float: Scaling factor.
        """
        if self.scale_fn_type == "exponential":
            return self.scale_fn_factor ** step
        return 1 - self.scale_fn_factor * step

    def get_lr(self):
        """
        Compute the current learning rate based on the cycle, progress, and annealing strategy.

        Returns:
            list: List of learning rates for each parameter group in the optimizer.
        """
        step_num = self.last_epoch

        # Check if we need to start a new cycle
        if step_num >= self.step_counter + self.cur_cycle_length:
            self.cycle_counter += 1
            self.step_counter += self.cur_cycle_length
            self.cur_cycle_length = int(self.cur_cycle_length * self.mult_factor)

        # Compute the learning rates for the current cycle
        init_lr, peak_lr, final_lr = self._compute_cycle_lrs()
        progress = (step_num - self.step_counter) / self.cur_cycle_length

        # Determine the current learning rate based on progress within the cycle
        if progress < self.pct_start:
            lr = self.anneal_func(init_lr, peak_lr, progress / self.pct_start)
        else:
            pct = (progress - self.pct_start) / (1 - self.pct_start)
            lr = self.anneal_func(peak_lr, final_lr, pct)

        return [lr] * len(self.optimizer.param_groups)

    def _compute_cycle_lrs(self):
        """
        Compute the initial, peak, and final learning rates for the current cycle.

        Returns:
            tuple: (init_lr, peak_lr, final_lr)
        """
        if self.scale_mode == "cycle":
            init_lr = self.base_lr * self._default_scale_fn(self.cycle_counter)
            final_lr = self.base_lr * self._default_scale_fn(self.cycle_counter + 1)
            peak_lr = self.max_lr * self._default_scale_fn(self.cycle_counter + self.pct_start)
        else:
            init_lr = self.base_lr * self._default_scale_fn(self.step_counter)
            final_lr = self.base_lr * self._default_scale_fn(self.step_counter + self.cur_cycle_length)
            peak_lr = self.max_lr * self._default_scale_fn(self.step_counter + self.pct_start * self.cur_cycle_length)
        return init_lr, peak_lr, final_lr

    def state_dict(self):
        """
        Return the state of the scheduler for checkpointing.

        Returns:
            dict: Scheduler state.
        """
        state = super().state_dict()
        state.update({
            "cycle_counter": self.cycle_counter,
            "step_counter": self.step_counter,
            "cur_cycle_length": self.cur_cycle_length,
        })
        return state

    def load_state_dict(self, state_dict):
        """
        Load the state of the scheduler from a checkpoint.

        Args:
            state_dict (dict): Scheduler state.
        """
        super().load_state_dict(state_dict)
        self.cycle_counter = state_dict.get("cycle_counter", 0)
        self.step_counter = state_dict.get("step_counter", 0)
        self.cur_cycle_length = state_dict.get("cur_cycle_length", self.steps_per_cycle)
