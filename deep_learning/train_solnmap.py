import argparse 
import os 
import numpy as np
import math 
import datetime
import torch
from torch import nn
from data import DataModule 
from model import SolutionMap
from initialization import *
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.profilers import AdvancedProfiler


print('Using pytorch', torch.__version__)
print('Using pytorch lightning', pl.__version__)


class FixedSequenceWeights(Callback):
    def __init__(self, weights):
        self.weights = torch.tensor(weights)
        
    def on_fit_start(self, trainer, pl_module):
        pl_module.set_sequence_weights(self.weights.to(pl_module.dtype).to(pl_module.device))
        

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
        self.prob = self.prob.to(pl_module.device)
        self.base_weights = self.base_weights.to(pl_module.dtype).to(pl_module.device)
#         self.base_factors = self.base_factors.to(pl_module.dtype).to(pl_module.device)
        pl_module.set_sequence_weights(self.base_weights*self.base_factors)
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        indices = torch.multinomial(self.prob, self.n2)
        random_weights = nn.functional.one_hot(indices, num_classes=self.sequence_len).sum(dim=0)
        pl_module.set_sequence_weights((random_weights + self.base_weights)*self.base_factors)


if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', default='lennardjones', help='problem name')
    ap.add_argument('--omega', default=300, type=float, help='FPU problem parameter')
    ap.add_argument('--Delta_t', default=1e-2, type=float, help='Delta t')
    ap.add_argument('--random_seed', default=42, type=int, help='random seed')
    ap.add_argument('--data_dir', default='.', help='data directory')
    ap.add_argument('--batch_size', default=100, type=int, help='batch size')
    ap.add_argument('--h2h_model', default='ResMLP', help='model (MLP, ResMLP, or HamiltonianReversibleNetwork)')
    ap.add_argument('--h2h_layer_sizes', default=[1000, 1000], nargs='+', type=int, help='h2h layer sizes')
    ap.add_argument('--i2h_layer_sizes', default=None, nargs='+', type=int, help='i2h layer sizes')
    ap.add_argument('--h2o_layer_sizes', default=None, nargs='+', type=int, help='h2o layer sizes')
    ap.add_argument('--activation', default='ELU', help='activation function')
    ap.add_argument('--sequence_weights', default=[1, 1, 1, 1, 1], nargs='+', type=int, help='sequence weights')
    ap.add_argument('--WS_strength', default=0., type=float, help='weight smoothness regularization strength')
    ap.add_argument('--S_strength', default=0., type=float, help='lagrangian regularization strength')
    ap.add_argument('--V_strength', default=0., type=float, help='transport cost regularization strength')
    ap.add_argument('--Comm_strength', default=0., type=float, help='commute with F_dt regularization strength')
    ap.add_argument('--weight_init', default='', help='weight initialization')
    ap.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    ap.add_argument('--lr_decay', default=1e-3, type=float, help='dacay rate of lambda lr scheduler')
    ap.add_argument('--lr_mult', default=1., type=float, help='multiplicative factor for lr of last layer')
    ap.add_argument('--num_epochs', default=1000, type=int, help='number of epochs')
    ap.add_argument('--steps_per_cycle', default=200000, type=int, help='steps per cycle of the custom cyclic lr scheduler')
    ap.add_argument('--gpus', default=[0], nargs='+', type=int, help='gpus')
    ap.add_argument('--resume_from_ckpt', default=None, help='resume from checkpoint')
    ap.add_argument('--init_model_ckpt', default=None, help='initialize model with checkpoint')
    args = ap.parse_args()
    
    # Config dictionary
    CONFIG = dict (
        group = args.group,
        problem_kwargs = {'Omega': args.omega},
        Delta_t = args.Delta_t,
        seed = args.random_seed,
        train_dir = os.path.join(args.data_dir, 'train'),
        test_dir = os.path.join(args.data_dir, 'test'),
        batch_size = args.batch_size,
        h2h_model = args.h2h_model,
        h2h_layer_sizes = args.h2h_layer_sizes,
        i2h_layer_sizes = args.i2h_layer_sizes,
        h2o_layer_sizes = args.h2o_layer_sizes,
        activation_fn = args.activation,
        activation_kwargs = {},
        use_bn = False,
        use_scale = True,
        init_gamma = 0.,
    #     loss_fn = 'MSELoss', loss_kwargs = {},
        loss_fn = 'MeanEnergyNormSquaredLoss',
        loss_kwargs = {'problem': args.group, 'problem_kwargs': {'Omega': args.omega}},
        optimizer_fn = 'AdamW',
    #     optimizer_fn = 'SGD',
    #     optimizer_kwargs = {},
        optimizer_kwargs = {'lr': args.lr, 'weight_decay': 1e-2}, 
    #     optimizer_kwargs = {'lr': args.lr, 'nesterov': False, 'momentum': 0.}, 
        lr_mult = args.lr_mult,
    #     lr_scheduler_fn = None,
        lr_scheduler_fn = 'OneCycleLR',
        lr_scheduler_kwargs = {'max_lr': [args.lr * args.lr_mult, args.lr], 
    #         'total_steps': 1600000,
            'epochs': args.num_epochs, 'steps_per_epoch': int(math.ceil(160000/args.batch_size)), 
            'anneal_strategy': 'cos', 'cycle_momentum': False, 'three_phase': False, 'pct_start': 0.0},
    #     lr_scheduler_fn = 'ReduceLROnPlateau',
    #     lr_scheduler_kwargs = {'factor': 0.85, 'patience': 5, 'cooldown': 5},
    #     lr_scheduler_fn = 'LambdaLR', 
    #     lr_scheduler_kwargs = {'lr_lambda': lambda epoch: 1./(1.+args.lr_decay*epoch)},
    #     lr_scheduler_fn = 'CustomCyclicLR', 
    #     lr_scheduler_kwargs = {'steps_per_cycle': args.steps_per_cycle, 'mult_factor': 1.0, 'base_lr': 0.0001, 'max_lr': args.lr, 'scale_mode': 'iterations', 'pct_start': 0.3, 'base_lr_scale_fn': lambda i: 0.999997**i, 'max_lr_scale_fn': lambda i: 0.999997**i, 'anneal_strategy': 'cos'}, 
        lr_scheduler_interval = 'step',
        weight_initialization = args.weight_init,
        H_strength = 0.,
        WS_strength = args.WS_strength,
        S_strength = args.S_strength, 
        V_strength = args.V_strength,
        Comm_strength = args.Comm_strength,
        sequence_weights = args.sequence_weights,
        sequence_len = len(args.sequence_weights),
        n1 = 3,
        n2 = 2,
        gpus = args.gpus,
        strategy = 'auto',
        num_epochs = args.num_epochs
    )

    # Seed everything
    seed_everything(CONFIG['seed'])

    # Get data
    data_module = DataModule(CONFIG['train_dir'], CONFIG['test_dir'], CONFIG['sequence_len'], CONFIG['batch_size'])

    # Define checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        save_top_k=3,
    #     every_n_epochs=10,
        save_last=True,
        save_weights_only=False,
        dirpath=None,
        filename='epoch{epoch:02d}-val_loss{val/loss:.3e}',
        auto_insert_metric_name=False,
        verbose=True,
        mode='min')

    # Define learning rate monitor 
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True)

    # Define sequence weights 
    fixed_seq_weights = FixedSequenceWeights(weights=CONFIG['sequence_weights'])
    # randomized_seq_weights = RandomizedSequenceWeights(sequence_len=CONFIG['sequence_len'], n1=CONFIG['n1'], n2=CONFIG['n2'])


    # Initialize model 
    lit_model = SolutionMap(
        h2h_model_name=CONFIG['h2h_model'],
        h2h_layer_sizes=CONFIG['h2h_layer_sizes'], 
        i2h_layer_sizes=CONFIG['i2h_layer_sizes'], 
        h2o_layer_sizes=CONFIG['h2o_layer_sizes'], 
        activation_fn=CONFIG['activation_fn'],
        activation_kwargs=CONFIG['activation_kwargs'],
        use_bn=CONFIG['use_bn'],
        use_scale=CONFIG['use_scale'],
        init_gamma=CONFIG['init_gamma'], 
        loss_fn=CONFIG['loss_fn'],
        loss_kwargs=CONFIG['loss_kwargs'],
        optimizer_fn=CONFIG['optimizer_fn'],
        optimizer_kwargs=CONFIG['optimizer_kwargs'],
        lr_scheduler_fn=CONFIG['lr_scheduler_fn'],
        lr_scheduler_kwargs=CONFIG['lr_scheduler_kwargs'],
        lr_scheduler_interval=CONFIG['lr_scheduler_interval'],
        lr_mult=CONFIG['lr_mult'],
        H_strength=CONFIG['H_strength'],
        WS_strength=CONFIG['WS_strength'],
        S_strength=CONFIG['S_strength'],
        V_strength=CONFIG['V_strength'],
        Comm_strength=CONFIG['Comm_strength'],
        problem=CONFIG['group'],
        problem_kwargs=CONFIG['problem_kwargs'],
        Delta_t=CONFIG['Delta_t'],
    ).double()

    if CONFIG['weight_initialization'] == 'xavier_uniform':
        xavier_uniform_init(lit_model)
    elif CONFIG['weight_initialization'] == 'xavier_normal':
        xavier_normal_init(lit_model)
    elif CONFIG['weight_initialization'] == 'kaiming_uniform':
        kaiming_uniform_init(lit_model)
    elif CONFIG['weight_initialization'] == 'kaiming_normal':
        kaiming_normal_init(lit_model)
    
    if args.init_model_ckpt is not None:
        checkpoint = torch.load(args.init_model_ckpt)
        lit_model.load_state_dict(checkpoint["state_dict"], strict=False)


    # Initialize W&B logger 
    if args.resume_from_ckpt is not None: 
        id = args.resume_from_ckpt.split('/')[-3]
        wandb_logger = WandbLogger(project='solutionmap', group=CONFIG['group'], id=id, resume='must')
    else:
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        wandb_logger = WandbLogger(name=current_time, project='solutionmap', group=CONFIG['group'], version=0, config=CONFIG)

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=CONFIG['gpus'],
        strategy=CONFIG['strategy'],
        max_epochs=CONFIG['num_epochs'],
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger,
        callbacks=[
            fixed_seq_weights,
    #                randomized_seq_weights,
                   lr_monitor, 
                   checkpoint_callback
                  ],
        profiler=AdvancedProfiler(dirpath=".", filename="prof_logs"),
    )

    # Log gradients and model topology
    wandb_logger.watch(lit_model, log="all", log_freq=500)

    # Fit data 
    if args.resume_from_ckpt is not None:
        trainer.fit(lit_model, datamodule=data_module, ckpt_path=args.resume_from_ckpt)
    else:
        trainer.fit(lit_model, datamodule=data_module)

    print("Best model saved to:\n", checkpoint_callback.best_model_path)

    # Close W&B logger
    wandb_logger.finalize("success")
