import argparse 
import os 
import numpy as np
import pandas as pd
import datetime
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from model import CorrectionOperator
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback


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



def split_dataset(ds, train_fraction, seed):
    n_full = len(ds)
    n_train = int(train_fraction*n_full)
    n_test = n_full - n_train 
    ds_train, ds_test = random_split(ds, [n_train, n_test], generator=torch.Generator().manual_seed(seed))
    return ds_train, ds_test 


def get_dataset(data_dir, sequence_len):

    filenames = [f"U{n}.csv" for n in range(sequence_len+1)]
    data = []
    for fname in filenames: 
        u = pd.read_csv(os.path.join(data_dir, fname)).to_numpy()
    #    v, x = u[:, :14], u[:, 14:]
    #    u = np.concatenate((v/100., x), axis=1)
        data.append(torch.tensor(u))
    ds = TensorDataset(*data)
    return ds



if __name__ == '__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', default='lennardjones', help='problem name')
    ap.add_argument('--Delta_t', default=1e-4, type=float, help='Delta t')
    ap.add_argument('--coarse_h', default=1e-4, type=float, help='coarse h')
    ap.add_argument('--random_seed', default=42, type=int, help='random seed')
    ap.add_argument('--data_dir', default='.', help='data directory')
    ap.add_argument('--batch_size', default=100, type=int, help='batch size')
    ap.add_argument('--model', default='ResMLP', help='model (MLP, ResMLP, or HamiltonianReversibleNetwork)')
    ap.add_argument('--layer_sizes', default=[28, 1000, 1000, 28], nargs='+', type=int, help='layer sizes')
    ap.add_argument('--ws_strength', default=0., type=float, help='weight smoothness regularization strength')
    ap.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    ap.add_argument('--num_epochs', default=1000, type=int, help='number of epochs')
    ap.add_argument('--sequence_weights', default=[1, 1, 1, 1, 1], nargs='+', type=int, help='sequence weights')
    ap.add_argument('--gpus', default=[0], nargs='+', type=int, help='gpus')
    ap.add_argument('--resume_from_ckpt', default=None, help='resume from checkpoint')
    ap.add_argument('--init_model_ckpt', default=None, help='initialize model with checkpoint')
    args = ap.parse_args()
    
    # Config dictionary
    CONFIG = dict (
        group = args.group,
        Delta_t = args.Delta_t,
        coarse_h = args.coarse_h,
        seed = args.random_seed,
        train_dir = os.path.join(args.data_dir, 'train'),
        test_dir = os.path.join(args.data_dir, 'test'),
        batch_size = args.batch_size,
        model = args.model,
        layer_sizes = args.layer_sizes,
        activation_fn = 'ELU',
        activation_kwargs = {},
        use_bn = False,
        use_scale = True,
        loss_fn = 'MSELoss',
        optimizer_fn = 'AdamW',
    #     optimizer_fn = 'SGD',
        optimizer_kwargs = {'lr': args.lr, 'weight_decay': 1e-2}, 
    #     optimizer_kwargs = {'lr': 1e-2, 'momentum': 0.9, 'weight_decay': 1e-4}, 
    #     lr_scheduler_fn = None, lr_scheduler_kwargs = {},
    #     lr_scheduler_fn = 'CyclicLR',
    #     lr_scheduler_kwargs = {'brase_lr': 1e-5, 'max_lr': 1e-3, 'step_size_up': 50000, 'step_size_down': 50000, 'mode': 'triangular2', 'cycle_momentum': True},
        lr_scheduler_fn = 'OneCycleLR',
        lr_scheduler_kwargs = {'max_lr': args.lr, 'epochs': args.num_epochs, 'steps_per_epoch': 1600, 'anneal_strategy': 'cos', 'cycle_momentum': False, 'three_phase': False, 'pct_start': 0.3},
    #     lr_scheduler_fn = 'ReduceLROnPlateau',
    #     lr_scheduler_kwargs = {'factor': 0.85, 'patience': 5, 'cooldown': 5},
        lr_scheduler_interval = 'step',
        H_strength = 0.,
        WS_strength = args.ws_strength,
        sequence_weights = args.sequence_weights,
        sequence_len = len(args.sequence_weights),
#         n1 = 3,
#         n2 = 2,
        gpus = args.gpus,
        strategy = None,
        num_epochs = args.num_epochs
    )

    # Seed everything
    seed_everything(CONFIG['seed'])

    # Get datasets
    ds_train = get_dataset(CONFIG['train_dir'], CONFIG['sequence_len'])
    ds_test = get_dataset(CONFIG['test_dir'], CONFIG['sequence_len'])

    print("U_n (n=0,1,...,{0}) train: {1}".format(len(ds_train[:])-1, ds_train[:][0].shape))
    print("U_n (n=0,1,...,{0}) test: {1}".format(len(ds_test[:])-1, ds_test[:][0].shape))

    train_loader = DataLoader(ds_train, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # Define checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/loss',
        save_top_k=3,
        save_last=True,
        save_weights_only=False,
        dirpath=None,
        filename='epoch{epoch:02d}-val_loss{val/loss:.3e}',
        auto_insert_metric_name=False,
        verbose=True,
        mode='min')

    # Define learning rate monitor 
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    
    # Define sequence weights 
    fixed_seq_weights = FixedSequenceWeights(weights=CONFIG['sequence_weights'])
    # randomized_seq_weights = RandomizedSequenceWeights(sequence_len=CONFIG['sequence_len'], n1=CONFIG['n1'], n2=CONFIG['n2'])

    # Initialize model 
    if args.init_model_ckpt is not None:
        lit_model = CorrectionOperator.load_from_checkpoint(args.init_model_ckpt,
            loss_fn=CONFIG['loss_fn'],
            optimizer_fn=CONFIG['optimizer_fn'],
            optimizer_kwargs=CONFIG['optimizer_kwargs'],
            lr_scheduler_fn=CONFIG['lr_scheduler_fn'],
            lr_scheduler_kwargs=CONFIG['lr_scheduler_kwargs'],
            lr_scheduler_interval=CONFIG['lr_scheduler_interval'],
            H_strength=CONFIG['H_strength'],
            WS_strength=CONFIG['WS_strength']
        ).double()    
    else:
        lit_model = CorrectionOperator(
            model_name=CONFIG['model'],
            layer_sizes=CONFIG['layer_sizes'], 
            activation_fn=CONFIG['activation_fn'],
            activation_kwargs=CONFIG['activation_kwargs'],
            use_bn=CONFIG['use_bn'],
            use_scale=CONFIG['use_scale'],
            loss_fn=CONFIG['loss_fn'],
            optimizer_fn=CONFIG['optimizer_fn'],
            optimizer_kwargs=CONFIG['optimizer_kwargs'],
            lr_scheduler_fn=CONFIG['lr_scheduler_fn'],
            lr_scheduler_kwargs=CONFIG['lr_scheduler_kwargs'],
            lr_scheduler_interval=CONFIG['lr_scheduler_interval'],
            H_strength=CONFIG['H_strength'],
            WS_strength=CONFIG['WS_strength'],
            problem=CONFIG['group'],
            Delta_t=CONFIG['Delta_t'],
            coarse_h=CONFIG['coarse_h'],
        ).double()

    
    # Initialize W&B logger
    if args.resume_from_ckpt is not None:
        id = args.resume_from_ckpt.split('/')[-3]
        wandb_logger = WandbLogger(project='solutionmap', group=CONFIG['group'], id=id, resume='must')
    else:
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        wandb_logger = WandbLogger(name=current_time, project='solutionmap', group=CONFIG['group'], version=0, config=CONFIG)

    # Initialize trainer
    trainer = pl.Trainer(
        gpus=CONFIG['gpus'],
        strategy=CONFIG['strategy'],
        max_epochs=CONFIG['num_epochs'],
        logger=wandb_logger,
        callbacks=[
            fixed_seq_weights,
    #                randomized_seq_weights,
                   lr_monitor, 
                   checkpoint_callback
             ], 
    )
    # Log gradients and model topology
    wandb_logger.watch(lit_model, log='all', log_freq=500)

    # Fit data 
    if args.resume_from_ckpt is not None:
        trainer.fit(lit_model, train_loader, test_loader, ckpt_path=args.resume_from_ckpt)
    else:
        trainer.fit(lit_model, train_loader, test_loader)

    print("Best model saved to:\n", checkpoint_callback.best_model_path)

    # Close W&B logger
    wandb_logger.finalize('success')
