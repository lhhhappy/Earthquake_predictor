# train.py
import json
from torch.utils.data import DataLoader
from model import ES_net
from argparse import ArgumentParser
from lightning.pytorch import loggers as pl_loggers
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from model import LightingModel
from dataset import get_dataset
from loss import get_loss

parser = ArgumentParser()

parser.add_argument('--energy-loss', type=str, default='NNSE', help='Type of energy loss function')
parser.add_argument('--day-loss', type=str, default='cross_entropy', help='Type of day prediction loss function')
parser.add_argument("--data-path", type=str, required=True, help='Path to the dataset')
parser.add_argument("--batch-size", type=int, default=4, help='Batch size for training')
parser.add_argument("--val-batch-size", type=int, default=1, help='Batch size for validation')
parser.add_argument("--log-dir", type=str, default="logs", help='Directory for TensorBoard logs')
parser.add_argument("--model-arch", type=str, default="ES_net", help='Model architecture to use')
parser.add_argument("--save-dir", type=str, default="save_dir", help='Directory to save checkpoints')
parser.add_argument("--max-epochs", type=int, default=100, help='Maximum number of epochs')
parser.add_argument("--device", type=int, default=0, help='Device index for training')
parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate')
parser.add_argument("--model_params", type=str, default="{}", help='JSON string for model parameters')
parser.add_argument("--window-size", type=int, default=14*100, help='Window size for input data')
parser.add_argument("--forecast-horizon", type=int, default=14, help='Forecast horizon')
parser.add_argument("--lape-dim", type=int, default=30, help='Dimensionality for Laplacian embedding')
parser.add_argument("--far-mask-delta", type=int, default=30, help='Delta value for far masks')
parser.add_argument("--dtw-delta", type=int, default=10, help='Delta value for DTW masks')

args = parser.parse_args()

# Load dataset
dataset = get_dataset(
    data_dir=args.data_path,
    window_size=args.window_size,
    forecast_horizon=args.forecast_horizon,
    lape_dim=args.lape_dim,
    far_mask_delta=args.far_mask_delta,
    dtw_delta=args.dtw_delta
)

# Define loss function
loss_fn = get_loss(energy_loss=args.energy_loss, day_loss=args.day_loss)

# Split dataset into training and validation sets
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]
)

# Create DataLoaders for training and validation
train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=dataset.collate_fn
)

# Load model parameters from JSON string
with open(args.model_params, 'r') as f:
    model_par = json.load(f)

model_arch_dict = {
    "ES_net": ES_net
}

# Initialize the model
model_class = model_arch_dict.get(args.model_arch)
if model_class:
    model = LightingModel(
        model_class,
        lr=args.lr,
        max_epoch=args.max_epochs,
        loss_fns=loss_fn,
        **model_par
    )
else:
    raise ValueError(f"Unknown model architecture: {args.model_arch}")

# Define checkpoint and learning rate monitor callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_psnr',
    dirpath=args.save_dir,
    filename=args.save_dir + 'Val-{epoch:02d}-{val_psnr:.2f}',
    save_top_k=1,
    mode='max',
    save_last=True,
    verbose=True,
    every_n_epochs=50
)

lr_monitor = LearningRateMonitor(logging_interval='step')

# Initialize Trainer
trainer = Trainer(
    max_epochs=args.max_epochs,
    devices=[args.device],
    logger=pl_loggers.TensorBoardLogger(args.log_dir),
    callbacks=[checkpoint_callback, lr_monitor],
    log_every_n_steps=20
)

# Start training
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)