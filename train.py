# train.py
import json
from torch.utils.data import DataLoader
from model import ES_net, ES_net_mixer, Mlpbaseline
from argparse import ArgumentParser
from lightning.pytorch import loggers as pl_loggers
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from model import LightingModel
from dataset import get_dataset, create_data_loader
from loss import get_loss
import lightning
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


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
parser.add_argument("--history-window", type=int, default=14*100, help='Window size for input data')
parser.add_argument("--forecast-window", type=int, default=14, help='Forecast horizon')
parser.add_argument("--lape-dim", type=int, default=30, help='Dimensionality for Laplacian embedding')
parser.add_argument("--geo-percentage", type=float, default=0.5, help='Percentage of geo features')
parser.add_argument("--sem-percentage", type=float, default=0.5, help='Percentage of semantic features')
parser.add_argument("--time-resolution", type=int, default=1, help='Time resolution for the dataset')
parser.add_argument("--seed", type=int, default=42, help='Seed for reproducibility')
parser.add_argument("--earthquake-catalog-window", type=int, default=14, help='Window size for earthquake catalog')
parser.add_argument("--finetune-from-model", type=str, default=None, help='Path to the model to finetune')
parser.add_argument("--start-date", type=str, default=None, help='Start date for training')
parser.add_argument("--last-date", type=str, default=None, help='End date for training')
parser.add_argument("--val-date", type=str, default=None, help='Date for validation')
parser.add_argument("--train-percentage", type=float, default=0.8, help='Percentage of data to use for training')
parser.add_argument("--use-area", type=str, default=None, help='Area to use for training')
parser.add_argument("--spilt-by-earthquake-happened", type=bool, default=False, help='Split the dataset by earthquake happened or not') 
parser.add_argument("--use-train-loader", type=str, default=None, help='Use train loader for training')
parser.add_argument("--use-val-loader", type=str, default=None, help='Use val loader for validation')

args = parser.parse_args()

# Load dataset
lightning.seed_everything(args.seed)


dataset_dict = get_dataset(
    data_dir=args.data_path,
    window_size=args.history_window,
    forecast_horizon=args.forecast_window,
    lape_dim=args.lape_dim,
    geo_percentage=args.geo_percentage,
    sem_percentage=args.sem_percentage,
    time_resolution=args.time_resolution,
    earthquake_catalog_window=args.earthquake_catalog_window,
    start_date=args.start_date,
    last_date=args.last_date,
    val_date=args.val_date,
    train_percentage=args.train_percentage,
    use_area=args.use_area,
    split_by_earthquake_happen=args.spilt_by_earthquake_happened
)

# 创建 DataLoader
loader_dict = {}
for key, dataset in dataset_dict.items():
    is_train = "train" in key
    loader_dict[key] = DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else args.val_batch_size,
        shuffle=is_train,
        collate_fn=dataset.collate_fn,
        num_workers=127
    )

train_loader_key = args.use_train_loader or ("train" if not args.spilt_by_earthquake_happened else "train_happen")
val_loader_key = args.use_val_loader or ("val" if not args.spilt_by_earthquake_happened else "val_happen")

train_loader = loader_dict.get(train_loader_key)
val_loader = loader_dict.get(val_loader_key)

if train_loader is None or val_loader is None:
    raise ValueError(f"Invalid loader keys: train_loader_key={train_loader_key}, val_loader_key={val_loader_key}")


loss_fn = get_loss(energy_loss=args.energy_loss, day_loss=args.day_loss)

with open(args.model_params, 'r') as f:
    model_par = json.load(f)

model_arch_dict = {
    "ES_net": ES_net,
    "ES_net_mixer": ES_net_mixer,
    "Mlpbaseline": Mlpbaseline
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
    monitor='TPR',
    dirpath=args.save_dir,
    filename='Val-{epoch:02d}-{TPR:.2f}',
    save_top_k=1,
    mode='max',
    save_last=True,
    verbose=True,
    every_n_epochs=1
)

lr_monitor = LearningRateMonitor(logging_interval='step')

# Initialize Trainer
trainer = Trainer(
    max_epochs=args.max_epochs,
    devices=[args.device],
    logger=pl_loggers.TensorBoardLogger(args.log_dir),
    callbacks=[checkpoint_callback, lr_monitor],
    log_every_n_steps=20,
    gradient_clip_val=1,
    gradient_clip_algorithm="norm")


if args.finetune_from_model:
    model = LightingModel.load_from_checkpoint(args.finetune_from_model,map_location='cpu',loss_fns=loss_fn)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
else:
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)