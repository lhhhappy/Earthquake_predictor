import torch
import lightning as L
from .ES_net import ES_net

class LightingModel(L.LightningModule):
    def __init__(self, model, lr=1e-3, max_epoch=300, loss_fns=None, **kwargs):
        super(LightingModel, self).__init__()
        self.model = model(**kwargs)
        self.loss_fns = loss_fns if loss_fns is not None else {}
        self.max_epoch = max_epoch
        self.lr = lr
        self.save_hyperparameters(ignore=['loss_fns'])
    
    def compute_loss(self, energy_predict, log_energy_future, day_predict, earthquake_data_future_day):
        
        total_loss = 0.0
        # Iterate through the loss functions and apply them

        for name, loss_fn in self.loss_fns.items():
            if name == 'energy_loss':
                loss = loss_fn(energy_predict, log_energy_future)
            elif name == 'day_loss':
                loss = loss_fn(day_predict, earthquake_data_future_day)
            else:
                raise ValueError(f"Unknown loss function: {name}")
            total_loss += loss
            # Log each loss component for better monitoring
            self.log(f"{name}", loss, prog_bar=True)
        return total_loss
    
    def training_step(self, batch, batch_idx):
        log_energy_history = batch['log_energy_history']
        gnss_data_history = batch['gnss_data_history']
        log_energy_future = batch['log_energy_future']
        earthquake_data_future_day = batch['earthquake_data_future_day']
        es_geo_masks = batch['es_geo_mask']
        es_sem_masks = batch['es_sem_mask']
        combined_gnss_masks = batch['combined_gnss_mask']
        lap_ex_masks = batch['lap_ex']  
        lap_gnss_masks = batch['lap_gnss']
        
        # Forward pass through the model
        energy_predict, day_predict = self.model(
            log_energy_history, 
            gnss_data_history, 
            lap_mx=lap_ex_masks,
            gnss_lap_mx=lap_gnss_masks,
            es_geo_mask=es_geo_masks,
            es_sem_mask=es_sem_masks,
            gnss_geo_mask=combined_gnss_masks
        )
        
        # Compute the loss using the helper function
        train_loss = self.compute_loss(
            energy_predict, log_energy_future, 
            day_predict, earthquake_data_future_day
        )
        self.log("train_loss", train_loss)
        return train_loss
    

    def validation_step(self, batch, batch_idx):
        log_energy_history = batch['log_energy_history']
        gnss_data_history = batch['gnss_data_history']
        log_energy_future = batch['log_energy_future']
        earthquake_data_future_day = batch['earthquake_data_future_day']
        es_geo_masks = batch['es_geo_mask']
        es_sem_masks = batch['es_sem_mask']
        combined_gnss_masks = batch['combined_gnss_mask']
        lap_ex_masks = batch['lap_ex']  
        lap_gnss_masks = batch['lap_gnss']
        
        # Forward pass through the model
        energy_predict, day_predict = self.model(
            log_energy_history, 
            gnss_data_history, 
            lap_mx=lap_ex_masks,
            gnss_lap_mx=lap_gnss_masks,
            es_geo_mask=es_geo_masks,
            es_sem_mask=es_sem_masks,
            gnss_geo_mask=combined_gnss_masks
        )
        
        # Compute the loss using the helper function
        val_loss = self.compute_loss(
            energy_predict, log_energy_future, 
            day_predict, earthquake_data_future_day
        )
        self.log("val_loss", val_loss)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch, eta_min=1e-5)
        return [optimizer], [scheduler]
    
    def forward(self, batch, batch_idx = None):
        log_energy_history = batch['log_energy_history']
        gnss_data_history = batch['gnss_data_history']
        es_geo_masks = batch['es_geo_mask']
        es_sem_masks = batch['es_sem_mask']
        combined_gnss_masks = batch['combined_gnss_mask']
        lap_ex_masks = batch['lap_ex']  
        lap_gnss_masks = batch['lap_gnss']
        
        # Forward pass through the model
        energy_predict, day_predict = self.model(
            log_energy_history, 
            gnss_data_history, 
            lap_mx=lap_ex_masks,
            gnss_lap_mx=lap_gnss_masks,
            es_geo_mask=es_geo_masks,
            es_sem_mask=es_sem_masks,
            gnss_geo_mask=combined_gnss_masks
        )
        return energy_predict, day_predict
