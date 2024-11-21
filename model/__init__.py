import torch
import lightning as L
from .ES_net import ES_net
from .ES_net_mixer import ES_net_mixer

def cuda_dict(d,model_device):
    for k, v in d.items():
        if v.device.type == model_device:
            continue    
        else:
            d[k] = v.to(model_device)
    return d

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
            if name == 'energy_loss' and loss_fn is not None:
                loss = loss_fn(energy_predict, log_energy_future)
            elif name == 'day_loss' and loss_fn is not None and day_predict is not None:
                loss = loss_fn(day_predict, earthquake_data_future_day)
            else:
                continue
            total_loss += loss
            self.log(f"{name}", loss, prog_bar=True)
        return total_loss
    
    def on_after_backward(self):
        
        total_grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_grad_norm = p.grad.data.norm(2)
                total_grad_norm += param_grad_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        # 记录梯度 norm
        self.log("train_grad_norm", total_grad_norm, prog_bar=True)

    def training_step(self, batch, batch_idx):
        
        log_energy_history = batch['log_energy_history']
        gnss_data_history = batch['gnss_data_history']
        log_energy_future = batch['log_energy_future']
        earthquake_data_future_day = batch['earthquake_data_future_day']
        es_geo_masks = batch['es_geo_mask']
        es_sem_masks = batch['es_sem_mask']
        combined_gnss_masks = batch['gnss_geo_mask']
        es_lap_masks = batch['lap_ex']  
        lap_gnss_masks = batch['lap_gnss']
        gnss_paddding_mask = batch['gnss_padding_mask']
        earthquake_loaction = batch['earthquake_location']
        station_loaction = batch['station_location']

        # Forward pass through the model
        energy_predict, day_predict = self.model(
            log_energy_history, 
            gnss_data_history, 
            es_lap_mx=es_lap_masks,
            gnss_lap_mx=lap_gnss_masks,
            es_geo_mask=es_geo_masks,
            es_sem_mask=es_sem_masks,
            gnss_geo_mask=combined_gnss_masks,
            gnss_padding_mask=gnss_paddding_mask,
            es_loc = earthquake_loaction,
            gnss_loc = station_loaction
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
        combined_gnss_masks = batch['gnss_geo_mask']
        es_lap_masks = batch['lap_ex']  
        lap_gnss_masks = batch['lap_gnss']
        gnss_paddding_mask = batch['gnss_padding_mask']
        earthquake_loaction = batch['earthquake_location']
        station_loaction = batch['station_location']

        energy_predict, day_predict = self.model(
            log_energy_history, 
            gnss_data_history, 
            es_lap_mx=es_lap_masks,
            gnss_lap_mx=lap_gnss_masks,
            es_geo_mask=es_geo_masks,
            es_sem_mask=es_sem_masks,
            gnss_geo_mask=combined_gnss_masks,
            gnss_padding_mask=gnss_paddding_mask,
            es_loc = earthquake_loaction,
            gnss_loc = station_loaction
        )
        
        # Compute the loss using the helper function
        val_loss = self.compute_loss(
            energy_predict, log_energy_future, 
            day_predict, earthquake_data_future_day
        )
        self.log("val_loss", val_loss)
        return val_loss
    
    def predict_step(self, batch, batch_idx):
        batch = cuda_dict(batch,self.device)
        log_energy_history = batch['log_energy_history']
        gnss_data_history = batch['gnss_data_history']
        log_energy_future = batch['log_energy_future']
        earthquake_data_future_day = batch['earthquake_data_future_day']
        es_geo_masks = batch['es_geo_mask']
        es_sem_masks = batch['es_sem_mask']
        combined_gnss_masks = batch['gnss_geo_mask']
        es_lap_masks = batch['lap_ex']
        lap_gnss_masks = batch['lap_gnss']
        gnss_paddding_mask = batch['gnss_padding_mask']
        earthquake_loaction = batch['earthquake_location']
        station_loaction = batch['station_location']
        earthquake_happen = batch['earthquake_happen']

        energy_predict, day_predict = self.model(
            log_energy_history, 
            gnss_data_history, 
            es_lap_mx=es_lap_masks,
            gnss_lap_mx=lap_gnss_masks,
            es_geo_mask=es_geo_masks,
            es_sem_mask=es_sem_masks,
            gnss_geo_mask=combined_gnss_masks,
            gnss_padding_mask=gnss_paddding_mask,
            es_loc = earthquake_loaction,
            gnss_loc = station_loaction,
            
        )

        return {
            'log_energy_history': log_energy_history,
            'gnss_data_history': gnss_data_history,
            'log_energy_future': log_energy_future,
            'earthquake_data_future_day': earthquake_data_future_day,
            'es_geo_mask': es_geo_masks,
            'es_sem_mask': es_sem_masks,
            'gnss_geo_mask': combined_gnss_masks,
            'lap_ex': es_lap_masks,
            'lap_gnss': lap_gnss_masks,
            'gnss_padding_mask': gnss_paddding_mask,
            'earthquake_location': earthquake_loaction,
            'station_location': station_loaction,
            'energy_predict': energy_predict,
            'day_predict': day_predict,
            'earthquake_happen': earthquake_happen
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch, eta_min=1e-5)
        return [optimizer], [scheduler]
    
    def forward(self, batch, batch_idx = None):
        batch = cuda_dict(batch,self.device)
        log_energy_history = batch['log_energy_history']
        gnss_data_history = batch['gnss_data_history']
        log_energy_future = batch['log_energy_future']
        earthquake_data_future_day = batch['earthquake_data_future_day']
        es_geo_masks = batch['es_geo_mask']
        es_sem_masks = batch['es_sem_mask']
        combined_gnss_masks = batch['gnss_geo_mask']
        es_lap_masks = batch['lap_ex']
        lap_gnss_masks = batch['lap_gnss']
        gnss_paddding_mask = batch['gnss_padding_mask']
        earthquake_loaction = batch['earthquake_location']
        station_loaction = batch['station_location']
        earthquake_happen = batch['earthquake_happen']

        energy_predict, day_predict = self.model(
            log_energy_history, 
            gnss_data_history, 
            es_lap_mx=es_lap_masks,
            gnss_lap_mx=lap_gnss_masks,
            es_geo_mask=es_geo_masks,
            es_sem_mask=es_sem_masks,
            gnss_geo_mask=combined_gnss_masks,
            gnss_padding_mask=gnss_paddding_mask,
            es_loc = earthquake_loaction,
            gnss_loc = station_loaction
        )
        
        return {
            'log_energy_history': log_energy_history,
            'gnss_data_history': gnss_data_history,
            'log_energy_future': log_energy_future,
            'earthquake_data_future_day': earthquake_data_future_day,
            'es_geo_mask': es_geo_masks,
            'es_sem_mask': es_sem_masks,
            'gnss_geo_mask': combined_gnss_masks,
            'lap_ex': es_lap_masks,
            'lap_gnss': lap_gnss_masks,
            'gnss_padding_mask': gnss_paddding_mask,
            'earthquake_location': earthquake_loaction,
            'station_location': station_loaction,
            'energy_predict': energy_predict,
            'day_predict': day_predict,
            'earthquake_happen': earthquake_happen
        }
