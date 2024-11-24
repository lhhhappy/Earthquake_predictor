import torch
import lightning as L
from .ES_net import ES_net
from .ES_net_mixer import ES_net_mixer
import torch.nn as nn
from .Mlpbaseline import Mlpbaseline



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
        
        self.save_hyperparameters()

        self.aggregative_score = AggregativeScoreLoss(threshold=3)
        self.nnse_loss = NNSELoss()
    def compute_loss(self, energy_predict, log_energy_future, day_predict, earthquake_data_future_day):
        loss_metrics = {}
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
            loss_metrics[name] = loss.item()
        nnse_loss = self.nnse_loss(energy_predict, log_energy_future)
        loss_metrics['nnse_loss'] = nnse_loss.item()
        return total_loss, loss_metrics
    
    def on_after_backward(self):
        
        total_grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_grad_norm = p.grad.data.norm(2)
                total_grad_norm += param_grad_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        self.log('grad_norm/train', total_grad_norm)

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
        train_loss, loss_metric = self.compute_loss(
            energy_predict, log_energy_future, 
            day_predict, earthquake_data_future_day,
        )

        metrics = self.aggregative_score(energy_predict, log_energy_future)
        metrics["total_loss"] = train_loss.item()
        for k,v in loss_metric.items():
            metrics[k] = v

        for k, v in metrics.items():
            self.log(f'{k}/train', v, prog_bar=False, logger=True)

        return train_loss
    

    def validation_step(self, batch, batch_idx):
        # 提取批次数据
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

        # 模型前向计算
        energy_predict, day_predict = self.model(
            log_energy_history,
            gnss_data_history,
            es_lap_mx=es_lap_masks,
            gnss_lap_mx=lap_gnss_masks,
            es_geo_mask=es_geo_masks,
            es_sem_mask=es_sem_masks,
            gnss_geo_mask=combined_gnss_masks,
            gnss_padding_mask=gnss_paddding_mask,
            es_loc=earthquake_loaction,
            gnss_loc=station_loaction
        )

        # 计算验证损失
        val_loss,loss_metric = self.compute_loss(
            energy_predict, log_energy_future,
            day_predict, earthquake_data_future_day
        )

        metrics = self.aggregative_score(energy_predict, log_energy_future)
        metrics["total_loss"] = val_loss.item()
        
        for k,v in loss_metric.items():
            metrics[k] = v
        for k, v in metrics.items():
            self.log(f'{k}/valid', v, prog_bar=False, logger=True)

        self.log("TPR", metrics["TPR"], prog_bar=False, logger=False)
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epoch, eta_min=1e-6)
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


class AggregativeScoreLoss:

    def __init__(self, threshold=3.5):
        """
        初始化类
        Args:
            threshold: 能量阈值 (float)
        """
        self.threshold = threshold

    def __call__(self, energy_predict, earthquake_target):
        """
        计算 Aggregative Score 和相关指标。
        Args:
            energy_predict: 预测的能量值 (torch.Tensor, GPU 上)
            earthquake_target: 地震发生的真实标签 (torch.Tensor, GPU 上)
        Returns:
            aggregative_score: 聚合得分 (float)
            metrics: 包含其他评估指标的字典
        """
        # 转换为分类标签
        
        energy_predict = energy_predict.flatten()
        earthquake_target = earthquake_target.flatten()

        earthquake_happen_predict = (energy_predict > self.threshold).long()
        earthquake_happen_real = (earthquake_target > self.threshold).long()

        # 计算混淆矩阵的四个区域
        confusion_vector = earthquake_happen_predict * 2 + earthquake_happen_real
        TP = torch.sum(confusion_vector == 3).item()  # True Positive
        TN = torch.sum(confusion_vector == 0).item()  # True Negative
        FP = torch.sum(confusion_vector == 2).item()  # False Positive
        FN = torch.sum(confusion_vector == 1).item()  # False Negative

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate (Recall)
        TNR = TN / (FP + TN) if (FP + TN) > 0 else 0  # True Negative Rate (Specificity)
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        Accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        Aggregative_Score = (TPR + Precision + Accuracy) / 3

        # 返回结果
        metrics = {
            "TPR": TPR,
            "TNR": TNR,
            "Precision": Precision,
            "Accuracy": Accuracy,
            "Aggregative_Score": Aggregative_Score
        }
        return metrics
    
class NNSELoss(nn.Module):
    """
    Negative Nash-Sutcliffe Efficiency (NNSE) Loss.
    """
    def __init__(self):
        super(NNSELoss, self).__init__()

    def forward(self, energy_predict, earthquake_target):
        """
        Args:
            energy_predict (torch.Tensor): Predicted energy values, shape (B, *).
            earthquake_target (torch.Tensor): Ground truth energy values, shape (B, *).

        Returns:
            torch.Tensor: Computed NNSE loss.
        """
        # Flatten the inputs to ensure 1D computation
        energy_predict = energy_predict.flatten()
        earthquake_target = earthquake_target.flatten()

        # Compute the numerator (sum of squared errors)
        numerator = torch.sum(torch.pow(earthquake_target - energy_predict, 2))
        # Compute the denominator (sum of squared deviations from the mean)
        denominator = torch.sum(torch.pow(earthquake_target - torch.mean(earthquake_target), 2))

        # Compute the Nash-Sutcliffe Efficiency (NSE)
        nse = 1 - (numerator / denominator)

        # Compute the NNSE loss
        nnse = 1 / (2 - nse)

        return nnse