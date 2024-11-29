LOSS_REGISTRY = {}

def register_loss(name):
    """
    注册损失函数的装饰器。
    
    参数：
    name (str): 损失函数的名称。
    """
    def decorator(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


import torch
import torch.nn as nn

@register_loss('nse')
class NSELoss(nn.Module):
    def __init__(self):
        super(NSELoss, self).__init__()
    
    def forward(self, predictions, targets):
        # 计算NSE损失的实现
        mean_observed = torch.mean(targets)
        numerator = torch.sum((targets - predictions) ** 2)
        denominator = torch.sum((targets - mean_observed) ** 2)
        nse = 1 - numerator / denominator
        return nse

@register_loss('nnse')
class NNSELoss(nn.Module):
    def __init__(self):
        super(NNSELoss, self).__init__()
        self.nse_loss = NSELoss()
    
    def forward(self, predictions, targets):
        # 计算NNSE损失的实现
        nse = self.nse_loss(predictions, targets)
        nnse = 1 / (2 - nse)
        return nnse

@register_loss('mse')
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    
    def forward(self, predictions, targets):
        # 计算MSE损失的实现
        mse = torch.mean((predictions - targets) ** 2)
        return mse

@register_loss('tss')
class TSSLoss(nn.Module):
    def __init__(self, threshold, lambda_weight=10, alpha=1.0, gamma=10.0, smooth_type="sigmoid"):
        super(TSSLoss, self).__init__()
        self.threshold = threshold
        self.lambda_weight = lambda_weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth_type = smooth_type
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        # 计算TSS损失的实现
        mse_loss = self.mse(y_pred, y_true)
        same_side = ((y_pred - self.threshold) * (y_true - self.threshold)) >= 0
        diff_side = ~same_side
        penalty = torch.zeros_like(y_pred)
        penalty[diff_side] = self._smooth_function(torch.abs(y_pred[diff_side] - self.threshold))
        cross_boundary_penalty = penalty * self.alpha * torch.abs(y_pred - self.threshold)
        classification_loss = cross_boundary_penalty.mean()
        total_loss = mse_loss + self.lambda_weight * classification_loss
        return total_loss
    
    def _smooth_function(self, x):
        if self.smooth_type == "sigmoid":
            return torch.sigmoid(self.gamma * x)
        elif self.smooth_type == "tanh":
            return (torch.tanh(self.gamma * x) + 1) / 2
        else:
            raise ValueError("Unsupported smooth type. Choose 'sigmoid' or 'tanh'.")

@register_loss('powerlaw_mse')

class PowerLawWeightedMSELoss(nn.Module):
    def __init__(self, alpha=2.0):
        super(PowerLawWeightedMSELoss, self).__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        # Compute weights based on targets
        weights = self.alpha ** targets  # Shape: (batch_size, 100)
        
        # Normalize weights per sample (along dimension 1)
        mean_weights = torch.mean(weights, dim=1, keepdim=True)  # Shape: (batch_size, 1)
        weights = weights / mean_weights  # Broadcast normalization
        mse = torch.mean(weights * (predictions - targets) ** 2)

        return mse

@register_loss('cross_entropy')
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, predict_day_class=15):
        super(CustomCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.predict_day_class = predict_day_class
    
    def forward(self, logits, targets):
        # 计算自定义的交叉熵损失
        logits = logits.view(-1, self.predict_day_class)
        targets = targets.view(-1)
        loss = self.criterion(logits, targets.long())
        return loss

@register_loss('power_tss')
class PowerTSSLoss(nn.Module):
    def __init__(self, threshold, lambda_weight=10, alpha=1.0, gamma=10.0, smooth_type="sigmoid"):
        super(PowerTSSLoss, self).__init__()
        self.threshold = threshold
        self.lambda_weight = lambda_weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth_type = smooth_type
        self.plmse = PowerLawWeightedMSELoss(alpha=2.0)
    
    def forward(self, y_pred, y_true):
        mse_loss = self.plmse(y_pred, y_true)
        same_side = ((y_pred - self.threshold) * (y_true - self.threshold)) >= 0
        diff_side = ~same_side
        penalty = torch.zeros_like(y_pred)
        penalty[diff_side] = self._smooth_function(torch.abs(y_pred[diff_side] - self.threshold))
        cross_boundary_penalty = penalty * self.alpha * torch.abs(y_pred - self.threshold)
        classification_loss = cross_boundary_penalty.mean()
        total_loss = mse_loss + self.lambda_weight * classification_loss
        return total_loss
    
    def _smooth_function(self, x):
        if self.smooth_type == "sigmoid":
            return torch.sigmoid(self.gamma * x)
        elif self.smooth_type == "tanh":
            return (torch.tanh(self.gamma * x) + 1) / 2
        else:
            raise ValueError("Unsupported smooth type. Choose 'sigmoid' or 'tanh'.")

def get_loss(energy_loss='nse', day_loss='cross_entropy'):
    """
    根据损失函数名称获取损失函数实例。
    
    参数：
    energy_loss (str): 能量损失函数的名称。
    day_loss (str): 日期损失函数的名称。
    
    返回：
    dict: 包含损失函数实例的字典。
    """
    loss_fns = {}
    
    # 获取能量损失函数
    if energy_loss in LOSS_REGISTRY:
        if energy_loss == 'tss':
            loss_fns['energy_loss'] = LOSS_REGISTRY[energy_loss](threshold=3.29)
        elif energy_loss == 'powerlaw_mse':
            loss_fns['energy_loss'] = LOSS_REGISTRY[energy_loss](alpha=2.0)
        elif energy_loss == 'nse':
            loss_fns['energy_loss'] = LOSS_REGISTRY[energy_loss]()
        elif energy_loss == 'power_tss':
            loss_fns['energy_loss'] = LOSS_REGISTRY[energy_loss](threshold=3.29)
    else:
        raise ValueError(f"未知的能量损失函数: {energy_loss}")
    # 获取日期损失函数
    if day_loss in LOSS_REGISTRY:
        if day_loss == 'cross_entropy':
            loss_fns['day_loss'] = LOSS_REGISTRY[day_loss]()
        else:
            loss_fns['day_loss'] = LOSS_REGISTRY[day_loss]()
    elif day_loss == 'None':
        loss_fns['day_loss'] = None
    else:
        raise ValueError(f"未知的日期损失函数: {day_loss}")
    
    return loss_fns