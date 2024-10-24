import torch
import torch.nn as nn

class NSELoss(nn.Module):
    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Compute the Nash-Sutcliffe Efficiency (NSE) loss.

        Parameters:
        predictions (torch.Tensor): Predicted values (P_i).
        targets (torch.Tensor): Observed values (O_i).

        Returns:
        torch.Tensor: Computed NSE loss.
        """
        # Compute mean of observed values
        mean_observed = torch.mean(targets)
        
        # Compute the numerator: sum of squared differences between observed and predicted values
        numerator = torch.sum((targets - predictions) ** 2)
        
        # Compute the denominator: sum of squared differences between observed values and their mean
        denominator = torch.sum((targets - mean_observed) ** 2)
        
        # Compute NSE
        nse = 1 - numerator / denominator

        return nse

class NNSELoss(nn.Module):
    def __init__(self):
        super(NNSELoss, self).__init__()
        self.nse_loss = NSELoss()

    def forward(self, predictions, targets):
        """
        Compute the Normalized Nash-Sutcliffe Efficiency (NNSE) loss.

        Parameters:
        predictions (torch.Tensor): Predicted values (P_i).
        targets (torch.Tensor): Observed values (O_i).

        Returns:
        torch.Tensor: Computed NNSE loss.
        """
        # Compute NSE
        nse = self.nse_loss(predictions, targets)
        
        # Compute NNSE
        nnse = 1 / (2 - nse)

        return nnse


class CustomCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for a three-class classification task.
    """
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()
        # Initialize the cross-entropy loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets):
        """
        Forward pass to compute the loss.

        :param logits: Model output logits of shape (batch_size, num, 3)
        :param targets: Ground truth labels of shape (batch_size, num), with values 0, 1, or 2
        :return: Scalar loss value
        """
        # Reshape logits from (batch_size, num, 3) to (batch_size * num, 3)
        logits = logits.view(-1, 3)
        
        # Reshape targets from (batch_size, num) to (batch_size * num)
        targets = targets.view(-1)
        
        # Compute cross-entropy loss
        loss = self.criterion(logits, targets)
        
        return loss
def get_loss(energy_loss='nse', day_loss='cross_entropy'):
    """
    Get the loss function based on the loss name.

    Parameters:
    loss_name (str): Name of the loss function.

    Returns:
    nn.Module: Loss function.
    """
    loss_fns = {}
    if energy_loss == 'nse':
        loss_fns['energy_loss'] = NSELoss()
    elif energy_loss == 'nnse':
        loss_fns['energy_loss'] = NNSELoss()
    else:
        raise ValueError(f"Unknown energy loss function: {energy_loss}")
    if day_loss == 'cross_entropy':
        loss_fns['day_loss'] = CustomCrossEntropyLoss()
    else:
        raise ValueError(f"Unknown day loss function: {day_loss}")
    return loss_fns