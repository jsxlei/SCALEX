#!/usr/bin/env python
"""
# Author: Xiong Lei
# Created Time : Mon 18 Nov 2019 01:25:24 PM CST

# File Name: utils.py
# Description:

"""

import numpy as np
import torch


def onehot(y, n):
    """
    Make the input tensor one hot tensors
    
    Parameters
    ----------
    y
        input tensors
    n
        number of classes
        
    Return
    ------
    Tensor
    """
    if (y is None) or (n<2):
        return None
    assert torch.max(y).item() < n
    y = y.view(y.size(0), 1)
    y_cat = torch.zeros(y.size(0), n).to(y.device)
    y_cat.scatter_(1, y.data, 1)
    return y_cat


class EarlyStopping:
    """
    Early stops the training if loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False, checkpoint_file=''):
        """
        Parameters
        ----------
        patience 
            How long to wait after last time loss improved. Default: 10
        verbose
            If True, prints a message for each loss improvement. Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.checkpoint_file = checkpoint_file

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.checkpoint_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''
        Saves model when loss decrease.
        '''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        if self.checkpoint_file:
            torch.save(model.state_dict(), self.checkpoint_file)
        self.loss_min = loss
        
    